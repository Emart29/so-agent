"""The public API: the thing someone actually imports.

Everything below this file is a layer; this is where they compose. Three design
rules, each chosen because its opposite is a bug this project exists to prevent:

* **Failures are loud and typed, never silent.** Returning ``None`` or an empty
  dict on a parse failure pushes the problem downstream where nobody can
  attribute it. Every result says what happened and why.
* **Every path writes a log record**, including the one that succeeded on the
  first attempt. The base rate is what the failure rate is measured against.
* **The agent works with no store and no critic.** It is a library first; the
  measurement apparatus is optional and its absence degrades reporting rather
  than breaking requests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from config import settings
from contracts.schemas import Contract
from contracts.translate import SchemaTranslator, Translation
from critic.semantic import SemanticCritic, SemanticResult, Verdict
from enforce.ladder import EnforcementPlan, build_plan
from enforce.repair import RepairOutcome, repair_loop
from enforce.tools import ToolResult, validate_tool_call
from provider.capabilities import ModelCapabilities, load_capabilities
from provider.client import LLMClient
from store.log import AttemptLog, AttemptRow, new_run_id

logger = logging.getLogger("agent")

DEFAULT_MAX_TOKENS = 2000


@dataclass
class Result:
    """What one request produced, and everything needed to account for it."""

    ok: bool
    value: Any = None
    run_id: str = ""
    provider: str = ""
    model: str = ""
    tier: str = ""
    downgraded_from: str | None = None
    outcome: RepairOutcome | None = None
    critic: SemanticResult | None = None
    translation: Translation | None = None
    error: str = ""

    @property
    def attempts(self) -> int:
        return self.outcome.attempt_count if self.outcome else 0

    @property
    def first_attempt_ok(self) -> bool:
        return bool(self.outcome and self.outcome.first_attempt_ok)

    @property
    def total_tokens(self) -> int:
        base = self.outcome.total_tokens if self.outcome else 0
        if self.critic:
            base += (self.critic.prompt_tokens or 0) + (
                self.critic.completion_tokens or 0
            )
        return base

    @property
    def total_latency_ms(self) -> float:
        base = self.outcome.total_latency_ms if self.outcome else 0.0
        return base + (self.critic.latency_ms if self.critic else 0.0)

    @property
    def semantically_sound(self) -> bool | None:
        """Whether the critic approved, or ``None`` if it did not run.

        Deliberately tri-state. Collapsing "not checked" into "failed" would
        understate quality, and into "passed" would let an absent critic look
        like coverage.
        """
        if self.critic is None or not self.critic.checked:
            return None
        return self.critic.passed

    def summary(self) -> str:
        status = "ok" if self.ok else f"FAILED ({self.error})"
        parts = [
            f"{status}",
            f"tier={self.tier}",
            f"attempts={self.attempts}",
            f"tokens={self.total_tokens}",
        ]
        sound = self.semantically_sound
        if sound is not None:
            parts.append(f"sound={sound}")
        return "  ".join(parts)


class StructuredAgent:
    """Extracts validated objects from text, and records how well that went."""

    def __init__(
        self,
        provider: str = "groq",
        model: str | None = None,
        critic_model: str | None = None,
        max_repairs: int | None = None,
        store: AttemptLog | None = None,
        translator: SchemaTranslator | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        """
        Args:
            provider: Provider name. Defaults to the configured one; a metered
                provider must be named explicitly.
            model: Model id. Defaults to the configured primary, which the
                capability probe fills in.
            critic_model: Model for semantic review. ``None`` disables the
                critic, and the result then reports "not checked" rather than
                claiming soundness.
            max_repairs: Attempts including the first.
            store: Where attempts are logged. ``None`` disables logging.
            translator: Schema translator, for overriding provider rules.
            max_tokens: Starting output ceiling.
        """
        self.client = LLMClient(provider)
        self.provider = self.client.provider.name
        self.model = model or settings.PRIMARY_MODEL
        if not self.model:
            raise ValueError(
                "no model configured. Run the capability probe first, or pass "
                "model= explicitly — guessing a model id measures the guess."
            )

        self.translator = translator or SchemaTranslator()
        self.max_repairs = max_repairs or settings.MAX_REPAIR_ATTEMPTS
        self.max_tokens = max_tokens
        self.store = store

        self.capabilities: ModelCapabilities | None = (
            load_capabilities().get(self.provider, {}).get(self.model)
        )
        if self.capabilities is None:
            logger.info(
                "%s/%s has not been probed; requests will use the weakest tier",
                self.provider, self.model,
            )

        self.critic = (
            SemanticCritic(self.client, critic_model, self.translator)
            if critic_model
            else None
        )

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def extract(
        self,
        text: str,
        contract: type[BaseModel],
        instruction: str | None = None,
        tier: str | None = None,
        review: bool = True,
    ) -> Result:
        """Extract a validated object from text.

        Args:
            text: Source text.
            contract: Pydantic model the result must satisfy.
            instruction: What to do with the text. Defaults to extraction.
            tier: Force an enforcement tier rather than selecting one.
            review: Run the semantic critic, when one is configured.

        Returns:
            The result, successful or explained.
        """
        run_id = new_run_id()
        name = _schema_name(contract)
        translation = self.translator.translate(contract)
        plan = build_plan(name, translation, self.capabilities, tier)

        task = instruction or "Extract the required fields from the text above."
        messages = _messages(text, task, plan)

        def generate(repair_message: str | None, max_tokens: int | None):
            turn = list(messages)
            if repair_message:
                turn.append({"role": "user", "content": repair_message})
            return self.client.chat(
                messages=turn,
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                **plan.request_kwargs,
            )

        try:
            outcome = repair_loop(
                generate,
                contract,
                tier=plan.tier,
                max_attempts=self.max_repairs,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            result = Result(
                ok=False, run_id=run_id, provider=self.provider, model=self.model,
                tier=plan.tier, downgraded_from=plan.downgraded_from,
                translation=translation, error=f"{type(exc).__name__}: {exc}",
            )
            self._log(result, contract, plan, text)
            return result

        critic_result: SemanticResult | None = None
        if outcome.ok and review and self.critic is not None:
            critic_result = self.critic.review(text, outcome.parsed, contract)

        result = Result(
            ok=outcome.ok,
            value=outcome.parsed,
            run_id=run_id,
            provider=self.provider,
            model=self.model,
            tier=plan.tier,
            downgraded_from=plan.downgraded_from,
            outcome=outcome,
            critic=critic_result,
            translation=translation,
            error="" if outcome.ok else outcome.stopped_because,
        )
        self._log(result, contract, plan, text)
        return result

    def choose(
        self,
        text: str,
        union_envelope: type[BaseModel],
        instruction: str | None = None,
        tier: str | None = None,
        review: bool = False,
    ) -> Result:
        """Let the model pick a response shape rather than fill a fixed one."""
        return self.extract(
            text,
            union_envelope,
            instruction=instruction
            or "Choose the response variant that fits, and fill it in.",
            tier=tier,
            review=review,
        )

    def call_tool(
        self,
        text: str,
        tools: list[dict[str, Any]],
        expected: dict[str, type[BaseModel]],
        instruction: str | None = None,
        require_one: bool = True,
    ) -> tuple[ToolResult, Result]:
        """Ask for a tool call and validate its arguments.

        Returns:
            The validated call and a :class:`Result` carrying the accounting.
            Tool arguments are model output and are validated like any other.
        """
        run_id = new_run_id()
        messages = [
            {"role": "user", "content": f"{text}\n\n{instruction or 'Use a tool.'}"}
        ]

        try:
            response = self.client.chat(
                messages=messages,
                model=self.model,
                tools=tools,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            failed = Result(
                ok=False, run_id=run_id, provider=self.provider, model=self.model,
                tier="tools", error=f"{type(exc).__name__}: {exc}",
            )
            return ToolResult(ok=False, detail=failed.error), failed

        tool_result = validate_tool_call(response.tool_calls, expected, require_one)
        result = Result(
            ok=tool_result.ok,
            value=tool_result.arguments,
            run_id=run_id,
            provider=self.provider,
            model=self.model,
            tier="tools",
            error="" if tool_result.ok else tool_result.detail,
        )

        if self.store is not None:
            self.store.record(
                AttemptRow(
                    run_id=run_id,
                    attempt_index=1,
                    provider=self.provider,
                    model=self.model,
                    tier="tools",
                    schema_name=tool_result.tool_name or "unknown_tool",
                    success=tool_result.ok,
                    failure_type=None if tool_result.ok else tool_result.failure.value,
                    failure_detail=tool_result.detail or None,
                    raw_output=None if tool_result.ok else tool_result.raw_arguments,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    latency_ms=response.latency_ms,
                )
            )

        return tool_result, result

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(
        self,
        result: Result,
        contract: type[BaseModel],
        plan: EnforcementPlan,
        source: str,
    ) -> None:
        """Write one row per attempt, including successful first attempts.

        The source text goes in with them. Without it a logged failure can be
        counted but not reproduced, and reproducing it elsewhere is the whole
        argument for keeping the raw output.
        """
        if self.store is None:
            return

        difficulty = (
            contract.difficulty().value
            if isinstance(contract, type) and issubclass(contract, Contract)
            else None
        )
        name = _schema_name(contract)

        if result.outcome is None:
            self.store.record(
                AttemptRow(
                    run_id=result.run_id, attempt_index=1, provider=result.provider,
                    model=result.model, tier=result.tier,
                    requested_tier=plan.tier, downgraded_from=plan.downgraded_from,
                    schema_name=name, schema_difficulty=difficulty,
                    success=False, failure_type="error", failure_detail=result.error,
                    source_text=source,
                )
            )
            return

        rows: list[AttemptRow] = []
        last = len(result.outcome.attempts)
        for attempt in result.outcome.attempts:
            # The critic judges the run's final object, so its verdict belongs
            # on the attempt that produced it rather than on every attempt.
            is_final = attempt.index == last
            rows.append(
                AttemptRow(
                    run_id=result.run_id,
                    attempt_index=attempt.index,
                    provider=result.provider,
                    model=result.model,
                    tier=attempt.tier,
                    requested_tier=plan.tier,
                    downgraded_from=plan.downgraded_from,
                    schema_name=name,
                    schema_difficulty=difficulty,
                    success=attempt.ok,
                    failure_type=None if attempt.ok else attempt.failure.value,
                    failure_detail=attempt.detail or None,
                    raw_output=None if attempt.ok else attempt.raw,
                    recovered_by_extraction=attempt.recovered_by_extraction,
                    repaired_from=attempt.repaired_from,
                    prompt_tokens=attempt.prompt_tokens,
                    completion_tokens=attempt.completion_tokens,
                    latency_ms=attempt.latency_ms,
                    max_tokens=attempt.max_tokens,
                    critic_verdict=(
                        result.critic.verdict.value
                        if is_final and result.critic
                        else None
                    ),
                    critic_reason=(
                        result.critic.reason if is_final and result.critic else None
                    ),
                    source_text=source,
                )
            )
        self.store.record_many(rows)


def _schema_name(contract: type[BaseModel]) -> str:
    """Provider-safe schema name derived from the model class."""
    return contract.__name__.lower()


def _messages(text: str, task: str, plan: EnforcementPlan) -> list[dict[str, str]]:
    """Build the conversation, adding the schema when the tier needs it."""
    messages: list[dict[str, str]] = []
    if plan.system_suffix:
        messages.append({"role": "system", "content": plan.system_suffix})
    messages.append({"role": "user", "content": f"{text}\n\n{task}"})
    return messages
