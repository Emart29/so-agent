"""Checks whether structurally valid output is actually correct.

Schema validation proves shape. It says nothing about truth, and solving shape
made that gap harder to see: a green checkmark now hides a wrong answer that
would previously have crashed. ``{"priority": "critical", "confidence": 0.98}``
passes every check ever written and can still be the wrong call on a ticket
about a typo.

A second, cheap model reviews the extracted values against the source text and
answers a short list of specific questions. Specific is the operative word — a
critic asked "is this good?" produces agreeable noise, while one asked "does
every extracted value appear in or follow from the source?" produces something
checkable.

Two rules the design turns on:

* **The critic's own output is model output.** It goes through the same
  validation path as everything else and gets no special trust.
* **The critic is itself fallible and its error rate is published.** An
  unmeasured judge is a second unvalidated model, not a safety net — which is
  why :func:`measure_agreement` exists and why the README carries its number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from contracts.translate import SchemaTranslator
from enforce.ladder import build_plan
from enforce.validate import validate_response
from provider.capabilities import ModelCapabilities, load_capabilities

logger = logging.getLogger("critic.semantic")

#: Kept small deliberately. The critic runs on every request, and one that costs
#: more than the primary call does not survive contact with a budget.
CRITIC_MAX_TOKENS = 1200


class Verdict(str, Enum):
    """What the critic concluded."""

    SOUND = "sound"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    #: The critic itself failed — its output would not validate, or the call
    #: errored. Recorded separately so a broken critic is never read as a pass.
    UNAVAILABLE = "unavailable"


class CriticReport(BaseModel):
    """The critic's structured answer, validated like any other model output."""

    model_config = ConfigDict(extra="forbid")

    grounded: bool = Field(
        description=(
            "True only if every extracted value appears in, or follows directly "
            "from, the source text. False if any value was inferred or invented."
        )
    )
    plausible_confidence: bool = Field(
        description=(
            "True if the stated confidence matches how clearly the source "
            "supports the extraction. False if it is over- or under-stated."
        )
    )
    categories_consistent: bool = Field(
        description=(
            "True if every enum choice matches the content, rather than merely "
            "being a valid member of the enum."
        )
    )
    substantive: bool = Field(
        description=(
            "True if fields required by meaning carry real content, rather than "
            "placeholder text, restated field names, or empty strings."
        )
    )
    problem: str = Field(
        description=(
            "The single most serious problem, in one sentence. Empty string if "
            "there is none."
        )
    )


@dataclass
class SemanticResult:
    """The critic's verdict on one extraction."""

    verdict: Verdict
    report: CriticReport | None = None
    reason: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float = 0.0

    @property
    def passed(self) -> bool:
        """Whether the extraction is semantically sound.

        An unavailable critic is not a pass. Treating it as one would let a
        broken critic silently approve everything, which is worse than having
        no critic at all because it looks like coverage.
        """
        return self.verdict is Verdict.SOUND

    @property
    def checked(self) -> bool:
        """Whether a verdict was actually reached."""
        return self.verdict is not Verdict.UNAVAILABLE


CRITIC_INSTRUCTIONS = (
    "You are checking whether a structured extraction is faithful to its source.\n"
    "The extraction is already known to be well-formed and to satisfy its schema. "
    "Do not comment on its shape, field names, JSON validity, or whether a value "
    "belongs to an allowed set — those are already guaranteed.\n\n"
    "Two kinds of field are judged differently, and confusing them is the main "
    "way this review goes wrong:\n"
    "  - EXTRACTED facts (names, plans, dates, quantities) must actually appear "
    "in the source. Inventing one is a failure.\n"
    "  - INFERRED judgements (priority, sentiment, category, confidence) are "
    "conclusions the source never states outright. They are correct when they "
    "are reasonable given the content. Do not fault an inferred value merely "
    "for being absent from the text — that is what inferring means.\n\n"
    "Judge only these questions, on the evidence in the source:\n"
    "  - grounded: is every EXTRACTED fact present in the source, and is every "
    "INFERRED judgement reasonable given it? A summary may paraphrase freely "
    "and does not need the source's wording. Judge meaning, not phrasing.\n"
    "  - plausible_confidence: does any stated confidence match how clearly the "
    "source supports the extraction?\n"
    "  - categories_consistent: does each category or priority fit the content? "
    "Choose only from the values the schema allows — do not propose alternatives "
    "that are not in it.\n"
    "  - substantive: do meaning-bearing fields carry real content, rather than "
    "invented placeholders such as \"Anonymous\" or \"Unnamed Customer\" for "
    "details the source never gave?\n\n"
    "Mark a field false only when you can point to the specific problem. An "
    "extraction that fairly represents the source is sound even if you would "
    "have worded it differently."
)


class SemanticCritic:
    """Reviews an extraction against its source with a cheap second model."""

    def __init__(
        self,
        client,
        model: str,
        translator: SchemaTranslator | None = None,
        capabilities: ModelCapabilities | None = None,
    ):
        """
        Args:
            client: Provider client used for the critique.
            model: Model id. Should be the cheapest one that reliably returns
                valid JSON; the critic does not need to be clever.
            translator: Schema translator, so the critic's own schema goes
                through the same provider-compatibility rules as everything else.
            capabilities: Measured capabilities for the critic's model. Looked
                up when omitted.

        The critic is a model call like any other and goes through the same
        enforcement ladder. Assuming native schema support here would break it
        on exactly the cheap models it is supposed to run on — most of them do
        not support it, which is why the ladder exists.
        """
        self.client = client
        self.model = model
        self.translator = translator or SchemaTranslator()
        self._translation = self.translator.translate(CriticReport)

        if capabilities is None:
            capabilities = (
                load_capabilities().get(client.provider.name, {}).get(model)
            )
        self.capabilities = capabilities
        self._plan = build_plan("critic_report", self._translation, capabilities)

        if not self._plan.is_strongest:
            logger.info(
                "critic %s runs at %s (%s)",
                model, self._plan.tier, self._plan.guarantee,
            )

    def review(
        self, source: str, extraction: Any, contract: type[BaseModel] | None = None
    ) -> SemanticResult:
        """Judge one extraction against the text it came from.

        Args:
            source: The original text.
            extraction: The validated object produced from it.
            contract: The model the extraction satisfies. Supplying it lets the
                critic see which enum values are permitted — without it, a
                critic will confidently propose categories the schema does not
                contain and mark sound output as inconsistent for using the
                only values available to it.

        Returns:
            The verdict, including the case where the critic itself failed.
        """
        payload = (
            extraction.model_dump_json(indent=2)
            if isinstance(extraction, BaseModel)
            else str(extraction)
        )

        contract = contract or (
            type(extraction) if isinstance(extraction, BaseModel) else None
        )
        allowed = _allowed_values(contract) if contract else ""
        # Below the top tier the provider enforces nothing, so the critic's own
        # schema has to travel in the prompt exactly as it does for any other
        # request. Most cheap models — the ones a critic should run on — are
        # below the top tier.
        instructions = CRITIC_INSTRUCTIONS
        if self._plan.system_suffix:
            instructions = f"{instructions}\n\n{self._plan.system_suffix}"

        user_content = f"SOURCE TEXT:\n{source}\n\nEXTRACTION:\n{payload}"
        if allowed:
            user_content += f"\n\nVALUES THE SCHEMA ALLOWS:\n{allowed}"

        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_content},
        ]

        try:
            response = self.client.chat(
                messages=messages,
                model=self.model,
                max_tokens=CRITIC_MAX_TOKENS,
                **self._plan.request_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - a failed critic must not fail the request
            logger.warning("critic call failed: %s", exc)
            return SemanticResult(
                verdict=Verdict.UNAVAILABLE, reason=f"{type(exc).__name__}: {exc}"
            )

        # The critic's output is model output and earns no special trust.
        validated = validate_response(
            response.text, CriticReport, truncated=response.truncated
        )
        if not validated.ok:
            return SemanticResult(
                verdict=Verdict.UNAVAILABLE,
                reason=f"critic output did not validate: {validated.failure.value}",
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                latency_ms=response.latency_ms,
            )

        report: CriticReport = validated.parsed
        return SemanticResult(
            verdict=_verdict_from(report),
            report=report,
            reason=report.problem,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=response.latency_ms,
        )


def _allowed_values(contract: type[BaseModel]) -> str:
    """List the enum values a contract permits, for the critic to judge against.

    A critic that cannot see the permitted set invents alternatives and then
    faults the extraction for not using them, which reads as a semantic failure
    and is really the critic misunderstanding the contract.
    """
    schema = contract.model_json_schema()
    definitions = schema.get("$defs", {})

    lines: list[str] = []
    for name, body in definitions.items():
        values = body.get("enum")
        if values:
            lines.append(f"  {name}: {', '.join(map(str, values))}")

    for field_name, body in schema.get("properties", {}).items():
        values = body.get("enum")
        if values:
            lines.append(f"  {field_name}: {', '.join(map(str, values))}")

    return "\n".join(lines)


def _verdict_from(report: CriticReport) -> Verdict:
    """Reduce the critic's answers to a verdict.

    Groundedness is separated from the rest because it is the failure that
    matters: a value contradicted by the source is wrong, while an overstated
    confidence or a thin summary is merely unsupported.
    """
    if not report.grounded:
        return Verdict.CONTRADICTED
    if not (
        report.plausible_confidence
        and report.categories_consistent
        and report.substantive
    ):
        return Verdict.UNSUPPORTED
    return Verdict.SOUND


@dataclass
class AgreementScore:
    """How closely the critic matches human judgement on a labelled set."""

    total: int
    agreed: int
    false_pass: int
    false_fail: int
    unavailable: int

    @property
    def agreement(self) -> float:
        """Share of judged cases where the critic matched the label."""
        judged = self.total - self.unavailable
        return self.agreed / judged if judged else 0.0

    def summary(self) -> str:
        return (
            f"agreement {self.agreement:.0%} over {self.total - self.unavailable} "
            f"judged cases ({self.false_pass} approved a bad extraction, "
            f"{self.false_fail} rejected a good one, {self.unavailable} unavailable)"
        )


def measure_agreement(
    critic: SemanticCritic, labelled: list[tuple[str, Any, bool]]
) -> AgreementScore:
    """Score the critic against hand-labelled cases.

    Without this the critic is a second unvalidated model rather than a check on
    the first, and any semantic failure rate it produces inherits an unknown
    error rate. The result belongs in the README beside the numbers it qualifies.

    Args:
        critic: The critic to score.
        labelled: ``(source, extraction, is_actually_sound)`` per case.

    Returns:
        The agreement score, counting the two error directions separately —
        approving a bad extraction is a different failure from rejecting a
        good one, and they have different consequences.
    """
    agreed = false_pass = false_fail = unavailable = 0

    for source, extraction, truly_sound in labelled:
        result = critic.review(source, extraction)
        if not result.checked:
            unavailable += 1
            continue
        if result.passed == truly_sound:
            agreed += 1
        elif result.passed:
            false_pass += 1
        else:
            false_fail += 1

    return AgreementScore(
        total=len(labelled),
        agreed=agreed,
        false_pass=false_pass,
        false_fail=false_fail,
        unavailable=unavailable,
    )
