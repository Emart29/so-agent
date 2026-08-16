"""Retries a failed generation with the validation error fed back.

The repair message is the whole technique. Feeding the model the exact error —
*"confidence: Input should be less than or equal to 1"* — alongside its own bad
output gives it something specific to change. A generic "that was invalid, try
again" gives it nothing, and the second attempt fails the same way.

Retrying is a cost, so it is refused wherever it cannot help:

* nothing is retried when the free repair already succeeded,
* refusals and empty responses are not retried at all,
* truncation is retried only with a larger budget, since the same ceiling
  reproduces the same truncation,
* an attempt that returns byte-identical output stops the loop, because a model
  repeating itself will keep repeating itself.

The full attempt history is returned rather than a final verdict. A run that
reports "3 attempts, failed" hides the case this loop is most likely to hit:
the model fixing the field it was told about and breaking a different one.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel

from config import settings
from enforce.validate import FailureType, ValidationResult, validate_response

logger = logging.getLogger("enforce.repair")

#: Multiplier applied to the token ceiling when retrying a truncated response.
TRUNCATION_BUDGET_MULTIPLIER = 2


@dataclass
class Attempt:
    """One generation and what became of it."""

    index: int
    tier: str
    ok: bool
    failure: FailureType
    detail: str = ""
    raw: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    max_tokens: int | None = None
    recovered_by_extraction: bool = False
    repaired_from: int | None = None


@dataclass
class RepairOutcome:
    """The result of the loop, with every attempt it took."""

    ok: bool
    parsed: Any = None
    attempts: list[Attempt] = field(default_factory=list)
    stopped_because: str = ""

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def first_attempt_ok(self) -> bool:
        """Whether the first generation validated without any repair.

        The headline number: how often enforcement worked unaided.
        """
        return bool(self.attempts) and self.attempts[0].ok

    @property
    def needed_only_extraction(self) -> bool:
        """Whether success came from local extraction rather than a retry."""
        return self.ok and any(
            a.ok and a.recovered_by_extraction for a in self.attempts
        )

    @property
    def total_tokens(self) -> int:
        return sum(
            (a.prompt_tokens or 0) + (a.completion_tokens or 0) for a in self.attempts
        )

    @property
    def total_latency_ms(self) -> float:
        return sum(a.latency_ms for a in self.attempts)

    @property
    def error_sequence(self) -> list[str]:
        """Failure types in order, so oscillation is visible.

        A model that fixes the named field and breaks another produces a
        changing sequence here, which a single final verdict would hide.
        """
        return [a.failure.value for a in self.attempts]


def build_repair_message(result: ValidationResult, contract: type[BaseModel]) -> str:
    """Compose the corrective prompt for one failed attempt."""
    errors = result.error_summary()

    if result.failure is FailureType.NOT_JSON:
        return (
            "That response was not valid JSON. Return only a JSON object "
            "matching the required schema, with no explanation, no markdown "
            "fences, and no text before or after it."
        )

    if result.failure is FailureType.PROVIDER_REJECTED:
        return (
            "The provider rejected that response because it did not match the "
            "required schema:\n"
            f"{errors}\n\n"
            "Return a JSON object that matches the schema exactly. Every "
            "required field must be present, with the correct type."
        )

    if result.failure is FailureType.CONSTRAINT_FAIL:
        return (
            "The JSON was structurally correct but violated a value constraint:\n"
            f"{errors}\n\n"
            "Return the corrected JSON object. Change only the fields listed "
            "above; leave every other value exactly as it was."
        )

    return (
        "That response did not match the required schema:\n"
        f"{errors}\n\n"
        "Return the corrected JSON object. Fix only the fields listed above and "
        "keep every other value unchanged."
    )


def repair_loop(
    generate: Callable[..., Any],
    contract: type[BaseModel],
    tier: str,
    max_attempts: int | None = None,
    max_tokens: int | None = None,
) -> RepairOutcome:
    """Generate, validate, and retry with the error fed back until it holds.

    Args:
        generate: Callable taking ``(repair_message, max_tokens)`` and returning
            an object with ``text``, ``truncated``, ``latency_ms``,
            ``prompt_tokens`` and ``completion_tokens``. Passing a callable
            rather than a client keeps this loop testable without a network.
        contract: The Pydantic model the response must satisfy.
        tier: Enforcement tier in use, recorded on every attempt.
        max_attempts: Attempts including the first. Defaults to configuration.
        max_tokens: Starting output ceiling.

    Returns:
        The outcome, carrying every attempt made.
    """
    limit = max_attempts if max_attempts is not None else settings.MAX_REPAIR_ATTEMPTS
    outcome = RepairOutcome(ok=False)
    repair_message: str | None = None
    budget = max_tokens
    previous_raw: str | None = None

    for index in range(1, limit + 1):
        response = generate(repair_message, budget)
        result = validate_response(
            response.text,
            contract,
            truncated=getattr(response, "truncated", False),
            provider_rejected=getattr(response, "rejected_by_provider", False),
        )

        attempt = Attempt(
            index=index,
            tier=tier,
            ok=result.ok,
            failure=result.failure,
            detail=result.detail,
            raw=result.raw,
            latency_ms=getattr(response, "latency_ms", 0.0),
            prompt_tokens=getattr(response, "prompt_tokens", None),
            completion_tokens=getattr(response, "completion_tokens", None),
            max_tokens=budget,
            recovered_by_extraction=result.recovered_by_extraction,
            repaired_from=index - 1 if index > 1 else None,
        )
        outcome.attempts.append(attempt)

        if result.ok:
            outcome.ok = True
            outcome.parsed = result.parsed
            outcome.stopped_because = (
                "validated after local extraction"
                if result.recovered_by_extraction and index == 1
                else "validated"
            )
            return outcome

        if not result.retryable:
            outcome.stopped_because = (
                f"{result.failure.value} cannot be fixed by asking again"
            )
            return outcome

        if index == limit:
            outcome.stopped_because = f"exhausted {limit} attempts"
            return outcome

        # A model that returns byte-identical output has nothing new to say, and
        # further attempts cost the same to reproduce it.
        if previous_raw is not None and result.raw == previous_raw:
            outcome.stopped_because = "the model repeated its previous output"
            return outcome
        previous_raw = result.raw

        if result.failure is FailureType.TRUNCATED:
            if budget is None:
                outcome.stopped_because = (
                    "truncated with no token ceiling to raise"
                )
                return outcome
            # Retrying a truncation at the same ceiling reproduces it exactly.
            budget = budget * TRUNCATION_BUDGET_MULTIPLIER
            repair_message = (
                "Your previous response was cut off before it finished. "
                "Return the complete JSON object, and keep it concise."
            )
            logger.info("raising token ceiling to %d after truncation", budget)
        else:
            repair_message = build_repair_message(result, contract)

        time.sleep(settings.REPAIR_BACKOFF_BASE * (2 ** (index - 1)))

    outcome.stopped_because = f"exhausted {limit} attempts"
    return outcome
