"""Parses a response, and classifies precisely what went wrong when it fails.

The taxonomy matters more than the parsing. These are not one failure with
different messages — they need different responses, and treating them alike is
how a retry loop burns money to fail identically:

* ``JSON_WRAPPED`` costs nothing to fix. The JSON is there, inside a fence or
  behind a preamble; extracting it needs no second request at all.
* ``TRUNCATED`` needs a larger token budget. Retrying with the same ceiling
  reproduces the same truncation, for the same price.
* ``REFUSAL`` will not be fixed by asking again in the same words.
* ``SCHEMA_MISMATCH`` is the one repair genuinely helps, because the validation
  error names what to change.

The free repair is attempted before anything is declared a failure. On models
without native enforcement, fences and preambles are most of what goes wrong,
and a loop that pays for a retry to fix a markdown fence is measuring its own
impatience.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, ValidationError


class FailureType(str, Enum):
    """Why a response could not be turned into a validated object."""

    NONE = "none"
    NOT_JSON = "not_json"
    JSON_WRAPPED = "json_wrapped"
    SCHEMA_MISMATCH = "schema_mismatch"
    CONSTRAINT_FAIL = "constraint_fail"
    TRUNCATED = "truncated"
    REFUSAL = "refusal"
    EMPTY = "empty"
    #: The provider enforced the constraint, the generation did not satisfy
    #: it, and the provider refused to return it. Enforcement working, and
    #: distinct from the same mistake caught locally after the fact.
    PROVIDER_REJECTED = "provider_rejected"


#: Failures a second request cannot fix as posed. Retrying these spends the
#: allowance to reproduce the same outcome.
UNRETRYABLE = frozenset({FailureType.REFUSAL, FailureType.EMPTY})

#: Failures fixed locally, with no request at all.
FREE_TO_REPAIR = frozenset({FailureType.JSON_WRAPPED})

#: Openings models use to decline. Matched at the start of the response only:
#: the same words appear mid-answer in perfectly good output about refusals.
REFUSAL_OPENERS = (
    "i cannot", "i can't", "i won't", "i am unable", "i'm unable",
    "sorry, i cannot", "sorry, i can't", "as an ai",
    "i do not feel comfortable", "i don't feel comfortable",
)

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


@dataclass
class ValidationResult:
    """The outcome of turning one response into an object."""

    ok: bool
    failure: FailureType = FailureType.NONE
    parsed: Any = None
    raw: str = ""
    detail: str = ""
    #: Field-level errors, when Pydantic produced them. This is what the repair
    #: prompt feeds back, so it is kept structured rather than flattened.
    field_errors: list[dict[str, Any]] = field(default_factory=list)
    recovered_by_extraction: bool = False

    @property
    def retryable(self) -> bool:
        """Whether asking again could plausibly produce a different result."""
        return self.failure not in UNRETRYABLE and not self.ok

    def error_summary(self) -> str:
        """A short, model-readable statement of what was wrong."""
        if self.ok:
            return ""
        if self.field_errors:
            lines = []
            for err in self.field_errors[:8]:
                location = ".".join(str(p) for p in err.get("loc", ())) or "(root)"
                lines.append(f"{location}: {err.get('msg', 'invalid')}")
            return "\n".join(lines)
        return self.detail


def extract_json(text: str) -> str | None:
    """Pull a JSON object out of surrounding prose or a markdown fence.

    Tried before any failure is declared, because it is the one repair that
    costs nothing. Brace matching is string-aware: a brace inside a string
    value is not structure, and counting naively truncates any object whose
    text contains one.
    """
    if not text:
        return None

    fenced = _FENCE.search(text)
    if fenced:
        candidate = fenced.group(1).strip()
        if candidate:
            return candidate

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def looks_like_refusal(text: str) -> bool:
    """Whether a response opens by declining rather than answering."""
    opening = text.strip().lower()[:120]
    return any(opening.startswith(marker) for marker in REFUSAL_OPENERS)


def validate_response(
    text: str,
    contract: type[BaseModel],
    truncated: bool = False,
    provider_rejected: bool = False,
) -> ValidationResult:
    """Turn a raw response into a validated object, or say precisely why not.

    Args:
        text: The model's response.
        contract: Pydantic model the response must satisfy.
        truncated: Whether generation stopped at the token ceiling. Passed in
            rather than guessed, because a truncated object is often still
            syntactically plausible and would otherwise be misreported as a
            schema mismatch — sending it back for repair instead of raising the
            budget that actually caused it.
        provider_rejected: Whether the provider refused the generation because
            it failed the constraint being enforced. Passed in for the same
            reason as ``truncated``: the local classification would otherwise
            call an empty body "the model returned nothing", which is a
            different event with a different remedy.

    Returns:
        The result, carrying the failure type and the detail a repair needs.
    """
    raw = text or ""

    if provider_rejected:
        # Retryable on purpose. The provider caught the mistake instead of the
        # validator, which says nothing about whether asking again would work —
        # and treating an empty rejection as an empty response would stop the
        # repair loop dead on the tier where enforcement is doing its job.
        inner = _validate_object(_try_parse(raw), contract, raw, recovered=False)             if _try_parse(raw) is not None else None
        return ValidationResult(
            ok=False,
            failure=FailureType.PROVIDER_REJECTED,
            raw=raw,
            detail=(
                inner.detail if inner is not None and inner.detail
                else "the provider refused the generation: it did not satisfy "
                     "the schema being enforced"
            ),
            field_errors=inner.field_errors if inner is not None else [],
        )

    if not raw.strip():
        # Truncation is reported first: an empty body after hitting the ceiling
        # is a budget problem, and calling it "empty" hides the cause.
        if truncated:
            return ValidationResult(
                ok=False, failure=FailureType.TRUNCATED, raw=raw,
                detail="generation stopped at the token limit before any output",
            )
        return ValidationResult(
            ok=False, failure=FailureType.EMPTY, raw=raw,
            detail="the model returned no content",
        )

    if looks_like_refusal(raw):
        return ValidationResult(
            ok=False, failure=FailureType.REFUSAL, raw=raw,
            detail=f"the model declined: {raw.strip()[:120]}",
        )

    direct = _try_parse(raw)
    if direct is not None:
        return _validate_object(direct, contract, raw, recovered=False)

    extracted = extract_json(raw)
    if extracted is not None:
        parsed = _try_parse(extracted)
        if parsed is not None:
            result = _validate_object(parsed, contract, raw, recovered=True)
            if result.ok:
                # Structurally correct, merely wrapped. Recorded as a distinct
                # failure so the benchmark can report how much of the raw
                # failure rate needs no retry at all.
                result.failure = FailureType.JSON_WRAPPED
            return result

    if truncated:
        return ValidationResult(
            ok=False, failure=FailureType.TRUNCATED, raw=raw,
            detail="generation stopped at the token limit mid-object",
        )

    return ValidationResult(
        ok=False, failure=FailureType.NOT_JSON, raw=raw,
        detail="the response did not contain a JSON object",
    )


def _try_parse(text: str) -> Any | None:
    """Parse JSON, returning ``None`` rather than raising."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _validate_object(
    data: Any, contract: type[BaseModel], raw: str, recovered: bool
) -> ValidationResult:
    """Validate parsed data against the contract."""
    try:
        parsed = contract.model_validate(data)
    except ValidationError as exc:
        errors = exc.errors()
        return ValidationResult(
            ok=False,
            # A bound violated is a different problem from a shape violated: the
            # provider never enforced the bound, so this is the translator's
            # stripped constraint surfacing rather than the model ignoring the
            # schema. Separating them keeps the two causes distinguishable in
            # the benchmark.
            failure=(
                FailureType.CONSTRAINT_FAIL
                if _is_constraint_only(errors)
                else FailureType.SCHEMA_MISMATCH
            ),
            raw=raw,
            detail=str(exc).splitlines()[0],
            field_errors=errors,
            recovered_by_extraction=recovered,
        )

    return ValidationResult(
        ok=True, parsed=parsed, raw=raw, recovered_by_extraction=recovered
    )


#: Pydantic error types raised by a value constraint rather than a shape
#: problem. These correspond to the keywords the translator strips.
_CONSTRAINT_ERROR_TYPES = frozenset(
    {
        "greater_than", "greater_than_equal", "less_than", "less_than_equal",
        "multiple_of", "string_too_short", "string_too_long",
        "string_pattern_mismatch", "too_short", "too_long",
    }
)


def _is_constraint_only(errors: list[dict[str, Any]]) -> bool:
    """Whether every error is a bound violation rather than a shape problem."""
    return bool(errors) and all(
        err.get("type") in _CONSTRAINT_ERROR_TYPES for err in errors
    )
