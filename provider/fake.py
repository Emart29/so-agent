"""A scripted stand-in for a provider, for testing the loop deterministically.

The repair loop cannot be proven against a live model. A model that happens to
succeed first time exercises none of it, and one that fails does so differently
on every run — so the branches that matter would be tested by luck or not at all.

Scripting the responses makes the loop's behaviour assertable: fail unparseably,
then fail validation, then succeed, and check the loop did exactly the right
thing at each step and stopped for the right reason.

The malformed fixtures below are real. Every one was produced by an actual model
during the capability probe or the contract checks, rather than invented — a
suite that defends only against imagined failures proves the code handles
imaginary problems.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class FakeResponse:
    """Shaped like the real client's result, carrying only what the loop reads."""

    text: str
    truncated: bool = False
    latency_ms: float = 1.0
    prompt_tokens: int = 10
    completion_tokens: int = 10


VALID_SUMMARY = json.dumps(
    {
        "category": "billing",
        "priority": "high",
        "summary": "Customer was charged twice and wants a refund.",
    }
)

#: Failure shapes observed from real models. The first three are what actually
#: happens on the weaker enforcement tiers; the rest come from validation.
MALFORMED: dict[str, str] = {
    # Seen constantly on prompt-only tiers. Structurally perfect JSON that
    # json.loads rejects purely because of the fence.
    "fenced": f"```json\n{VALID_SUMMARY}\n```",
    "preamble": f"Here is the triage:\n\n{VALID_SUMMARY}",
    "trailing_prose": f"{VALID_SUMMARY}\n\nLet me know if you need anything else.",
    # Reasoning models emit this when asked for JSON without enforcement.
    "thinking_block": (
        "<think>The user wants JSON with category, priority and summary. "
        "This is a billing issue and it sounds urgent.</think>\n"
        f"{VALID_SUMMARY}"
    ),
    "single_quotes": VALID_SUMMARY.replace('"', "'"),
    "trailing_comma": VALID_SUMMARY[:-1] + ",}",
    "truncated_object": VALID_SUMMARY[: len(VALID_SUMMARY) // 2],
    "wrong_enum": json.dumps(
        {"category": "billing", "priority": "extremely urgent", "summary": "x"}
    ),
    "missing_field": json.dumps({"category": "billing", "priority": "high"}),
    "extra_field": json.dumps(
        {
            "category": "billing",
            "priority": "high",
            "summary": "x",
            "confidence_note": "fairly sure",
        }
    ),
    "null_required": json.dumps(
        {"category": None, "priority": "high", "summary": "x"}
    ),
    "wrong_schema_entirely": json.dumps({"city": "Tokyo", "population": 37000000}),
    "prose_only": "This ticket is about billing and seems fairly urgent.",
    "empty": "",
    "refusal": "I cannot help with that request.",
}

#: A confidence outside its bounds. Separated because the provider never
#: enforced the bound — it is the translator's stripped constraint surfacing,
#: not the model ignoring a schema.
CONSTRAINT_VIOLATION = json.dumps(
    {
        "customer": {"name": "Marta", "account_tier": "business"},
        "issues": [{"description": "Double charge", "category": "billing"}],
        "priority": "high",
        "sentiment": "frustrated",
        "steps": [{"action": "Refund", "requires_customer_reply": False}],
        "confidence": 40.0,
        "assignee": None,
    }
)


class ScriptedClient:
    """Returns pre-arranged responses in order.

    Args:
        script: Responses to return, one per call. Each may be a string, a
            ``FakeResponse``, or the name of a fixture in ``MALFORMED``.
        loop_last: Repeat the final response once the script runs out, rather
            than raising. Useful for asserting that the loop stops on repeated
            identical output.
    """

    def __init__(self, script: list[Any], loop_last: bool = False) -> None:
        self.script = list(script)
        self.loop_last = loop_last
        self.calls: list[dict[str, Any]] = []

    def __call__(self, repair_message: str | None = None, max_tokens: int | None = None):
        """Match the signature the repair loop calls its generator with."""
        self.calls.append({"repair_message": repair_message, "max_tokens": max_tokens})
        index = len(self.calls) - 1

        if index >= len(self.script):
            if not self.loop_last or not self.script:
                raise AssertionError(
                    f"scripted client called {len(self.calls)} times but only "
                    f"{len(self.script)} responses were scripted"
                )
            item = self.script[-1]
        else:
            item = self.script[index]

        return _as_response(item)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def repair_messages(self) -> list[str]:
        """Repair prompts sent, excluding the initial call's ``None``."""
        return [c["repair_message"] for c in self.calls if c["repair_message"]]

    @property
    def token_budgets(self) -> list[int | None]:
        return [c["max_tokens"] for c in self.calls]


def _as_response(item: Any) -> FakeResponse:
    """Turn a script entry into a response."""
    if isinstance(item, FakeResponse):
        return item
    if isinstance(item, str):
        return FakeResponse(text=MALFORMED.get(item, item))
    raise TypeError(f"cannot script a {type(item).__name__}")
