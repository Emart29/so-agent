"""Picks the strongest enforcement a model will honour, and says which it used.

Three tiers, guaranteeing genuinely different things:

============  ==========================================================
json_schema   The provider constrains generation. Shape is guaranteed.
json_object   Valid JSON is guaranteed. The shape is not.
prompt_only   Nothing is guaranteed.
============  ==========================================================

The capability probe measured that only three of eight models on Groq support
the first tier, so this is a working fallback chain rather than a formality.

Two rules matter more than the selection itself:

* **The tier used is recorded on every result.** A silent fall to a weaker tier
  turns "we enforce schemas" into a claim nobody can check, and makes a
  benchmark row uninterpretable.
* **Assistant prefill is never used to force JSON.** It breaks tool use, several
  providers now reject it outright, and the ladder replaces what it was for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from contracts.translate import Translation, response_format_for
from provider.capabilities import ModelCapabilities, TierSupport

#: Strongest first. Selection walks this order and takes the first supported.
TIER_ORDER = ("json_schema", "json_object", "prompt_only")

#: What each tier actually guarantees, for reporting alongside results.
TIER_GUARANTEE = {
    "json_schema": "shape enforced during generation",
    "json_object": "valid JSON, shape not enforced",
    "prompt_only": "nothing enforced",
}


@dataclass
class EnforcementPlan:
    """How one request will be made, and what that buys."""

    tier: str
    request_kwargs: dict[str, Any]
    system_suffix: str | None = None
    downgraded_from: str | None = None

    @property
    def guarantee(self) -> str:
        return TIER_GUARANTEE[self.tier]

    @property
    def is_strongest(self) -> bool:
        return self.tier == TIER_ORDER[0]


def select_tier(
    caps: ModelCapabilities | None,
    requested: str | None = None,
) -> tuple[str, str | None]:
    """Choose the tier to use, and the one it was downgraded from.

    Args:
        caps: Measured capabilities, or ``None`` when the model has not been
            probed. Unprobed models fall to the weakest tier rather than
            optimistically assuming support — guessing produces a benchmark row
            that measures the guess.
        requested: Force a specific tier. Honoured even when unsupported, so the
            benchmark can measure what a tier does on a model that lacks it.

    Returns:
        ``(tier, downgraded_from)``.
    """
    if requested:
        return requested, None

    if caps is None:
        return "prompt_only", "json_schema"

    for tier in TIER_ORDER:
        if tier == "prompt_only":
            return tier, "json_schema"
        if caps.tiers.get(tier) == TierSupport.CONFORMED:
            return tier, None if tier == TIER_ORDER[0] else "json_schema"

    return "prompt_only", "json_schema"


def build_plan(
    name: str,
    translation: Translation,
    caps: ModelCapabilities | None,
    requested_tier: str | None = None,
) -> EnforcementPlan:
    """Build the request configuration for the strongest available tier.

    Args:
        name: Schema name sent to the provider.
        translation: The provider-acceptable schema and what it cost.
        caps: Measured capabilities for the target model.
        requested_tier: Force a tier rather than selecting one.
    """
    tier, downgraded_from = select_tier(caps, requested_tier)

    if tier == "json_schema":
        return EnforcementPlan(
            tier=tier,
            request_kwargs={"response_format": response_format_for(name, translation.schema)},
            downgraded_from=downgraded_from,
        )

    # Below the top tier the schema is no longer enforced by the provider, so it
    # has to travel in the prompt instead — the model cannot satisfy a contract
    # it was never shown.
    instruction = render_schema_instruction(translation.schema)

    if tier == "json_object":
        return EnforcementPlan(
            tier=tier,
            request_kwargs={"response_format": {"type": "json_object"}},
            system_suffix=instruction,
            downgraded_from=downgraded_from,
        )

    return EnforcementPlan(
        tier="prompt_only",
        request_kwargs={},
        system_suffix=instruction + "\n\nReturn only the JSON object, with no other text.",
        downgraded_from=downgraded_from,
    )


def render_schema_instruction(schema: dict[str, Any]) -> str:
    """Render a schema into prompt text, with a worked example beside it.

    The example is not decoration. On smaller models a concrete instance moves
    output shape more reliably than the schema text does, and the weaker tiers
    are exactly where the smaller models end up.
    """
    example = example_from_schema(schema)
    return (
        "Respond with a JSON object matching this schema:\n"
        f"{json.dumps(schema, separators=(',', ':'))}\n\n"
        "Example of a valid response:\n"
        f"{json.dumps(example, indent=2)}"
    )


def example_from_schema(schema: dict[str, Any]) -> Any:
    """Build a minimal instance satisfying a schema, for use as an example."""
    if "anyOf" in schema:
        non_null = [s for s in schema["anyOf"] if s.get("type") != "null"]
        return example_from_schema(non_null[0]) if non_null else None
    if "enum" in schema:
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]

    declared = schema.get("type")
    # A nullable field carries a list of types; the example should show the
    # useful one rather than null.
    if isinstance(declared, list):
        declared = next((t for t in declared if t != "null"), "string")

    if declared == "object":
        return {
            name: example_from_schema(sub)
            for name, sub in schema.get("properties", {}).items()
        }
    if declared == "array":
        return [example_from_schema(schema.get("items", {"type": "string"}))]
    if declared == "integer":
        return 0
    if declared == "number":
        return 0.0
    if declared == "boolean":
        return False
    if declared == "null":
        return None
    return "..."
