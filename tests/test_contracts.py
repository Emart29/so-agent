"""Tests for the contracts, the schema translator, and the enforcement ladder.

The translator's job is to stop a provider rejecting a request without weakening
the contract. Those two goals pull against each other, and the failure mode is
silent: a constraint dropped to satisfy the provider and then never enforced
anywhere produces a pipeline that validates everything and guarantees nothing.
Most of what follows pins that boundary.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

from contracts.schemas import (
    CONTRACTS,
    Customer,
    Difficulty,
    TicketAnalysis,
    TicketSummary,
    TicketTriage,
    get_contract,
)
from contracts.translate import (
    COSMETIC_KEYS,
    SchemaTranslator,
    VALUE_CONSTRAINTS,
    SchemaTranslator,
    response_format_for,
)
from enforce.ladder import (
    TIER_ORDER,
    build_plan,
    example_from_schema,
    render_schema_instruction,
    select_tier,
)
from provider.capabilities import ModelCapabilities, TierSupport


def caps_with(**tiers) -> ModelCapabilities:
    return ModelCapabilities(
        provider="groq", model="m", probed_at="2026-08-14T00:00:00+00:00", tiers=tiers
    )


class TestContracts:
    def test_every_contract_declares_a_difficulty(self):
        for name, contract in CONTRACTS.items():
            assert isinstance(contract.difficulty(), Difficulty), name

    def test_difficulties_span_the_range(self):
        """A benchmark cannot ask whether complexity predicts failure with one level."""
        levels = {c.difficulty() for c in CONTRACTS.values()}
        assert levels == set(Difficulty)

    def test_every_field_carries_a_description(self):
        """Descriptions ship to the model; a missing one is a silent quality loss."""
        for name, contract in CONTRACTS.items():
            for field_name, info in contract.model_fields.items():
                assert info.description, f"{name}.{field_name} has no description"

    def test_unknown_contract_names_the_alternatives(self):
        with pytest.raises(KeyError, match="ticket_summary"):
            get_contract("nope")

    def test_the_hard_contract_carries_both_measured_landmines(self):
        """It must exercise the translator, not just the model."""
        schema = TicketTriage.model_json_schema()
        assert "assignee" not in schema["required"], "needs the rejected optional form"
        assert "minimum" in schema["properties"]["confidence"], "needs a stripped bound"


class TestTranslation:
    def test_optional_fields_become_nullable_and_required(self):
        """The exact rewrite Groq's strict mode demands."""
        result = SchemaTranslator().translate(TicketTriage)

        assert "assignee" in result.schema["required"]
        assert result.made_nullable == ["assignee"]
        variants = result.schema["properties"]["assignee"]["anyOf"]
        assert {"type": "null"} in variants

    def test_a_nullable_rewrite_keeps_the_description(self):
        """The description is shipped to the model and must survive the rewrite."""
        prop = SchemaTranslator().translate(TicketTriage).schema["properties"]["assignee"]
        assert prop.get("description")

    def test_value_constraints_are_reported_not_silently_dropped(self):
        """A dropped-and-forgotten bound is how invalid data passes validation."""
        result = SchemaTranslator().translate(TicketTriage)

        stripped = {(c.keyword, c.value) for c in result.stripped}
        assert ("minimum", 0.0) in stripped
        assert ("maximum", 1.0) in stripped
        assert not result.is_lossless

    def test_stripped_constraints_name_their_location(self):
        result = SchemaTranslator().translate(TicketTriage)
        assert all("confidence" in c.path for c in result.stripped)

    def test_pydantic_still_enforces_what_the_provider_cannot(self):
        """The bound leaves the schema; it must not leave the contract."""
        payload = {
            "customer": {"name": "A", "account_tier": "pro"},
            "issues": [{"description": "d", "category": "billing"}],
            "priority": "high",
            "sentiment": "angry",
            "steps": [{"action": "a", "requires_customer_reply": False}],
            "confidence": 40.0,
            "assignee": None,
        }
        with pytest.raises(Exception):
            TicketTriage.model_validate(payload)

    def test_objects_are_closed(self):
        """An open object lets a model invent fields the contract never declared."""
        schema = SchemaTranslator().translate(TicketAnalysis).schema
        assert schema["additionalProperties"] is False

    def test_nested_objects_are_closed_too(self):
        schema = SchemaTranslator().translate(TicketAnalysis).schema
        customer = schema["$defs"]["Customer"]
        assert customer["additionalProperties"] is False

    def test_cosmetic_keys_are_dropped(self):
        """They cost tokens on every request and change nothing about generation."""
        schema = SchemaTranslator().translate(TicketSummary).schema
        assert "title" not in json.dumps(schema)

    def test_a_contract_with_no_constraints_is_lossless(self):
        result = SchemaTranslator().translate(TicketSummary)
        assert result.is_lossless
        assert "no constraints stripped" in result.report()

    def test_report_names_what_the_provider_will_not_enforce(self):
        report = SchemaTranslator().translate(TicketTriage).report()
        assert "NOT enforced" in report
        assert "confidence" in report

    def test_stripping_can_be_disabled(self):
        result = SchemaTranslator(strip_value_constraints=False).translate(TicketTriage)
        assert result.stripped == []
        assert "minimum" in result.schema["properties"]["confidence"]

    def test_nullable_rewrite_can_be_disabled(self):
        """Providers that accept partial `required` do not need the rewrite."""
        result = SchemaTranslator(nullable_optionals=False).translate(TicketTriage)
        assert "assignee" not in result.schema["required"]
        assert result.made_nullable == []

    def test_refs_survive_by_default(self):
        """Both providers probed accept $ref, so inlining is opt-in."""
        result = SchemaTranslator().translate(TicketAnalysis)
        assert "$defs" in result.schema
        assert result.inlined_refs == 0

    def test_refs_can_be_inlined(self):
        result = SchemaTranslator(inline_refs=True).translate(TicketAnalysis)
        assert "$defs" not in result.schema
        assert result.inlined_refs > 0
        assert result.schema["properties"]["customer"]["type"] == "object"

    def test_translation_is_stable_across_runs(self):
        """An unchanged contract must produce an identical schema, or provider-side
        schema caches are invalidated on every request for no reason."""
        a = SchemaTranslator().translate(TicketTriage).schema
        b = SchemaTranslator().translate(TicketTriage).schema
        assert json.dumps(a, sort_keys=False) == json.dumps(b, sort_keys=False)

    def test_already_nullable_types_are_not_widened_twice(self):
        translator = SchemaTranslator()
        result = translator.translate_schema(
            {
                "type": "object",
                "properties": {"x": {"type": ["string", "null"]}},
                "required": [],
            }
        )
        assert result.schema["properties"]["x"]["type"] == ["string", "null"]

    def test_response_format_marks_the_schema_strict(self):
        directive = response_format_for("n", {"type": "object"})
        assert directive["type"] == "json_schema"
        assert directive["json_schema"]["strict"] is True


class TestTierSelection:
    def test_the_strongest_supported_tier_wins(self):
        caps = caps_with(
            json_schema=TierSupport.CONFORMED.value,
            json_object=TierSupport.CONFORMED.value,
        )
        assert select_tier(caps) == ("json_schema", None)

    def test_selection_falls_back_when_the_top_tier_is_rejected(self):
        caps = caps_with(
            json_schema=TierSupport.REJECTED.value,
            json_object=TierSupport.CONFORMED.value,
        )
        tier, downgraded = select_tier(caps)
        assert tier == "json_object"
        assert downgraded == "json_schema"

    def test_an_ignored_tier_is_never_selected(self):
        """Accepted-and-not-honoured is worse than rejected: it looks like it worked."""
        caps = caps_with(
            json_schema=TierSupport.IGNORED.value,
            json_object=TierSupport.CONFORMED.value,
        )
        assert select_tier(caps)[0] == "json_object"

    def test_an_unprobed_model_falls_to_the_weakest_tier(self):
        """Assuming support would produce a benchmark row measuring the assumption."""
        tier, downgraded = select_tier(None)
        assert tier == "prompt_only"
        assert downgraded == "json_schema"

    def test_a_requested_tier_is_honoured_even_when_unsupported(self):
        """The benchmark must be able to measure a tier on a model that lacks it."""
        caps = caps_with(json_schema=TierSupport.REJECTED.value)
        assert select_tier(caps, requested="json_schema") == ("json_schema", None)

    def test_tier_order_runs_strongest_to_weakest(self):
        assert TIER_ORDER[0] == "json_schema"
        assert TIER_ORDER[-1] == "prompt_only"


class TestPlans:
    def _translation(self, contract=TicketSummary):
        return SchemaTranslator().translate(contract)

    def test_the_top_tier_sends_a_schema_and_no_prompt_text(self):
        plan = build_plan("t", self._translation(), caps_with(
            json_schema=TierSupport.CONFORMED.value))

        assert plan.tier == "json_schema"
        assert plan.request_kwargs["response_format"]["type"] == "json_schema"
        assert plan.system_suffix is None
        assert plan.is_strongest

    def test_weaker_tiers_carry_the_schema_in_the_prompt(self):
        """The model cannot satisfy a contract it was never shown."""
        plan = build_plan("t", self._translation(), caps_with(
            json_schema=TierSupport.REJECTED.value,
            json_object=TierSupport.CONFORMED.value,
        ))

        assert plan.tier == "json_object"
        assert plan.request_kwargs["response_format"] == {"type": "json_object"}
        assert plan.system_suffix and "schema" in plan.system_suffix.lower()

    def test_the_weakest_tier_sends_no_directive_at_all(self):
        plan = build_plan("t", self._translation(), None)

        assert plan.tier == "prompt_only"
        assert plan.request_kwargs == {}
        assert "only the JSON object" in plan.system_suffix

    def test_a_downgrade_is_always_recorded(self):
        """A silent fall to a weaker tier makes the result uninterpretable."""
        plan = build_plan("t", self._translation(), None)
        assert plan.downgraded_from == "json_schema"
        assert not plan.is_strongest

    def test_every_tier_states_what_it_guarantees(self):
        for tier in TIER_ORDER:
            plan = build_plan("t", self._translation(), None, requested_tier=tier)
            assert plan.guarantee


class TestPromptRendering:
    def test_the_instruction_carries_a_worked_example(self):
        """On smaller models the example moves output shape more than the schema."""
        schema = SchemaTranslator().translate(TicketSummary).schema
        text = render_schema_instruction(schema)

        assert "Example of a valid response" in text
        example_part = text.split("Example of a valid response:")[1]
        json.loads(example_part.strip())

    def test_examples_pick_a_real_enum_member(self):
        assert example_from_schema({"enum": ["low", "high"]}) == "low"

    def test_examples_avoid_null_for_nullable_fields(self):
        """Showing null teaches the model that omitting the value is the norm."""
        assert example_from_schema({"type": ["string", "null"]}) == "..."

    def test_examples_recurse_through_nesting(self):
        schema = SchemaTranslator().translate(TicketAnalysis, ).schema
        example = example_from_schema(
            {"type": "object", "properties": {"items": {
                "type": "array",
                "items": {"type": "object", "properties": {"n": {"type": "integer"}}},
            }}}
        )
        assert example == {"items": [{"n": 0}]}

    def test_generated_examples_are_serialisable(self):
        for contract in CONTRACTS.values():
            schema = SchemaTranslator(inline_refs=True).translate(contract).schema
            json.dumps(example_from_schema(schema))


class TestConstraintTables:
    def test_bounds_and_lengths_are_treated_as_value_constraints(self):
        for keyword in ("minimum", "maximum", "minLength", "pattern", "minItems"):
            assert keyword in VALUE_CONSTRAINTS

    def test_structural_keywords_are_never_stripped(self):
        """Removing these would change the contract rather than relax it."""
        for keyword in ("type", "properties", "required", "items", "enum", "anyOf"):
            assert keyword not in VALUE_CONSTRAINTS
            assert keyword not in COSMETIC_KEYS


class TestArbitraryModels:
    def test_a_model_with_only_optionals_still_requires_everything(self):
        class AllOptional(BaseModel):
            a: str | None = None
            b: int | None = None

        result = SchemaTranslator().translate(AllOptional)
        assert set(result.schema["required"]) == {"a", "b"}
        assert len(result.made_nullable) == 2

    def test_a_constrained_list_reports_its_bounds(self):
        class Bounded(BaseModel):
            items: list[str] = Field(min_length=1, max_length=5)

        result = SchemaTranslator().translate(Bounded)
        assert {c.keyword for c in result.stripped} & {"minItems", "maxItems"}


class TestAbsenceIsExpressible:
    """A field the source may omit must have a legal way to say "not stated".

    Without one the model has to write something, and the benchmark then scores
    the schema's mistake as the model inventing a fact. An earlier run measured
    exactly that: 292 of the flagged grounding failures were the value the field
    description itself asked for.
    """

    @pytest.mark.parametrize("field", ["name", "account_tier"])
    def test_a_customer_detail_accepts_null(self, field):
        customer = Customer(**{"name": None, "account_tier": None})
        assert getattr(customer, field) is None

    def test_a_class_docstring_does_not_bloat_every_request(self):
        """Docstrings are serialised into the schema and sent on every call, so
        rationale for the reader belongs in a comment."""
        for model in (Customer, TicketAnalysis, TicketTriage):
            assert len(model.__doc__.strip()) < 200, model.__name__

    def test_the_optional_assignee_still_accepts_null(self):
        assert TicketTriage.model_fields["assignee"].default is None

    @pytest.mark.parametrize("name", ["name", "account_tier"])
    def test_the_description_asks_for_null_rather_than_a_placeholder(self, name):
        """The instruction and the scoring have to agree on what absence looks
        like; a description asking for "unknown" contradicts the check."""
        description = Customer.model_fields[name].description.lower()
        assert "null" in description
        assert "placeholder" in description

    def test_nullable_fields_stay_required_after_translation(self):
        """Groq rejects a schema that omits a key from `required`, so optionality
        is carried by the type rather than by absence from the list."""
        translated = SchemaTranslator().translate(TicketAnalysis).schema
        customer = translated["$defs"]["Customer"]
        assert set(customer["required"]) == {"name", "account_tier"}
        # Pydantic expresses `str | None` as anyOf rather than a type list. The
        # probe measured both forms as accepted, so the shape is checked rather
        # than assumed to be one of them.
        variants = customer["properties"]["name"]["anyOf"]
        assert {v["type"] for v in variants} == {"string", "null"}
