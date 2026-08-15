"""Tests for discriminated unions and tool-call validation.

The union failure worth catching is a payload that contradicts its own tag:
`{"kind": "answer", "question": "..."}` parses as JSON, looks structured, and
means the opposite of what it claims. The tag has to select the variant before
the fields are checked, or the mismatch is silently coerced away.

Tool arguments get no special trust. They are model output that arrived in a
different envelope, and the probe measured a model that accepts a tool
definition and never calls it.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field, ValidationError

from contracts.translate import SchemaTranslator
from contracts.unions import (
    VARIANTS,
    Answer,
    Clarification,
    DecisionEnvelope,
    Lookup,
    parse_decision,
    variant_for,
)
from enforce.tools import ToolFailure, tool_definition, validate_tool_call


class FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, name, arguments):
        self.function = FakeFunction(name, arguments)


ANSWER = {
    "kind": "answer",
    "category": "billing",
    "priority": "high",
    "summary": "Double charge.",
}


class TestUnions:
    def test_each_variant_parses_to_its_own_type(self):
        assert isinstance(parse_decision(ANSWER), Answer)
        assert isinstance(
            parse_decision(
                {"kind": "clarification", "question": "Which invoice?", "reason": "vague"}
            ),
            Clarification,
        )
        assert isinstance(
            parse_decision({"kind": "lookup", "resource": "billing", "identifier": "42"}),
            Lookup,
        )

    def test_a_payload_contradicting_its_tag_is_rejected(self):
        """The failure this union exists to catch."""
        with pytest.raises(ValidationError):
            parse_decision({"kind": "answer", "question": "Which invoice?"})

    def test_an_unknown_tag_is_rejected(self):
        with pytest.raises(ValidationError):
            parse_decision({"kind": "escalate", "summary": "x"})

    def test_a_missing_tag_is_rejected(self):
        with pytest.raises(ValidationError):
            parse_decision({"category": "billing", "priority": "high", "summary": "x"})

    def test_fields_from_another_variant_are_rejected(self):
        """Otherwise a mixed payload validates as whichever variant it named."""
        with pytest.raises(ValidationError):
            parse_decision({**ANSWER, "question": "Which invoice?"})

    def test_variant_lookup_names_the_alternatives(self):
        assert variant_for("answer") is Answer
        with pytest.raises(KeyError, match="clarification"):
            variant_for("nope")

    def test_every_variant_is_registered(self):
        """A variant missing from the table breaks the two-call fallback."""
        assert set(VARIANTS) == {"answer", "clarification", "lookup"}


class TestUnionSchema:
    def test_the_envelope_translates_to_a_provider_schema(self):
        result = SchemaTranslator().translate(DecisionEnvelope)
        assert result.schema["type"] == "object"
        assert "decision" in result.schema["properties"]

    def test_the_union_survives_translation(self):
        """Pydantic emits oneOf for a discriminated union, not anyOf."""
        schema = SchemaTranslator(inline_refs=True).translate(DecisionEnvelope).schema
        assert "oneOf" in schema["properties"]["decision"]
        assert len(schema["properties"]["decision"]["oneOf"]) == len(VARIANTS)

    def test_the_discriminator_annotation_is_stripped(self):
        """It is an OpenAPI keyword a strict validator may reject, and its
        mapping holds $defs pointers that dangle once definitions are inlined."""
        schema = SchemaTranslator(inline_refs=True).translate(DecisionEnvelope).schema
        assert "discriminator" not in json.dumps(schema)
        assert "$defs" not in json.dumps(schema)

    def test_each_variant_keeps_its_const_tag(self):
        """The tag is what determines the variant once the discriminator is gone."""
        schema = SchemaTranslator(inline_refs=True).translate(DecisionEnvelope).schema
        tags = {
            v["properties"]["kind"]["const"]
            for v in schema["properties"]["decision"]["oneOf"]
        }
        assert tags == set(VARIANTS)

    def test_translation_keeps_every_variant(self):
        schema = json.dumps(
            SchemaTranslator(inline_refs=True).translate(DecisionEnvelope).schema
        )
        for tag in VARIANTS:
            assert tag in schema


class TestToolDefinitions:
    def test_a_definition_is_built_from_a_model(self):
        class SearchArgs(BaseModel):
            query: str = Field(description="What to search for.")

        definition = tool_definition("search", "Search tickets.", SearchArgs)
        assert definition["type"] == "function"
        assert definition["function"]["name"] == "search"
        assert "query" in definition["function"]["parameters"]["properties"]

    def test_tool_schemas_go_through_the_same_translator(self):
        """Strict-mode rules apply here too; an untranslated optional is rejected."""

        class Args(BaseModel):
            required_field: str = Field(description="r")
            optional_field: str | None = Field(default=None, description="o")

        parameters = tool_definition("t", "d", Args)["function"]["parameters"]
        assert set(parameters["required"]) == {"required_field", "optional_field"}


class TestToolValidation:
    def _expected(self):
        class SearchArgs(BaseModel):
            query: str
            limit: int = 10

        return {"search": SearchArgs}, SearchArgs

    def test_a_well_formed_call_validates(self):
        expected, _ = self._expected()
        result = validate_tool_call(
            [FakeToolCall("search", '{"query": "refund", "limit": 5}')], expected
        )
        assert result.ok
        assert result.arguments.query == "refund"

    def test_prose_instead_of_a_call_is_reported(self):
        """The probe measured a model that accepts a tool and never calls it."""
        expected, _ = self._expected()
        result = validate_tool_call([], expected)
        assert not result.ok
        assert result.failure is ToolFailure.NO_CALL

    def test_a_tool_that_was_never_offered_is_rejected(self):
        expected, _ = self._expected()
        result = validate_tool_call([FakeToolCall("delete_account", "{}")], expected)
        assert result.failure is ToolFailure.UNKNOWN_TOOL
        assert "search" in result.detail

    def test_unparseable_arguments_are_an_ordinary_parse_failure(self):
        expected, _ = self._expected()
        result = validate_tool_call(
            [FakeToolCall("search", "{'query': 'refund'}")], expected
        )
        assert result.failure is ToolFailure.ARGS_NOT_JSON
        assert result.raw_arguments

    def test_arguments_that_parse_but_mismatch_are_separated(self):
        expected, _ = self._expected()
        result = validate_tool_call(
            [FakeToolCall("search", '{"limit": "five"}')], expected
        )
        assert result.failure is ToolFailure.ARGS_MISMATCH
        assert result.field_errors

    def test_extra_calls_are_reported_not_silently_dropped(self):
        """The discarded call may have been the one that mattered."""
        expected, _ = self._expected()
        result = validate_tool_call(
            [FakeToolCall("search", '{"query": "a"}'),
             FakeToolCall("search", '{"query": "b"}')],
            expected,
        )
        assert not result.ok
        assert "got 2" in result.detail

    def test_parallel_calls_are_allowed_when_expected(self):
        expected, _ = self._expected()
        result = validate_tool_call(
            [FakeToolCall("search", '{"query": "a"}'),
             FakeToolCall("search", '{"query": "b"}')],
            expected,
            require_one=False,
        )
        assert result.ok

    def test_dict_shaped_calls_are_handled(self):
        """Not every provider returns SDK objects."""
        expected, _ = self._expected()
        result = validate_tool_call(
            [{"function": {"name": "search", "arguments": '{"query": "x"}'}}], expected
        )
        assert result.ok

    def test_empty_arguments_validate_when_the_schema_allows_it(self):
        class NoArgs(BaseModel):
            pass

        result = validate_tool_call([FakeToolCall("ping", "")], {"ping": NoArgs})
        assert result.ok
