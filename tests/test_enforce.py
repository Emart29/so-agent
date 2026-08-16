"""Tests for the failure taxonomy and the repair loop.

Two things are being pinned. The taxonomy must separate failures that need
different responses — retrying a refusal or a truncation at the same budget
spends the allowance to reproduce the outcome exactly. And the loop must stop
for the right reason, since "3 attempts, failed" is indistinguishable from
"1 attempt, then two identical repeats" unless it is asserted.

Everything here runs against a scripted client. A live model that happens to
succeed on the first call exercises none of this.
"""

from __future__ import annotations

import json

import pytest

from contracts.schemas import TicketSummary, TicketTriage
from enforce.repair import (
    TRUNCATION_BUDGET_MULTIPLIER,
    build_repair_message,
    repair_loop,
)
from enforce.validate import (
    FailureType,
    extract_json,
    looks_like_refusal,
    validate_response,
)
from provider.fake import CONSTRAINT_VIOLATION, MALFORMED, VALID_SUMMARY, ScriptedClient


def run(script, contract=TicketSummary, max_attempts=3, max_tokens=500, loop_last=False):
    client = ScriptedClient(script, loop_last=loop_last)
    outcome = repair_loop(
        client, contract, tier="json_object",
        max_attempts=max_attempts, max_tokens=max_tokens,
    )
    return outcome, client


class TestExtraction:
    def test_a_fenced_object_is_recovered(self):
        assert json.loads(extract_json(MALFORMED["fenced"]))["category"] == "billing"

    def test_a_preamble_is_stripped(self):
        assert json.loads(extract_json(MALFORMED["preamble"]))["priority"] == "high"

    def test_trailing_prose_is_ignored(self):
        assert json.loads(extract_json(MALFORMED["trailing_prose"]))["priority"] == "high"

    def test_a_reasoning_block_is_stripped(self):
        assert json.loads(extract_json(MALFORMED["thinking_block"]))["category"] == "billing"

    def test_braces_inside_strings_do_not_end_the_object(self):
        """Naive brace counting truncates any value containing a brace."""
        text = 'Here: {"summary": "use {curly} braces", "category": "other"}'
        assert json.loads(extract_json(text))["summary"] == "use {curly} braces"

    def test_escaped_quotes_do_not_end_the_string(self):
        text = r'{"summary": "she said \"hi\"", "category": "other"}'
        assert json.loads(extract_json(text))["summary"] == 'she said "hi"'

    def test_prose_with_no_object_yields_nothing(self):
        assert extract_json(MALFORMED["prose_only"]) is None

    def test_empty_input_yields_nothing(self):
        assert extract_json("") is None


class TestRefusalDetection:
    @pytest.mark.parametrize(
        "text", ["I cannot help with that.", "I'm unable to do this.", "As an AI, I..."]
    )
    def test_openers_are_detected(self, text):
        assert looks_like_refusal(text)

    def test_the_same_words_mid_answer_are_not_a_refusal(self):
        """Output *about* refusals is a valid answer, not a refusal."""
        assert not looks_like_refusal(
            '{"summary": "The agent said I cannot process this", "category": "other"}'
        )


class TestTaxonomy:
    def test_valid_output_passes(self):
        result = validate_response(VALID_SUMMARY, TicketSummary)
        assert result.ok
        assert result.failure is FailureType.NONE

    def test_a_fence_is_recovered_and_recorded_as_wrapped(self):
        """Recorded distinctly so the benchmark can report the free-repair share."""
        result = validate_response(MALFORMED["fenced"], TicketSummary)
        assert result.ok
        assert result.failure is FailureType.JSON_WRAPPED
        assert result.recovered_by_extraction

    @pytest.mark.parametrize(
        "fixture,expected",
        [
            ("prose_only", FailureType.NOT_JSON),
            ("single_quotes", FailureType.NOT_JSON),
            ("wrong_enum", FailureType.SCHEMA_MISMATCH),
            ("missing_field", FailureType.SCHEMA_MISMATCH),
            ("extra_field", FailureType.SCHEMA_MISMATCH),
            ("null_required", FailureType.SCHEMA_MISMATCH),
            ("wrong_schema_entirely", FailureType.SCHEMA_MISMATCH),
            ("empty", FailureType.EMPTY),
            ("refusal", FailureType.REFUSAL),
        ],
    )
    def test_real_failures_classify_correctly(self, fixture, expected):
        assert validate_response(MALFORMED[fixture], TicketSummary).failure is expected

    def test_a_violated_bound_is_not_a_schema_mismatch(self):
        """The provider never enforced it — this is a stripped constraint surfacing."""
        result = validate_response(CONSTRAINT_VIOLATION, TicketTriage)
        assert result.failure is FailureType.CONSTRAINT_FAIL

    def test_truncation_is_reported_as_truncation(self):
        """Otherwise it is repaired instead of given the budget it needs."""
        result = validate_response(
            MALFORMED["truncated_object"], TicketSummary, truncated=True
        )
        assert result.failure is FailureType.TRUNCATED

    def test_an_empty_body_after_truncation_reports_the_cause(self):
        result = validate_response("", TicketSummary, truncated=True)
        assert result.failure is FailureType.TRUNCATED

    def test_refusals_and_empties_are_not_retryable(self):
        for fixture in ("refusal", "empty"):
            assert not validate_response(MALFORMED[fixture], TicketSummary).retryable

    def test_a_schema_mismatch_is_retryable(self):
        assert validate_response(MALFORMED["wrong_enum"], TicketSummary).retryable

    def test_field_errors_name_the_offending_field(self):
        """This text is what the repair prompt feeds back."""
        summary = validate_response(MALFORMED["wrong_enum"], TicketSummary).error_summary()
        assert "priority" in summary


class TestRepairLoop:
    def test_a_first_time_success_makes_one_call(self):
        outcome, client = run([VALID_SUMMARY])
        assert outcome.ok and outcome.first_attempt_ok
        assert client.call_count == 1

    def test_a_fence_costs_no_retry_at_all(self):
        """The free repair. Paying for a retry here measures impatience."""
        outcome, client = run(["fenced"])
        assert outcome.ok
        assert client.call_count == 1
        assert outcome.needed_only_extraction

    def test_a_schema_error_is_repaired_on_the_second_attempt(self):
        outcome, client = run(["wrong_enum", VALID_SUMMARY])
        assert outcome.ok
        assert not outcome.first_attempt_ok
        assert client.call_count == 2
        assert outcome.error_sequence == ["schema_mismatch", "none"]

    def test_the_repair_prompt_names_the_broken_field(self):
        """A generic 'try again' gives the model nothing to change."""
        _, client = run(["wrong_enum", VALID_SUMMARY])
        assert "priority" in client.repair_messages[0]

    def test_a_refusal_stops_the_loop_immediately(self):
        outcome, client = run(["refusal", VALID_SUMMARY])
        assert not outcome.ok
        assert client.call_count == 1
        assert "cannot be fixed" in outcome.stopped_because

    def test_an_empty_response_stops_the_loop(self):
        outcome, client = run(["empty", VALID_SUMMARY])
        assert not outcome.ok
        assert client.call_count == 1

    def test_repeated_identical_output_stops_the_loop_early(self):
        """A model repeating itself will keep repeating itself."""
        outcome, client = run(["wrong_enum"], loop_last=True, max_attempts=5)
        assert not outcome.ok
        assert client.call_count == 2
        assert "repeated" in outcome.stopped_because

    def test_attempts_are_capped(self):
        outcome, client = run(
            ["wrong_enum", "missing_field", "null_required", VALID_SUMMARY],
            max_attempts=3,
        )
        assert not outcome.ok
        assert client.call_count == 3
        assert "exhausted" in outcome.stopped_because

    def test_oscillation_is_visible_in_the_history(self):
        """A model fixing one field and breaking another must not look like one failure."""
        outcome, _ = run(
            ["wrong_enum", "missing_field", "null_required"], max_attempts=3
        )
        assert outcome.error_sequence == [
            "schema_mismatch", "schema_mismatch", "schema_mismatch"
        ]
        assert outcome.attempt_count == 3


class TestTruncationHandling:
    def test_the_budget_is_raised_rather_than_the_request_repeated(self):
        """Retrying at the same ceiling reproduces the truncation exactly."""
        from provider.fake import FakeResponse

        truncated = FakeResponse(text=MALFORMED["truncated_object"], truncated=True)
        client = ScriptedClient([truncated, VALID_SUMMARY])
        outcome = repair_loop(
            client, TicketSummary, tier="json_schema", max_attempts=3, max_tokens=100
        )

        assert outcome.ok
        assert client.token_budgets == [100, 100 * TRUNCATION_BUDGET_MULTIPLIER]

    def test_truncation_with_no_ceiling_to_raise_stops(self):
        from provider.fake import FakeResponse

        client = ScriptedClient(
            [FakeResponse(text="", truncated=True), VALID_SUMMARY]
        )
        outcome = repair_loop(
            client, TicketSummary, tier="json_schema", max_attempts=3, max_tokens=None
        )
        assert not outcome.ok
        assert "no token ceiling" in outcome.stopped_because


class TestAccounting:
    def test_tokens_and_latency_accumulate_across_attempts(self):
        outcome, _ = run(["wrong_enum", VALID_SUMMARY])
        assert outcome.total_tokens == 40
        assert outcome.total_latency_ms == pytest.approx(2.0)

    def test_every_attempt_records_its_tier(self):
        outcome, _ = run(["wrong_enum", VALID_SUMMARY])
        assert all(a.tier == "json_object" for a in outcome.attempts)

    def test_repairs_link_back_to_the_attempt_they_followed(self):
        outcome, _ = run(["wrong_enum", VALID_SUMMARY])
        assert outcome.attempts[0].repaired_from is None
        assert outcome.attempts[1].repaired_from == 1


class TestRepairMessages:
    def test_a_constraint_failure_says_to_change_only_that_field(self):
        result = validate_response(CONSTRAINT_VIOLATION, TicketTriage)
        message = build_repair_message(result, TicketTriage)
        assert "confidence" in message
        assert "value constraint" in message

    def test_unparseable_output_is_told_to_drop_the_fences(self):
        result = validate_response(MALFORMED["prose_only"], TicketSummary)
        message = build_repair_message(result, TicketSummary)
        assert "markdown" in message.lower()


class TestProviderRejection:
    """The provider caught the mistake instead of the validator.

    Worth its own failure type: it is the only direct evidence that enforcement
    ran at all, and it must stay retryable — an empty rejection classified as an
    empty response would stop the repair loop on exactly the tier where
    enforcement is doing its job.
    """

    def test_it_is_classified_as_a_rejection_not_an_empty_response(self):
        result = validate_response("", TicketSummary, provider_rejected=True)
        assert result.failure is FailureType.PROVIDER_REJECTED
        assert not result.ok

    def test_a_rejection_is_retryable(self):
        assert validate_response("", TicketSummary, provider_rejected=True).retryable

    def test_an_ordinary_empty_response_is_still_unretryable(self):
        """Asking a model that returned nothing to try again returns nothing."""
        assert not validate_response("", TicketSummary).retryable

    def test_the_rejected_text_is_kept_and_explained(self):
        """Groq hands back what it refused, which is the evidence."""
        result = validate_response(
            '{"category": 5}', TicketSummary, provider_rejected=True
        )
        assert result.raw == '{"category": 5}'
        assert result.field_errors
        assert result.failure is FailureType.PROVIDER_REJECTED

    def test_a_rejection_without_text_still_explains_itself(self):
        detail = validate_response("", TicketSummary, provider_rejected=True).detail
        assert "refused" in detail

    def test_the_repair_message_asks_for_the_whole_object(self):
        """There is no partial output to preserve: the provider returned none."""
        result = validate_response("", TicketSummary, provider_rejected=True)
        message = build_repair_message(result, TicketSummary)
        assert "matches the schema exactly" in message
