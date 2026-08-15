"""Tests for the benchmark's scoring and cell logic.

The scorer decides every semantic number the article quotes, so its failure
modes matter. Two in particular: scoring a defensible classification as wrong
would record the labeller's preference as a model error, and accepting a
placeholder as a real value would let invented data pass as grounded.
"""

from __future__ import annotations

import pytest

from bench.cases import CASES, CASES_BY_ID
from bench.run import BENCH_TIERS, CellResult, tier_is_possible
from bench.score import PLACEHOLDER_MARKERS, is_placeholder, score_case
from contracts.schemas import (
    Category,
    Customer,
    Issue,
    Priority,
    Sentiment,
    TicketSummary,
    TicketTriage,
)
from provider.capabilities import ModelCapabilities, TierSupport


def triage(**overrides) -> TicketTriage:
    base = dict(
        customer=Customer(name="Marta Silva", account_tier="Business"),
        issues=[Issue(description="Billed twice.", category=Category.BILLING)],
        priority=Priority.HIGH,
        sentiment=Sentiment.FRUSTRATED,
        steps=[],
        confidence=0.8,
        assignee="Daniel",
    )
    return TicketTriage(**{**base, **overrides})


class TestCases:
    def test_every_case_has_a_unique_id(self):
        assert len({c.id for c in CASES}) == len(CASES)

    def test_every_case_allows_at_least_one_answer(self):
        for case in CASES:
            assert case.categories, case.id
            assert case.priorities, case.id

    def test_ambiguous_cases_allow_several_answers(self):
        """Forcing one answer on a genuinely ambiguous ticket measures the labeller."""
        assert len(CASES_BY_ID["vague"].priorities) > 1
        assert len(CASES_BY_ID["add_seat"].categories) > 1

    def test_stated_and_absent_facts_do_not_overlap(self):
        """A fact cannot be both given and withheld."""
        for case in CASES:
            assert not (set(case.stated_facts) & set(case.absent_facts)), case.id


class TestPlaceholders:
    @pytest.mark.parametrize(
        "value", ["unknown", "N/A", "Anonymous", "Unnamed", "not specified", ""]
    )
    def test_filler_is_recognised(self, value):
        assert is_placeholder(value)

    @pytest.mark.parametrize("value", ["Marta Silva", "Business", "Tom Reyes"])
    def test_real_values_are_not_filler(self, value):
        assert not is_placeholder(value)

    def test_an_explicit_null_is_honest(self):
        """Saying "no value" is correct; inventing a fake name is not."""
        assert not is_placeholder(None)

    def test_markers_are_stored_lowercase(self):
        assert all(m == m.lower() for m in PLACEHOLDER_MARKERS)


class TestScoring:
    def test_a_faithful_extraction_scores_clean(self):
        score = score_case(CASES_BY_ID["billing_duplicate"], triage())
        assert score.accurate
        assert score.grounded
        assert score.problems() == []

    def test_any_acceptable_category_passes(self):
        """A ticket straddling two teams has two defensible answers."""
        case = CASES_BY_ID["add_seat"]
        for category in case.categories:
            summary = TicketSummary(
                category=category, priority=Priority.LOW, summary="Add a seat."
            )
            assert score_case(case, summary).category_ok

    def test_a_category_outside_the_set_fails(self):
        case = CASES_BY_ID["dark_mode"]
        summary = TicketSummary(
            category=Category.BILLING, priority=Priority.LOW, summary="x"
        )
        assert not score_case(case, summary).category_ok

    def test_an_invented_name_is_caught(self):
        """The failure that matters most: a value the source never gave."""
        case = CASES_BY_ID["dark_mode"]
        score = score_case(
            case, triage(customer=Customer(name="Jane Doe", account_tier="Pro"))
        )
        assert not score.grounded
        assert any("name" in item for item in score.invented)

    def test_a_placeholder_is_reported_separately_from_invention(self):
        """Filler is a distinct failure: it looks like data downstream."""
        case = CASES_BY_ID["dark_mode"]
        score = score_case(
            case,
            triage(
                customer=Customer(name="Anonymous", account_tier="unknown"),
                assignee=None,
            ),
        )
        assert not score.grounded
        assert score.placeholders
        assert not score.invented

    def test_correctly_leaving_an_absent_field_null_is_grounded(self):
        case = CASES_BY_ID["dark_mode"]
        score = score_case(
            case,
            TicketSummary(
                category=Category.FEATURE_REQUEST,
                priority=Priority.LOW,
                summary="Customer wants a dark theme.",
            ),
        )
        assert score.grounded

    def test_a_stated_fact_contradicted_counts_as_invention(self):
        case = CASES_BY_ID["billing_duplicate"]
        score = score_case(
            case, triage(customer=Customer(name="Someone Else", account_tier="Free"))
        )
        assert not score.grounded

    def test_a_missed_stated_fact_is_reported_but_is_not_invention(self):
        """Omitting a given detail is weaker output; inventing one is wrong output."""
        case = CASES_BY_ID["billing_duplicate"]
        score = score_case(case, triage(assignee=None))
        assert "assignee" in score.missed
        assert score.grounded

    def test_nested_categories_are_read_from_the_issues(self):
        case = CASES_BY_ID["billing_duplicate"]
        score = score_case(
            case,
            triage(issues=[Issue(description="d", category=Category.BILLING)]),
        )
        assert score.category_ok

    def test_problems_name_what_went_wrong(self):
        case = CASES_BY_ID["dark_mode"]
        score = score_case(
            case,
            TicketSummary(
                category=Category.BILLING, priority=Priority.CRITICAL, summary="x"
            ),
        )
        assert len(score.problems()) == 2


class TestTierFeasibility:
    def _caps(self, **tiers) -> ModelCapabilities:
        return ModelCapabilities(
            provider="groq", model="m", probed_at="2026-08-15T00:00:00+00:00",
            tiers=tiers,
        )

    def test_a_supported_tier_is_runnable(self):
        caps = self._caps(json_schema=TierSupport.CONFORMED.value)
        assert tier_is_possible(caps, "json_schema")[0]

    def test_a_rejected_tier_is_skipped_with_a_reason(self):
        """Running it would 400 on every call and produce no data."""
        caps = self._caps(json_schema=TierSupport.REJECTED.value)
        possible, reason = tier_is_possible(caps, "json_schema")
        assert not possible
        assert "rejected" in reason

    def test_an_ignored_tier_is_not_runnable(self):
        caps = self._caps(tools=TierSupport.IGNORED.value)
        assert not tier_is_possible(caps, "tools")[0]

    def test_prompt_only_always_runs(self):
        """It sends no directive, so there is nothing for a provider to reject."""
        assert tier_is_possible(None, "prompt_only")[0]

    def test_an_unprobed_model_cannot_run_enforced_tiers(self):
        possible, reason = tier_is_possible(None, "json_schema")
        assert not possible
        assert "not probed" in reason


class TestCellResult:
    def test_rates_are_computed_from_the_totals(self):
        cell = CellResult(
            provider="groq", model="m", tier="json_schema",
            contract="TicketSummary", difficulty="simple",
            first_attempt_ok=8, final_ok=10, total=10,
        )
        assert cell.first_attempt_rate == 0.8
        assert cell.final_rate == 1.0

    def test_an_empty_cell_does_not_divide_by_zero(self):
        cell = CellResult(
            provider="groq", model="m", tier="json_schema",
            contract="c", difficulty="simple",
        )
        assert cell.first_attempt_rate == 0.0
        assert cell.accuracy == 0.0
        assert cell.critic_rate == 0.0

    def test_a_skipped_cell_carries_its_reason(self):
        """A missing cell is indistinguishable from one never attempted."""
        cell = CellResult(
            provider="groq", model="m", tier="json_schema", contract="c",
            difficulty="simple", skipped_reason="json_schema is rejected",
        )
        assert cell.skipped_reason
        assert cell.total == 0

    def test_tools_is_excluded_from_the_measured_tiers(self):
        """A different envelope with its own failure modes, measured separately."""
        assert "tools" not in BENCH_TIERS
