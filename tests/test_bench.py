"""Tests for the benchmark's scoring and cell logic.

The scorer decides every semantic number the article quotes, so its failure
modes matter. Two in particular: scoring a defensible classification as wrong
would record the labeller's preference as a model error, and accepting a
placeholder as a real value would let invented data pass as grounded.
"""

from __future__ import annotations

import pytest

from bench.cases import CASES, CASES_BY_ID
from bench.results import (
    failure_table,
    interval_advice,
    load_results,
    save_results,
    summary_table,
    tier_table,
    trajectory_table,
)
from bench.run import BENCH_TIERS, CellResult, run_matrix, tier_is_possible
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


class TestResults:
    def _cell(self, **overrides) -> CellResult:
        base = dict(
            provider="groq", model="m", tier="json_object",
            contract="TicketSummary", difficulty="simple",
            first_attempt_ok=9, final_ok=10, total=10,
            failures={"not_json": 1},
            scores=[score_case(CASES_BY_ID["billing_duplicate"], triage())],
        )
        return CellResult(**{**base, **overrides})

    def test_a_run_survives_a_round_trip(self, tmp_path):
        """A matrix costs hours of real requests; it must reload exactly."""
        path = tmp_path / "results.json"
        save_results([self._cell()], path, sampling={"repeats": 3})

        payload = load_results(path)
        restored = payload["cells"][0]
        assert payload["sampling"]["repeats"] == 3
        assert restored.final_rate == 1.0
        assert restored.failures == {"not_json": 1}
        assert restored.scores[0].accurate

    def test_a_skipped_cell_is_kept_in_the_file(self, tmp_path):
        path = tmp_path / "results.json"
        save_results([CellResult(
            provider="groq", model="m", tier="json_schema", contract="c",
            difficulty="simple", skipped_reason="json_schema is rejected",
        )], path)
        assert load_results(path)["cells"][0].skipped_reason

    def test_the_tables_render_a_mixed_run(self):
        """Skipped and measured cells share a table, so both paths must render."""
        cells = [
            self._cell(),
            CellResult(
                provider="groq", model="m", tier="json_schema", contract="c",
                difficulty="simple", skipped_reason="rejected",
            ),
        ]
        for build in (summary_table, tier_table, failure_table, trajectory_table):
            assert build(cells).row_count >= 1

    def test_sizing_advice_counts_thin_cells(self):
        advice = interval_advice([self._cell()])
        assert "n<30" in advice

    def test_sizing_advice_passes_a_large_run(self):
        advice = interval_advice([self._cell(total=40, final_ok=38)])
        assert "n>=30" in advice

    def test_sizing_advice_ignores_skipped_cells(self):
        """A cell that never ran says nothing about whether the run was big enough."""
        skipped = CellResult(
            provider="groq", model="m", tier="json_schema", contract="c",
            difficulty="simple", skipped_reason="rejected",
        )
        assert "No cell produced data" in interval_advice([skipped])

    def test_the_projection_reports_the_best_and_worst_cells(self):
        cells = [self._cell(), self._cell(tier="prompt_only", final_ok=6)]
        rendered = trajectory_table(cells)
        assert rendered.row_count == 2


class TestUnavailableModel:
    """A model whose daily quota runs out must not take the run down with it."""

    class _Agent:
        def __init__(self, model: str, fail: bool) -> None:
            self.provider, self.model, self._fail = "groq", model, fail

        def extract(self, *_args, **_kwargs):
            if self._fail:
                raise RuntimeError("groq failed after 4 attempts: rate limited")
            raise AssertionError("the healthy model should not have been reached")

    def test_the_rest_of_the_ladder_still_runs(self, monkeypatch):
        monkeypatch.setattr(
            "bench.run.load_capabilities",
            lambda: {"groq": {}},
        )
        cells = run_matrix(
            lambda model: self._Agent(model, fail=True),
            provider="groq",
            models=["dead", "alive"],
            tiers=["prompt_only"],
            contracts=["ticket_summary", "ticket_analysis"],
            repeats=1,
            cases=[CASES_BY_ID["dark_mode"]],
        )
        assert len(cells) == 4
        assert all(c.skipped_reason for c in cells)
        assert any(c.model == "alive" for c in cells)

    def test_the_remaining_cells_say_why_they_did_not_run(self, monkeypatch):
        monkeypatch.setattr("bench.run.load_capabilities", lambda: {"groq": {}})
        cells = run_matrix(
            lambda model: self._Agent(model, fail=True),
            provider="groq",
            models=["dead"],
            tiers=["prompt_only"],
            contracts=["ticket_summary", "ticket_analysis"],
            repeats=1,
            cases=[CASES_BY_ID["dark_mode"]],
        )
        assert all("unavailable" in c.skipped_reason for c in cells)
