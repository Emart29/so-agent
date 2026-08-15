"""Tests for the log, the metrics, the critic's verdicts, and the agent.

Two failures here are silent and consequential. A rate reported without its
sample size invites a reader to believe a difference the data cannot support,
and an unavailable critic read as a pass lets a broken judge approve everything
while looking like coverage. Both are asserted directly.
"""

from __future__ import annotations

import math

import pytest

from critic.semantic import (
    AgreementScore,
    CriticReport,
    SemanticResult,
    Verdict,
    _verdict_from,
)
from store.log import AttemptLog, AttemptRow, new_run_id
from store.metrics import Metrics, Rate, trajectory_reliability


@pytest.fixture
def log(tmp_path):
    return AttemptLog(tmp_path / "attempts.db")


def add(log, *, run=None, index=1, ok=True, tier="json_schema", failure=None,
        provider="groq", model="m", difficulty="simple", extraction=False,
        verdict=None, tokens=10, latency=5.0, schema="s"):
    log.record(AttemptRow(
        run_id=run or new_run_id(), attempt_index=index, provider=provider,
        model=model, tier=tier, schema_name=schema, schema_difficulty=difficulty,
        success=ok, failure_type=failure, recovered_by_extraction=extraction,
        critic_verdict=verdict, prompt_tokens=tokens, completion_tokens=tokens,
        latency_ms=latency,
    ))


class TestRate:
    def test_a_rate_carries_its_interval(self):
        rate = Rate(50, 100)
        low, high = rate.interval
        assert rate.value == 0.5
        assert low < 0.5 < high

    def test_the_interval_narrows_as_evidence_accumulates(self):
        assert Rate(5000, 10000).stderr < Rate(50, 100).stderr

    def test_a_small_sample_is_flagged_as_unmeasurable(self):
        """4% from 25 samples is not a measurement."""
        assert not Rate(1, 25).is_measurable
        assert "n too small" in str(Rate(1, 25))

    def test_an_empty_rate_does_not_divide_by_zero(self):
        assert Rate(0, 0).value == 0.0
        assert Rate(0, 0).stderr == 0.0

    def test_the_interval_stays_within_zero_and_one(self):
        low, high = Rate(100, 100).interval
        assert low >= 0.0 and high <= 1.0


class TestComparison:
    def test_overlapping_intervals_yield_no_winner(self):
        log_metrics = Metrics.__new__(Metrics)
        from store.metrics import Comparison

        comparison = Comparison("a", "b", Rate(52, 100), Rate(48, 100))
        assert comparison.intervals_overlap
        assert "no distinguishable difference" in comparison.verdict

    def test_a_clear_separation_names_the_better_one(self):
        from store.metrics import Comparison

        comparison = Comparison("a", "b", Rate(95, 100), Rate(20, 100))
        assert not comparison.intervals_overlap
        assert "a is better" in comparison.verdict

    def test_thin_data_refuses_a_verdict_entirely(self):
        from store.metrics import Comparison

        comparison = Comparison("a", "b", Rate(5, 5), Rate(0, 5))
        assert "not enough data" in comparison.verdict


class TestLog:
    def test_attempts_are_stored_and_counted(self, log):
        add(log)
        add(log, ok=False, failure="not_json")
        assert log.count() == 2

    def test_a_run_groups_its_attempts_in_order(self, log):
        run = new_run_id()
        add(log, run=run, index=1, ok=False, failure="schema_mismatch")
        add(log, run=run, index=2, ok=True)

        rows = log.attempts_for(run)
        assert [r["attempt_index"] for r in rows] == [1, 2]

    def test_combinations_are_discoverable(self, log):
        add(log, provider="groq", model="a", tier="json_schema")
        add(log, provider="openrouter", model="b", tier="json_object")
        assert len(log.combinations()) == 2

    def test_export_writes_one_line_per_attempt(self, log, tmp_path):
        add(log)
        add(log)
        target = tmp_path / "out.jsonl"
        assert log.export_jsonl(target) == 2
        assert len(target.read_text().strip().splitlines()) == 2


class TestMetrics:
    def test_first_attempt_success_ignores_repairs(self, log):
        """The headline number: how often enforcement worked unaided."""
        run = new_run_id()
        add(log, run=run, index=1, ok=False, failure="schema_mismatch")
        add(log, run=run, index=2, ok=True)
        add(log, index=1, ok=True)

        metrics = Metrics(log)
        assert metrics.first_attempt_success().successes == 1
        assert metrics.first_attempt_success().total == 2

    def test_final_success_counts_the_run_not_the_attempt(self, log):
        run = new_run_id()
        add(log, run=run, index=1, ok=False, failure="schema_mismatch")
        add(log, run=run, index=2, ok=True)

        assert Metrics(log).final_success().value == 1.0

    def test_repair_lift_reports_what_it_cost(self, log):
        run = new_run_id()
        add(log, run=run, index=1, ok=False, failure="schema_mismatch")
        add(log, run=run, index=2, ok=True)

        lift = Metrics(log).repair_lift()
        assert lift["lift"] == pytest.approx(1.0)
        assert lift["retries"] == 1
        assert lift["extra_tokens"] == 20

    def test_free_repairs_are_counted_separately(self, log):
        """Fixed locally, no retry — the share that costs nothing."""
        add(log, ok=True, extraction=True)
        add(log, ok=True, extraction=False)
        assert Metrics(log).free_repair_share().value == 0.5

    def test_error_breakdown_covers_first_attempts_only(self, log):
        add(log, ok=False, failure="not_json")
        add(log, ok=False, failure="not_json")
        add(log, ok=False, failure="truncated")

        breakdown = Metrics(log).error_breakdown()
        assert breakdown == {"not_json": 2, "truncated": 1}

    def test_attempts_to_success_reports_the_distribution(self, log):
        """The mean hides the shape."""
        one = new_run_id()
        add(log, run=one, index=1, ok=True)
        two = new_run_id()
        add(log, run=two, index=1, ok=False, failure="x")
        add(log, run=two, index=2, ok=True)

        assert Metrics(log).attempts_to_success() == {1: 1, 2: 1}

    def test_difficulty_grouping_separates_the_contracts(self, log):
        add(log, difficulty="simple", ok=True)
        add(log, difficulty="hard", ok=False, failure="schema_mismatch")

        by_difficulty = Metrics(log).by_schema_difficulty()
        assert by_difficulty["simple"].value == 1.0
        assert by_difficulty["hard"].value == 0.0

    def test_filters_narrow_the_population(self, log):
        add(log, provider="groq", ok=True)
        add(log, provider="openrouter", ok=False, failure="x")

        metrics = Metrics(log)
        assert metrics.first_attempt_success(provider="groq").value == 1.0
        assert metrics.first_attempt_success(provider="openrouter").value == 0.0


class TestSemanticMetrics:
    def test_accuracy_by_tier_only_counts_judged_output(self, log):
        """The query that asks whether enforcement costs quality."""
        add(log, tier="json_schema", ok=True, verdict="sound")
        add(log, tier="json_schema", ok=True, verdict="contradicted")
        add(log, tier="prompt_only", ok=True, verdict="sound")

        by_tier = Metrics(log).accuracy_by_tier()
        assert by_tier["json_schema"].value == 0.5
        assert by_tier["prompt_only"].value == 1.0

    def test_an_unavailable_critic_is_excluded_not_counted_as_sound(self, log):
        """A broken judge must not silently approve anything."""
        add(log, ok=True, verdict="sound")
        add(log, ok=True, verdict="unavailable")

        assert Metrics(log).accuracy_by_tier()["json_schema"].total == 1

    def test_semantic_failures_are_measured_among_valid_output(self, log):
        """The gap: '99% valid' while a third of those are wrong."""
        add(log, ok=True, verdict="sound")
        add(log, ok=True, verdict="sound")
        add(log, ok=True, verdict="contradicted")

        assert Metrics(log).semantic_failure_rate().value == pytest.approx(1 / 3)

    def test_failed_extractions_are_not_judged_semantically(self, log):
        add(log, ok=False, failure="not_json")
        assert Metrics(log).semantic_failure_rate().total == 0


class TestTrajectoryReliability:
    def test_reliability_compounds_across_a_chain(self):
        """99% per call is roughly 60% over fifty steps."""
        projection = trajectory_reliability(0.99)
        assert projection[1] == pytest.approx(0.99)
        assert projection[50] == pytest.approx(0.605, abs=0.01)

    def test_a_small_per_call_drop_collapses_a_long_chain(self):
        assert trajectory_reliability(0.90)[50] < 0.01

    def test_perfect_reliability_stays_perfect(self):
        assert trajectory_reliability(1.0)[50] == 1.0

    def test_the_step_counts_are_configurable(self):
        assert set(trajectory_reliability(0.9, steps=(2, 3))) == {2, 3}


class TestCriticVerdicts:
    def _report(self, **overrides) -> CriticReport:
        base = dict(
            grounded=True, plausible_confidence=True,
            categories_consistent=True, substantive=True, problem="",
        )
        return CriticReport(**{**base, **overrides})

    def test_everything_passing_is_sound(self):
        assert _verdict_from(self._report()) is Verdict.SOUND

    def test_ungrounded_output_is_contradicted_not_merely_unsupported(self):
        """A value the source does not support is wrong, not just weak."""
        assert _verdict_from(self._report(grounded=False)) is Verdict.CONTRADICTED

    @pytest.mark.parametrize(
        "flag", ["plausible_confidence", "categories_consistent", "substantive"]
    )
    def test_other_problems_are_unsupported(self, flag):
        assert _verdict_from(self._report(**{flag: False})) is Verdict.UNSUPPORTED

    def test_an_unavailable_critic_is_not_a_pass(self):
        """Treating it as one lets a broken critic approve everything."""
        result = SemanticResult(verdict=Verdict.UNAVAILABLE)
        assert not result.passed
        assert not result.checked

    def test_a_sound_verdict_is_both_passed_and_checked(self):
        result = SemanticResult(verdict=Verdict.SOUND)
        assert result.passed and result.checked


class TestAgreementScore:
    def test_agreement_excludes_unavailable_cases(self):
        """They were never judged, so they cannot count either way."""
        score = AgreementScore(total=10, agreed=6, false_pass=1, false_fail=1, unavailable=2)
        assert score.agreement == pytest.approx(6 / 8)

    def test_the_two_error_directions_are_reported_separately(self):
        """Approving a bad extraction differs in consequence from rejecting a good one."""
        score = AgreementScore(total=10, agreed=8, false_pass=2, false_fail=0, unavailable=0)
        assert "2 approved a bad extraction" in score.summary()
        assert "0 rejected a good one" in score.summary()

    def test_no_judged_cases_does_not_divide_by_zero(self):
        assert AgreementScore(total=3, agreed=0, false_pass=0, false_fail=0, unavailable=3).agreement == 0.0
