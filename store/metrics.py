"""Turns the attempt log into the numbers the article rests on.

Every rate here carries its sample size and a confidence interval. A failure
rate of 4% from 25 samples is not a measurement, and quoting it without the
interval invites a reader to believe a difference that the data cannot support.

Two queries matter more than the rest:

* :func:`accuracy_by_tier` asks whether enforcement costs reasoning quality.
  Parse rate answers "did it come back well-formed"; this answers "was it still
  right". If the strictest tier parses perfectly and scores worse on
  correctness, that is a real tradeoff almost everyone assumes away.
* :func:`trajectory_reliability` converts a per-call rate into what it means
  across a chain. 99% per call is roughly 60% over fifty steps, and that
  compounding is why this layer matters to anything agentic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from store.log import AttemptLog

#: z for a 95% normal-approximation interval.
Z_95 = 1.96


def _filters(
    provider: str | None = None,
    model: str | None = None,
    tier: str | None = None,
    difficulty: str | None = None,
) -> tuple[str, tuple]:
    """Build an optional WHERE fragment and its parameters.

    Returned as a fragment appended to a query that already has a condition, so
    every clause here starts with AND. Values are always bound rather than
    interpolated — model ids and provider names come from a config file and a
    command line, and neither is a safe thing to splice into SQL.
    """
    clauses: list[str] = []
    params: list[str] = []

    for column, value in (
        ("provider", provider),
        ("model", model),
        ("tier", tier),
        ("schema_difficulty", difficulty),
    ):
        if value is not None:
            clauses.append(f"AND {column} = ?")
            params.append(value)

    return (" " + " ".join(clauses) if clauses else ""), tuple(params)


@dataclass
class Rate:
    """A proportion with the evidence behind it."""

    successes: int
    total: int

    @property
    def value(self) -> float:
        return self.successes / self.total if self.total else 0.0

    @property
    def stderr(self) -> float:
        """Standard error, or 0 when there is nothing to measure."""
        if self.total <= 0:
            return 0.0
        p = self.value
        return math.sqrt(max(p * (1.0 - p), 0.0) / self.total)

    @property
    def interval(self) -> tuple[float, float]:
        """95% interval, clamped to [0, 1]."""
        margin = Z_95 * self.stderr
        return (max(self.value - margin, 0.0), min(self.value + margin, 1.0))

    @property
    def is_measurable(self) -> bool:
        """Whether there is enough data for the number to mean anything.

        Thirty is a convention rather than a law, but a rate below it has an
        interval wide enough that reporting the point estimate alone misleads.
        """
        return self.total >= 30

    def __str__(self) -> str:
        low, high = self.interval
        caveat = "" if self.is_measurable else "  [n too small to conclude]"
        return f"{self.value:.1%} ({low:.1%}-{high:.1%}, n={self.total}){caveat}"


@dataclass
class Comparison:
    """Two rates and whether the data can tell them apart."""

    label_a: str
    label_b: str
    rate_a: Rate
    rate_b: Rate

    @property
    def difference(self) -> float:
        return self.rate_a.value - self.rate_b.value

    @property
    def intervals_overlap(self) -> bool:
        a_low, a_high = self.rate_a.interval
        b_low, b_high = self.rate_b.interval
        return a_low <= b_high and b_low <= a_high

    @property
    def verdict(self) -> str:
        """Refuses to name a winner the data cannot support."""
        if not (self.rate_a.is_measurable and self.rate_b.is_measurable):
            return "not enough data to compare"
        if self.intervals_overlap:
            return "no distinguishable difference"
        better = self.label_a if self.difference > 0 else self.label_b
        return f"{better} is better by {abs(self.difference):.1%}"


class Metrics:
    """Queries over the attempt log."""

    def __init__(self, log: AttemptLog) -> None:
        self.log = log

    # ------------------------------------------------------------------
    # Success rates
    # ------------------------------------------------------------------

    def first_attempt_success(
        self, provider: str | None = None, model: str | None = None,
        tier: str | None = None, difficulty: str | None = None,
    ) -> Rate:
        """How often the first generation validated with no repair.

        The headline number: how often enforcement worked unaided.
        """
        where, params = _filters(provider, model, tier, difficulty)
        row = self.log.query(
            f"SELECT COUNT(*) AS total, SUM(success) AS ok FROM attempts "
            f"WHERE attempt_index = 1 {where}",
            params,
        )[0]
        return Rate(int(row["ok"] or 0), int(row["total"] or 0))

    def final_success(
        self, provider: str | None = None, model: str | None = None,
        tier: str | None = None, difficulty: str | None = None,
    ) -> Rate:
        """How often a run ended validated, after any repair."""
        where, params = _filters(provider, model, tier, difficulty)
        rows = self.log.query(
            f"SELECT run_id, MAX(success) AS ok FROM attempts "
            f"WHERE 1=1 {where} GROUP BY run_id",
            params,
        )
        return Rate(sum(int(r["ok"] or 0) for r in rows), len(rows))

    def repair_lift(
        self, provider: str | None = None, model: str | None = None,
        tier: str | None = None,
    ) -> dict[str, float]:
        """What repair added, and what it cost to add it."""
        first = self.first_attempt_success(provider, model, tier)
        final = self.final_success(provider, model, tier)
        where, params = _filters(provider, model, tier)
        extra = self.log.query(
            f"SELECT COUNT(*) AS retries, "
            f"COALESCE(SUM(prompt_tokens + completion_tokens), 0) AS tokens, "
            f"COALESCE(SUM(latency_ms), 0) AS latency FROM attempts "
            f"WHERE attempt_index > 1 {where}",
            params,
        )[0]
        return {
            "first_attempt": first.value,
            "after_repair": final.value,
            "lift": final.value - first.value,
            "retries": int(extra["retries"] or 0),
            "extra_tokens": int(extra["tokens"] or 0),
            "extra_latency_ms": float(extra["latency"] or 0.0),
        }

    def free_repair_share(
        self, provider: str | None = None, model: str | None = None,
        tier: str | None = None,
    ) -> Rate:
        """Share of first attempts fixed locally, with no retry.

        On weak tiers this is often most of the raw failure rate. A loop that
        pays for a retry to strip a markdown fence measures its own impatience.
        """
        where, params = _filters(provider, model, tier)
        row = self.log.query(
            f"SELECT COUNT(*) AS total, "
            f"SUM(recovered_by_extraction) AS recovered FROM attempts "
            f"WHERE attempt_index = 1 {where}",
            params,
        )[0]
        return Rate(int(row["recovered"] or 0), int(row["total"] or 0))

    # ------------------------------------------------------------------
    # Failure shape
    # ------------------------------------------------------------------

    def error_breakdown(
        self, provider: str | None = None, model: str | None = None,
        tier: str | None = None,
    ) -> dict[str, int]:
        """Which failures occurred on first attempts, and how often."""
        where, params = _filters(provider, model, tier)
        rows = self.log.query(
            f"SELECT failure_type, COUNT(*) AS n FROM attempts "
            f"WHERE attempt_index = 1 AND success = 0 {where} "
            f"GROUP BY failure_type ORDER BY n DESC",
            params,
        )
        return {r["failure_type"] or "unknown": int(r["n"]) for r in rows}

    def attempts_to_success(self) -> dict[int, int]:
        """Distribution of attempts needed, not just the mean.

        The mean hides the shape: two runs at one attempt and one at five is a
        different system from three runs at two or three.
        """
        rows = self.log.query(
            "SELECT COUNT(*) AS attempts FROM attempts "
            "GROUP BY run_id HAVING MAX(success) = 1"
        )
        distribution: dict[int, int] = {}
        for row in rows:
            n = int(row["attempts"])
            distribution[n] = distribution.get(n, 0) + 1
        return dict(sorted(distribution.items()))

    def by_schema_difficulty(
        self, provider: str | None = None, model: str | None = None,
        tier: str | None = None,
    ) -> dict[str, Rate]:
        """First-attempt success per difficulty — does complexity predict failure?"""
        where, params = _filters(provider, model, tier)
        rows = self.log.query(
            f"SELECT schema_difficulty AS d, COUNT(*) AS total, SUM(success) AS ok "
            f"FROM attempts WHERE attempt_index = 1 {where} GROUP BY d",
            params,
        )
        return {
            r["d"] or "unknown": Rate(int(r["ok"] or 0), int(r["total"]))
            for r in rows
        }

    # ------------------------------------------------------------------
    # The two that make this current
    # ------------------------------------------------------------------

    def accuracy_by_tier(
        self, provider: str | None = None, model: str | None = None
    ) -> dict[str, Rate]:
        """Semantic soundness per enforcement tier, among valid outputs.

        The project's most important query. Parse rate says the output was
        well-formed; this says it was also right. If the strictest tier parses
        perfectly and scores worse here, constraining generation is costing
        reasoning quality — a tradeoff nearly everyone assumes away, and one
        that only shows up if correctness is measured separately from shape.
        """
        where, params = _filters(provider, model)
        rows = self.log.query(
            f"SELECT tier, COUNT(*) AS total, "
            f"SUM(CASE WHEN critic_verdict = 'sound' THEN 1 ELSE 0 END) AS sound "
            f"FROM attempts "
            f"WHERE success = 1 AND critic_verdict IS NOT NULL "
            f"AND critic_verdict != 'unavailable' {where} "
            f"GROUP BY tier",
            params,
        )
        return {
            r["tier"]: Rate(int(r["sound"] or 0), int(r["total"])) for r in rows
        }

    def semantic_failure_rate(
        self, provider: str | None = None, model: str | None = None,
        tier: str | None = None,
    ) -> Rate:
        """How often structurally valid output was semantically wrong.

        The gap this project exists to expose. A system reporting "99% valid"
        while a third of those are wrong is reporting the wrong number.
        """
        where, params = _filters(provider, model, tier)
        row = self.log.query(
            f"SELECT COUNT(*) AS total, "
            f"SUM(CASE WHEN critic_verdict != 'sound' THEN 1 ELSE 0 END) AS bad "
            f"FROM attempts WHERE success = 1 AND critic_verdict IS NOT NULL "
            f"AND critic_verdict != 'unavailable' {where}",
            params,
        )[0]
        return Rate(int(row["bad"] or 0), int(row["total"] or 0))

    # ------------------------------------------------------------------
    # Comparisons
    # ------------------------------------------------------------------

    def by_provider(self, model_filter: str | None = None) -> dict[str, Rate]:
        """First-attempt success per provider.

        Where two providers serve the same model, any difference is
        attributable to the serving stack rather than the weights.
        """
        rows = self.log.query(
            "SELECT provider, COUNT(*) AS total, SUM(success) AS ok FROM attempts "
            "WHERE attempt_index = 1 "
            + ("AND model LIKE ? " if model_filter else "")
            + "GROUP BY provider",
            (f"%{model_filter}%",) if model_filter else (),
        )
        return {r["provider"]: Rate(int(r["ok"] or 0), int(r["total"])) for r in rows}

    def compare(
        self, a: tuple[str, str], b: tuple[str, str]
    ) -> Comparison:
        """Compare two (provider, model) pairs, refusing an unsupported verdict."""
        rate_a = self.first_attempt_success(provider=a[0], model=a[1])
        rate_b = self.first_attempt_success(provider=b[0], model=b[1])
        return Comparison(f"{a[0]}/{a[1]}", f"{b[0]}/{b[1]}", rate_a, rate_b)


def trajectory_reliability(
    per_call: float, steps: tuple[int, ...] = (1, 5, 10, 50)
) -> dict[int, float]:
    """Project a per-call success rate across a chain of calls.

    Four lines of arithmetic, and the number that connects this project to every
    agent built on top of it: 99% per call is roughly 60% across fifty steps.

    The independence assumption is not true — real failures correlate, since a
    model that struggles with a schema struggles with it repeatedly — so this is
    an upper bound on chain reliability, not a forecast. Reported as such.
    """
    return {n: per_call**n for n in steps}
