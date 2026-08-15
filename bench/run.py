"""The measurement harness: every cell of the matrix the article rests on.

Two runs with different budgets and different questions, deliberately not
conflated:

* **Groq** is the workhorse — the full model ladder against every tier the
  capability matrix says is possible. This answers how structured output
  behaves across model sizes, and whether the repair loop earns its place.
* **OpenRouter** is one slice, sized to fit a daily allowance. It runs the
  single model both providers serve, so the weights are held constant and any
  difference is attributable to the serving stack.

Semantic accuracy is scored against hand-written labels rather than against the
critic. The critic's own error rate is measured but not zero, and deriving the
headline accuracy figure from it would fold that error into every number. The
critic runs alongside as a second opinion, and where the two disagree that
disagreement is itself worth reporting.

Repeats are samples, not reproductions. Providers differ in what sampling
settings they honour, and batching makes identical output unlikely regardless,
so nothing here claims determinism.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from bench.cases import CASES, Case
from bench.score import CaseScore, score_case
from contracts.schemas import CONTRACTS, Contract
from provider.capabilities import ModelCapabilities, TierSupport, load_capabilities
from provider.client import BudgetExhaustedError

logger = logging.getLogger("bench.run")

#: Tiers measured. `tools` is excluded: it is a different envelope with its own
#: failure modes, measured separately rather than mixed into these rates.
BENCH_TIERS = ("json_schema", "json_object", "prompt_only")


@dataclass
class CellResult:
    """Every run for one (provider, model, tier, contract) combination."""

    provider: str
    model: str
    tier: str
    contract: str
    difficulty: str
    first_attempt_ok: int = 0
    final_ok: int = 0
    free_repairs: int = 0
    total: int = 0
    failures: dict[str, int] = field(default_factory=dict)
    scores: list[CaseScore] = field(default_factory=list)
    critic_sound: int = 0
    critic_judged: int = 0
    tokens: int = 0
    latency_ms: float = 0.0
    skipped_reason: str = ""

    @property
    def first_attempt_rate(self) -> float:
        return self.first_attempt_ok / self.total if self.total else 0.0

    @property
    def final_rate(self) -> float:
        return self.final_ok / self.total if self.total else 0.0

    @property
    def accuracy(self) -> float:
        """Share of valid extractions that were also faithful to the source."""
        return (
            sum(1 for s in self.scores if s.accurate) / len(self.scores)
            if self.scores
            else 0.0
        )

    @property
    def grounded_rate(self) -> float:
        """Share that invented nothing — the failure that matters most."""
        return (
            sum(1 for s in self.scores if s.grounded) / len(self.scores)
            if self.scores
            else 0.0
        )

    @property
    def critic_rate(self) -> float:
        return self.critic_sound / self.critic_judged if self.critic_judged else 0.0


def tier_is_possible(caps: ModelCapabilities | None, tier: str) -> tuple[bool, str]:
    """Whether a tier can be measured on a model, and why not when it cannot.

    A tier the provider rejects produces no data — every call 400s — so running
    it wastes an allowance to learn what the probe already established.
    """
    if tier == "prompt_only":
        return True, ""
    if caps is None:
        return False, "model not probed"
    support = caps.tiers.get(tier)
    if support == TierSupport.CONFORMED:
        return True, ""
    return False, f"{tier} is {support or 'untested'} on this model"


def run_cell(
    agent,
    contract: type[Contract],
    tier: str,
    cases: list[Case],
    repeats: int,
    review: bool,
) -> CellResult:
    """Run every case in one cell and accumulate its result."""
    cell = CellResult(
        provider=agent.provider,
        model=agent.model,
        tier=tier,
        contract=contract.__name__,
        difficulty=contract.difficulty().value,
    )

    for _ in range(repeats):
        for case in cases:
            result = agent.extract(case.text, contract, tier=tier, review=review)
            cell.total += 1
            cell.tokens += result.total_tokens
            cell.latency_ms += result.total_latency_ms

            if result.first_attempt_ok:
                cell.first_attempt_ok += 1
            elif result.outcome and result.outcome.needed_only_extraction:
                cell.free_repairs += 1

            if result.outcome and result.outcome.attempts:
                first = result.outcome.attempts[0]
                if not first.ok:
                    key = first.failure.value
                    cell.failures[key] = cell.failures.get(key, 0) + 1

            if not result.ok:
                continue

            cell.final_ok += 1
            cell.scores.append(score_case(case, result.value))

            sound = result.semantically_sound
            if sound is not None:
                cell.critic_judged += 1
                cell.critic_sound += int(sound)

    return cell


def run_matrix(
    make_agent,
    provider: str,
    models: list[str],
    tiers: list[str] | None = None,
    contracts: list[str] | None = None,
    repeats: int = 3,
    cases: list[Case] | None = None,
    review: bool = True,
    on_cell=None,
) -> list[CellResult]:
    """Run every possible cell for one provider.

    Args:
        make_agent: Callable taking ``(model)`` and returning a configured agent.
            Passed in so the harness does not own agent construction, which
            keeps it usable with a scripted client in tests.
        provider: Provider name, used to look up capabilities.
        models: Models to measure.
        tiers: Tiers to attempt. Defaults to all measured tiers.
        contracts: Contract names. Defaults to all.
        repeats: Runs per case per cell. Repeats are samples, not reproductions.
        cases: Task set. Defaults to the full set.
        review: Run the semantic critic alongside the label-based scoring.
        on_cell: Optional callback invoked with each finished cell, for progress.

    Returns:
        Every cell attempted, including those skipped, each carrying its reason.
    """
    tiers = tiers or list(BENCH_TIERS)
    contract_types = [
        CONTRACTS[name] for name in (contracts or list(CONTRACTS))
    ]
    cases = cases or CASES
    capabilities = load_capabilities().get(provider, {})

    results: list[CellResult] = []

    for model in models:
        caps = capabilities.get(model)
        agent = None

        for tier in tiers:
            possible, reason = tier_is_possible(caps, tier)

            for contract in contract_types:
                if not possible:
                    # Recorded rather than omitted: a missing cell is
                    # indistinguishable from one that was never attempted.
                    skipped = CellResult(
                        provider=provider, model=model, tier=tier,
                        contract=contract.__name__,
                        difficulty=contract.difficulty().value,
                        skipped_reason=reason,
                    )
                    results.append(skipped)
                    if on_cell:
                        on_cell(skipped)
                    continue

                if agent is None:
                    agent = make_agent(model)

                try:
                    cell = run_cell(agent, contract, tier, cases, repeats, review)
                except BudgetExhaustedError as exc:
                    # Stop and report the partial matrix. Rerouting to another
                    # provider would attribute rows to the wrong one, which
                    # corrupts the comparison the second provider exists for.
                    logger.warning("stopping: %s", exc)
                    partial = CellResult(
                        provider=provider, model=model, tier=tier,
                        contract=contract.__name__,
                        difficulty=contract.difficulty().value,
                        skipped_reason=f"budget exhausted: {exc}",
                    )
                    results.append(partial)
                    if on_cell:
                        on_cell(partial)
                    return results

                results.append(cell)
                if on_cell:
                    on_cell(cell)

    return results
