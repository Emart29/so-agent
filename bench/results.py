"""Persisting and presenting a benchmark run.

A matrix costs hundreds of real requests, so the result is written to disk the
moment it exists and every later view — the summary table, the HTML report —
reads that file rather than re-running anything.

Two presentation rules the tables here follow, both borrowed from the metrics
layer because the same mistakes are available in both places:

* **A rate is printed with its interval and its n.** Ten samples in a cell make
  a point estimate that looks precise and is not.
* **A skipped cell is shown, not dropped.** A missing row reads as an oversight;
  a row saying "json_schema is rejected on this model" is a finding.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from rich.console import Console
from rich.table import Table

from bench.run import CellResult
from bench.score import CaseScore
from store.metrics import Rate, trajectory_reliability

#: Where a run is written when no path is given.
DEFAULT_RESULTS_PATH = Path("bench_results.json")

#: Chain lengths the trajectory projection is reported at.
TRAJECTORY_STEPS = (1, 5, 10, 50)


def _score_to_dict(score: CaseScore) -> dict[str, Any]:
    return {
        "case_id": score.case_id,
        "category_ok": score.category_ok,
        "priority_ok": score.priority_ok,
        "invented": score.invented,
        "missed": score.missed,
        "placeholders": score.placeholders,
    }


def _cell_to_dict(cell: CellResult) -> dict[str, Any]:
    return {
        "provider": cell.provider,
        "model": cell.model,
        "tier": cell.tier,
        "contract": cell.contract,
        "difficulty": cell.difficulty,
        "first_attempt_ok": cell.first_attempt_ok,
        "final_ok": cell.final_ok,
        "free_repairs": cell.free_repairs,
        "total": cell.total,
        "failures": cell.failures,
        "errors": cell.errors,
        "critic_sound": cell.critic_sound,
        "critic_judged": cell.critic_judged,
        "tokens": cell.tokens,
        "latency_ms": cell.latency_ms,
        "skipped_reason": cell.skipped_reason,
        "scores": [_score_to_dict(s) for s in cell.scores],
    }


def _cell_from_dict(data: dict[str, Any]) -> CellResult:
    scores = [CaseScore(**s) for s in data.get("scores", [])]
    return CellResult(**{**data, "scores": scores})


def save_results(
    cells: Iterable[CellResult],
    path: Path | str = DEFAULT_RESULTS_PATH,
    sampling: dict[str, Any] | None = None,
) -> Path:
    """Write a run to disk, with the settings it was produced under.

    Args:
        cells: Every cell attempted, skipped ones included.
        path: Destination file.
        sampling: Temperature, repeats, and anything else that would change the
            numbers. Recorded because a rate without its sampling settings is
            not reproducible even as a sample.

    Returns:
        The path written.
    """
    path = Path(path)
    payload = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "sampling": sampling or {},
        "cells": [_cell_to_dict(c) for c in cells],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_results(path: Path | str = DEFAULT_RESULTS_PATH) -> dict[str, Any]:
    """Read a saved run back, rebuilding the cell objects."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["cells"] = [_cell_from_dict(c) for c in payload.get("cells", [])]
    return payload


def _rate(successes: int, total: int) -> str:
    return str(Rate(successes=successes, total=total)) if total else "-"


def _short_model(model: str) -> str:
    """Drop the vendor prefix so the table fits a terminal.

    ``openai/gpt-oss-120b`` and ``gpt-oss-120b`` are the same row to a reader;
    the prefix only matters when calling the API.
    """
    return model.split("/")[-1]


def summary_table(cells: list[CellResult]) -> Table:
    """One row per cell: the numbers the article quotes.

    Schema difficulty stands in for the contract name because difficulty is the
    axis being varied, and each contract exists to occupy one level of it.
    """
    table = Table(title="Benchmark matrix", show_lines=False)
    for column in ("model", "tier", "schema"):
        table.add_column(column)
    table.add_column("first attempt", justify="right")
    table.add_column("after repair", justify="right")
    table.add_column("free repairs", justify="right")
    table.add_column("accuracy", justify="right")
    table.add_column("grounded", justify="right")
    table.add_column("tok/call", justify="right")
    table.add_column("no output", justify="right")

    for cell in cells:
        if cell.skipped_reason:
            table.add_row(
                _short_model(cell.model), cell.tier, cell.difficulty,
                f"[dim]skipped: {cell.skipped_reason}[/dim]", "", "", "", "", "", "",
            )
            continue
        scored = len(cell.scores)
        table.add_row(
            _short_model(cell.model),
            cell.tier,
            cell.difficulty,
            _rate(cell.first_attempt_ok, cell.total),
            _rate(cell.final_ok, cell.total),
            str(cell.free_repairs),
            _rate(sum(1 for s in cell.scores if s.accurate), scored),
            _rate(sum(1 for s in cell.scores if s.grounded), scored),
            f"{cell.tokens / cell.total:.0f}" if cell.total else "-",
            str(cell.errors) if cell.errors else "",
        )
    return table


def tier_table(cells: list[CellResult]) -> Table:
    """Rates pooled by enforcement tier — the comparison the project exists for."""
    table = Table(title="By enforcement tier")
    table.add_column("tier")
    table.add_column("first attempt", justify="right")
    table.add_column("after repair", justify="right")
    table.add_column("semantic accuracy", justify="right")

    for tier in sorted({c.tier for c in cells if not c.skipped_reason}):
        run = [c for c in cells if c.tier == tier and not c.skipped_reason]
        scores = [s for c in run for s in c.scores]
        table.add_row(
            tier,
            _rate(sum(c.first_attempt_ok for c in run), sum(c.total for c in run)),
            _rate(sum(c.final_ok for c in run), sum(c.total for c in run)),
            _rate(sum(1 for s in scores if s.accurate), len(scores)),
        )
    return table


def failure_table(cells: list[CellResult]) -> Table:
    """What went wrong on first attempt, by tier.

    Split matters more than the total: fenced JSON is repaired for free, while a
    refusal is not repairable at any price.
    """
    table = Table(title="First-attempt failures by type")
    table.add_column("tier")
    table.add_column("failure")
    table.add_column("count", justify="right")

    for tier in sorted({c.tier for c in cells if not c.skipped_reason}):
        totals: dict[str, int] = {}
        for cell in cells:
            if cell.tier != tier or cell.skipped_reason:
                continue
            for kind, count in cell.failures.items():
                totals[kind] = totals.get(kind, 0) + count
        if not totals:
            table.add_row(tier, "[dim]none[/dim]", "0")
            continue
        for kind, count in sorted(totals.items(), key=lambda kv: -kv[1]):
            table.add_row(tier, kind, str(count))
    return table


def trajectory_table(cells: list[CellResult]) -> Table:
    """The best and worst measured cells projected across a chain of calls.

    The parse rate is the measurement; this is what makes a reader care. Failures
    correlate in practice, so an independent-call projection is an upper bound
    rather than a forecast, and is labelled as one.
    """
    measured = [c for c in cells if c.total and not c.skipped_reason]
    table = Table(title="Projected chain reliability (upper bound)")
    table.add_column("cell")
    table.add_column("per call", justify="right")
    for step in TRAJECTORY_STEPS:
        table.add_column(f"{step} step{'s' if step > 1 else ''}", justify="right")

    if not measured:
        return table

    best = max(measured, key=lambda c: c.final_rate)
    worst = min(measured, key=lambda c: c.final_rate)
    for label, cell in (("best", best), ("worst", worst)):
        projection = trajectory_reliability(cell.final_rate, TRAJECTORY_STEPS)
        table.add_row(
            f"{label}: {_short_model(cell.model)} / {cell.tier} / {cell.difficulty}",
            f"{cell.final_rate:.1%}",
            *[f"{projection[s]:.1%}" for s in TRAJECTORY_STEPS],
        )
    return table


def interval_advice(cells: list[CellResult]) -> str:
    """Whether the run is large enough to say anything, in one line.

    Sizing is checked on n rather than on the printed width, because a normal
    approximation collapses to a zero-width interval at 0% and 100% — exactly
    the cells a small run produces most often, and exactly where a narrow
    interval means the least.
    """
    measured = [c for c in cells if c.total and not c.skipped_reason]
    if not measured:
        return "No cell produced data."

    smallest = min(c.total for c in measured)
    thin = sum(1 for c in measured if c.total < 30)
    intervals = [Rate(c.final_ok, c.total).interval for c in measured]
    widest = max(high - low for low, high in intervals)

    if thin:
        factor = math.ceil(30 / smallest)
        return (
            f"{thin} of {len(measured)} cells have n<30 (smallest n={smallest}); "
            f"widest interval is +/-{widest / 2:.1%}. Around {factor}x the "
            "repeats would put every cell over the threshold. Cells sitting at "
            "0% or 100% show a zero-width interval, which is an artefact of the "
            "approximation rather than precision."
        )
    return f"Every cell has n>=30; widest interval is +/-{widest / 2:.1%}."


def print_summary(
    cells: list[CellResult], console: Console | None = None
) -> None:
    """Print every table for a finished run."""
    console = console or Console()
    for build in (summary_table, tier_table, failure_table, trajectory_table):
        console.print(build(cells))
        console.print()
    console.print(interval_advice(cells))
