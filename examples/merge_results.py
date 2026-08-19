"""Merges the per-model runs of a parallel benchmark into one result set.

The ladder is run one process per model, because Groq meters tokens per minute
per model and a serial run leaves every other allowance idle. That leaves the
measurement split across files, which this puts back together: one results file
for the report, and one attempt log for the failure samples.

Usage::

    python examples/merge_results.py runs/*.json --out bench_results_groq.json
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.results import load_results, print_summary, save_results  # noqa: E402
from store.log import COLUMNS, AttemptLog  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def merge_logs(sources: list[Path], destination: Path) -> int:
    """Copy every attempt from the per-model databases into one.

    Returns:
        How many rows were copied.
    """
    target = AttemptLog(destination)
    copied = 0
    columns = ", ".join(COLUMNS)

    for source in sources:
        if not source.exists():
            continue
        # ATTACH rather than reading and re-inserting through Python: the row
        # count is in the thousands and the schemas are identical.
        target._conn.execute(f"ATTACH DATABASE '{source.as_posix()}' AS src")
        cursor = target._conn.execute(
            f"INSERT INTO attempts ({columns}) SELECT {columns} FROM src.attempts"
        )
        copied += cursor.rowcount
        target._conn.commit()
        target._conn.execute("DETACH DATABASE src")

    target.close()
    return copied


def keep_best_sampled(cells: list) -> list:
    """Collapse duplicate cells, keeping the one with the most data.

    A cell can be measured more than once — a run starved by a daily token
    budget leaves an n of one or two, and a later top-up fills it in. Keeping
    both would double-count; keeping the first would throw away the better
    measurement. A skipped cell only survives if nothing ever ran it.
    """
    best: dict[tuple, object] = {}
    for cell in cells:
        key = (cell.provider, cell.model, cell.tier, cell.contract)
        seen = best.get(key)
        if seen is None or cell.total > seen.total:
            best[key] = cell
    return list(best.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", help="per-model result files")
    parser.add_argument("--out", default="bench_results_groq.json")
    parser.add_argument("--db-out", default="runs.db")
    parser.add_argument("--logs", nargs="*", default=None, help="per-model databases")
    args = parser.parse_args()

    cells: list = []
    sampling: dict = {}
    elapsed = 0.0

    for path in args.results:
        payload = load_results(path)
        cells.extend(payload["cells"])
        # Wall-clock times do not add up across processes that ran side by side,
        # so the longest one is the run's duration.
        elapsed = max(elapsed, payload.get("sampling", {}).get("elapsed_seconds", 0.0))
        sampling.update(payload.get("sampling", {}))

    cells = keep_best_sampled(cells)

    sampling["elapsed_seconds"] = round(elapsed, 1)
    sampling["run_shape"] = f"{len(args.results)} models in parallel, one process each"
    save_results(cells, args.out, sampling=sampling)

    logs = [Path(p) for p in (args.logs or [])]
    if not logs:
        logs = [Path(p).with_suffix(".db") for p in args.results]
    copied = merge_logs(logs, Path(args.db_out))

    print(f"merged {len(cells)} cells into {args.out}")
    print(f"merged {copied} attempts into {args.db_out}\n")
    print_summary(cells)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
