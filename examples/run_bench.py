"""Runs the benchmark matrix and writes the results the report is built from.

Two runs with different budgets and different questions:

* **Groq** is the headline matrix — a model size ladder against every tier the
  capability probe says is possible. It answers how structured output behaves
  as models get larger, and whether the repair loop earns its place.
* **OpenRouter** is one slice, sized to the daily allowance. It runs the single
  model both providers serve, so the weights are held constant and any
  difference is attributable to the serving stack rather than to the model.

Run Groq first and completely. The OpenRouter slice isolates one variable; it is
not a second matrix, and the budget guard will refuse it if treated as one.

Usage::

    python examples/run_bench.py [--provider groq] [--k 3] [--models A B]
                                 [--tiers T] [--contracts C] [--no-review]
                                 [--out bench_results.json]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402

from agent import StructuredAgent  # noqa: E402
from bench.cases import CASES  # noqa: E402
from bench.results import DEFAULT_RESULTS_PATH, print_summary, save_results  # noqa: E402
from bench.run import BENCH_TIERS, run_matrix  # noqa: E402
from config import settings  # noqa: E402
from contracts.schemas import CONTRACTS  # noqa: E402
from store.log import AttemptLog  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: A size ladder rather than a shortlist of favourites. The question is how
#: enforcement behaves as capacity grows, which needs the small models in it.
GROQ_LADDER = [
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
]

#: The one model both providers serve. Holding the weights constant is the
#: entire point of the second provider.
OPENROUTER_SLICE = ["openai/gpt-oss-20b:free"]

DEFAULT_MODELS = {"groq": GROQ_LADDER, "openrouter": OPENROUTER_SLICE}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--tiers", nargs="*", default=list(BENCH_TIERS))
    parser.add_argument("--contracts", nargs="*", default=list(CONTRACTS))
    parser.add_argument("--k", type=int, default=3, help="repeats per case")
    parser.add_argument("--cases", type=int, default=0, help="limit the case set")
    parser.add_argument("--no-review", action="store_true", help="skip the critic")
    parser.add_argument("--out", default=str(DEFAULT_RESULTS_PATH))
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    console = Console()

    models = args.models or DEFAULT_MODELS.get(args.provider)
    if not models:
        console.print(f"[red]No default model ladder for {args.provider}.[/red]")
        return 1

    cases = CASES[: args.cases] if args.cases else CASES
    review = not args.no_review
    critic_model = settings.CRITIC_MODEL if review else None
    store = AttemptLog(settings.LOG_DB_PATH)

    if critic_model and critic_model in models:
        # Worth stating rather than burying: in one row of the ladder the critic
        # is reviewing its own output. The label-based scoring is unaffected,
        # but the critic column for that model is not independent of it.
        console.print(
            f"[yellow]Note: {critic_model} is both a subject and the critic, so "
            "its critic column reviews its own output.[/yellow]"
        )

    planned = len(models) * len(args.tiers) * len(args.contracts)
    console.print(
        f"[bold]{args.provider}[/bold]: {len(models)} models x {len(args.tiers)} "
        f"tiers x {len(args.contracts)} contracts, {len(cases)} cases x {args.k} "
        f"repeats -> up to {planned} cells"
    )

    started = time.time()
    done: list = []

    def _sampling(elapsed: float) -> dict:
        """The settings behind the numbers.

        Saved with the run because a rate quoted without them is not
        reproducible even as a sample.
        """
        return {
            "provider": args.provider,
            "repeats": args.k,
            "cases": len(cases),
            "review": review,
            "critic_model": critic_model or "",
            "temperature": "provider default (not pinned)",
            "critic_judges_itself": bool(critic_model) and critic_model in models,
            "elapsed_seconds": round(elapsed, 1),
        }

    def progress(cell) -> None:
        done.append(cell)
        label = f"{cell.model} / {cell.tier} / {cell.contract}"
        if cell.skipped_reason:
            console.print(f"  [dim]{len(done)}/{planned} skip {label}: "
                          f"{cell.skipped_reason}[/dim]")
        else:
            console.print(
                f"  {len(done)}/{planned} {label}: "
                f"first {cell.first_attempt_rate:.0%}, "
                f"final {cell.final_rate:.0%}, accurate {cell.accuracy:.0%}"
            )
        # Checkpoint after every cell. A full matrix is hours of real requests,
        # and losing all of it to a dropped connection in the last cell would
        # mean paying for it twice.
        save_results(done, args.out, sampling=_sampling(time.time() - started))

    def make_agent(model: str) -> StructuredAgent:
        return StructuredAgent(
            provider=args.provider,
            model=model,
            critic_model=critic_model,
            store=store,
        )

    try:
        cells = run_matrix(
            make_agent,
            provider=args.provider,
            models=models,
            tiers=args.tiers,
            contracts=args.contracts,
            repeats=args.k,
            cases=cases,
            review=review,
            on_cell=progress,
        )
    except KeyboardInterrupt:
        console.print(
            f"[yellow]Interrupted after {len(done)} cells; "
            f"{args.out} holds what finished.[/yellow]"
        )
        return 130

    elapsed = time.time() - started
    path = save_results(cells, args.out, sampling=_sampling(elapsed))

    console.print()
    print_summary(cells, console)
    console.print(f"\nWrote {path} in {elapsed / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
