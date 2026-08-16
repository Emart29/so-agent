"""End to end in under a minute: the failure, the repair, and the headline table.

Three things happen, in the order that makes the argument:

1. What the configured model actually enforces, read from the cached probe.
2. One extraction run at the weakest tier and one at the strongest, on the same
   input, so the difference is visible rather than asserted.
3. The headline table from the saved benchmark run, if one is present.

Usage::

    python examples/demo.py [--provider groq] [--model M] [--weak M]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402

from agent import StructuredAgent  # noqa: E402
from bench.results import load_results, tier_table  # noqa: E402
from config import settings  # noqa: E402
from contracts.schemas import TicketTriage  # noqa: E402
from provider.capabilities import load_capabilities  # noqa: E402
from store.log import AttemptLog  # noqa: E402
from store.metrics import Metrics, trajectory_reliability  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: A ticket with everything the hard contract asks for: a named person, a plan,
#: two distinct problems, and a tone worth classifying.
TICKET = (
    "I'm Marta Silva on the Business plan. I was billed twice on 3 October and "
    "the second charge still hasn't been refunded. On top of that the export "
    "button does nothing on Firefox, which is blocking our month-end reporting. "
    "Please have Daniel look at the billing side, he handled it last time. This "
    "is getting frustrating."
)

RESULTS_FILES = ("bench_results_groq.json", "bench_results.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=settings.PROVIDER)
    parser.add_argument("--model", default=None, help="model with native enforcement")
    parser.add_argument("--weak", default=None, help="model without it")
    args = parser.parse_args()

    console = Console()
    capabilities = load_capabilities().get(args.provider, {})
    if not capabilities:
        console.print(
            f"[red]No capability probe for {args.provider}.[/red] "
            f"Run: so-agent probe --provider {args.provider}"
        )
        return 1

    model = args.model or settings.PRIMARY_MODEL
    caps = capabilities.get(model)
    if caps is None:
        console.print(f"[red]{model} has not been probed.[/red]")
        return 1

    console.rule("[bold]1. What this model actually enforces")
    console.print(
        f"{args.provider}/{model}: "
        + ", ".join(f"{tier}={verdict}" for tier, verdict in caps.tiers.items())
    )
    console.print(f"strongest tier available: [bold]{caps.best_tier}[/bold]")
    if caps.silently_ignores:
        console.print(
            f"[red]accepts but ignores: {', '.join(caps.silently_ignores)}[/red] "
            "— the request succeeds and nothing is enforced"
        )

    store = AttemptLog(settings.LOG_DB_PATH)
    agent = StructuredAgent(provider=args.provider, model=model, store=store)

    console.rule("[bold]2. The same ticket, at two enforcement tiers")
    for tier in ("prompt_only", caps.best_tier):
        result = agent.extract(TICKET, TicketTriage, tier=tier, review=False)
        console.print(f"\n[bold]{tier}[/bold]: {result.summary()}")

        if result.outcome:
            for attempt in result.outcome.attempts:
                if attempt.ok:
                    console.print(f"  attempt {attempt.index}: validated")
                    continue
                console.print(
                    f"  attempt {attempt.index}: [red]{attempt.failure.value}[/red]"
                    + ("  (recoverable by extraction, no second call)"
                       if attempt.recovered_by_extraction else "")
                )
                if attempt.detail:
                    console.print(f"    {attempt.detail[:160]}")

        if result.ok:
            triage = result.value
            console.print(
                f"  -> {len(triage.issues)} issues, priority={triage.priority.value}, "
                f"sentiment={triage.sentiment.value}, assignee={triage.assignee}"
            )

    console.rule("[bold]3. What the benchmark measured")
    saved = next((Path(f) for f in RESULTS_FILES if Path(f).exists()), None)
    if saved:
        payload = load_results(saved)
        console.print(tier_table(payload["cells"]))
        console.print(f"[dim]from {saved}[/dim]")
    else:
        console.print(
            "[yellow]No saved benchmark run. Falling back to this log.[/yellow]"
        )

    stats = Metrics(store)
    final = stats.final_success(provider=args.provider)
    console.print(f"\nlogged success rate for {args.provider}: {final}")
    projection = trajectory_reliability(final.value)
    console.print(
        "projected across a chain (upper bound): "
        + ", ".join(f"{n} steps {v:.0%}" for n, v in projection.items())
    )
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
