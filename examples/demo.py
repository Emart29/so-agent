"""End to end in under a minute: the failure, the repair, and the headline table.

Three things happen, in the order that makes the argument:

1. What the configured model actually enforces, read from the cached probe.
2. The same ticket run twice: on a small model with nothing enforcing the shape,
   then on a model that enforces it natively. Both halves use the hardest
   contract in the set, because an easy one succeeds everywhere and shows
   nothing.
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


def _weakest_probed(capabilities: dict, exclude: str) -> str | None:
    """Pick a model with no native schema enforcement, to show the gap.

    Running both halves on a model that enforces natively would show two
    successes and demonstrate nothing.
    """
    candidates = [
        name for name, caps in capabilities.items()
        if name != exclude and not caps.enforces_schema
        and caps.tiers.get("prompt_only") == "conformed"
    ]
    return candidates[0] if candidates else None


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

    console.rule("[bold]2. The same ticket, with and without enforcement")
    console.print(
        "The hardest contract in the set, run on a small model with nothing "
        "enforcing the shape, then on a model that enforces it natively.\n"
    )

    weak = args.weak or _weakest_probed(capabilities, exclude=model)
    runs = [("no enforcement", weak, "prompt_only"), ("enforced", model, caps.best_tier)]

    for label, run_model, tier in runs:
        if run_model is None:
            continue
        agent = StructuredAgent(provider=args.provider, model=run_model, store=store)
        result = agent.extract(TICKET, TicketTriage, tier=tier, review=False)
        console.print(
            f"[bold]{label}[/bold] — {run_model} at {tier}: {result.summary()}"
        )

        for attempt in result.outcome.attempts if result.outcome else []:
            if attempt.ok:
                console.print(f"  attempt {attempt.index}: [green]validated[/green]")
                continue
            note = (
                "  (recovered locally, no second call)"
                if attempt.recovered_by_extraction else ""
            )
            console.print(
                f"  attempt {attempt.index}: [red]{attempt.failure.value}[/red]{note}"
            )
            if attempt.detail:
                console.print(f"    [dim]{attempt.detail[:150]}[/dim]")

        if result.ok:
            triage = result.value
            console.print(
                f"  -> {len(triage.issues)} issues, priority={triage.priority.value}, "
                f"sentiment={triage.sentiment.value}, assignee={triage.assignee}\n"
            )
        else:
            console.print(f"  -> [red]no usable object: {result.error}[/red]\n")

    console.rule("[bold]3. What the benchmark measured")
    saved = next((Path(f) for f in RESULTS_FILES if Path(f).exists()), None)
    if saved:
        payload = load_results(saved)
        console.print(tier_table(payload["cells"]))
        console.print(
            "[dim]Tiers pooled across models are confounded — a model that "
            "cannot run json_schema contributes only to the weaker tiers. The "
            "per-model comparison is in the report; there, enforcement wins on "
            "none of the four.[/dim]"
        )
        console.print(f"[dim]from {saved}[/dim]")
    else:
        console.print(
            "[yellow]No saved benchmark run. Falling back to this log.[/yellow]"
        )

    stats = Metrics(store)
    final = stats.final_success(provider=args.provider)
    console.print(
        f"\nEvery attempt in this log for {args.provider}, including the two "
        f"runs above and anything logged before them: {final}"
    )
    projection = trajectory_reliability(final.value)
    console.print(
        "projected across a chain of independent calls — a ceiling, because "
        "real failures correlate: "
        + ", ".join(f"{n} steps {v:.0%}" for n, v in projection.items())
    )
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
