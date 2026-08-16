"""The command line: everything the library does, without writing Python.

Two rules shape the surface:

* **``--provider`` is accepted everywhere and defaults to the configured one.**
  A command that only works against the default would quietly make the default
  the only measurable thing, which is the opposite of what a provider-agnostic
  library is for.
* **Nothing that spends a metered allowance runs implicitly.** ``providers``
  answers "am I set up?" and "how much budget is left?" without making a single
  request, because that is the first thing anyone runs on a fresh clone.

Failures exit non-zero so the commands compose in a shell.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from bench.results import DEFAULT_RESULTS_PATH, load_results, print_summary
from config import settings
from contracts.schemas import CONTRACTS, get_contract
from provider.capabilities import (
    CAPABILITIES_PATH,
    CapabilityProber,
    ModelCapabilities,
    TierSupport,
    chat_models,
    is_stale,
    load_capabilities,
    save_capabilities,
)
from provider.client import BudgetExhaustedError, LLMClient
from provider.registry import PROVIDERS, UnknownProviderError, get_provider
from store.log import AttemptLog
from store.metrics import Metrics, trajectory_reliability
from store.requests import RequestLog

app = typer.Typer(
    add_completion=False,
    help="Structured output across providers, measured rather than assumed.",
    no_args_is_help=True,
)
console = Console()

#: Tier order, weakest last. Used wherever tiers are displayed as columns.
TIERS = ("json_schema", "json_object", "tools", "prompt_only")

#: How each probe verdict reads in a table. "IGNORED" is shouted because it is
#: the dangerous one: the request succeeded and nothing was enforced.
MARKS = {
    TierSupport.CONFORMED.value: "[green]yes[/green]",
    TierSupport.IGNORED.value: "[red]IGNORED[/red]",
    TierSupport.GEN_FAILED.value: "[yellow]gen-fail[/yellow]",
    TierSupport.REJECTED.value: "[dim]no[/dim]",
    TierSupport.ERROR.value: "[dim]err[/dim]",
    TierSupport.UNTESTED.value: "-",
}


def _fail(message: str) -> None:
    """Print an error and exit non-zero."""
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(1)


def _resolve_provider(name: str | None) -> str:
    """Validate a provider name, naming the alternatives when it is wrong."""
    chosen = name or settings.PROVIDER
    try:
        get_provider(chosen)
    except UnknownProviderError as exc:
        _fail(str(exc))
    return chosen


@app.command()
def providers() -> None:
    """Show configured providers, their keys, probes, and budget left today.

    Makes no requests. Reporting how much of a daily allowance is left should
    never cost part of it.
    """
    cache = load_capabilities()
    usage = RequestLog(settings.LOG_DB_PATH).usage_today()

    table = Table(title="Providers")
    table.add_column("provider")
    table.add_column("key")
    table.add_column("probe")
    table.add_column("models", justify="right")
    table.add_column("used today", justify="right")
    table.add_column("default")

    for name, provider in PROVIDERS.items():
        probed = cache.get(name, {})
        if not probed:
            probe = "[yellow]not probed[/yellow]"
        else:
            newest = max(c.probed_at for c in probed.values())
            stale = any(is_stale(c) for c in probed.values())
            probe = f"{newest[:10]}" + (" [yellow](stale)[/yellow]" if stale else "")

        used = usage.get(name, 0)
        if provider.daily_budget is None:
            budget = f"{used}" if used else "-"
        else:
            left = max(provider.daily_budget - used, 0)
            colour = "red" if left == 0 else "yellow" if left < 10 else "green"
            budget = f"{used}/{provider.daily_budget} ([{colour}]{left} left[/{colour}])"

        if provider.key_env is None:
            # A local endpoint needs no credential, which is not the same thing
            # as having one configured.
            key = "[dim]not needed[/dim]"
        else:
            key = "[green]set[/green]" if provider.has_key() else "[dim]missing[/dim]"

        table.add_row(
            name,
            key,
            probe,
            str(len(probed)) if probed else "-",
            budget,
            "yes" if provider.is_default_eligible else "[dim]on request[/dim]",
        )

    console.print(table)
    console.print(
        f"\nDefault provider: [bold]{settings.PROVIDER}[/bold]  "
        f"primary model: [bold]{settings.PRIMARY_MODEL or 'not set'}[/bold]"
    )
    if not settings.PRIMARY_MODEL:
        console.print("[yellow]Run `probe` first; no model is configured.[/yellow]")


@app.command()
def probe(
    provider: str = typer.Option(None, help="Provider to probe."),
    models: list[str] = typer.Option(None, "--model", "-m", help="Limit to these ids."),
    refresh: bool = typer.Option(False, help="Re-probe models already cached."),
    features: bool = typer.Option(True, help="Also probe schema feature support."),
    limit: int = typer.Option(0, help="Cap how many discovered models are probed."),
) -> None:
    """Measure what a provider's models actually enforce, and cache it.

    Everything downstream reads this rather than a hardcoded list of what
    providers claim to support.
    """
    name = _resolve_provider(provider)
    client = LLMClient(name)
    prober = CapabilityProber(client)

    cache = load_capabilities()
    cached = cache.get(name, {})

    targets = list(models) if models else chat_models(client.list_models())
    if limit:
        targets = targets[:limit]
    if not targets:
        _fail(f"no models to probe on {name}. Pass --model explicitly.")

    console.print(f"probing {len(targets)} models on [bold]{name}[/bold]")

    results: list[ModelCapabilities] = []
    for model in targets:
        known = cached.get(model)
        if known and not refresh and not is_stale(known):
            console.print(f"  [dim]{model} (cached {known.probed_at[:10]})[/dim]")
            results.append(known)
            continue
        console.print(f"  {model}", end="")
        try:
            caps = prober.probe_model(model, features=features)
        except BudgetExhaustedError as exc:
            console.print(f"\n[yellow]{exc}[/yellow]")
            break
        results.append(caps)
        cached[model] = caps
        console.print(f" -> {caps.best_tier}")

    cache[name] = cached
    save_capabilities(cache)
    console.print(f"saved to {CAPABILITIES_PATH}\n")
    console.print(_capability_table(results))

    ignoring = [c for c in results if c.silently_ignores]
    if ignoring:
        console.print(
            "\n[red]Accepted an enforcement directive without honouring it:[/red]"
        )
        for caps in ignoring:
            console.print(f"  {caps.model}: {', '.join(caps.silently_ignores)}")
        console.print(
            "[dim]These are the dangerous case: the request succeeds and nothing "
            "is enforced, which is indistinguishable from working code until the "
            "output is checked.[/dim]"
        )


def _capability_table(results: list[ModelCapabilities]) -> Table:
    table = Table(title="Enforcement support")
    table.add_column("model")
    for tier in TIERS:
        table.add_column(tier)
    table.add_column("best")
    for caps in sorted(results, key=lambda c: c.model):
        table.add_row(
            caps.model,
            *[MARKS.get(caps.tiers.get(t, "-"), "?") for t in TIERS],
            caps.best_tier,
        )
    return table


@app.command()
def extract(
    schema: str = typer.Option(..., "--schema", "-s", help=f"One of: {', '.join(CONTRACTS)}"),
    text: str = typer.Option(None, "--text", "-t", help="Source text."),
    file: Path = typer.Option(None, "--file", "-f", help="Read source text from a file."),
    provider: str = typer.Option(None, help="Provider to use."),
    model: str = typer.Option(None, help="Model id. Defaults to the configured one."),
    tier: str = typer.Option(None, help="Force an enforcement tier."),
    review: bool = typer.Option(False, help="Run the semantic critic as well."),
) -> None:
    """Extract one validated object from text and print it as JSON."""
    from agent import StructuredAgent

    if not text and not file:
        _fail("pass --text or --file")
    source = text or file.read_text(encoding="utf-8")

    try:
        contract = get_contract(schema)
    except KeyError as exc:
        _fail(str(exc))

    name = _resolve_provider(provider)
    agent = StructuredAgent(
        provider=name,
        model=model,
        critic_model=settings.CRITIC_MODEL if review else None,
        store=AttemptLog(settings.LOG_DB_PATH),
    )

    result = agent.extract(source, contract, tier=tier)
    console.print(f"[dim]{result.summary()}[/dim]")

    if not result.ok:
        _fail(f"extraction failed: {result.summary()}")

    console.print_json(result.value.model_dump_json())
    if result.semantically_sound is False:
        # Valid and wrong is the failure a schema cannot catch, so it is not
        # allowed to exit zero.
        console.print("[red]The critic judged this unsound.[/red]")
        raise typer.Exit(2)


@app.command()
def bench(
    provider: str = typer.Option(None, help="Provider to measure."),
    models: list[str] = typer.Option(None, "--model", "-m", help="Models to measure."),
    tiers: list[str] = typer.Option(None, "--tier", help="Tiers to attempt."),
    contracts: list[str] = typer.Option(None, "--contract", help="Contracts to run."),
    k: int = typer.Option(3, help="Repeats per case. Samples, not reproductions."),
    review: bool = typer.Option(True, help="Run the critic alongside the labels."),
    out: Path = typer.Option(DEFAULT_RESULTS_PATH, help="Where to write results."),
) -> None:
    """Run the benchmark matrix and write the results the report is built from.

    Every cell is checkpointed as it finishes, so an interrupted run keeps what
    it already paid for.
    """
    from agent import StructuredAgent
    from bench.cases import CASES
    from bench.results import save_results
    from bench.run import BENCH_TIERS, run_matrix

    name = _resolve_provider(provider)
    if not models:
        _fail("pass --model at least once; there is no default ladder here")

    store = AttemptLog(settings.LOG_DB_PATH)
    critic_model = settings.CRITIC_MODEL if review else None
    finished: list = []

    def on_cell(cell) -> None:
        finished.append(cell)
        label = f"{cell.model} / {cell.tier} / {cell.contract}"
        if cell.skipped_reason:
            console.print(f"  [dim]skip {label}: {cell.skipped_reason}[/dim]")
        else:
            console.print(
                f"  {label}: first {cell.first_attempt_rate:.0%}, "
                f"final {cell.final_rate:.0%}, accurate {cell.accuracy:.0%}"
            )
        save_results(finished, out, sampling={"provider": name, "repeats": k})

    cells = run_matrix(
        lambda m: StructuredAgent(
            provider=name, model=m, critic_model=critic_model, store=store
        ),
        provider=name,
        models=list(models),
        tiers=list(tiers) if tiers else list(BENCH_TIERS),
        contracts=list(contracts) if contracts else list(CONTRACTS),
        repeats=k,
        cases=CASES,
        review=review,
        on_cell=on_cell,
    )
    save_results(cells, out, sampling={"provider": name, "repeats": k})
    console.print()
    print_summary(cells, console)
    console.print(f"\nWrote {out}")


@app.command()
def metrics(
    provider: str = typer.Option(None, help="Restrict to one provider."),
    model: str = typer.Option(None, help="Restrict to one model."),
    tier: str = typer.Option(None, help="Restrict to one enforcement tier."),
) -> None:
    """Print the tables computed from the attempt log."""
    log = AttemptLog(settings.LOG_DB_PATH)
    if log.count() == 0:
        _fail("the log is empty. Run `extract` or `bench` first.")

    name = provider or None
    stats = Metrics(log)

    headline = Table(title="Success")
    headline.add_column("measure")
    headline.add_column("rate")
    first = stats.first_attempt_success(name, model, tier)
    final = stats.final_success(name, model, tier)
    headline.add_row("first attempt", str(first))
    headline.add_row("after repair", str(final))
    headline.add_row("repairs that were free", str(stats.free_repair_share(name, model, tier)))
    console.print(headline)

    breakdown = stats.error_breakdown(name, model, tier)
    if breakdown:
        errors = Table(title="First-attempt failures")
        errors.add_column("type")
        errors.add_column("count", justify="right")
        for kind, count in sorted(breakdown.items(), key=lambda kv: -kv[1]):
            errors.add_row(kind, str(count))
        console.print(errors)

    by_difficulty = stats.by_schema_difficulty(name, model, tier)
    if by_difficulty:
        difficulty = Table(title="By schema difficulty")
        difficulty.add_column("difficulty")
        difficulty.add_column("first attempt")
        for level, rate in by_difficulty.items():
            difficulty.add_row(level, str(rate))
        console.print(difficulty)

    accuracy = stats.accuracy_by_tier(name, model)
    if accuracy:
        semantic = Table(title="Critic verdicts by tier")
        semantic.add_column("tier")
        semantic.add_column("judged sound")
        for level, rate in accuracy.items():
            semantic.add_row(level, str(rate))
        console.print(semantic)

    projection = trajectory_reliability(final.value)
    chain = Table(title="Projected chain reliability (upper bound)")
    chain.add_column("steps", justify="right")
    chain.add_column("still correct", justify="right")
    for steps, value in projection.items():
        chain.add_row(str(steps), f"{value:.1%}")
    console.print(chain)
    console.print(
        "[dim]Failures correlate in practice, so this is a ceiling rather than "
        "a forecast.[/dim]"
    )


@app.command()
def report(
    results: list[Path] = typer.Option(
        None, "--results", "-r", help="Benchmark result files to include."
    ),
    out: Path = typer.Option(Path("report.html"), help="Where to write the report."),
) -> None:
    """Build the self-contained HTML report."""
    from report.build import build_report

    files = list(results) if results else [DEFAULT_RESULTS_PATH]
    missing = [f for f in files if not f.exists()]
    if missing:
        _fail(f"no such results file: {', '.join(str(f) for f in missing)}")

    path = build_report(files, out, log=AttemptLog(settings.LOG_DB_PATH))
    console.print(f"wrote {path}")


@app.command()
def replay(
    run_id: str = typer.Argument(..., help="Run id from the log."),
    provider: str = typer.Option(None, help="Provider to replay against."),
    model: str = typer.Option(None, help="Model to replay against."),
    tier: str = typer.Option(None, help="Tier to replay under."),
) -> None:
    """Re-run a logged attempt somewhere else.

    The practical argument for keeping raw output: "this failed in production,
    would native enforcement have caught it?" is answerable directly, against
    the same input, without reconstructing anything by hand.
    """
    from agent import StructuredAgent

    log = AttemptLog(settings.LOG_DB_PATH)
    rows = log.attempts_for(run_id)
    if not rows:
        _fail(f"no attempts logged for run {run_id}")

    original = rows[0]
    source = _source_text(log, run_id)
    if not source:
        _fail(
            f"run {run_id} has no stored source text, so it cannot be replayed. "
            "Only runs recorded with input capture can be."
        )

    console.print(
        f"[dim]original: {original['provider']}/{original['model']} at "
        f"{original['tier']} -> "
        f"{'ok' if original['success'] else original['failure_type']}[/dim]"
    )

    contract = get_contract(_contract_name(original["schema_name"]))
    agent = StructuredAgent(
        provider=_resolve_provider(provider or original["provider"]),
        model=model or original["model"],
        store=log,
    )
    result = agent.extract(source, contract, tier=tier)

    console.print(
        f"[dim]replay:   {agent.provider}/{agent.model} at "
        f"{tier or 'selected'} -> {result.summary()}[/dim]"
    )
    if result.ok:
        console.print_json(result.value.model_dump_json())
    else:
        raise typer.Exit(1)


def _source_text(log: AttemptLog, run_id: str) -> str | None:
    """Recover the input a run was given, if it was recorded."""
    rows = log.query(
        "SELECT source_text FROM attempts WHERE run_id = ? AND source_text IS NOT NULL"
        " LIMIT 1",
        (run_id,),
    )
    return rows[0]["source_text"] if rows else None


def _contract_name(schema_name: str) -> str:
    """Map a logged schema name back to its registry key."""
    for key, contract in CONTRACTS.items():
        if contract.__name__ == schema_name or key == schema_name:
            return key
    return schema_name


def main() -> None:
    """Entry point, so failures exit non-zero rather than tracebacking."""
    try:
        app()
    except (BudgetExhaustedError, UnknownProviderError) as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
