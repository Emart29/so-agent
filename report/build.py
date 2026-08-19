"""Builds a self-contained HTML report from a benchmark run and the log.

One file, no server, no build step, no external requests. Charts are inline SVG
rather than a charting library, because the point of a report is that it still
opens in ten years from a directory nobody has touched since.

What goes in it is chosen to make the numbers checkable rather than impressive:
every rate carries its interval and its n, every skipped cell says why it was
skipped, and the failure sample shows the raw bytes that failed rather than a
summary of them.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from bench.results import load_results
from bench.run import CellResult
from provider.capabilities import TierSupport, load_capabilities
from store.log import AttemptLog
from store.metrics import Metrics, Rate, trajectory_reliability

#: Tier order, strongest first. Every table that has a tier axis uses it, so the
#: reader compares the same columns in the same order throughout.
TIER_ORDER = ("json_schema", "json_object", "tools", "prompt_only")

#: How many real failures to show. Enough to be evidence, few enough to read.
FAILURE_SAMPLE = 12

#: Longest raw output shown before truncation. A five-thousand-token truncated
#: generation is not more informative than its first lines.
RAW_LIMIT = 600

CSS = """
:root {
  --ink: #1a1a1a; --muted: #666; --line: #e0e0e0; --bg: #fff;
  --good: #1a7f37; --bad: #b3261e; --warn: #9a6700; --accent: #24417a;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2.5rem 1.5rem 4rem; background: var(--bg); color: var(--ink);
  font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 2rem; margin: 0 0 .25rem; letter-spacing: -.02em; }
h2 { font-size: 1.35rem; margin: 3rem 0 .5rem; letter-spacing: -.01em; }
h3 { font-size: 1rem; margin: 2rem 0 .5rem; color: var(--muted);
     text-transform: uppercase; letter-spacing: .06em; }
p { margin: .5rem 0 1rem; max-width: 68ch; }
.sub { color: var(--muted); margin: 0 0 2rem; }
.note { color: var(--muted); font-size: .9rem; max-width: 68ch; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0 .5rem;
        font-size: .92rem; }
th, td { text-align: left; padding: .5rem .7rem; border-bottom: 1px solid var(--line);
         vertical-align: top; }
th { font-weight: 600; color: var(--muted); font-size: .8rem;
     text-transform: uppercase; letter-spacing: .04em; white-space: nowrap; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr:hover td { background: #fafafa; }
.yes { color: var(--good); font-weight: 600; }
.no { color: var(--muted); }
.ignored { color: var(--bad); font-weight: 700; }
.genfail { color: var(--warn); font-weight: 600; }
.skip { color: var(--muted); font-style: italic; }
.n { color: var(--muted); font-size: .85em; white-space: nowrap; }
.cards { display: flex; flex-wrap: wrap; gap: 1rem; margin: 1.5rem 0; }
.card { flex: 1 1 200px; border: 1px solid var(--line); border-radius: 8px;
        padding: 1rem 1.2rem; }
.card .value { font-size: 1.8rem; font-weight: 600; letter-spacing: -.02em; }
.card .label { color: var(--muted); font-size: .82rem; text-transform: uppercase;
               letter-spacing: .05em; }
pre { background: #f6f6f6; border: 1px solid var(--line); border-radius: 6px;
      padding: .7rem .9rem; overflow-x: auto; font-size: .82rem; margin: .4rem 0 0;
      white-space: pre-wrap; word-break: break-word; }
figure { margin: 1rem 0 2rem; }
svg { max-width: 100%; height: auto; }
footer { margin-top: 4rem; padding-top: 1rem; border-top: 1px solid var(--line);
         color: var(--muted); font-size: .85rem; }
"""

TIER_CLASS = {
    TierSupport.CONFORMED.value: "yes",
    TierSupport.IGNORED.value: "ignored",
    TierSupport.GEN_FAILED.value: "genfail",
    TierSupport.REJECTED.value: "no",
    TierSupport.ERROR.value: "no",
    TierSupport.UNTESTED.value: "no",
}

TIER_LABEL = {
    TierSupport.CONFORMED.value: "yes",
    TierSupport.IGNORED.value: "IGNORED",
    TierSupport.GEN_FAILED.value: "gen-fail",
    TierSupport.REJECTED.value: "no",
    TierSupport.ERROR.value: "error",
    TierSupport.UNTESTED.value: "-",
}


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def short_model(model: str) -> str:
    return model.split("/")[-1]


def rate_cell(successes: int, total: int) -> str:
    """A rate with its interval and n, or a dash when there is nothing to show."""
    if not total:
        return '<td class="num no">-</td>'
    rate = Rate(successes, total)
    low, high = rate.interval
    return (
        f'<td class="num">{rate.value:.0%}'
        f'<div class="n">{low:.0%}-{high:.0%}, n={total}</div></td>'
    )


def bar_chart(rows: list[tuple[str, float, int]], title: str) -> str:
    """A horizontal bar chart as inline SVG.

    Hand-written rather than drawn by a library: a chart that needs a CDN is a
    chart that stops rendering the day the CDN moves.
    """
    if not rows:
        return ""
    row_height, label_width, width = 30, 260, 900
    height = row_height * len(rows) + 30
    bars = []
    for i, (label, value, n) in enumerate(rows):
        y = i * row_height + 14
        bar_width = max((width - label_width - 90) * value, 1)
        colour = "#1a7f37" if value >= 0.95 else "#9a6700" if value >= 0.8 else "#b3261e"
        bars.append(
            f'<text x="0" y="{y + 13}" font-size="13" fill="#1a1a1a">'
            f"{esc(label)}</text>"
            f'<rect x="{label_width}" y="{y}" width="{bar_width:.1f}" height="18" '
            f'rx="3" fill="{colour}"></rect>'
            f'<text x="{label_width + bar_width + 8:.1f}" y="{y + 13}" font-size="12" '
            f'fill="#666">{value:.0%} (n={n})</text>'
        )
    return (
        f"<figure><figcaption class='note'>{esc(title)}</figcaption>"
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{esc(title)}">{"".join(bars)}</svg></figure>'
    )


def capability_section(capabilities: dict) -> str:
    """What each model actually enforced when asked, and when that was measured."""
    rows = []
    for provider, models in sorted(capabilities.items()):
        for model, caps in sorted(models.items()):
            cells = "".join(
                f'<td class="{TIER_CLASS.get(caps.tiers.get(t, "-"), "no")}">'
                f'{TIER_LABEL.get(caps.tiers.get(t, "-"), "-")}</td>'
                for t in TIER_ORDER
            )
            rows.append(
                f"<tr><td>{esc(provider)}</td><td>{esc(short_model(model))}</td>"
                f"{cells}<td>{esc(caps.best_tier)}</td>"
                f'<td class="n">{esc(caps.probed_at[:10])}</td></tr>'
            )
    if not rows:
        return ""

    headers = "".join(f"<th>{esc(t)}</th>" for t in TIER_ORDER)
    ignoring = [
        (p, m, c) for p, ms in capabilities.items() for m, c in ms.items()
        if c.silently_ignores
    ]
    warning = ""
    if ignoring:
        listed = "".join(
            f"<li>{esc(p)}/{esc(short_model(m))}: "
            f"{esc(', '.join(c.silently_ignores))}</li>"
            for p, m, c in ignoring
        )
        warning = (
            "<p class='note'><strong>Accepted a directive without honouring "
            "it.</strong> These are the dangerous case: the request succeeds, "
            "nothing is enforced, and the code looks correct until the output is "
            f"checked.</p><ul class='note'>{listed}</ul>"
        )

    return (
        "<h2>What each model actually enforces</h2>"
        "<p>Measured by sending each directive and checking the response, not "
        "read from documentation. A provider that accepts an enforcement "
        "directive and ignores it is indistinguishable from one that honours "
        "it until you look at the output.</p>"
        f"<table><thead><tr><th>provider</th><th>model</th>{headers}"
        f"<th>best</th><th>probed</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>{warning}"
    )


def matrix_section(cells: list[CellResult]) -> str:
    """Every cell of the run, skipped ones included."""
    rows = []
    for cell in cells:
        model = esc(short_model(cell.model))
        head = (
            f"<tr><td>{esc(cell.provider)}</td><td>{model}</td>"
            f"<td>{esc(cell.tier)}</td><td>{esc(cell.difficulty)}</td>"
        )
        if cell.skipped_reason:
            rows.append(
                head + f'<td class="skip" colspan="7">{esc(cell.skipped_reason)}'
                "</td></tr>"
            )
            continue
        if not cell.total:
            # Attempted and produced nothing: every call errored before a
            # generation came back. Distinct from a skipped cell, which was
            # never tried, and shown rather than dropped so the reader can see
            # the difference.
            rows.append(
                head + '<td class="skip" colspan="7">every call errored '
                f'({cell.errors} attempts, no output)</td></tr>'
            )
            continue
        scored = len(cell.scores)
        rows.append(
            head
            + rate_cell(cell.first_attempt_ok, cell.total)
            + rate_cell(cell.final_ok, cell.total)
            + rate_cell(sum(1 for s in cell.scores if s.accurate), scored)
            + rate_cell(sum(1 for s in cell.scores if s.grounded), scored)
            + f'<td class="num">{cell.tokens / cell.total:.0f}</td>'
            + f'<td class="num">{cell.latency_ms / cell.total:.0f}</td>'
            + f'<td class="num">{cell.errors or "-"}</td></tr>'
        )
    return (
        "<h2>The matrix</h2>"
        "<p>One row per (model, tier, schema difficulty). Structural success is "
        "whether the output parsed and validated; semantic accuracy is whether "
        "it was faithful to the source, scored against hand-written labels "
        "rather than against a model. <em>Grounded</em> asks whether any field "
        "was invented, so it only bites on contracts that carry fields the "
        "source may not supply — a summary has none, and scores 100% by "
        "construction. <em>No output</em> counts calls where the transport gave "
        "up after its retries, almost always a rate limit; those are excluded "
        "from the denominators, because a rate limit says nothing about whether "
        "a model can satisfy a schema.</p>"
        "<table><thead><tr><th>provider</th><th>model</th><th>tier</th>"
        "<th>schema</th><th class='num'>first attempt</th>"
        "<th class='num'>after repair</th><th class='num'>accurate</th>"
        "<th class='num'>grounded</th><th class='num'>tokens</th>"
        "<th class='num'>ms</th><th class='num'>no output</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def tier_section(cells: list[CellResult]) -> str:
    """Structural success and semantic accuracy side by side, per tier.

    The comparison the project exists for. If they diverge, "valid" and
    "correct" are measuring different things.
    """
    run = [c for c in cells if not c.skipped_reason and c.total]
    if not run:
        return ""

    rows, chart_rows = [], []
    for tier in TIER_ORDER:
        cut = [c for c in run if c.tier == tier]
        if not cut:
            continue
        total = sum(c.total for c in cut)
        scores = [s for c in cut for s in c.scores]
        rows.append(
            f"<tr><td>{esc(tier)}</td>"
            + rate_cell(sum(c.first_attempt_ok for c in cut), total)
            + rate_cell(sum(c.final_ok for c in cut), total)
            + rate_cell(sum(1 for s in scores if s.accurate), len(scores))
            + rate_cell(sum(1 for s in scores if s.grounded), len(scores))
            + "</tr>"
        )
        chart_rows.append(
            (f"{tier} — first attempt", Rate(
                sum(c.first_attempt_ok for c in cut), total
            ).value, total)
        )
        if scores:
            chart_rows.append((
                f"{tier} — semantically accurate",
                Rate(sum(1 for s in scores if s.accurate), len(scores)).value,
                len(scores),
            ))

    return (
        "<h2>Valid versus correct</h2>"
        "<p>Pooled across models and schemas. A tier that parses everything and "
        "gets the content wrong has solved the easier half of the problem.</p>"
        "<table><thead><tr><th>tier</th><th class='num'>first attempt</th>"
        "<th class='num'>after repair</th><th class='num'>semantically accurate</th>"
        "<th class='num'>invented nothing</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        + bar_chart(chart_rows, "Structural success against semantic accuracy")
    )


def failure_section(cells: list[CellResult]) -> str:
    """What went wrong on first attempt, split by whether repair was free."""
    run = [c for c in cells if not c.skipped_reason and c.total]
    if not run:
        return ""

    per_tier: dict[str, dict[str, int]] = {}
    for cell in run:
        bucket = per_tier.setdefault(cell.tier, {})
        for kind, count in cell.failures.items():
            bucket[kind] = bucket.get(kind, 0) + count

    kinds = sorted({k for b in per_tier.values() for k in b})
    if not kinds:
        return (
            "<h2>First-attempt failures</h2>"
            "<p>No first attempt failed anywhere in this run.</p>"
        )

    header = "".join(f"<th class='num'>{esc(k)}</th>" for k in kinds)
    rows = []
    for tier in TIER_ORDER:
        if tier not in per_tier:
            continue
        bucket = per_tier[tier]
        cut = [c for c in run if c.tier == tier]
        cells_html = "".join(
            f"<td class='num'>{bucket.get(k, 0) or '-'}</td>" for k in kinds
        )
        rows.append(
            f"<tr><td>{esc(tier)}</td>{cells_html}"
            f"<td class='num'>{sum(c.free_repairs for c in cut)}</td></tr>"
        )

    return (
        "<h2>First-attempt failures</h2>"
        "<p>The split matters more than the total. Markdown fences around "
        "otherwise-valid JSON are repaired by extracting them, costing nothing; "
        "a truncated generation needs a larger budget rather than another "
        "attempt; a refusal is not repairable at any price.</p>"
        f"<table><thead><tr><th>tier</th>{header}"
        "<th class='num'>free repairs</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def repair_section(cells: list[CellResult]) -> str:
    """What the repair loop bought, and what it cost."""
    run = [c for c in cells if not c.skipped_reason and c.total]
    if not run:
        return ""

    rows = []
    for tier in TIER_ORDER:
        cut = [c for c in run if c.tier == tier]
        if not cut:
            continue
        total = sum(c.total for c in cut)
        first = Rate(sum(c.first_attempt_ok for c in cut), total)
        final = Rate(sum(c.final_ok for c in cut), total)
        free = sum(c.free_repairs for c in cut)
        recovered = final.successes - first.successes
        rows.append(
            f"<tr><td>{esc(tier)}</td>"
            f"<td class='num'>{first.value:.0%}</td>"
            f"<td class='num'>{final.value:.0%}</td>"
            f"<td class='num'>+{final.value - first.value:.1%}</td>"
            f"<td class='num'>{recovered}</td>"
            f"<td class='num'>{free}</td>"
            f"<td class='num'>{sum(c.tokens for c in cut) / total:.0f}</td>"
            f"<td class='num'>{sum(c.latency_ms for c in cut) / total:.0f}</td></tr>"
        )

    return (
        "<h2>What repair bought</h2>"
        "<p>Tokens and latency are per completed extraction, so they already "
        "include whatever the retries cost.</p>"
        "<table><thead><tr><th>tier</th><th class='num'>first attempt</th>"
        "<th class='num'>after repair</th><th class='num'>lift</th>"
        "<th class='num'>recovered</th><th class='num'>free</th>"
        "<th class='num'>tokens/call</th><th class='num'>ms/call</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def trajectory_section(cells: list[CellResult]) -> str:
    """The per-call rate projected across a chain, for the best and worst cells."""
    run = [c for c in cells if not c.skipped_reason and c.total]
    if not run:
        return ""

    steps = (1, 5, 10, 50)
    best = max(run, key=lambda c: c.final_rate)
    worst = min(run, key=lambda c: c.final_rate)
    rows = []
    for label, cell in (("best cell", best), ("worst cell", worst)):
        projected = trajectory_reliability(cell.final_rate, steps)
        rows.append(
            f"<tr><td>{esc(label)}: {esc(short_model(cell.model))} / "
            f"{esc(cell.tier)} / {esc(cell.difficulty)}</td>"
            f"<td class='num'>{cell.final_rate:.1%}</td>"
            + "".join(f"<td class='num'>{projected[s]:.1%}</td>" for s in steps)
            + "</tr>"
        )
    header = "".join(f"<th class='num'>{s} steps</th>" for s in steps)
    return (
        "<h2>What this means for a chain</h2>"
        "<p>A per-call success rate compounds. The projection assumes calls fail "
        "independently, which they do not — a model that struggles with a schema "
        "struggles with it repeatedly — so these are ceilings rather than "
        "forecasts, and the real numbers are worse.</p>"
        f"<table><thead><tr><th>cell</th><th class='num'>per call</th>{header}"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def sample_failures(log: AttemptLog, limit: int = FAILURE_SAMPLE) -> str:
    """Real failures with the bytes that caused them.

    A failure taxonomy is an assertion until the reader can see the output that
    was classified, so the raw text goes in rather than a description of it.
    """
    try:
        rows = log.query(
            "SELECT provider, model, tier, schema_name, failure_type,"
            " failure_detail, raw_output FROM attempts"
            " WHERE success = 0 AND raw_output IS NOT NULL AND raw_output != ''"
            " GROUP BY failure_type, tier, model"
            " ORDER BY failure_type LIMIT ?",
            (limit,),
        )
    except Exception:  # noqa: BLE001 - a missing log is not a broken report
        return ""
    if not rows:
        return ""

    blocks = []
    for row in rows:
        raw = row["raw_output"] or ""
        clipped = raw[:RAW_LIMIT] + ("\n... [truncated]" if len(raw) > RAW_LIMIT else "")
        blocks.append(
            f"<h3>{esc(row['failure_type'])} — {esc(short_model(row['model']))} "
            f"at {esc(row['tier'])} on {esc(row['schema_name'])}</h3>"
            + (f"<p class='note'>{esc(row['failure_detail'])}</p>"
               if row["failure_detail"] else "")
            + f"<pre>{esc(clipped)}</pre>"
        )
    return (
        "<h2>What failure actually looks like</h2>"
        "<p>One example per failure type, model, and tier, taken from the log "
        "with the output that caused it.</p>" + "".join(blocks)
    )


def headline_cards(cells: list[CellResult]) -> str:
    """The four numbers a reader should leave with."""
    run = [c for c in cells if not c.skipped_reason and c.total]
    if not run:
        return ""
    total = sum(c.total for c in run)
    first = Rate(sum(c.first_attempt_ok for c in run), total)
    final = Rate(sum(c.final_ok for c in run), total)
    scores = [s for c in run for s in c.scores]
    accurate = Rate(sum(1 for s in scores if s.accurate), len(scores))
    skipped = sum(1 for c in cells if c.skipped_reason)

    def card(value: str, label: str) -> str:
        return f'<div class="card"><div class="value">{value}</div>' \
               f'<div class="label">{esc(label)}</div></div>'

    return (
        '<div class="cards">'
        + card(f"{first.value:.0%}", f"valid first try (n={total})")
        + card(f"{final.value:.0%}", "valid after repair")
        + card(
            f"{accurate.value:.0%}" if scores else "-",
            f"also correct (n={len(scores)})",
        )
        + card(str(skipped), "cells the provider could not run")
        + "</div>"
    )


def build_report(
    result_files: Iterable[Path | str],
    out: Path | str = "report.html",
    log: AttemptLog | None = None,
) -> Path:
    """Render one HTML file from saved benchmark runs and the attempt log.

    Args:
        result_files: Saved runs to include. Several may be given so a second
            provider's slice appears alongside the headline matrix.
        out: Destination path.
        log: Attempt log, used for the raw failure samples. Optional: without it
            the report loses that section rather than failing to build.

    Returns:
        The path written.
    """
    cells: list[CellResult] = []
    sampling: dict[str, Any] = {}
    sources: list[str] = []

    for file in result_files:
        payload = load_results(file)
        cells.extend(payload["cells"])
        sampling.update(payload.get("sampling", {}))
        sources.append(f"{Path(file).name} ({payload.get('written_at', '')[:19]})")

    parts = [
        "<h1>Structured output, measured</h1>",
        "<p class='sub'>What enforcement actually guarantees across providers, "
        "models, and schema difficulty — and what it costs when it fails.</p>",
        headline_cards(cells),
        capability_section(load_capabilities()),
        tier_section(cells),
        matrix_section(cells),
        failure_section(cells),
        repair_section(cells),
        trajectory_section(cells),
        sample_failures(log) if log is not None else "",
    ]

    settings_note = ", ".join(f"{k}={v}" for k, v in sorted(sampling.items()))
    footer = (
        "<footer>"
        f"<p>Built {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
        f"from {esc(', '.join(sources))}.</p>"
        f"<p>Sampling: {esc(settings_note) or 'not recorded'}. Repeats are "
        "samples, not reproductions: providers differ in what sampling settings "
        "they honour, and batching makes identical output unlikely regardless.</p>"
        "</footer>"
    )

    document = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Structured output, measured</title>"
        f"<style>{CSS}</style></head><body><main>"
        + "".join(p for p in parts if p)
        + footer
        + "</main></body></html>"
    )

    path = Path(out)
    path.write_text(document, encoding="utf-8")
    return path


def build_from_log(log: AttemptLog, out: Path | str = "report.html") -> Path:
    """Build a report from the log alone, with no benchmark run.

    Useful after ad-hoc extractions, where there is a log but no matrix.
    """
    stats = Metrics(log)
    summary = {
        "first_attempt": str(stats.first_attempt_success()),
        "final": str(stats.final_success()),
        "errors": stats.error_breakdown(),
    }
    parts = [
        "<h1>Structured output, measured</h1>",
        "<p class='sub'>Built from the attempt log; no benchmark run was "
        "supplied.</p>",
        f"<pre>{esc(json.dumps(summary, indent=2))}</pre>",
        sample_failures(log),
    ]
    path = Path(out)
    path.write_text(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Structured output, measured</title>"
        f"<style>{CSS}</style></head><body><main>{''.join(parts)}</main>"
        "</body></html>",
        encoding="utf-8",
    )
    return path
