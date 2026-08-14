"""Measures what a provider's models actually enforce, and caches the result.

Everything downstream reads the matrix this produces rather than a hardcoded
list, so this runs before the contracts, the ladder, or the benchmark mean
anything.

Usage::

    python examples/run_probe.py --provider groq
    python examples/run_probe.py --provider openrouter --models openai/gpt-oss-20b:free
    python examples/run_probe.py --provider groq --refresh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from provider.capabilities import (  # noqa: E402
    CAPABILITIES_PATH,
    CapabilityProber,
    ModelCapabilities,
    TierSupport,
    chat_models,
    is_stale,
    load_capabilities,
    save_capabilities,
)
from provider.client import BudgetExhaustedError, LLMClient  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TIERS = ("json_schema", "json_object", "tools", "prompt_only")

MARKS = {
    TierSupport.CONFORMED.value: "yes",
    TierSupport.IGNORED.value: "IGNORED",
    # The tier applied and the model could not satisfy it. Distinct from "no",
    # which means the tier is unavailable.
    TierSupport.GEN_FAILED.value: "gen-fail",
    TierSupport.REJECTED.value: "no",
    TierSupport.ERROR.value: "err",
    TierSupport.UNTESTED.value: "-",
}


def print_matrix(results: list[ModelCapabilities]) -> None:
    """Print the enforcement matrix, widest column first."""
    if not results:
        print("nothing probed")
        return

    width = max(len(c.model) for c in results) + 2
    header = f"{'model':<{width}}" + "".join(f"{t:<14}" for t in TIERS) + "best"
    print(header)
    print("-" * len(header))

    for caps in sorted(results, key=lambda c: c.model):
        row = f"{caps.model:<{width}}"
        for tier in TIERS:
            row += f"{MARKS.get(caps.tiers.get(tier, '-'), '?'):<14}"
        row += caps.best_tier
        print(row)


def print_features(results: list[ModelCapabilities]) -> None:
    """Print schema-feature support for models that enforce natively."""
    enforcing = [c for c in results if c.features]
    if not enforcing:
        return

    features = sorted({f for c in enforcing for f in c.features})
    width = max(len(c.model) for c in enforcing) + 2

    print("\nschema features, where native enforcement holds")
    header = f"{'model':<{width}}" + "".join(f"{f[:13]:<15}" for f in features) + "depth"
    print(header)
    print("-" * len(header))

    for caps in sorted(enforcing, key=lambda c: c.model):
        row = f"{caps.model:<{width}}"
        for feature in features:
            row += f"{MARKS.get(caps.features.get(feature, '-'), '?'):<15}"
        row += str(caps.max_nesting_depth if caps.max_nesting_depth is not None else "-")
        print(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--models", nargs="*", help="limit to these model ids")
    parser.add_argument("--refresh", action="store_true", help="re-probe cached models")
    parser.add_argument("--no-features", action="store_true", help="tiers only")
    parser.add_argument("--limit", type=int, default=None, help="cap models probed")
    args = parser.parse_args()

    client = LLMClient(args.provider)
    prober = CapabilityProber(client)
    cache = load_capabilities()
    cached_for_provider = cache.get(args.provider, {})

    if args.models:
        targets = list(args.models)
        discovered = False
    else:
        listed = client.list_models()
        discovered = bool(listed)
        targets = chat_models(listed)
        if args.limit:
            targets = targets[: args.limit]

    if not targets:
        print(f"no models to probe on {args.provider}. Pass --models explicitly.")
        return 1

    source = "discovered from the provider" if discovered else "supplied explicitly"
    print(f"probing {len(targets)} models on {args.provider} ({source})")

    remaining = client.remaining_budget()
    if remaining is not None:
        print(f"budget: {remaining} requests left today")
    print()

    results: list[ModelCapabilities] = []
    for model in targets:
        cached = cached_for_provider.get(model)
        if cached and not args.refresh and not is_stale(cached):
            print(f"  {model}  (cached {cached.probed_at[:10]})")
            results.append(cached)
            continue

        print(f"  {model}", end="", flush=True)
        try:
            caps = prober.probe_model(model, features=not args.no_features)
        except BudgetExhaustedError as exc:
            print(f"\n\n{exc}")
            print("stopping; partial results below are saved")
            break
        results.append(caps)
        cached_for_provider[model] = caps
        print(f"  -> {caps.best_tier}")

    cache[args.provider] = cached_for_provider
    save_capabilities(cache)

    print(f"\nsaved to {CAPABILITIES_PATH}\n")
    print_matrix(results)
    print_features(results)

    enforcing = [c for c in results if c.enforces_schema]
    ignoring = [c for c in results if c.silently_ignores]

    print(f"\n{len(enforcing)} of {len(results)} models enforce a native JSON schema")

    if ignoring:
        print("\naccepted an enforcement directive without honouring it:")
        for caps in ignoring:
            print(f"  {caps.model}: {', '.join(caps.silently_ignores)}")
        print(
            "  These are the dangerous case — the request succeeds and nothing\n"
            "  is enforced, which is indistinguishable from working code until\n"
            "  the output is checked."
        )

    remaining = client.remaining_budget()
    if remaining is not None:
        print(f"\nbudget left: {remaining}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
