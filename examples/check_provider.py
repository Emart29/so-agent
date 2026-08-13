"""Confirms a provider is reachable and reports what came back.

Run this before anything else. It answers the two questions that otherwise get
diagnosed much later and much less clearly: are the credentials working, and does
the normalised result carry the metadata the rest of the project depends on.

Usage::

    python examples/check_provider.py [--provider groq] [--model MODEL]
"""

from __future__ import annotations

import argparse
import sys

from provider.client import BudgetExhaustedError, LLMClient, MissingCredentialsError
from provider.registry import PROVIDERS, get_provider


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--model", default=None, help="defaults to the first listed")
    args = parser.parse_args()

    print("configured providers")
    for provider in PROVIDERS.values():
        key = "key set" if provider.has_key() else "no key"
        budget = f"{provider.daily_budget}/day" if provider.is_metered else "unmetered"
        default = "" if provider.is_default_eligible else "  (explicit only)"
        print(f"  {provider.name:<12} {key:<8} {budget:<10}{default}")

    target = get_provider(args.provider)
    print(f"\nchecking {target.name} at {target.base_url}")

    try:
        client = LLMClient(target.name)
    except MissingCredentialsError as exc:
        print(f"\n  {exc}")
        return 1

    models = client.list_models()
    if models:
        print(f"  {len(models)} models visible to this key")
        for model_id in models[:12]:
            print(f"    {model_id}")
        if len(models) > 12:
            print(f"    ... and {len(models) - 12} more")
    else:
        print("  this provider does not list models; pass --model explicitly")

    model = args.model or (models[0] if models else None)
    if model is None:
        print("\n  no model to call. Re-run with --model.")
        return 1

    print(f"\ncalling {model}")
    try:
        result = client.chat(
            messages=[{"role": "user", "content": "Reply with exactly: ok"}],
            model=model,
            max_tokens=16,
        )
    except BudgetExhaustedError as exc:
        print(f"\n  {exc}")
        return 1

    print(f"  text            {result.text.strip()!r}")
    print(f"  provider        {result.provider}")
    print(f"  model returned  {result.model}")
    print(f"  finish_reason   {result.finish_reason}")
    print(f"  tokens          {result.prompt_tokens} in / {result.completion_tokens} out")
    print(f"  latency         {result.latency_ms:.0f} ms")
    print(f"  attempts        {result.attempts}")
    print(f"  truncated       {result.truncated}")

    remaining = client.remaining_budget()
    if remaining is not None:
        print(f"  budget left     {remaining} of {target.daily_budget} today")

    missing = [
        name
        for name, value in (
            ("text", result.text),
            ("model", result.model),
            ("finish_reason", result.finish_reason),
        )
        if not value
    ]
    if missing:
        print(f"\n  note: provider omitted {', '.join(missing)}")

    print("\nprovider reachable, metadata normalised")
    return 0


if __name__ == "__main__":
    sys.exit(main())
