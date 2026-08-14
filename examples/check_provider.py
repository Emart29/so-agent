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
from pathlib import Path

# Running this as a script puts examples/ on the import path rather than the
# project root, so the packages next to it are invisible. Adding the root keeps
# the documented `python examples/check_provider.py` working from a fresh clone
# without requiring an editable install first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from provider.client import BudgetExhaustedError, LLMClient, MissingCredentialsError  # noqa: E402
from provider.registry import PROVIDERS, get_provider  # noqa: E402

# Model output is arbitrary text in any language, and the Windows console
# defaults to a codepage that cannot represent most of it. Without this, a model
# that replies in Arabic crashes the check that was meant to prove it works.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: Substrings marking models that do not serve chat completions. Used only to
#: pick a sensible default for this smoke test — what each model genuinely
#: supports is established by the capability probe, not by its name.
NON_CHAT_HINTS = ("whisper", "orpheus", "prompt-guard", "tts", "embed", "guard")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--model", default=None, help="defaults to the first listed")
    # Reasoning models spend output tokens on reasoning before emitting any
    # visible text, so a budget sized for the answer alone returns an empty
    # string with finish_reason "length". That is a truncation, not a failure to
    # respond, and it needs headroom rather than a retry.
    parser.add_argument("--max-tokens", type=int, default=512)
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

    model = args.model
    if model is None:
        chat_models = [
            m for m in models
            if not any(hint in m.lower() for hint in NON_CHAT_HINTS)
        ]
        model = chat_models[0] if chat_models else None

    if model is None:
        print("\n  no chat model to call. Re-run with --model.")
        return 1

    picked = "" if args.model else "  (auto-selected; pass --model to choose)"
    print(f"\ncalling {model}{picked}")
    try:
        result = client.chat(
            messages=[{"role": "user", "content": "Reply with exactly: ok"}],
            model=model,
            max_tokens=args.max_tokens,
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

    if result.truncated:
        print(
            f"\n  note: output hit the {args.max_tokens}-token ceiling. On a "
            "reasoning model the budget is spent before any visible text is "
            "emitted; re-run with a larger --max-tokens."
        )

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
        print(f"  note: provider omitted {', '.join(missing)}")

    print("\nprovider reachable, metadata normalised")
    return 0


if __name__ == "__main__":
    sys.exit(main())
