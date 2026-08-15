"""Runs the repair loop against a live model and reports what actually failed.

The scripted tests prove the loop branches correctly. They cannot show whether
the failures it handles are the ones real models produce, so this samples a
weak model on a weak tier — where failures are common — and reports the
breakdown.

The number worth watching is how much of the raw failure rate the free local
extraction removes. If most failures are markdown fences, a loop that pays for
a retry to fix them is measuring its own impatience rather than the model.

Usage::

    python examples/check_repair.py [--model M] [--tier T] [--n 8]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.schemas import TicketSummary  # noqa: E402
from contracts.translate import SchemaTranslator  # noqa: E402
from enforce.ladder import build_plan  # noqa: E402
from enforce.repair import repair_loop  # noqa: E402
from provider.capabilities import load_capabilities  # noqa: E402
from provider.client import LLMClient  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TICKETS = [
    "I was charged twice for October and nobody has replied to my emails.",
    "The export button does nothing on Firefox. I'm on the Pro plan.",
    "Please cancel my account, I've found something better.",
    "Can you add dark mode? It's hard to read at night.",
    "Login has failed for two days. This is costing us money.",
    "Invoice shows the wrong VAT rate for Germany.",
    "The API returns 500 on every POST since yesterday's deploy.",
    "How do I add a second seat to my subscription?",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--model", default="llama-3.1-8b-instant")
    parser.add_argument("--tier", default="prompt_only")
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()

    client = LLMClient(args.provider)
    caps = load_capabilities().get(args.provider, {}).get(args.model)
    translation = SchemaTranslator().translate(TicketSummary)
    plan = build_plan("ticket_summary", translation, caps, requested_tier=args.tier)

    print(f"{args.model} on {args.provider}, tier: {plan.tier}")
    print(f"guarantee: {plan.guarantee}\n")

    first_failures: Counter[str] = Counter()
    final_ok = 0
    first_ok = 0
    free_repairs = 0
    retries_used = 0

    for i, ticket in enumerate(TICKETS[: args.n], start=1):
        messages = [{"role": "user", "content": f"{ticket}\n\nTriage this ticket."}]
        if plan.system_suffix:
            messages.insert(0, {"role": "system", "content": plan.system_suffix})

        def generate(repair_message, max_tokens, _messages=messages):
            turn = list(_messages)
            if repair_message:
                turn.append({"role": "user", "content": repair_message})
            return client.chat(
                messages=turn,
                model=args.model,
                max_tokens=max_tokens or 1200,
                **plan.request_kwargs,
            )

        outcome = repair_loop(
            generate, TicketSummary, tier=plan.tier, max_attempts=3, max_tokens=1200
        )

        first = outcome.attempts[0]
        first_failures[first.failure.value] += 1
        if outcome.first_attempt_ok:
            first_ok += 1
        if outcome.needed_only_extraction:
            free_repairs += 1
        if outcome.attempt_count > 1:
            retries_used += outcome.attempt_count - 1
        if outcome.ok:
            final_ok += 1

        status = "ok" if outcome.ok else "FAILED"
        print(
            f"  {i}. {status:6} attempts={outcome.attempt_count} "
            f"{' -> '.join(outcome.error_sequence)}"
        )
        if not outcome.ok:
            print(f"      stopped: {outcome.stopped_because}")

    n = min(args.n, len(TICKETS))
    print(f"\nfirst attempt clean       {first_ok}/{n}")
    print(f"fixed by local extraction {free_repairs}/{n}  (no retry, no cost)")
    print(f"valid after repair        {final_ok}/{n}")
    print(f"retries spent             {retries_used}")

    print("\nwhat the first attempt produced:")
    for failure, count in first_failures.most_common():
        print(f"  {failure:18} {count}")

    return 0 if final_ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
