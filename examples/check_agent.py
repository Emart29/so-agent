"""Runs the assembled agent end to end and reports what the log then says.

This is the first point where all the layers run together against a live
provider: enforcement selection, repair, semantic review, logging, and the
metrics computed from what was logged.

The number to watch is the gap between structural success and semantic
soundness. If they diverge, "valid" and "correct" are measuring different
things — which is the argument the critic exists to make concrete.

Usage::

    python examples/check_agent.py [--model M] [--critic M] [--tier T]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import StructuredAgent  # noqa: E402
from contracts.schemas import TicketSummary, TicketTriage  # noqa: E402
from store.log import AttemptLog  # noqa: E402
from store.metrics import Metrics, trajectory_reliability  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TICKETS = [
    "I'm Marta Silva on the Business plan. I was billed twice on 3 October and "
    "the second charge hasn't been refunded. Please have Daniel look at it.",
    "The export button does nothing on Firefox. I'm on the Pro plan and this "
    "blocks our month-end reporting.",
    "Please cancel my account. I've found a cheaper alternative.",
    "Could you add dark mode? Reading the dashboard at night is painful.",
    "Login has failed for two days now. We cannot access anything and it is "
    "costing us real money. This is unacceptable.",
    "The invoice shows 19% VAT but we're registered in Ireland, so it should "
    "be 23%. Can you reissue it?",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--critic", default="llama-3.1-8b-instant")
    parser.add_argument("--tier", default=None)
    parser.add_argument("--db", default="check_agent.db")
    args = parser.parse_args()

    db_path = Path(args.db)
    if db_path.exists():
        db_path.unlink()

    store = AttemptLog(db_path)
    agent = StructuredAgent(
        provider=args.provider,
        model=args.model,
        critic_model=args.critic,
        store=store,
    )

    print(f"agent   {args.provider}/{args.model}")
    print(f"critic  {args.critic}")
    print(f"probed  {agent.capabilities is not None}\n")

    for contract in (TicketSummary, TicketTriage):
        print(f"{contract.__name__} ({contract.difficulty().value})")
        for i, ticket in enumerate(TICKETS, start=1):
            result = agent.extract(ticket, contract, tier=args.tier)
            sound = result.semantically_sound
            mark = {True: "sound", False: "UNSOUND", None: "unchecked"}[sound]
            print(f"  {i}. {result.summary()}  critic={mark}")
            if sound is False and result.critic:
                print(f"       {result.critic.reason[:100]}")
        print()

    metrics = Metrics(store)
    print("measured from the log")
    print(f"  first attempt clean   {metrics.first_attempt_success()}")
    print(f"  after repair          {metrics.final_success()}")
    print(f"  fixed for free        {metrics.free_repair_share()}")

    breakdown = metrics.error_breakdown()
    if breakdown:
        print(f"  first-attempt errors  {breakdown}")

    by_difficulty = metrics.by_schema_difficulty()
    if by_difficulty:
        print("\n  by schema difficulty")
        for level, rate in sorted(by_difficulty.items()):
            print(f"    {level:10} {rate}")

    semantic = metrics.semantic_failure_rate()
    if semantic.total:
        print(f"\n  valid but semantically wrong  {semantic}")
        print("  (structural validity and correctness are different measurements)")

    by_tier = metrics.accuracy_by_tier()
    if len(by_tier) > 1:
        print("\n  semantic soundness by tier")
        for tier, rate in by_tier.items():
            print(f"    {tier:14} {rate}")

    rate = metrics.first_attempt_success().value
    print(f"\n  projected across a chain at {rate:.0%} per call")
    for steps, survival in trajectory_reliability(rate).items():
        print(f"    {steps:>3} steps   {survival:.1%}")
    print("  (independent-failure upper bound; real failures correlate)")

    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
