"""Scores the critic against hand-labelled cases before trusting its output.

The critic produces a semantic failure rate, and that rate inherits the critic's
own error rate. Publishing it without this number would be publishing an
unvalidated model's opinion as a measurement.

Run this before quoting any semantic figure, and put the result beside it.

Usage::

    python examples/check_critic.py [--critic MODEL]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from critic.labelled import LABELLED_CASES  # noqa: E402
from critic.semantic import SemanticCritic  # noqa: E402
from provider.client import LLMClient  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="groq")
    # Read from configuration rather than hardcoded: the model this defaulted
    # to was retired by its provider mid-project, and a default that names a
    # dead model fails with a 404 that explains nothing.
    parser.add_argument("--critic", default=settings.CRITIC_MODEL)
    args = parser.parse_args()

    client = LLMClient(args.provider)
    critic = SemanticCritic(client, args.critic)

    sound_cases = sum(1 for _, _, s in LABELLED_CASES if s)
    print(f"critic  {args.provider}/{args.critic}  (tier: {critic._plan.tier})")
    print(
        f"cases   {len(LABELLED_CASES)} labelled "
        f"({sound_cases} sound, {len(LABELLED_CASES) - sound_cases} unsound)\n"
    )

    agreed = false_pass = false_fail = unavailable = 0

    for i, (source, extraction, truly_sound) in enumerate(LABELLED_CASES, start=1):
        result = critic.review(source, extraction, type(extraction))
        label = "sound" if truly_sound else "unsound"

        if not result.checked:
            unavailable += 1
            print(f"  {i}. label={label:8} critic=unavailable  ({result.reason[:60]})")
            continue

        said = "sound" if result.passed else "unsound"
        if result.passed == truly_sound:
            agreed += 1
            mark = "agree"
        elif result.passed:
            false_pass += 1
            mark = "FALSE PASS"
        else:
            false_fail += 1
            mark = "FALSE FAIL"

        print(f"  {i}. label={label:8} critic={said:8} {mark}")
        if mark != "agree" and result.reason:
            print(f"       {result.reason[:100]}")

    judged = len(LABELLED_CASES) - unavailable
    agreement = agreed / judged if judged else 0.0

    print(f"\nagreement       {agreement:.0%} over {judged} judged cases")
    print(f"false passes    {false_pass}  (approved a bad extraction)")
    print(f"false failures  {false_fail}  (rejected a good one)")
    if unavailable:
        print(f"unavailable     {unavailable}")

    print()
    if judged == 0:
        print("The critic never produced a verdict. Its output cannot be used.")
    elif agreement >= 0.8:
        print(
            "Agreement is high enough that the critic's failure rate is worth\n"
            "reporting, with this number stated alongside it."
        )
    else:
        print(
            "Agreement is too low to quote the critic's failure rate as a finding.\n"
            "Either use a stronger critic model or treat its output as a signal\n"
            "for review rather than a measurement."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
