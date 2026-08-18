"""Chooses the model ladder from the probe rather than from a hardcoded list.

Written after a run lost two of its five models mid-benchmark: both Llama models
Groq served on 17 August returned 404 on 18 August. A list of model ids checked
into a repository is wrong the moment a provider changes its line-up, and this
project's whole argument is that capabilities are measured rather than assumed —
which has to apply to which models exist, not only to what they enforce.

Usage::

    python examples/ladder.py --provider groq        # one id per line
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from provider.capabilities import ModelCapabilities, load_capabilities  # noqa: E402

#: Models to leave out of the benchmark, with the reason. Not a blocklist of
#: things that perform badly — only of things the benchmark cannot measure.
EXCLUDED = {
    "groq/compound": "an agentic system rather than a plain chat model",
}


def is_measurable(caps: ModelCapabilities) -> bool:
    """Whether a model can produce data for at least one tier.

    A model that fails even the prompt-only probe has nothing to contribute:
    every cell would be an error, and error cells measure the harness.
    """
    if caps.retired_at:
        return False  # The provider no longer serves it; every cell would 404.
    return any(
        caps.tiers.get(tier) == "conformed"
        for tier in ("json_schema", "json_object", "prompt_only")
    )


def build_ladder(provider: str = "groq") -> list[str]:
    """Return the measurable models for a provider, native enforcement last.

    Ordered so the models that enforce natively come last. If a daily token
    budget runs out mid-run, the cells lost are the ones a reader can most
    easily predict, rather than the ones carrying the finding.
    """
    probed = load_capabilities().get(provider, {})
    usable = [
        (name, caps) for name, caps in sorted(probed.items())
        if name not in EXCLUDED and is_measurable(caps)
    ]
    return [name for name, caps in usable if not caps.enforces_schema] + [
        name for name, caps in usable if caps.enforces_schema
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="groq")
    args = parser.parse_args()

    ladder = build_ladder(args.provider)
    if not ladder:
        print(
            f"no probed models for {args.provider}. Run: so-agent probe "
            f"--provider {args.provider}",
            file=sys.stderr,
        )
        return 1
    print("\n".join(ladder))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
