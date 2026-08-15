"""Confirms the translator produces schemas the provider accepts.

The point of the translation layer is that Pydantic's natural output is rejected
by strict modes. That claim is only worth making if the translated form is then
accepted, so this sends both and reports which survived.

Usage::

    python examples/check_contracts.py [--provider groq] [--model MODEL]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openai  # noqa: E402

from contracts.schemas import CONTRACTS  # noqa: E402
from contracts.translate import SchemaTranslator, response_format_for  # noqa: E402
from enforce.ladder import build_plan  # noqa: E402
from provider.capabilities import _short_error, load_capabilities  # noqa: E402
from provider.client import LLMClient  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TICKET = (
    "Subject: Charged twice for October\n\n"
    "Hi, I'm Marta Silva on the Business plan. I was billed twice on 3 October "
    "and the second charge hasn't been refunded. The billing page also errors "
    "with a 500 when I open invoices. This is the third time this year. "
    "Please have Daniel look at it, he handled it last time."
)


def send(client, model, contract, plan, name):
    """Send one request under a plan and report what came back."""
    messages = [{"role": "user", "content": f"{TICKET}\n\nTriage this ticket."}]
    if plan.system_suffix:
        messages.insert(0, {"role": "system", "content": plan.system_suffix})
    result = client.chat(
        messages=messages, model=model, max_tokens=2000, **plan.request_kwargs
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    args = parser.parse_args()

    client = LLMClient(args.provider)
    caps = load_capabilities().get(args.provider, {}).get(args.model)
    translator = SchemaTranslator()

    print(f"provider {args.provider}, model {args.model}")
    print(f"capabilities: {'measured' if caps else 'NOT PROBED — will use prompt_only'}\n")

    for name, contract in CONTRACTS.items():
        translation = translator.translate(contract)
        print(f"{name}  ({contract.difficulty().value})")
        for line in translation.report().splitlines():
            print(f"  {line}")

        # The untranslated schema first: this is what a naive integration sends,
        # and the probe measured it being rejected.
        if not translation.is_lossless or translation.made_nullable:
            raw = contract.model_json_schema()
            try:
                client.chat(
                    messages=[{"role": "user", "content": "test"}],
                    model=args.model,
                    response_format=response_format_for(name, raw),
                    max_tokens=64,
                )
                print("  raw pydantic schema:        accepted")
            except openai.BadRequestError as exc:
                print(f"  raw pydantic schema:        REJECTED — {_short_error(exc)[:95]}")

        plan = build_plan(name, translation, caps)
        try:
            result = send(client, args.model, contract, plan, name)
        except openai.BadRequestError as exc:
            reason = str(exc).split("'message': '")[-1].split("'")[0][:90]
            print(f"  translated schema:          REJECTED — {reason}\n")
            continue

        print(f"  translated schema:          accepted (tier: {plan.tier})")

        try:
            parsed = contract.model_validate_json(result.text)
        except Exception as exc:  # noqa: BLE001 - reporting, not handling
            print(f"  pydantic validation:        FAILED — {type(exc).__name__}")
            print(f"    {str(exc).splitlines()[0][:110]}")
            print(f"    raw: {result.text[:110]}\n")
            continue

        print("  pydantic validation:        passed")
        if hasattr(parsed, "confidence"):
            print(f"    confidence={parsed.confidence} assignee={parsed.assignee!r}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
