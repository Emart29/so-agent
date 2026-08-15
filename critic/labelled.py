"""Hand-labelled cases for scoring the critic itself.

The critic produces a semantic failure rate. That rate inherits the critic's own
error rate, so quoting it without knowing how often the critic is right would be
publishing an unvalidated number dressed as a measurement.

Every case here is labelled by hand, and the labels are deliberately obvious in
both directions — clear-cut cases make disagreement attributable to the critic
rather than to a judgement call a reasonable person could go either way on.

The unsound cases are drawn from failures actually observed: invented customer
names, placeholder account tiers, priorities that do not follow from the text.
The sound cases include paraphrased summaries, because an over-literal critic
faulting a summary for not quoting its source was a real failure mode.
"""

from __future__ import annotations

from contracts.schemas import (
    Category,
    Customer,
    Issue,
    Priority,
    Sentiment,
    TicketSummary,
    TicketTriage,
)


def _triage(**overrides) -> TicketTriage:
    base = dict(
        customer=Customer(name="Marta Silva", account_tier="Business"),
        issues=[Issue(description="Charged twice in October.", category=Category.BILLING)],
        priority=Priority.HIGH,
        sentiment=Sentiment.FRUSTRATED,
        steps=[],
        confidence=0.8,
        assignee=None,
    )
    return TicketTriage(**{**base, **overrides})


BILLING_TICKET = (
    "I'm Marta Silva on the Business plan. I was billed twice on 3 October and "
    "the second charge hasn't been refunded."
)

FEATURE_TICKET = "Could you add dark mode? Reading the dashboard at night is painful."

OUTAGE_TICKET = (
    "Login has failed for two days now. We cannot access anything and it is "
    "costing us real money. This is unacceptable."
)

#: ``(source, extraction, is_actually_sound)``. Half sound, half not, so the
#: agreement score is not flattered by an unbalanced set.
LABELLED_CASES: list[tuple[str, object, bool]] = [
    # --- sound -------------------------------------------------------------
    (
        BILLING_TICKET,
        TicketSummary(
            category=Category.BILLING,
            priority=Priority.HIGH,
            summary="Customer was charged twice in October and wants the duplicate refunded.",
        ),
        True,
    ),
    (
        FEATURE_TICKET,
        TicketSummary(
            category=Category.FEATURE_REQUEST,
            priority=Priority.LOW,
            # Deliberately paraphrased. A critic that demands the source's own
            # wording marks this unsound, which is the failure being tested for.
            summary="Customer asks for a dark theme to make night-time reading easier.",
        ),
        True,
    ),
    (
        OUTAGE_TICKET,
        TicketSummary(
            category=Category.TECHNICAL,
            priority=Priority.CRITICAL,
            summary="Login has been failing for two days, blocking all access.",
        ),
        True,
    ),
    (
        BILLING_TICKET,
        _triage(),
        True,
    ),
    # --- unsound -----------------------------------------------------------
    (
        FEATURE_TICKET,
        TicketSummary(
            category=Category.BILLING,  # nothing in the ticket is about billing
            priority=Priority.CRITICAL,  # a theme request is not critical
            summary="Customer reports a billing error requiring urgent attention.",
        ),
        False,
    ),
    (
        BILLING_TICKET,
        # An invented name and tier for details the source never gave.
        _triage(customer=Customer(name="Anonymous", account_tier="unknown")),
        False,
    ),
    (
        FEATURE_TICKET,
        TicketSummary(
            category=Category.FEATURE_REQUEST,
            priority=Priority.LOW,
            summary="summary",  # placeholder rather than content
        ),
        False,
    ),
    (
        OUTAGE_TICKET,
        # Confidence stated as near-certain on a claim the source contradicts.
        _triage(
            issues=[
                Issue(
                    description="Customer is happy with the service.",
                    category=Category.OTHER,
                )
            ],
            sentiment=Sentiment.SATISFIED,
            confidence=0.99,
        ),
        False,
    ),
]
