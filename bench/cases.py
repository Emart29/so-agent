"""The fixed task set the benchmark runs, with hand-labelled expectations.

Every case carries what a correct extraction would contain. Hand-labelling is
tedious and it is the only thing that makes semantic accuracy a measurement
rather than one model's opinion of another — scoring against the critic alone
would inherit the critic's error rate into every number derived from it.

The tickets are deliberately varied in the ways that matter to the contracts:
some name a person and a plan, some name neither; some are plainly urgent and
some only sound urgent; a few are terse enough that any confident triage is
overreach. A set where every case is easy measures nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from contracts.schemas import Category, Priority


@dataclass(frozen=True)
class Case:
    """One benchmark input and what a faithful extraction would say."""

    id: str
    text: str
    #: Categories a reasonable triage could choose. More than one where the
    #: ticket genuinely straddles teams — scoring a defensible answer wrong
    #: would measure the labeller's preference rather than the model.
    categories: tuple[Category, ...]
    #: Priorities that would be defensible given the content.
    priorities: tuple[Priority, ...]
    #: Facts stated in the text. An extraction inventing a value for any field
    #: not represented here is ungrounded.
    stated_facts: dict[str, str] = field(default_factory=dict)
    #: Details the text does not give. An extraction supplying one has invented
    #: it, which is the failure that matters most.
    absent_facts: tuple[str, ...] = ()


CASES: list[Case] = [
    Case(
        id="billing_duplicate",
        text=(
            "I'm Marta Silva on the Business plan. I was billed twice on 3 October "
            "and the second charge hasn't been refunded. Please have Daniel look "
            "at it, he handled this last time."
        ),
        categories=(Category.BILLING,),
        priorities=(Priority.MEDIUM, Priority.HIGH),
        stated_facts={
            "name": "Marta Silva",
            "account_tier": "Business",
            "assignee": "Daniel",
        },
    ),
    Case(
        id="export_broken",
        text=(
            "The export button does nothing on Firefox. I'm on the Pro plan and "
            "this blocks our month-end reporting."
        ),
        categories=(Category.TECHNICAL,),
        priorities=(Priority.MEDIUM, Priority.HIGH),
        stated_facts={"account_tier": "Pro"},
        absent_facts=("name", "assignee"),
    ),
    Case(
        id="cancel_account",
        text="Please cancel my account. I've found a cheaper alternative.",
        categories=(Category.ACCOUNT,),
        priorities=(Priority.LOW, Priority.MEDIUM),
        absent_facts=("name", "account_tier", "assignee"),
    ),
    Case(
        id="dark_mode",
        text="Could you add dark mode? Reading the dashboard at night is painful.",
        categories=(Category.FEATURE_REQUEST,),
        priorities=(Priority.LOW,),
        absent_facts=("name", "account_tier", "assignee"),
    ),
    Case(
        id="total_outage",
        text=(
            "Login has failed for two days now. We cannot access anything and it "
            "is costing us real money. This is unacceptable."
        ),
        categories=(Category.TECHNICAL,),
        priorities=(Priority.HIGH, Priority.CRITICAL),
        absent_facts=("name", "account_tier", "assignee"),
    ),
    Case(
        id="vat_rate",
        text=(
            "The invoice shows 19% VAT but we're registered in Ireland, so it "
            "should be 23%. Can you reissue it?"
        ),
        categories=(Category.BILLING,),
        priorities=(Priority.LOW, Priority.MEDIUM),
        absent_facts=("name", "account_tier", "assignee"),
    ),
    Case(
        id="api_500s",
        text=(
            "The API returns 500 on every POST since yesterday's deploy. Our "
            "integration is down. I'm Tom Reyes, Enterprise."
        ),
        categories=(Category.TECHNICAL,),
        priorities=(Priority.HIGH, Priority.CRITICAL),
        stated_facts={"name": "Tom Reyes", "account_tier": "Enterprise"},
        absent_facts=("assignee",),
    ),
    Case(
        id="add_seat",
        text="How do I add a second seat to my subscription?",
        categories=(Category.ACCOUNT, Category.BILLING),
        priorities=(Priority.LOW,),
        absent_facts=("name", "account_tier", "assignee"),
    ),
    Case(
        # Terse to the point that any confident triage is overreach. Included
        # because a set of clear cases cannot show a model over-committing.
        id="vague",
        text="it broke",
        categories=(Category.TECHNICAL, Category.OTHER),
        priorities=(Priority.LOW, Priority.MEDIUM, Priority.HIGH),
        absent_facts=("name", "account_tier", "assignee"),
    ),
    Case(
        id="angry_refund",
        text=(
            "This is the third time I've written. I want a refund and I want it "
            "today. Nobody has replied to any of my emails."
        ),
        categories=(Category.BILLING, Category.OTHER),
        priorities=(Priority.HIGH, Priority.CRITICAL),
        absent_facts=("name", "account_tier", "assignee"),
    ),
]

CASES_BY_ID = {case.id: case for case in CASES}
