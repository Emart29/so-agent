"""The data contracts, as Pydantic models.

One domain throughout — support-ticket triage — because comparing failure rates
across schemas only means something if the task behind them is held roughly
constant. The models climb in structural complexity so the benchmark can ask
whether complexity predicts failure.

Field descriptions are not documentation. They are serialised into the schema
and shipped to the model on every request, so they are written for the model:
what the field is, what shape the value takes, how to decide. A description
that reads well to a human and vaguely to a model is a bug in the contract.

Two fields here are deliberate landmines, both chosen because the capability
probe measured them failing:

* ``confidence`` carries numeric bounds, which strict schema modes strip. The
  bound still has to hold, so something must enforce it after parsing.
* ``assignee`` is optional in the ordinary Python sense, which Pydantic emits by
  leaving it out of ``required`` — the exact form Groq rejects outright.

A contract that avoided both would be easier to satisfy and would test nothing.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Difficulty(str, Enum):
    """How structurally demanding a contract is, for benchmark grouping."""

    SIMPLE = "simple"
    NESTED = "nested"
    HARD = "hard"


class Priority(str, Enum):
    """How urgently a ticket needs attention."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Category(str, Enum):
    """Which team a ticket belongs to."""

    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    FEATURE_REQUEST = "feature_request"
    OTHER = "other"


class Sentiment(str, Enum):
    """The customer's tone, as expressed in the ticket text."""

    ANGRY = "angry"
    FRUSTRATED = "frustrated"
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"


class Contract(BaseModel):
    """Base for every contract, carrying its benchmark difficulty label."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def difficulty(cls) -> Difficulty:
        raise NotImplementedError


class TicketSummary(Contract):
    """Flat scalars only — the easiest thing a model can be asked for."""

    category: Category = Field(
        description="The single team this ticket should be routed to."
    )
    priority: Priority = Field(
        description="Urgency, judged from impact and tone rather than politeness."
    )
    summary: str = Field(
        description="One sentence, under 20 words, stating the customer's problem."
    )

    @classmethod
    def difficulty(cls) -> Difficulty:
        return Difficulty.SIMPLE


class Customer(BaseModel):
    """Who reported a ticket."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Customer's name exactly as it appears in the ticket.")
    account_tier: str = Field(
        description='Plan named in the ticket, or "unknown" if none is mentioned.'
    )


class Issue(BaseModel):
    """One distinct problem raised within a ticket."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(description="The problem, in one sentence.")
    category: Category = Field(description="Team responsible for this specific issue.")


class TicketAnalysis(Contract):
    """A nested object and a list of objects — one structural step up."""

    customer: Customer = Field(description="Who reported the ticket.")
    issues: list[Issue] = Field(
        description="Every distinct problem raised. One entry per problem."
    )
    priority: Priority = Field(description="Overall urgency across all issues.")

    @classmethod
    def difficulty(cls) -> Difficulty:
        return Difficulty.NESTED


class ResolutionStep(BaseModel):
    """One action an agent should take."""

    model_config = ConfigDict(extra="forbid")

    action: str = Field(description="What to do, phrased as an instruction.")
    requires_customer_reply: bool = Field(
        description="True when this step cannot proceed without the customer."
    )


class TicketTriage(Contract):
    """Everything at once: nesting, lists, enums, bounds, and an optional field.

    Both of the fields the capability probe found problematic appear here on
    purpose. ``confidence`` carries bounds that strict modes strip, and
    ``assignee`` is the optional form Groq rejects. This is the contract that
    exercises the translator rather than the model.
    """

    customer: Customer = Field(description="Who reported the ticket.")
    issues: list[Issue] = Field(description="Every distinct problem raised.")
    priority: Priority = Field(description="Overall urgency.")
    sentiment: Sentiment = Field(description="The customer's tone.")
    steps: list[ResolutionStep] = Field(
        description="Ordered actions that would resolve this ticket."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="How certain this triage is, from 0.0 to 1.0 inclusive.",
    )
    assignee: str | None = Field(
        default=None,
        description=(
            "Name of a specific person the ticket asks for, or null if none is named."
        ),
    )

    @classmethod
    def difficulty(cls) -> Difficulty:
        return Difficulty.HARD


#: Every contract the benchmark runs, by name.
CONTRACTS: dict[str, type[Contract]] = {
    "ticket_summary": TicketSummary,
    "ticket_analysis": TicketAnalysis,
    "ticket_triage": TicketTriage,
}


def get_contract(name: str) -> type[Contract]:
    """Look up a contract by name, naming the alternatives when it is missing."""
    try:
        return CONTRACTS[name]
    except KeyError:
        known = ", ".join(sorted(CONTRACTS))
        raise KeyError(f"unknown contract {name!r}. Known contracts: {known}") from None


def contracts_by_difficulty(difficulty: Difficulty) -> list[type[Contract]]:
    """Return every contract at one difficulty level."""
    return [c for c in CONTRACTS.values() if c.difficulty() is difficulty]
