"""Letting the agent choose a response shape, not just fill one in.

An agent that can only fill a fixed form has to pretend it knows the answer.
Choosing between shapes is what lets it say "the request is ambiguous" or "I
need to look something up" instead of inventing a confident triage from a
one-line ticket.

Pydantic expresses this as a discriminated union: a literal tag field selects
which variant applies. The capability probe measured `anyOf` accepted by every
model that enforces natively, so the union can usually be sent whole — but not
always, and the two-call fallback below exists for the models where it cannot.

The failure worth testing for is a well-formed variant that contradicts its own
tag: `{"kind": "answer", "question": "..."}`. It parses, it looks structured,
and it means the opposite of what it says. Pydantic catches it because the tag
selects the variant before the fields are validated.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from contracts.schemas import Category, Priority


class Answer(BaseModel):
    """The request was clear enough to act on."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["answer"] = "answer"
    category: Category = Field(description="Team this ticket should be routed to.")
    priority: Priority = Field(description="Urgency, judged from impact and tone.")
    summary: str = Field(description="One sentence stating the customer's problem.")


class Clarification(BaseModel):
    """The request was ambiguous, and guessing would be worse than asking."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["clarification"] = "clarification"
    question: str = Field(
        description="The single question that would resolve the ambiguity."
    )
    reason: str = Field(description="Why the ticket cannot be triaged as written.")


class Lookup(BaseModel):
    """Answering needs data the ticket does not contain."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["lookup"] = "lookup"
    resource: str = Field(
        description='What to fetch, such as "billing_history" or "account_status".'
    )
    identifier: str = Field(description="Which record to fetch it for.")


#: The tag field selects the variant before its fields are validated, so a
#: payload that contradicts its own tag fails rather than being coerced.
TriageDecision = Annotated[
    Union[Answer, Clarification, Lookup],
    Field(discriminator="kind"),
]

DECISION_ADAPTER: TypeAdapter[TriageDecision] = TypeAdapter(TriageDecision)

#: Every variant, by tag. Used by the two-call fallback to pick the schema for
#: the second request once the tag is known.
VARIANTS: dict[str, type[BaseModel]] = {
    "answer": Answer,
    "clarification": Clarification,
    "lookup": Lookup,
}


class DecisionEnvelope(BaseModel):
    """Wraps the union in an object, since a bare union is not a JSON object.

    Providers expect an object at the top level of a schema, so the union
    travels one level down.
    """

    model_config = ConfigDict(extra="forbid")

    decision: TriageDecision = Field(
        description=(
            "Choose one: answer when the ticket can be triaged as written, "
            "clarification when it is ambiguous, lookup when external data is needed."
        )
    )


class VariantChoice(BaseModel):
    """First half of the two-call fallback: which shape applies.

    Used only where a provider cannot enforce `anyOf`. It costs an extra round
    trip, which is a real cost and is reported rather than hidden.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["answer", "clarification", "lookup"] = Field(
        description=(
            "answer if the ticket can be triaged as written; clarification if it "
            "is ambiguous; lookup if external data is required."
        )
    )


def variant_for(kind: str) -> type[BaseModel]:
    """Return the model for a tag, naming the alternatives when unknown."""
    try:
        return VARIANTS[kind]
    except KeyError:
        known = ", ".join(sorted(VARIANTS))
        raise KeyError(f"unknown variant {kind!r}. Known variants: {known}") from None


def parse_decision(data: dict) -> TriageDecision:
    """Validate a decision payload against the discriminated union.

    Raises:
        pydantic.ValidationError: When the tag is missing or unknown, or when
            the payload does not match the variant its tag selects.
    """
    return DECISION_ADAPTER.validate_python(data)
