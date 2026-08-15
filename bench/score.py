"""Scores an extraction against a case's hand-written labels.

Separate from the critic on purpose. The critic is an LLM whose own error rate
is measured but not zero, so a semantic accuracy figure derived only from it
inherits that error. These checks are deterministic: they ask whether the
extraction invented a fact the source never gave, and whether its classification
sits within the set a careful reader would accept.

Two rules keep this from measuring the labeller instead of the model:

* **Classification is scored against a set, not a single answer.** A ticket that
  genuinely straddles two teams has two defensible categories, and marking one
  wrong would record a preference as an error.
* **Invented facts are the failure that counts.** A model that says a customer
  is named "Anonymous" when the ticket named nobody has fabricated a value, and
  that is checkable without judgement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from bench.cases import Case

#: Values models use to fill a field they have no answer for. Treated as
#: invention rather than as an honest blank, because downstream code cannot
#: tell them apart from real data.
PLACEHOLDER_MARKERS = (
    "unknown", "n/a", "na", "none", "not specified", "not provided",
    "unnamed", "anonymous", "unspecified", "customer", "user", "no name",
    "not mentioned", "not available", "-", "",
)


@dataclass
class CaseScore:
    """How faithfully one extraction represented its source."""

    case_id: str
    category_ok: bool
    priority_ok: bool
    invented: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    placeholders: list[str] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        """Whether nothing was invented and no placeholder was passed off as data."""
        return not self.invented and not self.placeholders

    @property
    def accurate(self) -> bool:
        """Whether the extraction is defensible on every checked dimension."""
        return self.category_ok and self.priority_ok and self.grounded

    def problems(self) -> list[str]:
        issues: list[str] = []
        if not self.category_ok:
            issues.append("category outside the acceptable set")
        if not self.priority_ok:
            issues.append("priority outside the acceptable set")
        if self.invented:
            issues.append(f"invented: {', '.join(self.invented)}")
        if self.placeholders:
            issues.append(f"placeholder values: {', '.join(self.placeholders)}")
        if self.missed:
            issues.append(f"missed stated facts: {', '.join(self.missed)}")
        return issues


def is_placeholder(value: Any) -> bool:
    """Whether a value is filler standing in for an absent answer."""
    if value is None:
        return False  # An explicit null is honest; a fake name is not.
    text = str(value).strip().lower()
    return text in PLACEHOLDER_MARKERS


def _as_json_dict(extraction: Any) -> dict[str, Any]:
    """Dump a model to plain JSON types.

    ``mode="json"`` matters: a plain dump returns enum *members*, so an enum
    field stringifies as ``"Category.ACCOUNT"`` rather than ``"account"`` and
    every comparison against the allowed values fails. That would score each
    classification wrong and report semantic accuracy as zero everywhere.
    """
    if hasattr(extraction, "model_dump"):
        return extraction.model_dump(mode="json")
    return dict(extraction)


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _field_values(extraction: Any) -> dict[str, Any]:
    """Flatten the fields this scorer knows how to check.

    Contracts differ in shape — a summary carries no customer — so this reads
    what is present rather than assuming a structure.
    """
    data = _as_json_dict(extraction)
    values: dict[str, Any] = {}

    customer = data.get("customer") or {}
    if isinstance(customer, dict):
        values["name"] = customer.get("name")
        values["account_tier"] = customer.get("account_tier")

    if "assignee" in data:
        values["assignee"] = data.get("assignee")

    return values


def score_case(case: Case, extraction: Any) -> CaseScore:
    """Score one extraction against its case labels.

    Args:
        case: The benchmark case, carrying acceptable answers and known absences.
        extraction: The validated object produced for it.

    Returns:
        A per-dimension score, with the specific problems named.
    """
    data = _as_json_dict(extraction)

    category = data.get("category")
    if category is None:
        # Nested contracts carry categories on the issues rather than the root.
        issues = data.get("issues") or []
        found = {i.get("category") for i in issues if isinstance(i, dict)}
        acceptable = {c.value for c in case.categories}
        category_ok = bool(found) and bool(found & acceptable)
    else:
        category_ok = str(category) in {c.value for c in case.categories}

    priority = data.get("priority")
    priority_ok = (
        priority is not None and str(priority) in {p.value for p in case.priorities}
    )

    values = _field_values(extraction)
    invented: list[str] = []
    placeholders: list[str] = []
    missed: list[str] = []

    for field_name in case.absent_facts:
        value = values.get(field_name)
        if value is None:
            continue  # Correctly left blank.
        if is_placeholder(value):
            # A filler string is worse than a null: downstream code cannot tell
            # it from real data.
            placeholders.append(f"{field_name}={value!r}")
        else:
            invented.append(f"{field_name}={value!r}")

    for field_name, expected in case.stated_facts.items():
        value = values.get(field_name)
        if value is None:
            missed.append(field_name)
            continue
        if _normalise(expected) not in _normalise(value) and _normalise(
            value
        ) not in _normalise(expected):
            invented.append(f"{field_name}={value!r} (source says {expected!r})")

    return CaseScore(
        case_id=case.id,
        category_ok=category_ok,
        priority_ok=priority_ok,
        invented=invented,
        missed=missed,
        placeholders=placeholders,
    )
