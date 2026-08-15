"""Rewrites a Pydantic schema into one a provider will accept.

Pydantic emits valid JSON Schema. Providers accept a subset of it. The gap
between those two is where this project's first real bug lives, and the
capability probe measured exactly where it falls:

* **Optional fields.** Pydantic marks a field optional by leaving it out of
  ``required``. Groq's strict mode rejects that outright — *"`required` is
  required"* — while OpenRouter accepts it, for the same model. Optionality has
  to be re-expressed as a type admitting null with the field still required.
* **Value constraints.** Bounds, lengths, and patterns are dropped by strict
  modes. They are still part of the contract, so dropping them silently is how
  a "validated" pipeline starts accepting a confidence of 40.

The second point is the one that matters most. Every constraint removed here is
returned to the caller, not discarded, so it can be enforced after parsing.
Pydantic still applies all of them on the way back in — the translator's job is
to stop the provider rejecting the request, not to weaken the contract.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

#: Keywords that constrain a *value* rather than its shape. Strict schema modes
#: generally reject these, and none of them can be enforced by the provider
#: anyway — they are checked after parsing instead.
VALUE_CONSTRAINTS = frozenset(
    {
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
        "minLength", "maxLength", "pattern", "format",
        "minItems", "maxItems", "uniqueItems",
        "minProperties", "maxProperties",
    }
)

#: Annotations that carry no meaning for generation and only add tokens.
#:
#: ``discriminator`` is here for two reasons beyond token cost. It is an OpenAPI
#: keyword rather than core JSON Schema, so a strict validator may reject it
#: outright; and its ``mapping`` holds ``#/$defs/`` pointers that become dangling
#: references the moment definitions are inlined. It is also redundant — a
#: discriminated union's variants already carry a const tag field that
#: determines the variant on its own, and Pydantic applies its own discriminator
#: when validating the response regardless of what the provider was sent.
COSMETIC_KEYS = frozenset(
    {"title", "examples", "default", "$schema", "discriminator"}
)


@dataclass
class StrippedConstraint:
    """One constraint removed from a schema so a provider would accept it."""

    path: str
    keyword: str
    value: Any

    def __str__(self) -> str:
        return f"{self.path}: {self.keyword}={self.value!r}"


@dataclass
class Translation:
    """A provider-acceptable schema and an account of what it cost."""

    schema: dict[str, Any]
    stripped: list[StrippedConstraint] = field(default_factory=list)
    made_nullable: list[str] = field(default_factory=list)
    inlined_refs: int = 0

    @property
    def is_lossless(self) -> bool:
        """Whether the provider will enforce the whole contract."""
        return not self.stripped

    def report(self) -> str:
        """Describe what the provider will and will not enforce."""
        lines: list[str] = []
        if self.made_nullable:
            lines.append(
                f"{len(self.made_nullable)} optional field(s) rewritten as nullable "
                "and required: " + ", ".join(self.made_nullable)
            )
        if self.inlined_refs:
            lines.append(f"{self.inlined_refs} $ref(s) inlined")
        if self.stripped:
            lines.append(
                f"{len(self.stripped)} constraint(s) stripped — NOT enforced by the "
                "provider, enforced locally after parsing:"
            )
            lines.extend(f"    {c}" for c in self.stripped)
        else:
            lines.append("no constraints stripped; the provider enforces the contract")
        return "\n".join(lines)


class SchemaTranslator:
    """Rewrites schemas for a provider, reporting everything it changed.

    Args:
        strip_value_constraints: Remove bounds, lengths, and patterns. Required
            by strict modes; the removed constraints are reported so they can be
            enforced after parsing.
        nullable_optionals: Rewrite optional fields as nullable-and-required.
            Needed on providers that demand every property appear in
            ``required``, and harmless on those that do not — which is why it
            defaults on. A schema targeting the stricter validator works on both.
        inline_refs: Resolve ``$defs``/``$ref`` into the schema body. Measured as
            supported on both providers probed, so off by default, but kept for
            providers that reject references.
    """

    def __init__(
        self,
        strip_value_constraints: bool = True,
        nullable_optionals: bool = True,
        inline_refs: bool = False,
    ) -> None:
        self.strip_value_constraints = strip_value_constraints
        self.nullable_optionals = nullable_optionals
        self.inline_refs = inline_refs

    def translate(self, model: type[BaseModel]) -> Translation:
        """Translate a Pydantic model's schema for the target provider."""
        raw = model.model_json_schema()
        return self.translate_schema(raw)

    def translate_schema(self, raw: dict[str, Any]) -> Translation:
        """Translate a raw JSON Schema."""
        schema = copy.deepcopy(raw)
        result = Translation(schema={})

        defs = schema.pop("$defs", {}) or {}
        if self.inline_refs and defs:
            schema = _inline(schema, defs, result)
            defs = {}

        cleaned = self._walk(schema, defs, path="$", result=result)
        if defs:
            cleaned["$defs"] = {
                name: self._walk(body, defs, f"$defs.{name}", result)
                for name, body in defs.items()
            }

        result.schema = cleaned
        return result

    def _walk(
        self,
        node: Any,
        defs: dict[str, Any],
        path: str,
        result: Translation,
    ) -> Any:
        """Rewrite one schema node, recording every change on the way through."""
        if isinstance(node, list):
            return [self._walk(item, defs, f"{path}[{i}]", result) for i, item in enumerate(node)]
        if not isinstance(node, dict):
            return node

        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in COSMETIC_KEYS:
                continue
            if key in VALUE_CONSTRAINTS and self.strip_value_constraints:
                # Recorded rather than dropped: the contract still requires it,
                # and something has to enforce it after the response parses.
                result.stripped.append(StrippedConstraint(path, key, value))
                continue
            if key in ("properties", "$defs"):
                out[key] = {
                    name: self._walk(sub, defs, f"{path}.{name}", result)
                    for name, sub in value.items()
                }
            elif key in ("anyOf", "oneOf", "allOf", "prefixItems"):
                out[key] = [
                    self._walk(sub, defs, f"{path}.{key}[{i}]", result)
                    for i, sub in enumerate(value)
                ]
            elif key == "items":
                out[key] = self._walk(value, defs, f"{path}[]", result)
            else:
                out[key] = value

        if out.get("type") == "object" or "properties" in out:
            self._normalise_object(out, path, result)

        return out

    def _normalise_object(
        self, node: dict[str, Any], path: str, result: Translation
    ) -> None:
        """Apply the object-level rules strict modes impose."""
        properties = node.get("properties")
        if properties is None:
            return

        # Strict modes reject open objects, and an open object also lets a model
        # add plausible-looking fields the contract never declared.
        node["additionalProperties"] = False

        if not self.nullable_optionals:
            return

        required = list(node.get("required", []))
        missing = [name for name in properties if name not in required]
        for name in missing:
            _make_nullable(properties[name])
            result.made_nullable.append(f"{path}.{name}" if path != "$" else name)

        if missing:
            # Every property must appear in `required`; optionality now lives in
            # the type instead. Order is stabilised so an unchanged contract
            # produces an unchanged schema, which keeps provider-side schema
            # caches warm across runs.
            node["required"] = sorted(properties)
        elif required:
            node["required"] = sorted(required)


def _make_nullable(prop: dict[str, Any]) -> None:
    """Widen a property's type so it admits null."""
    if "type" in prop:
        current = prop["type"]
        if isinstance(current, list):
            if "null" not in current:
                prop["type"] = [*current, "null"]
        elif current != "null":
            prop["type"] = [current, "null"]
        return

    # A property defined by composition rather than a plain type — a $ref, an
    # anyOf, an enum — needs the null alternative added beside it rather than
    # merged into a type keyword that isn't there.
    for combinator in ("anyOf", "oneOf"):
        if combinator in prop:
            if {"type": "null"} not in prop[combinator]:
                prop[combinator] = [*prop[combinator], {"type": "null"}]
            return

    inner = {k: v for k, v in prop.items() if k != "description"}
    if inner:
        description = prop.get("description")
        prop.clear()
        prop["anyOf"] = [inner, {"type": "null"}]
        if description:
            prop["description"] = description


def _inline(schema: dict[str, Any], defs: dict[str, Any], result: Translation) -> dict[str, Any]:
    """Replace every ``$ref`` with the definition it points at."""

    def resolve(node: Any) -> Any:
        if isinstance(node, list):
            return [resolve(item) for item in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.split("/")[-1]
            target = defs.get(name)
            if target is not None:
                result.inlined_refs += 1
                merged = {**resolve(copy.deepcopy(target))}
                # Keep any sibling keywords, which carry the description.
                for key, value in node.items():
                    if key != "$ref":
                        merged.setdefault(key, value)
                return merged
        return {key: resolve(value) for key, value in node.items()}

    return resolve(schema)


def response_format_for(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Build the native-enforcement directive for a translated schema."""
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }
