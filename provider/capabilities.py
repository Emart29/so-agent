"""Measures what a model actually does, rather than what its docs claim.

Provider documentation on structured output is inconsistent, goes stale, and
never covers the case that matters most: an endpoint that *accepts* a
``response_format`` and then ignores it. That failure is invisible from the
request side — the call succeeds, the response looks fine, and nothing enforced
anything. The only way to catch it is to check that the output actually
conformed, which is what this module does.

Four outcomes are distinguished per enforcement tier:

* ``REJECTED``   — the API refused the request outright. The tier is unavailable
  on this model and a caller should drop to a weaker one.
* ``GEN_FAILED`` — the API applied the constraint and the model could not
  satisfy it. Also a 400, and the opposite conclusion: the tier works here, this
  model is merely unreliable at it, and a retry may succeed.
* ``IGNORED``    — the API accepted the directive and the output did not
  conform. The dangerous one: real constrained decoding cannot emit
  non-conforming output, so this proves nothing was enforced.
* ``CONFORMED``  — accepted, and the output matched. Evidence of support, not
  proof of reliability; how often it holds is what the benchmark measures.

Results are cached to disk because probing costs real requests, and on a metered
provider re-probing on every run would spend the day's allowance before the work
that needed it.
"""

from __future__ import annotations

import json
import re
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import openai

from config import settings
from provider.client import BudgetExhaustedError, LLMClient

logger = logging.getLogger("provider.capabilities")

CAPABILITIES_PATH = Path("capabilities.json")

#: Substrings marking models that do not serve chat completions. Probing a
#: speech or classifier model wastes a request to learn nothing.
NON_CHAT_HINTS = (
    "whisper", "orpheus", "tts", "embed", "prompt-guard",
    "guard", "moderation", "rerank", "distil-whisper",
)

#: Reasoning models spend output tokens before emitting visible text, so a
#: ceiling sized for the answer alone returns an empty string and reads as a
#: failure. Probes are budgeted for the thinking as well as the answer.
PROBE_MAX_TOKENS = 2000


class TierSupport(str, Enum):
    """What happened when an enforcement tier was tried.

    ``REJECTED`` and ``GEN_FAILED`` both arrive as a 400 and mean opposite
    things. Rejected means the tier is unavailable and the caller should drop to
    a weaker one. Generation-failed means the tier *was* applied, the provider
    tried to enforce it, and the model could not produce conforming output —
    the tier works, this model is unreliable at it, and a retry may succeed.
    Collapsing them would hide a model that is merely unreliable behind one that
    is unsupported.
    """

    CONFORMED = "conformed"
    IGNORED = "ignored"
    GEN_FAILED = "gen_failed"
    REJECTED = "rejected"
    ERROR = "error"
    UNTESTED = "untested"


#: Markers in a 400 body indicating the provider applied the constraint and the
#: model failed to satisfy it, rather than refusing the request outright. These
#: are provider-specific wordings; anything unrecognised falls back to
#: ``REJECTED``, which is the safer of the two to be wrong about.
GENERATION_FAILURE_MARKERS = (
    "failed_generation",
    "failed to validate json",
    "did not match the expected schema",
)


#: Tiers that send an actual enforcement directive. The prompt-only baseline
#: sends none, so it cannot be "ignored" — it is the control the others are
#: measured against.
DIRECTIVE_TIERS = ("json_schema", "json_object", "tools")


def classify_400(message: str) -> TierSupport:
    """Decide whether a 400 means unsupported or attempted-and-failed."""
    lowered = message.lower()
    if any(marker in lowered for marker in GENERATION_FAILURE_MARKERS):
        return TierSupport.GEN_FAILED
    return TierSupport.REJECTED


#: The schema every tier is probed with. Deliberately trivial: the question is
#: whether enforcement happens at all, so anything a model could plausibly fail
#: on its own merits would confound the answer.
PROBE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "city": {"type": "string", "description": "The city's name"},
        "population": {"type": "integer", "description": "Population as an integer"},
    },
    "required": ["city", "population"],
    "additionalProperties": False,
}

PROBE_PROMPT = (
    "Return the name and approximate population of Tokyo as a JSON object "
    'with exactly the keys "city" (string) and "population" (integer).'
)

#: Schema fragments probed one at a time, each isolating a single feature so a
#: rejection names the thing that caused it.
FEATURE_SCHEMAS: dict[str, dict[str, Any]] = {
    "nested_object": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "location": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
                "required": ["city", "country"],
                "additionalProperties": False,
            },
        },
        "required": ["name", "location"],
        "additionalProperties": False,
    },
    "array_of_objects": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"label": {"type": "string"}, "count": {"type": "integer"}},
                    "required": ["label", "count"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    },
    "enum": {
        "type": "object",
        "properties": {"priority": {"type": "string", "enum": ["low", "medium", "high"]}},
        "required": ["priority"],
        "additionalProperties": False,
    },
    # Two ways of saying "this field may be absent", probed separately because
    # they are not interchangeable and the difference decides how a schema
    # translator has to rewrite an optional field.
    #
    # `partial_required` is what Pydantic emits: the property simply does not
    # appear in `required`. Strict modes commonly reject that, demanding every
    # property be listed.
    #
    # `nullable_field` is the strict-legal form: every property is required and
    # optionality is carried by the type admitting null.
    "partial_required": {
        "type": "object",
        "properties": {"name": {"type": "string"}, "nickname": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    },
    "nullable_field": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "nickname": {"type": ["string", "null"]},
        },
        "required": ["name", "nickname"],
        "additionalProperties": False,
    },
    # Pydantic emits `oneOf` for a discriminated union, not `anyOf`. They are
    # different keywords and a validator may accept one and reject the other, so
    # measuring only `anyOf` and assuming unions work would leave the shape this
    # project actually sends unmeasured.
    "oneof_discriminated": {
        "type": "object",
        "properties": {
            "decision": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "kind": {"const": "answer"},
                            "text": {"type": "string"},
                        },
                        "required": ["kind", "text"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "kind": {"const": "question"},
                            "ask": {"type": "string"},
                        },
                        "required": ["kind", "ask"],
                        "additionalProperties": False,
                    },
                ]
            }
        },
        "required": ["decision"],
        "additionalProperties": False,
    },
    "anyof_union": {
        "type": "object",
        "properties": {
            "result": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {"kind": {"const": "answer"}, "text": {"type": "string"}},
                        "required": ["kind", "text"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {"kind": {"const": "question"}, "ask": {"type": "string"}},
                        "required": ["kind", "ask"],
                        "additionalProperties": False,
                    },
                ]
            }
        },
        "required": ["result"],
        "additionalProperties": False,
    },
    "ref_defs": {
        "type": "object",
        "$defs": {
            "point": {
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                "required": ["x", "y"],
                "additionalProperties": False,
            }
        },
        "properties": {"start": {"$ref": "#/$defs/point"}},
        "required": ["start"],
        "additionalProperties": False,
    },
}

FEATURE_PROMPTS: dict[str, str] = {
    "nested_object": "Return a person named Ada located in London, United Kingdom.",
    "array_of_objects": "Return two items: apples with count 3, pears with count 5.",
    "enum": "A server is on fire. Return its priority.",
    "partial_required": "Return a person named Ada, with no nickname.",
    "nullable_field": "Return a person named Ada, with a null nickname.",
    "anyof_union": "Answer this: what is 2+2? Use the answer variant.",
    "oneof_discriminated": "Answer this: what is 2+2? Use the answer variant.",
    "ref_defs": "Return a start point at coordinates x=1, y=2.",
}


@dataclass
class ModelCapabilities:
    """What one model on one provider was measured to support."""

    provider: str
    model: str
    probed_at: str
    tiers: dict[str, str] = field(default_factory=dict)
    features: dict[str, str] = field(default_factory=dict)
    max_nesting_depth: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def best_tier(self) -> str:
        """Strongest enforcement tier this model was measured to honour."""
        for tier in ("json_schema", "json_object", "tools"):
            if self.tiers.get(tier) == TierSupport.CONFORMED:
                return tier
        return "prompt_only"

    @property
    def enforces_schema(self) -> bool:
        """Whether native schema enforcement conformed when probed."""
        return self.tiers.get("json_schema") == TierSupport.CONFORMED

    @property
    def silently_ignores(self) -> list[str]:
        """Tiers where a directive was sent, accepted, and not honoured.

        Only tiers that actually carry an enforcement directive can be ignored.
        The prompt-only baseline sends none, so non-conforming output there is
        the expected result rather than a provider silently dropping a
        constraint — counting it would inflate the finding this project exists
        to report.
        """
        return [
            tier
            for tier, result in self.tiers.items()
            if result == TierSupport.IGNORED and tier in DIRECTIVE_TIERS
        ]


def _conforms_to_probe(text: str) -> bool:
    """Whether a response matches the probe schema exactly.

    Strict on purpose. Constrained decoding cannot produce a wrong shape, so any
    deviation — a markdown fence, a missing key, a stringified number — proves
    the constraint was not applied, and treating those leniently would report
    unenforced models as enforced.
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(data, dict):
        return False
    if set(data) != {"city", "population"}:
        return False
    population = data["population"]
    # bool is a subclass of int in Python but not an integer to a JSON schema,
    # so an unguarded isinstance check would accept `true` as a population and
    # record an unenforced model as enforcing.
    if isinstance(population, bool) or not isinstance(population, int):
        return False
    return isinstance(data["city"], str)


def _is_valid_json(text: str) -> bool:
    """Whether the text parses as JSON at all."""
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def chat_models(models: list[str]) -> list[str]:
    """Filter a model list down to plausible chat-completion models."""
    return [m for m in models if not any(h in m.lower() for h in NON_CHAT_HINTS)]


class CapabilityProber:
    """Runs enforcement and schema probes against one provider."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.provider = client.provider.name

    def _confirm(
        self,
        verdict: TierSupport,
        note: str | None,
        retry: "Callable[[], tuple[TierSupport, str | None]]",
    ) -> tuple[TierSupport, str | None]:
        """Re-run a probe before recording that a directive was ignored.

        Ignoring is the most consequential verdict this module can reach — it
        says a provider accepted an enforcement directive and did not honour
        it — and a single sample is not enough to support it. Transient empty
        responses and one-off malformed generations both look identical to a
        provider that never enforced anything, and recording either as ignored
        publishes a false claim about the provider.

        Only the surprising verdict pays for the extra request; everything else
        is recorded from the first sample.
        """
        if verdict is not TierSupport.IGNORED:
            return verdict, note

        second, second_note = retry()
        if second is not TierSupport.IGNORED:
            return second, f"first sample did not conform ({note}); second did"
        return second, f"{second_note} (confirmed over 2 samples)"

    def _ask(
        self,
        model: str,
        prompt: str,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[str | None, Any, str | None]:
        """Send one probe.

        Returns:
            ``(text, result, error)``. ``error`` is a short reason string when
            the request failed, and ``None`` when it succeeded.
        """
        try:
            result = self.client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                response_format=response_format,
                tools=tools,
                max_tokens=PROBE_MAX_TOKENS,
            )
            return result.text, result, None
        except openai.BadRequestError as exc:
            return None, None, _short_error(exc)
        except BudgetExhaustedError:
            raise
        except Exception as exc:  # noqa: BLE001 - probing must not abort a sweep
            return None, None, f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------
    # Enforcement tiers
    # ------------------------------------------------------------------

    def probe_json_schema(self, model: str) -> tuple[TierSupport, str | None]:
        """Probe native JSON-schema enforcement."""
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "city_population",
                "strict": True,
                "schema": PROBE_SCHEMA,
            },
        }
        text, result, error = self._ask(model, PROBE_PROMPT, response_format)
        if error:
            return classify_400(error), error
        if result is not None and result.truncated:
            return TierSupport.ERROR, "truncated before producing output"
        if _conforms_to_probe(text or ""):
            return TierSupport.CONFORMED, None
        if not (text or "").strip():
            # No output at all says nothing about whether a constraint was
            # applied, so it must not be recorded as the provider ignoring one.
            return TierSupport.ERROR, "empty response"
        # Accepted, produced non-conforming content. Enforcement did not happen.
        return TierSupport.IGNORED, _sample(text)

    def probe_json_object(self, model: str) -> tuple[TierSupport, str | None]:
        """Probe JSON mode, which guarantees valid JSON but not a shape."""
        prompt = f"{PROBE_PROMPT}\n\nRespond with JSON only."
        text, result, error = self._ask(model, prompt, {"type": "json_object"})
        if error:
            return classify_400(error), error
        if result is not None and result.truncated:
            return TierSupport.ERROR, "truncated before producing output"
        if _is_valid_json(text or ""):
            return TierSupport.CONFORMED, None
        if not (text or "").strip():
            return TierSupport.ERROR, "empty response"
        return TierSupport.IGNORED, _sample(text)

    def probe_tools(self, model: str) -> tuple[TierSupport, str | None]:
        """Probe function calling, which is structured output by another name."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "record_city",
                    "description": "Record a city and its population.",
                    "parameters": PROBE_SCHEMA,
                },
            }
        ]
        _, result, error = self._ask(
            model, "Record Tokyo and its population using the tool.", tools=tools
        )
        if error:
            return classify_400(error), error
        if result is None or not result.tool_calls:
            return TierSupport.IGNORED, "no tool call emitted"
        try:
            args = json.loads(result.tool_calls[0].function.arguments)
        except (json.JSONDecodeError, AttributeError, TypeError) as exc:
            return TierSupport.IGNORED, f"unparseable tool arguments: {exc}"
        if not isinstance(args, dict) or "city" not in args:
            return TierSupport.IGNORED, f"unexpected arguments: {args}"
        return TierSupport.CONFORMED, None

    def probe_prompt_only(self, model: str) -> tuple[TierSupport, str | None]:
        """Baseline: ask for JSON with no enforcement and see what arrives."""
        prompt = f"{PROBE_PROMPT}\n\nReturn only the JSON object, nothing else."
        text, result, error = self._ask(model, prompt)
        if error:
            return TierSupport.ERROR, error
        if result is not None and result.truncated:
            return TierSupport.ERROR, "truncated before producing output"
        if _conforms_to_probe(text or ""):
            return TierSupport.CONFORMED, None
        if not (text or "").strip():
            return TierSupport.ERROR, "empty response"
        return TierSupport.IGNORED, _sample(text)

    # ------------------------------------------------------------------
    # Schema features
    # ------------------------------------------------------------------

    def probe_feature(self, model: str, feature: str) -> tuple[TierSupport, str | None]:
        """Probe one schema feature under native enforcement."""
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": f"probe_{feature}",
                "strict": True,
                "schema": FEATURE_SCHEMAS[feature],
            },
        }
        text, result, error = self._ask(model, FEATURE_PROMPTS[feature], response_format)
        if error:
            return classify_400(error), error
        if result is not None and result.truncated:
            return TierSupport.ERROR, "truncated before producing output"
        if _is_valid_json(text or ""):
            return TierSupport.CONFORMED, None
        return TierSupport.IGNORED, _sample(text)

    def probe_nesting_depth(self, model: str, ceiling: int = 6) -> int | None:
        """Find how deeply a schema may nest before the provider refuses.

        Searched rather than assumed: providers document this rarely and
        inconsistently, and it is the limit most likely to be hit by a realistic
        schema. Returns the deepest level accepted, or ``None`` if even one
        level failed.
        """
        deepest: int | None = None
        for depth in range(1, ceiling + 1):
            schema = _nested_schema(depth)
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": f"depth_{depth}",
                    "strict": True,
                    "schema": schema,
                },
            }
            _, _, error = self._ask(
                model, f"Return a nested object {depth} levels deep.", response_format
            )
            if error:
                # A generation failure means the schema was accepted and the
                # model merely could not satisfy it, which says nothing about
                # the structural limit being searched for here.
                if classify_400(error) is TierSupport.GEN_FAILED:
                    deepest = depth
                    continue
                break
            deepest = depth
        return deepest

    # ------------------------------------------------------------------
    # Full sweep
    # ------------------------------------------------------------------

    def probe_model(self, model: str, features: bool = True) -> ModelCapabilities:
        """Run every probe against one model."""
        caps = ModelCapabilities(
            provider=self.provider,
            model=model,
            probed_at=datetime.now(timezone.utc).isoformat(),
        )

        for tier, probe in (
            ("json_schema", self.probe_json_schema),
            ("json_object", self.probe_json_object),
            ("tools", self.probe_tools),
            ("prompt_only", self.probe_prompt_only),
        ):
            support, note = probe(model)
            if tier in DIRECTIVE_TIERS:
                support, note = self._confirm(
                    support, note, lambda p=probe: p(model)
                )
            caps.tiers[tier] = support.value
            if note:
                caps.notes.append(f"{tier}: {note}")

        # Feature probes only mean something where native enforcement holds.
        # Running them otherwise would measure the model's willingness to
        # comply, which is a different question.
        if features and caps.enforces_schema:
            for feature in FEATURE_SCHEMAS:
                support, note = self.probe_feature(model, feature)
                caps.features[feature] = support.value
                if note:
                    caps.notes.append(f"{feature}: {note}")
            caps.max_nesting_depth = self.probe_nesting_depth(model)

        return caps


def _nested_schema(depth: int) -> dict[str, Any]:
    """Build an object schema nested to a given depth."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    for _ in range(depth):
        schema = {
            "type": "object",
            "properties": {"child": schema},
            "required": ["child"],
            "additionalProperties": False,
        }
    return schema


def _short_error(exc: Exception) -> str:
    """Condense a provider error to the part that identifies the cause.

    Error bodies arrive as a stringified dict whose quoting is not consistent:
    the key may be single-quoted while the value is double-quoted, because
    Python's repr switches quote style when the value itself contains an
    apostrophe. Matching a fixed marker therefore truncates exactly the errors
    worth reading — the ones naming a schema field.
    """
    message = str(exc)
    match = re.search(r"""["']message["']\s*:\s*(["'])(.*?)\1(?=\s*[,}])""", message, re.S)
    if match:
        return match.group(2)[:200]
    return message[:200]


def _sample(text: str | None) -> str:
    """Return a short excerpt of a non-conforming response, for the notes."""
    if not text:
        return "empty response"
    flat = " ".join(text.split())
    return flat[:120] + ("..." if len(flat) > 120 else "")


# ----------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------

def load_capabilities(path: Path = CAPABILITIES_PATH) -> dict[str, dict[str, ModelCapabilities]]:
    """Load cached probe results, keyed by provider then model."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        provider: {model: ModelCapabilities(**caps) for model, caps in models.items()}
        for provider, models in raw.items()
    }


def save_capabilities(
    data: dict[str, dict[str, ModelCapabilities]], path: Path = CAPABILITIES_PATH
) -> None:
    """Write probe results to disk."""
    serialisable = {
        provider: {model: asdict(caps) for model, caps in models.items()}
        for provider, models in data.items()
    }
    path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")


def is_stale(caps: ModelCapabilities, max_age_hours: int | None = None) -> bool:
    """Whether a cached result is old enough to be worth re-measuring.

    Capability data describes a provider at a moment. Lineups and enforcement
    behaviour change, and a stale matrix that still looks authoritative is worse
    than none — so results expire rather than being trusted indefinitely.
    """
    max_age = max_age_hours or settings.CAPABILITY_MAX_AGE_HOURS
    try:
        probed = datetime.fromisoformat(caps.probed_at)
    except ValueError:
        return True
    return datetime.now(timezone.utc) - probed > timedelta(hours=max_age)
