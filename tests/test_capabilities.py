"""Tests for probe classification and the capability cache.

The conformance check decides whether a model is recorded as enforcing a schema
or silently ignoring one, and getting it wrong is not a visible failure — it
produces a matrix that looks authoritative and says the opposite of the truth.
It is strict on purpose, and these tests pin that strictness.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'examples'))
from ladder import build_ladder, is_measurable  # noqa: E402
from provider.capabilities import (
    classify_400,
    FEATURE_PROMPTS,
    FEATURE_SCHEMAS,
    ModelCapabilities,
    TierSupport,
    _conforms_to_probe,
    _is_valid_json,
    _nested_schema,
    _sample,
    _short_error,
    chat_models,
    is_stale,
    load_capabilities,
    mark_retired,
    save_capabilities,
)

VALID = '{"city": "Tokyo", "population": 37000000}'


class TestConformance:
    def test_exact_match_conforms(self):
        assert _conforms_to_probe(VALID)

    def test_key_order_does_not_matter(self):
        assert _conforms_to_probe('{"population": 1, "city": "Tokyo"}')

    @pytest.mark.parametrize(
        "text,reason",
        [
            (f"```json\n{VALID}\n```", "markdown fence"),
            (f"Here you go:\n{VALID}", "prose preamble"),
            ('{"city": "Tokyo"}', "missing required key"),
            ('{"city": "Tokyo", "population": "37000000"}', "number sent as string"),
            ('{"city": "Tokyo", "population": 1, "extra": true}', "extra key"),
            ('{"city": 5, "population": 1}', "wrong type"),
            ("[1, 2, 3]", "not an object"),
            ("not json at all", "unparseable"),
            ("", "empty"),
        ],
    )
    def test_deviations_do_not_conform(self, text, reason):
        """Constrained decoding cannot emit these, so none may pass as enforced."""
        assert not _conforms_to_probe(text), reason

    def test_booleans_are_not_accepted_as_integers(self):
        """Python treats bool as int; a schema does not."""
        assert not _conforms_to_probe('{"city": "Tokyo", "population": true}')

    def test_json_validity_is_separate_from_conformance(self):
        """JSON mode guarantees parseability only, so the checks differ."""
        wrong_shape = '{"anything": 1}'
        assert _is_valid_json(wrong_shape)
        assert not _conforms_to_probe(wrong_shape)


class TestModelFiltering:
    @pytest.mark.parametrize(
        "model",
        [
            "whisper-large-v3",
            "canopylabs/orpheus-v1-english",
            "meta-llama/llama-prompt-guard-2-86m",
            "openai/gpt-oss-safeguard-20b",
            "text-embedding-3-small",
        ],
    )
    def test_non_chat_models_are_excluded(self, model):
        assert chat_models([model]) == []

    @pytest.mark.parametrize(
        "model",
        ["llama-3.3-70b-versatile", "openai/gpt-oss-20b", "groq/compound", "allam-2-7b"],
    )
    def test_chat_models_are_kept(self, model):
        assert chat_models([model]) == [model]


class TestNestedSchema:
    def test_depth_zero_is_a_flat_object(self):
        assert _nested_schema(0)["properties"] == {"value": {"type": "string"}}

    def test_each_level_adds_one_child(self):
        schema = _nested_schema(3)
        for _ in range(3):
            schema = schema["properties"]["child"]
        assert schema["properties"] == {"value": {"type": "string"}}


class TestFeatureProbes:
    def test_every_feature_has_a_prompt(self):
        """A schema with no prompt would probe nothing and report success."""
        assert set(FEATURE_SCHEMAS) == set(FEATURE_PROMPTS)

    def test_feature_schemas_are_valid_json(self):
        for name, schema in FEATURE_SCHEMAS.items():
            json.dumps(schema), name


class TestCapabilityRecord:
    def _caps(self, **tiers) -> ModelCapabilities:
        return ModelCapabilities(
            provider="groq",
            model="m",
            probed_at=datetime.now(timezone.utc).isoformat(),
            tiers=tiers,
        )

    def test_best_tier_prefers_the_strongest_that_conformed(self):
        caps = self._caps(
            json_schema=TierSupport.CONFORMED.value,
            json_object=TierSupport.CONFORMED.value,
        )
        assert caps.best_tier == "json_schema"
        assert caps.enforces_schema

    def test_best_tier_falls_back_when_schema_is_rejected(self):
        caps = self._caps(
            json_schema=TierSupport.REJECTED.value,
            json_object=TierSupport.CONFORMED.value,
        )
        assert caps.best_tier == "json_object"
        assert not caps.enforces_schema

    def test_best_tier_is_prompt_only_when_nothing_conformed(self):
        caps = self._caps(
            json_schema=TierSupport.REJECTED.value,
            json_object=TierSupport.IGNORED.value,
            tools=TierSupport.REJECTED.value,
        )
        assert caps.best_tier == "prompt_only"

    def test_an_ignored_tier_is_not_treated_as_support(self):
        """Accepted-and-ignored is the dangerous case and must never count."""
        caps = self._caps(json_schema=TierSupport.IGNORED.value)
        assert not caps.enforces_schema
        assert caps.best_tier != "json_schema"
        assert "json_schema" in caps.silently_ignores

    def test_rejection_is_not_reported_as_silent_ignoring(self):
        caps = self._caps(json_schema=TierSupport.REJECTED.value)
        assert caps.silently_ignores == []

    def test_prompt_only_non_conformance_is_not_silent_ignoring(self):
        """No directive is sent in the baseline tier, so none can be dropped.

        Counting it would report every model that declines to emit bare JSON as
        a provider silently discarding a constraint, which inflates the exact
        finding this project exists to report.
        """
        caps = self._caps(prompt_only=TierSupport.IGNORED.value)
        assert caps.silently_ignores == []

    def test_a_dropped_directive_is_still_reported(self):
        caps = self._caps(
            tools=TierSupport.IGNORED.value,
            prompt_only=TierSupport.IGNORED.value,
        )
        assert caps.silently_ignores == ["tools"]


class TestCache:
    def test_round_trip_preserves_the_record(self, tmp_path):
        path = tmp_path / "caps.json"
        caps = ModelCapabilities(
            provider="groq",
            model="m",
            probed_at=datetime.now(timezone.utc).isoformat(),
            tiers={"json_schema": TierSupport.CONFORMED.value},
            features={"enum": TierSupport.CONFORMED.value},
            max_nesting_depth=4,
            notes=["a note"],
        )
        save_capabilities({"groq": {"m": caps}}, path)
        loaded = load_capabilities(path)["groq"]["m"]

        assert loaded.enforces_schema
        assert loaded.max_nesting_depth == 4
        assert loaded.notes == ["a note"]

    def test_missing_file_is_empty_not_an_error(self, tmp_path):
        assert load_capabilities(tmp_path / "absent.json") == {}

    def test_fresh_results_are_reused(self):
        caps = ModelCapabilities(
            provider="groq", model="m",
            probed_at=datetime.now(timezone.utc).isoformat(),
        )
        assert not is_stale(caps, max_age_hours=24)

    def test_old_results_expire(self):
        """A stale matrix that still looks authoritative is worse than none."""
        old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        assert is_stale(ModelCapabilities(provider="g", model="m", probed_at=old), 24)

    def test_unparseable_timestamp_counts_as_stale(self):
        assert is_stale(ModelCapabilities(provider="g", model="m", probed_at="nonsense"))


class TestFourHundredClassification:
    """A 400 means two opposite things and the difference changes what to do."""

    @pytest.mark.parametrize(
        "message",
        [
            "Failed to validate JSON. Please adjust your prompt. See 'failed_generation'",
            "generation did not match the expected schema",
            "FAILED_GENERATION: model output invalid",
        ],
    )
    def test_generation_failures_are_distinguished(self, message):
        """The tier applied; the model could not satisfy it. A retry may work."""
        assert classify_400(message) is TierSupport.GEN_FAILED

    @pytest.mark.parametrize(
        "message",
        [
            "This model does not support response format `json_schema`",
            "invalid JSON schema for response_format: `required` is required",
            "tool calling is not supported with this model",
        ],
    )
    def test_unsupported_requests_are_rejections(self, message):
        """The tier is unavailable; drop to a weaker one."""
        assert classify_400(message) is TierSupport.REJECTED

    def test_unrecognised_wording_falls_back_to_rejection(self):
        """Markers are provider-specific, so the safer conclusion is the default."""
        assert classify_400("something entirely new") is TierSupport.REJECTED

    def test_a_generation_failure_is_not_counted_as_support(self):
        caps = ModelCapabilities(
            provider="groq", model="m",
            probed_at=datetime.now(timezone.utc).isoformat(),
            tiers={"json_schema": TierSupport.GEN_FAILED.value},
        )
        assert not caps.enforces_schema
        # Nor as the silent-ignore case: the provider did report the failure.
        assert caps.silently_ignores == []


class TestOptionalFieldForms:
    """The two ways of expressing an optional field are not interchangeable."""

    def test_partial_required_omits_the_property(self):
        schema = FEATURE_SCHEMAS["partial_required"]
        assert "nickname" in schema["properties"]
        assert "nickname" not in schema["required"]

    def test_nullable_field_requires_everything_and_admits_null(self):
        schema = FEATURE_SCHEMAS["nullable_field"]
        assert set(schema["required"]) == set(schema["properties"])
        assert "null" in schema["properties"]["nickname"]["type"]

    def test_they_are_probed_separately(self):
        """Conflating them would report a strict-mode rule as a model limitation."""
        assert "partial_required" in FEATURE_PROMPTS
        assert "nullable_field" in FEATURE_PROMPTS


class TestErrorFormatting:
    def test_provider_message_is_extracted(self):
        raw = "Error code: 400 - {'error': {'message': 'not supported', 'code': 400}}"
        assert _short_error(Exception(raw)) == "not supported"

    def test_unrecognised_shape_falls_back_to_the_raw_text(self):
        assert "boom" in _short_error(Exception("boom"))

    def test_sample_reports_empty_rather_than_nothing(self):
        assert _sample("") == "empty response"
        assert _sample(None) == "empty response"

    def test_sample_collapses_whitespace_and_truncates(self):
        assert _sample("a\n\n  b") == "a b"
        assert len(_sample("x" * 500)) < 130


class TestRetirement:
    """Provider line-ups change. A benchmark that hardcodes model ids breaks
    silently when they do — this run lost two of five models overnight."""

    def _caps(self, model: str, retired: str = "") -> ModelCapabilities:
        return ModelCapabilities(
            provider="groq", model=model, probed_at="2026-08-14T00:00:00+00:00",
            tiers={"json_object": "conformed"}, retired_at=retired,
        )

    def test_a_vanished_model_is_marked(self):
        cached = {"gone": self._caps("gone"), "here": self._caps("here")}
        newly = mark_retired(cached, ["here"])
        assert newly == ["gone"]
        assert cached["gone"].retired_at
        assert not cached["here"].retired_at

    def test_its_measurement_is_kept(self):
        """The date it was measured and the date it disappeared are both
        evidence about how long any of these numbers last."""
        cached = {"gone": self._caps("gone")}
        mark_retired(cached, ["other"])
        assert cached["gone"].tiers == {"json_object": "conformed"}
        assert cached["gone"].probed_at == "2026-08-14T00:00:00+00:00"

    def test_an_empty_listing_retires_nothing(self):
        """A failed list call is not evidence that every model is gone."""
        cached = {"here": self._caps("here")}
        assert mark_retired(cached, []) == []
        assert not cached["here"].retired_at

    def test_marking_twice_keeps_the_first_date(self):
        cached = {"gone": self._caps("gone", retired="2026-08-18T00:00:00+00:00")}
        assert mark_retired(cached, ["other"]) == []
        assert cached["gone"].retired_at == "2026-08-18T00:00:00+00:00"

    def test_a_returning_model_can_be_probed_again(self):
        """Retirement records history; it does not blocklist."""
        cached = {"back": self._caps("back", retired="2026-08-18T00:00:00+00:00")}
        mark_retired(cached, ["back"])
        assert cached["back"].retired_at  # unchanged; the probe overwrites on refresh


class TestLadder:
    def test_a_retired_model_is_left_out(self):
        assert not is_measurable(
            ModelCapabilities(
                provider="groq", model="gone", probed_at="x",
                tiers={"json_schema": "conformed"},
                retired_at="2026-08-18T00:00:00+00:00",
            )
        )

    def test_a_model_that_conforms_nowhere_is_left_out(self):
        """Every cell would be an error, and error cells measure the harness."""
        assert not is_measurable(
            ModelCapabilities(
                provider="groq", model="broken", probed_at="x",
                tiers={"json_schema": "rejected", "prompt_only": "error"},
            )
        )

    def test_models_without_native_enforcement_run_first(self):
        """If a daily token budget runs out, the cells lost should be the ones a
        reader can most easily predict rather than the ones carrying the finding."""
        ladder = build_ladder("groq")
        if len(ladder) < 2:
            pytest.skip("needs a probed provider with several models")
        caps = load_capabilities()["groq"]
        enforcing = [bool(caps[m].enforces_schema) for m in ladder]
        assert enforcing == sorted(enforcing)
