"""Tests for client construction and the daily budget guard.

None of these make a network call. The guard's whole job is to refuse before the
request leaves, so it is verifiable offline — and it needs to be, because the
resource it protects takes a day to replenish.
"""

from __future__ import annotations

import httpx
import openai
import pytest

from provider.client import (
    BudgetExhaustedError,
    GenerationRejectedError,
    LLMClient,
    MissingCredentialsError,
    _rejected_generation,
)
from provider.registry import get_provider
from store.requests import RequestLog


@pytest.fixture
def log(tmp_path):
    return RequestLog(tmp_path / "client.db")


@pytest.fixture
def keyed(monkeypatch):
    """Credentials present for both providers, without touching the real ones."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")


class TestConstruction:
    def test_missing_key_names_the_variable_to_set(self, monkeypatch, log):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(MissingCredentialsError) as exc:
            LLMClient("groq", request_log=log)
        assert "GROQ_API_KEY" in str(exc.value)

    def test_keyless_provider_constructs_without_credentials(self, log):
        client = LLMClient("ollama", request_log=log)
        assert client.provider.name == "ollama"

    def test_default_never_resolves_to_a_metered_provider(self, monkeypatch, keyed, log):
        """The whole point of the explicit-only rule, asserted directly."""
        monkeypatch.setattr("provider.client.settings.PROVIDER", "openrouter")
        with pytest.raises(ValueError, match="explicitly"):
            LLMClient(request_log=log)

    def test_metered_provider_is_usable_when_asked_for_by_name(self, keyed, log):
        client = LLMClient("openrouter", request_log=log)
        assert client.provider.name == "openrouter"

    def test_default_provider_is_used_when_none_given(self, keyed, log):
        client = LLMClient(request_log=log)
        assert client.provider.name == "groq"


class TestBudgetGuard:
    def test_unmetered_provider_reports_no_limit(self, keyed, log):
        assert LLMClient("groq", request_log=log).remaining_budget() is None

    def test_remaining_budget_falls_as_requests_are_recorded(self, keyed, log):
        client = LLMClient("openrouter", request_log=log)
        allowance = get_provider("openrouter").daily_budget

        assert client.remaining_budget() == allowance
        log.record("openrouter", "m")
        assert client.remaining_budget() == allowance - 1

    def test_another_provider_does_not_consume_this_budget(self, keyed, log):
        client = LLMClient("openrouter", request_log=log)
        allowance = get_provider("openrouter").daily_budget

        for _ in range(10):
            log.record("groq", "m")

        assert client.remaining_budget() == allowance

    def test_call_is_refused_once_the_allowance_is_spent(self, keyed, log):
        client = LLMClient("openrouter", request_log=log)
        for _ in range(get_provider("openrouter").daily_budget):
            log.record("openrouter", "m")

        with pytest.raises(BudgetExhaustedError) as exc:
            client.chat(messages=[{"role": "user", "content": "hi"}], model="m")

        message = str(exc.value)
        assert "45" in message
        assert "UTC" in message

    def test_refusal_happens_before_the_request_is_counted(self, keyed, log):
        """A refused call must not consume more of the allowance it just refused."""
        allowance = get_provider("openrouter").daily_budget
        for _ in range(allowance):
            log.record("openrouter", "m")

        client = LLMClient("openrouter", request_log=log)
        with pytest.raises(BudgetExhaustedError):
            client.chat(messages=[{"role": "user", "content": "hi"}], model="m")

        assert log.count_today("openrouter") == allowance

    def test_exhaustion_is_not_confused_with_a_retryable_failure(self, keyed, log):
        """Backing off or rerouting on exhaustion would defeat the guard."""
        assert not issubclass(BudgetExhaustedError, ConnectionError)
        assert not issubclass(BudgetExhaustedError, TimeoutError)

    def test_budget_is_never_negative(self, keyed, log):
        client = LLMClient("openrouter", request_log=log)
        for _ in range(get_provider("openrouter").daily_budget + 5):
            log.record("openrouter", "m")

        assert client.remaining_budget() == 0


class TestGenerationRejected:
    """A 400 that means "the model could not comply" is a measurement, not a bug.

    Discarding it would delete the loudest evidence enforcement produces: the
    provider held the model to a schema, the model missed, and the provider said
    so — and handed back the text it refused.
    """

    def _error(self, message: str, failed: str | None = None) -> openai.BadRequestError:
        body = {"error": {"message": message, "code": "json_validate_failed"}}
        if failed is not None:
            body["error"]["failed_generation"] = failed
        response = httpx.Response(
            400,
            request=httpx.Request("POST", "https://example.test/v1/chat/completions"),
        )
        return openai.BadRequestError(message, response=response, body=body)

    def test_it_is_recognised_and_carries_the_rejected_text(self):
        exc = _rejected_generation(
            self._error(
                "Failed to generate JSON. See 'failed_generation' for details.",
                failed='{"category",\n "priority"}',
            )
        )
        assert exc is not None
        assert "category" in exc.failed_generation

    def test_an_ordinary_bad_request_is_left_alone(self):
        """A malformed request will be malformed next time; it is not a failure
        of the model."""
        assert _rejected_generation(
            self._error("Invalid value for 'model': no such model")
        ) is None

    def test_a_rejection_without_the_text_still_classifies(self):
        """Not every provider returns what it refused."""
        exc = _rejected_generation(self._error("json_validate_failed"))
        assert exc is not None
        assert exc.failed_generation == ""

    def test_it_is_not_retried_at_the_transport(self, monkeypatch):
        """Retrying re-runs the same generation and spends the allowance twice."""
        calls = []

        class _Completions:
            def create(self, **_kwargs):
                calls.append(1)
                raise self_outer._error("failed_generation was not valid")

        self_outer = self
        client = LLMClient("groq")
        monkeypatch.setattr(
            client._client.chat, "completions", _Completions(), raising=False
        )
        with pytest.raises(GenerationRejectedError):
            client.chat(messages=[{"role": "user", "content": "x"}], model="m")
        assert len(calls) == 1
