"""Tests for client construction and the daily budget guard.

None of these make a network call. The guard's whole job is to refuse before the
request leaves, so it is verifiable offline — and it needs to be, because the
resource it protects takes a day to replenish.
"""

from __future__ import annotations

import pytest

from provider.client import BudgetExhaustedError, LLMClient, MissingCredentialsError
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
