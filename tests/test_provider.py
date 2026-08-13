"""Tests for the provider registry, the request counter, and the budget guard.

The guard is the piece worth testing first. It protects a resource that takes a
full day to replenish, and every one of its failure modes is silent: overspending
looks like working code right up until the provider starts refusing, and a guard
that falls back to another provider produces results attributed to the wrong one.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from provider.registry import (
    PROVIDERS,
    UnknownProviderError,
    available_providers,
    get_provider,
)
from store.requests import RequestLog, utc_day


@pytest.fixture
def log(tmp_path):
    """A request log isolated from the real database."""
    return RequestLog(tmp_path / "test.db")


class TestRegistry:
    def test_groq_is_the_default_and_unmetered(self):
        groq = get_provider("groq")
        assert groq.is_default_eligible
        assert not groq.is_metered

    def test_metered_providers_are_never_default_eligible(self):
        """A daily allowance must not be reachable by accident."""
        for provider in PROVIDERS.values():
            if provider.is_metered:
                assert not provider.is_default_eligible, (
                    f"{provider.name} has a daily budget but may be selected "
                    "implicitly, which is how an allowance disappears"
                )

    def test_openrouter_is_metered_and_explicit_only(self):
        openrouter = get_provider("openrouter")
        assert openrouter.is_metered
        assert not openrouter.is_default_eligible

    def test_lookup_is_case_insensitive(self):
        assert get_provider("GROQ").name == "groq"
        assert get_provider("  Groq ").name == "groq"

    def test_unknown_provider_names_the_known_ones(self):
        """A typo should be diagnosable here, not as a connection error later."""
        with pytest.raises(UnknownProviderError) as exc:
            get_provider("grok")
        assert "groq" in str(exc.value)

    def test_keyless_provider_is_usable_without_credentials(self):
        """A local endpoint needs no key and must not be treated as unconfigured."""
        ollama = get_provider("ollama")
        assert ollama.key_env is None
        assert ollama.has_key()
        assert ollama.api_key() is None

    def test_missing_key_is_reported_as_unavailable(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        assert not get_provider("groq").has_key()
        assert "groq" not in {p.name for p in available_providers()}

    def test_empty_key_counts_as_missing(self, monkeypatch):
        """An empty string is a common .env mistake and must not read as set."""
        monkeypatch.setenv("GROQ_API_KEY", "")
        assert not get_provider("groq").has_key()


class TestRequestLog:
    def test_counts_start_at_zero(self, log):
        assert log.count_today("groq") == 0
        assert log.usage_today() == {}

    def test_records_are_counted_per_provider(self, log):
        for _ in range(3):
            log.record("groq", "model-a")
        log.record("openrouter", "model-b")

        assert log.count_today("groq") == 3
        assert log.count_today("openrouter") == 1
        assert log.usage_today() == {"groq": 3, "openrouter": 1}

    def test_counts_survive_a_new_connection(self, tmp_path):
        """The count has to outlive the process, or the guard resets on restart."""
        path = tmp_path / "persist.db"
        RequestLog(path).record("openrouter", "m")
        assert RequestLog(path).count_today("openrouter") == 1

    def test_only_today_is_counted(self, log):
        """Yesterday's spend must not consume today's allowance."""
        yesterday = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        with log._connect() as conn:
            conn.execute(
                "INSERT INTO requests (provider, model, created_at) VALUES (?, ?, ?)",
                ("openrouter", "m", yesterday),
            )
        log.record("openrouter", "m")

        assert log.count_today("openrouter") == 1

    def test_day_boundary_is_utc(self, log):
        """Quotas reset on the provider's clock, not the machine's."""
        assert utc_day() == datetime.now(timezone.utc).strftime("%Y-%m-%d")
