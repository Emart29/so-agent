"""Known providers, as data rather than code.

Every provider here speaks the OpenAI chat-completions wire format, which is the
entire surface this project needs. Supporting a new one is a row in this table:
no client subclass, no branching, no per-provider request building. If a provider
ever requires a code path of its own, the abstraction has failed and the fix
belongs in the client, not here.

Daily budgets live alongside the provider they constrain rather than in global
config, because a budget is a property of the endpoint's free tier and travels
with it. A provider with no budget is unmetered.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    """An OpenAI-compatible endpoint and the terms of using it.

    Attributes:
        name: Short identifier used on the command line and in the log.
        base_url: Root the OpenAI SDK is pointed at.
        key_env: Environment variable holding the API key, or ``None`` for
            endpoints that need no authentication, such as a local server.
        daily_budget: Requests permitted per day, or ``None`` for unmetered.
            Free tiers with a hard daily cap are declared here so a single
            careless benchmark run cannot cost a day of access.
        is_default_eligible: Whether this provider may be selected implicitly.
            Metered providers set this to ``False``: reaching one by accident is
            how a daily allowance disappears before the work that needed it.
    """

    name: str
    base_url: str
    key_env: str | None
    daily_budget: int | None = None
    is_default_eligible: bool = True

    @property
    def is_metered(self) -> bool:
        """Whether this provider enforces a daily request ceiling."""
        return self.daily_budget is not None

    def api_key(self) -> str | None:
        """Return the configured key, or ``None`` if the endpoint needs none."""
        if self.key_env is None:
            return None
        return os.environ.get(self.key_env) or None

    def has_key(self) -> bool:
        """Whether this provider is usable with the current environment."""
        return self.key_env is None or bool(self.api_key())


PROVIDERS: dict[str, Provider] = {
    "groq": Provider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        key_env="GROQ_API_KEY",
    ),
    # Roughly fifty requests a day on the free tier, so it is metered and never
    # selected implicitly. It earns its place by serving models other providers
    # also serve, which turns a provider comparison into a serving-stack
    # comparison with the weights held constant.
    "openrouter": Provider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        key_env="OPENROUTER_API_KEY",
        daily_budget=45,
        is_default_eligible=False,
    ),
    "openai": Provider(
        name="openai",
        base_url="https://api.openai.com/v1",
        key_env="OPENAI_API_KEY",
    ),
    "together": Provider(
        name="together",
        base_url="https://api.together.xyz/v1",
        key_env="TOGETHER_API_KEY",
    ),
    "cerebras": Provider(
        name="cerebras",
        base_url="https://api.cerebras.ai/v1",
        key_env="CEREBRAS_API_KEY",
    ),
    "ollama": Provider(
        name="ollama",
        base_url="http://localhost:11434/v1",
        key_env=None,
    ),
    "vllm": Provider(
        name="vllm",
        base_url="http://localhost:8000/v1",
        key_env=None,
    ),
}


class UnknownProviderError(ValueError):
    """Raised when a provider name has no entry in the registry."""


def get_provider(name: str) -> Provider:
    """Look up a provider by name.

    Args:
        name: Registry key, case-insensitive.

    Returns:
        The matching provider.

    Raises:
        UnknownProviderError: With the available names, since a typo here is
            otherwise reported as a connection failure much later.
    """
    key = name.strip().lower()
    try:
        return PROVIDERS[key]
    except KeyError:
        known = ", ".join(sorted(PROVIDERS))
        raise UnknownProviderError(
            f"unknown provider {name!r}. Known providers: {known}"
        ) from None


def available_providers() -> list[Provider]:
    """Return providers whose credentials are present in the environment."""
    return [p for p in PROVIDERS.values() if p.has_key()]
