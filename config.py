"""Runtime settings.

Model fields are deliberately empty by default. Provider line-ups change often
enough that any model id hardcoded here would eventually be wrong, and a wrong
default that happens to work today fails confusingly later. An empty one fails
immediately, with an instruction to run the capability probe — which is the
correct behaviour, because the probe is what establishes which models exist and
what they actually support.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration resolved from the environment and an optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    #: Provider used when none is given explicitly. Only providers marked
    #: default-eligible in the registry may appear here; metered ones must be
    #: requested by name so a daily allowance is never spent by accident.
    PROVIDER: str = "groq"

    #: Filled in by the capability probe, not guessed. Empty means "not probed
    #: yet" and every entry point should say so rather than falling back.
    PRIMARY_MODEL: str = ""
    CRITIC_MODEL: str = ""
    FALLBACK_MODEL: str = ""

    #: How many times a failed generation may be sent back for repair. Attempts
    #: past a small number rarely succeed and reliably cost money.
    MAX_REPAIR_ATTEMPTS: int = 3

    #: Base for exponential backoff between repair attempts, in seconds.
    REPAIR_BACKOFF_BASE: float = 0.5

    #: Per-request timeout. Generous, because a slow structured generation is
    #: still a result whereas a timeout is a discarded call.
    REQUEST_TIMEOUT: float = 60.0

    #: Transport-level retries for rate limits and server errors. Client errors
    #: are never retried: a request the API rejects will be rejected identically.
    MAX_TRANSPORT_RETRIES: int = 4

    #: Longest a rate-limit response may ask us to wait before we give up rather
    #: than block. Providers occasionally return very long values under load.
    MAX_RETRY_AFTER: float = 60.0

    LOG_DB_PATH: str = "runs.db"

    #: Age past which a cached capability probe is considered stale. Probing
    #: costs real requests, so results are reused until they expire.
    CAPABILITY_MAX_AGE_HOURS: int = 168


settings = Settings()
