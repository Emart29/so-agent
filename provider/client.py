"""One client for every provider.

This is the only file allowed to know that providers differ. Everything above it
works with a normalised result, so a provider quirk that leaks past this module
is a bug here rather than a special case to handle upstream.

Three behaviours are deliberate and worth stating, because each one is a mistake
that is easy to make and expensive to discover later:

* **Client errors are never retried.** A request the API rejects as malformed is
  rejected identically every time. Retrying it wastes an allowance and hides the
  error message that would have explained the problem.
* **Rate limits are obeyed, not guessed at.** A 429 carries the wait the provider
  wants; sleeping for a made-up interval either wastes time or earns another 429.
* **Budgets refuse rather than reroute.** When a metered provider is exhausted the
  call fails. Silently serving it from elsewhere would attribute the result to the
  wrong provider, which quietly corrupts any comparison between them.
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

import openai

from config import settings
from provider.registry import Provider, get_provider
from store.requests import RequestLog

logger = logging.getLogger("provider.client")


class MissingCredentialsError(RuntimeError):
    """Raised when a provider is selected but its API key is not configured."""


class BudgetExhaustedError(RuntimeError):
    """Raised when a metered provider's daily allowance is spent.

    Deliberately not a subclass of any retryable error: the correct response is
    to stop and report, not to back off or to try somewhere else.
    """


class GenerationRejectedError(RuntimeError):
    """The model could not satisfy a constraint the provider was enforcing.

    Arrives as a 400 and is not one: the request was well formed, and the
    generation failed to validate against the schema the provider was asked to
    hold it to. Treating it as a malformed request would discard a real
    measurement — this is enforcement working, loudly.

    Groq returns the text it could not validate. That is the raw output of a
    failed attempt, so it is carried here rather than dropped, which turns an
    opaque error into a classified failure with the bytes attached.
    """

    def __init__(self, message: str, failed_generation: str = "") -> None:
        super().__init__(message)
        self.failed_generation = failed_generation


#: Rate-limit resets arrive as durations rather than seconds, in a format the
#: OpenAI spec does not define: "410ms", "7.66s", "2m59.56s". Parsed rather than
#: coerced, because ``float("2m36s")`` raises and the silent fallback to
#: exponential backoff waits far longer than the provider asked for.
_DURATION = re.compile(
    r"(?:(?P<h>[\d.]+)h)?(?:(?P<m>[\d.]+)m(?!s))?"
    r"(?:(?P<s>[\d.]+)s)?(?:(?P<ms>[\d.]+)ms)?$"
)


def parse_duration(raw: str | None) -> float | None:
    """Read a rate-limit duration into seconds, or ``None`` if unreadable."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:  # A bare number is already seconds.
        return float(text)
    except ValueError:
        pass

    match = _DURATION.match(text)
    if not match or not any(match.groupdict().values()):
        return None
    parts = {k: float(v) for k, v in match.groupdict().items() if v}
    return (
        parts.get("h", 0.0) * 3600
        + parts.get("m", 0.0) * 60
        + parts.get("s", 0.0)
        + parts.get("ms", 0.0) / 1000
    )


def _is_zero(value: str | None) -> bool:
    """Whether a remaining-quota header says the bucket is empty."""
    try:
        return value is not None and float(value) <= 0
    except (TypeError, ValueError):
        return False


#: Finish reason stamped on a result rebuilt from a rejected generation. Not a
#: value any provider returns; it carries the rejection through the layers that
#: only see a completion.
GENERATION_REJECTED_FINISH_REASON = "generation_rejected"

#: Substrings that mark a 400 as a generation failure rather than a bad request.
GENERATION_REJECTED_MARKERS = (
    "failed_generation",
    "json_validate_failed",
    "failed to generate json",
    "failed to validate json",
)


def _rejected_generation(exc: Exception) -> GenerationRejectedError | None:
    """Recognise a 400 that means "the model could not comply"."""
    message = str(exc)
    if not any(m in message.lower() for m in GENERATION_REJECTED_MARKERS):
        return None

    body = getattr(exc, "body", None)
    failed = ""
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            failed = error.get("failed_generation") or ""
    return GenerationRejectedError(message, failed_generation=failed)


@dataclass
class ChatResult:
    """A completion plus the metadata every later layer needs.

    Attributes:
        text: Assistant message content, or an empty string when the model
            returned only tool calls.
        provider: Which provider served this, recorded on every row so results
            can be grouped by it later.
        model: Model id as the provider reported it, which is not always the one
            that was requested.
        finish_reason: Why generation stopped. ``length`` here is the signal that
            output was truncated, which needs a larger budget rather than a retry.
        tool_calls: Raw tool calls, left unparsed. Their arguments are model
            output and are validated like any other model output.
        prompt_tokens, completion_tokens: Usage, or ``None`` where a provider
            omits it.
        latency_ms: Wall-clock time for the call that produced this result.
        attempts: Transport attempts made, so retry cost is visible.
        raw: The provider's response object, for anything not normalised here.
    """

    text: str
    provider: str
    model: str
    finish_reason: str | None
    tool_calls: list[Any] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float = 0.0
    attempts: int = 1
    raw: Any = None

    @property
    def truncated(self) -> bool:
        """Whether generation stopped because the token limit was reached."""
        return self.finish_reason == "length"

    @property
    def rejected_by_provider(self) -> bool:
        """Whether the provider refused this generation for failing its schema."""
        return self.finish_reason == GENERATION_REJECTED_FINISH_REASON


class LLMClient:
    """Calls any OpenAI-compatible endpoint, with retries and a budget guard."""

    def __init__(
        self,
        provider: str | Provider = None,
        request_log: RequestLog | None = None,
        timeout: float | None = None,
    ) -> None:
        """
        Args:
            provider: Provider name or instance. Defaults to the configured one.
            request_log: Where outbound requests are counted. Defaults to the
                configured database. Pass an explicit log in tests to keep counts
                out of the real one.
            timeout: Per-request timeout override.

        Raises:
            MissingCredentialsError: If the provider needs a key and none is set.
            ValueError: If a metered provider was reached implicitly, which
                should be impossible and indicates a defaulting bug.
        """
        if provider is None:
            provider = settings.PROVIDER
            resolved = get_provider(provider)
            if not resolved.is_default_eligible:
                raise ValueError(
                    f"provider {resolved.name!r} has a daily allowance and must be "
                    "requested explicitly rather than reached as a default"
                )
        resolved = provider if isinstance(provider, Provider) else get_provider(provider)

        if not resolved.has_key():
            raise MissingCredentialsError(
                f"{resolved.name} needs {resolved.key_env}. Set it in the "
                f"environment or add it to .env, then try again."
            )

        self.provider = resolved
        self.requests = request_log or RequestLog(settings.LOG_DB_PATH)
        self.timeout = timeout if timeout is not None else settings.REQUEST_TIMEOUT

        self._client = openai.OpenAI(
            api_key=resolved.api_key() or "not-required",
            base_url=resolved.base_url,
            timeout=self.timeout,
            # Retries are handled here rather than by the SDK so that a 429's
            # retry-after is honoured and every attempt is counted against the
            # daily allowance, which the SDK's internal retries would not be.
            max_retries=0,
        )

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    def remaining_budget(self) -> int | None:
        """Return requests left today, or ``None`` when unmetered."""
        if not self.provider.is_metered:
            return None
        used = self.requests.count_today(self.provider.name)
        return max(self.provider.daily_budget - used, 0)

    def _check_budget(self) -> None:
        """Refuse the call if today's allowance for this provider is spent."""
        remaining = self.remaining_budget()
        if remaining is not None and remaining <= 0:
            used = self.requests.count_today(self.provider.name)
            raise BudgetExhaustedError(
                f"{self.provider.name} daily allowance spent: {used} of "
                f"{self.provider.daily_budget} requests used today (UTC). "
                "It resets at midnight UTC."
            )

    # ------------------------------------------------------------------
    # Calling
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **extra: Any,
    ) -> ChatResult:
        """Send one chat completion and return a normalised result.

        Args:
            messages: Conversation in OpenAI format.
            model: Model id, as the provider names it.
            response_format: Enforcement directive, when the provider supports one.
            tools: Function-calling definitions.
            max_tokens: Output ceiling.
            temperature: Sampling temperature, where the provider accepts it.
            **extra: Passed through untouched, for provider-specific parameters.

        Returns:
            The completion and its metadata.

        Raises:
            BudgetExhaustedError: The provider's daily allowance is spent.
            openai.BadRequestError: The request was rejected. Surfaced unchanged
                rather than retried, because its message is usually the only
                explanation of what the schema got wrong.
        """
        self._check_budget()

        payload: dict[str, Any] = {"model": model, "messages": messages}
        if response_format is not None:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        payload.update(extra)

        started = time.perf_counter()
        response = self._send_with_retries(payload, model)
        latency_ms = (time.perf_counter() - started) * 1000.0

        return self._normalise(response, latency_ms)

    def _send_with_retries(self, payload: dict[str, Any], model: str) -> Any:
        """Send the request, retrying only what retrying can fix."""
        last_error: Exception | None = None

        for attempt in range(1, settings.MAX_TRANSPORT_RETRIES + 1):
            # Counted before the call is made: a request that errors still
            # consumed the allowance, and not counting it would let the guard
            # overspend exactly when things are going wrong.
            self.requests.record(self.provider.name, model)
            self._attempts = attempt

            try:
                return self._client.chat.completions.create(**payload)

            except openai.BadRequestError as exc:
                # A 400 usually means the request is wrong and will be wrong
                # next time too. One kind is not: the provider enforced a
                # constraint and the model could not satisfy it, which is a
                # generation failure carrying its own evidence.
                rejected = _rejected_generation(exc)
                if rejected is not None:
                    raise rejected from exc
                raise

            except openai.RateLimitError as exc:
                last_error = exc
                wait = self._retry_after(exc) or self._backoff(attempt)
                if attempt == settings.MAX_TRANSPORT_RETRIES:
                    break
                logger.warning(
                    "%s rate limited, waiting %.1fs (attempt %d/%d)",
                    self.provider.name, wait, attempt, settings.MAX_TRANSPORT_RETRIES,
                )
                time.sleep(wait)
                self._check_budget()

            except (openai.APIConnectionError, openai.APITimeoutError) as exc:
                last_error = exc
                if attempt == settings.MAX_TRANSPORT_RETRIES:
                    break
                time.sleep(self._backoff(attempt))

            except openai.APIStatusError as exc:
                last_error = exc
                # 4xx other than 429 means the request is unacceptable as sent.
                if exc.status_code < 500:
                    raise
                if attempt == settings.MAX_TRANSPORT_RETRIES:
                    break
                time.sleep(self._backoff(attempt))

        raise RuntimeError(
            f"{self.provider.name} failed after {settings.MAX_TRANSPORT_RETRIES} "
            f"attempts: {last_error}"
        ) from last_error

    def _retry_after(self, exc: Exception) -> float | None:
        """Read the wait a rate-limit response asked for, if it gave one.

        Which bucket ran out decides the wait, because they refill on very
        different timescales. A token-per-minute limit typically resets in under
        a second; a daily request quota does not. Waiting the wrong one turns a
        400ms pause into a minute, which across a benchmark is the difference
        between hours and days.
        """
        response = getattr(exc, "response", None)
        header = getattr(response, "headers", None)
        if not header:
            return None

        exhausted = [
            header.get(f"x-ratelimit-reset-{bucket}")
            for bucket in ("tokens", "requests")
            if _is_zero(header.get(f"x-ratelimit-remaining-{bucket}"))
        ]
        candidates = [w for w in (parse_duration(v) for v in exhausted) if w is not None]

        if not candidates:
            wait = parse_duration(header.get("retry-after"))
            if wait is None:
                return None
            candidates = [wait]

        # Providers occasionally return very long waits under load. Past a
        # threshold it is better to fail visibly than to block for minutes.
        return min(max(max(candidates), 0.0), settings.MAX_RETRY_AFTER)

    @staticmethod
    def _backoff(attempt: int) -> float:
        """Exponential backoff with jitter, so parallel workers desynchronise."""
        return min(2.0 ** (attempt - 1), 8.0) + random.uniform(0, 0.25)

    def _normalise(self, response: Any, latency_ms: float) -> ChatResult:
        """Flatten a provider response into the shape the rest of the code uses.

        Providers vary in what they omit — usage blocks, tool-call fields, even
        the message itself on a pure tool call. Every one of those differences is
        absorbed here so no caller has to know which provider it is talking to.
        """
        choice = response.choices[0] if response.choices else None
        message = getattr(choice, "message", None)

        text = (getattr(message, "content", None) or "") if message else ""
        tool_calls = list(getattr(message, "tool_calls", None) or []) if message else []

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None

        return ChatResult(
            text=text,
            provider=self.provider.name,
            model=getattr(response, "model", "") or "",
            finish_reason=getattr(choice, "finish_reason", None) if choice else None,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            attempts=getattr(self, "_attempts", 1),
            raw=response,
        )

    def list_models(self) -> list[str]:
        """Return model ids this key can reach.

        Returns an empty list where a provider does not implement the endpoint,
        rather than raising: callers fall back to an explicit model list and say
        so, which is a reportable state rather than a failure.
        """
        try:
            return sorted(m.id for m in self._client.models.list().data)
        except Exception as exc:
            logger.info("%s does not list models: %s", self.provider.name, exc)
            return []
