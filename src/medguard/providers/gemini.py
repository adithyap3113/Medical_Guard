"""Gemini generation provider.

**This is the only module in MedGuard permitted to import ``google.genai``.**
Everything it raises is a :mod:`medguard.errors` type and everything it returns is a
:mod:`medguard.providers.base` type, so the vendor stops here.

Retry policy
------------
Retrying is only correct for failures that are actually transient, and the hard part
is that HTTP 429 covers two very different situations:

* **rate limit** (too many requests this minute) -- clears in seconds, retry;
* **quota exhaustion** (free-tier daily cap) -- will not clear for hours, do not retry.

Retrying the second wastes the caller's time and, worse, makes the real cause look
like a flaky network. They are separated below by inspecting the error payload. That
inspection is a heuristic against Google's message wording, so it is written to fail
safe: anything not clearly identifiable as a daily/hard quota is treated as a rate
limit and retried, which is the recoverable direction to be wrong in.
"""

from __future__ import annotations

import re
import time
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from medguard.config import GeminiSettings, get_settings
from medguard.errors import (
    ProviderAuthError,
    ProviderBlockedError,
    ProviderError,
    ProviderModelNotFoundError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from medguard.logging import get_logger
from medguard.providers.base import LLMResponse, ProviderHealth

__all__ = ["GeminiProvider"]

log = get_logger(__name__)

#: finish_reason values that mean "the model refused or was stopped", not "done".
_BLOCKING_FINISH_REASONS = frozenset(
    {
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
    }
)

#: Substrings that indicate a *hard* quota (daily/lifetime cap) rather than a
#: per-minute throttle. Matched case-insensitively against the error message.
_HARD_QUOTA_MARKERS = (
    "per day",
    "perday",
    "per_day",
    "daily",
    "requests per day",
    "free_tier",
    "free tier",
    "billing",
    "exceeded your current quota",
)

_RETRY_DELAY_RE = re.compile(r'"?retryDelay"?\s*[:=]\s*"?(\d+(?:\.\d+)?)s', re.IGNORECASE)


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, ProviderError) and exc.retryable


def _extract_retry_after(exc: genai_errors.APIError) -> float | None:
    """Pull ``RetryInfo.retryDelay`` out of the error payload when Google supplies it.

    Honouring the server's own backoff hint is strictly better than guessing with
    exponential backoff, and it is the difference between a 429 that resolves and one
    that keeps tripping the same limit.
    """
    details: Any = getattr(exc, "details", None)
    if isinstance(details, dict):
        inner = details.get("error", details)
        items = inner.get("details") if isinstance(inner, dict) else None
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and "retryDelay" in item:
                    match = _RETRY_DELAY_RE.search(f'"retryDelay": "{item["retryDelay"]}"')
                    if match:
                        return float(match.group(1))
    match = _RETRY_DELAY_RE.search(str(getattr(exc, "message", "") or ""))
    return float(match.group(1)) if match else None


def _looks_like_hard_quota(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _HARD_QUOTA_MARKERS)


def _map_api_error(exc: genai_errors.APIError) -> ProviderError:
    """Translate a Gemini SDK error into MedGuard's vendor-neutral taxonomy.

    Note what is *not* used here: ``str(exc)``. The SDK's string form embeds the whole
    response JSON, which can echo request content back. Only the structured ``code``,
    ``status`` and ``message`` fields are carried forward.
    """
    code = getattr(exc, "code", None)
    status = str(getattr(exc, "status", "") or "").upper()
    message = str(getattr(exc, "message", "") or "").strip() or "no message provided"
    lowered = message.lower()

    is_key_problem = any(
        marker in lowered
        for marker in ("api key not valid", "api_key_invalid", "invalid api key", "unauthenticated")
    )

    if code in (401, 403) or status in ("UNAUTHENTICATED", "PERMISSION_DENIED") or is_key_problem:
        return ProviderAuthError(
            f"Gemini rejected the credentials (HTTP {code}, {status}): {message}",
            status_code=code,
        )

    if code == 404 or status == "NOT_FOUND":
        return ProviderModelNotFoundError(
            f"Gemini model not found or not enabled for this API project: {message}",
            status_code=code,
        )

    if code == 429 or status == "RESOURCE_EXHAUSTED":
        if _looks_like_hard_quota(message):
            return ProviderQuotaError(
                f"Gemini quota exhausted (HTTP {code}): {message}", status_code=code
            )
        return ProviderRateLimitError(
            f"Gemini rate limit hit (HTTP {code}): {message}",
            retry_after_s=_extract_retry_after(exc),
            status_code=code,
        )

    if status == "DEADLINE_EXCEEDED" or code == 504:
        return ProviderTimeoutError(f"Gemini deadline exceeded: {message}", status_code=code)

    if (isinstance(code, int) and code >= 500) or status in ("UNAVAILABLE", "INTERNAL", "ABORTED"):
        return ProviderUnavailableError(
            f"Gemini upstream error (HTTP {code}, {status}): {message}", status_code=code
        )

    if isinstance(code, int) and 400 <= code < 500:
        # Unknown client error: our request was wrong, so retrying it changes nothing.
        return ProviderError(
            f"Gemini rejected the request (HTTP {code}, {status}): {message}",
            user_message="The request to the language model was invalid.",
            status_code=code,
        )

    return ProviderResponseError(f"Unexpected Gemini error (HTTP {code}): {message}")


def _map_transport_error(exc: Exception) -> ProviderError:
    """Translate httpx transport failures, which never reach the SDK's error types."""
    if isinstance(exc, httpx.TimeoutException):
        return ProviderTimeoutError(f"Gemini request timed out: {type(exc).__name__}")
    return ProviderUnavailableError(f"Could not reach Gemini: {type(exc).__name__}: {exc}")


class GeminiProvider:
    """Concrete :class:`~medguard.providers.base.LLMProvider` backed by Gemini.

    The SDK client is built lazily and cached. Lazy because constructing it with an
    absent or malformed key should surface when the provider is *used* or explicitly
    health-checked -- not at import time, which would make ``medguard doctor``
    impossible to run on a machine that has no key yet.
    """

    name = "gemini"

    def __init__(self, settings: GeminiSettings | None = None) -> None:
        self._s = settings if settings is not None else get_settings().gemini
        self._client: genai.Client | None = None

    # -- client -----------------------------------------------------------------

    def _get_client(self) -> genai.Client:
        if self._client is None:
            if not self._s.configured:
                raise ProviderAuthError(
                    "GEMINI_API_KEY is not set.",
                    user_message="The language model is not configured.",
                )
            assert self._s.api_key is not None  # noqa: S101 - guarded by `configured`
            self._client = genai.Client(
                api_key=self._s.api_key.get_secret_value(),
                # HttpOptions.timeout is in milliseconds, not seconds.
                http_options=genai_types.HttpOptions(timeout=self._s.timeout_ms),
            )
        return self._client

    # -- generation -------------------------------------------------------------

    def generate(
        self,
        *,
        user_prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a completion, retrying only genuinely transient failures."""
        target_model = model or self._s.model

        retryer = Retrying(
            stop=stop_after_attempt(self._s.max_retries + 1),
            wait=wait_exponential_jitter(
                initial=self._s.retry_initial_backoff_s,
                max=self._s.retry_max_backoff_s,
            ),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        )

        for attempt in retryer:
            with attempt:
                try:
                    return self._generate_once(
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        model=target_model,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    )
                except ProviderRateLimitError as exc:
                    # Prefer the server's own backoff hint over our exponential guess.
                    if exc.retry_after_s:
                        capped = min(exc.retry_after_s, self._s.retry_max_backoff_s)
                        log.warning(
                            "gemini.rate_limited",
                            model=target_model,
                            sleeping_s=capped,
                            attempt=attempt.retry_state.attempt_number,
                        )
                        time.sleep(capped)
                    raise

        raise ProviderResponseError("Retry loop exited without a result.")  # pragma: no cover

    def _generate_once(
        self,
        *,
        user_prompt: str,
        system_prompt: str | None,
        model: str,
        temperature: float | None,
        max_output_tokens: int | None,
    ) -> LLMResponse:
        client = self._get_client()
        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self._s.temperature if temperature is None else temperature,
            max_output_tokens=(
                self._s.max_output_tokens if max_output_tokens is None else max_output_tokens
            ),
        )

        started = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=model, contents=user_prompt, config=config
            )
        except genai_errors.APIError as exc:
            raise _map_api_error(exc) from exc
        except httpx.HTTPError as exc:
            raise _map_transport_error(exc) from exc
        latency_ms = (time.perf_counter() - started) * 1000

        return self._to_response(response, model=model, latency_ms=latency_ms)

    @staticmethod
    def _to_response(response: Any, *, model: str, latency_ms: float) -> LLMResponse:
        """Convert an SDK response, treating "blocked" as an error rather than empty text.

        Accessing ``response.text`` naively is a known trap: when generation is stopped
        by a safety filter there is no candidate, and the property raises or returns
        ``None``. Silently turning that into an empty answer would be the worst outcome
        for a medical assistant, because the user cannot tell refusal from "no
        information found". So the refusal is detected explicitly and raised.
        """
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            feedback = getattr(response, "prompt_feedback", None)
            reason = str(getattr(feedback, "block_reason", None) or "unknown")
            raise ProviderBlockedError(
                f"Gemini returned no candidates (block_reason={reason}).",
                finish_reason=reason,
            )

        raw_finish = getattr(candidates[0], "finish_reason", None)
        finish_reason = getattr(raw_finish, "name", None) or (
            str(raw_finish) if raw_finish is not None else None
        )
        if finish_reason and finish_reason.upper() in _BLOCKING_FINISH_REASONS:
            raise ProviderBlockedError(
                f"Gemini stopped generation (finish_reason={finish_reason}).",
                finish_reason=finish_reason,
            )

        text = getattr(response, "text", None)
        if not text:
            raise ProviderResponseError(
                f"Gemini returned an empty response (finish_reason={finish_reason})."
            )

        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            text=text,
            model=model,
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            completion_tokens=getattr(usage, "candidates_token_count", None),
            total_tokens=getattr(usage, "total_token_count", None),
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            truncated=bool(finish_reason and finish_reason.upper() == "MAX_TOKENS"),
        )

    # -- diagnostics ------------------------------------------------------------

    def list_models(self) -> list[str]:
        """Model ids this API key can use for text generation.

        Filtered by ``supported_actions`` so embedding-only and legacy models do not
        appear as generation candidates.
        """
        client = self._get_client()
        try:
            models: list[str] = []
            for entry in client.models.list():
                name = str(getattr(entry, "name", "") or "")
                if not name:
                    continue
                actions = getattr(entry, "supported_actions", None)
                if actions and "generateContent" not in actions:
                    continue
                models.append(name.removeprefix("models/"))
            return sorted(models)
        except genai_errors.APIError as exc:
            raise _map_api_error(exc) from exc
        except httpx.HTTPError as exc:
            raise _map_transport_error(exc) from exc

    def health(self, *, smoke_test: bool = False) -> ProviderHealth:
        """Report configuration and live availability without ever raising.

        When ``smoke_test`` is true this spends one real generation call (a handful of
        tokens) to prove the key can generate, not merely list. Listing models and
        generating are separately permissioned, so the cheap check can pass while the
        expensive one fails.
        """
        if not self._s.configured:
            return ProviderHealth(
                provider=self.name,
                configured=False,
                model=self._s.model,
                error="GEMINI_API_KEY is not set.",
            )

        started = time.perf_counter()
        try:
            available = self.list_models()
        except ProviderError as exc:
            return ProviderHealth(
                provider=self.name,
                configured=True,
                reachable=False,
                model=self._s.model,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=exc.detail,
            )

        latency_ms = (time.perf_counter() - started) * 1000
        model_available = self._s.model in available

        error: str | None = None
        if not model_available:
            error = (
                f"Configured model {self._s.model!r} is not available to this API key. "
                f"Fallback {self._s.fallback_model!r} is "
                f"{'available' if self._s.fallback_model in available else 'also unavailable'}."
            )
        elif smoke_test:
            try:
                self.generate(
                    user_prompt="Reply with the single word: ok",
                    system_prompt="You are a health check. Answer in one word.",
                    max_output_tokens=64,
                )
            except ProviderError as exc:
                error = f"Model listed but generation failed: {exc.detail}"

        return ProviderHealth(
            provider=self.name,
            configured=True,
            reachable=True,
            model=self._s.model,
            model_available=model_available,
            available_models=available,
            latency_ms=latency_ms,
            error=error,
        )
