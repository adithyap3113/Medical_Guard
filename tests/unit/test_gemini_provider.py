"""Gemini provider error-handling tests.

No network access. Errors are constructed with the SDK's real ``APIError`` signature
``(code, response_json)`` so these tests exercise the same objects production sees,
rather than a hand-rolled approximation that could drift from the SDK.

The behaviour under test is the part that is easy to get wrong and expensive in
production: **which failures are worth retrying.** HTTP 429 means two different
things, and treating them identically either wastes the daily quota or gives up on a
recoverable blip.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from google.genai import errors as genai_errors

from medguard.config import GeminiSettings
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
from medguard.providers.base import LLMResponse, ProviderHealth
from medguard.providers.gemini import GeminiProvider, _map_api_error, _map_transport_error


def api_error(code: int, message: str, status: str = "") -> genai_errors.APIError:
    """Build a real SDK error the way the transport layer would."""
    cls = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return cls(code, {"error": {"message": message, "status": status}})


class TestErrorMapping:
    def test_401_maps_to_auth_error(self) -> None:
        mapped = _map_api_error(api_error(401, "Unauthenticated", "UNAUTHENTICATED"))
        assert isinstance(mapped, ProviderAuthError)
        assert mapped.retryable is False

    def test_403_maps_to_auth_error(self) -> None:
        assert isinstance(
            _map_api_error(api_error(403, "Permission denied", "PERMISSION_DENIED")),
            ProviderAuthError,
        )

    def test_invalid_key_400_maps_to_auth_error(self) -> None:
        """Google returns 400, not 401, for a malformed key -- a real trap."""
        mapped = _map_api_error(api_error(400, "API key not valid.", "INVALID_ARGUMENT"))
        assert isinstance(mapped, ProviderAuthError)

    def test_404_maps_to_model_not_found(self) -> None:
        mapped = _map_api_error(api_error(404, "models/gemini-9 is not found", "NOT_FOUND"))
        assert isinstance(mapped, ProviderModelNotFoundError)
        assert mapped.retryable is False

    def test_429_rate_limit_is_retryable(self) -> None:
        mapped = _map_api_error(
            api_error(429, "Resource has been exhausted (per minute).", "RESOURCE_EXHAUSTED")
        )
        assert isinstance(mapped, ProviderRateLimitError)
        assert mapped.retryable is True

    @pytest.mark.parametrize(
        "message",
        [
            "Quota exceeded for quota metric 'Generate requests per day'",
            "You exceeded your current quota, please check your plan (free_tier).",
            "GenerateRequestsPerDayPerProjectPerModel limit reached",
            "Daily limit exceeded",
        ],
    )
    def test_429_hard_quota_is_not_retryable(self, message: str) -> None:
        """Retrying a daily-quota failure burns the remaining budget for nothing."""
        mapped = _map_api_error(api_error(429, message, "RESOURCE_EXHAUSTED"))
        assert isinstance(mapped, ProviderQuotaError)
        assert mapped.retryable is False

    @pytest.mark.parametrize("code", [500, 502, 503])
    def test_5xx_is_retryable(self, code: int) -> None:
        mapped = _map_api_error(api_error(code, "The model is overloaded", "UNAVAILABLE"))
        assert isinstance(mapped, ProviderUnavailableError)
        assert mapped.retryable is True

    def test_504_maps_to_timeout(self) -> None:
        mapped = _map_api_error(api_error(504, "Deadline exceeded", "DEADLINE_EXCEEDED"))
        assert isinstance(mapped, ProviderTimeoutError)
        assert mapped.retryable is True

    def test_unknown_4xx_is_not_retryable(self) -> None:
        """Our request was wrong; repeating it verbatim cannot help."""
        mapped = _map_api_error(api_error(422, "Unprocessable", "INVALID_ARGUMENT"))
        assert isinstance(mapped, ProviderError)
        assert mapped.retryable is False

    def test_retry_after_hint_is_extracted(self) -> None:
        """Honouring the server's own backoff beats guessing exponentially."""
        exc = genai_errors.ClientError(
            429,
            {
                "error": {
                    "message": "rate limited",
                    "status": "RESOURCE_EXHAUSTED",
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.RetryInfo",
                            "retryDelay": "38s",
                        }
                    ],
                }
            },
        )
        mapped = _map_api_error(exc)
        assert isinstance(mapped, ProviderRateLimitError)
        assert mapped.retry_after_s == 38.0

    def test_timeout_transport_error_maps_to_timeout(self) -> None:
        mapped = _map_transport_error(httpx.ReadTimeout("timed out"))
        assert isinstance(mapped, ProviderTimeoutError)
        assert mapped.retryable is True

    def test_connection_error_maps_to_unavailable(self) -> None:
        mapped = _map_transport_error(httpx.ConnectError("connection refused"))
        assert isinstance(mapped, ProviderUnavailableError)
        assert mapped.retryable is True


class TestErrorMessageSafety:
    def test_user_messages_never_expose_internals(self) -> None:
        """The client-facing string must not carry status codes or vendor payloads."""
        mapped = _map_api_error(
            api_error(401, "API key AIzaSyLEAKED_KEY_VALUE_123456 is invalid", "UNAUTHENTICATED")
        )
        assert "AIzaSyLEAKED" not in mapped.user_message
        assert "401" not in mapped.user_message

    def test_technical_detail_is_kept_for_logs(self) -> None:
        """The detail must stay useful; the logging layer scrubs secrets from it."""
        mapped = _map_api_error(api_error(503, "backend overloaded", "UNAVAILABLE"))
        assert "503" in mapped.detail
        assert "overloaded" in mapped.detail

    def test_quota_message_tells_the_user_what_to_do(self) -> None:
        mapped = _map_api_error(api_error(429, "quota exceeded per day", "RESOURCE_EXHAUSTED"))
        assert "quota" in mapped.user_message.lower()


class _FakeUsage:
    prompt_token_count = 120
    candidates_token_count = 45
    total_token_count = 165


class _FakeCandidate:
    def __init__(self, finish_reason: str = "STOP") -> None:
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(
        self, text: str | None = "answer", finish_reason: str = "STOP", candidates: Any = None
    ) -> None:
        self.text = text
        self.candidates = [_FakeCandidate(finish_reason)] if candidates is None else candidates
        self.usage_metadata = _FakeUsage()
        self.prompt_feedback = None


class TestResponseConversion:
    def test_successful_response_carries_token_and_latency_metadata(self) -> None:
        result = GeminiProvider._to_response(_FakeResponse(), model="m", latency_ms=123.4)
        assert isinstance(result, LLMResponse)
        assert (result.prompt_tokens, result.completion_tokens) == (120, 45)
        assert result.latency_ms == 123.4
        assert result.truncated is False

    @pytest.mark.parametrize(
        "reason", ["SAFETY", "PROHIBITED_CONTENT", "RECITATION", "BLOCKLIST", "SPII"]
    )
    def test_safety_stop_raises_rather_than_returning_empty(self, reason: str) -> None:
        """A refusal must be distinguishable from "no information found".

        Silently returning empty text would let a safety block masquerade as an
        absence of evidence -- the worst possible confusion for a medical assistant.
        """
        with pytest.raises(ProviderBlockedError) as exc_info:
            GeminiProvider._to_response(
                _FakeResponse(text=None, finish_reason=reason), model="m", latency_ms=1.0
            )
        assert exc_info.value.finish_reason == reason
        assert exc_info.value.retryable is False

    def test_no_candidates_raises_blocked(self) -> None:
        """The known trap: `.text` on a blocked response has no candidate behind it."""
        with pytest.raises(ProviderBlockedError):
            GeminiProvider._to_response(_FakeResponse(candidates=[]), model="m", latency_ms=1.0)

    def test_empty_text_raises_response_error(self) -> None:
        with pytest.raises(ProviderResponseError):
            GeminiProvider._to_response(_FakeResponse(text=""), model="m", latency_ms=1.0)

    def test_max_tokens_is_flagged_as_truncated(self) -> None:
        """A truncated answer can lose its closing citations and look unsupported."""
        result = GeminiProvider._to_response(
            _FakeResponse(finish_reason="MAX_TOKENS"), model="m", latency_ms=1.0
        )
        assert result.truncated is True


def _provider(**overrides: Any) -> GeminiProvider:
    defaults: dict[str, Any] = {
        "api_key": "AIzaTEST_key_for_unit_tests_000000000",
        "max_retries": 2,
        "retry_initial_backoff_s": 0.001,
        "retry_max_backoff_s": 0.002,
    }
    return GeminiProvider(GeminiSettings(**{**defaults, **overrides}))


class TestRetryBehaviour:
    def test_transient_failure_is_retried_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        def flaky(**_: Any) -> LLMResponse:
            calls["n"] += 1
            if calls["n"] < 3:
                raise ProviderUnavailableError("upstream 503")
            return LLMResponse(text="ok", model="m", latency_ms=1.0)

        provider = _provider()
        monkeypatch.setattr(provider, "_generate_once", flaky)
        assert provider.generate(user_prompt="q").text == "ok"
        assert calls["n"] == 3

    def test_auth_failure_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A bad key fails identically every time; retrying only delays the truth."""
        calls = {"n": 0}

        def always_auth_error(**_: Any) -> LLMResponse:
            calls["n"] += 1
            raise ProviderAuthError("bad key")

        provider = _provider()
        monkeypatch.setattr(provider, "_generate_once", always_auth_error)
        with pytest.raises(ProviderAuthError):
            provider.generate(user_prompt="q")
        assert calls["n"] == 1

    def test_hard_quota_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}

        def quota(**_: Any) -> LLMResponse:
            calls["n"] += 1
            raise ProviderQuotaError("daily quota exhausted")

        provider = _provider()
        monkeypatch.setattr(provider, "_generate_once", quota)
        with pytest.raises(ProviderQuotaError):
            provider.generate(user_prompt="q")
        assert calls["n"] == 1

    def test_retries_are_bounded_and_the_error_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        def always_down(**_: Any) -> LLMResponse:
            calls["n"] += 1
            raise ProviderUnavailableError("still down")

        provider = _provider(max_retries=2)
        monkeypatch.setattr(provider, "_generate_once", always_down)
        with pytest.raises(ProviderUnavailableError):
            provider.generate(user_prompt="q")
        assert calls["n"] == 3, "should attempt once plus max_retries"

    def test_zero_retries_attempts_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}

        def down(**_: Any) -> LLMResponse:
            calls["n"] += 1
            raise ProviderUnavailableError("down")

        provider = _provider(max_retries=0)
        monkeypatch.setattr(provider, "_generate_once", down)
        with pytest.raises(ProviderUnavailableError):
            provider.generate(user_prompt="q")
        assert calls["n"] == 1


class TestUnconfiguredProvider:
    def test_health_reports_unconfigured_without_network(self) -> None:
        health = GeminiProvider(GeminiSettings(api_key=None)).health()
        assert isinstance(health, ProviderHealth)
        assert health.configured is False
        assert health.reachable is False
        assert health.ok is False

    def test_generate_without_key_raises_auth_error(self) -> None:
        with pytest.raises(ProviderAuthError):
            GeminiProvider(GeminiSettings(api_key=None)).generate(user_prompt="q")

    def test_health_never_raises_when_listing_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Diagnostics report failure; they do not abort the doctor run."""
        provider = _provider()
        monkeypatch.setattr(
            provider, "list_models", lambda: (_ for _ in ()).throw(ProviderAuthError("nope"))
        )
        health = provider.health()
        assert health.configured is True
        assert health.reachable is False
        assert health.error is not None
