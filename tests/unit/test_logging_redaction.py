"""Secret-redaction tests.

These are the security tests of Phase 1. The guarantee under test is:
**no configured credential can reach a log sink**, regardless of which field it is
placed in, whether it is nested, or whether it arrives inside a traceback.

Each test names the leak path it defends against, because a redaction test that only
checks the obvious case gives false confidence.
"""

from __future__ import annotations

import logging

import pytest
import structlog
from pydantic import SecretStr

from medguard.logging import (
    REDACTED,
    configure_logging,
    get_logger,
    redact_processor,
    redact_value,
    register_secrets,
)

GEMINI_KEY = "AIzaSyD_fake_key_for_tests_0123456789abcdef"
LANGFUSE_SECRET = "sk-lf-deadbeef-1111-2222-3333-444455556666"


class TestKeyNameRedaction:
    """Defence 1: any field whose *name* looks sensitive is dropped."""

    @pytest.mark.parametrize(
        "key",
        [
            "api_key",
            "API_KEY",
            "gemini_api_key",
            "secret",
            "secret_key",
            "password",
            "authorization",
            "access_token",
            "credential",
            "private_key",
            "cookie",
        ],
    )
    def test_sensitive_key_names_are_dropped(self, key: str) -> None:
        assert redact_value("whatever-the-value-is", key=key) == REDACTED

    def test_ordinary_key_names_pass_through(self) -> None:
        """Over-redaction destroys debuggability, so this must not be indiscriminate."""
        assert redact_value("gemini-2.5-flash-lite", key="model") == "gemini-2.5-flash-lite"
        assert redact_value(42, key="top_k") == 42

    def test_nested_structures_are_walked(self) -> None:
        """A secret one level down is the common real-world leak."""
        payload = {"config": {"gemini": {"api_key": GEMINI_KEY, "model": "flash"}}}
        result = redact_value(payload)
        assert GEMINI_KEY not in str(result)
        assert result["config"]["gemini"]["model"] == "flash"

    def test_secrets_inside_lists_are_redacted(self) -> None:
        register_secrets([GEMINI_KEY])
        result = redact_value({"headers": [f"Authorization: Bearer {GEMINI_KEY}"]})
        assert GEMINI_KEY not in str(result)


class TestValuePatternRedaction:
    """Defence 2: credential-shaped values are caught under innocent field names."""

    def test_google_key_in_free_text_is_redacted(self) -> None:
        text = f"request failed for key {GEMINI_KEY} at endpoint /v1/models"
        assert GEMINI_KEY not in redact_value(text, key="message")

    def test_langfuse_key_is_redacted(self) -> None:
        assert LANGFUSE_SECRET not in redact_value(f"auth={LANGFUSE_SECRET}", key="note")

    def test_bearer_token_is_redacted(self) -> None:
        out = redact_value("Authorization: Bearer abcdef0123456789XYZ", key="header_line")
        assert "abcdef0123456789XYZ" not in out

    def test_key_in_url_query_string_is_redacted(self) -> None:
        """The classic leak: an SDK logs the request URL with the key in the query."""
        url = f"https://generativelanguage.googleapis.com/v1/models?key={GEMINI_KEY}"
        out = redact_value(url, key="url")
        assert GEMINI_KEY not in out
        # The structure survives, so the log line is still diagnostically useful.
        assert "generativelanguage.googleapis.com" in out

    def test_database_password_in_url_is_redacted(self) -> None:
        out = redact_value("postgresql+psycopg://medguard:hunter2pass@db:5432/medguard", key="dsn")
        assert "hunter2pass" not in out
        assert "db:5432" in out


class TestKnownLiteralRedaction:
    """Defence 3: the actual configured values, whatever shape they take."""

    def test_registered_secret_is_scrubbed_anywhere(self) -> None:
        odd_key = "zzz-custom-format-credential-abc123456789"
        register_secrets([odd_key])
        assert odd_key not in redact_value(f"connecting with {odd_key}", key="msg")

    def test_secretstr_is_accepted_for_registration(self) -> None:
        value = "another-secret-value-9876543210"
        register_secrets([SecretStr(value)])
        assert value not in redact_value(f"token={value}", key="msg")

    def test_short_values_are_not_registered(self) -> None:
        """Substring-replacing a short string would corrupt unrelated log text."""
        register_secrets(["abc"])
        assert redact_value("abc appears in abcdefg", key="msg") == "abc appears in abcdefg"

    def test_secretstr_value_is_never_rendered(self) -> None:
        assert redact_value(SecretStr(GEMINI_KEY)) == REDACTED


class TestProcessorIntegration:
    def test_processor_scrubs_the_whole_event_dict(self) -> None:
        register_secrets([GEMINI_KEY])
        event = {
            "event": f"calling gemini with {GEMINI_KEY}",
            "api_key": GEMINI_KEY,
            "model": "gemini-2.5-flash-lite",
            "nested": {"secret_key": LANGFUSE_SECRET},
        }
        out = redact_processor(None, "info", event)
        rendered = str(out)
        assert GEMINI_KEY not in rendered
        assert LANGFUSE_SECRET not in rendered
        assert out["model"] == "gemini-2.5-flash-lite"


class TestEndToEndLogging:
    """The guarantee that actually matters: nothing reaches the emitted output."""

    def test_configured_secret_never_appears_in_log_output(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging(level="DEBUG", fmt="json", secrets=[GEMINI_KEY, LANGFUSE_SECRET])
        log = get_logger("test")

        log.info("provider.call", api_key=GEMINI_KEY, url=f"https://x.test/v1?key={GEMINI_KEY}")
        log.warning("tracing.init", secret_key=LANGFUSE_SECRET)

        captured = capsys.readouterr().err
        assert GEMINI_KEY not in captured
        assert LANGFUSE_SECRET not in captured
        assert "provider.call" in captured

    def test_exceptions_are_scrubbed_but_remain_useful(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Tracebacks are a leak path; scrubbing must not destroy debuggability."""
        configure_logging(level="DEBUG", fmt="console", secrets=[GEMINI_KEY])
        log = get_logger("test")

        try:
            raise ValueError(f"auth failed for key={GEMINI_KEY}")
        except ValueError:
            log.exception("provider.failed", model="gemini-2.5-flash-lite")

        captured = capsys.readouterr().err
        assert GEMINI_KEY not in captured
        # Still diagnosable: error type, our message, and the redaction marker remain.
        assert "ValueError" in captured
        assert "provider.failed" in captured
        assert REDACTED in captured

    def test_third_party_stdlib_logs_are_also_redacted(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The point of routing stdlib logging through ProcessorFormatter.

        An SDK we do not control must not be able to emit an unredacted line.
        """
        configure_logging(level="DEBUG", fmt="json", secrets=[GEMINI_KEY])
        logging.getLogger("some_third_party_sdk").warning(
            "requesting https://api.test/v1?key=%s", GEMINI_KEY
        )
        captured = capsys.readouterr().err
        assert GEMINI_KEY not in captured

    def test_json_format_emits_parseable_lines(self, capsys: pytest.CaptureFixture[str]) -> None:
        import json

        configure_logging(level="INFO", fmt="json", secrets=[])
        get_logger("test").info("structured.event", chunk_count=7)

        line = capsys.readouterr().err.strip().splitlines()[-1]
        parsed = json.loads(line)
        assert parsed["event"] == "structured.event"
        assert parsed["chunk_count"] == 7
        assert parsed["level"] == "info"

    def test_database_password_in_exception_is_redacted(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        raw_password = "superSecretDbPassword987!"
        register_secrets([raw_password])
        configure_logging(level="DEBUG", fmt="console", secrets=[raw_password])
        log = get_logger("test.db")

        try:
            raise ConnectionError(
                f"Failed to connect to postgresql://medguard:{raw_password}@db.internal:5432/medguard"
            )
        except ConnectionError:
            log.exception("db.connect_failed")

        captured = capsys.readouterr().err
        assert raw_password not in captured
        assert REDACTED in captured


@pytest.fixture(autouse=True)
def _reset_structlog() -> None:
    """Undo global structlog state so configuration tests do not bleed into others."""
    structlog.reset_defaults()
