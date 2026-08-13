"""Configuration and environment-validation tests."""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from medguard.config import GeminiSettings, Settings, get_settings


class TestGeminiSettings:
    def test_absent_key_is_valid_and_reported_unconfigured(self) -> None:
        """A missing key must not crash startup.

        The app has to boot far enough to *tell* you the key is missing; failing at
        import time would make `medguard doctor` useless on a fresh clone.
        """
        s = GeminiSettings()
        assert s.api_key is None
        assert s.configured is False

    def test_key_is_read_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaTESTKEY_not_a_real_credential_000")
        s = GeminiSettings()
        assert s.configured is True

    def test_whitespace_only_key_is_not_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "   ")
        assert GeminiSettings().configured is False

    def test_model_is_environment_driven(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The model must be swappable without touching code."""
        monkeypatch.setenv("GEMINI_MODEL", "gemini-3-something-future")
        assert GeminiSettings().model == "gemini-3-something-future"

    def test_timeout_converts_seconds_to_milliseconds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Gemini's HttpOptions.timeout is milliseconds; ours is seconds."""
        monkeypatch.setenv("GEMINI_TIMEOUT_S", "12.5")
        assert GeminiSettings().timeout_ms == 12_500

    @pytest.mark.parametrize(
        ("var", "value"),
        [
            ("GEMINI_TEMPERATURE", "2.5"),
            ("GEMINI_TEMPERATURE", "-0.1"),
            ("GEMINI_TIMEOUT_S", "0"),
            ("GEMINI_MAX_RETRIES", "-1"),
            ("GEMINI_MAX_OUTPUT_TOKENS", "8"),
        ],
    )
    def test_out_of_range_values_fail_fast(
        self, monkeypatch: pytest.MonkeyPatch, var: str, value: str
    ) -> None:
        """Bad config must fail at startup, not on the first user request."""
        monkeypatch.setenv(var, value)
        with pytest.raises(ValidationError):
            GeminiSettings()

    def test_backoff_ordering_is_enforced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_RETRY_INITIAL_BACKOFF_S", "30")
        monkeypatch.setenv("GEMINI_RETRY_MAX_BACKOFF_S", "5")
        with pytest.raises(ValidationError, match="must be >="):
            GeminiSettings()


class TestAppSettings:
    def test_sqlite_is_the_default_store(self, settings: Settings) -> None:
        assert settings.app.database_url.startswith("sqlite:")

    def test_postgres_url_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Switching to Postgres must be a config change only."""
        monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/medguard")
        get_settings.cache_clear()
        assert get_settings().app.database_url.startswith("postgresql+psycopg:")

    def test_unknown_database_scheme_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", "mysql://u:p@localhost/medguard")
        get_settings.cache_clear()
        with pytest.raises(ValidationError):
            get_settings()

    def test_log_level_is_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "debug")
        get_settings.cache_clear()
        assert get_settings().app.log_level == "DEBUG"


class TestCollectionNaming:
    def test_collection_name_encodes_model_and_dimension(self, settings: Settings) -> None:
        """Two embedding models must never share a collection.

        Mixing incompatible vector spaces silently degrades retrieval in a way that is
        extremely hard to diagnose, so the name makes the collision impossible.
        """
        name = settings.collection_name
        assert name.startswith("medguard__")
        assert "bge_small_en_v1_5" in name
        assert name.endswith("__384")

    def test_changing_the_model_changes_the_collection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        get_settings.cache_clear()
        baseline = get_settings().collection_name

        monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        monkeypatch.setenv("EMBEDDING_DIMENSION", "768")
        get_settings.cache_clear()
        assert get_settings().collection_name != baseline


class TestRedaction:
    def test_redacted_dump_hides_every_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`redacted()` is the only sanctioned way to display configuration."""
        secret = "AIzaSUPERSECRET_value_that_must_never_appear_1234"
        monkeypatch.setenv("GEMINI_API_KEY", secret)
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-11111111-2222-3333-4444-555555555555")
        get_settings.cache_clear()

        dumped = get_settings().redacted()
        assert secret not in str(dumped)
        assert "sk-lf-" not in str(dumped)
        assert dumped["gemini"]["api_key"] == "<set>"

    def test_redacted_dump_does_not_leak_key_length_or_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A prefix or length is still usable information about a credential."""
        short, long = "AIza" + "a" * 32, "AIza" + "b" * 64
        monkeypatch.setenv("GEMINI_API_KEY", short)
        get_settings.cache_clear()
        first = get_settings().redacted()["gemini"]["api_key"]

        monkeypatch.setenv("GEMINI_API_KEY", long)
        get_settings.cache_clear()
        second = get_settings().redacted()["gemini"]["api_key"]

        assert first == second == "<set>"

    def test_secret_is_not_exposed_by_repr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SecretStr must survive accidental interpolation into a log line."""
        secret = "AIzaREPR_LEAK_CHECK_00000000000000000"
        monkeypatch.setenv("GEMINI_API_KEY", secret)
        s = GeminiSettings()
        assert secret not in repr(s)
        assert secret not in str(s.api_key)
        assert isinstance(s.api_key, SecretStr)
        assert s.api_key.get_secret_value() == secret


class TestSettingsSingleton:
    def test_get_settings_is_cached(self) -> None:
        assert get_settings() is get_settings()
