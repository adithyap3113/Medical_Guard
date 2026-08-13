"""Diagnostics and CLI tests.

``medguard doctor`` is the Phase 1 acceptance test, so it needs its own tests. Two
invariants matter most:

* it must **never raise**, whatever is broken -- otherwise it cannot report on
  anything past the first failure;
* it must **never print a secret**, since it is the command people paste into bug
  reports and screenshots.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from medguard.cli.app import app
from medguard.config import Settings, get_settings
from medguard.diagnostics import (
    Check,
    CheckStatus,
    DiagnosticReport,
    check_embedding_config,
    check_gemini_key,
    check_gemini_models,
    check_python,
    run_diagnostics,
)

runner = CliRunner()


class TestIndividualChecks:
    def test_python_version_check_passes_and_is_critical(self) -> None:
        check = check_python()
        assert check.status is CheckStatus.OK
        assert check.critical is True

    def test_missing_gemini_key_warns_rather_than_fails(self, settings: Settings) -> None:
        """No key is a legitimate state early on; a red light would train people to
        ignore the doctor output."""
        check = check_gemini_key(settings)
        assert check.status is CheckStatus.WARN
        assert "aistudio.google.com" in check.detail

    def test_present_gemini_key_is_reported_without_its_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        secret = "AIzaDOCTOR_leak_check_00000000000000"
        monkeypatch.setenv("GEMINI_API_KEY", secret)
        get_settings.cache_clear()

        check = check_gemini_key(get_settings())
        assert check.status is CheckStatus.OK
        assert secret not in check.detail

    def test_model_checks_are_skipped_without_a_key(self, settings: Settings) -> None:
        """No key must mean no network call, so the suite stays offline."""
        checks = check_gemini_models(settings)
        assert all(c.status is CheckStatus.SKIP for c in checks)

    def test_embedding_check_confirms_nothing_is_loaded(self, settings: Settings) -> None:
        """Phase 1 acceptance: no model weights are downloaded."""
        check = check_embedding_config(settings)
        assert check.status is CheckStatus.SKIP
        assert "not loaded" in check.detail
        assert settings.collection_name in check.detail


class TestReport:
    def test_report_is_ok_when_only_non_critical_checks_fail(self) -> None:
        report = DiagnosticReport(
            checks=[
                Check(name="Python", status=CheckStatus.OK, critical=True),
                Check(name="Gemini models", status=CheckStatus.FAIL, critical=False),
            ]
        )
        assert report.ok is True
        assert report.exit_code() == 0
        assert len(report.failed) == 1

    def test_report_fails_on_a_critical_failure(self) -> None:
        report = DiagnosticReport(
            checks=[Check(name="Database", status=CheckStatus.FAIL, critical=True)]
        )
        assert report.ok is False
        assert report.exit_code() == 1


class TestRunner:
    def test_full_run_never_raises_and_covers_every_area(self, settings: Settings) -> None:
        report = run_diagnostics(settings, smoke_test=False, include_hardware=True)
        names = {c.name for c in report.checks}
        assert {"Python", "CPU", "RAM", "GPU", "Docker", "Database", "Qdrant", "Langfuse"} <= names

    def test_hardware_probes_can_be_skipped(self, settings: Settings) -> None:
        report = run_diagnostics(settings, include_hardware=False)
        assert "CPU" not in {c.name for c in report.checks}

    def test_unreachable_qdrant_warns_instead_of_crashing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:1")  # nothing listens here
        get_settings.cache_clear()

        report = run_diagnostics(get_settings(), include_hardware=False)
        qdrant = next(c for c in report.checks if c.name == "Qdrant")
        assert qdrant.status is CheckStatus.WARN
        assert "docker compose" in qdrant.detail


class TestDoctorCommand:
    def test_doctor_runs_and_reports(self) -> None:
        result = runner.invoke(app, ["doctor", "--skip-hardware"])
        assert result.exit_code == 0, result.output
        assert "MedGuard environment diagnostics" in result.output

    def test_doctor_never_prints_a_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """This is the command people paste into bug reports."""
        secret = "AIzaCLI_LEAK_CHECK_000000000000000000"
        monkeypatch.setenv("GEMINI_API_KEY", secret)
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-99999999-8888-7777-6666-555555555555")
        get_settings.cache_clear()

        result = runner.invoke(app, ["doctor", "--skip-hardware", "--show-config"])
        assert secret not in result.output
        assert "sk-lf-" not in result.output

    def test_show_config_displays_the_derived_collection_name(self) -> None:
        result = runner.invoke(app, ["doctor", "--skip-hardware", "--show-config"])
        assert "collection_name" in result.output

    def test_version_flag(self) -> None:
        from medguard import __version__

        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output


class TestUnimplementedCommands:
    """Stubs must fail loudly and say which phase implements them."""

    @pytest.mark.parametrize(
        ("argv", "phase"),
        [
            (["ingest"], "Phase 3"),
            (["query", "what is hypertension?"], "Phases 5-11"),
            (["eval"], "Phase 12"),
        ],
    )
    def test_stub_raises_not_implemented(self, argv: list[str], phase: str) -> None:
        result = runner.invoke(app, argv)
        assert result.exit_code != 0
        assert isinstance(result.exception, NotImplementedError)
        assert phase in str(result.exception)

    def test_commands_are_registered_with_real_signatures(self) -> None:
        result = runner.invoke(app, ["--help"])
        for command in ("doctor", "serve", "ingest", "query", "eval"):
            assert command in result.output
