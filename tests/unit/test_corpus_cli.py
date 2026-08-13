"""CLI exit codes and safety behaviour.

Exit codes are the contract these commands have with CI. `verify` returning 0 on a
corrupt corpus would make every pipeline that depends on it useless, and the failure
would be invisible -- a green build over a broken corpus.

`sync` gets its own attention because it writes the committed provenance record. The
property that matters is that it never overwrites an existing entry: an entry that
disagrees with its file is an integrity failure for `verify` to report, not something
to paper over.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from typer.testing import CliRunner

from medguard.cli.app import app
from medguard.config import Settings

runner = CliRunner()

SOURCES = """
schema_version: 1
documents:
  - document_id: test_guideline_2024
    title: A Test Clinical Guideline, 2024
    source_url: https://example.invalid/guideline.pdf
    source_organization: CDC
    document_type: guideline
    publication_date: 2024-01-15
    local_filename: test_guideline_2024.pdf
    license_or_usage_note: Public domain test fixture, stated by the publisher.
"""


def _manifest_payload(pdf: bytes, **overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "document_id": "test_guideline_2024",
        "title": "A Test Clinical Guideline, 2024",
        "source_url": "https://example.invalid/guideline.pdf",
        "source_organization": "CDC",
        "document_type": "guideline",
        "publication_date": "2024-01-15",
        "license_or_usage_note": "Public domain test fixture, stated by the publisher.",
        "local_filename": "test_guideline_2024.pdf",
        "sha256": hashlib.sha256(pdf).hexdigest(),
        "file_size_bytes": len(pdf),
        "page_count": None,
        "retrieved_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
        "version_or_revision": None,
    }
    entry.update(overrides)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "documents": [entry],
    }


class TestVerifyExitCodes:
    def test_a_clean_corpus_exits_zero(self, settings: Settings, pdf_bytes: bytes) -> None:
        (settings.corpus.raw_dir / "test_guideline_2024.pdf").write_bytes(pdf_bytes)
        settings.corpus.manifest_path.write_text(
            json.dumps(_manifest_payload(pdf_bytes)), encoding="utf-8"
        )

        result = runner.invoke(app, ["corpus", "verify"])
        assert result.exit_code == 0

    def test_a_corrupt_file_exits_one(self, settings: Settings, pdf_bytes: bytes) -> None:
        """The CI gate. A green build over a corrupt corpus is worse than no gate."""
        (settings.corpus.raw_dir / "test_guideline_2024.pdf").write_bytes(
            b"%PDF-1.7\n" + b"tampered\n" * 900
        )
        settings.corpus.manifest_path.write_text(
            json.dumps(_manifest_payload(pdf_bytes)), encoding="utf-8"
        )

        result = runner.invoke(app, ["corpus", "verify"])
        assert result.exit_code == 1

    def test_a_missing_manifest_exits_one(self, settings: Settings) -> None:
        result = runner.invoke(app, ["corpus", "verify"])
        assert result.exit_code == 1

    def test_a_missing_file_exits_one(self, settings: Settings, pdf_bytes: bytes) -> None:
        """Phase 3A raised this from a warning; run `corpus download` before `verify`."""
        settings.corpus.manifest_path.write_text(
            json.dumps(_manifest_payload(pdf_bytes)), encoding="utf-8"
        )
        result = runner.invoke(app, ["corpus", "verify"])
        assert result.exit_code == 1

    def test_a_null_publication_date_does_not_fail_the_build(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        """Correct under "never invent metadata"; nine committed documents have one."""
        (settings.corpus.raw_dir / "test_guideline_2024.pdf").write_bytes(pdf_bytes)
        settings.corpus.manifest_path.write_text(
            json.dumps(_manifest_payload(pdf_bytes, publication_date=None)), encoding="utf-8"
        )
        result = runner.invoke(app, ["corpus", "verify"])
        assert result.exit_code == 0

    def test_verbose_runs_without_a_database(self, settings: Settings, pdf_bytes: bytes) -> None:
        """The freshness comparison is optional enrichment, never a reason to fail."""
        (settings.corpus.raw_dir / "test_guideline_2024.pdf").write_bytes(pdf_bytes)
        settings.corpus.manifest_path.write_text(
            json.dumps(_manifest_payload(pdf_bytes)), encoding="utf-8"
        )
        result = runner.invoke(app, ["corpus", "verify", "--verbose"])
        assert result.exit_code == 0

    def test_a_malformed_manifest_exits_one_without_a_traceback(self, settings: Settings) -> None:
        settings.corpus.manifest_path.write_text("{not json", encoding="utf-8")
        result = runner.invoke(app, ["corpus", "verify"])
        assert result.exit_code == 1
        assert result.exception is None or isinstance(result.exception, SystemExit)


class TestDownloadExitCodes:
    def test_a_missing_sources_file_exits_one(self, settings: Settings) -> None:
        result = runner.invoke(app, ["corpus", "download"])
        assert result.exit_code == 1

    def test_a_failing_download_exits_one(self, settings: Settings) -> None:
        """The Phase 1 script always exited 0, which made it useless in CI."""
        settings.corpus.sources_path.write_text(SOURCES, encoding="utf-8")
        result = runner.invoke(
            app, ["corpus", "download", "--no-write-manifest"], catch_exceptions=True
        )
        # example.invalid does not resolve, so the download fails and the command says so.
        assert result.exit_code == 1


class TestSync:
    def test_adopts_an_unrecorded_file(self, settings: Settings, pdf_bytes: bytes) -> None:
        settings.corpus.sources_path.write_text(SOURCES, encoding="utf-8")
        (settings.corpus.raw_dir / "test_guideline_2024.pdf").write_bytes(pdf_bytes)

        result = runner.invoke(app, ["corpus", "sync", "--yes"])
        assert result.exit_code == 0

        payload = json.loads(settings.corpus.manifest_path.read_text(encoding="utf-8"))
        assert len(payload["documents"]) == 1
        assert payload["documents"][0]["sha256"] == hashlib.sha256(pdf_bytes).hexdigest()

    def test_refuses_to_overwrite_an_existing_entry(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        """A wrong hash is an integrity failure to report, not one to silently fix."""
        settings.corpus.sources_path.write_text(SOURCES, encoding="utf-8")
        (settings.corpus.raw_dir / "test_guideline_2024.pdf").write_bytes(pdf_bytes)
        stale = _manifest_payload(pdf_bytes, sha256="f" * 64)
        settings.corpus.manifest_path.write_text(json.dumps(stale), encoding="utf-8")

        result = runner.invoke(app, ["corpus", "sync", "--yes"])
        assert result.exit_code == 0

        payload = json.loads(settings.corpus.manifest_path.read_text(encoding="utf-8"))
        assert payload["documents"][0]["sha256"] == "f" * 64

    def test_will_not_invent_provenance_for_an_unknown_file(
        self,
        settings: Settings,
        pdf_bytes: bytes,
    ) -> None:
        """A PDF with no sources.yaml entry has no verifiable origin or licence."""
        settings.corpus.sources_path.write_text(SOURCES, encoding="utf-8")
        (settings.corpus.raw_dir / "mystery_document.pdf").write_bytes(pdf_bytes)

        result = runner.invoke(app, ["corpus", "sync", "--yes"])
        assert result.exit_code == 0
        assert not settings.corpus.manifest_path.exists()

    def test_nothing_to_add_is_not_a_failure(self, settings: Settings) -> None:
        settings.corpus.sources_path.write_text(SOURCES, encoding="utf-8")
        result = runner.invoke(app, ["corpus", "sync", "--yes"])
        assert result.exit_code == 0

    def test_declining_the_prompt_records_nothing(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        settings.corpus.sources_path.write_text(SOURCES, encoding="utf-8")
        (settings.corpus.raw_dir / "test_guideline_2024.pdf").write_bytes(pdf_bytes)

        result = runner.invoke(app, ["corpus", "sync"], input="n\n")
        assert result.exit_code == 0
        assert not settings.corpus.manifest_path.exists()


class TestSyncFlags:
    """Both modes are reachable, and they do not bleed into one another."""

    def test_the_backfill_flags_are_documented(self) -> None:
        result = runner.invoke(app, ["corpus", "sync", "--help"])
        assert result.exit_code == 0
        assert "--backfill-page-counts" in result.stdout
        assert "--backfill-publication-dates" in result.stdout

    def test_the_help_text_states_the_never_overwrites_rule(self) -> None:
        """The widened contract still has to advertise its limit."""
        result = runner.invoke(app, ["corpus", "sync", "--help"])
        assert "Never overwrites" in result.stdout

    def test_a_backfill_flag_does_not_adopt_files(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        """The two modes are exclusive: backfill must not quietly also record new files."""
        settings.corpus.sources_path.write_text(SOURCES, encoding="utf-8")
        (settings.corpus.raw_dir / "test_guideline_2024.pdf").write_bytes(pdf_bytes)
        settings.corpus.manifest_path.write_text(
            json.dumps(
                {"schema_version": 1, "generated_at": "2026-01-01T00:00:00Z", "documents": []}
            ),
            encoding="utf-8",
        )

        runner.invoke(app, ["corpus", "sync", "--backfill-page-counts", "--yes"])

        payload = json.loads(settings.corpus.manifest_path.read_text(encoding="utf-8"))
        assert payload["documents"] == []

    def test_plain_sync_does_not_backfill(self, settings: Settings, pdf_bytes: bytes) -> None:
        settings.corpus.sources_path.write_text(SOURCES, encoding="utf-8")
        (settings.corpus.raw_dir / "test_guideline_2024.pdf").write_bytes(pdf_bytes)

        assert runner.invoke(app, ["corpus", "sync", "--yes"]).exit_code == 0
        payload = json.loads(settings.corpus.manifest_path.read_text(encoding="utf-8"))
        assert payload["documents"][0]["page_count"] is None


class TestCommandRegistration:
    def test_corpus_and_db_are_registered(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "corpus" in result.stdout
        assert "db" in result.stdout

    def test_db_revision_requires_a_message(self, settings: Settings) -> None:
        """An unnamed migration is unreadable exactly when it matters."""
        result = runner.invoke(app, ["db", "revision"])
        assert result.exit_code != 0

    def test_db_downgrade_asks_before_dropping_tables(self, settings: Settings) -> None:
        result = runner.invoke(app, ["db", "downgrade", "base"], input="n\n")
        assert result.exit_code == 1
