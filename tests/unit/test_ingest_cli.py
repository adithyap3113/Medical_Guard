"""`medguard ingest`: exit codes, flags, and the guards that run before any work.

Exit codes are the contract with CI. A run that indexes eighteen of twenty documents and
exits 0 produces a green build over a corpus that cannot answer a third of the questions
it should, and nobody looks again.

The preflight tests matter for a different reason. Every one of them is a guard that
fires *before* the expensive part, and a guard that has silently stopped working is
indistinguishable from one that was never there -- right up until a machine with 200 MB
free starts a run that the OS kills forty minutes in.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from medguard.cli.app import app
from medguard.config import Settings
from medguard.storage.database import Database
from tests.fixtures.make_pdf import write_structured_pdf

runner = CliRunner()


@pytest.fixture
def corpus(settings: Settings) -> dict[str, Path]:
    """Two generated PDFs, a manifest describing them, and a migrated database."""
    from medguard.corpus.manifest import sha256_file

    raw = settings.corpus.raw_dir
    paths = {
        "doc_one": write_structured_pdf(raw / "doc_one.pdf", pages=3),
        "doc_two": write_structured_pdf(raw / "doc_two.pdf", pages=2),
    }

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "documents": [
            {
                "document_id": document_id,
                "title": f"Title for {document_id}",
                "source_url": f"https://example.invalid/{document_id}.pdf",
                "source_organization": "CDC",
                "document_type": "guideline",
                "publication_date": "2024-01-15",
                "license_or_usage_note": "Public domain test fixture.",
                "local_filename": path.name,
                "sha256": sha256_file(path),
                "file_size_bytes": path.stat().st_size,
                "page_count": None,
                "retrieved_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
                "version_or_revision": None,
            }
            for document_id, path in sorted(paths.items())
        ],
    }
    settings.corpus.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    database = Database(settings)
    database.create_all()
    database.dispose()
    return paths


class TestRegistration:
    def test_ingest_is_a_top_level_command(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "ingest" in result.stdout

    def test_the_flags_are_documented(self) -> None:
        result = runner.invoke(app, ["ingest", "--help"])
        assert result.exit_code == 0
        for flag in ("--dry-run", "--only", "--limit", "--force-recreate"):
            assert flag in result.stdout

    def test_force_recreate_advertises_that_it_is_destructive(self) -> None:
        """A flag that deletes an hour of work should say so where it is read."""
        result = runner.invoke(app, ["ingest", "--help"])
        assert "DESTRUCTIVE" in result.stdout


class TestPreflight:
    def test_a_missing_manifest_exits_one(self, settings: Settings) -> None:
        result = runner.invoke(app, ["ingest"])
        assert result.exit_code == 1
        assert "corpus download" in result.stdout + result.stderr

    def test_low_memory_refuses_to_start(
        self, settings: Settings, corpus: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Forty minutes in, an OOM kill leaves a half-populated collection."""
        from medguard import diagnostics

        monkeypatch.setattr(diagnostics, "available_ram_mb", lambda: 100.0)

        result = runner.invoke(app, ["ingest"])
        assert result.exit_code == 1
        assert "Available RAM below" in result.stdout + result.stderr

    def test_the_refusal_names_a_way_forward(
        self, settings: Settings, corpus: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from medguard import diagnostics

        monkeypatch.setattr(diagnostics, "available_ram_mb", lambda: 100.0)

        output = runner.invoke(app, ["ingest"]).stdout + runner.invoke(app, ["ingest"]).stderr
        assert "--dry-run" in output
        assert "MEDGUARD_INGEST_FORCE" in output

    def test_the_override_lets_it_through(
        self, settings: Settings, corpus: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The person at the keyboard may know better than a threshold does."""
        from medguard import diagnostics

        monkeypatch.setattr(diagnostics, "available_ram_mb", lambda: 100.0)
        monkeypatch.setenv("MEDGUARD_INGEST_FORCE", "1")

        result = runner.invoke(app, ["ingest", "--dry-run"])
        assert "Available RAM below" not in result.stdout

    def test_a_dry_run_skips_the_memory_check(
        self, settings: Settings, corpus: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dry run loads no model, so it is not what the threshold is protecting."""
        from medguard import diagnostics

        monkeypatch.setattr(diagnostics, "available_ram_mb", lambda: 50.0)

        result = runner.invoke(app, ["ingest", "--dry-run"])
        assert result.exit_code == 0


class TestDryRun:
    def test_it_exits_zero_and_writes_nothing(
        self, settings: Settings, corpus: dict[str, Path]
    ) -> None:
        from sqlalchemy import select

        from medguard.storage.schema import ChunkRow

        result = runner.invoke(app, ["ingest", "--dry-run"])
        assert result.exit_code == 0

        database = Database(settings)
        try:
            with database.session() as session:
                assert session.scalars(select(ChunkRow)).all() == []
        finally:
            database.dispose()

    def test_it_reports_pages_and_chunks(self, settings: Settings, corpus: dict[str, Path]) -> None:
        result = runner.invoke(app, ["ingest", "--dry-run"])
        assert "dry run" in result.stdout
        assert "ingested" in result.stdout

    def test_it_shows_sample_chunks(self, settings: Settings, corpus: dict[str, Path]) -> None:
        """Counts cannot tell a paragraph-aware chunker from one that cuts mid-sentence."""
        result = runner.invoke(app, ["ingest", "--dry-run"])
        assert "Sample chunks from" in result.stdout
        assert "::0000" in result.stdout

    def test_it_leaves_the_documents_pending(
        self, settings: Settings, corpus: dict[str, Path]
    ) -> None:
        from medguard.models import IngestionStatus
        from medguard.storage.schema import DocumentRow

        runner.invoke(app, ["ingest", "--dry-run"])

        database = Database(settings)
        try:
            with database.session() as session:
                row = session.get(DocumentRow, "doc_one")
                assert row is not None
                assert row.ingestion_status is IngestionStatus.PENDING
        finally:
            database.dispose()


class TestSelection:
    def test_only_restricts_to_one_document(
        self, settings: Settings, corpus: dict[str, Path]
    ) -> None:
        result = runner.invoke(app, ["ingest", "--dry-run", "--only", "doc_two"])
        assert result.exit_code == 0
        assert "doc_two" in result.stdout
        assert "doc_one" not in result.stdout.replace("doc_one.pdf", "")

    def test_limit_caps_the_run(self, settings: Settings, corpus: dict[str, Path]) -> None:
        result = runner.invoke(app, ["ingest", "--dry-run", "--limit", "1"])
        assert result.exit_code == 0
        assert "1 ingested" in result.stdout

    def test_an_unknown_document_id_is_not_an_error(
        self, settings: Settings, corpus: dict[str, Path]
    ) -> None:
        """Nothing to do is not a failure; it is the answer to the question asked."""
        result = runner.invoke(app, ["ingest", "--dry-run", "--only", "no_such_document"])
        assert result.exit_code == 0
        assert "Nothing to ingest" in result.stdout

    def test_a_zero_limit_is_rejected_by_the_parser(
        self, settings: Settings, corpus: dict[str, Path]
    ) -> None:
        result = runner.invoke(app, ["ingest", "--dry-run", "--limit", "0"])
        assert result.exit_code != 0


class TestExitCodes:
    def test_a_failed_document_exits_one(self, settings: Settings, corpus: dict[str, Path]) -> None:
        """The CI gate: a partially-indexed corpus must not report success."""
        from medguard.corpus.manifest import sha256_file

        broken = settings.corpus.raw_dir / "doc_three.pdf"
        broken.write_bytes(b"not a pdf at all")

        payload = json.loads(settings.corpus.manifest_path.read_text(encoding="utf-8"))
        payload["documents"].append(
            {
                "document_id": "doc_three",
                "title": "Title for doc_three",
                "source_url": "https://example.invalid/doc_three.pdf",
                "source_organization": "CDC",
                "document_type": "guideline",
                "publication_date": "2024-01-15",
                "license_or_usage_note": "Public domain test fixture.",
                "local_filename": broken.name,
                "sha256": sha256_file(broken),
                "file_size_bytes": broken.stat().st_size,
                "page_count": None,
                "retrieved_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
                "version_or_revision": None,
            }
        )
        settings.corpus.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

        result = runner.invoke(app, ["ingest", "--dry-run"])
        assert result.exit_code == 1
        assert "1 failed" in result.stdout

    def test_the_other_documents_still_run(
        self, settings: Settings, corpus: dict[str, Path]
    ) -> None:
        from medguard.corpus.manifest import sha256_file

        broken = settings.corpus.raw_dir / "aaa_broken.pdf"
        broken.write_bytes(b"not a pdf at all")

        payload = json.loads(settings.corpus.manifest_path.read_text(encoding="utf-8"))
        payload["documents"].insert(
            0,
            {
                "document_id": "aaa_broken",
                "title": "Broken",
                "source_url": "https://example.invalid/aaa.pdf",
                "source_organization": "CDC",
                "document_type": "guideline",
                "publication_date": None,
                "license_or_usage_note": "Public domain test fixture.",
                "local_filename": broken.name,
                "sha256": sha256_file(broken),
                "file_size_bytes": broken.stat().st_size,
                "page_count": None,
                "retrieved_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
                "version_or_revision": None,
            },
        )
        settings.corpus.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

        result = runner.invoke(app, ["ingest", "--dry-run"])
        assert result.exit_code == 1
        assert "2 ingested" in result.stdout

    def test_a_clean_run_exits_zero(self, settings: Settings, corpus: dict[str, Path]) -> None:
        result = runner.invoke(app, ["ingest", "--dry-run"])
        assert result.exit_code == 0


class TestForceRecreate:
    def test_it_asks_before_destroying_an_index(
        self, settings: Settings, corpus: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from medguard import diagnostics

        monkeypatch.setattr(diagnostics, "available_ram_mb", lambda: 4000.0)

        result = runner.invoke(app, ["ingest", "--force-recreate"], input="n\n")
        assert result.exit_code == 0
        assert "Nothing was changed" in result.stdout

    def test_declining_leaves_the_collection_alone(
        self, settings: Settings, corpus: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from medguard import diagnostics
        from medguard.storage.vector_store import VectorStore

        monkeypatch.setattr(diagnostics, "available_ram_mb", lambda: 4000.0)

        recreated: list[str] = []
        monkeypatch.setattr(
            VectorStore, "recreate_collection", lambda self: recreated.append("called")
        )

        runner.invoke(app, ["ingest", "--force-recreate"], input="n\n")
        assert recreated == []
