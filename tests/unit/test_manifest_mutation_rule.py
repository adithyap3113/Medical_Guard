"""Only two callers may write ``data/manifest.json``.

The rule, stated in :mod:`medguard.corpus` and enforced here:

1. ``corpus.manifest.write_manifest`` (reached via ``build_manifest`` on
   ``medguard corpus download``)
2. ``medguard corpus sync`` -- adopt mode and the ``--backfill-*`` modes

``medguard ingest`` must never write to it. The manifest is the committed record of what
the corpus *is*; ingestion consumes that record, and a consumer that rewrites its own
input has destroyed the only independent check on itself. ``corpus verify`` would then
be comparing ingestion's output against ingestion's own claims.

Phase 3A does not implement ``medguard ingest``, which is exactly why the rule is
encoded now: the scaffolding has to carry the constraint *before* the code that would
violate it gets written.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from medguard.cli.app import app
from medguard.config import Settings, project_root

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


def _seed_manifest(settings: Settings, pdf: bytes) -> float:
    """Write a valid manifest and return its mtime."""
    (settings.corpus.raw_dir / "test_guideline_2024.pdf").write_bytes(pdf)
    settings.corpus.sources_path.write_text(SOURCES, encoding="utf-8")
    assert runner.invoke(app, ["corpus", "sync", "--yes"]).exit_code == 0
    return settings.corpus.manifest_path.stat().st_mtime_ns


class TestOnlySanctionedWritersExist:
    def test_the_source_tree_has_exactly_one_write_site(self) -> None:
        """`write_manifest` is the only function that opens the file for writing.

        A second write path -- however well-intentioned -- is how ingestion ends up
        editing provenance, so the count is asserted rather than assumed.
        """
        src = project_root() / "src"
        writers = sorted(
            path.relative_to(src).as_posix()
            for path in src.rglob("*.py")
            if "def write_manifest" in path.read_text(encoding="utf-8")
        )
        assert writers == ["medguard/corpus/manifest.py"]

    def test_only_the_download_and_sync_commands_call_it(self) -> None:
        src = project_root() / "src"
        callers = sorted(
            path.relative_to(src).as_posix()
            for path in src.rglob("*.py")
            if "write_manifest(" in path.read_text(encoding="utf-8")
        )
        assert callers == ["medguard/cli/corpus.py", "medguard/corpus/manifest.py"]

    def test_no_ingest_module_writes_the_manifest(self) -> None:
        """Holds trivially in Phase 3A. It must still hold once Phase 3B lands."""
        src = project_root() / "src"
        offenders = [
            path.relative_to(src).as_posix()
            for path in src.rglob("*.py")
            if "ingest" in path.name and "write_manifest" in path.read_text(encoding="utf-8")
        ]
        assert offenders == []

    def test_the_rule_is_written_down_where_an_implementer_will_see_it(self) -> None:
        text = (project_root() / "src" / "medguard" / "corpus" / "__init__.py").read_text(
            encoding="utf-8"
        )
        assert "must never write to it" in text


class TestNonSyncOperationsLeaveTheFileAlone:
    def test_verify_does_not_touch_the_manifest(self, settings: Settings, pdf_bytes: bytes) -> None:
        before = _seed_manifest(settings, pdf_bytes)
        assert runner.invoke(app, ["corpus", "verify"]).exit_code == 0
        assert settings.corpus.manifest_path.stat().st_mtime_ns == before

    def test_verbose_verify_does_not_touch_the_manifest(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        """--verbose reads the database too; reading must stay read-only."""
        before = _seed_manifest(settings, pdf_bytes)
        runner.invoke(app, ["corpus", "verify", "--verbose"])
        assert settings.corpus.manifest_path.stat().st_mtime_ns == before

    def test_doctor_does_not_touch_the_manifest(self, settings: Settings, pdf_bytes: bytes) -> None:
        before = _seed_manifest(settings, pdf_bytes)
        runner.invoke(app, ["doctor", "--no-hardware"])
        assert settings.corpus.manifest_path.stat().st_mtime_ns == before

    def test_a_download_that_writes_no_manifest_leaves_it_alone(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        before = _seed_manifest(settings, pdf_bytes)
        runner.invoke(app, ["corpus", "download", "--no-write-manifest"])
        assert settings.corpus.manifest_path.stat().st_mtime_ns == before

    def test_a_no_op_sync_leaves_the_file_byte_identical(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        """The `generated_at` rule: a run that changes nothing must not restamp it."""
        _seed_manifest(settings, pdf_bytes)
        path = settings.corpus.manifest_path
        before_bytes = path.read_bytes()
        before_mtime = path.stat().st_mtime_ns

        result = runner.invoke(app, ["corpus", "sync", "--yes"])

        assert result.exit_code == 0
        assert path.stat().st_mtime_ns == before_mtime
        assert path.read_bytes() == before_bytes

    def test_a_no_op_backfill_leaves_the_file_byte_identical(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        """A readable database with nothing new to offer must not restamp the file."""
        from medguard.storage.database import Database

        _seed_manifest(settings, pdf_bytes)
        database = Database(settings)
        database.create_all()  # tables exist, no rows: nothing to promote
        database.dispose()

        path = settings.corpus.manifest_path
        before_bytes = path.read_bytes()

        result = runner.invoke(app, ["corpus", "sync", "--backfill-page-counts", "--yes"])

        assert result.exit_code == 0
        assert path.read_bytes() == before_bytes


class TestGeneratedAtMovesOnlyOnRealChange:
    def test_adopting_a_document_restamps_generated_at(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        """The counterpart: a run that *does* change something must say so."""
        path = settings.corpus.manifest_path
        _seed_manifest(settings, pdf_bytes)
        first = json.loads(path.read_text(encoding="utf-8"))["generated_at"]

        settings.corpus.sources_path.write_text(
            SOURCES
            + """
  - document_id: test_second_2025
    title: A Second Test Guideline, 2025
    source_url: https://example.invalid/second.pdf
    source_organization: CDC
    document_type: guideline
    local_filename: test_second_2025.pdf
    license_or_usage_note: Public domain test fixture, stated by the publisher.
""",
            encoding="utf-8",
        )
        (settings.corpus.raw_dir / "test_second_2025.pdf").write_bytes(pdf_bytes + b"second")

        assert runner.invoke(app, ["corpus", "sync", "--yes"]).exit_code == 0
        second = json.loads(path.read_text(encoding="utf-8"))["generated_at"]
        assert second != first

    def test_the_committed_manifest_is_deterministic_to_reserialise(self) -> None:
        """Rewriting the shipped manifest unchanged must produce identical bytes."""
        from medguard.corpus.manifest import load_manifest, write_manifest

        real = project_root() / "data" / "manifest.json"
        original = real.read_text(encoding="utf-8")

        scratch = Path(str(real) + ".roundtrip")
        try:
            write_manifest(load_manifest(real), scratch)
            assert scratch.read_text(encoding="utf-8") == original
        finally:
            scratch.unlink(missing_ok=True)
