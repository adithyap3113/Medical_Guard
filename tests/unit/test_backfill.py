"""Backfill promotes database values into the manifest, and never overwrites.

Two properties, and both are load-bearing.

**Null-only.** A manifest value was transcribed from a publisher by a human; a database
value was derived by parsing a file. When they disagree, the publisher wins and the
disagreement is a finding for a person, not something to resolve by overwriting. Any
backfill that can clobber a recorded value has quietly made the manifest a cache of
whatever ingestion last believed.

**Round-tripping.** `source_to_manifest_entry` used to hard-code ``page_count=None``,
which meant a backfilled value survived ``corpus download`` (``build_manifest``
preserved it) and was silently erased by the next ``corpus sync`` that rebuilt the entry.
The bug produced no error and no diff anyone would read as wrong -- the field just went
back to null. The regression tests at the bottom of this file exist for that.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

from typer.testing import CliRunner

from medguard.cli.app import app
from medguard.config import Settings
from medguard.corpus.manifest import (
    BackfillField,
    Manifest,
    ManifestDocument,
    apply_backfill,
    build_manifest,
    plan_backfill,
    source_to_manifest_entry,
)
from medguard.corpus.sources import SourceDocument, SourceList

runner = CliRunner()

SOURCES = """
schema_version: 1
documents:
  - document_id: test_guideline_2024
    title: A Test Clinical Guideline, 2024
    source_url: https://example.invalid/guideline.pdf
    source_organization: CDC
    document_type: guideline
    publication_date: null
    local_filename: test_guideline_2024.pdf
    license_or_usage_note: Public domain test fixture, stated by the publisher.
"""


def _entry(**overrides: object) -> ManifestDocument:
    base: dict[str, object] = {
        "document_id": "cdc_example_2024",
        "title": "An Example Guideline, 2024",
        "source_url": "https://example.invalid/a.pdf",
        "source_organization": "CDC",
        "document_type": "guideline",
        "license_or_usage_note": "Public domain.",
        "local_filename": "cdc_example_2024.pdf",
        "sha256": "a" * 64,
        "file_size_bytes": 12345,
        "retrieved_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    base.update(overrides)
    return ManifestDocument.model_validate(base)


def _source(**overrides: object) -> SourceDocument:
    base: dict[str, object] = {
        "document_id": "cdc_example_2024",
        "title": "An Example Guideline, 2024",
        "source_url": "https://example.invalid/a.pdf",
        "source_organization": "CDC",
        "document_type": "guideline",
        "local_filename": "cdc_example_2024.pdf",
        "license_or_usage_note": "Public domain.",
    }
    base.update(overrides)
    return SourceDocument.model_validate(base)


def _manifest(*docs: ManifestDocument) -> Manifest:
    return Manifest(generated_at=datetime(2026, 1, 1, tzinfo=UTC), documents=list(docs))


class TestPlanBackfillFillsOnlyNulls:
    def test_a_null_field_is_a_candidate(self) -> None:
        changes = plan_backfill(
            _manifest(_entry(page_count=None)),
            {"cdc_example_2024": {BackfillField.PAGE_COUNT: 97}},
            {BackfillField.PAGE_COUNT},
        )
        assert [(c.document_id, c.field, c.value) for c in changes] == [
            ("cdc_example_2024", BackfillField.PAGE_COUNT, 97)
        ]

    def test_a_populated_field_is_never_a_candidate(self) -> None:
        """The publisher's value outranks anything derived from parsing a file."""
        changes = plan_backfill(
            _manifest(_entry(page_count=42)),
            {"cdc_example_2024": {BackfillField.PAGE_COUNT: 97}},
            {BackfillField.PAGE_COUNT},
        )
        assert changes == []

    def test_a_null_database_value_proposes_nothing(self) -> None:
        changes = plan_backfill(
            _manifest(_entry(page_count=None)),
            {"cdc_example_2024": {BackfillField.PAGE_COUNT: None}},
            {BackfillField.PAGE_COUNT},
        )
        assert changes == []

    def test_a_document_absent_from_the_database_proposes_nothing(self) -> None:
        changes = plan_backfill(_manifest(_entry(page_count=None)), {}, {BackfillField.PAGE_COUNT})
        assert changes == []

    def test_only_the_requested_fields_are_considered(self) -> None:
        """`--backfill-page-counts` must not quietly also write dates."""
        changes = plan_backfill(
            _manifest(_entry(page_count=None, publication_date=None)),
            {
                "cdc_example_2024": {
                    BackfillField.PAGE_COUNT: 97,
                    BackfillField.PUBLICATION_DATE: date(2024, 3, 1),
                }
            },
            {BackfillField.PAGE_COUNT},
        )
        assert [c.field for c in changes] == [BackfillField.PAGE_COUNT]

    def test_both_fields_can_be_requested_together(self) -> None:
        changes = plan_backfill(
            _manifest(_entry(page_count=None, publication_date=None)),
            {
                "cdc_example_2024": {
                    BackfillField.PAGE_COUNT: 97,
                    BackfillField.PUBLICATION_DATE: date(2024, 3, 1),
                }
            },
            {BackfillField.PAGE_COUNT, BackfillField.PUBLICATION_DATE},
        )
        assert {c.field for c in changes} == {
            BackfillField.PAGE_COUNT,
            BackfillField.PUBLICATION_DATE,
        }

    def test_planning_mutates_nothing(self) -> None:
        manifest = _manifest(_entry(page_count=None))
        plan_backfill(
            manifest,
            {"cdc_example_2024": {BackfillField.PAGE_COUNT: 97}},
            {BackfillField.PAGE_COUNT},
        )
        assert manifest.documents[0].page_count is None


class TestApplyBackfill:
    def test_the_value_lands_on_the_right_document(self) -> None:
        manifest = _manifest(
            _entry(page_count=None),
            _entry(document_id="cdc_other_2024", local_filename="b.pdf", sha256="b" * 64),
        )
        changes = plan_backfill(
            manifest,
            {"cdc_example_2024": {BackfillField.PAGE_COUNT: 97}},
            {BackfillField.PAGE_COUNT},
        )
        updated = apply_backfill(manifest, changes)
        assert updated.by_id("cdc_example_2024").page_count == 97  # type: ignore[union-attr]
        assert updated.by_id("cdc_other_2024").page_count is None  # type: ignore[union-attr]

    def test_generated_at_moves_when_something_changed(self) -> None:
        manifest = _manifest(_entry(page_count=None))
        changes = plan_backfill(
            manifest,
            {"cdc_example_2024": {BackfillField.PAGE_COUNT: 97}},
            {BackfillField.PAGE_COUNT},
        )
        assert apply_backfill(manifest, changes).generated_at != manifest.generated_at

    def test_generated_at_is_untouched_when_nothing_changed(self) -> None:
        """`generated_at` is how a diff answers "did this run alter the record?"."""
        manifest = _manifest(_entry(page_count=None))
        assert apply_backfill(manifest, []).generated_at == manifest.generated_at

    def test_an_empty_change_list_returns_the_manifest_unchanged(self) -> None:
        manifest = _manifest(_entry(page_count=None))
        assert apply_backfill(manifest, []) == manifest

    def test_no_other_field_is_disturbed(self) -> None:
        manifest = _manifest(_entry(page_count=None))
        changes = plan_backfill(
            manifest,
            {"cdc_example_2024": {BackfillField.PAGE_COUNT: 97}},
            {BackfillField.PAGE_COUNT},
        )
        before = manifest.documents[0]
        after = apply_backfill(manifest, changes).documents[0]
        for field in (
            "sha256",
            "file_size_bytes",
            "title",
            "source_url",
            "source_organization",
            "document_type",
            "license_or_usage_note",
            "retrieved_at",
            "version_or_revision",
        ):
            assert getattr(after, field) == getattr(before, field)


class TestBackfillSurvivesARebuild:
    """The `source_to_manifest_entry` regression: a backfilled value must not evaporate."""

    def test_sync_after_backfill_preserves_the_page_count(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        """The exact bug. `sync` rebuilt the entry from its source and lost the value."""
        path = settings.corpus.raw_dir / "cdc_example_2024.pdf"
        path.write_bytes(pdf_bytes)

        rebuilt = source_to_manifest_entry(_source(), path, page_count=97)
        assert rebuilt.page_count == 97

    def test_sync_after_backfill_preserves_the_publication_date(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        path = settings.corpus.raw_dir / "cdc_example_2024.pdf"
        path.write_bytes(pdf_bytes)

        rebuilt = source_to_manifest_entry(_source(), path, publication_date=date(2024, 3, 1))
        assert rebuilt.publication_date == date(2024, 3, 1)

    def test_a_publisher_stated_date_still_wins_over_a_carried_one(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        """sources.yaml is transcribed from a publisher; a carried value is derived."""
        path = settings.corpus.raw_dir / "cdc_example_2024.pdf"
        path.write_bytes(pdf_bytes)

        rebuilt = source_to_manifest_entry(
            _source(publication_date=date(2020, 1, 1)),
            path,
            publication_date=date(2024, 3, 1),
        )
        assert rebuilt.publication_date == date(2020, 1, 1)

    def test_download_preserves_a_backfilled_publication_date(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        """`build_manifest` had the same asymmetry for dates that it avoided for pages."""
        import hashlib

        from medguard.corpus.download import DownloadOutcome, DownloadStatus

        (settings.corpus.raw_dir / "cdc_example_2024.pdf").write_bytes(pdf_bytes)
        digest = hashlib.sha256(pdf_bytes).hexdigest()
        previous = _manifest(
            _entry(sha256=digest, page_count=97, publication_date=date(2024, 3, 1))
        )
        outcomes = [
            DownloadOutcome(
                document_id="cdc_example_2024",
                local_filename="cdc_example_2024.pdf",
                status=DownloadStatus.PRESENT,
                file_size_bytes=len(pdf_bytes),
                sha256=digest,
            )
        ]
        manifest = build_manifest(
            SourceList(documents=[_source()]), outcomes, settings, previous=previous
        )
        assert manifest.documents[0].page_count == 97
        assert manifest.documents[0].publication_date == date(2024, 3, 1)


class TestBackfillCli:
    def _seed(self, settings: Settings, pdf: bytes) -> None:
        (settings.corpus.raw_dir / "test_guideline_2024.pdf").write_bytes(pdf)
        settings.corpus.sources_path.write_text(SOURCES, encoding="utf-8")
        assert runner.invoke(app, ["corpus", "sync", "--yes"]).exit_code == 0

    def _write_rows(self, settings: Settings, **values: object) -> None:
        """Put a documents row in place so backfill has something to read."""
        from medguard.corpus.manifest import load_manifest
        from medguard.storage.database import Database
        from medguard.storage.schema import DocumentRow

        entry = load_manifest(settings.corpus.manifest_path).documents[0]
        database = Database(settings)
        database.create_all()
        with database.session() as session:
            session.add(
                DocumentRow(
                    document_id=entry.document_id,
                    title=entry.title,
                    source_url=entry.source_url,
                    source_organization=entry.source_organization,
                    document_type=entry.document_type,
                    local_filename=entry.local_filename,
                    sha256=entry.sha256,
                    file_size_bytes=entry.file_size_bytes,
                    retrieved_at=entry.retrieved_at,
                    license_or_usage_note=entry.license_or_usage_note,
                    **values,
                )
            )
        database.dispose()

    def test_page_counts_are_promoted(self, settings: Settings, pdf_bytes: bytes) -> None:
        self._seed(settings, pdf_bytes)
        self._write_rows(settings, page_count=61)

        result = runner.invoke(app, ["corpus", "sync", "--backfill-page-counts", "--yes"])
        assert result.exit_code == 0

        payload = json.loads(settings.corpus.manifest_path.read_text(encoding="utf-8"))
        assert payload["documents"][0]["page_count"] == 61

    def test_publication_dates_are_promoted(self, settings: Settings, pdf_bytes: bytes) -> None:
        self._seed(settings, pdf_bytes)
        self._write_rows(settings, publication_date=date(2024, 3, 1))

        result = runner.invoke(app, ["corpus", "sync", "--backfill-publication-dates", "--yes"])
        assert result.exit_code == 0

        payload = json.loads(settings.corpus.manifest_path.read_text(encoding="utf-8"))
        assert payload["documents"][0]["publication_date"] == "2024-03-01"

    def test_page_count_flag_does_not_write_dates(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        self._seed(settings, pdf_bytes)
        self._write_rows(settings, page_count=61, publication_date=date(2024, 3, 1))

        runner.invoke(app, ["corpus", "sync", "--backfill-page-counts", "--yes"])

        payload = json.loads(settings.corpus.manifest_path.read_text(encoding="utf-8"))
        assert payload["documents"][0]["page_count"] == 61
        assert payload["documents"][0]["publication_date"] is None

    def test_declining_the_prompt_writes_nothing(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        """Backfill edits committed provenance, so it asks before every field."""
        self._seed(settings, pdf_bytes)
        self._write_rows(settings, page_count=61)
        before = settings.corpus.manifest_path.read_bytes()

        result = runner.invoke(app, ["corpus", "sync", "--backfill-page-counts"], input="n\n")
        assert result.exit_code == 0
        assert settings.corpus.manifest_path.read_bytes() == before

    def test_an_existing_value_is_not_overwritten(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        self._seed(settings, pdf_bytes)
        self._write_rows(settings, page_count=61)
        runner.invoke(app, ["corpus", "sync", "--backfill-page-counts", "--yes"])

        # The database now disagrees with a recorded manifest value.
        from medguard.storage.database import Database
        from medguard.storage.schema import DocumentRow

        database = Database(settings)
        with database.session() as session:
            row = session.get(DocumentRow, "test_guideline_2024")
            assert row is not None
            row.page_count = 999
        database.dispose()

        runner.invoke(app, ["corpus", "sync", "--backfill-page-counts", "--yes"])
        payload = json.loads(settings.corpus.manifest_path.read_text(encoding="utf-8"))
        assert payload["documents"][0]["page_count"] == 61

    def test_backfill_without_a_manifest_fails_clearly(self, settings: Settings) -> None:
        result = runner.invoke(app, ["corpus", "sync", "--backfill-page-counts", "--yes"])
        assert result.exit_code == 1

    def test_backfill_without_a_database_fails_clearly(
        self, settings: Settings, pdf_bytes: bytes, tmp_path: Path
    ) -> None:
        """No database means no values; that is an error, not a silent no-op."""
        self._seed(settings, pdf_bytes)
        result = runner.invoke(app, ["corpus", "sync", "--backfill-page-counts", "--yes"])
        assert result.exit_code == 1
