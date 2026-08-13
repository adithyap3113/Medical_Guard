"""Manifest schema, determinism and verification.

The manifest is the only committed evidence of what the corpus contained. If it can
drift from the files on disk without anything noticing, then every citation the system
later produces rests on an unchecked assumption.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from medguard.config import Settings
from medguard.corpus.manifest import (
    Manifest,
    ManifestDocument,
    build_manifest,
    load_manifest,
    sha256_file,
    verify_corpus,
    write_manifest,
)
from medguard.corpus.sources import SourceDocument, SourceList
from medguard.errors import ManifestError


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
        # Stated by default: a null date is now a reported warning, so leaving it unset
        # would add an unrelated finding to every verification assertion below.
        "publication_date": date(2024, 1, 15),
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


class TestManifestDocument:
    def test_page_count_defaults_to_none(self) -> None:
        """Phase 2 has no PDF parser. Null means 'not counted', never 'zero pages'."""
        assert _entry().page_count is None

    @pytest.mark.parametrize(
        "bad_hash",
        ["A" * 64, "a" * 63, "sha256:" + "a" * 57, "z" * 64, ""],
    )
    def test_rejects_malformed_hashes(self, bad_hash: str) -> None:
        """Lowercase hex, no prefix, exactly 64 chars -- so `sha256sum` output matches."""
        with pytest.raises(ValueError, match="sha256"):
            _entry(sha256=bad_hash)

    def test_rejects_unexpected_fields(self) -> None:
        with pytest.raises(ValueError, match="checksum"):
            _entry(checksum="abc")

    def test_negative_file_size_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="file_size_bytes"):
            _entry(file_size_bytes=-1)


class TestManifestInvariants:
    def test_duplicate_sha256_is_rejected(self) -> None:
        """Same bytes under two ids double-counts one source in every metric."""
        with pytest.raises(ValueError, match="duplicate sha256"):
            Manifest(
                generated_at=datetime.now(UTC),
                documents=[
                    _entry(),
                    _entry(document_id="cdc_other_2024", local_filename="cdc_other_2024.pdf"),
                ],
            )

    def test_duplicate_local_filename_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate local_filename"):
            Manifest(
                generated_at=datetime.now(UTC),
                documents=[
                    _entry(),
                    _entry(document_id="cdc_other_2024", sha256="b" * 64),
                ],
            )

    def test_duplicate_document_id_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate document_id"):
            Manifest(
                generated_at=datetime.now(UTC),
                documents=[
                    _entry(),
                    _entry(local_filename="cdc_other_2024.pdf", sha256="b" * 64),
                ],
            )


class TestSha256File:
    def test_matches_hashlib_over_the_same_bytes(self, tmp_path: Path, pdf_bytes: bytes) -> None:
        path = tmp_path / "x.pdf"
        path.write_bytes(pdf_bytes)
        assert sha256_file(path) == hashlib.sha256(pdf_bytes).hexdigest()

    def test_is_deterministic(self, tmp_path: Path, pdf_bytes: bytes) -> None:
        path = tmp_path / "x.pdf"
        path.write_bytes(pdf_bytes)
        assert sha256_file(path) == sha256_file(path)

    def test_changes_when_one_byte_changes(self, tmp_path: Path, pdf_bytes: bytes) -> None:
        a = tmp_path / "a.pdf"
        b = tmp_path / "b.pdf"
        a.write_bytes(pdf_bytes)
        b.write_bytes(pdf_bytes[:-1] + b"X")
        assert sha256_file(a) != sha256_file(b)


class TestRoundTrip:
    def test_write_then_load_preserves_every_field(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        original = Manifest(generated_at=datetime(2026, 5, 1, tzinfo=UTC), documents=[_entry()])
        write_manifest(original, path)
        assert load_manifest(path).documents == original.documents

    def test_output_is_sorted_by_document_id(self, tmp_path: Path) -> None:
        """A committed file must diff on real changes, not on dict ordering."""
        path = tmp_path / "manifest.json"
        write_manifest(
            Manifest(
                generated_at=datetime.now(UTC),
                documents=[
                    _entry(document_id="zzz_last", local_filename="z.pdf", sha256="b" * 64),
                    _entry(document_id="aaa_first", local_filename="a.pdf", sha256="c" * 64),
                ],
            ),
            path,
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert [d["document_id"] for d in payload["documents"]] == ["aaa_first", "zzz_last"]

    def test_non_ascii_titles_are_written_readably(self, tmp_path: Path) -> None:
        """CDC titles contain em dashes; escaping them makes the file unreviewable."""
        path = tmp_path / "manifest.json"
        write_manifest(
            Manifest(
                generated_at=datetime.now(UTC),
                documents=[_entry(title="Hepatitis C Screening — United States, 2020")],
            ),
            path,
        )
        assert "—" in path.read_text(encoding="utf-8")

    def test_missing_file_raises_manifest_error(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match="not found"):
            load_manifest(tmp_path / "absent.json")

    def test_invalid_json_raises_manifest_error(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ManifestError, match="not valid JSON"):
            load_manifest(path)

    def test_schema_violation_raises_manifest_error(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        path.write_text(
            json.dumps({"generated_at": "2026-01-01T00:00:00Z", "documents": [{"nope": 1}]}),
            encoding="utf-8",
        )
        with pytest.raises(ManifestError, match="failed validation"):
            load_manifest(path)


class TestBuildManifest:
    def test_only_records_documents_that_are_on_disk(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        """Recording a file nobody has asserts provenance for bytes that do not exist."""
        from medguard.corpus.download import DownloadOutcome, DownloadStatus

        (settings.corpus.raw_dir / "cdc_example_2024.pdf").write_bytes(pdf_bytes)
        sources = SourceList(
            documents=[
                _source(),
                _source(
                    document_id="cdc_absent_2024",
                    local_filename="cdc_absent_2024.pdf",
                    source_url="https://example.invalid/b.pdf",
                ),
            ]
        )
        outcomes = [
            DownloadOutcome(
                document_id="cdc_example_2024",
                local_filename="cdc_example_2024.pdf",
                status=DownloadStatus.DOWNLOADED,
                file_size_bytes=len(pdf_bytes),
                sha256=hashlib.sha256(pdf_bytes).hexdigest(),
            ),
            DownloadOutcome(
                document_id="cdc_absent_2024",
                local_filename="cdc_absent_2024.pdf",
                status=DownloadStatus.FAILED,
                error="HTTP 404",
            ),
        ]
        manifest = build_manifest(sources, outcomes, settings)
        assert [d.document_id for d in manifest.documents] == ["cdc_example_2024"]

    def test_retired_documents_still_get_provenance(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        """A retired source whose file exists must keep its manifest entry.

        Otherwise `verify` reports it as an orphan forever, and the corpus loses the
        record of a document it genuinely contains.
        """
        from medguard.corpus.download import DownloadOutcome, DownloadStatus

        (settings.corpus.raw_dir / "who_old_2019.pdf").write_bytes(pdf_bytes)
        sources = SourceList(
            documents=[
                _source(
                    document_id="who_old_2019",
                    local_filename="who_old_2019.pdf",
                    source_organization="WHO",
                    retired=True,
                    retired_reason="The publisher withdrew the URL.",
                )
            ]
        )
        outcomes = [
            DownloadOutcome(
                document_id="who_old_2019",
                local_filename="who_old_2019.pdf",
                status=DownloadStatus.RETIRED,
            )
        ]
        manifest = build_manifest(sources, outcomes, settings)
        assert len(manifest.documents) == 1
        assert manifest.documents[0].sha256 == hashlib.sha256(pdf_bytes).hexdigest()

    def test_retrieved_at_is_preserved_for_unchanged_files(
        self,
        settings: Settings,
        pdf_bytes: bytes,
    ) -> None:
        """Re-running download must not restamp a file that did not change."""
        from medguard.corpus.download import DownloadOutcome, DownloadStatus

        (settings.corpus.raw_dir / "cdc_example_2024.pdf").write_bytes(pdf_bytes)
        digest = hashlib.sha256(pdf_bytes).hexdigest()
        original_time = datetime(2020, 6, 1, tzinfo=UTC)
        previous = Manifest(
            generated_at=datetime.now(UTC),
            documents=[_entry(sha256=digest, retrieved_at=original_time)],
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
        assert manifest.documents[0].retrieved_at == original_time

    def test_page_count_survives_a_rebuild(self, settings: Settings, pdf_bytes: bytes) -> None:
        """Phase 3 fills page_count; re-running download must not wipe it."""
        from medguard.corpus.download import DownloadOutcome, DownloadStatus

        (settings.corpus.raw_dir / "cdc_example_2024.pdf").write_bytes(pdf_bytes)
        digest = hashlib.sha256(pdf_bytes).hexdigest()
        previous = Manifest(
            generated_at=datetime.now(UTC),
            documents=[_entry(sha256=digest, page_count=97)],
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


class TestVerifyCorpus:
    def _write(self, settings: Settings, pdf_bytes: bytes) -> Manifest:
        path = settings.corpus.raw_dir / "cdc_example_2024.pdf"
        path.write_bytes(pdf_bytes)
        return Manifest(
            generated_at=datetime.now(UTC),
            documents=[
                _entry(
                    sha256=hashlib.sha256(pdf_bytes).hexdigest(),
                    file_size_bytes=len(pdf_bytes),
                )
            ],
        )

    def test_a_matching_corpus_produces_no_issues(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        report = verify_corpus(self._write(settings, pdf_bytes), settings)
        assert report.issues == []
        assert report.ok
        assert report.exit_code() == 0

    def test_a_missing_file_is_an_error(self, settings: Settings, pdf_bytes: bytes) -> None:
        """Phase 3A raised this from a warning.

        With 20 documents across four publishers, "some are absent" stopped being a
        benign state: it is the difference between an index over 20 documents and an
        index over 14, with nothing visible to say which. A fresh clone runs
        `medguard corpus download` before it runs `verify`.
        """
        manifest = self._write(settings, pdf_bytes)
        (settings.corpus.raw_dir / "cdc_example_2024.pdf").unlink()

        report = verify_corpus(manifest, settings)
        assert not report.ok
        assert report.exit_code() == 1
        assert any("absent" in i.detail for i in report.errors)

    def test_a_null_publication_date_is_a_warning_not_an_error(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        """Correct under "never invent metadata", so it must not fail the build.

        Nine of the twenty committed documents have no publisher-stated date. The
        warning keeps the count visible rather than letting it be forgotten.
        """
        path = settings.corpus.raw_dir / "cdc_example_2024.pdf"
        path.write_bytes(pdf_bytes)
        manifest = Manifest(
            generated_at=datetime.now(UTC),
            documents=[
                _entry(
                    sha256=hashlib.sha256(pdf_bytes).hexdigest(),
                    file_size_bytes=len(pdf_bytes),
                    publication_date=None,
                )
            ],
        )

        report = verify_corpus(manifest, settings)
        assert report.ok
        assert report.exit_code() == 0
        assert any("publication_date is null" in i.detail for i in report.warnings)

    def test_duplicate_content_under_two_ids_is_an_error(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        """Two ids over identical bytes double-counts one source in every metric."""
        (settings.corpus.raw_dir / "cdc_example_2024.pdf").write_bytes(pdf_bytes)
        (settings.corpus.raw_dir / "cdc_other_2024.pdf").write_bytes(pdf_bytes)
        digest = hashlib.sha256(pdf_bytes).hexdigest()
        manifest = Manifest(
            generated_at=datetime.now(UTC),
            documents=[
                _entry(sha256=digest, file_size_bytes=len(pdf_bytes)),
                # A different recorded hash, so the Manifest model itself permits it;
                # the files on disk are what have converged.
                _entry(
                    document_id="cdc_other_2024",
                    local_filename="cdc_other_2024.pdf",
                    sha256="b" * 64,
                    file_size_bytes=len(pdf_bytes),
                ),
            ],
        )

        report = verify_corpus(manifest, settings)
        assert not report.ok
        assert any("duplicate content" in i.detail for i in report.errors)

    def test_a_changed_file_is_an_error(self, settings: Settings, pdf_bytes: bytes) -> None:
        """Silent drift here corrupts retrieval in a way that surfaces phases later."""
        manifest = self._write(settings, pdf_bytes)
        (settings.corpus.raw_dir / "cdc_example_2024.pdf").write_bytes(
            b"%PDF-1.7\n" + b"different content\n" * 900
        )

        report = verify_corpus(manifest, settings)
        assert not report.ok
        assert report.exit_code() == 1
        assert any("sha256 mismatch" in i.detail for i in report.errors)

    def test_a_non_pdf_file_is_an_error(self, settings: Settings, pdf_bytes: bytes) -> None:
        manifest = self._write(settings, pdf_bytes)
        (settings.corpus.raw_dir / "cdc_example_2024.pdf").write_bytes(b"<html>gotcha</html>")

        report = verify_corpus(manifest, settings)
        assert not report.ok
        assert any("not a PDF" in i.detail for i in report.errors)

    def test_an_unrecorded_file_is_an_error(self, settings: Settings, pdf_bytes: bytes) -> None:
        """Phase 3A raised this from a warning: no provenance means nothing can cite it."""
        manifest = self._write(settings, pdf_bytes)
        (settings.corpus.raw_dir / "mystery.pdf").write_bytes(pdf_bytes + b"extra")

        report = verify_corpus(manifest, settings)
        assert not report.ok
        assert any("no manifest entry" in i.detail for i in report.errors)

    def test_a_leftover_part_file_is_reported(self, settings: Settings, pdf_bytes: bytes) -> None:
        """Visible rather than mysterious: it tells the user a download was interrupted."""
        manifest = self._write(settings, pdf_bytes)
        (settings.corpus.raw_dir / "cdc_other.pdf.part").write_bytes(b"%PDF-partial")

        report = verify_corpus(manifest, settings)
        assert any("partial download" in i.detail for i in report.warnings)

    def test_verification_never_raises_on_a_bad_corpus(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        """It must report all twelve problems, not die on the first."""
        manifest = Manifest(
            generated_at=datetime.now(UTC),
            documents=[
                _entry(
                    document_id=f"cdc_doc_{i:02d}", local_filename=f"d{i}.pdf", sha256=str(i) * 64
                )
                for i in range(1, 6)
            ],
        )
        report = verify_corpus(manifest, settings)
        assert len(report.errors) == 5
        assert report.checked == 5


class TestFreshnessNotes:
    """`verify --verbose` reports what the database knows and the manifest does not."""

    def _clean(self, settings: Settings, pdf_bytes: bytes, **overrides: object) -> Manifest:
        path = settings.corpus.raw_dir / "cdc_example_2024.pdf"
        path.write_bytes(pdf_bytes)
        return Manifest(
            generated_at=datetime.now(UTC),
            documents=[
                _entry(
                    sha256=hashlib.sha256(pdf_bytes).hexdigest(),
                    file_size_bytes=len(pdf_bytes),
                    **overrides,
                )
            ],
        )

    def test_a_gap_is_reported_as_a_note(self, settings: Settings, pdf_bytes: bytes) -> None:
        from medguard.corpus.manifest import BackfillField

        report = verify_corpus(
            self._clean(settings, pdf_bytes),
            settings,
            database_values={"cdc_example_2024": {BackfillField.PAGE_COUNT: 61}},
        )
        assert report.ok
        assert len(report.notes) == 1
        assert "database has newer operational values" in report.notes[0].detail

    def test_the_note_names_the_command_that_promotes_it(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        from medguard.corpus.manifest import BackfillField

        report = verify_corpus(
            self._clean(settings, pdf_bytes),
            settings,
            database_values={"cdc_example_2024": {BackfillField.PAGE_COUNT: 61}},
        )
        assert "--backfill-page-counts" in report.notes[0].detail

    def test_notes_do_not_affect_the_exit_code(self, settings: Settings, pdf_bytes: bytes) -> None:
        """Work being available is not a failure."""
        from medguard.corpus.manifest import BackfillField

        report = verify_corpus(
            self._clean(settings, pdf_bytes),
            settings,
            database_values={"cdc_example_2024": {BackfillField.PAGE_COUNT: 61}},
        )
        assert report.exit_code() == 0

    def test_a_populated_manifest_field_produces_no_note(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        from medguard.corpus.manifest import BackfillField

        report = verify_corpus(
            self._clean(settings, pdf_bytes, page_count=61),
            settings,
            database_values={"cdc_example_2024": {BackfillField.PAGE_COUNT: 61}},
        )
        assert report.notes == []

    def test_omitting_database_values_skips_the_comparison(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        """`verify` must stay runnable with no database at all."""
        report = verify_corpus(self._clean(settings, pdf_bytes), settings)
        assert report.notes == []


class TestShippedManifest:
    """The committed manifest must stay valid and consistent with sources.yaml."""

    def test_the_committed_manifest_loads(self) -> None:
        from medguard.config import project_root

        manifest = load_manifest(project_root() / "data" / "manifest.json")
        assert manifest.schema_version == 1
        assert len(manifest.documents) >= 18

    def test_the_corpus_spans_several_publishers(self) -> None:
        """A single-publisher corpus cannot demonstrate provenance-aware behaviour.

        With one publisher, "which organisation issued this recommendation?" has one
        answer, and nothing downstream that depends on provenance is testable at all.
        """
        from collections import Counter

        from medguard.config import project_root

        manifest = load_manifest(project_root() / "data" / "manifest.json")
        by_publisher = Counter(d.source_organization for d in manifest.documents)
        assert len(by_publisher) >= 3
        # No publisher may be so dominant that the others are decorative.
        assert max(by_publisher.values()) / len(manifest.documents) < 0.75

    def test_no_publisher_is_represented_by_a_single_document(self) -> None:
        """One document from a publisher is a token, not coverage."""
        from collections import Counter

        from medguard.config import project_root
        from medguard.corpus.sources import load_sources

        root = project_root()
        manifest = load_manifest(root / "data" / "manifest.json")
        sources = load_sources(root / "data" / "sources.yaml")
        active = {d.document_id for d in sources.active}

        by_publisher = Counter(
            d.source_organization for d in manifest.documents if d.document_id in active
        )
        singletons = [p for p, n in by_publisher.items() if n < 2]
        assert singletons == [], f"publishers with fewer than two active documents: {singletons}"

    def test_every_manifest_entry_has_a_source_entry(self) -> None:
        from medguard.config import project_root
        from medguard.corpus.sources import load_sources

        root = project_root()
        manifest = load_manifest(root / "data" / "manifest.json")
        sources = load_sources(root / "data" / "sources.yaml")
        for doc in manifest.documents:
            assert sources.by_id(doc.document_id) is not None, doc.document_id

    def test_page_count_is_null_before_a_parser_exists(self) -> None:
        """Filling this needs a PDF parser. A guess would be indistinguishable from data."""
        from medguard.config import project_root

        manifest = load_manifest(project_root() / "data" / "manifest.json")
        assert all(d.page_count is None for d in manifest.documents)

    def test_every_null_publication_date_is_accounted_for_in_sources_md(self) -> None:
        """ "Never invent metadata" only holds if each absence has a recorded reason."""
        from medguard.config import project_root

        root = project_root()
        manifest = load_manifest(root / "data" / "manifest.json")
        prose = (root / "data" / "raw" / "SOURCES.md").read_text(encoding="utf-8")

        for doc in manifest.documents:
            if doc.publication_date is None:
                assert doc.document_id in prose, (
                    f"{doc.document_id} has a null publication_date with no reason "
                    "recorded in data/raw/SOURCES.md"
                )

    def test_no_licence_note_is_silently_assumed(self) -> None:
        """A publisher that states no terms must say so, not borrow someone else's."""
        from medguard.config import project_root

        manifest = load_manifest(project_root() / "data" / "manifest.json")
        for doc in manifest.documents:
            if doc.source_organization.value == "NIH":
                assert "not published" in doc.license_or_usage_note

    def test_every_entry_records_usage_terms(self) -> None:
        """Provenance without licence terms is not provenance."""
        from medguard.config import project_root

        manifest = load_manifest(project_root() / "data" / "manifest.json")
        assert all(len(d.license_or_usage_note.strip()) > 20 for d in manifest.documents)
