"""Two editions of one guideline must stay distinguishable end to end.

The corpus deliberately contains only the 2022 edition of the ICMR antimicrobial
guidelines, because two live editions recommending different drugs is a retrieval hazard
(see ``data/raw/SOURCES.md``). That decision costs nothing in capability *only if* the
version machinery is actually tested -- otherwise "we handle versions" is an untested
claim.

So it is tested here, against generated documents that share a title and a topic and
recommend different drugs, with no possibility of real clinical guidance reaching a user.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from medguard.config import Settings
from medguard.corpus.manifest import Manifest, source_to_manifest_entry, verify_corpus
from medguard.corpus.sources import SourceDocument, SourceList
from medguard.storage.schema import make_chunk_id
from tests.fixtures.make_pdf import (
    DOCUMENT_ID_ROOT,
    SHARED_TITLE,
    SYNTHETIC_VERSIONS,
    write_synthetic_pdf,
    write_synthetic_versions,
)


def _source_for(document_id: str, filename: str, version: str) -> SourceDocument:
    return SourceDocument.model_validate(
        {
            "document_id": document_id,
            "title": SHARED_TITLE,
            "source_url": f"https://example.invalid/{filename}",
            "source_organization": "ICMR",
            "document_type": "guideline",
            "local_filename": filename,
            "license_or_usage_note": "Synthetic test fixture; not a real publication.",
            "version_or_revision": version,
        }
    )


class TestFixtureGeneration:
    def test_both_editions_are_written(self, tmp_path: Path) -> None:
        written = write_synthetic_versions(tmp_path)
        assert sorted(written) == ["test_amr_2020", "test_amr_2023"]
        assert all(p.is_file() for p in written.values())

    def test_they_are_real_pdfs(self, tmp_path: Path) -> None:
        for path in write_synthetic_versions(tmp_path).values():
            assert path.read_bytes().startswith(b"%PDF-")

    def test_they_clear_the_minimum_size_floor(self, tmp_path: Path, settings: Settings) -> None:
        """Below CORPUS_MIN_PDF_BYTES the downloader treats a file as an error page."""
        for path in write_synthetic_versions(tmp_path).values():
            assert path.stat().st_size > settings.corpus.min_pdf_bytes

    def test_generation_is_byte_stable(self, tmp_path: Path) -> None:
        """A fixture that re-hashed per run would make every sha256 assertion useless."""
        spec = SYNTHETIC_VERSIONS[0]
        first = write_synthetic_pdf(tmp_path / "a" / spec.filename, spec).read_bytes()
        second = write_synthetic_pdf(tmp_path / "b" / spec.filename, spec).read_bytes()
        assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()

    def test_the_two_editions_have_different_content(self, tmp_path: Path) -> None:
        v1, v2 = (p.read_bytes() for p in write_synthetic_versions(tmp_path).values())
        assert hashlib.sha256(v1).hexdigest() != hashlib.sha256(v2).hexdigest()

    def test_they_carry_conflicting_recommendations(self) -> None:
        """The point of the pair: same question, different answer."""
        first, second = SYNTHETIC_VERSIONS
        assert "Drug A" in first.recommendation
        assert "Drug B" in second.recommendation
        assert first.recommendation != second.recommendation


class TestDocumentIdentity:
    def test_the_editions_share_a_root(self) -> None:
        """A version-aware retriever needs to see them as editions of one another."""
        assert all(s.document_id.startswith(DOCUMENT_ID_ROOT) for s in SYNTHETIC_VERSIONS)

    def test_the_full_document_ids_differ(self) -> None:
        first, second = SYNTHETIC_VERSIONS
        assert first.document_id != second.document_id

    def test_the_version_is_what_distinguishes_them(self) -> None:
        first, second = SYNTHETIC_VERSIONS
        assert first.version != second.version
        assert first.document_id.removeprefix(f"{DOCUMENT_ID_ROOT}_") == first.version

    def test_chunk_ids_do_not_collide_across_versions(self) -> None:
        """Position 0 of the 2020 edition is not position 0 of the 2023 edition.

        This is the property Decision B rests on: the chunk id is position-addressed,
        and version identity is carried by ``document_id`` rather than by hashing text.
        """
        first, second = SYNTHETIC_VERSIONS
        assert make_chunk_id(first.document_id, 0) != make_chunk_id(second.document_id, 0)


class TestManifestRepresentation:
    def test_both_editions_become_separate_manifest_entries(self, settings: Settings) -> None:
        written = write_synthetic_versions(settings.corpus.raw_dir)
        entries = [
            source_to_manifest_entry(
                _source_for(spec.document_id, spec.filename, spec.version),
                written[spec.document_id],
            )
            for spec in SYNTHETIC_VERSIONS
        ]
        manifest = Manifest(generated_at=datetime.now(UTC), documents=entries)

        assert len(manifest.documents) == 2
        assert {d.version_or_revision for d in manifest.documents} == {"2020", "2023"}
        assert manifest.by_id("test_amr_2020") is not None
        assert manifest.by_id("test_amr_2023") is not None

    def test_the_shared_title_does_not_merge_them(self, settings: Settings) -> None:
        """Identical titles are exactly the case a naive dedupe would collapse."""
        written = write_synthetic_versions(settings.corpus.raw_dir)
        entries = [
            source_to_manifest_entry(
                _source_for(spec.document_id, spec.filename, spec.version),
                written[spec.document_id],
            )
            for spec in SYNTHETIC_VERSIONS
        ]
        assert entries[0].title == entries[1].title
        assert entries[0].sha256 != entries[1].sha256

    def test_the_duplicate_check_does_not_fire_on_two_versions(self, settings: Settings) -> None:
        """Distinct editions must not trip the "same bytes twice" invariant."""
        written = write_synthetic_versions(settings.corpus.raw_dir)
        entries = [
            source_to_manifest_entry(
                _source_for(spec.document_id, spec.filename, spec.version),
                written[spec.document_id],
            )
            for spec in SYNTHETIC_VERSIONS
        ]
        manifest = Manifest(generated_at=datetime.now(UTC), documents=entries)
        report = verify_corpus(manifest, settings)
        assert not any("duplicate content" in i.detail for i in report.errors)

    def test_identical_bytes_under_two_ids_is_still_rejected(self, settings: Settings) -> None:
        """The invariant still holds: the *same* document twice is an error."""
        spec = SYNTHETIC_VERSIONS[0]
        first = write_synthetic_pdf(settings.corpus.raw_dir / "copy_a.pdf", spec)
        second = write_synthetic_pdf(settings.corpus.raw_dir / "copy_b.pdf", spec)
        entries = [
            source_to_manifest_entry(_source_for("test_amr_a", "copy_a.pdf", "2020"), first),
            source_to_manifest_entry(_source_for("test_amr_b", "copy_b.pdf", "2020"), second),
        ]
        with pytest.raises(ValueError, match="duplicate sha256"):
            Manifest(generated_at=datetime.now(UTC), documents=entries)


class TestFixturesStayOutOfTheRealCorpus:
    def test_no_synthetic_id_is_in_the_committed_sources(self) -> None:
        from medguard.config import project_root
        from medguard.corpus.sources import load_sources

        sources = load_sources(project_root() / "data" / "sources.yaml")
        assert not any(d.document_id.startswith(DOCUMENT_ID_ROOT) for d in sources.documents)

    def test_no_synthetic_id_is_in_the_committed_manifest(self) -> None:
        from medguard.config import project_root
        from medguard.corpus.manifest import load_manifest

        manifest = load_manifest(project_root() / "data" / "manifest.json")
        assert not any(d.document_id.startswith(DOCUMENT_ID_ROOT) for d in manifest.documents)

    def test_no_synthetic_pdf_is_committed(self) -> None:
        """They are generated into tmp_path; none may reach the repository."""
        from medguard.config import project_root

        stray = sorted((project_root() / "data" / "raw").glob("test_v*.pdf"))
        assert stray == [], f"synthetic fixtures leaked into the corpus: {stray}"

    def test_the_generator_does_not_import_a_pdf_parser(self) -> None:
        """PyMuPDF is Phase 3B. Test infrastructure must not smuggle it in early.

        Checked against the parsed import statements, not the raw text: the module
        docstring explains at length *why* PyMuPDF is absent, and a grep would trip on
        its own justification.
        """
        import ast

        from medguard.config import project_root

        source = (project_root() / "tests" / "fixtures" / "make_pdf.py").read_text(encoding="utf-8")
        imported: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

        assert imported.isdisjoint({"fitz", "pymupdf", "PyPDF2", "pypdf", "pdfplumber"})
        assert "reportlab" in imported

    def test_a_source_list_of_both_versions_validates(self) -> None:
        """Proves the pair *could* be a corpus, which is what makes the exclusion a choice."""
        sources = SourceList(
            documents=[
                _source_for(spec.document_id, spec.filename, spec.version)
                for spec in SYNTHETIC_VERSIONS
            ]
        )
        assert len(sources.documents) == 2
        assert len(sources.active) == 2
