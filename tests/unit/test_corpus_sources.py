"""Corpus source-list parsing and validation.

The contract worth pinning down: a malformed entry must fail *here*, before any
network traffic. An entry that validates loosely and fails later produces a
half-downloaded corpus whose manifest disagrees with the files on disk, and untangling
that costs far more than a strict parser does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from medguard.corpus.sources import SourceDocument, SourceList, load_sources
from medguard.errors import CorpusError
from medguard.models import DocumentType, SourceOrganization


def _doc(**overrides: object) -> dict[str, object]:
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
    return base


class TestSourceDocument:
    def test_parses_a_well_formed_entry(self) -> None:
        doc = SourceDocument.model_validate(_doc(publication_date="2024-03-01"))
        assert doc.source_organization is SourceOrganization.CDC
        assert doc.document_type is DocumentType.GUIDELINE
        assert doc.publication_date is not None
        assert doc.publication_date.year == 2024

    def test_publication_date_defaults_to_none(self) -> None:
        """Absent means 'the publisher states no date', which must not become a guess."""
        assert SourceDocument.model_validate(_doc()).publication_date is None

    @pytest.mark.parametrize(
        "bad_id",
        ["Has-Caps", "with space", "hy-phen", "ab", "", "9" * 65],
    )
    def test_rejects_ids_that_would_need_quoting_downstream(self, bad_id: str) -> None:
        """Ids become filenames, primary keys and chunk id prefixes.

        Restricting the character set once here avoids quoting bugs in four places.
        """
        with pytest.raises(ValueError, match="document_id"):
            SourceDocument.model_validate(_doc(document_id=bad_id))

    @pytest.mark.parametrize("bad_url", ["ftp://x/a.pdf", "example.invalid/a.pdf", "/local/a.pdf"])
    def test_rejects_non_http_urls(self, bad_url: str) -> None:
        with pytest.raises(ValueError, match="source_url"):
            SourceDocument.model_validate(_doc(source_url=bad_url))

    def test_rejects_non_pdf_filenames(self) -> None:
        with pytest.raises(ValueError, match=r"must end in \.pdf"):
            SourceDocument.model_validate(_doc(local_filename="notes.txt"))

    @pytest.mark.parametrize(
        "bad_name", ["../escape.pdf", "sub/dir.pdf", "sub\\dir.pdf", ".hidden.pdf"]
    )
    def test_filename_cannot_escape_the_corpus_directory(self, bad_name: str) -> None:
        """A path separator here would write outside data/raw/ on download."""
        with pytest.raises(ValueError, match="bare filename"):
            SourceDocument.model_validate(_doc(local_filename=bad_name))

    def test_unknown_organisation_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="source_organization"):
            SourceDocument.model_validate(_doc(source_organization="RandomBlog"))

    def test_unknown_document_type_is_rejected(self) -> None:
        """'unknown' is not a document type. Every intentional addition is classified."""
        with pytest.raises(ValueError, match="document_type"):
            SourceDocument.model_validate(_doc(document_type="unknown"))

    def test_unexpected_field_is_rejected(self) -> None:
        """extra='forbid' catches a typo'd key instead of silently ignoring it."""
        with pytest.raises(ValueError, match="publisher"):
            SourceDocument.model_validate(_doc(publisher="CDC"))

    def test_retired_entry_must_state_a_reason(self) -> None:
        """A retired entry with no reason is an unexplained gap in the corpus record."""
        with pytest.raises(ValueError, match="retired_reason"):
            SourceDocument.model_validate(_doc(retired=True))

    def test_retired_entry_with_a_reason_is_accepted(self) -> None:
        doc = SourceDocument.model_validate(_doc(retired=True, retired_reason="URL withdrawn."))
        assert doc.retired is True


class TestSourceList:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("document_id", "cdc_example_2024"),
            ("local_filename", "cdc_example_2024.pdf"),
            ("source_url", "https://example.invalid/a.pdf"),
        ],
    )
    def test_rejects_duplicates(self, field: str, value: str) -> None:
        """Two entries sharing any of these means one silently overwrites the other."""
        second = _doc(
            document_id="cdc_other_2024",
            local_filename="cdc_other_2024.pdf",
            source_url="https://example.invalid/b.pdf",
        )
        second[field] = value
        with pytest.raises(ValueError, match=f"duplicate {field}"):
            SourceList.model_validate({"documents": [_doc(), second]})

    def test_active_excludes_retired_entries(self) -> None:
        sources = SourceList.model_validate(
            {
                "documents": [
                    _doc(),
                    _doc(
                        document_id="who_old_2019",
                        local_filename="who_old_2019.pdf",
                        source_url="https://example.invalid/b.pdf",
                        retired=True,
                        retired_reason="URL withdrawn.",
                    ),
                ]
            }
        )
        assert [d.document_id for d in sources.active] == ["cdc_example_2024"]
        assert len(sources.documents) == 2

    def test_by_id_finds_retired_entries_too(self) -> None:
        """Retired means 'do not download', not 'not part of the corpus'."""
        sources = SourceList.model_validate(
            {
                "documents": [
                    _doc(retired=True, retired_reason="URL withdrawn."),
                ]
            }
        )
        assert sources.by_id("cdc_example_2024") is not None
        assert sources.by_id("nope") is None

    def test_empty_document_list_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="documents"):
            SourceList.model_validate({"documents": []})


class TestLoadSources:
    def test_loads_a_valid_file(self, tmp_path: Path, sources_yaml: str) -> None:
        path = tmp_path / "sources.yaml"
        path.write_text(sources_yaml, encoding="utf-8")
        sources = load_sources(path)
        assert len(sources.documents) == 2
        assert len(sources.active) == 1

    def test_missing_file_raises_corpus_error(self, tmp_path: Path) -> None:
        """One exception type for callers, not three from three libraries."""
        with pytest.raises(CorpusError, match="not found"):
            load_sources(tmp_path / "absent.yaml")

    def test_invalid_yaml_raises_corpus_error(self, tmp_path: Path) -> None:
        path = tmp_path / "sources.yaml"
        path.write_text("documents: [unclosed\n", encoding="utf-8")
        with pytest.raises(CorpusError, match="not valid YAML"):
            load_sources(path)

    def test_non_mapping_top_level_raises_corpus_error(self, tmp_path: Path) -> None:
        path = tmp_path / "sources.yaml"
        path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(CorpusError, match="mapping at the top level"):
            load_sources(path)

    def test_schema_violation_raises_corpus_error(self, tmp_path: Path) -> None:
        path = tmp_path / "sources.yaml"
        path.write_text(
            "documents:\n  - document_id: x\n    title: Too short an id\n", encoding="utf-8"
        )
        with pytest.raises(CorpusError, match="failed validation"):
            load_sources(path)


class TestShippedCorpus:
    """The real data/sources.yaml must stay loadable and guideline-shaped."""

    def test_the_committed_corpus_definition_is_valid(self) -> None:
        from medguard.config import project_root

        sources = load_sources(project_root() / "data" / "sources.yaml")
        assert len(sources.documents) >= 10

    def test_every_retired_entry_explains_itself(self) -> None:
        from medguard.config import project_root

        sources = load_sources(project_root() / "data" / "sources.yaml")
        for doc in sources.documents:
            if doc.retired:
                assert doc.retired_reason and len(doc.retired_reason) > 20

    def test_the_corpus_is_predominantly_guidelines(self) -> None:
        """Phase 2 chose a guideline-shaped corpus deliberately; keep it that way.

        A corpus that drifts toward primary research answers clinical questions from
        single studies, which no amount of reranking fixes.
        """
        from medguard.config import project_root

        sources = load_sources(project_root() / "data" / "sources.yaml")
        guidelines = [d for d in sources.documents if d.document_type is DocumentType.GUIDELINE]
        assert len(guidelines) / len(sources.documents) >= 0.75
