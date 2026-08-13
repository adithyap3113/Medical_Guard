"""Domain model tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from medguard.models import (
    MEDICAL_DISCLAIMER,
    Answer,
    Candidate,
    Chunk,
    Citation,
    DocumentMeta,
    QueryStrategy,
    RetrievalResult,
    RetrieverKind,
)


def make_chunk(
    doc: str = "who_hypertension_2021", index: int = 0, text: str = "Some text."
) -> Chunk:
    return Chunk(
        chunk_id=Chunk.make_id(doc, index, text),
        document_id=doc,
        text=text,
        chunk_index=index,
        page_start=3,
        page_end=4,
    )


class TestChunk:
    def test_ids_are_deterministic(self) -> None:
        """Re-ingesting unchanged content must produce identical ids.

        This is what makes ingestion idempotent; random ids would let a second run
        silently duplicate the corpus and quietly distort every retrieval metric.
        """
        a = Chunk.make_id("doc", 3, "identical text")
        b = Chunk.make_id("doc", 3, "identical text")
        assert a == b

    def test_changed_text_changes_the_id(self) -> None:
        assert Chunk.make_id("doc", 3, "original") != Chunk.make_id("doc", 3, "edited")

    def test_changed_position_changes_the_id(self) -> None:
        assert Chunk.make_id("doc", 3, "text") != Chunk.make_id("doc", 4, "text")

    def test_empty_text_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Chunk(chunk_id="x", document_id="d", text="", chunk_index=0)

    def test_inverted_page_range_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="page_end"):
            Chunk(chunk_id="x", document_id="d", text="t", chunk_index=0, page_start=9, page_end=2)

    def test_chunks_are_immutable(self) -> None:
        """Frozen so a retrieval stage cannot mutate a chunk another stage still holds."""
        chunk = make_chunk()
        with pytest.raises(ValidationError):
            chunk.text = "mutated"


class TestDocumentMeta:
    def test_minimal_document_is_valid(self) -> None:
        meta = DocumentMeta(document_id="d", title="T", source_path="data/raw/d.pdf")
        assert meta.ingested_at.tzinfo is not None, "timestamps must be timezone-aware"

    def test_implausible_year_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DocumentMeta(document_id="d", title="T", source_path="p", published_year=1200)


class TestCandidate:
    def test_per_stage_scores_are_retained(self) -> None:
        """Keeping every stage's score is what makes Phase 12 comparisons possible."""
        candidate = Candidate(
            chunk=make_chunk(),
            score=0.91,
            rank=0,
            retriever=RetrieverKind.RERANKED,
            scores={"dense": 0.62, "bm25": 8.4, "rrf": 0.031, "rerank": 0.91},
        )
        assert candidate.scores["bm25"] == 8.4
        assert candidate.document_id == "who_hypertension_2021"

    def test_negative_rank_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Candidate(chunk=make_chunk(), score=0.5, rank=-1, retriever=RetrieverKind.DENSE)


class TestRetrievalResult:
    def _result(self) -> RetrievalResult:
        chunks = [make_chunk("doc_a", 0), make_chunk("doc_a", 1), make_chunk("doc_b", 0)]
        return RetrievalResult(
            query="what is the first-line treatment for hypertension?",
            candidates=[
                Candidate(chunk=c, score=1.0 - i / 10, rank=i, retriever=RetrieverKind.HYBRID)
                for i, c in enumerate(chunks)
            ],
            strategy=QueryStrategy.HYDE,
            retriever=RetrieverKind.HYBRID,
        )

    def test_returned_count_is_computed(self) -> None:
        assert self._result().returned_count == 3

    def test_top_k_truncates(self) -> None:
        assert len(self._result().top_k(2)) == 2

    def test_unique_documents_preserves_rank_order(self) -> None:
        assert self._result().unique_documents() == ["doc_a", "doc_b"]

    def test_strategy_is_recorded_for_reproducibility(self) -> None:
        assert self._result().strategy is QueryStrategy.HYDE

    def test_default_strategy_is_the_baseline(self) -> None:
        assert RetrievalResult(query="q").strategy is QueryStrategy.BASIC


class TestCitation:
    def test_unvalidated_citation_is_none_not_true(self) -> None:
        """`None` means "not checked". Defaulting to True would assume validity."""
        citation = Citation(marker="[1]", chunk_id="c", document_id="d")
        assert citation.supported is None

    def test_validation_verdict_is_recordable(self) -> None:
        citation = Citation(
            marker="[1]",
            chunk_id="c",
            document_id="d",
            supported=False,
            validation_note="quote not found in the cited chunk",
        )
        assert citation.supported is False


class TestAnswer:
    def test_disclaimer_is_attached_by_default(self) -> None:
        """The disclaimer must not depend on any caller remembering to add it."""
        answer = Answer(question="q", text="a")
        assert answer.disclaimer == MEDICAL_DISCLAIMER

    def test_disclaimer_states_it_is_not_a_diagnosis(self) -> None:
        lowered = MEDICAL_DISCLAIMER.lower()
        assert "not medical advice" in lowered
        assert "not a diagnosis" in lowered
        assert "qualified healthcare professional" in lowered

    def test_disclaimer_makes_no_safety_or_validation_claim(self) -> None:
        """The project must not claim clinical validation it does not have."""
        lowered = MEDICAL_DISCLAIMER.lower()
        for forbidden in ("clinically validated", "medically safe", "fda", "approved"):
            assert forbidden not in lowered

    def test_abstention_is_a_first_class_outcome(self) -> None:
        """ "The sources do not cover this" is a correct answer, not a failure."""
        answer = Answer(
            question="unanswerable from the corpus",
            text="The indexed sources do not contain information on this.",
            abstained=True,
        )
        assert answer.abstained is True
        assert answer.citation_count == 0

    def test_cited_document_ids_are_deduplicated(self) -> None:
        answer = Answer(
            question="q",
            text="a [1][2][3]",
            citations=[
                Citation(marker="[1]", chunk_id="c1", document_id="doc_a"),
                Citation(marker="[2]", chunk_id="c2", document_id="doc_a"),
                Citation(marker="[3]", chunk_id="c3", document_id="doc_b"),
            ],
        )
        assert answer.citation_count == 3
        assert answer.cited_document_ids == ["doc_a", "doc_b"]
