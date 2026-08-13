"""Core domain models.

Why typed models instead of dicts
---------------------------------
A RAG pipeline is a chain of transformations: document -> chunk -> candidate ->
reranked candidate -> citation -> answer. If each stage passes a bare ``dict``, then
a typo in a key surfaces as a ``KeyError`` three stages downstream, retrieval scores
silently become strings, and no editor can tell you what a function returns.

These models are the contracts between stages. They are also what makes Phase 12
evaluation possible: you cannot compute nDCG over an object that does not reliably
record which retriever produced a result and what it scored.

Nothing here loads a model, embeds text, or talks to a database. This module is
pure data, so it stays importable from tests with no infrastructure running.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

__all__ = [
    "MEDICAL_DISCLAIMER",
    "Answer",
    "Candidate",
    "Chunk",
    "Citation",
    "DocumentMeta",
    "QueryStrategy",
    "RetrievalResult",
    "RetrieverKind",
]

#: Shown with every generated answer, in the API, and in the CLI.
#:
#: This wording is deliberate. MedGuard retrieves and summarises text from an indexed
#: corpus of published guidance. It does not diagnose, and it has not been clinically
#: validated. The disclaimer states what the system actually does rather than making
#: a safety claim the project cannot support.
MEDICAL_DISCLAIMER = (
    "This response is generated from indexed medical reference documents and is for "
    "informational purposes only. It is not medical advice, not a diagnosis, and not a "
    "substitute for consultation with a qualified healthcare professional. Verify all "
    "information against the cited sources before acting on it."
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class RetrieverKind(StrEnum):
    """Which retrieval mechanism produced a candidate.

    Recorded per-candidate rather than per-query because hybrid search merges results
    from several retrievers. Without this field, you cannot answer the question every
    interviewer asks: "how much did the sparse leg actually contribute?"
    """

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    RERANKED = "reranked"


class QueryStrategy(StrEnum):
    """Query-processing strategy, selected by the router.

    ``BASIC`` is the mandatory baseline. Every other strategy must be measured
    against it before it earns a place in the default path.
    """

    BASIC = "basic"
    REWRITE = "rewrite"
    HYDE = "hyde"
    MULTI_QUERY = "multi_query"
    DECOMPOSITION = "decomposition"


class DocumentMeta(BaseModel):
    """Provenance for one source document.

    Provenance is a correctness requirement, not bookkeeping: a citation that cannot
    name its source document, publisher and version is not a citation. ``sha256``
    additionally makes re-ingestion idempotent and detects a source changing under us.
    """

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(description="Stable identifier, typically the filename stem.")
    title: str
    source_path: str = Field(description="Path or URL the document was ingested from.")
    publisher: str | None = Field(default=None, description="e.g. 'WHO', 'NIH'.")
    published_year: int | None = Field(default=None, ge=1800, le=2200)
    licence: str | None = Field(default=None, description="Redistribution terms of the source.")
    sha256: str | None = Field(default=None, description="Content hash of the source file.")
    page_count: int | None = Field(default=None, ge=0)
    ingested_at: datetime = Field(default_factory=_utcnow)
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """One retrievable unit of text, with enough context to cite it.

    ``page_start``/``page_end`` are what let a citation point a clinician at a page
    rather than at a 400-page PDF, so they are carried from ingestion onward rather
    than reconstructed later.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    text: str = Field(min_length=1)
    chunk_index: int = Field(ge=0, description="Ordinal position within the document.")
    page_start: int | None = Field(default=None, ge=0)
    page_end: int | None = Field(default=None, ge=0)
    section: str | None = Field(default=None, description="Nearest enclosing heading.")
    token_count: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_pages(self) -> Chunk:
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            msg = f"page_end ({self.page_end}) < page_start ({self.page_start})"
            raise ValueError(msg)
        return self

    @staticmethod
    def make_id(document_id: str, chunk_index: int, text: str) -> str:
        """Derive a deterministic chunk id from content.

        Content-addressed rather than random, so re-ingesting an unchanged document
        produces identical ids and upserts become idempotent. Re-running ingestion
        therefore cannot silently duplicate the corpus -- a failure mode that quietly
        destroys retrieval metrics.
        """
        digest = hashlib.blake2b(
            f"{document_id}:{chunk_index}:{text}".encode(), digest_size=8
        ).hexdigest()
        return f"{document_id}:{chunk_index}:{digest}"


class Candidate(BaseModel):
    """A chunk retrieved for a query, with its score and origin.

    ``scores`` keeps every stage's raw score (``{"dense": 0.82, "bm25": 11.4,
    "rerank": 0.97}``) instead of overwriting one field, because comparing stages is
    the entire point of the evaluation phase. ``rank`` is the position at the stage
    that produced this object, needed for reciprocal-rank fusion and for MRR.
    """

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float = Field(description="Score from the stage that produced this candidate.")
    rank: int = Field(ge=0, description="0-based rank at the producing stage.")
    retriever: RetrieverKind
    scores: dict[str, float] = Field(
        default_factory=dict, description="Per-stage scores retained for analysis."
    )

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def document_id(self) -> str:
        return self.chunk.document_id


class RetrievalResult(BaseModel):
    """The full outcome of retrieving for one query.

    Carries the strategy and timing alongside the candidates so that a stored result
    is self-describing: an evaluation run recorded last week can still say which
    strategy produced it. Without that, experiments are not reproducible.
    """

    query: str
    candidates: list[Candidate] = Field(default_factory=list)
    strategy: QueryStrategy = QueryStrategy.BASIC
    retriever: RetrieverKind = RetrieverKind.DENSE
    expanded_queries: list[str] = Field(
        default_factory=list,
        description="Sub-queries or hypothetical documents generated by the strategy.",
    )
    latency_ms: float | None = Field(default=None, ge=0)
    total_candidates_considered: int = Field(
        default=0, ge=0, description="Pool size before top-k truncation."
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def returned_count(self) -> int:
        return len(self.candidates)

    def top_k(self, k: int) -> list[Candidate]:
        return self.candidates[:k]

    def unique_documents(self) -> list[str]:
        """Distinct source documents, preserving rank order."""
        return list(dict.fromkeys(c.document_id for c in self.candidates))


class Citation(BaseModel):
    """A link from a span of the answer back to the chunk that supports it.

    ``supported`` and ``validation_note`` exist because Phase 11 *validates* citations
    rather than trusting them. A model can emit a well-formed reference to a chunk
    that does not actually support the claim; that is a fabricated citation even
    though the chunk id is real. Recording the verdict keeps the distinction visible
    instead of assuming every emitted citation is sound.
    """

    model_config = ConfigDict(frozen=True)

    marker: str = Field(description="Inline marker as it appears in the answer, e.g. '[1]'.")
    chunk_id: str
    document_id: str
    document_title: str | None = None
    page_start: int | None = Field(default=None, ge=0)
    page_end: int | None = Field(default=None, ge=0)
    quote: str | None = Field(
        default=None, description="Verbatim supporting span from the source chunk."
    )
    supported: bool | None = Field(
        default=None,
        description="Validation verdict. None means not yet validated -- not 'valid'.",
    )
    validation_note: str | None = None


class Answer(BaseModel):
    """A generated answer plus everything needed to audit it.

    ``abstained`` is a first-class outcome. For a medical corpus, "the indexed sources
    do not cover this" is a correct and desirable response, and the evaluation set
    contains unanswerable questions specifically to measure it. Treating abstention as
    a failure would push the system toward confident fabrication.
    """

    question: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    retrieval: RetrievalResult | None = None
    model: str | None = Field(default=None, description="Generation model that produced this.")
    strategy: QueryStrategy = QueryStrategy.BASIC
    abstained: bool = Field(
        default=False, description="True when the system declined for lack of grounding."
    )
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    trace_id: str | None = Field(default=None, description="Langfuse trace id, when tracing is on.")
    created_at: datetime = Field(default_factory=_utcnow)
    disclaimer: str = Field(default=MEDICAL_DISCLAIMER)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def citation_count(self) -> int:
        return len(self.citations)

    @property
    def cited_document_ids(self) -> list[str]:
        return list(dict.fromkeys(c.document_id for c in self.citations))
