"""Retrieval contracts, and the one translation every retriever shares.

Why there is no ``RetrievedChunk`` type here
--------------------------------------------
Because :class:`~medguard.models.Candidate` already is one. It carries the chunk, the
score, the 0-based rank at the producing stage, which retriever produced it, and a
per-stage ``scores`` dict -- which is exactly the shape reciprocal-rank fusion and MRR
need, and exactly what a second parallel type would have had to grow into. Introducing
one would repeat the two-names-for-one-concept mistake this project already fought and
resolved over chunk ids.

The API renders a different shape on the wire, and that is a separate concern handled
at the API boundary. Inside the application there is one retrieval result type.

Why ``RetrievalFilter`` is not a dict
-------------------------------------
A dict of metadata constraints accepts ``{"source_org": "CDC"}`` -- a typo'd key that
filters nothing and reports nothing. The enums additionally close the value set, so a
misspelled publisher is a validation error rather than an empty result set that looks
like "the corpus does not cover that".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from qdrant_client import models as qmodels

from medguard.errors import RetrievalError
from medguard.models import Candidate, Chunk, DocumentType, RetrieverKind, SourceOrganization
from medguard.storage.vector_store import build_filter

__all__ = ["RetrievalFilter", "Retriever", "candidate_from_point"]


@dataclass(frozen=True, slots=True)
class RetrievalFilter:
    """Metadata constraints narrowing what a retriever may return.

    Values inside a field are OR-ed, fields are AND-ed. Empty or ``None`` means
    unconstrained on that field, and a filter constrained on nothing produces no Qdrant
    filter at all rather than an always-true one.

    There is no publication-date field. That is measured, not an oversight: 9 of the 20
    indexed documents state no publication date, and a Qdrant range condition drops
    every point where the field is absent -- 2,550 of 5,185 chunks -- without raising.
    See :func:`medguard.storage.vector_store.date_filter_unsupported_reason`.
    """

    source_organization: Sequence[SourceOrganization] | None = None
    document_type: Sequence[DocumentType] | None = None

    def is_empty(self) -> bool:
        return not self.source_organization and not self.document_type

    def to_qdrant(self) -> qmodels.Filter | None:
        """Lower to a Qdrant filter, passing enum *values* rather than names."""
        return build_filter(
            source_organization=[str(v) for v in self.source_organization or ()],
            document_type=[str(v) for v in self.document_type or ()],
        )


@runtime_checkable
class Retriever(Protocol):
    """Anything that turns a query string into ranked candidates.

    Deliberately narrow. Everything a retriever needs beyond the query -- the store, the
    embedder, the depth to fetch -- is supplied at construction, so that swapping dense
    for sparse for hybrid is a substitution at one call site rather than a change of
    signature at every one. The evaluation harness depends on exactly this: it runs
    whatever satisfies this protocol and does not know which leg it is measuring.
    """

    #: Cost of the most recent :meth:`retrieve`, split rather than summed. Part of the
    #: contract because the evaluation harness needs the split: on a CPU-only machine
    #: the ONNX query embed can cost more than the Qdrant search it feeds, and a single
    #: end-to-end figure would report that as slow retrieval. Settable attributes, not
    #: properties, since every implementation just assigns them.
    last_embed_ms: float
    last_search_ms: float

    @property
    def kind(self) -> RetrieverKind:
        """Which mechanism this retriever represents. Read-only; see DenseEmbedder."""
        ...

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        retrieval_filter: RetrievalFilter | None = None,
    ) -> list[Candidate]:
        """Return at most ``top_k`` candidates, best first, rank 0 first."""
        ...


#: Payload keys without which a citation cannot be rendered. A point missing any of
#: these was written by an older, incompatible indexer, and silently returning it would
#: put a chunk with no provenance into a result set that claims every item is cited.
_REQUIRED_PAYLOAD = ("chunk_id", "document_id", "text", "chunk_index")


def candidate_from_point(
    point: qmodels.ScoredPoint,
    *,
    rank: int,
    retriever: RetrieverKind,
) -> Candidate:
    """Rebuild a :class:`Candidate` from a Qdrant hit.

    The payload carries the chunk text (see ``ChunkWithVectors.payload``), so this needs
    no second lookup into SQL -- which is what keeps retrieval answerable while the
    relational store is down.

    ``scores`` is keyed by the retriever name rather than overwriting a single field,
    because the whole point of Phase 4 is comparing what each leg contributed. For a
    single-leg retriever it holds one entry; fusion in Phase 4B merges two.
    """
    payload: dict[str, Any] = dict(point.payload or {})
    missing = [k for k in _REQUIRED_PAYLOAD if payload.get(k) in (None, "")]
    if missing:
        raise RetrievalError(
            f"point {point.id} is missing required payload field(s) {missing}; the "
            f"collection was written by an incompatible indexer and cannot be cited"
        )

    chunk = Chunk(
        chunk_id=str(payload["chunk_id"]),
        document_id=str(payload["document_id"]),
        text=str(payload["text"]),
        chunk_index=int(payload["chunk_index"]),
        page_start=payload.get("page_start"),
        page_end=payload.get("page_end"),
        section=payload.get("section"),
        # Publisher-facing fields ride in metadata rather than as Chunk columns: they
        # describe the *document*, not the chunk, and duplicating them onto the chunk
        # type would invite them drifting apart from DocumentMeta.
        metadata={
            "title": payload.get("title"),
            "source_organization": payload.get("source_organization"),
            "document_type": payload.get("document_type"),
            "publication_date": payload.get("publication_date"),
        },
    )
    score = float(point.score)
    return Candidate(
        chunk=chunk,
        score=score,
        rank=rank,
        retriever=retriever,
        scores={str(retriever): score},
    )
