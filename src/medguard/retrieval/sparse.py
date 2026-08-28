"""Sparse retrieval: BM25 over the ``bm25`` named vector.

Structurally a mirror of :mod:`medguard.retrieval.dense`, and kept as a separate class
rather than a mode flag for the same reason the store has two search methods: the query
types genuinely differ, and a shared implementation would have to branch on the vector
type at the one place where getting it wrong is silent.

What differs from dense, and matters when reading the numbers: BM25 scores are
unbounded IDF-weighted sums, not cosine similarities in [-1, 1]. They are comparable
*within* one query's result list and meaningless across queries, which is precisely why
fusion in Phase 4B will operate on ranks rather than on these scores.
"""

from __future__ import annotations

import time

from medguard.config import RetrievalSettings, get_settings
from medguard.embeddings.base import SparseEmbedder
from medguard.errors import EmbeddingError, RetrievalError
from medguard.logging import get_logger
from medguard.models import Candidate, RetrieverKind
from medguard.retrieval.base import RetrievalFilter, candidate_from_point
from medguard.storage.vector_store import VectorStore

__all__ = ["SparseRetriever"]

log = get_logger(__name__)


class SparseRetriever:
    """Lexical retrieval, for the tokens dense embeddings blur together.

    Drug names, ``12.5 mg``, ``HbA1c``, ``ceftriaxone`` versus ``cefotaxime``: a dense
    model places those close together because they *are* close in meaning, which is
    exactly wrong when the question is which one to give.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: SparseEmbedder,
        settings: RetrievalSettings | None = None,
    ) -> None:
        self._store = vector_store
        self._embedder = embedder
        self._settings = settings if settings is not None else get_settings().retrieval
        self.last_embed_ms: float = 0.0
        self.last_search_ms: float = 0.0

    @property
    def kind(self) -> RetrieverKind:
        return RetrieverKind.SPARSE

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        retrieval_filter: RetrievalFilter | None = None,
    ) -> list[Candidate]:
        if not query.strip():
            raise RetrievalError("query is empty; there is nothing to retrieve")
        if top_k < 1:
            raise RetrievalError(f"top_k must be >= 1, got {top_k}")

        started = time.perf_counter()
        try:
            vector = self._embedder.embed_query(query)
        except EmbeddingError:
            raise
        # Anything the embedder raises that is not already an EmbeddingError is
        # still a failure to retrieve, and the caller should see it as one.
        except Exception as exc:
            raise RetrievalError(
                f"could not embed query for sparse retrieval: {type(exc).__name__}: {exc}"
            ) from exc
        self.last_embed_ms = (time.perf_counter() - started) * 1000.0

        depth = max(top_k, self._settings.fetch_k)
        qfilter = retrieval_filter.to_qdrant() if retrieval_filter is not None else None

        started = time.perf_counter()
        points = self._store.search_sparse(vector, limit=depth, query_filter=qfilter)
        self.last_search_ms = (time.perf_counter() - started) * 1000.0

        if not vector.indices:
            # Every term was dropped by the tokeniser. Worth saying out loud: it is the
            # one case where an empty result means "unanswerable query" rather than
            # "nothing in the corpus matched".
            log.info("retrieval.sparse.no_terms", query_length=len(query))

        log.debug(
            "retrieval.sparse",
            hits=len(points),
            depth=depth,
            top_k=top_k,
            terms=len(vector.indices),
            filtered=qfilter is not None,
            embed_ms=round(self.last_embed_ms, 2),
            search_ms=round(self.last_search_ms, 2),
        )
        return [
            candidate_from_point(point, rank=rank, retriever=RetrieverKind.SPARSE)
            for rank, point in enumerate(points[:top_k])
        ]
