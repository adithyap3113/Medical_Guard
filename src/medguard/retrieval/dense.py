"""Dense retrieval: embed the query, search the ``dense`` named vector.

The baseline. Every number Phase 4B reports is a delta against this one, which is why
it is built and measured before anything more interesting exists to compare it to.
"""

from __future__ import annotations

import time

from medguard.config import RetrievalSettings, get_settings
from medguard.embeddings.base import DenseEmbedder
from medguard.errors import EmbeddingError, RetrievalError
from medguard.logging import get_logger
from medguard.models import Candidate, RetrieverKind
from medguard.retrieval.base import RetrievalFilter, candidate_from_point
from medguard.storage.vector_store import VectorStore

__all__ = ["DenseRetriever"]

log = get_logger(__name__)


class DenseRetriever:
    """Semantic nearest-neighbour retrieval over BGE-small embeddings.

    Fetches ``fetch_k`` and returns ``top_k``. In Phase 4A those are the same list
    truncated, and over-fetching buys nothing at query time -- it exists so the
    evaluation harness can ask for depth 50 and measure recall@50, which is the signal
    that says whether a Phase 5 reranker has anything to work with.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: DenseEmbedder,
        settings: RetrievalSettings | None = None,
    ) -> None:
        self._store = vector_store
        self._embedder = embedder
        self._settings = settings if settings is not None else get_settings().retrieval
        #: Populated by the most recent :meth:`retrieve`, in milliseconds. Read by the
        #: evaluation runner, which needs the embed/search split: on a CPU-only machine
        #: the ONNX query embed can cost more than the search it feeds, and a single
        #: end-to-end number would attribute that to Qdrant.
        self.last_embed_ms: float = 0.0
        self.last_search_ms: float = 0.0

    @property
    def kind(self) -> RetrieverKind:
        return RetrieverKind.DENSE

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
                f"could not embed query for dense retrieval: {type(exc).__name__}: {exc}"
            ) from exc
        self.last_embed_ms = (time.perf_counter() - started) * 1000.0

        depth = max(top_k, self._settings.fetch_k)
        qfilter = retrieval_filter.to_qdrant() if retrieval_filter is not None else None

        started = time.perf_counter()
        points = self._store.search_dense(vector, limit=depth, query_filter=qfilter)
        self.last_search_ms = (time.perf_counter() - started) * 1000.0

        log.debug(
            "retrieval.dense",
            hits=len(points),
            depth=depth,
            top_k=top_k,
            filtered=qfilter is not None,
            embed_ms=round(self.last_embed_ms, 2),
            search_ms=round(self.last_search_ms, 2),
        )
        return [
            candidate_from_point(point, rank=rank, retriever=RetrieverKind.DENSE)
            for rank, point in enumerate(points[:top_k])
        ]
