"""Hybrid retrieval: Reciprocal Rank Fusion (RRF) over dense and sparse legs.

Fuses semantic similarity (dense BGE-small embeddings) with exact lexical match
(sparse BM25) into a unified ranked candidate pool.

Why Reciprocal Rank Fusion (RRF) instead of score normalization:
--------------------------------------------------------------
Dense cosine similarities reside in [-1, 1], while BM25 produces unbounded
IDF-weighted term frequency sums that vary wildly by document length and term rarity.
Linear combination of raw or min-max normalized scores is brittle across differing queries.
RRF operates strictly on rank positions:
    score(d) = sum_{m in models} 1.0 / (k_rrf + rank_m(d) + 1)
where rank is 0-indexed and k_rrf (typically 60) dampens the impact of high ranks.
"""

from __future__ import annotations

import time

from medguard.config import RetrievalSettings, get_settings
from medguard.errors import RetrievalError
from medguard.logging import get_logger
from medguard.models import Candidate, Chunk, RetrieverKind
from medguard.retrieval.base import RetrievalFilter
from medguard.retrieval.dense import DenseRetriever
from medguard.retrieval.sparse import SparseRetriever

__all__ = ["HybridRetriever", "reciprocal_rank_fusion"]

log = get_logger(__name__)


def reciprocal_rank_fusion(
    dense_candidates: list[Candidate],
    sparse_candidates: list[Candidate],
    *,
    rrf_k: int = 60,
    top_k: int = 5,
) -> list[Candidate]:
    """Merge ranked dense and sparse candidate lists via Reciprocal Rank Fusion.

    Preserves per-leg scores in each Candidate's ``scores`` dictionary.
    Ranks are 0-based; the formula adds 1 so rank 0 contributes 1 / (k + 1).
    """
    if rrf_k < 1:
        raise ValueError(f"rrf_k must be >= 1, got {rrf_k}")

    rrf_scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    leg_scores: dict[str, dict[str, float]] = {}

    for rank, cand in enumerate(dense_candidates):
        cid = cand.chunk_id
        chunks[cid] = cand.chunk
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))
        if cid not in leg_scores:
            leg_scores[cid] = {}
        leg_scores[cid]["dense"] = cand.score

    for rank, cand in enumerate(sparse_candidates):
        cid = cand.chunk_id
        chunks[cid] = cand.chunk
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))
        if cid not in leg_scores:
            leg_scores[cid] = {}
        leg_scores[cid]["sparse"] = cand.score

    # Sort descending by RRF score, breaking ties deterministically by chunk_id
    sorted_ids = sorted(
        rrf_scores.keys(),
        key=lambda cid: (-rrf_scores[cid], cid),
    )

    fused: list[Candidate] = []
    for rank, cid in enumerate(sorted_ids[:top_k]):
        scores = dict(leg_scores[cid])
        scores["rrf"] = rrf_scores[cid]
        fused.append(
            Candidate(
                chunk=chunks[cid],
                score=rrf_scores[cid],
                rank=rank,
                retriever=RetrieverKind.HYBRID,
                scores=scores,
            )
        )

    return fused


class HybridRetriever:
    """Hybrid search combining DenseRetriever and SparseRetriever via RRF."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        settings: RetrievalSettings | None = None,
    ) -> None:
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._settings = settings if settings is not None else get_settings().retrieval
        self.last_embed_ms: float = 0.0
        self.last_search_ms: float = 0.0

    @property
    def kind(self) -> RetrieverKind:
        return RetrieverKind.HYBRID

    @property
    def dense_retriever(self) -> DenseRetriever:
        return self._dense

    @property
    def sparse_retriever(self) -> SparseRetriever:
        return self._sparse

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

        depth = max(top_k, self._settings.fetch_k)

        started = time.perf_counter()
        dense_candidates = self._dense.retrieve(
            query, top_k=depth, retrieval_filter=retrieval_filter
        )
        sparse_candidates = self._sparse.retrieve(
            query, top_k=depth, retrieval_filter=retrieval_filter
        )

        self.last_embed_ms = self._dense.last_embed_ms + self._sparse.last_embed_ms
        self.last_search_ms = (
            self._dense.last_search_ms
            + self._sparse.last_search_ms
            + (time.perf_counter() - started) * 1000.0
        )

        rrf_k = getattr(self._settings, "rrf_k", 60)
        fused = reciprocal_rank_fusion(
            dense_candidates,
            sparse_candidates,
            rrf_k=rrf_k,
            top_k=top_k,
        )

        log.debug(
            "retrieval.hybrid",
            dense_hits=len(dense_candidates),
            sparse_hits=len(sparse_candidates),
            fused_count=len(fused),
            top_k=top_k,
            depth=depth,
            rrf_k=rrf_k,
            embed_ms=round(self.last_embed_ms, 2),
            search_ms=round(self.last_search_ms, 2),
        )

        return fused
