"""Tests for HybridRetriever and Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import pytest

from medguard.config import Settings
from medguard.errors import RetrievalError
from medguard.models import Candidate, Chunk, RetrieverKind
from medguard.retrieval.dense import DenseRetriever
from medguard.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from medguard.retrieval.sparse import SparseRetriever
from medguard.storage.vector_store import VectorStore
from tests.unit.test_retrieval import StubDense, StubSparse, _chunk


def _candidate(chunk_id: str, score: float, rank: int, kind: RetrieverKind) -> Candidate:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id=chunk_id.split("::")[0],
        chunk_index=int(chunk_id.split("::")[1]) if "::" in chunk_id else 0,
        text=f"Sample text for {chunk_id}",
        page_start=1,
        page_end=1,
    )
    return Candidate(
        chunk=chunk,
        score=score,
        rank=rank,
        retriever=kind,
        scores={str(kind): score},
    )


class TestReciprocalRankFusion:
    def test_candidates_in_both_legs_rank_highest(self) -> None:
        c1 = _candidate("doc::0001", 0.9, 0, RetrieverKind.DENSE)
        c2 = _candidate("doc::0002", 0.8, 1, RetrieverKind.DENSE)

        s2 = _candidate("doc::0002", 12.0, 0, RetrieverKind.SPARSE)
        s3 = _candidate("doc::0003", 8.0, 1, RetrieverKind.SPARSE)

        fused = reciprocal_rank_fusion([c1, c2], [s2, s3], rrf_k=60, top_k=5)

        # doc::0002 appears in both legs:
        # rank 1 in dense (1/62) + rank 0 in sparse (1/61) = ~0.0325
        # doc::0001: 1/61 = ~0.0164
        # doc::0003: 1/62 = ~0.0161
        assert len(fused) == 3
        assert fused[0].chunk_id == "doc::0002"
        assert fused[0].rank == 0
        assert fused[0].retriever is RetrieverKind.HYBRID
        assert "dense" in fused[0].scores
        assert "sparse" in fused[0].scores
        assert "rrf" in fused[0].scores
        assert fused[0].scores["dense"] == 0.8
        assert fused[0].scores["sparse"] == 12.0

        assert fused[1].chunk_id == "doc::0001"
        assert fused[2].chunk_id == "doc::0003"

    def test_top_k_truncates_output(self) -> None:
        dense = [
            _candidate(f"doc::{i:04d}", 1.0 - 0.1 * i, i, RetrieverKind.DENSE) for i in range(10)
        ]
        sparse = [_candidate(f"doc::{i:04d}", 10.0 - i, i, RetrieverKind.SPARSE) for i in range(10)]

        fused = reciprocal_rank_fusion(dense, sparse, rrf_k=60, top_k=3)
        assert len(fused) == 3
        assert [c.rank for c in fused] == [0, 1, 2]

    def test_invalid_rrf_k_raises(self) -> None:
        with pytest.raises(ValueError, match="rrf_k must be >= 1"):
            reciprocal_rank_fusion([], [], rrf_k=0)

    def test_empty_inputs_return_empty(self) -> None:
        assert reciprocal_rank_fusion([], []) == []


class TestHybridRetriever:
    @pytest.fixture
    def populated_store(self, settings: Settings):  # type: ignore[no-untyped-def]
        store = VectorStore(settings)
        store.ensure_collection()
        store.upsert_chunks([_chunk("cdc_doc", i, "CDC", "guideline") for i in range(5)])
        yield store
        store.close()

    def test_retrieve_combines_dense_and_sparse(
        self, populated_store: VectorStore, settings: Settings
    ) -> None:
        dense_ret = DenseRetriever(populated_store, StubDense(), settings.retrieval)
        sparse_ret = SparseRetriever(populated_store, StubSparse(), settings.retrieval)
        hybrid_ret = HybridRetriever(dense_ret, sparse_ret, settings.retrieval)

        assert hybrid_ret.kind is RetrieverKind.HYBRID
        assert hybrid_ret.dense_retriever is dense_ret
        assert hybrid_ret.sparse_retriever is sparse_ret

        results = hybrid_ret.retrieve("aspirin guideline", top_k=3)
        assert len(results) <= 3
        for rank, cand in enumerate(results):
            assert cand.rank == rank
            assert cand.retriever is RetrieverKind.HYBRID
            assert "rrf" in cand.scores

        assert hybrid_ret.last_embed_ms >= 0.0
        assert hybrid_ret.last_search_ms >= 0.0

    def test_empty_query_raises(self, populated_store: VectorStore, settings: Settings) -> None:
        dense_ret = DenseRetriever(populated_store, StubDense(), settings.retrieval)
        sparse_ret = SparseRetriever(populated_store, StubSparse(), settings.retrieval)
        hybrid_ret = HybridRetriever(dense_ret, sparse_ret, settings.retrieval)

        with pytest.raises(RetrievalError, match="query is empty"):
            hybrid_ret.retrieve("   ", top_k=5)

    def test_invalid_top_k_raises(self, populated_store: VectorStore, settings: Settings) -> None:
        dense_ret = DenseRetriever(populated_store, StubDense(), settings.retrieval)
        sparse_ret = SparseRetriever(populated_store, StubSparse(), settings.retrieval)
        hybrid_ret = HybridRetriever(dense_ret, sparse_ret, settings.retrieval)

        with pytest.raises(RetrievalError, match="top_k must be >= 1"):
            hybrid_ret.retrieve("test", top_k=0)
