"""The read path: query in, ranked cited candidates out.

Phase 4A ships the two single-leg retrievers and the filter they share. Hybrid fusion,
query rewriting and the HTTP surface arrive in Phase 4B; reranking in Phase 5. Import
the concrete classes from their own modules rather than eagerly here, so that importing
``medguard.retrieval`` does not pull in an ONNX runtime.
"""

from __future__ import annotations

from typing import Any

from medguard.retrieval.base import RetrievalFilter, Retriever, candidate_from_point
from medguard.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion

__all__ = [
    "HybridRetriever",
    "RetrievalFilter",
    "Retriever",
    "candidate_from_point",
    "get_retriever",
    "reciprocal_rank_fusion",
]


def get_retriever(
    kind: str,
    store: Any = None,
    settings: Any = None,
    *,
    validate_collection: bool = False,
) -> Retriever:
    """Build the named retriever from process-wide settings.

    Imports the implementation lazily: the embedders load a ~130 MB ONNX session on
    first use, and a CLI that pays that cost merely to print ``--help`` is a CLI nobody
    keeps using.

    If ``validate_collection`` is True, explicitly validates that the collection exists
    and satisfies the required schema without silently suppressing errors.
    """
    from medguard.config import get_settings
    from medguard.storage.vector_store import VectorStore

    s = settings if settings is not None else get_settings()
    vs = store if store is not None else VectorStore(s)
    if validate_collection:
        vs.validate_collection()

    if kind == "dense":
        from medguard.embeddings.fastembed_dense import get_dense_embedder
        from medguard.retrieval.dense import DenseRetriever

        return DenseRetriever(vs, get_dense_embedder(), s.retrieval)

    if kind == "sparse":
        from medguard.embeddings.fastembed_sparse import get_sparse_embedder
        from medguard.retrieval.sparse import SparseRetriever

        return SparseRetriever(vs, get_sparse_embedder(), s.retrieval)

    if kind == "hybrid":
        from medguard.embeddings.fastembed_dense import get_dense_embedder
        from medguard.embeddings.fastembed_sparse import get_sparse_embedder
        from medguard.retrieval.dense import DenseRetriever
        from medguard.retrieval.hybrid import HybridRetriever
        from medguard.retrieval.sparse import SparseRetriever

        dense = DenseRetriever(vs, get_dense_embedder(), s.retrieval)
        sparse = SparseRetriever(vs, get_sparse_embedder(), s.retrieval)
        return HybridRetriever(dense, sparse, s.retrieval)

    raise ValueError(f"unknown retriever {kind!r}; expected 'dense', 'sparse' or 'hybrid'")
