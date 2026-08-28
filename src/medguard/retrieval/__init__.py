"""The read path: query in, ranked cited candidates out.

Phase 4A ships the two single-leg retrievers and the filter they share. Hybrid fusion,
query rewriting and the HTTP surface arrive in Phase 4B; reranking in Phase 5. Import
the concrete classes from their own modules rather than eagerly here, so that importing
``medguard.retrieval`` does not pull in an ONNX runtime.
"""

from __future__ import annotations

from medguard.retrieval.base import RetrievalFilter, Retriever, candidate_from_point

__all__ = ["RetrievalFilter", "Retriever", "candidate_from_point", "get_retriever"]


def get_retriever(kind: str) -> Retriever:
    """Build the named retriever from process-wide settings.

    Imports the implementation lazily: the embedders load a ~130 MB ONNX session on
    first use, and a CLI that pays that cost merely to print ``--help`` is a CLI nobody
    keeps using.
    """
    from medguard.config import get_settings
    from medguard.storage.vector_store import VectorStore

    settings = get_settings()
    store = VectorStore(settings)

    if kind == "dense":
        from medguard.embeddings.fastembed_dense import get_dense_embedder
        from medguard.retrieval.dense import DenseRetriever

        return DenseRetriever(store, get_dense_embedder(), settings.retrieval)

    if kind == "sparse":
        from medguard.embeddings.fastembed_sparse import get_sparse_embedder
        from medguard.retrieval.sparse import SparseRetriever

        return SparseRetriever(store, get_sparse_embedder(), settings.retrieval)

    if kind == "hybrid":
        raise NotImplementedError(
            "hybrid retrieval and RRF fusion land in Phase 4B; Phase 4A measures the "
            "dense and sparse baselines that the hybrid delta is computed against"
        )

    raise ValueError(f"unknown retriever {kind!r}; expected 'dense', 'sparse' or 'hybrid'")
