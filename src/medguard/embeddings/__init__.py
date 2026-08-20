"""Embedding contracts and their fastembed implementations.

The contracts in :mod:`~medguard.embeddings.base` were written in Phase 1 with nothing
behind them, deliberately: an interface retrofitted after a concrete class has spread
through a codebase almost never recovers the abstraction. Phase 3B added the
implementations, and the claim that the embedding model is swappable by configuration is
now checkable rather than aspirational -- nothing outside these two modules imports
fastembed.

Importing this package does **not** load a model or download weights. Both embedders
build their ONNX session on first use, so a diagnostic can report the planned
configuration and the unit tests can verify the protocols are satisfied without any
network access at all.
"""

from __future__ import annotations

from medguard.embeddings.base import DenseEmbedder, SparseEmbedder, SparseVector
from medguard.embeddings.fastembed_dense import FastEmbedDense, get_dense_embedder
from medguard.embeddings.fastembed_sparse import FastEmbedSparse, get_sparse_embedder

__all__ = [
    "DenseEmbedder",
    "FastEmbedDense",
    "FastEmbedSparse",
    "SparseEmbedder",
    "SparseVector",
    "get_dense_embedder",
    "get_sparse_embedder",
]
