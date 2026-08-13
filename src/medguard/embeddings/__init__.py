"""Embedding contracts.

Phase 1 defines the interfaces only -- no implementation, no dependency on fastembed,
and **no model weights are downloaded**. Implementations land in Phase 4.
"""

from __future__ import annotations

from medguard.embeddings.base import DenseEmbedder, SparseEmbedder, SparseVector

__all__ = ["DenseEmbedder", "SparseEmbedder", "SparseVector"]
