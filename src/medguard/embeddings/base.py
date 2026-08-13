"""Embedding interfaces.

Why define these in Phase 1 with no implementation behind them
--------------------------------------------------------------
The claim "the embedding model can be swapped by changing configuration" is only true
if nothing in the codebase depends on a specific model's library, output shape, or
dimensionality. Writing the contract *before* the first implementation is what keeps
that true; retrofitting an interface after a concrete class has spread through the
codebase almost never recovers it.

Concretely, this is what makes the hardware decision reversible. ``bge-small`` via
ONNX was chosen because this development machine is CPU-only with roughly 2.5 GB of
headroom. On a machine with a real GPU, moving to ``bge-base`` or BGE-M3 should be an
environment-variable change plus a re-index -- not a refactor.

Two separate protocols, because dense and sparse retrieval return fundamentally
different things: a fixed-width float vector versus a variable set of (token id,
weight) pairs. Forcing them into one interface would mean one of them lies about its
output. Qdrant stores both natively in a single collection, which is why hybrid search
does not need a second database.

Nothing here imports a model runtime, so this module is free to import anywhere.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["DenseEmbedder", "SparseEmbedder", "SparseVector"]


class SparseVector(BaseModel):
    """A sparse vector as parallel index/value arrays -- Qdrant's native format.

    BM25 and SPLADE both produce a handful of weighted terms out of a large
    vocabulary, so storing a dense array would be enormous and mostly zeros.
    """

    model_config = ConfigDict(frozen=True)

    indices: list[int] = Field(description="Token/vocabulary ids with non-zero weight.")
    values: list[float] = Field(description="Weight per corresponding index.")

    @model_validator(mode="after")
    def _same_length(self) -> SparseVector:
        if len(self.indices) != len(self.values):
            msg = f"indices ({len(self.indices)}) and values ({len(self.values)}) must align"
            raise ValueError(msg)
        return self


@runtime_checkable
class DenseEmbedder(Protocol):
    """Produces fixed-width semantic vectors.

    ``embed_documents`` and ``embed_query`` are distinct because several retrieval
    models -- including the BGE family -- expect an instruction prefix on queries but
    not on passages. Collapsing them into one method is a subtle, silent quality
    regression that is very hard to spot without an evaluation harness.
    """

    model_id: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for indexing."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query for search."""
        ...


@runtime_checkable
class SparseEmbedder(Protocol):
    """Produces sparse lexical vectors (BM25, SPLADE) for exact-term matching.

    Retained alongside dense retrieval because clinical text is full of tokens that
    dense models blur together: drug names, dosages such as ``12.5 mg``, ICD codes and
    abbreviations. Whether that helps here is an empirical question for Phase 7, not
    an assumption -- which is exactly why the baseline is measured first.
    """

    model_id: str

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        """Embed passages for indexing."""
        ...

    def embed_query(self, text: str) -> SparseVector:
        """Embed a single query for search."""
        ...
