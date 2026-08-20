"""Sparse embeddings: BM25 via fastembed.

Why a sparse leg at all
-----------------------
Dense retrieval smooths. That is what makes it good at paraphrase and bad at exactly the
tokens clinical text turns on: drug names, ``12.5 mg``, ICD codes, ``HbA1c``, the
difference between ``ceftriaxone`` and ``cefotaxime``. A dense model places those near
each other because they *are* near each other in meaning, which is precisely wrong when
the question is which one to give. BM25 matches the string.

Whether the hybrid actually beats dense alone on this corpus is an empirical question
for the evaluation phase, not a claim made here. It is measured against a dense-only
baseline before it earns its place in the default path.

Why Qdrant's BM25 and not a separate search engine
--------------------------------------------------
``Qdrant/bm25`` produces sparse vectors that Qdrant stores natively as a named vector in
the same collection and the same point as the dense vector. One store, one write path,
one deletion path -- no second system to keep consistent, and no window in which a
document has been removed from one index and not the other.

What this deliberately does not do
----------------------------------
No dimension assertion: sparse vectors are variable-length by construction, one entry
per distinct term, so there is no fixed shape to check. No truncation warning either:
BM25 tokenises the whole input and weights terms, with no context window to overflow.
The dense side has both, and the asymmetry is real rather than an oversight.
"""

from __future__ import annotations

import functools
from typing import Any

from medguard.config import EmbeddingSettings, get_settings
from medguard.embeddings.base import SparseVector
from medguard.errors import EmbeddingError
from medguard.logging import get_logger

__all__ = ["FastEmbedSparse", "get_sparse_embedder"]

log = get_logger(__name__)


class FastEmbedSparse:
    """A :class:`~medguard.embeddings.base.SparseEmbedder` backed by fastembed."""

    def __init__(self, settings: EmbeddingSettings | None = None) -> None:
        self._settings = settings if settings is not None else get_settings().embedding
        self._model: Any | None = None

    @property
    def model_id(self) -> str:
        return self._settings.sparse_model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> Any:
        if self._model is None:
            from fastembed import SparseTextEmbedding

            log.info("embeddings.sparse.loading", model=self.model_id)
            try:
                self._model = SparseTextEmbedding(model_name=self.model_id)
            except Exception as exc:
                raise EmbeddingError(
                    f"could not load sparse model {self.model_id!r}: {type(exc).__name__}: {exc}"
                ) from exc
            log.info("embeddings.sparse.loaded", model=self.model_id)
        return self._model

    @staticmethod
    def _convert(raw: Any) -> SparseVector:
        """Translate fastembed's numpy-backed result into the project's own type.

        The conversion is the boundary. fastembed returns an object with numpy arrays
        attached; letting that escape would put numpy semantics into the retrieval code
        and make the sparse leg impossible to fake in a test. ``SparseVector`` validates
        that indices and values align, which catches a malformed result here rather than
        as a Qdrant rejection several stages later.
        """
        indices = [int(i) for i in getattr(raw, "indices", [])]
        values = [float(v) for v in getattr(raw, "values", [])]
        return SparseVector(indices=indices, values=values)

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        """Embed passages for indexing."""
        if not texts:
            return []

        model = self._load()
        try:
            vectors = [self._convert(raw) for raw in model.embed(texts)]
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                f"{self.model_id} failed to embed {len(texts)} passage(s): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if len(vectors) != len(texts):
            # Misalignment here indexes every subsequent chunk under the wrong text, and
            # nothing downstream can tell: retrieval keeps working and returns nonsense.
            raise EmbeddingError(
                f"{self.model_id} returned {len(vectors)} vectors for {len(texts)} texts"
            )
        return vectors

    def embed_query(self, text: str) -> SparseVector:
        """Embed a single query for search.

        BM25 scoring is asymmetric in general -- a query is weighted by IDF against the
        corpus while a document is weighted by term frequency -- and fastembed's
        ``Qdrant/bm25`` handles that split on the Qdrant side via the collection's IDF
        modifier. The call is therefore the same on both sides here, and the modifier is
        set when the collection is created. Getting that modifier wrong is what makes a
        BM25 leg quietly score everything almost equally.
        """
        return self.embed_documents([text])[0]


@functools.lru_cache(maxsize=1)
def get_sparse_embedder() -> FastEmbedSparse:
    """The process-wide sparse embedder. Cached for the same reason as the dense one."""
    return FastEmbedSparse()
