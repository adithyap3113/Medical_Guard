"""Dense embeddings: BGE-small via fastembed's ONNX runtime.

Why fastembed and not sentence-transformers
-------------------------------------------
sentence-transformers pulls PyTorch, which is roughly 2.5 GB of wheels and wants more
resident memory than this machine has spare. fastembed runs the same BGE weights through
ONNX Runtime: about 130 MB on disk, no CUDA stack, and inference that is competitive on
CPU. The model is the same; only the runtime differs, so the retrieval quality argument
is unaffected by the packaging decision.

Why the model is loaded lazily
------------------------------
Constructing :class:`FastEmbedDense` must not download 130 MB or allocate an ONNX
session. Two consequences follow directly. ``medguard doctor`` can report the *planned*
embedding configuration without paying for it, and the unit tests can assert that this
class satisfies the :class:`~medguard.embeddings.base.DenseEmbedder` protocol while
staying entirely offline. The first call to :meth:`embed_documents` is where the cost
lands, and that is a place where the caller has already decided to spend it.

The truncation warning
----------------------
This is the part worth reading. fastembed truncates any input past the model's 512-token
window **silently** -- no exception, no flag on the result, just a vector computed from
the first half of the passage. For a chunk containing a recommendation and its
contraindication, that means the contraindication is not in the index at all and no
query can retrieve it. Nothing downstream can detect this: the vector is well-formed and
retrieval keeps working. So oversize inputs are counted and logged by name before they
are embedded, which turns a silent quality loss into a line in the ingestion log.
"""

from __future__ import annotations

import functools
from typing import Any

from medguard.config import EmbeddingSettings, get_settings
from medguard.errors import EmbeddingError
from medguard.logging import get_logger

__all__ = ["FastEmbedDense", "get_dense_embedder"]

log = get_logger(__name__)

#: BGE-small's context window. Passages longer than this are truncated by the runtime
#: without complaint; see the module docstring.
MAX_TOKENS = 512

#: Characters per token, used only to decide whether a text is *worth warning about*
#: before the tokeniser has seen it. English clinical prose runs about four characters
#: per token; the estimate is deliberately pessimistic (lower ratio -> warns earlier)
#: because a false warning costs a log line and a missed one costs half a passage.
_CHARS_PER_TOKEN = 3.5


class FastEmbedDense:
    """A :class:`~medguard.embeddings.base.DenseEmbedder` backed by fastembed.

    Implements the protocol structurally rather than by inheritance, which is the point
    of the protocol: nothing outside this module imports fastembed, so swapping to a
    different runtime is a new class and a config change, not a refactor.
    """

    def __init__(self, settings: EmbeddingSettings | None = None) -> None:
        self._settings = settings if settings is not None else get_settings().embedding
        self._model: Any | None = None

    @property
    def model_id(self) -> str:
        return self._settings.model

    @property
    def dimension(self) -> int:
        return self._settings.dimension

    @property
    def loaded(self) -> bool:
        """True once the ONNX session exists. Used by diagnostics, never by callers."""
        return self._model is not None

    def _load(self) -> Any:
        """Build the ONNX session on first use, downloading weights if necessary."""
        if self._model is None:
            from fastembed import TextEmbedding

            log.info("embeddings.dense.loading", model=self.model_id)
            try:
                self._model = TextEmbedding(model_name=self.model_id)
            except Exception as exc:
                raise EmbeddingError(
                    f"could not load dense model {self.model_id!r}: {type(exc).__name__}: {exc}"
                ) from exc
            log.info("embeddings.dense.loaded", model=self.model_id, dimension=self.dimension)
        return self._model

    def _warn_on_truncation(self, texts: list[str]) -> None:
        """Log any passage long enough that the model will silently cut it."""
        limit = int(MAX_TOKENS * _CHARS_PER_TOKEN)
        for position, text in enumerate(texts):
            if len(text) > limit:
                log.warning(
                    "embeddings.dense.likely_truncated",
                    position=position,
                    chars=len(text),
                    limit_chars=limit,
                    max_tokens=MAX_TOKENS,
                    preview=text[:80],
                    detail=(
                        "text exceeds the model context window and will be truncated "
                        "without error; the tail will not be represented in the index"
                    ),
                )

    def _validate(self, vectors: list[list[float]], expected: int) -> list[list[float]]:
        """Reject anything whose shape would corrupt the collection.

        Both checks guard failures that are undetectable downstream. A short return list
        silently misaligns vectors with chunks, so every chunk from that batch onward is
        indexed under the wrong text. A wrong dimension is rejected by Qdrant for the
        first batch and would otherwise be diagnosed as a Qdrant problem.
        """
        if len(vectors) != expected:
            raise EmbeddingError(
                f"{self.model_id} returned {len(vectors)} vectors for {expected} texts"
            )
        for position, vector in enumerate(vectors):
            if len(vector) != self.dimension:
                raise EmbeddingError(
                    f"{self.model_id} returned a {len(vector)}-dimensional vector at "
                    f"position {position}; the collection expects {self.dimension}"
                )
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for indexing."""
        if not texts:
            return []

        self._warn_on_truncation(texts)
        model = self._load()
        try:
            vectors = [[float(v) for v in vector] for vector in model.embed(texts)]
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                f"{self.model_id} failed to embed {len(texts)} passage(s): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        return self._validate(vectors, len(texts))

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query for search.

        Separate from :meth:`embed_documents` because several retrieval models -- E5 and
        the instructor family among them -- require an instruction prefix on queries and
        not on passages, and collapsing the two methods makes that impossible to add
        later without touching every call site.

        For BGE-small-en-v1.5 specifically the two are identical. BAAI documents a query
        instruction ("Represent this sentence for searching relevant passages: ") for
        *retrieval* use, and also states it is unnecessary for short queries; applying it
        here without measuring would be a quality change made on faith. The measurement
        belongs to the evaluation phase, and this is the seam where the result would be
        applied.
        """
        return self.embed_documents([text])[0]


@functools.lru_cache(maxsize=1)
def get_dense_embedder() -> FastEmbedDense:
    """The process-wide dense embedder.

    Cached because the ONNX session is the expensive object, not the wrapper: building a
    second one doubles resident memory on a machine chosen for having very little to
    spare. The cache holds the wrapper, and the wrapper still loads lazily, so calling
    this in a diagnostic path costs nothing.

    Tests that need a fresh instance call ``get_dense_embedder.cache_clear()``.
    """
    return FastEmbedDense()
