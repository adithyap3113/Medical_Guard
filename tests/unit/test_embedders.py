"""Embedders: the contracts, the caching, and the shape checks.

Almost everything here runs without loading a model. That is the design being verified,
not a limitation of the tests: constructing an embedder must not download 130 MB or
allocate an ONNX session, because `medguard doctor` reports the embedding configuration
and the whole unit suite has to stay offline.

The shape checks earn their place by what they catch. A model returning fewer vectors
than it was given texts misaligns every chunk from that batch onward -- each one indexed
under someone else's text -- and no downstream component can detect it, because
retrieval keeps working and simply returns the wrong passage. Same for a dimension
mismatch, which otherwise surfaces as a confusing Qdrant rejection.

Tests that load the real BGE weights are marked `slow` and deselected by default.
"""

from __future__ import annotations

import pytest
from structlog.testing import capture_logs

from medguard.config import EmbeddingSettings
from medguard.embeddings import (
    DenseEmbedder,
    FastEmbedDense,
    FastEmbedSparse,
    SparseEmbedder,
    SparseVector,
    get_dense_embedder,
    get_sparse_embedder,
)
from medguard.embeddings.fastembed_dense import MAX_TOKENS
from medguard.errors import EmbeddingError, IngestionError

DIMENSION = 384


class StubModel:
    """Stands in for a loaded fastembed model."""

    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self.vectors = vectors
        self.seen: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.seen.append(list(texts))
        if self.vectors is not None:
            return self.vectors
        return [[0.1] * DIMENSION for _ in texts]


class StubSparseResult:
    def __init__(self, indices: list[int], values: list[float]) -> None:
        self.indices = indices
        self.values = values


class StubSparseModel:
    def __init__(self, results: list[StubSparseResult] | None = None) -> None:
        self.results = results

    def embed(self, texts: list[str]) -> list[StubSparseResult]:
        if self.results is not None:
            return self.results
        return [StubSparseResult([1, 7], [0.9, 0.4]) for _ in texts]


def _loaded(embedder: FastEmbedDense | FastEmbedSparse, model: object) -> None:
    """Attach a stub model so no real weights are ever fetched."""
    embedder._model = model


class TestProtocolConformance:
    def test_the_dense_embedder_satisfies_the_protocol(self) -> None:
        """The claim "the model is swappable by configuration" rests on this."""
        assert isinstance(FastEmbedDense(), DenseEmbedder)

    def test_the_sparse_embedder_satisfies_the_protocol(self) -> None:
        assert isinstance(FastEmbedSparse(), SparseEmbedder)

    def test_the_dense_embedder_reports_its_configuration(self) -> None:
        embedder = FastEmbedDense(EmbeddingSettings(model="BAAI/bge-small-en-v1.5"))
        assert embedder.model_id == "BAAI/bge-small-en-v1.5"
        assert embedder.dimension == DIMENSION

    def test_the_sparse_embedder_reports_its_model(self) -> None:
        assert FastEmbedSparse(EmbeddingSettings()).model_id == "Qdrant/bm25"


class TestLaziness:
    def test_constructing_loads_nothing(self) -> None:
        """`medguard doctor` constructs these to report configuration, and must be free."""
        assert FastEmbedDense().loaded is False
        assert FastEmbedSparse().loaded is False

    def test_embedding_an_empty_list_loads_nothing(self) -> None:
        """A document that chunked to nothing must not trigger a 130 MB download."""
        embedder = FastEmbedDense()
        assert embedder.embed_documents([]) == []
        assert embedder.loaded is False

    def test_the_sparse_embedder_is_lazy_too(self) -> None:
        embedder = FastEmbedSparse()
        assert embedder.embed_documents([]) == []
        assert embedder.loaded is False


class TestCaching:
    def test_the_dense_factory_returns_one_instance(self) -> None:
        """A second ONNX session doubles resident memory on a machine short of it."""
        get_dense_embedder.cache_clear()
        assert get_dense_embedder() is get_dense_embedder()

    def test_the_sparse_factory_returns_one_instance(self) -> None:
        get_sparse_embedder.cache_clear()
        assert get_sparse_embedder() is get_sparse_embedder()

    def test_the_cache_can_be_cleared(self) -> None:
        get_dense_embedder.cache_clear()
        first = get_dense_embedder()
        get_dense_embedder.cache_clear()
        assert get_dense_embedder() is not first


class TestDenseShapeChecks:
    def test_a_wrong_dimension_is_rejected(self) -> None:
        """Otherwise this surfaces later as a Qdrant error about the collection."""
        embedder = FastEmbedDense()
        _loaded(embedder, StubModel(vectors=[[0.1] * 512]))

        with pytest.raises(EmbeddingError, match="512-dimensional"):
            embedder.embed_documents(["one passage"])

    def test_a_short_return_list_is_rejected(self) -> None:
        """The silent one: every chunk after the gap is indexed under the wrong text."""
        embedder = FastEmbedDense()
        _loaded(embedder, StubModel(vectors=[[0.1] * DIMENSION]))

        with pytest.raises(EmbeddingError, match="1 vectors for 3 texts"):
            embedder.embed_documents(["a", "b", "c"])

    def test_a_correct_batch_passes(self) -> None:
        embedder = FastEmbedDense()
        _loaded(embedder, StubModel())

        vectors = embedder.embed_documents(["a", "b"])
        assert len(vectors) == 2
        assert all(len(v) == DIMENSION for v in vectors)

    def test_a_model_failure_becomes_an_embedding_error(self) -> None:
        class Broken:
            def embed(self, texts: list[str]) -> list[list[float]]:
                raise RuntimeError("onnxruntime session failed")

        embedder = FastEmbedDense()
        _loaded(embedder, Broken())

        with pytest.raises(EmbeddingError, match="onnxruntime"):
            embedder.embed_documents(["a"])

    def test_embedding_errors_are_ingestion_errors(self) -> None:
        """So the pipeline's per-document boundary catches them."""
        assert issubclass(EmbeddingError, IngestionError)


class TestTruncationWarning:
    """Asserted through `structlog.testing.capture_logs`, and neither of the obvious ways.

    `caplog` sees nothing: structlog here writes to stdout rather than through the stdlib
    logging handlers. `capsys` works in isolation and then fails in the full suite,
    because structlog binds its output stream when it is configured -- so once any test
    has run a CLI command under `CliRunner`, the logger is still holding that command's
    long-since-restored stdout.

    Both of those failure modes produce a test that passes while asserting nothing, which
    for a warning whose entire job is to make a silent problem visible would be a
    particularly bad joke. `capture_logs` intercepts the event before any of that.
    """

    def test_an_oversized_passage_is_reported(self) -> None:
        """fastembed truncates past 512 tokens with no error and no flag on the result.

        For a chunk holding a recommendation and its contraindication, that means the
        contraindication is not in the index at all and no query can retrieve it. The
        only way anyone finds out is if the ingestion log says so.
        """
        import medguard.embeddings.fastembed_dense as fed
        from medguard.logging import get_logger

        embedder = FastEmbedDense()
        _loaded(embedder, StubModel())

        with capture_logs() as events:
            fed.log = get_logger("medguard.embeddings.fastembed_dense")
            embedder.embed_documents(["x" * (MAX_TOKENS * 10)])

        warnings = [e for e in events if e.get("event") == "embeddings.dense.likely_truncated"]
        assert warnings, f"expected a truncation warning, saw {[e.get('event') for e in events]}"
        assert warnings[0]["log_level"] == "warning"
        assert warnings[0]["max_tokens"] == MAX_TOKENS

    def test_a_normal_passage_is_not_reported(self) -> None:
        """A warning on every chunk is a warning nobody reads."""
        import medguard.embeddings.fastembed_dense as fed
        from medguard.logging import get_logger

        embedder = FastEmbedDense()
        _loaded(embedder, StubModel())

        with capture_logs() as events:
            fed.log = get_logger("medguard.embeddings.fastembed_dense")
            embedder.embed_documents(["a normal-length clinical passage."])

        assert not [e for e in events if e.get("event") == "embeddings.dense.likely_truncated"]

    def test_the_warning_does_not_stop_the_embedding(self) -> None:
        """It is a quality signal, not a failure: the truncated vector is still useful."""
        import structlog

        structlog.reset_defaults()
        embedder = FastEmbedDense()
        _loaded(embedder, StubModel())

        assert len(embedder.embed_documents(["x" * (MAX_TOKENS * 10)])) == 1


class TestSparseConversion:
    def test_results_become_sparse_vectors(self) -> None:
        embedder = FastEmbedSparse()
        _loaded(embedder, StubSparseModel())

        vectors = embedder.embed_documents(["a", "b"])
        assert all(isinstance(v, SparseVector) for v in vectors)
        assert vectors[0].indices == [1, 7]
        assert vectors[0].values == [0.9, 0.4]

    def test_misaligned_indices_and_values_are_rejected(self) -> None:
        """`SparseVector` validates the pair; a malformed result fails here, not in Qdrant."""
        embedder = FastEmbedSparse()
        _loaded(embedder, StubSparseModel([StubSparseResult([1, 2, 3], [0.5])]))

        with pytest.raises(EmbeddingError):
            embedder.embed_documents(["a"])

    def test_a_short_return_list_is_rejected(self) -> None:
        embedder = FastEmbedSparse()
        _loaded(embedder, StubSparseModel([StubSparseResult([1], [0.5])]))

        with pytest.raises(EmbeddingError, match="1 vectors for 2 texts"):
            embedder.embed_documents(["a", "b"])

    def test_there_is_no_dimension_assertion(self) -> None:
        """Sparse vectors are variable-length by construction; asserting a size is wrong."""
        embedder = FastEmbedSparse()
        _loaded(
            embedder,
            StubSparseModel([StubSparseResult([1], [0.5]), StubSparseResult([1, 2, 3], [1, 2, 3])]),
        )

        vectors = embedder.embed_documents(["a", "b"])
        assert [len(v.indices) for v in vectors] == [1, 3]


class TestQueryPath:
    def test_a_dense_query_returns_one_vector(self) -> None:
        embedder = FastEmbedDense()
        _loaded(embedder, StubModel())
        assert len(embedder.embed_query("what is the first-line therapy?")) == DIMENSION

    def test_a_sparse_query_returns_one_vector(self) -> None:
        embedder = FastEmbedSparse()
        _loaded(embedder, StubSparseModel())
        assert isinstance(embedder.embed_query("ceftriaxone dose"), SparseVector)

    def test_queries_and_passages_take_the_same_path_for_bge_small(self) -> None:
        """Documented rather than assumed: BGE-small needs no query prefix.

        The two methods stay separate because E5 and the instructor family do need one,
        and collapsing them would make that impossible to add without touching every
        call site.
        """
        embedder = FastEmbedDense()
        model = StubModel()
        _loaded(embedder, model)

        embedder.embed_query("ceftriaxone dose")
        assert model.seen == [["ceftriaxone dose"]]


@pytest.mark.slow
class TestAgainstTheRealModel:
    """Downloads ~130 MB on first run. Deselected by default; run with `-m slow`."""

    def test_the_real_dense_model_produces_384_dimensions(self) -> None:
        vectors = FastEmbedDense().embed_documents(["Ceftriaxone 2 g IV once daily."])
        assert len(vectors) == 1
        assert len(vectors[0]) == DIMENSION

    def test_similar_passages_score_closer_than_unrelated_ones(self) -> None:
        """A dimension check passes on a model returning noise. This does not."""
        embedder = FastEmbedDense()
        a, b, c = embedder.embed_documents(
            [
                "Ceftriaxone is recommended as first-line therapy.",
                "First-line treatment is ceftriaxone.",
                "The building was completed in 1974 and has six floors.",
            ]
        )
        assert _cosine(a, b) > _cosine(a, c)

    def test_the_real_sparse_model_returns_aligned_pairs(self) -> None:
        vectors = FastEmbedSparse().embed_documents(["ceftriaxone 2 g IV once daily"])
        assert len(vectors[0].indices) == len(vectors[0].values)
        assert vectors[0].indices


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return dot / norm if norm else 0.0
