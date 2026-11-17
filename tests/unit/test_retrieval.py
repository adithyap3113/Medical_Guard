"""The read path, exercised offline against in-process Qdrant.

Qdrant's in-process mode is a genuinely different implementation from the server -- it
ignores payload indexes, so filtering here happens by full scan. That makes filter
*behaviour* testable without Docker while leaving filter *performance* to the
integration suite, which is the honest split.

The embedders are stubbed rather than loaded. A real BGE session is 130 MB and several
seconds, and none of the properties under test here are properties of the model: they
are properties of the plumbing between the embedder, the store and the Candidate.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qdrant_client import models

from medguard.config import RetrievalSettings, Settings
from medguard.embeddings.base import DenseEmbedder, SparseEmbedder, SparseVector
from medguard.errors import (
    CollectionMismatchError,
    CollectionNotFoundError,
    RetrievalError,
    VectorStoreAuthError,
    VectorStoreError,
    VectorStoreTimeoutError,
    VectorStoreUnavailableError,
)
from medguard.models import Candidate, DocumentType, RetrieverKind, SourceOrganization
from medguard.retrieval.base import RetrievalFilter, Retriever, candidate_from_point
from medguard.retrieval.dense import DenseRetriever
from medguard.retrieval.sparse import SparseRetriever
from medguard.storage.vector_store import (
    ChunkWithVectors,
    VectorStore,
    build_filter,
    date_filter_unsupported_reason,
)

DIMENSION = 384

# (document_id, publisher, type) -- two publishers and two types, so a filter has
# something to exclude in either dimension.
_DOCUMENTS = (
    ("cdc_doc", "CDC", "guideline"),
    ("icmr_doc", "ICMR", "guideline"),
    ("ref_doc", "CDC", "reference"),
)


class StubDense:
    """Satisfies DenseEmbedder without loading a model."""

    model_id = "stub-dense"
    dimension = DIMENSION

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * DIMENSION for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1] * DIMENSION


class StubSparse:
    model_id = "stub-sparse"

    def __init__(self, *, indices: list[int] | None = None) -> None:
        self._indices = [1, 5, 9] if indices is None else indices
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [SparseVector(indices=[1, 5, 9], values=[0.9, 0.5, 0.2]) for _ in texts]

    def embed_query(self, text: str) -> SparseVector:
        self.queries.append(text)
        return SparseVector(indices=self._indices, values=[0.9] * len(self._indices))


def _chunk(document_id: str, index: int, publisher: str, doc_type: str) -> ChunkWithVectors:
    return ChunkWithVectors(
        chunk_id=f"{document_id}::{index:04d}",
        document_id=document_id,
        chunk_index=index,
        text=f"Passage {index} of {document_id}.",
        page_start=index + 1,
        page_end=index + 1,
        title=f"Title of {document_id}",
        source_organization=publisher,
        document_type=doc_type,
        publication_date=None,
        dense=[0.1 * (index + 1)] * DIMENSION,
        sparse=SparseVector(indices=[1, 5, 9], values=[0.9, 0.5, 0.2]),
    )


@pytest.fixture
def store(settings: Settings):  # type: ignore[no-untyped-def]
    """An in-process store holding thirty points across three documents."""
    store = VectorStore(settings)
    store.ensure_collection()
    store.upsert_chunks(
        [
            _chunk(document_id, index, publisher, doc_type)
            for document_id, publisher, doc_type in _DOCUMENTS
            for index in range(10)
        ]
    )
    yield store
    store.close()


def must_conditions(built: models.Filter | None) -> list[models.FieldCondition]:
    """Narrow a built filter to its field conditions.

    ``Filter.must`` is typed as a union of every condition kind Qdrant supports, so
    reaching ``.match.any`` off it directly is unresolvable for a type checker. The
    assertions here are the narrowing, and they double as the shape check: anything
    other than a list of FieldConditions means build_filter produced the wrong thing.
    """
    assert built is not None
    assert isinstance(built.must, list)
    conditions = []
    for condition in built.must:
        assert isinstance(condition, models.FieldCondition)
        conditions.append(condition)
    return conditions


def match_any_values(built: models.Filter | None, index: int) -> list[str]:
    """The OR-ed values of the nth must-condition."""
    match = must_conditions(built)[index].match
    assert isinstance(match, models.MatchAny)
    return [str(value) for value in match.any]


class TestFilterConstruction:
    def test_no_constraints_produce_no_filter(self) -> None:
        """None rather than an always-true Filter, so the unfiltered path sends nothing."""
        assert build_filter() is None
        assert RetrievalFilter().to_qdrant() is None

    def test_values_within_a_field_are_or_ed(self) -> None:
        built = build_filter(source_organization=["CDC", "ICMR"])
        assert len(must_conditions(built)) == 1
        assert match_any_values(built, 0) == ["CDC", "ICMR"]

    def test_fields_are_and_ed(self) -> None:
        built = build_filter(source_organization=["CDC"], document_type=["guideline"])
        assert len(must_conditions(built)) == 2

    def test_enum_members_lower_to_their_values_not_their_names(self) -> None:
        """The trap: SQLAlchemy persists DocumentType.REFERENCE as the name 'REFERENCE',
        while the Qdrant payload holds the value 'reference'. A caller passing the name
        matches nothing, silently."""
        built = RetrievalFilter(
            source_organization=[SourceOrganization.CDC],
            document_type=[DocumentType.REFERENCE],
        ).to_qdrant()
        assert match_any_values(built, 1) == ["reference"]

    def test_is_empty_reports_whether_anything_is_constrained(self) -> None:
        assert RetrievalFilter().is_empty()
        assert not RetrievalFilter(document_type=[DocumentType.GUIDELINE]).is_empty()

    def test_the_absent_date_filter_documents_itself(self) -> None:
        reason = date_filter_unsupported_reason()
        assert "publication_date" in reason
        assert "2,550" in reason


class TestSearchMethods:
    def test_dense_search_returns_scored_points_best_first(self, store: VectorStore) -> None:
        points = store.search_dense([0.1] * DIMENSION, limit=5)
        assert len(points) == 5
        assert [p.score for p in points] == sorted((p.score for p in points), reverse=True)

    def test_sparse_search_returns_points(self, store: VectorStore) -> None:
        query = SparseVector(indices=[1, 5], values=[0.8, 0.4])
        assert store.search_sparse(query, limit=5)

    def test_limit_is_respected(self, store: VectorStore) -> None:
        assert len(store.search_dense([0.1] * DIMENSION, limit=3)) == 3

    def test_a_wrong_dimension_names_the_real_cause(self, store: VectorStore) -> None:
        """Relaying Qdrant's shape error would send the reader to the wrong layer: the
        query was embedded by a different model than the index was built with."""
        with pytest.raises(VectorStoreError, match="different model"):
            store.search_dense([0.1] * 128, limit=5)

    def test_a_sparse_query_with_no_terms_returns_empty_rather_than_searching(
        self, store: VectorStore
    ) -> None:
        assert store.search_sparse(SparseVector(indices=[], values=[]), limit=5) == []

    def test_a_zero_limit_is_refused(self, store: VectorStore) -> None:
        with pytest.raises(VectorStoreError, match="limit must be"):
            store.search_dense([0.1] * DIMENSION, limit=0)

    def test_a_filter_that_matches_nothing_returns_an_empty_list(self, store: VectorStore) -> None:
        built = build_filter(source_organization=["WHO"])
        assert store.search_dense([0.1] * DIMENSION, limit=5, query_filter=built) == []


class TestFilteringBehaviour:
    def test_publisher_filter_excludes_other_publishers(self, store: VectorStore) -> None:
        points = store.search_dense(
            [0.1] * DIMENSION, limit=30, query_filter=build_filter(source_organization=["ICMR"])
        )
        assert points
        assert {(p.payload or {})["source_organization"] for p in points} == {"ICMR"}

    def test_document_type_filter_selects_the_reference_document(self, store: VectorStore) -> None:
        points = store.search_dense(
            [0.1] * DIMENSION, limit=30, query_filter=build_filter(document_type=["reference"])
        )
        assert points
        assert {(p.payload or {})["document_id"] for p in points} == {"ref_doc"}

    def test_combining_fields_narrows_further(self, store: VectorStore) -> None:
        both = build_filter(source_organization=["CDC"], document_type=["reference"])
        points = store.search_dense([0.1] * DIMENSION, limit=30, query_filter=both)
        assert {(p.payload or {})["document_id"] for p in points} == {"ref_doc"}


class TestCandidateConstruction:
    def test_a_point_becomes_a_candidate_with_its_citation_fields(self) -> None:
        point = models.ScoredPoint(
            id="00000000-0000-0000-0000-000000000000",
            version=1,
            score=0.87,
            payload={
                "chunk_id": "doc::0003",
                "document_id": "doc",
                "chunk_index": 3,
                "text": "Some clinical guidance.",
                "page_start": 4,
                "page_end": 5,
                "section": "Treatment",
                "title": "A Guideline",
                "source_organization": "CDC",
                "document_type": "guideline",
                "publication_date": None,
            },
        )
        candidate = candidate_from_point(point, rank=0, retriever=RetrieverKind.DENSE)
        assert isinstance(candidate, Candidate)
        assert candidate.chunk_id == "doc::0003"
        assert candidate.chunk.page_start == 4
        assert candidate.chunk.metadata["source_organization"] == "CDC"
        assert candidate.rank == 0

    def test_the_score_is_recorded_under_the_producing_retriever(self) -> None:
        """Keyed rather than overwritten, because comparing what each leg contributed is
        the entire point of the phase -- and fusion in 4B merges two keys here."""
        point = models.ScoredPoint(
            id="00000000-0000-0000-0000-000000000000",
            version=1,
            score=11.4,
            payload={
                "chunk_id": "doc::0000",
                "document_id": "doc",
                "chunk_index": 0,
                "text": "text",
            },
        )
        candidate = candidate_from_point(point, rank=2, retriever=RetrieverKind.SPARSE)
        assert candidate.scores == {"sparse": 11.4}

    def test_a_payload_missing_citation_fields_is_refused(self) -> None:
        """Returning it would put an uncitable chunk into a result set that claims every
        item is cited."""
        point = models.ScoredPoint(
            id="00000000-0000-0000-0000-000000000000",
            version=1,
            score=0.5,
            payload={"chunk_id": "doc::0000"},
        )
        with pytest.raises(RetrievalError, match="missing required payload"):
            candidate_from_point(point, rank=0, retriever=RetrieverKind.DENSE)


class TestRetrievers:
    def test_both_satisfy_the_protocol(self, store: VectorStore) -> None:
        dense = DenseRetriever(store, StubDense(), RetrievalSettings())
        sparse = SparseRetriever(store, StubSparse(), RetrievalSettings())
        assert isinstance(dense, Retriever)
        assert isinstance(sparse, Retriever)

    def test_the_stubs_satisfy_the_embedder_protocols(self) -> None:
        assert isinstance(StubDense(), DenseEmbedder)
        assert isinstance(StubSparse(), SparseEmbedder)

    def test_dense_retrieval_returns_ranked_candidates(self, store: VectorStore) -> None:
        retriever = DenseRetriever(store, StubDense(), RetrievalSettings())
        candidates = retriever.retrieve("a question", top_k=5)
        assert len(candidates) == 5
        assert [c.rank for c in candidates] == [0, 1, 2, 3, 4]
        assert all(c.retriever is RetrieverKind.DENSE for c in candidates)

    def test_sparse_retrieval_labels_its_candidates(self, store: VectorStore) -> None:
        retriever = SparseRetriever(store, StubSparse(), RetrievalSettings())
        candidates = retriever.retrieve("a question", top_k=3)
        assert all(c.retriever is RetrieverKind.SPARSE for c in candidates)

    def test_the_query_reaches_the_embedder_unmodified(self, store: VectorStore) -> None:
        embedder = StubDense()
        DenseRetriever(store, embedder, RetrievalSettings()).retrieve("exact text", top_k=1)
        assert embedder.queries == ["exact text"]

    def test_it_fetches_deeper_than_it_returns(self, store: VectorStore) -> None:
        """fetch_k exists so the harness can measure recall at depth while a caller
        still gets top_k. Asking for 2 must not limit the underlying search to 2."""
        retriever = DenseRetriever(store, StubDense(), RetrievalSettings(top_k=2, fetch_k=20))
        assert len(retriever.retrieve("q", top_k=2)) == 2

    def test_a_filter_is_applied(self, store: VectorStore) -> None:
        retriever = DenseRetriever(store, StubDense(), RetrievalSettings())
        candidates = retriever.retrieve(
            "q",
            top_k=30,
            retrieval_filter=RetrievalFilter(source_organization=[SourceOrganization.ICMR]),
        )
        assert candidates
        assert {c.document_id for c in candidates} == {"icmr_doc"}

    def test_an_empty_query_is_refused(self, store: VectorStore) -> None:
        retriever = DenseRetriever(store, StubDense(), RetrievalSettings())
        with pytest.raises(RetrievalError, match="query is empty"):
            retriever.retrieve("   ", top_k=5)

    def test_a_non_positive_top_k_is_refused(self, store: VectorStore) -> None:
        retriever = DenseRetriever(store, StubDense(), RetrievalSettings())
        with pytest.raises(RetrievalError, match="top_k must be"):
            retriever.retrieve("q", top_k=0)

    def test_a_termless_sparse_query_yields_no_candidates(self, store: VectorStore) -> None:
        retriever = SparseRetriever(store, StubSparse(indices=[]), RetrievalSettings())
        assert retriever.retrieve("!!!", top_k=5) == []

    def test_timings_are_split_between_embed_and_search(self, store: VectorStore) -> None:
        """A single end-to-end number would attribute a slow CPU embed to Qdrant."""
        retriever = DenseRetriever(store, StubDense(), RetrievalSettings())
        retriever.retrieve("q", top_k=5)
        assert retriever.last_embed_ms >= 0.0
        assert retriever.last_search_ms >= 0.0


class TestRetrieverFactory:
    """``get_retriever`` is what the CLI calls, so its error paths are user-facing.

    Safe to exercise offline: the embedder factories build a wrapper whose ONNX session
    is loaded lazily on first embed, so constructing a retriever downloads nothing.
    """

    def test_it_builds_a_dense_retriever(self, settings: Settings) -> None:
        from medguard.retrieval import get_retriever

        retriever = get_retriever("dense")
        assert isinstance(retriever, Retriever)
        assert retriever.kind is RetrieverKind.DENSE

    def test_it_builds_a_sparse_retriever(self, settings: Settings) -> None:
        from medguard.retrieval import get_retriever

        assert get_retriever("sparse").kind is RetrieverKind.SPARSE

    def test_hybrid_returns_hybrid_retriever(self, settings: Settings) -> None:
        from medguard.retrieval import HybridRetriever, get_retriever

        retriever = get_retriever("hybrid")
        assert isinstance(retriever, HybridRetriever)
        assert retriever.kind is RetrieverKind.HYBRID

    def test_an_unknown_name_lists_the_valid_ones(self, settings: Settings) -> None:
        from medguard.retrieval import get_retriever

        with pytest.raises(ValueError, match="dense"):
            get_retriever("bm42")


class TestVectorStoreLifecycleAndErrors:
    def test_qdrant_unavailable_raises_unavailable_error(self, settings: Settings) -> None:
        store = VectorStore(settings)
        store._client = MagicMock()
        store._client.query_points.side_effect = Exception("Connection refused: [WinError 10061]")

        with pytest.raises(VectorStoreUnavailableError):
            store.search_dense([0.1] * DIMENSION, limit=5)

    def test_missing_collection_raises_collection_not_found_error(self, settings: Settings) -> None:
        store = VectorStore(settings)
        store._client = MagicMock()
        store._client.query_points.side_effect = Exception("Collection 'medguard' doesn't exist")

        with pytest.raises(CollectionNotFoundError):
            store.search_dense([0.1] * DIMENSION, limit=5)

    def test_collection_schema_mismatch_raises_mismatch_error(self, settings: Settings) -> None:
        store = VectorStore(settings)
        with pytest.raises(CollectionMismatchError, match="dimensions"):
            store.search_dense([0.1] * 128, limit=5)

    def test_authentication_failure_raises_auth_error(self, settings: Settings) -> None:
        store = VectorStore(settings)
        store._client = MagicMock()
        store._client.query_points.side_effect = Exception("Unauthorized: Invalid API key")

        with pytest.raises(VectorStoreAuthError):
            store.search_dense([0.1] * DIMENSION, limit=5)

    def test_timeout_raises_timeout_error(self, settings: Settings) -> None:
        store = VectorStore(settings)
        store._client = MagicMock()
        store._client.query_points.side_effect = Exception("Operation timed out after 30s")

        with pytest.raises(VectorStoreTimeoutError):
            store.search_dense([0.1] * DIMENSION, limit=5)

    def test_validate_collection_missing_raises(self, settings: Settings) -> None:
        store = VectorStore(settings)
        store._client = MagicMock()
        store._client.collection_exists.return_value = False

        with pytest.raises(CollectionNotFoundError):
            store.validate_collection()

    def test_store_close_cleans_up_client(self, settings: Settings) -> None:
        store = VectorStore(settings)
        mock_client = MagicMock()
        store._client = mock_client

        store.close()
        mock_client.close.assert_called_once()
        assert getattr(store, "_client") is None  # noqa: B009
