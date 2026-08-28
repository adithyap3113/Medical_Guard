"""Dense and sparse retrieval against the real, fully indexed corpus.

Unlike the other integration modules, these tests do **not** create a throwaway
collection. The properties under test are properties of the actual 5,185-chunk index --
whether BM25's IDF modifier is doing anything, whether a rare drug name reaches the one
document that contains it, whether a publisher filter really narrows -- and none of them
are observable on thirty synthetic points.

That means the collection here is read-only. Nothing in this module writes, deletes or
recreates anything; a test that did would destroy an hour of ingestion.
"""

from __future__ import annotations

import pytest

from medguard.config import Settings, get_settings
from medguard.embeddings.fastembed_dense import get_dense_embedder
from medguard.embeddings.fastembed_sparse import get_sparse_embedder
from medguard.models import DocumentType, RetrieverKind, SourceOrganization
from medguard.retrieval.base import RetrievalFilter
from medguard.retrieval.dense import DenseRetriever
from medguard.retrieval.sparse import SparseRetriever
from medguard.storage.vector_store import VectorStore

pytestmark = [pytest.mark.integration, pytest.mark.slow]

MIN_EXPECTED_POINTS = 1000


@pytest.fixture
def live_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Undo conftest's in-memory redirect and use the real collection name.

    Function-scoped and monkeypatch-based, matching ``test_indexing.py``: conftest's
    autouse ``isolated_env`` is function-scoped, so a module-scoped fixture here would
    be overwritten by it before the first test body runs.
    """
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.delenv("QDRANT_COLLECTION", raising=False)
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def store(live_settings: Settings):  # type: ignore[no-untyped-def]
    """A read-only handle on the indexed corpus, or a skip.

    Nothing in this module writes. The collection is the real one, and recreating it
    would destroy an hour of ingestion.
    """
    store = VectorStore(live_settings)
    if not store.health().reachable:
        store.close()
        pytest.skip("Qdrant is not running; start it with docker compose")
    try:
        count = store.collection_points_count()
    except Exception:  # noqa: BLE001 - an absent collection is a skip, not a failure
        store.close()
        pytest.skip("the corpus collection does not exist; run `medguard ingest` first")
    if count < MIN_EXPECTED_POINTS:
        store.close()
        pytest.skip(f"only {count} points indexed; run `medguard ingest` first")
    yield store
    store.close()


@pytest.fixture
def dense(store: VectorStore, live_settings: Settings) -> DenseRetriever:
    # The embedder factories are lru_cached process-wide, so the ONNX session is built
    # once for the whole session even though these fixtures are per-test.
    return DenseRetriever(store, get_dense_embedder(), live_settings.retrieval)


@pytest.fixture
def sparse(store: VectorStore, live_settings: Settings) -> SparseRetriever:
    return SparseRetriever(store, get_sparse_embedder(), live_settings.retrieval)


class TestDenseRetrieval:
    def test_it_returns_ranked_candidates(self, dense: DenseRetriever) -> None:
        candidates = dense.retrieve("How is latent tuberculosis infection treated?", top_k=10)
        assert len(candidates) == 10
        assert [c.rank for c in candidates] == list(range(10))
        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_every_candidate_can_be_cited(self, dense: DenseRetriever) -> None:
        """A result without a document and a page is not evidence, whatever its score."""
        for candidate in dense.retrieve("hepatitis C screening", top_k=5):
            assert candidate.chunk.text
            assert candidate.document_id
            assert candidate.chunk.metadata["source_organization"]

    def test_cosine_scores_stay_in_range(self, dense: DenseRetriever) -> None:
        for candidate in dense.retrieve("blood pressure treatment threshold", top_k=10):
            assert -1.0 <= candidate.score <= 1.0

    def test_it_finds_the_right_document_for_a_clear_question(self, dense: DenseRetriever) -> None:
        candidates = dense.retrieve(
            "What blood pressure threshold triggers antihypertensive treatment?", top_k=10
        )
        assert any(c.document_id == "who_hypertension_2021" for c in candidates)


class TestSparseRetrieval:
    def test_it_returns_ranked_candidates(self, sparse: SparseRetriever) -> None:
        candidates = sparse.retrieve("isoniazid rifapentine", top_k=10)
        assert candidates
        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_bm25_scores_are_spread_rather_than_near_uniform(self, sparse: SparseRetriever) -> None:
        """The check that the IDF modifier is actually doing something.

        Without ``Modifier.IDF`` on the sparse vector params, Qdrant stores raw term
        weights and scores almost everything equally -- which looks exactly like a
        working BM25 leg until someone measures it. A collection built wrong would show
        a top-to-bottom spread near zero here.
        """
        candidates = sparse.retrieve("ceftazidime-avibactam aztreonam", top_k=20)
        assert len(candidates) >= 10
        top, bottom = candidates[0].score, candidates[-1].score
        assert top > bottom
        assert (top - bottom) / top > 0.2, f"scores nearly uniform: {top} vs {bottom}"

    def test_a_rare_drug_name_reaches_the_one_document_that_carries_it(
        self, sparse: SparseRetriever
    ) -> None:
        """The qualitative check from the Phase 4 plan. A rare token is exactly what
        dense embeddings blur and BM25 exists to catch."""
        candidates = sparse.retrieve("ceftazidime-avibactam", top_k=5)
        assert candidates
        assert candidates[0].document_id == "icmr_diagnosis_management_cros"

    def test_a_punctuation_only_query_yields_nothing_rather_than_failing(
        self, sparse: SparseRetriever
    ) -> None:
        assert sparse.retrieve("!!! ???", top_k=5) == []


class TestFiltering:
    def test_a_publisher_filter_returns_only_that_publisher(self, dense: DenseRetriever) -> None:
        candidates = dense.retrieve(
            "infection control precautions",
            top_k=20,
            retrieval_filter=RetrievalFilter(source_organization=[SourceOrganization.ICMR]),
        )
        assert candidates
        assert {c.chunk.metadata["source_organization"] for c in candidates} == {"ICMR"}

    def test_the_reference_type_selects_the_one_reference_document(
        self, dense: DenseRetriever
    ) -> None:
        """``cdc_antibiotic_resistance_2019`` is the corpus's only document_type=reference,
        so this is both a filter test and a guard on the payload's enum casing: the
        payload holds the value 'reference' while SQL holds the name 'REFERENCE'."""
        candidates = dense.retrieve(
            "antibiotic resistant infections",
            top_k=20,
            retrieval_filter=RetrievalFilter(document_type=[DocumentType.REFERENCE]),
        )
        assert candidates
        assert {c.document_id for c in candidates} == {"cdc_antibiotic_resistance_2019"}

    def test_filters_narrow_rather_than_reorder(self, dense: DenseRetriever) -> None:
        query = "screening recommendations"
        unfiltered = {c.document_id for c in dense.retrieve(query, top_k=30)}
        filtered = {
            c.document_id
            for c in dense.retrieve(
                query,
                top_k=30,
                retrieval_filter=RetrievalFilter(source_organization=[SourceOrganization.WHO]),
            )
        }
        assert filtered <= {"who_hypertension_2021"}
        assert filtered != unfiltered

    def test_the_sparse_leg_honours_filters_too(self, sparse: SparseRetriever) -> None:
        candidates = sparse.retrieve(
            "hand hygiene gloves",
            top_k=20,
            retrieval_filter=RetrievalFilter(source_organization=[SourceOrganization.ICMR]),
        )
        assert candidates
        assert {c.chunk.metadata["source_organization"] for c in candidates} == {"ICMR"}


class TestBothLegs:
    def test_they_are_labelled_by_the_leg_that_produced_them(
        self, dense: DenseRetriever, sparse: SparseRetriever
    ) -> None:
        query = "opioid dosing for chronic pain"
        assert all(c.retriever is RetrieverKind.DENSE for c in dense.retrieve(query, top_k=5))
        assert all(c.retriever is RetrieverKind.SPARSE for c in sparse.retrieve(query, top_k=5))

    def test_they_disagree_somewhere(self, dense: DenseRetriever, sparse: SparseRetriever) -> None:
        """If the two legs returned identical rankings there would be nothing for
        Phase 4B's fusion to combine, and the hybrid experiment would be pointless."""
        query = "What glycated haemoglobin level should adults with diabetes aim for?"
        dense_ids = [c.chunk_id for c in dense.retrieve(query, top_k=10)]
        sparse_ids = [c.chunk_id for c in sparse.retrieve(query, top_k=10)]
        assert dense_ids != sparse_ids

    def test_the_embed_cost_is_measured_separately_from_the_search(
        self, dense: DenseRetriever
    ) -> None:
        dense.retrieve("a clinical question about treatment", top_k=10)
        assert dense.last_embed_ms > 0.0
        assert dense.last_search_ms > 0.0
