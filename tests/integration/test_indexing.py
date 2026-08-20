"""The Qdrant write path, against a real server.

Marked ``integration``: it needs Qdrant running. The in-process mode the unit suite uses
is a genuinely different implementation -- it ignores payload indexes, and its collection
metadata is not the server's -- so collection shape, index creation and mismatch
detection can only be verified here.

The property under test throughout is that a write is *replaceable*. Ingestion runs
repeatedly over the same corpus, and every operation below has to converge on the same
state no matter how many times it runs or where a previous run stopped.
"""

from __future__ import annotations

import uuid

import pytest

from medguard.config import Settings, get_settings
from medguard.embeddings.base import SparseVector
from medguard.errors import CollectionMismatchError
from medguard.storage.vector_store import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    ChunkWithVectors,
    VectorStore,
    point_id_for,
)

pytestmark = pytest.mark.integration

DIMENSION = 384


@pytest.fixture
def live_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Undo conftest's in-memory redirect, and use a collection unique to this run.

    A per-run collection name means these tests never touch a corpus someone spent an
    hour indexing, and two runs in parallel cannot interfere with each other.
    """
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", f"medguard_test_{uuid.uuid4().hex[:8]}")
    get_settings.cache_clear()
    settings = get_settings()

    store = VectorStore(settings)
    try:
        store.health()
    finally:
        store.close()
    return settings


@pytest.fixture
def store(live_settings: Settings):  # type: ignore[no-untyped-def]
    """A store with its collection created, torn down afterwards."""
    store = VectorStore(live_settings)
    if not store.health().reachable:
        store.close()
        pytest.skip("Qdrant is not running; start it with docker compose")

    store.ensure_collection()
    try:
        yield store
    finally:
        try:
            store.client.delete_collection(store.collection_name)
        finally:
            store.close()


def _chunk(document_id: str, index: int, *, text: str | None = None) -> ChunkWithVectors:
    return ChunkWithVectors(
        chunk_id=f"{document_id}::{index:04d}",
        document_id=document_id,
        chunk_index=index,
        text=text or f"Passage {index} of {document_id}.",
        page_start=index + 1,
        page_end=index + 1,
        title=f"Title of {document_id}",
        source_organization="CDC",
        document_type="guideline",
        publication_date="2024-01-15T00:00:00Z",
        dense=[0.01 * (index + 1)] * DIMENSION,
        sparse=SparseVector(indices=[1, 5, 9], values=[0.9, 0.5, 0.2]),
    )


class TestCollectionCreation:
    def test_the_named_vectors_are_created(self, store: VectorStore) -> None:
        info = store.client.get_collection(store.collection_name)
        assert DENSE_VECTOR in (info.config.params.vectors or {})
        assert SPARSE_VECTOR in (info.config.params.sparse_vectors or {})

    def test_the_dense_vector_has_the_configured_shape(self, store: VectorStore) -> None:
        from qdrant_client import models

        info = store.client.get_collection(store.collection_name)
        vectors = info.config.params.vectors
        assert isinstance(vectors, dict)
        dense = vectors[DENSE_VECTOR]
        assert dense.size == DIMENSION
        assert dense.distance == models.Distance.COSINE

    def test_the_sparse_vector_has_the_idf_modifier(self, store: VectorStore) -> None:
        """Without IDF, BM25 scores everything almost equally and looks like it works."""
        from qdrant_client import models

        info = store.client.get_collection(store.collection_name)
        sparse = (info.config.params.sparse_vectors or {})[SPARSE_VECTOR]
        assert sparse.modifier == models.Modifier.IDF

    def test_the_four_payload_indexes_exist(self, store: VectorStore) -> None:
        info = store.client.get_collection(store.collection_name)
        schema = info.payload_schema or {}
        assert {
            "document_id",
            "source_organization",
            "document_type",
            "publication_date",
        } <= set(schema)

    def test_ensure_collection_is_idempotent(self, store: VectorStore) -> None:
        """Called at the start of every ingestion run, so this is the common path."""
        store.upsert_chunks([_chunk("doc_a", 0)])
        store.ensure_collection()
        store.ensure_collection()

        assert store.count_for_document("doc_a") == 1


class TestMismatchDetection:
    def test_a_wrong_dimension_is_refused(self, store: VectorStore) -> None:
        """Two models in one space is retrieval that is wrong and looks fine."""
        from medguard.config import EmbeddingSettings

        other = store._s.model_copy(update={"embedding": EmbeddingSettings(dimension=768)})
        # Same collection name, different expectation.
        other.qdrant.collection = store._s.qdrant.collection
        divergent = VectorStore(other)
        divergent._client = store.client

        with pytest.raises(CollectionMismatchError, match="different vector spaces"):
            divergent._verify_collection(store.collection_name)

    def test_a_collection_without_the_named_vector_is_refused(
        self, store: VectorStore, live_settings: Settings
    ) -> None:
        from qdrant_client import models

        name = f"{store.collection_name}_legacy"
        store.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=DIMENSION, distance=models.Distance.COSINE),
        )
        try:
            with pytest.raises(CollectionMismatchError):
                store._verify_collection(name)
        finally:
            store.client.delete_collection(name)


class TestUpsertAndCount:
    def test_chunks_become_points(self, store: VectorStore) -> None:
        store.upsert_chunks([_chunk("doc_a", i) for i in range(5)])
        assert store.count_for_document("doc_a") == 5
        assert store.collection_points_count() == 5

    def test_the_same_chunk_id_upserts_rather_than_duplicating(self, store: VectorStore) -> None:
        """The deterministic point id is what makes re-ingestion idempotent."""
        store.upsert_chunks([_chunk("doc_a", 0, text="first version")])
        store.upsert_chunks([_chunk("doc_a", 0, text="second version")])

        assert store.count_for_document("doc_a") == 1
        point = store.client.retrieve(
            collection_name=store.collection_name,
            ids=[point_id_for("doc_a::0000")],
            with_payload=True,
        )[0]
        assert point.payload is not None
        assert point.payload["text"] == "second version"

    def test_the_payload_carries_what_a_citation_needs(self, store: VectorStore) -> None:
        """Retrieval must be able to render a citation without a second lookup."""
        store.upsert_chunks([_chunk("doc_a", 3)])
        point = store.client.retrieve(
            collection_name=store.collection_name,
            ids=[point_id_for("doc_a::0003")],
            with_payload=True,
        )[0]

        assert point.payload is not None
        for key in ("chunk_id", "document_id", "page_start", "page_end", "title", "text"):
            assert key in point.payload

    def test_an_empty_upsert_is_a_no_op(self, store: VectorStore) -> None:
        store.upsert_chunks([])
        assert store.collection_points_count() == 0

    def test_counting_an_absent_document_returns_zero(self, store: VectorStore) -> None:
        assert store.count_for_document("never_ingested") == 0


class TestDeleteByDocument:
    def test_it_removes_exactly_that_document(self, store: VectorStore) -> None:
        store.upsert_chunks([_chunk("doc_a", i) for i in range(4)])
        store.upsert_chunks([_chunk("doc_b", i) for i in range(3)])

        removed = store.delete_by_document("doc_a")

        assert removed == 4
        assert store.count_for_document("doc_a") == 0
        assert store.count_for_document("doc_b") == 3

    def test_deleting_an_absent_document_is_not_an_error(self, store: VectorStore) -> None:
        assert store.delete_by_document("never_ingested") == 0

    def test_a_shorter_revision_leaves_no_orphans(self, store: VectorStore) -> None:
        """Superseded clinical text staying retrievable is the failure being prevented."""
        store.upsert_chunks([_chunk("doc_a", i) for i in range(6)])
        store.delete_by_document("doc_a")
        store.upsert_chunks([_chunk("doc_a", i) for i in range(2)])

        assert store.count_for_document("doc_a") == 2


class TestPointIdentity:
    def test_the_mapping_is_deterministic(self) -> None:
        assert point_id_for("doc::0000") == point_id_for("doc::0000")

    def test_different_chunks_get_different_points(self) -> None:
        assert point_id_for("doc::0000") != point_id_for("doc::0001")

    def test_it_is_a_valid_uuid(self) -> None:
        """Qdrant rejects anything that is not an unsigned int or a UUID."""
        uuid.UUID(point_id_for("doc::0000"))

    def test_qdrant_accepts_it(self, store: VectorStore) -> None:
        store.upsert_chunks([_chunk("doc_a", 0)])
        found = store.client.retrieve(
            collection_name=store.collection_name, ids=[point_id_for("doc_a::0000")]
        )
        assert len(found) == 1


class TestRecreate:
    def test_it_empties_the_collection(self, store: VectorStore) -> None:
        store.upsert_chunks([_chunk("doc_a", i) for i in range(3)])
        store.recreate_collection()

        assert store.collection_points_count() == 0

    def test_the_collection_is_usable_again_afterwards(self, store: VectorStore) -> None:
        store.recreate_collection()
        store.upsert_chunks([_chunk("doc_a", 0)])

        assert store.count_for_document("doc_a") == 1
