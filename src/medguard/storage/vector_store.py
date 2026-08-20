"""Qdrant vector store adapter.

Phase 1 scope: **connectivity and health reporting only.** No collection creation, no
upserts, no search. Those arrive in Phases 3-4, once there is something to index.

The adapter exists this early for two reasons. First, ``medguard doctor`` needs to
tell you whether Qdrant is actually reachable before you spend an hour ingesting into
nothing. Second, fixing the client construction and collection-naming rules now means
Phase 4 adds methods to a working object instead of inventing infrastructure while
also debugging retrieval.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient, models

from medguard.config import Settings, get_settings
from medguard.constants import MEDGUARD_UUID_NAMESPACE
from medguard.embeddings.base import SparseVector
from medguard.errors import CollectionMismatchError, VectorStoreError
from medguard.logging import get_logger

__all__ = [
    "DENSE_VECTOR",
    "SPARSE_VECTOR",
    "ChunkWithVectors",
    "VectorStore",
    "VectorStoreHealth",
    "point_id_for",
]

log = get_logger(__name__)

#: Sentinel URL selecting Qdrant's in-process mode. Used by the test suite so the
#: full stack is exercisable without Docker running.
IN_MEMORY_URL = ":memory:"

#: Names of the two vectors stored on every point. Named vectors, not two collections:
#: one point carries both, so a dense hit and a sparse hit refer to the same object and
#: fusion has nothing to reconcile. Renaming either invalidates an existing collection,
#: so they are constants rather than literals scattered through the call sites.
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"

#: Payload fields that get an index. Only these four: an index costs memory and write
#: throughput, and is worth it exactly where a filter will actually be applied.
#: ``document_id`` is required by delete-by-document, which is the operation that makes
#: re-ingestion idempotent, and the other three are the metadata filters retrieval will
#: use. ``text`` deliberately has no index -- BM25 is the lexical leg, and a full-text
#: index over the same field would double the write cost to duplicate it.
_PAYLOAD_INDEXES: tuple[tuple[str, models.PayloadSchemaType], ...] = (
    ("document_id", models.PayloadSchemaType.KEYWORD),
    ("source_organization", models.PayloadSchemaType.KEYWORD),
    ("document_type", models.PayloadSchemaType.KEYWORD),
    ("publication_date", models.PayloadSchemaType.DATETIME),
)


def point_id_for(chunk_id: str) -> str:
    """Map a chunk id onto the Qdrant point id that stores it.

    Qdrant point ids must be an unsigned integer or a UUID; ``"cdc_anthrax_2023::0007"``
    is neither. ``uuid5`` is a hash, so the mapping is deterministic and needs no lookup
    table: the same chunk id always lands on the same point, which is what makes an
    upsert of an unchanged document a no-op instead of a duplicate.

    The determinism is entirely relative to :data:`MEDGUARD_UUID_NAMESPACE`. Changing
    that constant changes every point id at once, leaving the old points in place beside
    the new ones with nothing raising.
    """
    return str(uuid.uuid5(MEDGUARD_UUID_NAMESPACE, chunk_id))


class VectorStoreHealth(BaseModel):
    """Diagnostic snapshot of the vector database."""

    model_config = ConfigDict(frozen=True)

    reachable: bool = False
    url: str
    collection: str
    collection_exists: bool | None = Field(
        default=None, description="None means 'could not check', not 'absent'."
    )
    points_count: int | None = Field(default=None, ge=0)
    collections: list[str] = Field(default_factory=list)
    latency_ms: float | None = Field(default=None, ge=0)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.reachable


class VectorStore:
    """Thin wrapper over ``QdrantClient``.

    Owns client construction and the effective collection name so that no caller has
    to reason about either. The collection name is derived from the embedding model
    and dimension (see :meth:`Settings.collection_name`), which prevents the silent
    corruption that happens when vectors from two different models land in one
    collection.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._s = settings if settings is not None else get_settings()
        self._client: QdrantClient | None = None

    @property
    def collection_name(self) -> str:
        return self._s.collection_name

    @property
    def url(self) -> str:
        return self._s.qdrant.url

    def _build_client(self) -> QdrantClient:
        url = self._s.qdrant.url
        if url == IN_MEMORY_URL:
            # In-process Qdrant: no server, no Docker, no persistence. Tests only.
            return QdrantClient(location=IN_MEMORY_URL)

        api_key = self._s.qdrant.api_key
        return QdrantClient(
            url=url,
            api_key=api_key.get_secret_value() if api_key else None,
            timeout=self._s.qdrant.timeout_s,
            prefer_grpc=self._s.qdrant.prefer_grpc,
            # Skip the client/server version handshake: it emits a noisy warning on
            # perfectly workable version pairs and costs a round trip on every start.
            check_compatibility=False,
        )

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def health(self) -> VectorStoreHealth:
        """Check reachability. Never raises -- diagnostics report, they do not abort."""
        started = time.perf_counter()
        try:
            collections = [c.name for c in self.client.get_collections().collections]
        except Exception as exc:  # noqa: BLE001 - any failure here means "unreachable"
            return VectorStoreHealth(
                reachable=False,
                url=self.url,
                collection=self.collection_name,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )

        exists = self.collection_name in collections
        points: int | None = None
        if exists:
            try:
                info: Any = self.client.get_collection(self.collection_name)
                points = getattr(info, "points_count", None)
            except Exception as exc:  # noqa: BLE001
                log.debug("qdrant.collection_info_failed", error=str(exc))

        return VectorStoreHealth(
            reachable=True,
            url=self.url,
            collection=self.collection_name,
            collection_exists=exists,
            points_count=points,
            collections=sorted(collections),
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def require_healthy(self) -> None:
        """Raise :class:`VectorStoreError` unless Qdrant is reachable.

        For code paths that genuinely cannot proceed, as opposed to diagnostics.
        """
        status = self.health()
        if not status.reachable:
            raise VectorStoreError(f"Qdrant unreachable at {self.url}: {status.error}")

    # -- collection lifecycle ------------------------------------------------

    def _expected_vectors(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """The vector configuration this build requires, dense and sparse."""
        dense = {
            DENSE_VECTOR: models.VectorParams(
                size=self._s.embedding.dimension,
                distance=models.Distance.COSINE,
            )
        }
        sparse = {
            SPARSE_VECTOR: models.SparseVectorParams(
                # IDF is not optional. Without it Qdrant stores the raw term weights and
                # scores every document almost equally, so the sparse leg contributes
                # noise that looks like a working BM25 until someone measures it.
                modifier=models.Modifier.IDF,
            )
        }
        return dense, sparse

    def ensure_collection(self) -> None:
        """Create the collection if absent; verify its shape if present.

        Idempotent, and deliberately *not* self-healing. A collection whose vector
        configuration disagrees with this build raises rather than being recreated:
        dropping it silently would destroy an index that may have taken an hour to
        build, and writing into it anyway is worse -- vectors from two different models
        in one space produce retrieval that is confidently wrong and that no downstream
        check can detect. Refusing, and naming the difference, leaves the decision with
        the person who knows which side is the mistake.
        """
        name = self.collection_name
        dense, sparse = self._expected_vectors()

        try:
            exists = self.client.collection_exists(name)
        except Exception as exc:
            raise VectorStoreError(f"Qdrant unreachable at {self.url}: {exc}") from exc

        if exists:
            self._verify_collection(name)
            self._ensure_payload_indexes(name)
            log.debug("qdrant.collection_verified", collection=name)
            return

        log.info(
            "qdrant.creating_collection",
            collection=name,
            dimension=self._s.embedding.dimension,
        )
        try:
            self.client.create_collection(
                collection_name=name,
                vectors_config=dense,
                sparse_vectors_config=sparse,
            )
        except Exception as exc:
            # A concurrent creator is a success, not a failure: both callers wanted the
            # collection to exist, and it now does.
            if self.client.collection_exists(name):
                self._verify_collection(name)
            else:
                raise VectorStoreError(f"could not create collection {name!r}: {exc}") from exc

        self._ensure_payload_indexes(name)
        log.info("qdrant.collection_ready", collection=name)

    def _verify_collection(self, name: str) -> None:
        """Raise :class:`CollectionMismatchError` if the live shape is not the expected one."""
        try:
            info: Any = self.client.get_collection(name)
            config = info.config.params
        except Exception as exc:
            raise VectorStoreError(f"could not read collection {name!r}: {exc}") from exc

        vectors = config.vectors or {}
        if not isinstance(vectors, dict) or DENSE_VECTOR not in vectors:
            raise CollectionMismatchError(
                f"collection {name!r} has no {DENSE_VECTOR!r} named vector. It was "
                "probably created by an older build or by hand; drop it and re-ingest, "
                "or point QDRANT_COLLECTION at a new name."
            )

        live = vectors[DENSE_VECTOR]
        expected_size = self._s.embedding.dimension
        if int(live.size) != expected_size:
            raise CollectionMismatchError(
                f"collection {name!r} stores {live.size}-dimensional vectors but this "
                f"build produces {expected_size}. These are different vector spaces; "
                "writing into it would corrupt retrieval silently."
            )
        if live.distance != models.Distance.COSINE:
            raise CollectionMismatchError(
                f"collection {name!r} uses {live.distance} distance, not cosine. Scores "
                "would not be comparable with anything already indexed."
            )

        sparse_config = getattr(config, "sparse_vectors", None) or {}
        if SPARSE_VECTOR not in sparse_config:
            raise CollectionMismatchError(
                f"collection {name!r} has no {SPARSE_VECTOR!r} sparse vector; the "
                "lexical half of hybrid retrieval has nowhere to be stored."
            )
        modifier = getattr(sparse_config[SPARSE_VECTOR], "modifier", None)
        if modifier != models.Modifier.IDF:
            raise CollectionMismatchError(
                f"collection {name!r} stores {SPARSE_VECTOR!r} without the IDF modifier. "
                "BM25 scores would be near-uniform, which looks like a working sparse "
                "leg and is not one."
            )

    def _ensure_payload_indexes(self, name: str) -> None:
        """Create the payload indexes, tolerating the ones that already exist.

        Qdrant has no "create index if absent", and re-creating an existing index is an
        error rather than a no-op, so the exception is the control flow here. Failing to
        create one is logged and not raised: an index is a performance property, except
        for ``document_id``, whose absence is caught by the delete path itself.
        """
        if self.url == IN_MEMORY_URL:
            # In-process Qdrant ignores payload indexes and warns loudly about it once
            # per field per run. Filtering still works there -- by full scan, over a test
            # collection of a few dozen points -- so skipping is free and keeps the
            # warning out of every test's output.
            log.debug("qdrant.payload_indexes_skipped", reason="in-memory mode ignores them")
            return

        for field, schema in _PAYLOAD_INDEXES:
            try:
                self.client.create_payload_index(
                    collection_name=name, field_name=field, field_schema=schema
                )
            except Exception as exc:  # noqa: BLE001 - "already exists" is the common case
                log.debug("qdrant.payload_index_skipped", field=field, reason=str(exc))

    def recreate_collection(self) -> None:
        """Drop the collection and build it again. Destructive, and only ever explicit.

        Reachable solely through ``medguard ingest --force-recreate``. Every embedding in
        the collection is discarded, and rebuilding the shipped corpus takes on the order
        of an hour on the hardware this project targets.
        """
        name = self.collection_name
        log.warning("qdrant.recreating_collection", collection=name, detail="all points deleted")
        try:
            self.client.delete_collection(name)
        except Exception as exc:  # noqa: BLE001 - absent is an acceptable starting state
            log.debug("qdrant.delete_collection_failed", collection=name, error=str(exc))
        self.ensure_collection()

    # -- writes --------------------------------------------------------------

    def upsert_chunks(self, chunks: list[ChunkWithVectors]) -> None:
        """Write chunks as points, overwriting any point with the same id.

        Batching is the caller's decision, not this method's. The pipeline sizes batches
        against available memory and can halve them under pressure; a fixed batch size
        buried here would silently override that.
        """
        if not chunks:
            return

        points = [
            models.PointStruct(
                id=chunk.point_id,
                vector={
                    DENSE_VECTOR: chunk.dense,
                    SPARSE_VECTOR: models.SparseVector(
                        indices=chunk.sparse.indices, values=chunk.sparse.values
                    ),
                },
                payload=chunk.payload(),
            )
            for chunk in chunks
        ]

        try:
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        except Exception as exc:
            raise VectorStoreError(
                f"upsert of {len(points)} point(s) failed: {type(exc).__name__}: {exc}"
            ) from exc

        log.debug("qdrant.upserted", collection=self.collection_name, points=len(points))

    def delete_by_document(self, document_id: str) -> int:
        """Remove every point belonging to one document. Returns how many went.

        Called before re-indexing a document, so that a revision producing *fewer* chunks
        does not leave the tail of the previous revision in the index. Those orphans stay
        retrievable and citable, and they are superseded clinical text -- the exact
        failure the corpus rules exist to prevent.

        The count is measured before and after rather than read from the operation
        result, because Qdrant's delete response reports the operation's status, not the
        number of points it matched.
        """
        before = self.count_for_document(document_id)
        if before == 0:
            return 0

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=_document_filter(document_id)),
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"delete of document {document_id!r} failed: {type(exc).__name__}: {exc}"
            ) from exc

        after = self.count_for_document(document_id)
        removed = before - after
        log.debug("qdrant.deleted_document", document_id=document_id, points=removed)
        return removed

    # -- counts --------------------------------------------------------------

    def count_for_document(self, document_id: str) -> int:
        """How many points this document currently has in the collection."""
        try:
            result = self.client.count(
                collection_name=self.collection_name,
                count_filter=_document_filter(document_id),
                exact=True,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"count for document {document_id!r} failed: {type(exc).__name__}: {exc}"
            ) from exc
        return int(result.count)

    def collection_points_count(self) -> int:
        """Total points in the collection.

        Used by the ``SQL chunk count == Qdrant point count`` check, which is the
        cheapest available signal that the two stores have drifted apart.
        """
        try:
            return int(self.client.count(collection_name=self.collection_name, exact=True).count)
        except Exception as exc:
            raise VectorStoreError(
                f"count for collection {self.collection_name!r} failed: {type(exc).__name__}: {exc}"
            ) from exc

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("qdrant.close_failed", error=str(exc))
            finally:
                self._client = None


def _document_filter(document_id: str) -> models.Filter:
    """Match every point belonging to one document.

    Depends on the ``document_id`` payload index created by
    :meth:`VectorStore.ensure_collection`. Without it Qdrant still answers, by scanning
    every point in the collection -- correct, and slow enough to matter once the corpus
    is fully indexed.
    """
    return models.Filter(
        must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
    )


# ---------------------------------------------------------------------------
# Phase 3B: the write path
# ---------------------------------------------------------------------------
#
# Everything above is Phase 1 health reporting and is unchanged. What follows is the
# collection lifecycle and the write path -- creation, upsert, delete-by-document and
# the counts that make idempotency checkable.
#
# The single most important decision here is the point id. Qdrant accepts only an
# unsigned integer or a UUID, and MedGuard's chunk ids are position-addressed strings
# ("cdc_anthrax_2023::0007"). Bridging them with uuid5 over a fixed namespace makes the
# mapping deterministic: the same chunk id always produces the same point, so
# re-ingesting an unchanged document upserts over itself rather than doubling the index.
# See medguard.constants for why that namespace must never be regenerated.


class ChunkWithVectors(BaseModel):
    """One chunk, ready to be written to Qdrant: metadata plus both vectors.

    A single type carrying both vectors, rather than parallel lists, because the failure
    mode of parallel lists is silent. A dense list and a sparse list that fall out of
    step by one index every chunk under the wrong text, retrieval keeps working, and
    nothing reports an error.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    chunk_index: int = Field(ge=0)
    text: str
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    section: str | None = None
    title: str | None = None
    source_organization: str | None = None
    document_type: str | None = None
    publication_date: str | None = Field(
        default=None,
        description="ISO date string, or None. Qdrant's datetime index accepts RFC 3339; "
        "None means the publisher states no date, never 'unknown so guessed'.",
    )
    dense: list[float]
    sparse: SparseVector

    @property
    def point_id(self) -> str:
        """The deterministic Qdrant point id for this chunk."""
        return point_id_for(self.chunk_id)

    def payload(self) -> dict[str, Any]:
        """What gets stored alongside the vectors.

        The chunk text is included deliberately. It costs disk, and it means a retrieval
        result is self-contained: the API can render a citation without a second lookup
        into SQL, and an outage of the relational store degrades the system to "no
        ingestion" rather than "no answers".
        """
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section,
            "title": self.title,
            "source_organization": self.source_organization,
            "document_type": self.document_type,
            "publication_date": self.publication_date,
            "text": self.text,
        }
