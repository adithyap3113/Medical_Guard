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
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient

from medguard.config import Settings, get_settings
from medguard.errors import VectorStoreError
from medguard.logging import get_logger

__all__ = ["VectorStore", "VectorStoreHealth"]

log = get_logger(__name__)

#: Sentinel URL selecting Qdrant's in-process mode. Used by the test suite so the
#: full stack is exercisable without Docker running.
IN_MEMORY_URL = ":memory:"


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

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("qdrant.close_failed", error=str(exc))
            finally:
                self._client = None
