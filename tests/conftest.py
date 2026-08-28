"""Shared test fixtures.

The central problem this file solves: :func:`medguard.config.get_settings` is cached
and reads a ``.env`` file. Left alone, the suite would behave differently on a machine
that has real credentials than on one that does not -- tests would pass locally and
fail in CI, or worse, silently make live API calls and spend quota.

So :func:`isolated_env` runs for **every** test, stripping the real environment and
clearing the settings cache. Any test needing particular configuration sets it
explicitly. No test in this suite touches the network.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from medguard.config import Settings, get_settings
from medguard.observability.tracing import get_tracer

#: Every environment variable the application reads. Cleared before each test so a
#: developer's real `.env` or shell exports cannot leak into assertions.
_APP_ENV_VARS = (
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_FALLBACK_MODEL",
    "GEMINI_TIMEOUT_S",
    "GEMINI_MAX_RETRIES",
    "GEMINI_TEMPERATURE",
    "GEMINI_MAX_OUTPUT_TOKENS",
    "GEMINI_RETRY_INITIAL_BACKOFF_S",
    "GEMINI_RETRY_MAX_BACKOFF_S",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "QDRANT_COLLECTION",
    "QDRANT_PREFER_GRPC",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
    "LANGFUSE_HOST",
    "LANGFUSE_ENABLED",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_SPARSE_MODEL",
    "EMBEDDING_BATCH_SIZE",
    "INGESTION_CHUNK_SIZE",
    "INGESTION_CHUNK_OVERLAP",
    "INGESTION_CHUNK_MIN_CHARS",
    "INGESTION_CHUNK_MAX_CHARS",
    "INGESTION_STRIP_RUNNING_HEADERS",
    "INGESTION_DRY_RUN_SAMPLE_CHUNKS",
    "INGESTION_BATCH_SIZE",
    "RETRIEVAL_TOP_K",
    "RETRIEVAL_FETCH_K",
    "RETRIEVAL_DEFAULT_RETRIEVER",
    "CORPUS_DATA_DIR",
    "CORPUS_SOURCES_FILE",
    "CORPUS_MANIFEST_FILE",
    "CORPUS_USER_AGENT",
    "CORPUS_DOWNLOAD_TIMEOUT_S",
    "CORPUS_MAX_RETRIES",
    "CORPUS_RETRY_INITIAL_BACKOFF_S",
    "CORPUS_RETRY_MAX_BACKOFF_S",
    "CORPUS_MIN_PDF_BYTES",
    "DATABASE_URL",
    "LOG_LEVEL",
    "LOG_FORMAT",
    "ENVIRONMENT",
    "API_HOST",
    "API_PORT",
)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[None]:  # type: ignore[no-untyped-def]
    """Give every test a clean, offline configuration environment."""
    for name in _APP_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    # Point at throwaway infrastructure so nothing can reach a real service.
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")

    # Point the corpus at an empty temporary directory. Without this, tests would read
    # and (via `corpus sync`) potentially rewrite the developer's real corpus and its
    # committed manifest.
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    monkeypatch.setenv("CORPUS_DATA_DIR", str(corpus_dir))
    monkeypatch.setenv("CORPUS_SOURCES_FILE", str(tmp_path / "sources.yaml"))
    monkeypatch.setenv("CORPUS_MANIFEST_FILE", str(tmp_path / "manifest.json"))

    get_settings.cache_clear()
    get_tracer.cache_clear()
    yield
    get_settings.cache_clear()
    get_tracer.cache_clear()


@pytest.fixture
def settings() -> Settings:
    """Freshly built settings reflecting the current (isolated) environment."""
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def pdf_bytes() -> bytes:
    """Bytes that satisfy the corpus checks: correct magic, above the size floor.

    Not a real PDF -- nothing in Phase 2 parses one. It only has to be distinguishable
    from the HTML consent pages and WAF blocks that the download path exists to reject.
    """
    return b"%PDF-1.7\n" + b"medguard synthetic corpus fixture\n" * 400


@pytest.fixture
def sources_yaml() -> str:
    """A minimal, valid corpus definition with one active and one retired entry."""
    return """
schema_version: 1
documents:
  - document_id: test_guideline_2024
    title: A Test Clinical Guideline, 2024
    source_url: https://example.invalid/guideline.pdf
    source_organization: CDC
    document_type: guideline
    publication_date: 2024-01-15
    local_filename: test_guideline_2024.pdf
    license_or_usage_note: Public domain test fixture.
  - document_id: test_retired_2019
    title: A Retired Test Document, 2019
    source_url: https://example.invalid/retired.pdf
    source_organization: WHO
    document_type: reference
    local_filename: test_retired_2019.pdf
    license_or_usage_note: Public domain test fixture.
    retired: true
    retired_reason: The publisher withdrew the URL.
"""


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """A ``TestClient`` over an app built from the isolated settings.

    Used as a context manager so FastAPI's lifespan actually runs -- otherwise startup
    logging and tracer wiring are never exercised and bugs there go unnoticed.
    """
    from medguard.api.main import create_app

    with TestClient(create_app(settings)) as test_client:
        yield test_client
