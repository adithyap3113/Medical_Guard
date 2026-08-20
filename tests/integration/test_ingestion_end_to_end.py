"""The five idempotency runs, end to end against real Qdrant and real embedders.

Everything else in the suite tests a piece. This tests the promise: **running ingestion
twice over an unchanged corpus does no work, and running it after any kind of change does
exactly the work that change requires.** A RAG system that cannot make that claim has to
be rebuilt from scratch whenever anyone is unsure, which in practice means it is rebuilt
constantly and nobody trusts the index either way.

    Run 1  fresh corpus                      -> everything indexed
    Run 2  nothing changed                   -> zero writes, zero embeddings
    Run 3  one document edited               -> only that document replaced
    Run 4  --dry-run                         -> zero writes anywhere
    Run 5  collection deleted out of band    -> re-embedded, without re-parsing

Run 5 is the subtle one and the reason the skip predicate has three clauses rather than
two. The database still says every document is `indexed`, and every file still hashes to
what the manifest recorded -- so a two-clause predicate skips the entire corpus and
reports success over an empty collection. The third clause, comparing counts, is what
notices.

Marked ``integration`` and ``slow``: needs Qdrant running and downloads the real BGE
weights on first execution.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select

from medguard.config import Settings, get_settings
from medguard.ingestion.pipeline import IngestionOutcome, IngestionPipeline
from medguard.models import IngestionStatus
from medguard.storage.database import Database
from medguard.storage.schema import ChunkRow, DocumentRow
from medguard.storage.vector_store import VectorStore
from tests.fixtures.make_pdf import write_structured_pdf

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture
def live_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", f"medguard_e2e_{uuid.uuid4().hex[:8]}")
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def corpus(live_settings: Settings) -> dict[str, Path]:
    raw = live_settings.corpus.raw_dir
    paths = {
        "doc_one": write_structured_pdf(raw / "doc_one.pdf", pages=4),
        "doc_two": write_structured_pdf(raw / "doc_two.pdf", pages=2),
    }
    _write_manifest(live_settings, paths)
    return paths


def _write_manifest(settings: Settings, paths: dict[str, Path]) -> Path:
    from medguard.corpus.manifest import sha256_file

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "documents": [
            {
                "document_id": document_id,
                "title": f"Title for {document_id}",
                "source_url": f"https://example.invalid/{document_id}.pdf",
                "source_organization": "CDC",
                "document_type": "guideline",
                "publication_date": "2024-01-15",
                "license_or_usage_note": "Public domain test fixture.",
                "local_filename": path.name,
                "sha256": sha256_file(path),
                "file_size_bytes": path.stat().st_size,
                "page_count": None,
                "retrieved_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
                "version_or_revision": None,
            }
            for document_id, path in sorted(paths.items())
        ],
    }
    settings.corpus.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return settings.corpus.manifest_path


class CountingParser:
    """Wraps the real parser so "was this re-parsed?" is answerable.

    Run 5 asserts that a re-embed after the collection was dropped does *not* re-parse.
    Re-parsing 20 PDFs is minutes of work that the chunk rows in SQL already contain, and
    a pipeline that redoes it has misunderstood which artefact is expensive.
    """

    def __init__(self) -> None:
        self.parsed: list[str] = []

    def __call__(self, path: Path):  # type: ignore[no-untyped-def]
        from medguard.ingestion.parser import parse as real_parse

        self.parsed.append(Path(path).name)
        return real_parse(path)


@pytest.fixture
def harness(live_settings: Settings, corpus: dict[str, Path]):  # type: ignore[no-untyped-def]
    """A migrated database, a real collection, and real embedders."""
    from medguard.embeddings import get_dense_embedder, get_sparse_embedder

    store = VectorStore(live_settings)
    if not store.health().reachable:
        store.close()
        pytest.skip("Qdrant is not running; start it with docker compose")

    store.ensure_collection()
    database = Database(live_settings)
    database.create_all()

    dense = get_dense_embedder()
    sparse = get_sparse_embedder()

    def run(*, dry_run: bool = False, parser=None):  # type: ignore[no-untyped-def]
        with database.session() as session:
            pipeline = IngestionPipeline(
                session=session,
                vector_store=store,
                dense_embedder=dense,
                sparse_embedder=sparse,
                settings=live_settings.ingestion,
                raw_dir=live_settings.corpus.raw_dir,
            )
            pipeline.sync_documents_from_manifest(live_settings.corpus.manifest_path)
            if parser is not None:
                import medguard.ingestion.pipeline as module

                # `parse` is imported into the pipeline module but deliberately
                # absent from its __all__, which is why the type checker objects to
                # touching it here. Swapping the module-level symbol is the only way to
                # observe whether the pipeline parsed at all.
                original = module.parse  # type: ignore[attr-defined]
                module.parse = parser  # type: ignore[attr-defined]
                try:
                    return pipeline.ingest_all(dry_run=dry_run)
                finally:
                    module.parse = original  # type: ignore[attr-defined]
            return pipeline.ingest_all(dry_run=dry_run)

    try:
        yield run, store, database, live_settings
    finally:
        try:
            store.client.delete_collection(store.collection_name)
        finally:
            store.close()
            database.dispose()


def test_run_1_a_fresh_corpus_is_fully_indexed(harness) -> None:  # type: ignore[no-untyped-def]
    run, store, database, _ = harness

    summary = run()

    assert len(summary.ingested) == 2
    assert not summary.failed
    assert summary.total_chunks > 0
    assert store.collection_points_count() == summary.total_chunks

    with database.session() as session:
        rows = session.scalars(select(DocumentRow)).all()
        assert {r.ingestion_status for r in rows} == {IngestionStatus.INDEXED}
        assert all(r.page_count for r in rows)
        assert all(r.indexed_sha256 for r in rows)


def test_run_2_an_unchanged_corpus_does_no_work(harness) -> None:  # type: ignore[no-untyped-def]
    run, store, _database, _ = harness
    first = run()
    points_before = store.collection_points_count()

    parser = CountingParser()
    second = run(parser=parser)

    assert len(second.skipped) == 2
    assert not second.ingested
    assert parser.parsed == [], "an unchanged document must not even be parsed"
    assert store.collection_points_count() == points_before == first.total_chunks


def test_run_3_an_edited_document_is_replaced_alone(harness) -> None:  # type: ignore[no-untyped-def]
    run, store, database, settings = harness
    run()
    two_before = store.count_for_document("doc_two")

    # The publisher revises one document; `corpus download` records the new bytes.
    write_structured_pdf(settings.corpus.raw_dir / "doc_one.pdf", pages=6)
    _write_manifest(
        settings,
        {
            "doc_one": settings.corpus.raw_dir / "doc_one.pdf",
            "doc_two": settings.corpus.raw_dir / "doc_two.pdf",
        },
    )

    summary = run()
    outcomes = {r.document_id: r.outcome for r in summary.results}

    assert outcomes["doc_one"] is IngestionOutcome.INGESTED
    assert outcomes["doc_two"] is IngestionOutcome.SKIPPED
    assert store.count_for_document("doc_two") == two_before

    with database.session() as session:
        rows = session.scalars(select(ChunkRow).where(ChunkRow.document_id == "doc_one")).all()
        assert store.count_for_document("doc_one") == len(rows)


def test_run_4_a_dry_run_writes_nothing(harness) -> None:  # type: ignore[no-untyped-def]
    run, store, database, _ = harness
    run()
    points_before = store.collection_points_count()
    with database.session() as session:
        chunks_before = len(session.scalars(select(ChunkRow)).all())

    summary = run(dry_run=True)

    assert summary.dry_run
    assert store.collection_points_count() == points_before
    with database.session() as session:
        assert len(session.scalars(select(ChunkRow)).all()) == chunks_before


def test_run_5_a_dropped_collection_is_rebuilt_without_re_parsing(harness) -> None:  # type: ignore[no-untyped-def]
    """The case a two-clause skip predicate reports success on, over an empty index."""
    run, store, _database, _ = harness
    first = run()

    # Out of band: someone drops the collection. SQL still says everything is indexed.
    store.client.delete_collection(store.collection_name)
    store.ensure_collection()
    assert store.collection_points_count() == 0

    parser = CountingParser()
    second = run(parser=parser)

    assert len(second.ingested) == 2, "the count mismatch must be noticed"
    assert store.collection_points_count() == first.total_chunks
    assert second.total_chunks == first.total_chunks
    # The point of the run: the chunk rows in SQL are already exactly what a re-parse
    # would produce, so re-parsing 20 PDFs to arrive at a byte-identical answer is pure
    # waste. Chunks are the durable artefact; the vector store is derived from them.
    assert parser.parsed == [], "recovery must re-embed, not re-parse"
    assert all("re-embedded from stored chunks" in (r.reason or "") for r in second.ingested)


def test_the_index_and_the_database_agree_after_every_run(harness) -> None:  # type: ignore[no-untyped-def]
    """The cheapest available signal that the two stores have not drifted."""
    run, store, database, _ = harness
    run()

    with database.session() as session:
        chunks = len(session.scalars(select(ChunkRow)).all())

    assert store.collection_points_count() == chunks
