"""The orchestrator: state transitions, failure isolation, skip logic, dry run.

Fakes throughout, and that is the design working rather than a shortcut. The parser,
cleaner and chunker are pure and tested directly; what is left in the pipeline is
sequencing, error handling and state, and all three are far more thoroughly exercised by
a fake that can be made to fail on command than by a real embedder that succeeds.

The two properties that matter most:

**Document 7 failing must not touch documents 8 through 20.** A corpus contains scanned
PDFs, truncated downloads and files a publisher revised in place. If any one of those
ends the run, ingestion is unusable in practice.

**A dry run must write nothing.** Not "writes little" -- nothing. It is the command
someone runs to inspect chunk boundaries before committing to an hour of embedding, and
it is worthless if it can leave state behind.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from sqlalchemy import select

from medguard.config import IngestionSettings, Settings
from medguard.embeddings.base import SparseVector
from medguard.errors import EmbeddingError, ParseError
from medguard.ingestion.pipeline import (
    RAM_ABORT_MB,
    DocumentIngestionResult,
    IngestionOutcome,
    IngestionPipeline,
    IngestionSummary,
)
from medguard.models import DocumentType, IngestionStatus, SourceOrganization
from medguard.storage.database import Database
from medguard.storage.schema import ChunkRow, DocumentRow
from tests.fixtures.make_pdf import write_structured_pdf

DIMENSION = 384


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeDense:
    """A dense embedder with a call log and a switchable failure."""

    model_id = "fake-dense"
    dimension = DIMENSION

    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls: list[list[str]] = []
        self._fail_on_call = fail_on_call

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self._fail_on_call is not None and len(self.calls) >= self._fail_on_call:
            raise EmbeddingError("fake dense embedder failed")
        return [[0.01 * (i + 1)] * DIMENSION for i, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class FakeSparse:
    model_id = "fake-sparse"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        self.calls.append(list(texts))
        return [SparseVector(indices=[1, 2], values=[0.5, 0.25]) for _ in texts]

    def embed_query(self, text: str) -> SparseVector:
        return self.embed_documents([text])[0]


class FakeStore:
    """An in-memory stand-in for Qdrant that records everything done to it."""

    def __init__(self) -> None:
        self.points: dict[str, str] = {}  # point_id -> document_id
        self.upserts: list[int] = []
        self.deletes: list[str] = []
        self.ensure_calls = 0

    def ensure_collection(self) -> None:
        self.ensure_calls += 1

    def upsert_chunks(self, chunks: list[object]) -> None:
        self.upserts.append(len(chunks))
        for chunk in chunks:
            self.points[chunk.point_id] = chunk.document_id  # type: ignore[attr-defined]

    def delete_by_document(self, document_id: str) -> int:
        self.deletes.append(document_id)
        removed = [pid for pid, did in self.points.items() if did == document_id]
        for pid in removed:
            del self.points[pid]
        return len(removed)

    def count_for_document(self, document_id: str) -> int:
        return sum(1 for did in self.points.values() if did == document_id)

    def collection_points_count(self) -> int:
        return len(self.points)


class RefusingStore(FakeStore):
    """Raises on every write, so "a dry run writes nothing" is provable."""

    def upsert_chunks(self, chunks: list[object]) -> None:
        raise AssertionError("upsert_chunks must not be called")

    def delete_by_document(self, document_id: str) -> int:
        raise AssertionError("delete_by_document must not be called")


class RefusingDense(FakeDense):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("the embedder must not be loaded during a dry run")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def database(settings: Settings) -> Database:
    db = Database(settings)
    db.create_all()
    return db


@pytest.fixture
def corpus(settings: Settings) -> dict[str, Path]:
    """Two real generated PDFs in the configured raw directory."""
    raw = settings.corpus.raw_dir
    return {
        "doc_one": write_structured_pdf(raw / "doc_one.pdf", pages=3),
        "doc_two": write_structured_pdf(raw / "doc_two.pdf", pages=2),
    }


def _row(document_id: str, path: Path, **overrides: object) -> DocumentRow:
    from medguard.corpus.manifest import sha256_file

    values: dict[str, object] = {
        "document_id": document_id,
        "title": f"Title for {document_id}",
        "source_url": f"https://example.invalid/{document_id}.pdf",
        "source_organization": SourceOrganization.CDC,
        "document_type": DocumentType.GUIDELINE,
        "publication_date": date(2024, 1, 15),
        "local_filename": path.name,
        "sha256": sha256_file(path),
        "file_size_bytes": path.stat().st_size,
        "retrieved_at": datetime(2026, 1, 1, tzinfo=UTC),
        "ingestion_status": IngestionStatus.PENDING,
        "license_or_usage_note": "Public domain test fixture.",
    }
    values.update(overrides)
    return DocumentRow(**values)


def _pipeline(
    session: object,
    store: object,
    *,
    dense: object | None = None,
    sparse: object | None = None,
    settings: IngestionSettings | None = None,
    raw_dir: Path | None = None,
) -> IngestionPipeline:
    return IngestionPipeline(
        session=session,  # type: ignore[arg-type]
        vector_store=store,  # type: ignore[arg-type]
        dense_embedder=dense or FakeDense(),  # type: ignore[arg-type]
        sparse_embedder=sparse or FakeSparse(),  # type: ignore[arg-type]
        settings=settings or IngestionSettings(),
        raw_dir=raw_dir,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_a_document_reaches_indexed(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        store = FakeStore()
        with database.session() as session:
            session.add(_row("doc_one", corpus["doc_one"]))
            session.commit()
            document = session.get(DocumentRow, "doc_one")
            assert document is not None

            result = _pipeline(session, store, raw_dir=settings.corpus.raw_dir).ingest_document(
                document
            )

            assert result.outcome is IngestionOutcome.INGESTED
            assert document.ingestion_status is IngestionStatus.INDEXED
            assert document.ingestion_error is None

    def test_the_page_count_is_recorded(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        """Null meant "not counted yet". After ingestion it must mean the real number."""
        store = FakeStore()
        with database.session() as session:
            session.add(_row("doc_one", corpus["doc_one"]))
            session.commit()
            document = session.get(DocumentRow, "doc_one")
            assert document is not None

            result = _pipeline(session, store, raw_dir=settings.corpus.raw_dir).ingest_document(
                document
            )

            assert result.page_count == 3
            assert document.page_count == 3

    def test_chunks_land_in_both_stores(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        store = FakeStore()
        with database.session() as session:
            session.add(_row("doc_one", corpus["doc_one"]))
            session.commit()
            document = session.get(DocumentRow, "doc_one")
            assert document is not None

            result = _pipeline(session, store, raw_dir=settings.corpus.raw_dir).ingest_document(
                document
            )

            rows = session.scalars(select(ChunkRow).where(ChunkRow.document_id == "doc_one")).all()
            assert len(rows) == result.chunk_count
            assert store.count_for_document("doc_one") == result.chunk_count

    def test_chunk_rows_carry_their_page_span(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        store = FakeStore()
        with database.session() as session:
            session.add(_row("doc_one", corpus["doc_one"]))
            session.commit()
            document = session.get(DocumentRow, "doc_one")
            assert document is not None
            _pipeline(session, store, raw_dir=settings.corpus.raw_dir).ingest_document(document)

            rows = session.scalars(select(ChunkRow).where(ChunkRow.document_id == "doc_one")).all()
            for row in rows:
                assert row.page_start is not None
                assert row.page_end is not None
                assert row.page_start >= 1
                assert row.page_end >= row.page_start

    def test_embedding_happens_in_batches(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        dense = FakeDense()
        with database.session() as session:
            session.add(_row("doc_one", corpus["doc_one"]))
            session.commit()
            document = session.get(DocumentRow, "doc_one")
            assert document is not None

            ingestion = IngestionSettings(batch_size=2)
            _pipeline(
                session,
                FakeStore(),
                dense=dense,
                settings=ingestion,
                raw_dir=settings.corpus.raw_dir,
            ).ingest_document(document)

            assert all(len(call) <= 2 for call in dense.calls)
            assert len(dense.calls) > 1


class TestSkipPredicate:
    """Three clauses, and each is tested with the other two satisfied."""

    def _ingest_once(
        self, database: Database, corpus: dict[str, Path], settings: Settings, store: FakeStore
    ) -> None:
        with database.session() as session:
            session.add(_row("doc_one", corpus["doc_one"]))
            session.commit()
            document = session.get(DocumentRow, "doc_one")
            assert document is not None
            _pipeline(session, store, raw_dir=settings.corpus.raw_dir).ingest_document(document)

    def test_an_unchanged_document_is_skipped(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        store = FakeStore()
        self._ingest_once(database, corpus, settings, store)

        with database.session() as session:
            document = session.get(DocumentRow, "doc_one")
            assert document is not None
            dense = FakeDense()
            result = _pipeline(
                session, store, dense=dense, raw_dir=settings.corpus.raw_dir
            ).ingest_document(document)

            assert result.outcome is IngestionOutcome.SKIPPED
            assert result.reason == "unchanged"
            assert dense.calls == [], "a skipped document must not be re-embedded"

    def test_a_document_not_yet_indexed_is_not_skipped(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        """Clause one: status."""
        store = FakeStore()
        self._ingest_once(database, corpus, settings, store)

        with database.session() as session:
            document = session.get(DocumentRow, "doc_one")
            assert document is not None
            document.ingestion_status = IngestionStatus.CHUNKED
            session.commit()

            result = _pipeline(session, store, raw_dir=settings.corpus.raw_dir).ingest_document(
                document
            )
            assert result.outcome is IngestionOutcome.INGESTED

    def test_a_changed_file_is_re_ingested(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        """Clause two: the hash. A publisher revising a document in place is silent."""
        store = FakeStore()
        self._ingest_once(database, corpus, settings, store)

        with database.session() as session:
            document = session.get(DocumentRow, "doc_one")
            assert document is not None
            # The file now differs from what was indexed, and the manifest agrees with
            # the new bytes -- which is what `corpus download` would have produced.
            write_structured_pdf(corpus["doc_one"], pages=5)
            from medguard.corpus.manifest import sha256_file

            document.sha256 = sha256_file(corpus["doc_one"])
            session.commit()

            result = _pipeline(session, store, raw_dir=settings.corpus.raw_dir).ingest_document(
                document
            )
            assert result.outcome is IngestionOutcome.INGESTED
            assert result.page_count == 5

    def test_an_emptied_collection_is_re_indexed(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        """Clause three: the count. This is the "collection dropped by hand" case."""
        store = FakeStore()
        self._ingest_once(database, corpus, settings, store)
        store.points.clear()

        with database.session() as session:
            document = session.get(DocumentRow, "doc_one")
            assert document is not None
            result = _pipeline(session, store, raw_dir=settings.corpus.raw_dir).ingest_document(
                document
            )

            assert result.outcome is IngestionOutcome.INGESTED
            assert store.count_for_document("doc_one") == result.chunk_count

    def test_a_partial_index_is_re_indexed(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        """Half a document in the collection answers queries with half a document."""
        store = FakeStore()
        self._ingest_once(database, corpus, settings, store)
        for point in list(store.points)[:2]:
            del store.points[point]

        with database.session() as session:
            document = session.get(DocumentRow, "doc_one")
            assert document is not None
            result = _pipeline(session, store, raw_dir=settings.corpus.raw_dir).ingest_document(
                document
            )
            assert result.outcome is IngestionOutcome.INGESTED


class TestIndexedHash:
    """`documents.sha256` describes the file; `indexed_sha256` describes the index.

    Keeping them apart is what makes a revised document get re-ingested. With one column
    doing both jobs, `corpus download` moves the hash forward, the skip predicate then
    compares the file to itself, and the superseded edition stays in the index forever
    while every check reports success.
    """

    def test_it_is_null_before_ingestion(self, database: Database, corpus: dict[str, Path]) -> None:
        with database.session() as session:
            session.add(_row("doc_one", corpus["doc_one"]))
            session.commit()
            document = session.get(DocumentRow, "doc_one")
            assert document is not None
            assert document.indexed_sha256 is None

    def test_it_is_written_on_success(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        from medguard.corpus.manifest import sha256_file

        with database.session() as session:
            session.add(_row("doc_one", corpus["doc_one"]))
            session.commit()
            document = session.get(DocumentRow, "doc_one")
            assert document is not None

            _pipeline(session, FakeStore(), raw_dir=settings.corpus.raw_dir).ingest_document(
                document
            )

            assert document.indexed_sha256 == sha256_file(corpus["doc_one"])

    def test_it_is_not_written_on_failure(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        """Claiming the index holds these bytes when embedding failed is a lie."""
        with database.session() as session:
            session.add(_row("doc_one", corpus["doc_one"]))
            session.commit()
            document = session.get(DocumentRow, "doc_one")
            assert document is not None

            _pipeline(
                session,
                FakeStore(),
                dense=FakeDense(fail_on_call=1),
                raw_dir=settings.corpus.raw_dir,
            ).ingest_document(document)

            reloaded = session.get(DocumentRow, "doc_one")
            assert reloaded is not None
            assert reloaded.indexed_sha256 is None

    def test_a_manifest_sync_does_not_touch_it(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        """The regression guard. This is precisely how the evidence gets erased."""
        manifest = _write_manifest(settings, corpus)
        with database.session() as session:
            pipeline = _pipeline(session, FakeStore(), raw_dir=settings.corpus.raw_dir)
            pipeline.sync_documents_from_manifest(manifest)
            pipeline.ingest_all()

            document = session.get(DocumentRow, "doc_one")
            assert document is not None
            indexed = document.indexed_sha256
            assert indexed is not None

            # A publisher revises the document in place and `corpus download` records it.
            write_structured_pdf(corpus["doc_one"], pages=5)
            _write_manifest(settings, corpus)
            pipeline.sync_documents_from_manifest(manifest)

            refreshed = session.get(DocumentRow, "doc_one")
            assert refreshed is not None
            assert refreshed.sha256 != indexed, "the manifest hash should have moved"
            assert refreshed.indexed_sha256 == indexed, "the indexed hash must not have"

    def test_the_revised_document_is_then_re_ingested(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        manifest = _write_manifest(settings, corpus)
        with database.session() as session:
            pipeline = _pipeline(session, FakeStore(), raw_dir=settings.corpus.raw_dir)
            pipeline.sync_documents_from_manifest(manifest)
            pipeline.ingest_all()

            write_structured_pdf(corpus["doc_one"], pages=5)
            _write_manifest(settings, corpus)
            pipeline.sync_documents_from_manifest(manifest)

            summary = pipeline.ingest_all()
            outcomes = {r.document_id: r.outcome for r in summary.results}

            assert outcomes["doc_one"] is IngestionOutcome.INGESTED
            assert outcomes["doc_two"] is IngestionOutcome.SKIPPED


class TestHashVerification:
    def test_a_mismatched_file_fails_without_parsing(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        """The trust boundary. Everything downstream assumes these are the right bytes."""
        store = FakeStore()
        with database.session() as session:
            session.add(_row("doc_one", corpus["doc_one"], sha256="b" * 64))
            session.commit()
            document = session.get(DocumentRow, "doc_one")
            assert document is not None

            result = _pipeline(session, store, raw_dir=settings.corpus.raw_dir).ingest_document(
                document
            )

            assert result.outcome is IngestionOutcome.FAILED
            assert "does not match the manifest" in (result.reason or "")
            assert store.upserts == []

    def test_a_missing_file_fails_with_a_useful_message(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        store = FakeStore()
        with database.session() as session:
            session.add(_row("doc_one", corpus["doc_one"]))
            session.commit()
            document = session.get(DocumentRow, "doc_one")
            assert document is not None
            corpus["doc_one"].unlink()

            result = _pipeline(session, store, raw_dir=settings.corpus.raw_dir).ingest_document(
                document
            )

            assert result.outcome is IngestionOutcome.FAILED
            assert "corpus download" in (result.reason or "")


class TestFailureIsolation:
    def test_a_parse_failure_writes_no_chunks(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        store = FakeStore()
        broken = settings.corpus.raw_dir / "broken.pdf"
        broken.write_bytes(b"%PDF-1.7\nthis is not a pdf body\n" * 50)

        with database.session() as session:
            session.add(_row("broken", broken))
            session.commit()
            document = session.get(DocumentRow, "broken")
            assert document is not None

            result = _pipeline(session, store, raw_dir=settings.corpus.raw_dir).ingest_document(
                document
            )

            assert result.outcome is IngestionOutcome.FAILED
            assert document.ingestion_status is IngestionStatus.FAILED
            assert document.ingestion_error
            assert session.scalars(select(ChunkRow)).all() == []
            assert store.upserts == []

    def test_an_embedding_failure_leaves_no_points_for_that_document(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        """Chunks may survive -- they are correct and expensive. Points must not."""
        store = FakeStore()
        with database.session() as session:
            session.add(_row("doc_one", corpus["doc_one"]))
            session.commit()
            document = session.get(DocumentRow, "doc_one")
            assert document is not None

            result = _pipeline(
                session,
                store,
                dense=FakeDense(fail_on_call=2),
                settings=IngestionSettings(batch_size=2),
                raw_dir=settings.corpus.raw_dir,
            ).ingest_document(document)

            assert result.outcome is IngestionOutcome.FAILED
            assert document.ingestion_status is IngestionStatus.FAILED
            assert store.count_for_document("doc_one") == 0

    def test_one_failure_does_not_stop_the_batch(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        """The property the whole design exists for."""
        store = FakeStore()
        broken = settings.corpus.raw_dir / "broken.pdf"
        broken.write_bytes(b"not a pdf")

        with database.session() as session:
            session.add(_row("aaa_first", corpus["doc_one"]))
            session.add(_row("bbb_broken", broken))
            session.add(_row("ccc_last", corpus["doc_two"]))
            session.commit()

            summary = _pipeline(session, store, raw_dir=settings.corpus.raw_dir).ingest_all()

            assert [r.outcome for r in summary.results] == [
                IngestionOutcome.INGESTED,
                IngestionOutcome.FAILED,
                IngestionOutcome.INGESTED,
            ]
            assert len(summary.failed) == 1
            assert summary.exit_code() == 1

    def test_an_unexpected_exception_is_still_isolated(
        self,
        database: Database,
        corpus: dict[str, Path],
        settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A bug in the parser is fatal to one document and harmless to the rest."""
        from medguard.ingestion import pipeline as pipeline_module

        # Captured before the patch. Referring to `pipeline_module.parse` inside the
        # replacement would call the replacement.
        real_parse = pipeline_module.parse  # type: ignore[attr-defined]

        def explode(path: Path) -> object:
            if path.name == "doc_one.pdf":
                raise RuntimeError("unexpected internal failure")
            return real_parse(path)

        store = FakeStore()
        with database.session() as session:
            session.add(_row("aaa_first", corpus["doc_one"]))
            session.add(_row("bbb_second", corpus["doc_two"]))
            session.commit()
            monkeypatch.setattr(pipeline_module, "parse", explode)

            summary = _pipeline(session, store, raw_dir=settings.corpus.raw_dir).ingest_all()

            assert len(summary.failed) == 1
            assert len(summary.ingested) == 1
            assert "unexpected RuntimeError" in (summary.failed[0].reason or "")

    def test_a_failed_document_records_why(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        """ "Failed" with no reason is a support ticket; with a reason it is a fix."""
        store = FakeStore()
        broken = settings.corpus.raw_dir / "broken.pdf"
        broken.write_bytes(b"nope")

        with database.session() as session:
            session.add(_row("broken", broken))
            session.commit()
            document = session.get(DocumentRow, "broken")
            assert document is not None
            _pipeline(session, store, raw_dir=settings.corpus.raw_dir).ingest_document(document)

            assert "broken.pdf" in (document.ingestion_error or "")


class TestDryRun:
    def test_it_writes_nothing_to_either_store(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        store = RefusingStore()
        with database.session() as session:
            session.add(_row("doc_one", corpus["doc_one"]))
            session.commit()
            document = session.get(DocumentRow, "doc_one")
            assert document is not None

            result = _pipeline(
                session, store, dense=RefusingDense(), raw_dir=settings.corpus.raw_dir
            ).ingest_document(document, dry_run=True)

            assert result.outcome is IngestionOutcome.INGESTED
            assert session.scalars(select(ChunkRow)).all() == []
            assert store.points == {}
            assert document.ingestion_status is IngestionStatus.PENDING

    def test_it_still_reports_real_numbers(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        """A dry run that reported zeros would tell you nothing worth knowing."""
        with database.session() as session:
            session.add(_row("doc_one", corpus["doc_one"]))
            session.commit()
            document = session.get(DocumentRow, "doc_one")
            assert document is not None

            result = _pipeline(
                session, RefusingStore(), dense=RefusingDense(), raw_dir=settings.corpus.raw_dir
            ).ingest_document(document, dry_run=True)

            assert result.page_count == 3
            assert result.chunk_count > 0
            assert result.total_chars > 0

    def test_it_does_not_leave_the_status_changed(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        with database.session() as session:
            session.add(_row("doc_one", corpus["doc_one"]))
            session.commit()
            document = session.get(DocumentRow, "doc_one")
            assert document is not None
            _pipeline(
                session, RefusingStore(), dense=RefusingDense(), raw_dir=settings.corpus.raw_dir
            ).ingest_document(document, dry_run=True)

        with database.session() as session:
            reloaded = session.get(DocumentRow, "doc_one")
            assert reloaded is not None
            assert reloaded.ingestion_status is IngestionStatus.PENDING
            assert reloaded.page_count is None

    def test_a_dry_run_reports_failures_too(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        broken = settings.corpus.raw_dir / "broken.pdf"
        broken.write_bytes(b"nope")

        with database.session() as session:
            session.add(_row("broken", broken))
            session.commit()
            document = session.get(DocumentRow, "broken")
            assert document is not None

            result = _pipeline(
                session, RefusingStore(), dense=RefusingDense(), raw_dir=settings.corpus.raw_dir
            ).ingest_document(document, dry_run=True)

            assert result.outcome is IngestionOutcome.FAILED
            assert document.ingestion_status is IngestionStatus.PENDING


class TestReIngestion:
    def test_a_shorter_revision_leaves_no_orphans(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        """The failure this guards is superseded clinical text staying retrievable."""
        from medguard.corpus.manifest import sha256_file

        store = FakeStore()
        with database.session() as session:
            session.add(_row("doc_one", corpus["doc_one"]))
            session.commit()
            document = session.get(DocumentRow, "doc_one")
            assert document is not None
            first = _pipeline(session, store, raw_dir=settings.corpus.raw_dir).ingest_document(
                document
            )

            write_structured_pdf(corpus["doc_one"], pages=1)
            document.sha256 = sha256_file(corpus["doc_one"])
            session.commit()

            second = _pipeline(session, store, raw_dir=settings.corpus.raw_dir).ingest_document(
                document
            )

            assert second.chunk_count < first.chunk_count
            assert store.count_for_document("doc_one") == second.chunk_count
            rows = session.scalars(select(ChunkRow).where(ChunkRow.document_id == "doc_one")).all()
            assert len(rows) == second.chunk_count

    def test_re_ingestion_does_not_touch_other_documents(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        store = FakeStore()
        with database.session() as session:
            session.add(_row("doc_one", corpus["doc_one"]))
            session.add(_row("doc_two", corpus["doc_two"]))
            session.commit()
            pipeline = _pipeline(session, store, raw_dir=settings.corpus.raw_dir)
            pipeline.ingest_all()
            before = store.count_for_document("doc_two")

            document = session.get(DocumentRow, "doc_one")
            assert document is not None
            document.ingestion_status = IngestionStatus.PENDING
            session.commit()
            pipeline.ingest_document(document)

            assert store.count_for_document("doc_two") == before


class TestSelection:
    def test_only_restricts_to_one_document(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        with database.session() as session:
            session.add(_row("doc_one", corpus["doc_one"]))
            session.add(_row("doc_two", corpus["doc_two"]))
            session.commit()

            summary = _pipeline(session, FakeStore(), raw_dir=settings.corpus.raw_dir).ingest_all(
                only="doc_two"
            )

            assert [r.document_id for r in summary.results] == ["doc_two"]

    def test_limit_caps_the_run(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        with database.session() as session:
            session.add(_row("doc_one", corpus["doc_one"]))
            session.add(_row("doc_two", corpus["doc_two"]))
            session.commit()

            summary = _pipeline(session, FakeStore(), raw_dir=settings.corpus.raw_dir).ingest_all(
                limit=1
            )

            assert len(summary.results) == 1


class TestManifestSync:
    def test_new_entries_become_pending_rows(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        manifest = _write_manifest(settings, corpus)
        with database.session() as session:
            added = _pipeline(
                session, FakeStore(), raw_dir=settings.corpus.raw_dir
            ).sync_documents_from_manifest(manifest)

            assert added == 2
            rows = session.scalars(select(DocumentRow)).all()
            assert {r.ingestion_status for r in rows} == {IngestionStatus.PENDING}

    def test_syncing_twice_adds_nothing(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        manifest = _write_manifest(settings, corpus)
        with database.session() as session:
            pipeline = _pipeline(session, FakeStore(), raw_dir=settings.corpus.raw_dir)
            pipeline.sync_documents_from_manifest(manifest)
            assert pipeline.sync_documents_from_manifest(manifest) == 0

    def test_sync_does_not_reset_ingestion_state(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        """The manifest has no opinion about whether a document has been indexed."""
        manifest = _write_manifest(settings, corpus)
        with database.session() as session:
            pipeline = _pipeline(session, FakeStore(), raw_dir=settings.corpus.raw_dir)
            pipeline.sync_documents_from_manifest(manifest)
            pipeline.ingest_all()

            pipeline.sync_documents_from_manifest(manifest)
            rows = session.scalars(select(DocumentRow)).all()
            assert {r.ingestion_status for r in rows} == {IngestionStatus.INDEXED}

    def test_sync_refreshes_the_fields_the_manifest_owns(
        self, database: Database, corpus: dict[str, Path], settings: Settings
    ) -> None:
        manifest = _write_manifest(settings, corpus)
        with database.session() as session:
            pipeline = _pipeline(session, FakeStore(), raw_dir=settings.corpus.raw_dir)
            pipeline.sync_documents_from_manifest(manifest)

            row = session.get(DocumentRow, "doc_one")
            assert row is not None
            row.title = "a stale title"
            session.commit()

            pipeline.sync_documents_from_manifest(manifest)
            refreshed = session.get(DocumentRow, "doc_one")
            assert refreshed is not None
            assert refreshed.title == "Title for doc_one"


def _write_manifest(settings: Settings, corpus: dict[str, Path]) -> Path:
    import json

    from medguard.corpus.manifest import sha256_file

    documents = [
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
        for document_id, path in sorted(corpus.items())
    ]
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "documents": documents,
    }
    settings.corpus.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return settings.corpus.manifest_path


class TestMemoryGuard:
    def test_ingestion_aborts_below_the_floor(
        self,
        database: Database,
        corpus: dict[str, Path],
        settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Being killed by the OS mid-batch leaves no error anywhere at all."""
        from medguard import diagnostics

        monkeypatch.setattr(diagnostics, "available_ram_mb", lambda: RAM_ABORT_MB - 1.0)

        store = FakeStore()
        with database.session() as session:
            session.add(_row("doc_one", corpus["doc_one"]))
            session.commit()
            document = session.get(DocumentRow, "doc_one")
            assert document is not None

            result = _pipeline(session, store, raw_dir=settings.corpus.raw_dir).ingest_document(
                document
            )

            assert result.outcome is IngestionOutcome.FAILED
            assert "memory available" in (result.reason or "")
            assert store.count_for_document("doc_one") == 0

    def test_the_batch_is_halved_under_pressure(
        self,
        database: Database,
        corpus: dict[str, Path],
        settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from medguard import diagnostics

        monkeypatch.setattr(diagnostics, "available_ram_mb", lambda: 200.0)

        dense = FakeDense()
        with database.session() as session:
            session.add(_row("doc_one", corpus["doc_one"]))
            session.commit()
            document = session.get(DocumentRow, "doc_one")
            assert document is not None

            _pipeline(
                session,
                FakeStore(),
                dense=dense,
                settings=IngestionSettings(batch_size=8),
                raw_dir=settings.corpus.raw_dir,
            ).ingest_document(document)

            assert max(len(call) for call in dense.calls) < 8


class TestSummary:
    def test_exit_code_is_zero_when_nothing_failed(self) -> None:
        summary = IngestionSummary(
            results=[
                DocumentIngestionResult(document_id="a", outcome=IngestionOutcome.INGESTED),
                DocumentIngestionResult(document_id="b", outcome=IngestionOutcome.SKIPPED),
            ]
        )
        assert summary.exit_code() == 0

    def test_all_skipped_is_success_not_a_warning(self) -> None:
        summary = IngestionSummary(
            results=[DocumentIngestionResult(document_id="a", outcome=IngestionOutcome.SKIPPED)]
        )
        assert summary.exit_code() == 0

    def test_one_failure_makes_the_run_fail(self) -> None:
        """Eighteen of twenty documents indexed is not a successful run."""
        summary = IngestionSummary(
            results=[
                DocumentIngestionResult(document_id="a", outcome=IngestionOutcome.INGESTED),
                DocumentIngestionResult(document_id="b", outcome=IngestionOutcome.FAILED),
            ]
        )
        assert summary.exit_code() == 1

    def test_the_chunk_total_is_summed(self) -> None:
        summary = IngestionSummary(
            results=[
                DocumentIngestionResult(
                    document_id="a", outcome=IngestionOutcome.INGESTED, chunk_count=12
                ),
                DocumentIngestionResult(
                    document_id="b", outcome=IngestionOutcome.INGESTED, chunk_count=8
                ),
            ]
        )
        assert summary.total_chunks == 20


class TestErrorTaxonomy:
    def test_parse_errors_reach_the_result_intact(
        self, database: Database, settings: Settings
    ) -> None:
        """The reason must name the document; PyMuPDF's own messages do not."""
        broken = settings.corpus.raw_dir / "identifiable.pdf"
        broken.write_bytes(b"not a pdf")

        with database.session() as session:
            session.add(_row("identifiable", broken))
            session.commit()
            document = session.get(DocumentRow, "identifiable")
            assert document is not None

            result = _pipeline(
                session, FakeStore(), raw_dir=settings.corpus.raw_dir
            ).ingest_document(document)

            assert "identifiable.pdf" in (result.reason or "")

    def test_parse_error_is_what_the_parser_raises(self, tmp_path: Path) -> None:
        from medguard.ingestion.parser import parse

        broken = tmp_path / "x.pdf"
        broken.write_bytes(b"nope")
        with pytest.raises(ParseError):
            parse(broken)
