"""The ingestion orchestrator: manifest to indexed chunks, one document at a time.

This is the only module in the ingestion package that performs I/O or holds state. The
parser, cleaner and chunker are pure functions; everything that can be retried, rolled
back or left half-finished lives here, on purpose. That separation is what lets the
interesting logic be tested offline and the interesting failures be tested with fakes.

Failure isolation is the design constraint
------------------------------------------
Twenty documents are ingested in one run. Document 7 is a scanned PDF with no text
layer. The requirement is that documents 8 through 20 are unaffected, that 7 is left in
a state that says *what* went wrong, and that nothing partial is left behind in either
store. So each document is its own transaction and its own error boundary, and the
summary at the end counts failures rather than hiding them.

The skip predicate, and why it has three clauses
------------------------------------------------
Re-running ingestion on an unchanged corpus must do no work. Deciding that requires
three things to agree, and each one catches a different real situation:

* ``status == indexed`` -- the document finished last time. Cheap, and wrong on its own,
  because it believes a database that may be describing a Qdrant collection that no
  longer exists.
* the file's hash matches ``indexed_sha256`` -- the bytes on disk are the ones the
  current chunks were built from. Catches a publisher revising a guideline in place,
  which is otherwise completely silent. Note *which* column: ``documents.sha256`` tracks
  the manifest and is updated by ``corpus download``, so comparing against it would
  compare the file to itself and skip every revised document forever. That is a bug this
  pipeline had and this column exists to prevent.
* ``qdrant count == sql chunk count`` -- the index actually holds what the database
  says. Catches the collection being dropped out of band, a half-finished previous run,
  and a partial delete.

Dropping any one of them produces a system that reports success over an index that
cannot answer anything, which is the worst available outcome because it looks fine.

Status vocabulary
-----------------
The eight conceptual stages map onto the six values ``IngestionStatus`` already has.
``PARSING`` and ``PARSED`` are not two states worth persisting -- one is a moment, the
other is a fact -- and adding transient values to the enum would mean a crashed process
leaves rows in states that no longer mean anything after restart. Transient stages exist
in the log and the progress bar; the database records only what is durable.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import delete, select

from medguard.config import IngestionSettings, get_settings
from medguard.corpus.manifest import load_manifest, sha256_file
from medguard.errors import IngestionError, MedGuardError
from medguard.ingestion.chunker import chunk_document
from medguard.ingestion.cleaner import clean_document
from medguard.ingestion.parser import parse
from medguard.logging import get_logger
from medguard.models import Chunk, IngestionStatus
from medguard.storage.schema import ChunkRow, DocumentRow
from medguard.storage.vector_store import ChunkWithVectors, VectorStore

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from medguard.embeddings.base import DenseEmbedder, SparseEmbedder

__all__ = [
    "DocumentIngestionResult",
    "IngestionOutcome",
    "IngestionPipeline",
    "IngestionSummary",
]

log = get_logger(__name__)

#: Below this much available memory, embedding batches are halved. Chosen against the
#: measured peak transient allocation of BGE-small at batch 8 (about 100 MB) plus the
#: ONNX session itself, with room for the rest of the process.
RAM_WARN_MB = 300

#: Below this, the document is abandoned rather than attempted. A process killed by the
#: OS mid-batch leaves a half-populated collection and a row claiming to be mid-flight;
#: stopping deliberately leaves a failed row and an index that is still consistent.
RAM_ABORT_MB = 150

#: Batches never shrink below this. At a batch of one the per-call overhead dominates
#: and a document that was going to fail on memory fails slowly instead of quickly.
MIN_BATCH = 2


class IngestionOutcome(StrEnum):
    """What happened to one document in one run.

    Distinct from :class:`~medguard.models.IngestionStatus`, which is the *persisted*
    state of a document. A document can end a run as ``SKIPPED`` while its stored status
    is ``INDEXED``; conflating the two would make "was there work to do?" unanswerable
    after the fact.
    """

    INGESTED = "ingested"
    SKIPPED = "skipped"
    FAILED = "failed"


class DocumentIngestionResult(BaseModel):
    """The outcome of one document, whether or not anything was written."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    outcome: IngestionOutcome
    chunk_count: int = Field(default=0, ge=0)
    page_count: int | None = Field(default=None, ge=0)
    total_chars: int = Field(default=0, ge=0)
    reason: str | None = Field(
        default=None, description="Why it was skipped, or how it failed. Never a secret."
    )
    duration_s: float = Field(default=0.0, ge=0)

    @property
    def ok(self) -> bool:
        return self.outcome is not IngestionOutcome.FAILED


class IngestionSummary(BaseModel):
    """Everything one ``ingest`` run did."""

    results: list[DocumentIngestionResult] = Field(default_factory=list)
    duration_s: float = Field(default=0.0, ge=0)
    dry_run: bool = False

    @property
    def ingested(self) -> list[DocumentIngestionResult]:
        return [r for r in self.results if r.outcome is IngestionOutcome.INGESTED]

    @property
    def skipped(self) -> list[DocumentIngestionResult]:
        return [r for r in self.results if r.outcome is IngestionOutcome.SKIPPED]

    @property
    def failed(self) -> list[DocumentIngestionResult]:
        return [r for r in self.results if r.outcome is IngestionOutcome.FAILED]

    @property
    def total_chunks(self) -> int:
        return sum(r.chunk_count for r in self.results)

    def exit_code(self) -> int:
        """Non-zero if any document failed. All-skipped is success, not a warning."""
        return 1 if self.failed else 0


class IngestionPipeline:
    """Drives documents from ``pending`` to ``indexed``, isolating failures."""

    def __init__(
        self,
        session: Session,
        vector_store: VectorStore,
        dense_embedder: DenseEmbedder,
        sparse_embedder: SparseEmbedder,
        settings: IngestionSettings,
        *,
        raw_dir: Path | None = None,
    ) -> None:
        self._session = session
        self._store = vector_store
        self._dense = dense_embedder
        self._sparse = sparse_embedder
        self._settings = settings
        self._raw_dir = raw_dir if raw_dir is not None else get_settings().corpus.raw_dir

    # -- discovery -----------------------------------------------------------

    def sync_documents_from_manifest(self, manifest_path: Path) -> int:
        """Create ``pending`` rows for manifest entries the database has not seen.

        One-directional on purpose. New entries are added; existing rows are updated only
        where the manifest is authoritative and the database is not -- title, licence,
        publisher, and the hash and size that describe the file. Ingestion state is never
        touched here, because the manifest has no opinion about whether a document has
        been indexed.

        Rows are never deleted. A document removed from the manifest keeps its chunks
        until someone removes it deliberately; silently dropping indexed content because
        a YAML edit removed a line is not a decision this function should make.
        """
        manifest = load_manifest(manifest_path)
        existing = {
            row.document_id for row in self._session.execute(select(DocumentRow.document_id)).all()
        }

        added = 0
        for entry in manifest.documents:
            if entry.document_id in existing:
                self._refresh_provenance(entry.document_id, entry)
                continue

            self._session.add(
                DocumentRow(
                    document_id=entry.document_id,
                    title=entry.title,
                    source_url=entry.source_url,
                    source_organization=entry.source_organization,
                    document_type=entry.document_type,
                    publication_date=entry.publication_date,
                    local_filename=entry.local_filename,
                    sha256=entry.sha256,
                    file_size_bytes=entry.file_size_bytes,
                    page_count=entry.page_count,
                    retrieved_at=entry.retrieved_at,
                    ingestion_status=IngestionStatus.PENDING,
                    version_or_revision=entry.version_or_revision,
                    license_or_usage_note=entry.license_or_usage_note,
                )
            )
            added += 1

        self._session.commit()
        log.info("ingest.manifest_synced", added=added, total=len(manifest.documents))
        return added

    def _refresh_provenance(self, document_id: str, entry: object) -> None:
        """Update the fields the manifest owns, leaving ingestion state alone."""
        row = self._session.get(DocumentRow, document_id)
        if row is None:  # pragma: no cover - the caller just read the id from the table
            return
        # `indexed_sha256` is deliberately absent from this list, and so is every
        # ingestion-state field. The manifest is authoritative about the *file*; it has
        # no opinion about what has been indexed, and letting it write that column would
        # erase the only evidence that a re-downloaded document still needs re-ingesting.
        for field in (
            "title",
            "source_url",
            "publication_date",
            "sha256",
            "file_size_bytes",
            "version_or_revision",
            "license_or_usage_note",
        ):
            setattr(row, field, getattr(entry, field))

    # -- the run -------------------------------------------------------------

    def ingest_all(
        self,
        *,
        only: str | None = None,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> IngestionSummary:
        """Ingest every eligible document, continuing past any that fail."""
        started = time.perf_counter()

        statement = select(DocumentRow).order_by(DocumentRow.document_id)
        if only:
            statement = statement.where(DocumentRow.document_id == only)
        documents = list(self._session.scalars(statement).all())
        if limit is not None:
            documents = documents[:limit]

        results = [self.ingest_document(document, dry_run=dry_run) for document in documents]

        summary = IngestionSummary(
            results=results,
            duration_s=time.perf_counter() - started,
            dry_run=dry_run,
        )
        log.info(
            "ingest.run_complete",
            ingested=len(summary.ingested),
            skipped=len(summary.skipped),
            failed=len(summary.failed),
            chunks=summary.total_chunks,
            seconds=round(summary.duration_s, 2),
            dry_run=dry_run,
        )
        return summary

    def ingest_document(
        self, document: DocumentRow, *, dry_run: bool = False
    ) -> DocumentIngestionResult:
        """Ingest one document. Never raises for a per-document failure.

        The broad ``except`` is the point of this method rather than an oversight: a
        failure here must become a recorded outcome, not an exception that ends the run
        and leaves nineteen other documents unprocessed. Anything not derived from
        :class:`MedGuardError` is still recorded, because an unanticipated bug in the
        parser is exactly as fatal to *this* document and exactly as harmless to the
        others.
        """
        started = time.perf_counter()
        document_id = document.document_id

        try:
            return self._ingest(document, dry_run=dry_run, started=started)
        except MedGuardError as exc:
            return self._fail(document, exc.detail, started, dry_run=dry_run)
        except Exception as exc:
            log.exception("ingest.unexpected_failure", document_id=document_id)
            return self._fail(
                document, f"unexpected {type(exc).__name__}: {exc}", started, dry_run=dry_run
            )

    # -- the per-document stages --------------------------------------------

    def _ingest(
        self, document: DocumentRow, *, dry_run: bool, started: float
    ) -> DocumentIngestionResult:
        document_id = document.document_id
        path = self._raw_dir / document.local_filename

        # 1. Trust boundary. Everything downstream assumes the bytes are the bytes the
        #    manifest vouched for; if they are not, no amount of careful chunking helps.
        file_hash = self._verify_hash(document, path)

        # 2. Nothing to do?
        skip_reason = self._skip_reason(document, file_hash)
        if skip_reason and not dry_run:
            log.info("ingest.skipped", document_id=document_id, reason=skip_reason)
            return DocumentIngestionResult(
                document_id=document_id,
                outcome=IngestionOutcome.SKIPPED,
                chunk_count=self._sql_chunk_count(document_id),
                page_count=document.page_count,
                reason=skip_reason,
                duration_s=time.perf_counter() - started,
            )

        # 3. Can the index be rebuilt from what is already in SQL?
        if not dry_run and self._can_reindex_without_parsing(document, file_hash):
            return self._reindex_from_chunks(document, file_hash, started)

        # 4. Parse.
        parsed = parse(path)
        pages = list(parsed.pages())
        log.debug("ingest.parsed", document_id=document_id, pages=parsed.page_count)

        # 4. Clean and chunk. Both pure; neither writes anything.
        cleaned = clean_document(pages, self._settings)
        chunks = chunk_document(document_id, cleaned, self._settings)
        total_chars = sum(len(c.text) for c in chunks)

        if dry_run:
            log.info(
                "ingest.dry_run_complete",
                document_id=document_id,
                pages=parsed.page_count,
                chunks=len(chunks),
                chars=total_chars,
            )
            return DocumentIngestionResult(
                document_id=document_id,
                outcome=IngestionOutcome.INGESTED,
                chunk_count=len(chunks),
                page_count=parsed.page_count,
                total_chars=total_chars,
                reason="dry run: parsed and chunked, nothing written",
                duration_s=time.perf_counter() - started,
            )

        document.page_count = parsed.page_count
        self._set_status(document, IngestionStatus.PARSED)

        self._persist_chunks(document_id, chunks)
        self._set_status(document, IngestionStatus.CHUNKED)

        # 5 and 6. Embed and index, in batches, under memory supervision.
        indexed = self._embed_and_index(document, chunks)

        # Written only here, only on success. Setting it any earlier would claim the
        # index holds these bytes while a failure could still leave it holding the
        # previous ones.
        document.indexed_sha256 = file_hash
        self._set_status(document, IngestionStatus.INDEXED, error=None)
        log.info(
            "ingest.indexed",
            document_id=document_id,
            chunks=indexed,
            pages=parsed.page_count,
        )
        return DocumentIngestionResult(
            document_id=document_id,
            outcome=IngestionOutcome.INGESTED,
            chunk_count=indexed,
            page_count=parsed.page_count,
            total_chars=total_chars,
            duration_s=time.perf_counter() - started,
        )

    def _verify_hash(self, document: DocumentRow, path: Path) -> str:
        """Confirm the file on disk is the one the manifest recorded, and return its hash."""
        if not path.is_file():
            raise IngestionError(
                f"{document.document_id}: {path.name} is not on disk. "
                "Run `medguard corpus download` first."
            )

        actual = sha256_file(path)
        if actual != document.sha256:
            raise IngestionError(
                f"{document.document_id}: {path.name} does not match the manifest "
                f"(recorded {document.sha256[:12]}…, found {actual[:12]}…). The file "
                "changed after it was recorded; re-run `medguard corpus download` and "
                "`medguard corpus verify` before ingesting."
            )
        return actual

    def _skip_reason(self, document: DocumentRow, file_hash: str) -> str | None:
        """``None`` when there is work to do; otherwise why there is not.

        All three clauses are evaluated on every run. Checking only the cheap one is how
        a system ends up reporting "20 skipped, unchanged" over an empty collection.
        """
        if document.ingestion_status is not IngestionStatus.INDEXED:
            return None

        # The hash the *index* was built from, not the one the manifest vouches for.
        # Those two are equal in the steady state and differ for exactly as long as a
        # re-download has happened and a re-ingestion has not -- which is the case this
        # clause exists to catch. Comparing against `document.sha256` here would compare
        # the file to itself and skip every revised document forever.
        if document.indexed_sha256 != file_hash:
            log.info(
                "ingest.source_changed",
                document_id=document.document_id,
                indexed=(document.indexed_sha256 or "never")[:12],
                on_disk=file_hash[:12],
            )
            return None

        sql_chunks = self._sql_chunk_count(document.document_id)
        if sql_chunks == 0:
            return None

        indexed_points = self._store.count_for_document(document.document_id)
        if indexed_points != sql_chunks:
            log.info(
                "ingest.index_drifted",
                document_id=document.document_id,
                sql_chunks=sql_chunks,
                qdrant_points=indexed_points,
            )
            return None

        return "unchanged"

    def _can_reindex_without_parsing(self, document: DocumentRow, file_hash: str) -> bool:
        """True when the chunks in SQL are still valid and only Qdrant needs rebuilding.

        The case: someone dropped the collection, or a previous run died between writing
        chunk rows and finishing the upserts. The file has not changed, so the chunks
        derived from it are exactly what a re-parse would produce again -- for a 148-page
        guideline that is minutes of PyMuPDF work to arrive at a byte-identical answer.

        Chunk rows are the durable artefact and the vector store is derived from them.
        Treating it that way is what makes recovering from a lost collection cheap.
        """
        if document.indexed_sha256 != file_hash:
            return False
        return self._sql_chunk_count(document.document_id) > 0

    def _reindex_from_chunks(
        self, document: DocumentRow, file_hash: str, started: float
    ) -> DocumentIngestionResult:
        """Re-embed and re-upsert this document's stored chunks. No parsing."""
        document_id = document.document_id
        rows = self._session.scalars(
            select(ChunkRow)
            .where(ChunkRow.document_id == document_id)
            .order_by(ChunkRow.chunk_index)
        ).all()

        chunks = [
            Chunk(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                text=row.text,
                chunk_index=row.chunk_index,
                page_start=row.page_start,
                page_end=row.page_end,
                section=row.section,
                token_count=row.token_count,
            )
            for row in rows
        ]

        log.info("ingest.reindexing_from_chunks", document_id=document_id, chunks=len(chunks))
        indexed = self._embed_and_index(document, chunks)

        document.indexed_sha256 = file_hash
        self._set_status(document, IngestionStatus.INDEXED, error=None)
        return DocumentIngestionResult(
            document_id=document_id,
            outcome=IngestionOutcome.INGESTED,
            chunk_count=indexed,
            page_count=document.page_count,
            total_chars=sum(len(c.text) for c in chunks),
            reason="re-embedded from stored chunks; the source was unchanged",
            duration_s=time.perf_counter() - started,
        )

    def _sql_chunk_count(self, document_id: str) -> int:
        rows = self._session.scalars(
            select(ChunkRow.chunk_id).where(ChunkRow.document_id == document_id)
        ).all()
        return len(rows)

    def _persist_chunks(self, document_id: str, chunks: Sequence[Chunk]) -> None:
        """Replace this document's chunk rows wholesale.

        Delete-then-insert rather than upsert, because a revision that produces *fewer*
        chunks would otherwise leave the tail of the previous revision in the table.
        Those rows are superseded text that still matches retrieval, which is the exact
        failure the corpus rules exist to prevent.
        """
        self._session.execute(delete(ChunkRow).where(ChunkRow.document_id == document_id))
        self._session.add_all(
            ChunkRow(
                chunk_id=chunk.chunk_id,
                document_id=document_id,
                chunk_index=chunk.chunk_index,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section=chunk.section,
                text=chunk.text,
                char_count=len(chunk.text),
                token_count=chunk.token_count,
                chunk_metadata={},
            )
            for chunk in chunks
        )
        self._session.commit()

    def _embed_and_index(self, document: DocumentRow, chunks: Sequence[Chunk]) -> int:
        """Embed in batches and write to Qdrant, replacing whatever was there.

        The old points go first. Upserting over them would leave any chunk the new
        revision no longer produces sitting in the collection, still retrievable, with
        no row in SQL to notice it by.
        """
        document_id = document.document_id
        self._store.delete_by_document(document_id)

        if not chunks:
            self._set_status(document, IngestionStatus.EMBEDDED)
            return 0

        batch_size = self._settings.batch_size
        written = 0

        for batch in self._batches(chunks, batch_size):
            batch_size = self._adjusted_batch_size(batch_size, document_id)
            texts = [chunk.text for chunk in batch]

            dense = self._dense.embed_documents(texts)
            sparse = self._sparse.embed_documents(texts)
            if len(dense) != len(batch) or len(sparse) != len(batch):
                raise IngestionError(
                    f"{document_id}: embedder returned {len(dense)} dense and "
                    f"{len(sparse)} sparse vectors for {len(batch)} chunks"
                )

            self._store.upsert_chunks(
                [
                    ChunkWithVectors(
                        chunk_id=chunk.chunk_id,
                        document_id=document_id,
                        chunk_index=chunk.chunk_index,
                        text=chunk.text,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        section=chunk.section,
                        title=document.title,
                        source_organization=str(document.source_organization),
                        document_type=str(document.document_type),
                        publication_date=_as_rfc3339(document.publication_date),
                        dense=dense_vector,
                        sparse=sparse_vector,
                    )
                    for chunk, dense_vector, sparse_vector in zip(batch, dense, sparse, strict=True)
                ]
            )
            written += len(batch)

        self._set_status(document, IngestionStatus.EMBEDDED)

        # The index has to agree with what was just written. A silent shortfall here is
        # a collection that answers queries with part of a document.
        stored = self._store.count_for_document(document_id)
        if stored != len(chunks):
            raise IngestionError(
                f"{document_id}: wrote {len(chunks)} chunks but the collection holds "
                f"{stored} points for it"
            )
        return written

    def _batches(self, chunks: Sequence[Chunk], size: int) -> Iterator[list[Chunk]]:
        for start in range(0, len(chunks), size):
            yield list(chunks[start : start + size])

    def _adjusted_batch_size(self, current: int, document_id: str) -> int:
        """Shrink the batch under memory pressure; abandon the document under duress.

        Being killed by the OS mid-batch is the outcome this exists to avoid. It leaves
        a half-populated collection, a row claiming a state it is not in, and no error
        anywhere -- the process simply stops. Failing deliberately leaves a failed row
        and a consistent index.
        """
        from medguard.diagnostics import available_ram_mb

        available = available_ram_mb()
        if available < RAM_ABORT_MB:
            raise IngestionError(
                f"{document_id}: only {available:.0f} MB of memory available "
                f"(floor is {RAM_ABORT_MB} MB). Close applications and re-run, or use "
                "`--dry-run` to exercise parsing and chunking without loading a model."
            )
        if available < RAM_WARN_MB and current > MIN_BATCH:
            reduced = max(MIN_BATCH, current // 2)
            log.warning(
                "ingest.batch_reduced",
                document_id=document_id,
                available_mb=round(available),
                was=current,
                now=reduced,
            )
            return reduced
        return current

    # -- state ---------------------------------------------------------------

    def _set_status(
        self, document: DocumentRow, status: IngestionStatus, *, error: str | None = None
    ) -> None:
        document.ingestion_status = status
        document.ingestion_error = error
        self._session.commit()

    def _fail(
        self, document: DocumentRow, reason: str, started: float, *, dry_run: bool
    ) -> DocumentIngestionResult:
        """Record a failure and leave both stores consistent.

        Chunk rows are deliberately *not* deleted. They are the expensive artefact and
        the honest one -- if parsing and chunking succeeded and embedding failed, the
        chunks are correct and re-running can reuse them. Qdrant points are deleted,
        because a partially-indexed document is one that answers queries with part of
        itself.
        """
        document_id = document.document_id
        log.warning("ingest.failed", document_id=document_id, reason=reason)

        if not dry_run:
            # Three separate blocks, not one. Each step is independently useful and
            # independently able to fail, and grouping them means a Qdrant timeout during
            # cleanup would also skip recording *why* the document failed -- leaving a row
            # stuck in a mid-flight state with no explanation, which is the single most
            # confusing thing this method could produce.
            with _suppressed("ingest.rollback_failed", document_id=document_id):
                self._session.rollback()

            with _suppressed("ingest.point_cleanup_failed", document_id=document_id):
                self._store.delete_by_document(document_id)

            with _suppressed("ingest.status_write_failed", document_id=document_id):
                # Re-read after the rollback rather than reusing `document`: the rollback
                # expired the identity map, and if this document's row was itself part of
                # the rolled-back transaction it no longer exists. `None` then means there
                # is nothing to record against, which is correct rather than an error.
                row = self._session.get(DocumentRow, document_id)
                if row is not None:
                    self._set_status(row, IngestionStatus.FAILED, error=reason)

        return DocumentIngestionResult(
            document_id=document_id,
            outcome=IngestionOutcome.FAILED,
            reason=reason,
            duration_s=time.perf_counter() - started,
        )


@contextmanager
def _suppressed(event: str, **context: object) -> Iterator[None]:
    """Swallow and log a failure in cleanup code.

    Cleanup runs while something has already gone wrong. An exception raised here would
    replace the real reason with a less useful one, and would stop the remaining
    documents from being attempted.
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - cleanup must not mask the original failure
        log.warning(event, error=f"{type(exc).__name__}: {exc}", **context)


def _as_rfc3339(value: object) -> str | None:
    """Render a date for Qdrant's datetime payload index, or ``None``.

    ``None`` stays ``None``. Nine documents in the shipped corpus state no publication
    date, and substituting a placeholder would make them sort and filter as though the
    publisher had said something it did not.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return f"{value}T00:00:00Z"
