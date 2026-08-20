"""Relational schema for the corpus and its chunks.

Scope in Phase 2 is the *schema*, not the data. The tables exist, migrations create
them, and nothing writes rows until ingestion lands in Phase 3. Defining them now is
what lets the migration history start from a real baseline instead of an empty one --
an empty baseline records that the database once had no tables, which is not useful
information.

Three decisions worth defending
-------------------------------
**Enums are stored as constrained strings, not native database enum types.**
``native_enum=False`` renders a ``VARCHAR`` plus a ``CHECK`` constraint on both SQLite
and PostgreSQL. Native PostgreSQL enums would give marginally tighter storage in
exchange for ``ALTER TYPE`` migrations that SQLite cannot express at all, so the two
backends would need divergent migration paths. Identical DDL on both is worth more.

**Every constraint is named, via a metadata naming convention.** This is not
cosmetic. SQLite has no real ``ALTER TABLE``, so Alembic runs migrations in batch
mode, which works by building a new table, copying rows, and dropping the old one.
That process has to refer to constraints by name. Unnamed constraints get
auto-generated names that differ between backends and between SQLAlchemy versions,
and a migration that cannot name a constraint cannot drop it.

**Chunks cascade from documents.** Re-ingesting a document deletes and rebuilds its
chunks. Without ``ON DELETE CASCADE`` the old rows survive as orphans that still
match retrieval queries, which silently corrupts every evaluation metric computed
afterwards.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from medguard.models import DocumentType, IngestionStatus, SourceOrganization

__all__ = ["Base", "ChunkRow", "DocumentRow", "make_chunk_id"]

#: Deterministic constraint names across backends. See the module docstring: batch
#: migrations on SQLite cannot drop a constraint they cannot name.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base for every MedGuard table.

    ``Base.metadata`` is what ``migrations/env.py`` compares the live database against,
    so autogenerate only sees tables whose modules have been imported. Importing
    :mod:`medguard.storage.database` pulls this module in, which keeps that from being
    a thing anyone has to remember.
    """

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def make_chunk_id(document_id: str, chunk_index: int) -> str:
    """Build the storage id for a chunk: ``{document_id}::{chunk_index:04d}``.

    Position-addressed, and now the *only* chunk-id scheme in the project. Phase 2
    carried a second, content-addressed one (``Chunk.make_id``, a blake2b digest over
    the text); Phase 3A deleted it, because two id schemes for one concept is how a
    vector store and a relational table end up keyed differently.

    Position won for a specific reason. A content-addressed id changes when the text
    changes, which makes an edited chunk a *new* chunk: re-ingesting a revised document
    inserts the new text and leaves the superseded text in the index, still retrievable,
    still citable. Position-addressed ids overwrite in place, so ingestion is idempotent
    in the sense that actually matters -- the index converges on the current document.

    Document *versions* stay distinguishable regardless, because ``document_id`` encodes
    the version (``icmr_amr_2022`` versus a hypothetical ``icmr_amr_2019``). Version
    identity belongs to the document, not to the chunk's id scheme.

    Zero-padded to four digits so ids sort lexicographically in the same order as
    ``chunk_index``, which makes a raw scan of the table readable. Qdrant cannot use
    this string directly -- point ids must be an unsigned int or a UUID -- so Phase 3B
    bridges it with ``uuid5(MEDGUARD_UUID_NAMESPACE, chunk_id)``. See
    :mod:`medguard.constants`.
    """
    return f"{document_id}::{chunk_index:04d}"


class DocumentRow(Base):
    """One source document, mirroring its manifest entry plus ingestion state.

    The manifest is the portable provenance record; this table is the working copy
    that ingestion mutates. They overlap on purpose: the manifest is committed and
    survives a dropped database, while this table carries the per-run state
    (``ingestion_status``, ``ingestion_error``) that has no business in a committed
    file.
    """

    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    #: Source-authored and canonical: this is the title the *publisher* states, read
    #: from a landing page, DOI record or catalogue entry, and recorded in
    #: ``data/sources.yaml`` by a human who checked it.
    #:
    #: GUARDRAIL FOR PHASE 3B: when the PDF parser lands, a title extracted from PDF
    #: metadata or from the document's first page must never overwrite this value. In
    #: government PDFs the ``/Title`` field is frequently the production filename
    #: ("rr7103a1-H", "final_draft_v3"), so treating extraction as authoritative would
    #: replace verified titles with build artefacts -- and the citations shown to a
    #: clinician are rendered from this column. An extracted title may be recorded
    #: elsewhere for comparison, or used to *populate* a title that is somehow absent;
    #: it may not replace one that came from the publisher.
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    source_organization: Mapped[SourceOrganization] = mapped_column(
        SAEnum(SourceOrganization, native_enum=False, validate_strings=True, name="sourceorg"),
        nullable=False,
    )
    document_type: Mapped[DocumentType] = mapped_column(
        SAEnum(DocumentType, native_enum=False, validate_strings=True, name="documenttype"),
        nullable=False,
    )
    publication_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    local_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    #: What the *manifest* says this file is. Owned by the corpus commands, which keep
    #: it in step with the bytes on disk.
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    #: What ingestion last built chunks from. Owned by the pipeline, and deliberately a
    #: separate column from ``sha256``: that one tracks the file, this one tracks the
    #: index, and the whole point is that they can disagree. When a publisher revises a
    #: guideline in place, ``corpus download`` moves ``sha256`` forward; until a
    #: re-ingestion moves this one, the index is serving the superseded edition. Null
    #: means "never successfully indexed", never "unchanged".
    indexed_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    #: Null until Phase 3 parses the PDF. Null means "not counted yet", never "zero".
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ingestion_status: Mapped[IngestionStatus] = mapped_column(
        SAEnum(IngestionStatus, native_enum=False, validate_strings=True, name="ingestionstatus"),
        nullable=False,
        default=IngestionStatus.PENDING,
        server_default=IngestionStatus.PENDING.value,
    )
    ingestion_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    version_or_revision: Mapped[str | None] = mapped_column(String(255), nullable=True)
    license_or_usage_note: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    chunks: Mapped[list[ChunkRow]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        # Unique rather than merely indexed: two ids pointing at identical bytes would
        # double-count one source in every retrieval metric.
        UniqueConstraint("sha256", name="uq_documents_sha256"),
        Index("ix_documents_sha256", "sha256"),
        Index("ix_documents_ingestion_status", "ingestion_status"),
        CheckConstraint("file_size_bytes >= 0", name="file_size_non_negative"),
        CheckConstraint("length(sha256) = 64", name="sha256_length"),
        CheckConstraint(
            "indexed_sha256 IS NULL OR length(indexed_sha256) = 64",
            name="indexed_sha256_length",
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<DocumentRow {self.document_id!r} status={self.ingestion_status}>"


class ChunkRow(Base):
    """One retrievable span of a document.

    Empty in Phase 2. The table is created now so the baseline migration describes the
    schema the system is actually being built toward, and so Phase 3 adds rows rather
    than simultaneously inventing a schema and debugging a chunker.
    """

    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    #: A chunk is a span of text, and a span crosses page boundaries constantly -- the
    #: single `page` column this replaces could only record where a chunk started, so a
    #: citation into a chunk spanning pages 41-43 pointed a reader at page 41 and left
    #: them to find the rest. Both null means "page not known" (a parser that could not
    #: attribute the span), never "page zero".
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section: Mapped[str | None] = mapped_column(Text, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    #: Null until a tokeniser exists; counting tokens needs the embedding model.
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: Named `chunk_metadata`, not `metadata`: `Base.metadata` is SQLAlchemy's own
    #: attribute and declaring a column of that name shadows it.
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document: Mapped[DocumentRow] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        # A document cannot have two chunk 4s. Without this, a partially-failed
        # re-ingestion silently duplicates spans and inflates recall.
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_id_chunk_index"),
        CheckConstraint("chunk_index >= 0", name="chunk_index_non_negative"),
        CheckConstraint("char_count >= 0", name="char_count_non_negative"),
        # Both-null is legal (page attribution failed); one-null is legal (a chunk that
        # starts on a known page and runs off the end of what the parser could map).
        # An inverted range is not, and it is the shape a citation renderer would print
        # as "pages 43-41" without noticing anything was wrong.
        CheckConstraint(
            "page_start IS NULL OR page_end IS NULL OR page_end >= page_start",
            name="page_range_ordered",
        ),
        # Pages are 1-based because that is how every reader, every printed index and
        # every clinician counts them. PyMuPDF counts from 0, so "page 0" is the exact
        # shape an unconverted parser index takes -- an off-by-one that produces a
        # citation to a page that does not exist and is otherwise completely silent.
        # `NULL` still means "not known": a CHECK passes on NULL, but the null branches
        # are written out rather than relied on, because the reader of a constraint
        # should not have to recall SQL's three-valued logic to know what it permits.
        CheckConstraint(
            "(page_start IS NULL OR page_start >= 1) AND (page_end IS NULL OR page_end >= 1)",
            name="page_numbers_positive",
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ChunkRow {self.chunk_id!r} chars={self.char_count}>"
