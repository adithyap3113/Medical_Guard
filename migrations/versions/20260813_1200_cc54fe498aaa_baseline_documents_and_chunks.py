"""baseline: documents and chunks

The first migration, and deliberately not an empty one. An empty baseline records that
the database once had no tables, which is not information anyone needs; starting from
the real schema means `alembic upgrade head` on a fresh clone produces a working
database in one step.

Both tables are created inside `batch_alter_table` blocks for their indexes, because
`render_as_batch=True` is set in migrations/env.py. On PostgreSQL this compiles to
ordinary DDL; on SQLite it is what makes future column changes possible at all.

Revision ID: cc54fe498aaa
Revises:
Create Date: 2026-08-13 12:00:01.382317
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "cc54fe498aaa"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("document_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column(
            "source_organization",
            sa.Enum(
                "CDC", "NIH", "NICE", "WHO", "PMC", "ICMR", name="sourceorg", native_enum=False
            ),
            nullable=False,
        ),
        sa.Column(
            "document_type",
            sa.Enum(
                "GUIDELINE",
                "SYSTEMATIC_REVIEW",
                "REVIEW",
                "PROTOCOL",
                "REFERENCE",
                name="documenttype",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("publication_date", sa.Date(), nullable=True),
        sa.Column("local_filename", sa.String(length=255), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "ingestion_status",
            sa.Enum(
                "PENDING",
                "PARSED",
                "CHUNKED",
                "EMBEDDED",
                "INDEXED",
                "FAILED",
                name="ingestionstatus",
                native_enum=False,
            ),
            server_default="pending",
            nullable=False,
        ),
        sa.Column("ingestion_error", sa.Text(), nullable=True),
        sa.Column("version_or_revision", sa.String(length=255), nullable=True),
        sa.Column("license_or_usage_note", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "file_size_bytes >= 0", name=op.f("ck_documents_file_size_non_negative")
        ),
        sa.CheckConstraint("length(sha256) = 64", name=op.f("ck_documents_sha256_length")),
        sa.PrimaryKeyConstraint("document_id", name=op.f("pk_documents")),
        sa.UniqueConstraint("sha256", name="uq_documents_sha256"),
    )
    with op.batch_alter_table("documents", schema=None) as batch_op:
        batch_op.create_index("ix_documents_ingestion_status", ["ingestion_status"], unique=False)
        batch_op.create_index("ix_documents_sha256", ["sha256"], unique=False)

    op.create_table(
        "chunks",
        sa.Column("chunk_id", sa.String(length=128), nullable=False),
        sa.Column("document_id", sa.String(length=64), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("page", sa.Integer(), nullable=True),
        sa.Column("section", sa.Text(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("chunk_metadata", sa.JSON(), server_default="{}", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.CheckConstraint("char_count >= 0", name=op.f("ck_chunks_char_count_non_negative")),
        sa.CheckConstraint("chunk_index >= 0", name=op.f("ck_chunks_chunk_index_non_negative")),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.document_id"],
            name=op.f("fk_chunks_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("chunk_id", name=op.f("pk_chunks")),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_id_chunk_index"),
    )
    with op.batch_alter_table("chunks", schema=None) as batch_op:
        batch_op.create_index("ix_chunks_document_id", ["document_id"], unique=False)


def downgrade() -> None:
    # Chunks first: they hold the foreign key into documents.
    with op.batch_alter_table("chunks", schema=None) as batch_op:
        batch_op.drop_index("ix_chunks_document_id")

    op.drop_table("chunks")
    with op.batch_alter_table("documents", schema=None) as batch_op:
        batch_op.drop_index("ix_documents_sha256")
        batch_op.drop_index("ix_documents_ingestion_status")

    op.drop_table("documents")
