"""record the hash each document was indexed from

`documents.sha256` is provenance: it is the hash the *manifest* vouches for, and
`corpus download` / `corpus sync` keep it current with the file on disk. That makes it
the wrong column to answer the question ingestion actually needs to ask, which is "have
the bytes changed since we last built chunks from them?"

Concretely, without this column a revised document is never re-indexed. A publisher
updates a guideline in place, `corpus download` fetches the new bytes and records the new
hash, `medguard ingest` syncs that hash onto the row -- and the skip predicate then sees
a document whose status is `indexed`, whose recorded hash matches the file, and whose
chunk and point counts agree. It skips. The index keeps serving the superseded edition,
with a citation that points at the current one, and nothing anywhere reports a problem.

Two columns, two meanings, both nullable-by-design:

* ``sha256`` -- what the manifest says the file is. Owned by the corpus commands.
* ``indexed_sha256`` -- what ingestion last built chunks from. Owned by the pipeline.
  Null means "never successfully indexed", which is exactly true for every existing row
  at the moment this migration runs.

They are equal in the steady state. They differ for precisely as long as a re-download
has happened and a re-ingestion has not, which is the window this column exists to make
visible.

Revision ID: 3c81ad55e4f2
Revises: b6f2d40c9a71
Create Date: 2026-08-18 17:10:22.114903
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "3c81ad55e4f2"
down_revision: str | None = "b6f2d40c9a71"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("documents", schema=None) as batch_op:
        batch_op.add_column(sa.Column("indexed_sha256", sa.String(length=64), nullable=True))

    # Existing rows are *not* backfilled from `sha256`, deliberately. Copying it would
    # assert that every document currently in the table was indexed from the bytes now
    # on disk, which is the exact claim this column exists to stop the system making
    # without evidence. Null means "unknown", the skip predicate treats unknown as "do
    # the work", and one extra ingestion pass is a much cheaper mistake than a silently
    # stale index.
    with op.batch_alter_table("documents", schema=None) as batch_op:
        batch_op.create_check_constraint(
            batch_op.f("ck_documents_indexed_sha256_length"),
            "indexed_sha256 IS NULL OR length(indexed_sha256) = 64",
        )


def downgrade() -> None:
    with op.batch_alter_table("documents", schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f("ck_documents_indexed_sha256_length"), type_="check")
        batch_op.drop_column("indexed_sha256")
