"""widen chunk page metadata to a page range

A chunk is a span of text, and a span crosses page boundaries constantly. The baseline
schema gave it a single `page` column, which could only record where the span *started*:
a citation into a chunk covering pages 41-43 sent a clinician to page 41 and left them
to work out that the rest was overleaf. `page_start` / `page_end` records what was
always true about the data.

The check constraint permits both-null (the parser could not attribute the span) and
one-null (a chunk that starts on a known page and runs past what could be mapped). It
rejects only an inverted range, which is the shape a citation renderer would happily
print as "pages 43-41".

Data is carried in both directions rather than dropped, even though the `chunks` table
is empty at the time of writing. A migration that only works on an empty table is a
migration that will be run once and then be wrong. Downgrading is lossy in an honest
way: a range collapses to its first page, because the old column cannot hold more.

Revision ID: 125928f4b940
Revises: cc54fe498aaa
Create Date: 2026-08-13 20:57:53.504009
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "125928f4b940"
down_revision: str | None = "cc54fe498aaa"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_PAGE_RANGE_ORDERED = "page_start IS NULL OR page_end IS NULL OR page_end >= page_start"


def upgrade() -> None:
    # Three steps, not one: the copy has to happen while both the old and the new
    # columns exist, and the constraint can only be added once the copied data already
    # satisfies it.
    with op.batch_alter_table("chunks", schema=None) as batch_op:
        batch_op.add_column(sa.Column("page_start", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("page_end", sa.Integer(), nullable=True))

    # A single recorded page becomes a one-page range, which is what it always meant.
    op.execute("UPDATE chunks SET page_start = page, page_end = page WHERE page IS NOT NULL")

    with op.batch_alter_table("chunks", schema=None) as batch_op:
        batch_op.create_check_constraint(
            batch_op.f("ck_chunks_page_range_ordered"), _PAGE_RANGE_ORDERED
        )
        batch_op.drop_column("page")


def downgrade() -> None:
    with op.batch_alter_table("chunks", schema=None) as batch_op:
        batch_op.add_column(sa.Column("page", sa.INTEGER(), nullable=True))

    # Lossy by necessity: `page_end` has nowhere to go. Keeping the first page is
    # strictly better than keeping nothing, and it matches what the column meant before.
    op.execute("UPDATE chunks SET page = page_start WHERE page_start IS NOT NULL")

    with op.batch_alter_table("chunks", schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f("ck_chunks_page_range_ordered"), type_="check")
        batch_op.drop_column("page_end")
        batch_op.drop_column("page_start")
