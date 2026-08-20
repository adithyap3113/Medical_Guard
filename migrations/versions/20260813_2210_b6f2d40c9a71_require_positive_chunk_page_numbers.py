"""require positive chunk page numbers

The page range added in `125928f4b940` constrained the two columns *relative to each
other* -- `page_end >= page_start` -- and said nothing about their absolute value. That
leaves `page_start = 0` a legal row, and page zero is not a page: every reader, printed
index, table of contents and clinician counts from one.

It is also the exact shape of a specific bug. PyMuPDF, the parser landing in this phase,
indexes pages from 0. An extraction loop that forgets to convert produces a chunk whose
recorded page is one less than the page a clinician would turn to, and the last page of
every document becomes unreachable. Nothing about that fails loudly: the citation
renders, the number looks plausible, and it points at the wrong page -- or at a page
that does not exist. A constraint turns it into an `IntegrityError` at the moment of
the first bad insert, which is the only cheap time to find it.

Standalone on purpose. Bundling it into an ingestion-related migration would tie a
schema guarantee to a feature release, and the guarantee is what makes the feature safe
to write.

`NULL` remains legal in both columns and still means "the parser could not attribute
this span", never "page zero". A `CHECK` passes when its expression evaluates to `NULL`,
so `page_start >= 1` alone would already permit it, but the null branches are written
out explicitly: a reader should not have to recall SQL's three-valued logic to know what
a constraint allows.

Revision ID: b6f2d40c9a71
Revises: 125928f4b940
Create Date: 2026-08-13 22:10:41.882357
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "b6f2d40c9a71"
down_revision: str | None = "125928f4b940"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_PAGE_NUMBERS_POSITIVE = (
    "(page_start IS NULL OR page_start >= 1) AND (page_end IS NULL OR page_end >= 1)"
)


def upgrade() -> None:
    # No data step. Unlike the widening migration this one adds no column, so there is
    # nothing to carry -- but if any row already violated the constraint the batch copy
    # would fail here rather than silently dropping it, which is the correct outcome.
    with op.batch_alter_table("chunks", schema=None) as batch_op:
        batch_op.create_check_constraint(
            batch_op.f("ck_chunks_page_numbers_positive"), _PAGE_NUMBERS_POSITIVE
        )


def downgrade() -> None:
    with op.batch_alter_table("chunks", schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f("ck_chunks_page_numbers_positive"), type_="check")
