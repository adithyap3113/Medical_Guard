"""The chunk page range, at the database level.

`ChunkRow` used to carry a single `page`, which could only say where a chunk *started*.
A citation into a chunk spanning pages 41-43 pointed a clinician at page 41 and left the
rest for them to find. `page_start` / `page_end` records what was always true.

The check constraint is the part worth testing. Pydantic already rejects an inverted
range on `medguard.models.Chunk`, but rows can reach the table without passing through
that model -- a migration, a bulk load, a repair script -- and an inverted range is the
shape a citation renderer prints as "pages 43-41" without noticing anything is wrong.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy.exc import IntegrityError

from medguard.config import Settings
from medguard.models import DocumentType, IngestionStatus, SourceOrganization
from medguard.storage.database import Database
from medguard.storage.schema import ChunkRow, DocumentRow, make_chunk_id


@pytest.fixture
def database(settings: Settings) -> Database:
    db = Database(settings)
    db.create_all()
    with db.session() as session:
        session.add(
            DocumentRow(
                document_id="cdc_example_2024",
                title="An Example Guideline, 2024",
                source_url="https://example.invalid/a.pdf",
                source_organization=SourceOrganization.CDC,
                document_type=DocumentType.GUIDELINE,
                local_filename="cdc_example_2024.pdf",
                sha256="a" * 64,
                file_size_bytes=1234,
                retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
                ingestion_status=IngestionStatus.PENDING,
                license_or_usage_note="Public domain.",
            )
        )
    return db


def _chunk(index: int, start: int | None, end: int | None) -> ChunkRow:
    return ChunkRow(
        chunk_id=make_chunk_id("cdc_example_2024", index),
        document_id="cdc_example_2024",
        chunk_index=index,
        page_start=start,
        page_end=end,
        text="Recommended first-line therapy is …",
        char_count=35,
        chunk_metadata={},
    )


class TestAcceptedRanges:
    def test_a_multi_page_span_is_stored(self, database: Database) -> None:
        with database.session() as session:
            session.add(_chunk(0, 41, 43))
        with database.session() as session:
            row = session.get(ChunkRow, "cdc_example_2024::0000")
            assert row is not None
            assert (row.page_start, row.page_end) == (41, 43)

    def test_a_single_page_span_is_stored(self, database: Database) -> None:
        with database.session() as session:
            session.add(_chunk(0, 7, 7))
        with database.session() as session:
            assert session.get(ChunkRow, "cdc_example_2024::0000") is not None

    def test_both_null_is_allowed(self, database: Database) -> None:
        """Page attribution can fail. Null means "not known", never "page zero"."""
        with database.session() as session:
            session.add(_chunk(0, None, None))
        with database.session() as session:
            row = session.get(ChunkRow, "cdc_example_2024::0000")
            assert row is not None
            assert row.page_start is None
            assert row.page_end is None

    def test_a_null_end_is_allowed(self, database: Database) -> None:
        """A chunk that starts on a known page and runs past what could be mapped."""
        with database.session() as session:
            session.add(_chunk(0, 41, None))
        with database.session() as session:
            assert session.get(ChunkRow, "cdc_example_2024::0000") is not None

    def test_a_null_start_is_allowed(self, database: Database) -> None:
        with database.session() as session:
            session.add(_chunk(0, None, 43))
        with database.session() as session:
            assert session.get(ChunkRow, "cdc_example_2024::0000") is not None


class TestRejectedRanges:
    def test_an_inverted_range_is_rejected(self, database: Database) -> None:
        # The session exits first and raises on commit; pytest.raises catches it.
        with pytest.raises(IntegrityError, match="page_range_ordered"), database.session() as s:
            s.add(_chunk(0, 43, 41))

    def test_an_inverted_range_by_one_is_rejected(self, database: Database) -> None:
        with pytest.raises(IntegrityError, match="page_range_ordered"), database.session() as s:
            s.add(_chunk(0, 8, 7))


class TestTheOldColumnIsGone:
    def test_chunks_has_page_start_and_page_end_and_no_page(self, database: Database) -> None:
        from sqlalchemy import inspect

        columns = {c["name"] for c in inspect(database.engine).get_columns("chunks")}
        assert {"page_start", "page_end"} <= columns
        assert "page" not in columns

    def test_the_orm_model_exposes_the_pair(self) -> None:
        assert hasattr(ChunkRow, "page_start")
        assert hasattr(ChunkRow, "page_end")
        assert not hasattr(ChunkRow, "page")

    def test_the_constraint_is_named(self, database: Database) -> None:
        """Batch migrations on SQLite cannot drop a constraint they cannot name."""
        from sqlalchemy import inspect

        names = {c["name"] for c in inspect(database.engine).get_check_constraints("chunks")}
        assert "ck_chunks_page_range_ordered" in names


class TestTheDomainModelAgrees:
    def test_the_pydantic_chunk_uses_the_same_pair(self) -> None:
        """The row and the model must not disagree about what a page reference is."""
        from medguard.models import Chunk

        chunk = Chunk(
            chunk_id=make_chunk_id("cdc_example_2024", 0),
            document_id="cdc_example_2024",
            text="t",
            chunk_index=0,
            page_start=41,
            page_end=43,
        )
        assert (chunk.page_start, chunk.page_end) == (41, 43)

    def test_the_citation_model_uses_the_same_pair(self) -> None:
        from medguard.models import Citation

        citation = Citation(
            marker="[1]",
            chunk_id="cdc_example_2024::0000",
            document_id="cdc_example_2024",
            page_start=41,
            page_end=43,
        )
        assert (citation.page_start, citation.page_end) == (41, 43)
