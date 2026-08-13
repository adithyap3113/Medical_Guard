"""ORM schema behaviour.

Phase 2 stores no data, so these tests check the *constraints* rather than any
queries. That is the point: constraints are the part of a schema that silently stops
working if someone changes a column, and the failures they prevent -- duplicate
chunks inflating recall, orphaned rows surviving a re-ingest -- are invisible until
an evaluation run produces numbers nobody can explain.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy.exc import IntegrityError as SAIntegrityError
from sqlalchemy.exc import StatementError

from medguard.config import Settings
from medguard.models import DocumentType, IngestionStatus, SourceOrganization
from medguard.storage.database import Database
from medguard.storage.schema import ChunkRow, DocumentRow, make_chunk_id


@pytest.fixture
def db(settings: Settings) -> Database:
    """A migrated throwaway database.

    Foreign-key enforcement is not switched on here: `Database` does it for every
    SQLite connection it opens, which is what the cascade tests below actually rely on.
    Enabling it in the fixture instead would test the fixture, not the application.
    """
    database = Database(settings)
    database.create_all()
    return database


def _document(**overrides: object) -> DocumentRow:
    values: dict[str, object] = {
        "document_id": "cdc_example_2024",
        "title": "An Example Guideline, 2024",
        "source_url": "https://example.invalid/a.pdf",
        "source_organization": SourceOrganization.CDC,
        "document_type": DocumentType.GUIDELINE,
        "local_filename": "cdc_example_2024.pdf",
        "sha256": "a" * 64,
        "file_size_bytes": 12345,
        "retrieved_at": datetime(2026, 1, 1, tzinfo=UTC),
        "license_or_usage_note": "Public domain.",
    }
    values.update(overrides)
    return DocumentRow(**values)


def _chunk(**overrides: object) -> ChunkRow:
    values: dict[str, object] = {
        "chunk_id": make_chunk_id("cdc_example_2024", 0),
        "document_id": "cdc_example_2024",
        "chunk_index": 0,
        "text": "Recommendation 1: measure blood pressure at every visit.",
        "char_count": 56,
    }
    values.update(overrides)
    return ChunkRow(**values)


class TestChunkId:
    def test_is_zero_padded_for_lexical_ordering(self) -> None:
        """Unpadded indices sort '10' before '2', which scrambles document order."""
        assert make_chunk_id("doc", 7) == "doc::0007"
        assert make_chunk_id("doc", 1234) == "doc::1234"

    def test_sorts_in_document_order(self) -> None:
        ids = [make_chunk_id("doc", i) for i in (0, 2, 10, 100)]
        assert sorted(ids) == ids


class TestDocumentRow:
    def test_round_trips(self, db: Database) -> None:
        with db.session() as session:
            session.add(_document())

        with db.session() as session:
            row = session.get(DocumentRow, "cdc_example_2024")
            assert row is not None
            assert row.source_organization is SourceOrganization.CDC
            assert row.document_type is DocumentType.GUIDELINE

    def test_ingestion_status_defaults_to_pending(self, db: Database) -> None:
        """A newly recorded document has not been ingested, and must say so."""
        with db.session() as session:
            session.add(_document())
        with db.session() as session:
            row = session.get(DocumentRow, "cdc_example_2024")
            assert row is not None
            assert row.ingestion_status is IngestionStatus.PENDING

    def test_page_count_is_nullable(self, db: Database) -> None:
        """Phase 3 fills it. Null means 'not counted', not 'zero pages'."""
        with db.session() as session:
            session.add(_document())
        with db.session() as session:
            row = session.get(DocumentRow, "cdc_example_2024")
            assert row is not None
            assert row.page_count is None

    def test_timestamps_are_populated_by_the_database(self, db: Database) -> None:
        with db.session() as session:
            session.add(_document())
        with db.session() as session:
            row = session.get(DocumentRow, "cdc_example_2024")
            assert row is not None
            assert row.created_at is not None
            assert row.updated_at is not None

    def test_duplicate_sha256_is_rejected(self, db: Database) -> None:
        """Two ids over identical bytes double-count one source in every metric."""
        with db.session() as session:
            session.add(_document())
        with pytest.raises(SAIntegrityError), db.session() as session:
            session.add(_document(document_id="cdc_other_2024", local_filename="other.pdf"))

    def test_an_invalid_organisation_is_rejected(self, db: Database) -> None:
        """The CHECK constraint keeps 'RandomBlog' out of the publisher column."""
        with pytest.raises((SAIntegrityError, StatementError, LookupError)), db.session() as s:
            s.add(_document(source_organization="RandomBlog"))

    def test_negative_file_size_is_rejected(self, db: Database) -> None:
        with pytest.raises(SAIntegrityError), db.session() as session:
            session.add(_document(file_size_bytes=-1))

    def test_a_short_hash_is_rejected(self, db: Database) -> None:
        with pytest.raises(SAIntegrityError), db.session() as session:
            session.add(_document(sha256="abc"))


class TestChunkRow:
    def test_round_trips_with_json_metadata(self, db: Database) -> None:
        with db.session() as session:
            session.add(_document())
            session.add(_chunk(chunk_metadata={"heading": "Recommendations", "level": 2}))

        with db.session() as session:
            row = session.get(ChunkRow, "cdc_example_2024::0000")
            assert row is not None
            assert row.chunk_metadata == {"heading": "Recommendations", "level": 2}

    def test_chunk_metadata_defaults_to_an_empty_object(self, db: Database) -> None:
        with db.session() as session:
            session.add(_document())
            session.add(_chunk())
        with db.session() as session:
            row = session.get(ChunkRow, "cdc_example_2024::0000")
            assert row is not None
            assert row.chunk_metadata == {}

    def test_the_foreign_key_is_enforced(self, db: Database) -> None:
        with pytest.raises(SAIntegrityError), db.session() as session:
            session.add(_chunk(document_id="no_such_document"))

    def test_a_document_cannot_have_two_chunks_at_the_same_index(self, db: Database) -> None:
        """A partially-failed re-ingest would otherwise duplicate spans and inflate recall."""
        with db.session() as session:
            session.add(_document())
            session.add(_chunk())

        with pytest.raises(SAIntegrityError), db.session() as session:
            session.add(_chunk(chunk_id="cdc_example_2024::0000-dup"))

    def test_negative_chunk_index_is_rejected(self, db: Database) -> None:
        with db.session() as session:
            session.add(_document())
        with pytest.raises(SAIntegrityError), db.session() as session:
            session.add(_chunk(chunk_index=-1, chunk_id="bad"))

    def test_deleting_a_document_deletes_its_chunks(self, db: Database) -> None:
        """Orphaned chunks still match retrieval queries and corrupt every metric."""
        with db.session() as session:
            session.add(_document())
            session.add(_chunk())
            session.add(_chunk(chunk_id=make_chunk_id("cdc_example_2024", 1), chunk_index=1))

        with db.session() as session:
            document = session.get(DocumentRow, "cdc_example_2024")
            assert document is not None
            session.delete(document)

        with db.session() as session:
            assert session.query(ChunkRow).count() == 0


class TestSessionScope:
    def test_a_failed_transaction_leaves_nothing_behind(self, db: Database) -> None:
        """Ingestion depends on this: one bad document must not half-commit."""

        class Boom(Exception):
            pass

        with pytest.raises(Boom), db.session() as session:
            session.add(_document())
            session.flush()
            raise Boom

        with db.session() as session:
            assert session.query(DocumentRow).count() == 0
