"""ONE-SHOT: remove one document from the operational stores.

Not a CLI command, and deliberately not one. Deleting indexed content is rare, it is
irreversible without a re-ingest, and the safe version of it asks a human to confirm the
document id by typing it. A permanent ``medguard`` subcommand would make it routine,
which is the opposite of what it should be. If this needs to happen often enough to
justify a command, that is a Phase 4 conversation.

Written for the Phase 3B closure: ``icmr_amr_treatment_2022`` turned out to be a
one-page QR-code landing card rather than the guidelines it named, so it had to leave
SQL, Qdrant and -- unlike a normally retired document -- ``data/raw/`` as well. A
retired entry whose file stays on disk is rebuilt into the manifest and re-indexed by
the next ``medguard ingest``, so ``--delete-file`` is what actually makes the removal
stick.

Order matters. Qdrant is emptied first and the drop verified against the chunk count SQL
still holds; only then does SQL let go. Reversed, a failure between the two steps would
leave orphan vectors that nothing in the database knows to point at -- invisible to
``medguard doctor``'s index check, which compares two totals rather than two sets.

    uv run python scripts/retire_document.py <document_id> [--delete-file]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sqlalchemy import func, select  # noqa: E402

from medguard.config import get_settings  # noqa: E402
from medguard.storage.database import Database  # noqa: E402
from medguard.storage.schema import ChunkRow, DocumentRow  # noqa: E402
from medguard.storage.vector_store import VectorStore  # noqa: E402


def _log(message: str = "") -> None:
    print(message, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document_id")
    parser.add_argument(
        "--delete-file",
        action="store_true",
        help="also delete the PDF from data/raw/, so the manifest stops recording it",
    )
    args = parser.parse_args()

    settings = get_settings()
    database = Database(settings)
    store = VectorStore(settings)

    try:
        with database.session() as session:
            document = session.get(DocumentRow, args.document_id)
            if document is None:
                _log(f"No such document in the database: {args.document_id}")
                return 1

            sql_chunks = session.scalar(
                select(func.count())
                .select_from(ChunkRow)
                .where(ChunkRow.document_id == args.document_id)
            )
            path = settings.corpus.raw_dir / document.local_filename
            points_before = store.collection_points_count()
            doc_points = store.count_for_document(args.document_id)

            _log("About to remove, permanently:")
            _log(f"  document_id : {document.document_id}")
            _log(f"  title       : {document.title}")
            _log(f"  status      : {document.ingestion_status.value}")
            _log(f"  SQL chunks  : {sql_chunks}")
            _log(f"  Qdrant pts  : {doc_points}  (collection holds {points_before})")
            _log(f"  file        : {path}{'' if path.is_file() else '  [absent]'}")
            _log(f"  delete file : {'yes' if args.delete_file else 'no'}")
            _log()

            typed = input(f"Type the document id to confirm ({args.document_id}): ").strip()
            if typed != args.document_id:
                _log("Confirmation did not match. Nothing was changed.")
                return 1

            removed = store.delete_by_document(args.document_id)
            points_after = store.collection_points_count()
            dropped = points_before - points_after
            _log(
                f"Qdrant: reported {removed} removed; collection {points_before} -> "
                f"{points_after} (dropped {dropped})"
            )
            if dropped != doc_points:
                _log(
                    f"ABORT: expected the collection to drop by exactly {doc_points}. "
                    "SQL untouched; investigate before retrying."
                )
                return 1

            # Chunks cascade from the document row.
            session.delete(document)
            session.commit()
            _log(f"SQL: document row and {sql_chunks} chunk rows deleted")

        if args.delete_file and path.is_file():
            path.unlink()
            _log(f"Disk: deleted {path}")
        elif args.delete_file:
            _log(f"Disk: nothing to delete at {path}")

        _log()
        _log(
            "Done. Run `medguard corpus download` to rebuild the manifest, then "
            "`medguard doctor` to confirm SQL and Qdrant still agree."
        )
        return 0
    finally:
        store.close()
        database.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
