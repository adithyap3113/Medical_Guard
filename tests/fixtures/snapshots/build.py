"""Regenerate the committed chunker snapshot.

    python -m tests.fixtures.snapshots.build

Run this only when a chunk-boundary change is *intended*, and read the resulting diff
before committing it. The snapshot's job is to make an unintended boundary move visible
in review; regenerating without looking defeats the entire mechanism.

The snapshot is JSON rather than a pickle or a binary because the diff is the product.
Chunk text is stored in full so that a reviewer can see exactly where a boundary moved
and judge whether the new one is better -- a hash would report that something changed
and tell them nothing about what.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_TESTS = Path(__file__).resolve().parents[2]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from medguard.ingestion.chunker import chunk_document  # noqa: E402
from medguard.ingestion.cleaner import clean_document  # noqa: E402
from medguard.ingestion.parser import parse  # noqa: E402
from tests.fixtures.make_pdf import write_structured_pdf  # noqa: E402

OUTPUT = Path(__file__).resolve().parent / "structured_chunks.json"

#: Kept in step with `SNAPSHOT_SETTINGS` in tests/unit/test_chunker.py, which asserts
#: that the settings recorded in the file match the ones it chunks with.
DOCUMENT_ID = "snapshot_doc"
PAGES = 5


def build() -> dict[str, object]:
    from medguard.config import IngestionSettings

    settings = IngestionSettings(
        chunk_size=400,
        chunk_overlap=100,
        chunk_min_chars=80,
        chunk_max_chars=500,
    )

    with tempfile.TemporaryDirectory() as tmp:
        pdf = write_structured_pdf(Path(tmp) / "snapshot.pdf", pages=PAGES)
        pages = clean_document(list(parse(pdf).pages()), settings)
        chunks = chunk_document(DOCUMENT_ID, pages, settings)

    return {
        "fixture": {
            "generator": "tests/fixtures/make_pdf.py::write_structured_pdf",
            "pages": PAGES,
            "document_id": DOCUMENT_ID,
        },
        "settings": settings.model_dump(),
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "chunk_index": c.chunk_index,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "char_count": len(c.text),
                "text": c.text,
            }
            for c in chunks
        ],
    }


def main() -> None:
    payload = build()
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    chunks = payload["chunks"]
    assert isinstance(chunks, list)
    print(f"wrote {OUTPUT.relative_to(_TESTS.parent)} with {len(chunks)} chunks")


if __name__ == "__main__":
    main()
