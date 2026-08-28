"""Pooled candidate finder for authoring golden-set ground truth.

The problem this solves
-----------------------
Someone writing an evaluation question has to say which of 5,185 chunks answer it.
Reading all of them is not possible; grepping finds only literal string matches, which
biases the ground truth toward BM25 before a single metric has been computed.

So this pools candidates from three channels that fail differently -- dense (semantic,
misses rare tokens), sparse BM25 (lexical, misses paraphrase), and a SQL ``LIKE`` scan
(literal, misses everything else) -- and prints the union for a human to judge. This is
TREC-style pooled relevance assessment, and the honest limitation that comes with it is
stated in ``evals/README.md``: recall is measured against the pool, not against the
whole corpus, so a relevant chunk that no channel surfaced is invisible to the metric.

The circularity is real and worth naming: labelling from your own retriever's output
biases ground truth toward what that retriever already finds. Pooling three channels at
depth mitigates it; it does not remove it. The other half of the mitigation is human:
open the document around a candidate, because the *adjacent* chunk is often the better
answer and no channel will ever suggest it.

This is a development tool, not a product surface. It is deliberately a script rather
than a CLI command -- it is run while authoring a golden set and at no other time.

Usage:
    uv run python scripts/find_chunks.py "when should PrEP be offered"
    uv run python scripts/find_chunks.py --depth 40 --like "ceftriaxone" "gonorrhoea dose"
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medguard.config import get_settings  # noqa: E402
from medguard.errors import MedGuardError  # noqa: E402

PREVIEW_CHARS = 300


def _sqlite_path(database_url: str) -> Path | None:
    if not database_url.startswith("sqlite:"):
        return None
    return Path(database_url.split("///")[-1])


def _lexical_candidates(db: Path, term: str, limit: int) -> list[str]:
    """Literal substring matches. The channel neither vector model can be trusted for."""
    if not db.is_file():
        return []
    connection = sqlite3.connect(db)
    try:
        rows = connection.execute(
            "SELECT chunk_id FROM chunks WHERE text LIKE ? LIMIT ?",
            (f"%{term}%", limit),
        ).fetchall()
    finally:
        connection.close()
    return [row[0] for row in rows]


def _chunk_details(
    db: Path, chunk_ids: list[str]
) -> dict[str, tuple[str, int | None, int | None, str]]:
    if not chunk_ids or not db.is_file():
        return {}
    connection = sqlite3.connect(db)
    try:
        placeholders = ",".join("?" * len(chunk_ids))
        rows = connection.execute(
            f"SELECT chunk_id, document_id, page_start, page_end, text "  # noqa: S608
            f"FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
    finally:
        connection.close()
    return {row[0]: (row[1], row[2], row[3], row[4]) for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("query", help="The candidate golden-set question.")
    parser.add_argument("--depth", type=int, default=30, help="Depth per vector channel.")
    parser.add_argument(
        "--like",
        default=None,
        help="Literal term for the SQL channel. Defaults to the longest word in the query.",
    )
    args = parser.parse_args()

    settings = get_settings()
    db = _sqlite_path(settings.app.database_url)
    if db is None:
        print("This helper reads chunk text from SQLite; point DATABASE_URL at one.")
        return 2

    pools: dict[str, list[str]] = {}

    try:
        from medguard.embeddings.fastembed_dense import get_dense_embedder
        from medguard.embeddings.fastembed_sparse import get_sparse_embedder
        from medguard.storage.vector_store import VectorStore

        store = VectorStore(settings)
        if store.health().reachable:
            dense = store.search_dense(
                get_dense_embedder().embed_query(args.query), limit=args.depth
            )
            pools["dense"] = [str((p.payload or {}).get("chunk_id")) for p in dense]
            sparse = store.search_sparse(
                get_sparse_embedder().embed_query(args.query), limit=args.depth
            )
            pools["bm25"] = [str((p.payload or {}).get("chunk_id")) for p in sparse]
        else:
            print("!! Qdrant unreachable: pooling from the SQL channel only.")
            print("!! Ground truth labelled from one channel is materially weaker.\n")
        store.close()
    except MedGuardError as exc:
        print(f"!! vector channels unavailable ({exc}); SQL channel only.\n")

    term = args.like or max(args.query.split(), key=len)
    pools["like"] = _lexical_candidates(db, term, args.depth)

    ordered: list[str] = []
    for channel_ids in pools.values():
        for chunk_id in channel_ids:
            if chunk_id and chunk_id not in ordered:
                ordered.append(chunk_id)

    details = _chunk_details(db, ordered)

    print(f'query: "{args.query}"   (SQL channel term: {term!r})')
    print(
        f"pool: {len(ordered)} unique chunks from "
        + ", ".join(f"{name}={len(ids)}" for name, ids in pools.items())
    )
    print("=" * 100)
    for chunk_id in ordered:
        if chunk_id not in details:
            continue
        document_id, page_start, page_end, text = details[chunk_id]
        found_in = [name for name, ids in pools.items() if chunk_id in ids]
        ranks = ", ".join(f"{name}#{pools[name].index(chunk_id) + 1}" for name in found_in)
        pages = f"p{page_start}" + (f"-{page_end}" if page_end != page_start else "")
        preview = " ".join(text[:PREVIEW_CHARS].split())
        print(f"\n{chunk_id}  [{document_id} {pages}]  ({ranks})")
        print(f"    {preview}...")

    print("\n" + "=" * 100)
    print("A chunk found by only one channel is the interesting case: it is what the")
    print("other channel would have missed. Read the document around anything you label.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
