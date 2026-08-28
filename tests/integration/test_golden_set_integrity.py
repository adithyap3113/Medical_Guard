"""Does the committed golden set still describe the corpus it was authored against?

Three checks, in increasing order of how badly a failure would mislead:

* **A — ground truth resolves.** Every ``relevant_chunk_id`` names a chunk that exists.
* **B — pool coverage.** *Most* scored questions have at least one labelled chunk
  reachable by dense or sparse retrieval at depth. A threshold rather than a
  requirement, deliberately: demanding every question be reachable would forbid the
  hard ones worth keeping. Individual misses are printed for a human to read.
* **C — provenance.** The chunking settings match the ones the ids were authored under.

C is the one that matters most and shows least. Chunk ids are position-addressed, so a
re-chunk leaves every id resolving to *different text*: the file parses, the ids resolve,
check A passes, the metrics compute, and every number is quietly wrong.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from medguard.config import Settings, get_settings
from medguard.evaluation.golden_set import (
    GOLDEN_SET_PATH,
    GoldenQuestion,
    load_golden_set,
    load_meta,
    verify_provenance,
)

pytestmark = pytest.mark.integration

POOL_DEPTH = 50


@pytest.fixture
def golden_set() -> list[GoldenQuestion]:
    # require_review=False: the shipped v1 is an unreviewed worked example, and these
    # tests check that its ground truth is *sound*, which is a different question from
    # whether a human has signed it off.
    return load_golden_set(GOLDEN_SET_PATH, require_review=False)


@pytest.fixture
def live_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Point at the real store and the real index.

    Function-scoped and monkeypatch-based for the same reason as
    ``test_retrieval_live.py``: conftest's autouse isolation is function-scoped and
    would overwrite anything set at module scope.
    """
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.delenv("QDRANT_COLLECTION", raising=False)
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./medguard.db")
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def db(live_settings: Settings):  # type: ignore[no-untyped-def]
    path = Path(live_settings.app.database_url.split("///")[-1])
    if not path.is_file():
        pytest.skip("the relational store does not exist; run `medguard ingest` first")
    connection = sqlite3.connect(path)
    try:
        if not connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]:
            pytest.skip("no chunks ingested; run `medguard ingest` first")
        yield connection
    finally:
        connection.close()


class TestGroundTruthResolves:
    """Check A."""

    def test_every_labelled_chunk_exists(
        self, golden_set: list[GoldenQuestion], db: sqlite3.Connection
    ) -> None:
        missing: list[str] = []
        for question in golden_set:
            for chunk_id in question.relevant_chunk_ids:
                row = db.execute("SELECT 1 FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
                if row is None:
                    missing.append(f"{question.question_id} -> {chunk_id}")
        assert not missing, "ground truth points at chunks that do not exist: " + str(missing)

    def test_every_labelled_document_exists(
        self, golden_set: list[GoldenQuestion], db: sqlite3.Connection
    ) -> None:
        missing: list[str] = []
        for question in golden_set:
            for document_id in question.relevant_document_ids:
                row = db.execute(
                    "SELECT 1 FROM documents WHERE document_id = ?", (document_id,)
                ).fetchone()
                if row is None:
                    missing.append(f"{question.question_id} -> {document_id}")
        assert not missing, str(missing)

    def test_labelled_chunks_carry_text(
        self, golden_set: list[GoldenQuestion], db: sqlite3.Connection
    ) -> None:
        """An empty chunk would satisfy check A and answer nothing."""
        for question in golden_set:
            for chunk_id in question.relevant_chunk_ids:
                text = db.execute(
                    "SELECT text FROM chunks WHERE chunk_id = ?", (chunk_id,)
                ).fetchone()[0]
                assert text and text.strip(), f"{question.question_id} -> {chunk_id} is empty"


class TestProvenance:
    """Check C -- the tripwire."""

    def test_the_chunking_settings_still_match(self, live_settings: Settings) -> None:
        verify_provenance(load_meta(GOLDEN_SET_PATH), live_settings.ingestion)

    def test_the_recorded_embedding_model_matches_the_collection(
        self, live_settings: Settings
    ) -> None:
        """A different model means a different vector space; the ids would resolve and
        the retrieval would be measuring something else entirely."""
        meta = load_meta(GOLDEN_SET_PATH)
        assert meta.embedding_model == live_settings.embedding.model
        assert meta.embedding_dimension == live_settings.embedding.dimension


@pytest.mark.slow
class TestPoolCoverage:
    """Check B -- the smoke test that labels and retrieval agree at all."""

    @pytest.fixture
    def pools(self, golden_set: list[GoldenQuestion], live_settings: Settings):  # type: ignore[no-untyped-def]
        from medguard.embeddings.fastembed_dense import get_dense_embedder
        from medguard.embeddings.fastembed_sparse import get_sparse_embedder
        from medguard.storage.vector_store import VectorStore

        store = VectorStore(live_settings)
        if not store.health().reachable:
            store.close()
            pytest.skip("Qdrant is not running; start it with docker compose")

        dense_embedder = get_dense_embedder()
        sparse_embedder = get_sparse_embedder()
        result: dict[str, set[str]] = {}
        try:
            for question in golden_set:
                if not question.is_scored:
                    continue
                found: set[str] = set()
                for points in (
                    store.search_dense(
                        dense_embedder.embed_query(question.question_text), limit=POOL_DEPTH
                    ),
                    store.search_sparse(
                        sparse_embedder.embed_query(question.question_text), limit=POOL_DEPTH
                    ),
                ):
                    found.update(str((p.payload or {}).get("chunk_id")) for p in points)
                result[question.question_id] = found
        finally:
            store.close()
        return result

    def test_most_scored_questions_have_a_labelled_chunk_in_the_pool(
        self, golden_set: list[GoldenQuestion], pools: dict[str, set[str]]
    ) -> None:
        """Detects systematic breakage, not individual hard questions.

        A threshold rather than "every question", and the distinction matters. If this
        demanded that every labelled chunk be reachable, it would quietly forbid exactly
        the questions worth having -- the ones current retrieval fails and a reranker or
        a hybrid might fix -- and the golden set would drift toward whatever the present
        retriever already finds. That is the circularity ``evals/README.md`` warns about,
        enforced by a test.

        What it is really watching for is the systematic case: a set authored against a
        different corpus, a different chunking, or the wrong collection, where coverage
        collapses across the board rather than for one question.

        The individual misses are still printed, because they are worth a human look. In
        Phase 4A one of them was a genuine mislabel: gq_014 was labelled with a single
        chunk whose text is dominated by opioid dosing, and neither leg reached it -- the
        recommendation it carries is also stated in three other chunks that were missing
        from the label.
        """
        unreachable: list[str] = []
        scored = [q for q in golden_set if q.is_scored]
        for question in scored:
            if not set(question.relevant_chunk_ids) & pools[question.question_id]:
                unreachable.append(f"{question.question_id} {question.relevant_chunk_ids}")

        if unreachable:
            # Printed rather than logged: pytest -s surfaces it to the person
            # reviewing the golden set, which is who needs to read it.
            print(
                f"\n{len(unreachable)}/{len(scored)} scored question(s) have no labelled "
                f"chunk within depth {POOL_DEPTH} of either retriever:\n  "
                + "\n  ".join(unreachable)
                + "\nEach is either a mislabel or a genuine retrieval failure. Both are "
                "worth reading; only the first is a bug in the golden set."
            )

        covered = len(scored) - len(unreachable)
        assert covered / len(scored) >= 0.7, (
            f"only {covered}/{len(scored)} scored questions have any labelled chunk "
            f"reachable at depth {POOL_DEPTH}. That is too few to be bad luck: the "
            f"golden set and the index have most likely come apart -- different corpus, "
            f"different chunking, or the wrong collection."
        )

    def test_unanswerable_questions_are_not_silently_answerable(
        self, golden_set: list[GoldenQuestion], live_settings: Settings
    ) -> None:
        """The out-of-corpus questions were verified absent by term search when they were
        written. This re-checks the claim against the live index rather than trusting the
        note, since a corpus addition could quietly make one of them answerable."""
        path = Path(live_settings.app.database_url.split("///")[-1])
        if not path.is_file():
            pytest.skip("relational store absent")
        connection = sqlite3.connect(path)
        try:
            for term in ("semaglutide", "Alzheimer"):
                count = connection.execute(
                    "SELECT COUNT(*) FROM chunks WHERE text LIKE ?", (f"%{term}%",)
                ).fetchone()[0]
                assert count == 0, (
                    f"{term!r} now appears in {count} chunk(s); the golden set's "
                    f"unsupported questions need re-labelling"
                )
        finally:
            connection.close()
