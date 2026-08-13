"""Prove that Qdrant survives a container restart.

Why this needs proving
----------------------
A Qdrant container without a mounted volume works perfectly until the first restart,
then comes back empty. Nothing in normal development surfaces that: ingestion succeeds,
retrieval succeeds, and the corpus quietly disappears the next time Docker Desktop is
restarted -- typically discovered hours later, mid-evaluation, and misdiagnosed as a
retrieval bug.

So this script writes known vectors, restarts the container, and asserts the vectors
came back. It is the only way to know the compose file's volume actually works, as
opposed to looking like it should.

Deliberately safe:

* It creates and deletes its own throwaway collection. It never touches the real
  collection, which does not exist yet by design.
* It restarts the container. It never runs ``docker compose down -v``, which deletes
  the volume and would destroy exactly what this script exists to protect.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = PROJECT_ROOT / "docker" / "docker-compose.yml"

TEST_COLLECTION = "medguard__phase2_persistence_test"
VECTOR_SIZE = 4
TEST_POINTS: list[tuple[int, list[float]]] = [
    (1, [0.1, 0.2, 0.3, 0.4]),
    (2, [0.5, 0.6, 0.7, 0.8]),
    (3, [0.9, 1.0, 1.1, 1.2]),
]
RESTART_POLL_TIMEOUT_S = 60.0
RESTART_POLL_INTERVAL_S = 2.0


def _log(message: str) -> None:
    print(message, flush=True)


def _restart_qdrant() -> bool:
    _log(f"  restarting qdrant via {COMPOSE_FILE.name} ...")
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
            ["docker", "compose", "-f", str(COMPOSE_FILE), "restart", "qdrant"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _log(f"  FAILED: could not run docker compose: {exc}")
        return False
    if proc.returncode != 0:
        _log(f"  FAILED: docker compose restart exited {proc.returncode}: {proc.stderr.strip()}")
        return False
    return True


def main() -> int:
    from medguard.config import get_settings
    from medguard.storage.vector_store import IN_MEMORY_URL, VectorStore

    settings = get_settings()
    if settings.qdrant.url == IN_MEMORY_URL:
        _log("SKIP: QDRANT_URL is ':memory:'. In-process Qdrant has no persistence to test.")
        return 0
    if not COMPOSE_FILE.is_file():
        _log(f"FAIL: compose file not found at {COMPOSE_FILE}")
        return 1

    from qdrant_client import models as qmodels

    store = VectorStore(settings)
    started = time.perf_counter()

    try:
        health = store.health()
        if not health.reachable:
            _log(f"FAIL: Qdrant unreachable at {health.url} ({health.error}).")
            _log("      Start it: docker compose -f docker/docker-compose.yml up -d qdrant")
            return 1
        _log(f"  connected to {health.url}")

        _log(f"  creating throwaway collection {TEST_COLLECTION!r} ...")
        # Not `recreate_collection`: it is deprecated, and an unconditional drop would
        # be one typo in TEST_COLLECTION away from deleting the real corpus index.
        if store.client.collection_exists(TEST_COLLECTION):
            store.client.delete_collection(collection_name=TEST_COLLECTION)
        store.client.create_collection(
            collection_name=TEST_COLLECTION,
            vectors_config=qmodels.VectorParams(size=VECTOR_SIZE, distance=qmodels.Distance.COSINE),
        )

        _log(f"  upserting {len(TEST_POINTS)} vectors ...")
        store.client.upsert(
            collection_name=TEST_COLLECTION,
            wait=True,
            points=[
                qmodels.PointStruct(id=pid, vector=vec, payload={"marker": f"phase2-{pid}"})
                for pid, vec in TEST_POINTS
            ],
        )

        before = store.client.count(collection_name=TEST_COLLECTION, exact=True).count
        _log(f"  before restart: {before} points")
        if before != len(TEST_POINTS):
            _log(f"FAIL: expected {len(TEST_POINTS)} points before restart, got {before}")
            return 1
    finally:
        # The client holds a connection the restart will drop; reconnect afterwards.
        store.close()

    if not _restart_qdrant():
        return 1

    _log(f"  polling for readiness (up to {RESTART_POLL_TIMEOUT_S:.0f}s) ...")
    deadline = time.monotonic() + RESTART_POLL_TIMEOUT_S
    store = VectorStore(settings)
    ready = False
    while time.monotonic() < deadline:
        if store.health().reachable:
            ready = True
            break
        store.close()
        store = VectorStore(settings)
        time.sleep(RESTART_POLL_INTERVAL_S)

    if not ready:
        _log(f"FAIL: Qdrant did not become reachable within {RESTART_POLL_TIMEOUT_S:.0f}s")
        store.close()
        return 1
    _log("  reachable again")

    exit_code = 0
    try:
        after = store.client.count(collection_name=TEST_COLLECTION, exact=True).count
        _log(f"  after restart: {after} points")

        records, _ = store.client.scroll(
            collection_name=TEST_COLLECTION,
            limit=len(TEST_POINTS) + 1,
            with_payload=True,
            with_vectors=False,
        )
        recovered = sorted(int(r.id) for r in records)
        expected = sorted(pid for pid, _ in TEST_POINTS)

        if recovered != expected:
            _log(f"FAIL: expected ids {expected}, recovered {recovered}")
            _log("      Qdrant is NOT persisting. Check the qdrant_storage volume mount.")
            exit_code = 1
        else:
            _log(f"  recovered ids {recovered} — all present with correct ids")
    except Exception as exc:  # noqa: BLE001 - any failure here means the test did not pass
        _log(f"FAIL: {type(exc).__name__}: {exc}")
        exit_code = 1
    finally:
        try:
            store.client.delete_collection(collection_name=TEST_COLLECTION)
            _log(f"  deleted {TEST_COLLECTION!r}")
        except Exception as exc:  # noqa: BLE001 - cleanup failure must not mask the result
            _log(f"  WARNING: could not delete the test collection: {exc}")
        store.close()

    elapsed = time.perf_counter() - started
    _log(f"\n{'PASS' if exit_code == 0 else 'FAIL'}: Qdrant persistence ({elapsed:.1f}s)")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
