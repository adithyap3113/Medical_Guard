"""Qdrant persistence across a container restart.

Marked ``integration`` because it needs a running Docker daemon and restarts a
container. ``pytest -m "not integration"`` -- the default development loop -- skips it
entirely, so the unit suite stays offline and fast.

The reason this is worth a test at all: a Qdrant container without a working volume
mount behaves perfectly until its first restart, then comes back empty. Nothing in
normal development surfaces that, and the eventual symptom looks like a retrieval bug
rather than a storage one.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from medguard.config import get_settings, project_root
from medguard.storage.vector_store import IN_MEMORY_URL

pytestmark = pytest.mark.integration


@pytest.fixture
def live_qdrant_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo conftest's in-memory redirect: this test needs the real server."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    get_settings.cache_clear()


def test_qdrant_survives_a_container_restart(live_qdrant_env: None) -> None:
    """Runs the persistence script and asserts it reports PASS.

    The script rather than an inline reimplementation, so that what CI verifies and
    what a developer runs by hand are the same code.
    """
    settings = get_settings()
    if settings.qdrant.url == IN_MEMORY_URL:
        pytest.skip("QDRANT_URL is in-memory; there is no persistence to test")

    from medguard.storage.vector_store import VectorStore

    store = VectorStore(settings)
    try:
        reachable = store.health().reachable
    finally:
        store.close()
    if not reachable:
        pytest.skip("Qdrant is not running; start it with docker compose up -d qdrant")

    script = project_root() / "scripts" / "verify_qdrant_persistence.py"
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        cwd=str(project_root()),
    )

    assert proc.returncode == 0, f"persistence check failed:\n{proc.stdout}\n{proc.stderr}"
    assert "PASS" in proc.stdout
