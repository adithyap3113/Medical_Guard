"""Download the MedGuard source corpus.

A shim. Everything real lives in :mod:`medguard.corpus`, which sits under ``src/`` and
is therefore covered by mypy strict and exercised by the test suite -- neither of which
applies to ``scripts/``. Prefer ``medguard corpus download``; this file exists so the
Phase 1 entry point keeps working.
"""

from __future__ import annotations

import sys

from medguard.config import get_settings
from medguard.corpus.download import DownloadStatus, download_corpus
from medguard.corpus.manifest import build_manifest, write_manifest
from medguard.corpus.sources import load_sources
from medguard.errors import CorpusError


def main() -> int:
    settings = get_settings()
    try:
        sources = load_sources(settings.corpus.sources_path)
    except CorpusError as exc:
        print(f"FAILED: {exc.detail}", file=sys.stderr)
        return 1

    print(f"Writing to: {settings.corpus.raw_dir}\n")
    outcomes = download_corpus(sources, settings)
    for outcome in outcomes:
        size = f"{outcome.file_size_bytes / 1024:.0f} KB" if outcome.file_size_bytes else "—"
        detail = f" ({outcome.error})" if outcome.error else ""
        print(f"  {outcome.status.value:11s} {outcome.local_filename:44s} {size}{detail}")

    write_manifest(build_manifest(sources, outcomes, settings), settings.corpus.manifest_path)

    failed = [o for o in outcomes if o.status is DownloadStatus.FAILED]
    usable = [o for o in outcomes if o.ok]
    print(f"\nStatus: {len(usable)}/{len(outcomes)} documents ready, {len(failed)} failed.")
    # Non-zero on failure: the Phase 1 version always exited 0, which made this script
    # useless as a CI step because no caller could tell success from total failure.
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
