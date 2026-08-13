"""Source-document corpus: acquisition, provenance, and integrity.

Three concerns, deliberately separated:

* :mod:`medguard.corpus.sources` -- *what the corpus should contain*, read from an
  editable ``data/sources.yaml`` rather than hard-coded in Python.
* :mod:`medguard.corpus.download` -- *getting the bytes*, safely: atomic writes, magic
  byte validation, bounded retries.
* :mod:`medguard.corpus.manifest` -- *what the corpus actually contains*, recorded as
  committed provenance and checkable against the files on disk.

The PDFs are not committed; the manifest is. That is the point of the split. A clone
of this repository knows exactly which documents the system was built on, what they
hashed to, and where they came from, without carrying 35 MB of government PDFs in git
history.

Who may write ``data/manifest.json``
------------------------------------
Exactly two callers, and this is a rule, not a description of current behaviour:

1. :func:`medguard.corpus.manifest.write_manifest`, reached via
   :func:`~medguard.corpus.manifest.build_manifest` on ``medguard corpus download``.
2. ``medguard corpus sync`` -- in adopt mode, and in the ``--backfill-*`` modes.

**``medguard ingest`` must never write to it.** Not directly, not through a helper, not
"just the page counts". The manifest is the committed record of what the corpus *is*;
ingestion is a consumer of that record, and a consumer that rewrites its own input has
destroyed the only independent check on itself -- ``corpus verify`` would then be
comparing ingestion's output against ingestion's own claims.

Ingestion writes what it learns to the ``documents`` table instead. A human promotes it
with an explicit ``medguard corpus sync --backfill-*``, which shows the diff and asks.
Every change to committed provenance stays a reviewable event rather than a side effect
of a pipeline run.

``tests/unit/test_manifest_mutation_rule.py`` enforces this: it greps the source tree
for writers and asserts the set is exactly the two above.
"""

from __future__ import annotations

from medguard.corpus.download import (
    DownloadOutcome,
    DownloadStatus,
    download_corpus,
    download_document,
)
from medguard.corpus.manifest import (
    Manifest,
    ManifestDocument,
    VerificationReport,
    build_manifest,
    load_manifest,
    sha256_file,
    verify_corpus,
    write_manifest,
)
from medguard.corpus.sources import SourceDocument, SourceList, load_sources

__all__ = [
    "DownloadOutcome",
    "DownloadStatus",
    "Manifest",
    "ManifestDocument",
    "SourceDocument",
    "SourceList",
    "VerificationReport",
    "build_manifest",
    "download_corpus",
    "download_document",
    "load_manifest",
    "load_sources",
    "sha256_file",
    "verify_corpus",
    "write_manifest",
]
