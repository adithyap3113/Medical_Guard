"""Turning source PDFs into indexed, citable chunks.

The write path, in four pure-ish stages plus one orchestrator:

    parser   PDF bytes            -> pages of text, page numbers 1-based
    cleaner  pages                -> pages, de-hyphenated, headers stripped
    chunker  pages                -> chunks, each carrying the page span it covers
    pipeline chunks               -> SQL rows + Qdrant points, per document

The split is not decorative. Three of the four stages are deterministic functions of
their input and configuration -- no clock, no network, no database -- which is what
makes the chunk boundaries reproducible and testable offline. Everything that can fail
in an interesting way is concentrated in the parser (a PDF is arbitrary attacker-shaped
input) and the pipeline (which owns all the I/O and all the state).

**Page numbers are 1-based everywhere past :func:`parser.to_one_based_page`.** PyMuPDF
counts from 0; a clinician counts from 1; a citation is read by the clinician. The
conversion happens in exactly one place and the database rejects page 0 outright.
"""

from __future__ import annotations

__all__: list[str] = []
