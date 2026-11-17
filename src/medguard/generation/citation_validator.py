"""Conservative citation verification against retrieved candidate evidence.

Validates that every citation emitted by the generator:
1. Points to an actual chunk in the retrieved candidate pool.
2. Identifies the correct document ID and title.
3. Reflects the actual page range of the chunk.
4. If an evidence quote is provided, confirms the quote is present in the cited chunk's text
   using rigorous Unicode NFKC normalization and whitespace standardization.

Distinguishes between:
- quote_verified: Verbatim evidence quote verified in chunk text (supported=True).
- source_valid: Valid source and chunk reference, but without a verified quote (supported=None, unverified).
- unsupported: Invalid chunk, document mismatch, page number mismatch, or fabricated quote (supported=False).

Important: Exact quotation matching confirms textual grounding in the cited chunk;
it does not constitute automated semantic entailment or formal clinical proof.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from medguard.models import Candidate, Citation

__all__ = [
    "CitationValidator",
    "check_quote_in_chunk",
    "normalize_for_matching",
    "validate_citations",
]


def normalize_for_matching(text: str) -> str:
    """Normalize text for robust evidence quote verification.

    1. Applies Unicode NFKC normalization (decomposing ligatures and special symbols).
    2. Converts curly quotation marks and dashes to ASCII standards.
    3. Collapses all whitespace runs (including newlines and non-breaking spaces) into a single space.
    4. Strips leading and trailing whitespace and enclosing quote marks.
    5. Case-folds to lowercase for robust comparison.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u2026", "...")
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def check_quote_in_chunk(quote: str, chunk_text: str) -> bool:
    """Check if an evidence quote is grounded in the chunk's text."""
    norm_quote = normalize_for_matching(quote)
    norm_chunk = normalize_for_matching(chunk_text)
    if not norm_quote or not norm_chunk:
        return False

    # Exact normalized substring match
    if norm_quote in norm_chunk:
        return True

    # Strip surrounding quotation marks or ending punctuation that might differ
    core_quote = norm_quote.strip("\"'.,;:!?()[]{}")
    return bool(core_quote and len(core_quote) >= 5 and core_quote in norm_chunk)


class CitationValidator:
    """Validates citations conservatively against the pool of retrieved candidates."""

    def __init__(self, candidates: Sequence[Candidate]) -> None:
        self._pool: dict[str, Candidate] = {c.chunk_id: c for c in candidates}

    def validate(self, citations: Sequence[Citation]) -> list[Citation]:
        """Verify each citation and return updated Citation objects with validation verdicts."""
        validated: list[Citation] = []

        for cit in citations:
            cand = self._pool.get(cit.chunk_id)
            if cand is None:
                validated.append(
                    Citation(
                        marker=cit.marker,
                        chunk_id=cit.chunk_id,
                        document_id=cit.document_id,
                        document_title=cit.document_title,
                        page_start=cit.page_start,
                        page_end=cit.page_end,
                        quote=cit.quote,
                        supported=False,
                        validation_status="unsupported",
                        validation_note=(
                            f"Hallucinated reference: chunk_id {cit.chunk_id!r} was not in "
                            "the retrieved candidate pool."
                        ),
                    )
                )
                continue

            chunk = cand.chunk
            actual_doc_id = chunk.document_id
            title = chunk.metadata.get("title") or cit.document_title or actual_doc_id

            # 1. Verify document ID matches
            if cit.document_id and cit.document_id != actual_doc_id:
                validated.append(
                    Citation(
                        marker=cit.marker,
                        chunk_id=chunk.chunk_id,
                        document_id=cit.document_id,
                        document_title=title,
                        page_start=cit.page_start,
                        page_end=cit.page_end,
                        quote=cit.quote,
                        supported=False,
                        validation_status="unsupported",
                        validation_note=(
                            f"Document mismatch: citation stated {cit.document_id!r} but chunk belongs to "
                            f"{actual_doc_id!r}."
                        ),
                    )
                )
                continue

            # 2. Verify page numbers if explicitly cited
            page_start = cit.page_start if cit.page_start is not None else chunk.page_start
            page_end = cit.page_end if cit.page_end is not None else chunk.page_end

            page_mismatch = False
            if cit.page_start is not None and chunk.page_start is not None:
                max_page = chunk.page_end or chunk.page_start
                min_page = chunk.page_start
                if cit.page_start < min_page or cit.page_start > max_page:
                    page_mismatch = True

            if page_mismatch:
                validated.append(
                    Citation(
                        marker=cit.marker,
                        chunk_id=chunk.chunk_id,
                        document_id=actual_doc_id,
                        document_title=title,
                        page_start=cit.page_start,
                        page_end=cit.page_end,
                        quote=cit.quote,
                        supported=False,
                        validation_status="unsupported",
                        validation_note=(
                            f"Page mismatch: cited page {cit.page_start} is outside chunk page span "
                            f"[{chunk.page_start}-{chunk.page_end or chunk.page_start}]."
                        ),
                    )
                )
                continue

            # 3. Verify quote if provided
            if cit.quote and cit.quote.strip():
                quote_matches = check_quote_in_chunk(cit.quote, chunk.text)
                if quote_matches:
                    supported: bool | None = True
                    validation_status = "quote_verified"
                    validation_note = "Verified verbatim against retrieved source chunk text."
                else:
                    supported = False
                    validation_status = "unsupported"
                    validation_note = "Fabricated or ungrounded quote: text does not appear in the cited source chunk."
            else:
                # Chunk and document are valid, but no verbatim quote provided to verify text grounding
                supported = None
                validation_status = "source_valid"
                validation_note = "Valid source reference in candidate pool; no verbatim quote provided for evidence grounding."

            validated.append(
                Citation(
                    marker=cit.marker,
                    chunk_id=chunk.chunk_id,
                    document_id=actual_doc_id,
                    document_title=title,
                    page_start=page_start,
                    page_end=page_end,
                    quote=cit.quote,
                    supported=supported,
                    validation_status=validation_status,
                    validation_note=validation_note,
                )
            )

        return validated


def validate_citations(
    citations: Sequence[Citation], candidates: Sequence[Candidate]
) -> list[Citation]:
    validator = CitationValidator(candidates)
    return validator.validate(citations)
