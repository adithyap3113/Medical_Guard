"""Deterministic recursive chunking, with page spans preserved.

Takes cleaned pages and returns :class:`~medguard.models.Chunk` objects. Pure: no clock,
no randomness, no I/O, no dependence on dictionary ordering. The same pages and the same
:class:`~medguard.config.IngestionSettings` produce byte-identical output on any machine,
which is what a committed snapshot test is able to assert and what makes a citation
reproducible six months later.

Why this is written rather than imported
----------------------------------------
LangChain ships a recursive character splitter that does most of this, and it is not
usable here for a reason that is structural rather than aesthetic: it takes a flat
string. A flat string has no pages. Feeding it a document means either losing the page
attribution entirely -- and a citation that cannot name a page sends a clinician into a
400-page PDF -- or splitting page by page, which forbids a chunk from ever spanning a
page break and therefore guarantees that every passage crossing one is cut in half. The
mechanism below tracks character offsets through the split so a chunk can span pages
*and* say which ones. That is the whole feature, and it is about two hundred lines.

The secondary reason is dependency weight: ``langchain-text-splitters`` pulls
``langchain-core`` and ``langsmith``, which is a large surface and a telemetry client to
acquire for one function.

Why characters and not tokens
-----------------------------
Token-accurate chunking requires the embedding model's tokeniser, which means loading
the model. That would make the chunker untestable offline, make its output depend on the
model version, and put a 130 MB download in the path of a unit test. Characters are a
proxy -- roughly four per token for English clinical prose -- and the ceiling that
matters (``chunk_max_chars``) is set from that ratio with room to spare. A chunk that
exceeds the model's 512-token window is not an error, it is *silently truncated at
embedding time*, so the budget is deliberately conservative.

The split hierarchy
-------------------
``\\n\\n`` then ``\\n`` then ``. `` then ``" "`` then raw characters. Each level is tried
only when the level above leaves a piece too large, so a paragraph is preferred whole, a
line is preferred whole, and a sentence is only broken when a single sentence is longer
than the chunk budget. Falling all the way to raw characters is possible and it is the
right behaviour: a 3,000-character unbroken table row has to go somewhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from medguard.config import IngestionSettings
from medguard.errors import ChunkingError
from medguard.ingestion.parser import PageText
from medguard.logging import get_logger
from medguard.models import Chunk
from medguard.storage.schema import make_chunk_id

__all__ = ["PAGE_SEPARATOR", "SEPARATORS", "chunk_document"]

log = get_logger(__name__)

#: Tried in order. The last entry is the empty string, which means "split anywhere":
#: the terminating case for text containing none of the separators above it.
SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")

#: Joins pages into the single string the splitter walks. A blank line, so a page break
#: reads to the splitter as the strongest possible boundary -- it will prefer to cut
#: there over cutting mid-paragraph, without being forbidden from spanning it when the
#: text genuinely runs on.
PAGE_SEPARATOR = "\n\n"


@dataclass(frozen=True, slots=True)
class _Atom:
    """A piece of the document, small enough to be a chunk, with its exact offset.

    Offsets are what make page attribution possible. The alternative -- searching for
    the chunk text in the document afterwards -- silently attributes the wrong page as
    soon as a passage appears twice, which in a guideline full of repeated boilerplate
    is common rather than hypothetical.
    """

    text: str
    start: int

    @property
    def end(self) -> int:
        return self.start + len(self.text)


@dataclass(frozen=True, slots=True)
class _PageSpan:
    """Where one page's text sits in the concatenated document."""

    page_number: int
    start: int
    end: int


def _assemble(pages: Sequence[PageText]) -> tuple[str, list[_PageSpan]]:
    """Concatenate pages into one string and record where each page landed."""
    parts: list[str] = []
    spans: list[_PageSpan] = []
    offset = 0

    for index, page in enumerate(pages):
        if index:
            parts.append(PAGE_SEPARATOR)
            offset += len(PAGE_SEPARATOR)
        parts.append(page.text)
        spans.append(_PageSpan(page.page_number, offset, offset + len(page.text)))
        offset += len(page.text)

    return "".join(parts), spans


def _split_keeping_offsets(atom: _Atom, separator: str) -> list[_Atom]:
    """Split one atom on `separator`, keeping the separator attached to its left piece.

    Attaching rather than discarding is what makes the concatenation of every atom equal
    the original document, character for character. That identity is relied on twice: to
    reconstruct chunk text by joining atoms, and to trust the offsets used for page
    attribution.
    """
    if separator == "":
        return [_Atom(c, atom.start + i) for i, c in enumerate(atom.text)]

    pieces: list[_Atom] = []
    cursor = 0
    while True:
        found = atom.text.find(separator, cursor)
        if found == -1:
            break
        cut = found + len(separator)
        pieces.append(_Atom(atom.text[cursor:cut], atom.start + cursor))
        cursor = cut

    if cursor < len(atom.text):
        pieces.append(_Atom(atom.text[cursor:], atom.start + cursor))
    return pieces


def _atomise(text: str, limit: int) -> list[_Atom]:
    """Break the document into pieces no longer than `limit`, preserving offsets.

    Breadth-first over the separator hierarchy: everything is split on paragraphs, then
    only the still-oversized pieces are split on lines, and so on. A piece that fits is
    never broken further, which is what keeps whole paragraphs whole.
    """
    atoms = [_Atom(text, 0)]

    for separator in SEPARATORS:
        if all(len(a.text) <= limit for a in atoms):
            return atoms

        refined: list[_Atom] = []
        for atom in atoms:
            if len(atom.text) <= limit:
                refined.append(atom)
                continue
            pieces = _split_keeping_offsets(atom, separator)
            # A separator that appears only at the very end splits nothing; fall through
            # to the next level rather than looping on an unchanged list.
            refined.extend(pieces if len(pieces) > 1 else [atom])
        atoms = refined

    return atoms


def _pack(atoms: Sequence[_Atom], settings: IngestionSettings) -> list[list[_Atom]]:
    """Greedily group atoms into chunks, carrying `chunk_overlap` characters forward.

    Overlap is taken at atom granularity rather than by slicing characters off the end
    of the previous chunk. Slicing would produce a chunk whose text is not a contiguous
    run of atoms, and the offset arithmetic that page attribution depends on would stop
    being exact. Whole atoms cost a little precision in the overlap length and buy
    correctness in the page span.
    """
    groups: list[list[_Atom]] = []
    current: list[_Atom] = []
    size = 0

    for atom in atoms:
        if current and size + len(atom.text) > settings.chunk_size:
            groups.append(current)
            current = _overlap_tail(current, settings.chunk_overlap)
            size = sum(len(a.text) for a in current)
        current.append(atom)
        size += len(atom.text)

    if current:
        groups.append(current)
    return groups


def _overlap_tail(group: Sequence[_Atom], overlap: int) -> list[_Atom]:
    """The trailing atoms of `group` totalling at most `overlap` characters.

    ``group[1:]``, never the whole group. Returning every atom would mean the next chunk
    starts exactly where this one did, and with an atom larger than the overlap budget
    the packer would never advance.

    The slice is at the *front* for a reason worth recording, because getting it wrong
    is silent. An earlier version excluded the last atom instead of the first, which
    made every chunk after the first begin at the document's opening atom: chunk text is
    reconstructed as the contiguous slice from the first atom's offset to the last one's,
    so each successive chunk grew by one atom and re-contained the entire document
    prefix. Every chunk still passed the size checks that were in place, and the only
    visible symptom was an "overlap" that grew without bound.
    """
    if overlap <= 0 or len(group) < 2:
        return []

    tail: list[_Atom] = []
    total = 0
    for atom in reversed(group[1:]):
        if total + len(atom.text) > overlap:
            break
        tail.insert(0, atom)
        total += len(atom.text)
    return tail


def _trimmed_bounds(text: str, start: int, end: int) -> tuple[str, int, int]:
    """Strip surrounding whitespace and move the offsets to match.

    Adjusting the offsets rather than stripping alone matters at a page boundary: a
    chunk beginning with the two newlines that separate pages would otherwise be
    attributed to the page that ended, not the one whose text it actually contains.
    """
    stripped = text.strip()
    if not stripped:
        return "", start, start
    lead = len(text) - len(text.lstrip())
    trail = len(text) - len(text.rstrip())
    return stripped, start + lead, end - trail


def _pages_for(spans: Sequence[_PageSpan], start: int, end: int) -> tuple[int | None, int | None]:
    """The first and last page a character range touches.

    Returns ``(None, None)`` when the range intersects no page, which can only happen
    for a chunk consisting entirely of separator characters. Null means "not known",
    never page zero -- the database rejects page zero outright.
    """
    touched = [s.page_number for s in spans if s.start < end and s.end > start]
    if not touched:
        return None, None
    return touched[0], touched[-1]


def _merge_short(groups: list[list[_Atom]], settings: IngestionSettings) -> list[list[_Atom]]:
    """Fold undersized chunks into their predecessor where the ceiling allows.

    A 40-character fragment -- "Table 3. Continued", a stray page number that survived
    cleaning -- embeds to a vector that means almost nothing and still occupies a slot
    in every top-k result it wins. Merging it into the passage it belongs to is strictly
    better than indexing it alone.

    A single chunk that is simply short is left as it is. It has nothing to merge with,
    and dropping it would silently lose the only content of a one-paragraph document.
    """
    merged: list[list[_Atom]] = []
    for group in groups:
        size = sum(len(a.text) for a in group)
        if merged and size < settings.chunk_min_chars:
            previous = sum(len(a.text) for a in merged[-1])
            if previous + size <= settings.chunk_max_chars:
                merged[-1] = merged[-1] + group
                continue
        merged.append(group)
    return merged


def _enforce_ceiling(text: str, start: int, settings: IngestionSettings) -> list[tuple[str, int]]:
    """Hard-split anything still above `chunk_max_chars`.

    A safety net rather than a routine path: the packer already stops at ``chunk_size``,
    and overlap can only add ``chunk_overlap`` on top, so with the shipped defaults
    (1600 + 200 against a 2000 ceiling) this never fires. It exists because the ceiling
    is a real constraint -- above it the embedding model truncates without telling
    anyone -- and a configuration that puts those three numbers in a different relation
    must degrade by splitting rather than by silently indexing half a passage.
    """
    limit = settings.chunk_max_chars
    if len(text) <= limit:
        return [(text, start)]

    pieces: list[tuple[str, int]] = []
    for offset in range(0, len(text), limit):
        piece = text[offset : offset + limit]
        pieces.append((piece, start + offset))
    return pieces


def chunk_document(
    document_id: str,
    pages: Sequence[PageText],
    settings: IngestionSettings,
) -> list[Chunk]:
    """Split a cleaned document into chunks, each knowing which pages it covers.

    An empty document returns an empty list rather than raising. A PDF of scanned images
    has no text layer and extracts to nothing; that is a fact about the document worth
    reporting as "0 chunks", not an exception to be handled at every call site.

    ``section`` is left null. Detecting the nearest enclosing heading requires deciding
    what a heading *is*, and every available signal here -- short line, no full stop,
    larger font -- misfires on the numbered recommendations and multi-line table headers
    that fill clinical guidance. Under this project's standing rule that a field which
    cannot be derived from the source is left null rather than guessed, a wrong section
    label is worse than none: it would be shown in a citation as though it were read off
    the document.
    """
    text, spans = _assemble(pages)
    if not text.strip():
        log.debug("chunker.empty_document", document_id=document_id)
        return []

    groups = _merge_short(_pack(_atomise(text, settings.chunk_size), settings), settings)

    chunks: list[Chunk] = []
    for group in groups:
        if not group:
            continue
        raw = text[group[0].start : group[-1].end]
        for piece, piece_start in _enforce_ceiling(raw, group[0].start, settings):
            body, start, end = _trimmed_bounds(piece, piece_start, piece_start + len(piece))
            if not body:
                continue
            page_start, page_end = _pages_for(spans, start, end)
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(document_id, index),
                    document_id=document_id,
                    text=body,
                    chunk_index=index,
                    page_start=page_start,
                    page_end=page_end,
                )
            )

    _check_invariants(chunks, document_id, settings)
    log.debug("chunker.done", document_id=document_id, chunks=len(chunks), chars=len(text))
    return chunks


def _check_invariants(
    chunks: Sequence[Chunk], document_id: str, settings: IngestionSettings
) -> None:
    """Refuse to hand back output that violates the chunker's own contract.

    An assertion about this module's correctness, not about the document. If it fires,
    the chunker has a bug and the right response is to stop before writing a corrupt
    index: a chunk above the ceiling is truncated at embedding time with no error, so
    half a passage is indexed and nobody is told.

    The other invariant worth stating -- that a page span never runs backwards -- is
    *not* checked here, because :class:`~medguard.models.Chunk` already refuses to be
    constructed with one (see its ``_check_pages`` validator) and the database refuses
    to store one (``ck_chunks_page_range_ordered``). Repeating it a third time would add
    a branch that no test can reach, and unreachable safety code is indistinguishable
    from safety code that does not work.
    """
    for chunk in chunks:
        if len(chunk.text) > settings.chunk_max_chars:
            raise ChunkingError(
                f"{document_id}: chunk {chunk.chunk_index} is {len(chunk.text)} chars, "
                f"above the {settings.chunk_max_chars} ceiling"
            )
