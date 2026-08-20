"""PDF text extraction.

Scope, deliberately narrow: bytes on disk become text with a page number attached, and
nothing else. No cleaning, no chunking, no heuristics about what the text *means*. The
parser is the only module in the project that touches PyMuPDF, so replacing the
extractor -- with pdfplumber, with an OCR pass for scanned documents -- is a change to
one file rather than an archaeology project.

Three properties worth defending
--------------------------------
**Page numbers are converted exactly once.** PyMuPDF indexes pages from 0. Every reader,
every printed index and every clinician counts from 1. :func:`to_one_based_page` is the
single conversion site, which is why it is a named function with its own test rather
than a ``+ 1`` scattered through a loop: an off-by-one here renders a citation that
looks entirely plausible and points at the wrong page.

**The document is never held open past the iterator.** A ``pymupdf.Document`` holds
native memory and a file handle. Returning one to a caller means the file stays locked
until the caller remembers to close it, which on Windows makes the PDF undeletable and
turns a test-suite teardown into a permission error. :meth:`ParsedDocument.pages` opens
inside a ``with`` block and yields from within it, so the handle's lifetime is exactly
the iterator's.

**Nothing is swallowed.** Every failure becomes a :class:`ParseError` naming the
document, because PyMuPDF's own messages are frequently a bare "cannot open document"
with no filename in them, and "which of the twenty failed?" is the first question
anyone asks.

On memory
---------
Extraction is lazy per page. A 400-page guideline is roughly 1 MB of text, which is not
itself a problem -- but the cleaner needs a document-level view to find running headers,
so the pipeline does end up holding one document's pages at once. Holding *twenty*
documents at once is what laziness prevents, and that is the case that matters on an
8 GB machine.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pymupdf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from medguard.errors import ParseError
from medguard.logging import get_logger

__all__ = [
    "PageText",
    "ParsedDocument",
    "TextBlock",
    "parse",
    "to_one_based_page",
]

log = get_logger(__name__)

#: Fraction of page height treated as the header band and the footer band. 12% of a
#: US Letter page is about 95 points, comfortably more than a one-line running head and
#: comfortably less than the first paragraph of body text. Used only to decide which
#: blocks are *candidates* for the cleaner to consider; the cleaner decides what is
#: actually a header by looking for repetition across pages.
_BAND_FRACTION = 0.12

#: Cap on how many blocks per band are carried forward. A running header is one or two
#: lines. Keeping every block that happens to fall in the band would carry most of a
#: dense page's body text into memory for every page of every document, which is exactly
#: the "do not over-return" failure this bound exists to prevent.
_MAX_BAND_BLOCKS = 3


def to_one_based_page(zero_based: int) -> int:
    """Convert a PyMuPDF page index into the page number a human would cite.

    The entire reason this is a function: it is the one place in the codebase where the
    two numbering conventions meet. Inlining ``+ 1`` at each call site means the next
    person to add an extraction path has to notice the convention, and the failure when
    they do not is silent -- every citation in the document is one page early, and the
    final page becomes unreachable.
    """
    if zero_based < 0:
        msg = f"page index must be >= 0, got {zero_based}"
        raise ValueError(msg)
    return zero_based + 1


class TextBlock(BaseModel):
    """One layout block near the top or bottom of a page.

    Carries only what running-header detection needs: the text, where it sat vertically,
    and how large it was set. Font size is a secondary signal -- a running head is
    usually set smaller than body text -- and it is nearly free to record, so it is kept
    even though the current cleaner decides on repetition alone.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    y0: float = Field(description="Top edge, in points from the top of the page.")
    y1: float = Field(description="Bottom edge, in points from the top of the page.")
    font_size: float = Field(default=0.0, ge=0, description="Largest span size in the block.")


class PageText(BaseModel):
    """The text of one page, with the page number a citation would print.

    ``page_number`` is 1-based and validated as such. A ``PageText`` carrying page 0
    cannot be constructed, so an unconverted PyMuPDF index fails here rather than eight
    stages later in a rendered citation.
    """

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1, description="1-based, as printed and as cited.")
    text: str
    #: Defaulted rather than required, because the validator below derives it from
    #: ``text`` whenever a caller omits it. Making it required would force every
    #: construction site to compute ``len(text)`` and give each one its own chance to
    #: get it wrong, which is exactly what the consistency check exists to catch.
    char_count: int = Field(default=0, ge=0)
    top_blocks: tuple[TextBlock, ...] = ()
    bottom_blocks: tuple[TextBlock, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _default_char_count(cls, data: Any) -> Any:
        """Derive ``char_count`` from ``text`` unless a caller states it explicitly."""
        if isinstance(data, dict) and "char_count" not in data and "text" in data:
            text = data["text"]
            if isinstance(text, str):
                return {**data, "char_count": len(text)}
        return data

    @model_validator(mode="after")
    def _count_matches_text(self) -> PageText:
        if self.char_count != len(self.text):
            msg = f"char_count ({self.char_count}) != len(text) ({len(self.text)})"
            raise ValueError(msg)
        return self


BandBlocks = tuple[tuple[TextBlock, ...], tuple[TextBlock, ...]]


def _band_blocks(raw: dict[str, Any], page_height: float) -> BandBlocks:
    """Reduce a ``get_text("dict")`` payload to the header and footer candidates.

    Everything in the middle of the page is discarded here: it is already present in the
    plain-text extraction, and keeping a second structured copy of it would multiply the
    memory cost of a long document for no gain.
    """
    band = page_height * _BAND_FRACTION
    top: list[TextBlock] = []
    bottom: list[TextBlock] = []

    for block in raw.get("blocks", ()):
        # Type 0 is text; type 1 is an image, which has no lines to read.
        if block.get("type") != 0:
            continue
        bbox = block.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        y0, y1 = float(bbox[1]), float(bbox[3])

        pieces: list[str] = []
        largest = 0.0
        for line in block.get("lines", ()):
            for span in line.get("spans", ()):
                pieces.append(str(span.get("text", "")))
                largest = max(largest, float(span.get("size", 0.0)))
        text = " ".join(" ".join(pieces).split())
        if not text:
            continue

        entry = TextBlock(text=text, y0=y0, y1=y1, font_size=largest)
        if y1 <= band:
            top.append(entry)
        elif y0 >= page_height - band:
            bottom.append(entry)

    top.sort(key=lambda b: b.y0)
    bottom.sort(key=lambda b: b.y0)
    return tuple(top[:_MAX_BAND_BLOCKS]), tuple(bottom[-_MAX_BAND_BLOCKS:])


def _extract_page_text(page: Any, page_number: int) -> PageText:
    """Pull text and header/footer candidates from one open page.

    A named seam rather than an inline block, so the laziness of
    :meth:`ParsedDocument.pages` can be asserted by counting calls to it. A test that
    cannot observe when extraction happens cannot tell a lazy iterator from an eager one
    that returns a list.
    """
    text = str(page.get_text("text"))
    try:
        raw = page.get_text("dict")
        height = float(page.rect.height)
        top, bottom = _band_blocks(raw, height)
    except Exception as exc:  # noqa: BLE001 - layout is an optimisation, text is not
        # Losing the header candidates costs a cosmetic cleanup on this page. Losing the
        # page costs a section of a clinical guideline. Degrade, log, continue.
        log.debug("parser.layout_unavailable", page=page_number, error=str(exc))
        top, bottom = (), ()

    return PageText(page_number=page_number, text=text, top_blocks=top, bottom_blocks=bottom)


@contextmanager
def _opened(path: Path) -> Iterator[Any]:
    """Open a PDF, translating every failure into :class:`ParseError`.

    ``pymupdf.FileNotFoundError`` is *not* the builtin -- it subclasses ``RuntimeError``
    -- so an ``except FileNotFoundError`` written by reflex would not catch it. Both are
    handled, along with ``OSError`` for the permission and device cases PyMuPDF passes
    straight through.
    """
    try:
        document = pymupdf.open(path)
    except pymupdf.FileNotFoundError as exc:
        raise ParseError(f"{path.name}: file not found ({exc})") from exc
    except pymupdf.EmptyFileError as exc:
        raise ParseError(f"{path.name}: file is empty ({exc})") from exc
    except pymupdf.FileDataError as exc:
        raise ParseError(f"{path.name}: not a readable PDF ({exc})") from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise ParseError(f"{path.name}: could not be opened ({type(exc).__name__}: {exc})") from exc

    try:
        yield document
    finally:
        document.close()


class ParsedDocument:
    """A PDF that has been opened once and can be read page by page.

    Holds a path and a page count, never an open handle. Each call to :meth:`pages`
    opens the file again for the duration of the iteration. Re-opening costs a few
    milliseconds -- MuPDF parses the cross-reference table lazily -- and buys a class
    that cannot leak a file handle no matter how a caller abandons the iterator.
    """

    __slots__ = ("_page_count", "_path")

    def __init__(self, path: Path, page_count: int) -> None:
        self._path = path
        self._page_count = page_count

    @property
    def path(self) -> Path:
        return self._path

    @property
    def page_count(self) -> int:
        return self._page_count

    def pages(self) -> Iterator[PageText]:
        """Yield each page's text in order, extracting on demand.

        Lazy: nothing is read until the first ``next()``, and only one page is held at a
        time. Abandoning the iterator part-way through closes the document.
        """
        with _opened(self._path) as document:
            for index in range(document.page_count):
                page_number = to_one_based_page(index)
                try:
                    page = document.load_page(index)
                    yield _extract_page_text(page, page_number)
                except ParseError:
                    raise
                except Exception as exc:
                    raise ParseError(
                        f"{self._path.name}: page {page_number} of {self._page_count} "
                        f"could not be read ({type(exc).__name__}: {exc})"
                    ) from exc

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ParsedDocument {self._path.name!r} pages={self._page_count}>"


def parse(pdf_path: Path) -> ParsedDocument:
    """Open a PDF, validate that it has pages, and return a lazy reader over it.

    The page count is read eagerly because callers need it before deciding to iterate --
    the pipeline records it on the document row, and a zero-page file has to be rejected
    now rather than as a mysteriously empty chunk list later.

    A zero-page PDF is a :class:`ParseError` rather than an empty iterator. It is a
    structurally valid file containing nothing, which in this corpus means a truncated
    download or a placeholder the publisher put up in place of a withdrawn document.
    Treating it as "successfully ingested, zero chunks" would leave a document marked
    ``indexed`` that contributes nothing and answers no query, and nobody would look at
    it again.
    """
    path = Path(pdf_path)
    with _opened(path) as document:
        if not bool(getattr(document, "is_pdf", False)):
            # PyMuPDF is a document library, not a PDF library: it opens HTML, EPUB,
            # XPS and several image formats, and it does so *silently*. A consent
            # interstitial or a WAF challenge page saved under a .pdf URL parses
            # perfectly and yields text like "Are you a robot?", which would then be
            # chunked, embedded and returned to a clinician as though it came from the
            # CDC. The corpus download path already checks the %PDF- magic bytes; this
            # is the second lock, because the parser can be pointed at any file.
            fmt = str(document.metadata.get("format", "unknown")) if document.metadata else "?"
            raise ParseError(
                f"{path.name}: opened as {fmt}, not PDF. A non-PDF served under a .pdf "
                "URL is usually a consent page, an error page or a bot challenge."
            )

        try:
            page_count = int(document.page_count)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ParseError(f"{path.name}: page count unreadable ({exc})") from exc

    if page_count <= 0:
        raise ParseError(
            f"{path.name}: contains no pages. A structurally valid PDF with zero pages "
            "is usually a truncated download or a withdrawn-document placeholder."
        )

    log.debug("parser.opened", document=path.name, pages=page_count)
    return ParsedDocument(path, page_count)
