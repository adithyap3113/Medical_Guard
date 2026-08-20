"""Text normalisation between extraction and chunking.

Pure functions over :class:`~medguard.ingestion.parser.PageText`. No I/O, no clock, no
model. Given the same pages and the same settings this module returns the same output
forever, which is what makes the chunk boundaries downstream reproducible.

What it fixes, and why each one matters for retrieval
-----------------------------------------------------
**Line-break hyphens.** A PDF renders ``immunocompromised`` split across two lines as
``immuno-`` / ``compromised``. Extracted verbatim, neither half is the word, so BM25
cannot match a query for it and the dense model embeds two fragments. Rejoining is not
cosmetic; it is the difference between a passage being findable and not.

**Running headers and footers.** ``MMWR / Recommendations and Reports`` appears on all
60 pages of a CDC document. Left in, it is repeated 60 times in one document and almost
nowhere else in the corpus, which is precisely the shape BM25's IDF weighting rewards:
the least informative string in the document becomes one of its highest-scoring terms.
It also lands in the middle of chunk text, splitting sentences that ran across the page
break.

**Whitespace.** Extracted text is full of runs of spaces used for visual alignment.
Collapsing them shortens chunks without losing anything, and makes chunk boundaries
depend on the words rather than on the typesetting.

What it deliberately does not do
--------------------------------
It does not detect headings, and it does not reflow paragraphs into single lines. Both
would destroy structure the chunker uses: paragraph breaks are the first level of the
split hierarchy, and the single newlines inside a paragraph are the second. A cleaner
that "tidies" text into one long line hands the chunker nothing to split on but spaces.

Idempotence
-----------
``clean_document(clean_document(pages)) == clean_document(pages)``, asserted in the test
suite. This is not a nicety: the pipeline may retry a document, and a cleaner that
shortens text a little more on every pass would produce different chunks -- and
therefore different citations -- for the same input depending on how many times it ran.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

from medguard.config import IngestionSettings
from medguard.ingestion.parser import PageText, TextBlock
from medguard.logging import get_logger

__all__ = [
    "clean_document",
    "collapse_whitespace",
    "dehyphenate",
    "find_running_lines",
]

log = get_logger(__name__)

#: How many pages a line must appear on before it counts as a running header or footer.
#: Three is the smallest number that distinguishes "repeated furniture" from "a heading
#: that happens to appear on the facing page too". Below it, a two-page document's title
#: would be stripped from both pages and the document would lose its own name.
_MIN_REPEATS = 3

#: How far into a page a candidate may sit and still be treated as furniture. A running
#: head is the first line; a footer is the last. Widening this to "anywhere on the page"
#: is how a cleaner starts deleting body text that happens to repeat.
_EDGE_LINES = 2

#: Guard on the strip-until-nothing-new loop. Not a tuning knob: the loop is proved to
#: terminate by the fact that every continuing round removes at least one line, so
#: reaching this bound means that reasoning is wrong and the log line below is the bug
#: report. Ten is far above any real document's stack of repeated furniture.
_MAX_STRIP_ROUNDS = 10

#: How many times a line may appear on a single page and still be furniture. Two, not
#: one, because a document title is legitimately printed as both the running head and
#: the running foot of the same page. Anything more frequent is body text with a
#: repeating shape, and stripping it would remove content.
_MAX_OCCURRENCES_PER_PAGE = 2

#: Digits are replaced before counting so that "Page 4 of 60" and "Page 5 of 60" are
#: recognised as the same running footer. Without this, every page's footer is unique
#: and none of them ever reaches the repeat threshold -- which is the single most common
#: way running-footer stripping silently does nothing.
_DIGITS = re.compile(r"\d+")

#: A hyphen at end of line, followed by a lowercase continuation. The lowercase
#: requirement is what keeps "Category A -\nBacterial" from being welded together.
#:
#: The two non-ASCII members are U+2010 HYPHEN and U+2011 NON-BREAKING HYPHEN, written
#: as escapes rather than as the characters themselves: they are visually identical to
#: an ordinary "-" in every editor, so a literal here would look like a typo to the next
#: reader and be impossible to verify in review. Both appear in typeset PDFs.
_LINE_BREAK_HYPHEN = re.compile(r"(?<=\w)([-\u2010\u2011])[ \t]*\n[ \t]*(?=[a-z])")

#: Prefixes whose hyphen carries meaning rather than typesetting. ``beta-lactam`` and
#: ``non-inferiority`` are the terms clinicians type and the terms BM25 must match;
#: joining them into ``betalactam`` makes an exact-match query miss the one passage that
#: answers it. Everything not on this list is joined, which is the right default --
#: ``immuno-`` / ``compromised`` is one word broken by a line, not a compound.
#:
#: Deliberately short. Every entry is a claim that a hyphen is semantic, and a long
#: speculative list would start preserving the line-break hyphens this function exists
#: to remove. Known limitation: a genuine compound whose prefix is not listed loses its
#: hyphen. That degrades an exact-match query for that one term; the alternative --
#: leaving every broken word broken -- degrades every query.
_KEEP_HYPHEN_PREFIXES = frozenset(
    {
        "alpha",
        "anti",
        "beta",
        "co",
        "cross",
        "multi",
        "non",
        "post",
        "pre",
        "self",
    }
)

#: Any whitespace except a newline. Spelled as a double negative rather than as a
#: list of space characters because PDFs are full of exotic ones -- non-breaking,
#: figure, narrow no-break, ideographic -- and an explicit list silently misses
#: whichever one the typesetter happened to use.
_HORIZONTAL_RUN = re.compile(r"[^\S\n]{2,}")
_TRAILING_SPACE = re.compile(r"[^\S\n]+(?=\n)")
_BLANK_RUN = re.compile(r"\n{3,}")


def _normalise_key(line: str) -> str:
    """Reduce a line to the form used for counting repeats across pages."""
    return _DIGITS.sub("#", " ".join(line.split())).casefold()


def dehyphenate(text: str) -> str:
    """Rejoin words broken across a line by a typesetter's hyphen.

    Two guards keep this from damaging text. A capitalised continuation is left alone,
    because ``Table 3 -\\nSummary`` is a dash between phrases rather than a broken word.
    A blank line between the halves is left alone too: a paragraph boundary means the
    hyphen ended a line of its own, not a word.
    """

    def join(match: re.Match[str]) -> str:
        prefix = text[: match.start()]
        # The stem immediately before the hyphen decides whether it was semantic.
        stem = re.split(r"[^\w]", prefix)[-1].casefold() if prefix else ""
        return match.group(1) if stem in _KEEP_HYPHEN_PREFIXES else ""

    # Paragraph breaks are excluded by the pattern requiring a single newline, so a
    # hyphen followed by a blank line is never joined.
    return _LINE_BREAK_HYPHEN.sub(join, text)


def collapse_whitespace(text: str) -> str:
    """Squeeze horizontal runs to one space and blank-line runs to one blank line.

    Single newlines survive on purpose. They are where a PDF's visual lines were, they
    are how numbered recommendations and table rows stay separable, and they are the
    second level of the chunker's split hierarchy. Reflowing them into spaces reads
    better to a human and leaves the chunker with nothing but word boundaries to cut on.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # A *lone* non-breaking space survives the run-collapsing pass below, and it is
    # invisible in a diff while breaking any exact-match query over the phrase it sits
    # in. Normalise it explicitly rather than hoping it arrives in a run.
    text = text.replace("\u00a0", " ")
    text = _HORIZONTAL_RUN.sub(" ", text)
    text = _TRAILING_SPACE.sub("", text)
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()


def _edge_candidates(page: PageText) -> tuple[list[str], list[str]]:
    """The lines on this page that could be furniture: top few and bottom few.

    Draws on two sources at once. The plain-text edge lines are what stripping actually
    operates on, and the parser's layout bands are the stronger evidence -- "physically
    at the top of the page" beats "first in the extraction order", and it still works on
    a PDF whose reading order is scrambled. Neither alone is enough, so both are counted.
    """
    lines = [line for line in page.text.splitlines() if line.strip()]
    top = list(lines[:_EDGE_LINES])
    bottom = list(lines[-_EDGE_LINES:])

    # Both sources, not one or the other. A block is the stronger signal but its text is
    # the block's lines joined, so a two-line running head produces a key that matches no
    # single line of the extraction -- and stripping works on lines. Counting both means
    # detection gets the position evidence and stripping gets a key it can actually use.
    top += [b.text for b in page.top_blocks]
    bottom += [b.text for b in page.bottom_blocks]
    return top, bottom


def find_running_lines(pages: Sequence[PageText]) -> frozenset[str]:
    """Normalised keys of lines that repeat at the edge of at least three pages.

    The first of the two passes. Detection needs the whole document: a single page
    cannot tell a running head from a heading, and that is the entire reason this module
    takes a document rather than a page.
    """
    if len(pages) < _MIN_REPEATS:
        # Nothing can repeat often enough, and a short document is exactly where
        # over-eager stripping does the most damage.
        return frozenset()

    seen: Counter[str] = Counter()
    for page in pages:
        top, bottom = _edge_candidates(page)
        # How often each line occurs on *this* page, which is the discriminator below.
        occurrences: Counter[str] = Counter(
            _normalise_key(line) for line in page.text.splitlines() if line.strip()
        )

        # Per page, not per occurrence: a line printed twice on one page is not thereby
        # a running header.
        for key in {_normalise_key(line) for line in (*top, *bottom) if line.strip()}:
            if occurrences[key] > _MAX_OCCURRENCES_PER_PAGE:
                # Repeated *body*, not furniture. A running head appears once at the top
                # and a footer once at the bottom; a string that occurs twenty times on
                # the same page is a table row, a list marker or a boilerplate paragraph.
                # Without this the strip loop walks down the page removing two more
                # identical lines every round and eventually deletes the document.
                continue
            seen[key] += 1

    return frozenset(key for key, count in seen.items() if count >= _MIN_REPEATS)


def _strip_running_lines(text: str, running: frozenset[str]) -> str:
    """Remove matching lines from the head and tail of one page's text.

    Only the edges are considered. A string that is genuinely a running header will be
    at an edge on every page; one that appears mid-page is body text that happens to
    match, and deleting it would remove content while looking like a formatting fix.
    """
    lines = text.splitlines()

    def drop_from(indices: Iterable[int]) -> None:
        checked = 0
        for i in indices:
            if not lines[i].strip():
                continue  # blank lines do not count against the edge budget
            if checked >= _EDGE_LINES:
                return
            checked += 1
            if _normalise_key(lines[i]) in running:
                lines[i] = ""

    drop_from(range(len(lines)))
    drop_from(range(len(lines) - 1, -1, -1))
    return "\n".join(lines)


def _surviving_blocks(
    blocks: tuple[TextBlock, ...], running: frozenset[str]
) -> tuple[TextBlock, ...]:
    """Drop the band blocks that were just stripped from the text.

    This is what makes the whole function idempotent. If the bands still advertised the
    headers after they had been removed, a second pass would detect them again -- find
    nothing to strip, but only by luck. Filtering them means the second pass genuinely
    has nothing to find, and the property holds by construction rather than by accident.
    """
    return tuple(b for b in blocks if _normalise_key(b.text) not in running)


def _strip_to_fixed_point(
    texts: list[str], pages: Sequence[PageText]
) -> tuple[list[str], frozenset[str]]:
    """Strip running lines repeatedly until a round finds nothing new.

    One pass is not enough, and the reason is worth stating because it is not obvious.
    Removing the running head uncovers whatever was beneath it. A CDC page whose first
    two lines are the publication banner and the section rule leaves the section rule
    exposed once the banner goes, and a single pass would strip the banner from every
    page and leave the rule on every page -- furniture, repeated 60 times, doing exactly
    the damage to IDF that this function exists to prevent.

    It also buys idempotence outright. The loop exits only when a round detects nothing
    new, so the returned text is by definition a fixed point: running
    :func:`clean_document` on its own output cannot strip anything further. That is the
    property the pipeline depends on when it retries a document, and proving it by
    construction is much better than hoping one pass happened to be enough.

    Termination is guaranteed: every round that continues has removed at least one line,
    and there are finitely many lines. The round cap is a guard against a bug in that
    reasoning, not against the data.
    """
    working = list(texts)
    all_running: set[str] = set()
    round_number = 0

    for round_number in range(_MAX_STRIP_ROUNDS):  # noqa: B007 - used after the loop
        probe = [
            page.model_copy(update={"text": text, "char_count": len(text)})
            for page, text in zip(pages, working, strict=True)
        ]
        all_running |= find_running_lines(probe)
        if not all_running:
            break

        # Strip with everything known so far, not only what this round newly found. A
        # key discovered in round 1 can match a line that only reaches the page edge in
        # round 3, once the lines above it have gone; stripping with just the new keys
        # leaves those behind and the loop stops one round too early.
        stripped = [_strip_running_lines(text, frozenset(all_running)) for text in working]
        if stripped == working:
            # Nothing moved. This is the fixed point, and it is what makes a second call
            # to clean_document a no-op.
            break
        working = stripped
    else:
        # Not converged inside the budget. That means detection keeps finding new
        # "furniture" as the pages shrink, which on real input does not happen and on
        # pathological input means the loop is eating the document a line at a time.
        #
        # Strip nothing at all rather than hand back a half-eaten document. Losing a
        # running header costs some IDF quality; losing body text loses clinical content
        # and nothing downstream can tell that it happened. This also keeps the function
        # idempotent in the failure case: the fallback is "return the input", and
        # re-running on that input reaches the same fallback.
        log.warning(
            "cleaner.strip_abandoned_no_fixed_point",
            rounds=_MAX_STRIP_ROUNDS,
            detected=len(all_running),
            detail="running-line detection did not converge; left the text unmodified",
        )
        return list(texts), frozenset()

    if all_running:
        log.debug(
            "cleaner.running_lines_detected",
            count=len(all_running),
            pages=len(working),
            rounds=round_number,
        )
    return working, frozenset(all_running)


def clean_document(pages: Sequence[PageText], settings: IngestionSettings) -> list[PageText]:
    """Normalise a whole document's pages.

    Document-level rather than per-page, because running-header detection needs to see
    every page and a per-page function cannot. This holds one document's pages in memory
    at once, which for a 400-page guideline is roughly a megabyte of text -- bounded, and
    far cheaper than reading every PDF twice.

    Order is load-bearing. Headers are stripped first, while the text still has the line
    structure they were detected in. De-hyphenation runs next, while line breaks still
    exist to be rejoined across. Whitespace collapse runs last, because it is the step
    that destroys the information the other two depend on.
    """
    page_list = list(pages)
    texts = [page.text for page in page_list]
    running: frozenset[str] = frozenset()

    if settings.strip_running_headers:
        texts, running = _strip_to_fixed_point(texts, page_list)

    return [
        PageText(
            page_number=page.page_number,
            text=collapse_whitespace(dehyphenate(text)),
            top_blocks=_surviving_blocks(page.top_blocks, running),
            bottom_blocks=_surviving_blocks(page.bottom_blocks, running),
        )
        for page, text in zip(page_list, texts, strict=True)
    ]
