"""Generate the synthetic version-comparison PDFs used by the corpus tests.

Why generated and not committed
-------------------------------
Two reasons, and the second is the one that matters.

A committed binary PDF is opaque to review. Nobody diffs one, so nobody notices when it
changes, and "the fixture is wrong" becomes a very expensive thing to discover. Here the
*content* is in this file, in plain Python, where a reviewer can read what the fixture
actually says.

More importantly, these fixtures exist to test version handling -- what happens when the
corpus contains two editions of the same guidance that recommend **different drugs**.
Doing that with real documents would mean indexing contradictory live clinical guidance,
which is precisely what `data/raw/SOURCES.md` explains the corpus deliberately does not
do. Synthetic documents exercise the mechanism with no possibility of a real clinical
claim reaching a real user.

Why reportlab and not PyMuPDF
-----------------------------
PyMuPDF is the Phase 3B parser. Pulling it into Phase 3A *test infrastructure* would
make it impossible to tell later whether the ingestion pipeline or the fixtures
introduced the dependency, and it would put an AGPL-3.0 library in the tree ahead of the
decision to accept it. reportlab writes PDFs and cannot read them, which is exactly the
capability boundary wanted: these fixtures can be created without granting the test
suite the ability to parse anything.

These files never enter ``data/raw/``, ``data/sources.yaml`` or ``data/manifest.json``.
They exist for the duration of a test and are written to ``tmp_path``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

__all__ = [
    "DOCUMENT_ID_ROOT",
    "RUNNING_FOOTER",
    "RUNNING_HEADER",
    "STRUCTURED_SECTIONS",
    "SYNTHETIC_VERSIONS",
    "SyntheticVersion",
    "structured_page_body",
    "write_structured_pdf",
    "write_synthetic_pdf",
    "write_synthetic_versions",
]

#: Both fixtures describe the same logical document. The full ``document_id`` is
#: ``{root}_{version}``, so the two ids differ while the root stays comparable -- which
#: is how a version-aware retriever would recognise them as editions of one another.
DOCUMENT_ID_ROOT = "test_amr"

SHARED_TITLE = "Sample Antimicrobial Guideline"

#: Padding that pushes each fixture past ``CORPUS_MIN_PDF_BYTES`` and over onto a second
#: page. Deliberately bland: a fixture that reads like real clinical guidance is a
#: fixture someone will eventually quote as though it were.
_FILLER = (
    "This is a synthetic document generated for automated testing. It contains no real "
    "clinical guidance and must not be used for any clinical purpose. "
)


@dataclass(frozen=True)
class SyntheticVersion:
    """One edition of the synthetic guideline."""

    version: str
    recommendation: str

    @property
    def document_id(self) -> str:
        return f"{DOCUMENT_ID_ROOT}_{self.version}"

    @property
    def filename(self) -> str:
        return f"test_v{'1' if self.version == '2020' else '2'}_{self.version}.pdf"


#: Same topic, same title, deliberately conflicting recommendations. If a retriever
#: cannot keep these apart, it cannot keep two editions of a real guideline apart
#: either -- and that is the failure this pair exists to catch.
SYNTHETIC_VERSIONS: tuple[SyntheticVersion, SyntheticVersion] = (
    SyntheticVersion(
        version="2020", recommendation="recommended first-line: Drug A, 500mg twice daily"
    ),
    SyntheticVersion(
        version="2023", recommendation="recommended first-line: Drug B, 250mg twice daily"
    ),
)


def write_synthetic_pdf(path: Path, spec: SyntheticVersion, *, pages: int = 2) -> Path:
    """Write one synthetic guideline PDF and return its path.

    Byte-stable across runs for a given input: ``invariant=1`` strips the creation
    timestamp and the document id that reportlab would otherwise randomise, so the same
    fixture hashes identically every time. A fixture whose hash changed per run would
    make every sha256 assertion in the corpus tests untestable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # pageCompression=0 keeps the content streams as readable text, so a failing
    # fixture can be inspected with `strings`, and keeps the file comfortably above
    # CORPUS_MIN_PDF_BYTES without padding it with meaningless filler.
    pdf = canvas.Canvas(str(path), pagesize=LETTER, invariant=1, pageCompression=0)
    pdf.setTitle(SHARED_TITLE)

    for page in range(1, pages + 1):
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(72, 720, SHARED_TITLE)
        pdf.setFont("Helvetica", 11)
        pdf.drawString(72, 700, f"Version {spec.version} — page {page} of {pages}")

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(72, 660, spec.recommendation)

        pdf.setFont("Helvetica", 7)
        y = 630
        # Enough filler to clear CORPUS_MIN_PDF_BYTES, which exists to reject error
        # pages masquerading as documents.
        for line in range(70):
            pdf.drawString(72, y, f"{line:02d}. {_FILLER}")
            y -= 8
        pdf.showPage()

    pdf.save()
    return path


def write_synthetic_versions(directory: Path) -> dict[str, Path]:
    """Write both editions into ``directory``, keyed by full ``document_id``."""
    return {
        spec.document_id: write_synthetic_pdf(directory / spec.filename, spec)
        for spec in SYNTHETIC_VERSIONS
    }


# ---------------------------------------------------------------------------
# A structurally realistic document, for the cleaner and chunker
# ---------------------------------------------------------------------------
#
# `write_synthetic_pdf` above prints the same body on every page, which is fine for the
# version-identity tests it was built for and useless for testing a cleaner: a document
# whose every line repeats on every page has no way to distinguish furniture from
# content, and the cleaner would be right to strip all of it.
#
# What follows generates the shape the cleaner and chunker are actually specified
# against: one running header, one running footer carrying a page number that changes,
# and body text that differs page to page. It is still obviously synthetic -- the same
# rule as above applies, nothing here may read as quotable clinical guidance.

#: Printed at the top of every page. Modelled on the real thing: the CDC prints
#: "Morbidity and Mortality Weekly Report / Recommendations and Reports" on all 60 pages
#: of a guideline, which is exactly the string that must not survive into a chunk.
RUNNING_HEADER = "Synthetic Guidance Series / Recommendations and Reports"

#: Printed at the bottom of every page, with the page number substituted in. The number
#: is the point: a footer that differs on every page only registers as repeated
#: furniture once digits are normalised, and a cleaner that misses that strips nothing.
RUNNING_FOOTER = "Synthetic Guidance Series | Page {page} of {total}"

#: One section per page, so the body is distinguishable page to page and a chunk that
#: crosses a page boundary can be told apart from one that does not. Deliberately bland
#: and deliberately not clinical advice.
STRUCTURED_SECTIONS: tuple[tuple[str, str], ...] = (
    (
        "Section 1. Scope",
        "This synthetic document exists to exercise a text-processing pipeline. "
        "It states no clinical position and cites no evidence. The paragraphs below "
        "are filler of a known length so that chunk boundaries can be asserted.",
    ),
    (
        "Section 2. Definitions",
        # Explicit line breaks, not wrapped text. The hyphens have to land at the *end
        # of a rendered line* for the parser to extract "immuno-\ncompromised" -- which
        # is the exact input the de-hyphenation rule exists for. A wrapped paragraph
        # puts a space after the hyphen and quietly tests nothing.
        "An immuno-\n"
        "compromised host is defined here purely as a string used to check that a\n"
        "word broken across a line was rejoined. A beta-\n"
        "lactam is the opposite case: that hyphen is part of the term and must "
        "survive.",
    ),
    (
        "Section 3. Assessment",
        "Assessment in this synthetic document means checking that the parser reported "
        "the page number a reader would see, and that the chunker recorded a span "
        "rather than a single page when a passage runs over a page break.",
    ),
    (
        "Section 4. Management",
        "Management here is a placeholder heading. No drug, dose, route or interval "
        "appears in this section, because a fixture that reads like a prescription is "
        "a fixture somebody eventually quotes as one.",
    ),
    (
        "Section 5. Follow-up",
        "Follow-up text closing the synthetic document. The section exists so that the "
        "last page has content of its own and the final chunk is not an artefact of the "
        "document simply running out.",
    ),
)

#: Repeated to pad each page to a predictable length. Distinct per page via the section
#: number prefixed to it, so it never looks like a running line to the cleaner.
_BODY_FILLER = (
    "Paragraph {n} of section {section}. This sentence is filler with a stable length "
    "so that character-based chunk boundaries are reproducible across runs and across "
    "machines. It carries no meaning and should not be read as guidance."
)


def structured_page_body(page: int, total: int, paragraphs: int = 6) -> str:
    """The body text of one structured page, as the parser should extract it.

    Exposed so tests can assert on content without reverse-engineering the PDF: a test
    that only checks "some text came out" passes just as happily when the extraction is
    wrong.
    """
    heading, intro = STRUCTURED_SECTIONS[(page - 1) % len(STRUCTURED_SECTIONS)]
    lines = [heading, intro]
    lines += [_BODY_FILLER.format(n=n, section=page) for n in range(1, paragraphs + 1)]
    return "\n".join(lines)


def write_structured_pdf(path: Path, *, pages: int = 5, paragraphs: int = 6) -> Path:
    """Write a multi-page PDF with a running header, a running footer and real sections.

    Byte-stable for a given input, for the same reason as `write_synthetic_pdf`:
    `invariant=1` strips the timestamp and random document id that would otherwise make
    the file hash differently on every run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(path), pagesize=LETTER, invariant=1, pageCompression=0)
    pdf.setTitle("Synthetic Structured Guidance")

    _, height = LETTER
    for page in range(1, pages + 1):
        # Running header, in the top band the parser samples for candidates.
        pdf.setFont("Helvetica", 8)
        pdf.drawString(72, height - 36, RUNNING_HEADER)

        heading, intro = STRUCTURED_SECTIONS[(page - 1) % len(STRUCTURED_SECTIONS)]
        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(72, height - 108, heading)

        pdf.setFont("Helvetica", 9)
        y = height - 132
        for line in _lay_out(intro, 92):
            pdf.drawString(72, y, line)
            y -= 12

        y -= 10
        for n in range(1, paragraphs + 1):
            for line in _wrap(_BODY_FILLER.format(n=n, section=page), 92):
                pdf.drawString(72, y, line)
                y -= 12
            y -= 6

        # Running footer, in the bottom band, with the page number that makes digit
        # normalisation necessary.
        pdf.setFont("Helvetica", 8)
        pdf.drawString(72, 36, RUNNING_FOOTER.format(page=page, total=pages))
        pdf.showPage()

    pdf.save()
    return path


def _lay_out(text: str, width: int) -> list[str]:
    """Split on explicit newlines first, then wrap each piece.

    Honouring the newlines is what lets a section place a hyphen deliberately at the end
    of a rendered line; wrapping alone would put a space after it.
    """
    return [line for piece in text.split("\n") for line in _wrap(piece, width)]


def _wrap(text: str, width: int) -> list[str]:
    """Greedy wrap. reportlab has no paragraph flow on a bare canvas."""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if current and sum(len(w) + 1 for w in current) + len(word) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines
