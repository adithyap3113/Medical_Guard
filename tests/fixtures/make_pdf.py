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
    "SYNTHETIC_VERSIONS",
    "SyntheticVersion",
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
