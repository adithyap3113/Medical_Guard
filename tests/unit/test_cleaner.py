"""Text normalisation.

Every rule here is defended by what it costs retrieval when it is missing, not by how
tidy the output looks. A de-hyphenation bug makes a word unfindable; a header-stripping
bug hands BM25 a high-IDF string that carries no information; a whitespace bug moves
chunk boundaries and therefore moves citations.

Idempotence gets its own class. The pipeline can re-run a document, and a cleaner that
removes a little more on each pass would produce different chunks -- different chunk
ids, different cited pages -- for byte-identical input.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from medguard.config import IngestionSettings
from medguard.ingestion.cleaner import (
    clean_document,
    collapse_whitespace,
    dehyphenate,
    find_running_lines,
)
from medguard.ingestion.parser import PageText, TextBlock, parse
from tests.fixtures.make_pdf import RUNNING_HEADER, write_structured_pdf


@pytest.fixture
def settings() -> IngestionSettings:
    return IngestionSettings()


def _page(number: int, text: str, *, top: str = "", bottom: str = "") -> PageText:
    return PageText(
        page_number=number,
        text=text,
        top_blocks=(TextBlock(text=top, y0=10, y1=24, font_size=8),) if top else (),
        bottom_blocks=(TextBlock(text=bottom, y0=760, y1=774, font_size=8),) if bottom else (),
    )


#: Rotated so each page's body differs by *words*, not by a digit.
#:
#: This matters more than it looks. Detection normalises digits, so "Body text for page
#: 1." and "Body text for page 2." are the same key -- which is exactly what makes a
#: numbered footer detectable, and exactly what makes a naively-written fixture look
#: like furniture. A line sitting at the page edge on every page and differing only by a
#: number *is* a running footer by any reasonable definition, so the fixture has to be
#: realistic rather than the rule loosened.
#: Indexed by *page*, not by line, so no line is ever shared between two pages. Rotating
#: through the list per line instead would make page 1 line 3 identical to page 3 line 1,
#: and the cleaner would be entirely right to call that a running line.
_WORDS = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel")
_TOPICS = ("assessment", "monitoring", "referral", "follow-up", "documentation", "review")


def _body(page: int, lines: int = 6) -> list[str]:
    """Distinct body lines for one page, none of them furniture-shaped."""
    word = _WORDS[(page - 1) % len(_WORDS)]
    return [
        f"The {word} cohort section discusses {_TOPICS[i % len(_TOPICS)]} in detail."
        for i in range(lines)
    ]


def _realistic(
    number: int,
    *,
    header: str | None = None,
    footer: str | None = None,
    middle: str | None = None,
) -> PageText:
    """A page shaped like a real one: furniture at the edges, distinct body between."""
    body = _body(number)
    if middle:
        body.insert(len(body) // 2, middle)
    lines = ([header] if header else []) + body + ([footer] if footer else [])
    return _page(
        number,
        "\n".join(lines),
        top=header or "",
        bottom=footer or "",
    )


class TestDehyphenation:
    def test_a_word_broken_across_a_line_is_rejoined(self) -> None:
        assert dehyphenate("immuno-\ncompromised") == "immunocompromised"

    def test_a_capitalised_continuation_is_left_alone(self) -> None:
        """ "Table 3 -\\nSummary" is a dash between phrases, not a broken word."""
        assert dehyphenate("Table 3 -\nSummary") == "Table 3 -\nSummary"

    def test_a_paragraph_break_is_not_crossed(self) -> None:
        assert dehyphenate("word-\n\nnext paragraph") == "word-\n\nnext paragraph"

    def test_a_semantic_hyphen_survives(self) -> None:
        """ "betalactam" is not what a clinician types, so BM25 would never match it."""
        assert dehyphenate("beta-\nlactam") == "beta-lactam"

    def test_several_semantic_prefixes_survive(self) -> None:
        assert dehyphenate("non-\ninferiority") == "non-inferiority"
        assert dehyphenate("self-\nreported") == "self-reported"
        assert dehyphenate("multi-\ndrug") == "multi-drug"

    def test_a_hyphen_not_at_a_line_end_is_untouched(self) -> None:
        assert dehyphenate("first-line therapy") == "first-line therapy"

    def test_indentation_after_the_break_is_absorbed(self) -> None:
        assert dehyphenate("immuno-\n   compromised") == "immunocompromised"

    def test_several_breaks_in_one_passage(self) -> None:
        text = "immuno-\ncompromised and neutro-\npenic"
        assert dehyphenate(text) == "immunocompromised and neutropenic"


class TestWhitespace:
    def test_horizontal_runs_collapse_to_one_space(self) -> None:
        assert collapse_whitespace("a  \t  b") == "a b"

    def test_paragraph_breaks_survive(self) -> None:
        """The chunker's strongest split level. Losing it costs chunk quality."""
        assert collapse_whitespace("a\n\nb") == "a\n\nb"

    def test_runs_of_blank_lines_become_one_blank_line(self) -> None:
        assert collapse_whitespace("a\n\n\n\n\nb") == "a\n\nb"

    def test_single_newlines_survive(self) -> None:
        """Numbered recommendations and table rows stay separable."""
        assert collapse_whitespace("1. first\n2. second") == "1. first\n2. second"

    def test_trailing_spaces_are_removed(self) -> None:
        assert collapse_whitespace("a   \nb") == "a\nb"

    def test_a_lone_non_breaking_space_is_normalised(self) -> None:
        """Invisible in a diff, and it breaks any exact-match query over the phrase."""
        assert collapse_whitespace("12.5\u00a0mg") == "12.5 mg"

    def test_it_is_idempotent(self) -> None:
        once = collapse_whitespace("a  \t b\n\n\n\nc  \n")
        assert collapse_whitespace(once) == once


class TestRunningHeaders:
    def test_a_header_on_five_pages_is_stripped(self, settings: IngestionSettings) -> None:
        pages = [_realistic(n, header="MMWR Recommendations") for n in range(1, 6)]
        cleaned = clean_document(pages, settings)

        assert all("MMWR Recommendations" not in page.text for page in cleaned)
        # And the body it sat above is untouched.
        assert all(_body(page.page_number)[0] in page.text for page in cleaned)

    def test_a_footer_with_a_changing_page_number_is_stripped(
        self, settings: IngestionSettings
    ) -> None:
        """The case a naive implementation always misses: every footer differs."""
        pages = [_realistic(n, footer=f"Page {n} of 5") for n in range(1, 6)]
        cleaned = clean_document(pages, settings)

        assert all("of 5" not in page.text for page in cleaned)
        assert all(_body(page.page_number)[-1] in page.text for page in cleaned)

    def test_a_line_on_only_two_pages_is_kept(self, settings: IngestionSettings) -> None:
        """Below the threshold. A two-page document must not lose its own title."""
        pages = [
            _page(1, "Shared Title\nAlpha body."),
            _page(2, "Shared Title\nBeta body."),
            _page(3, "Different Title\nGamma body."),
        ]
        cleaned = clean_document(pages, settings)

        assert "Shared Title" in cleaned[0].text

    def test_body_text_that_repeats_mid_page_is_kept(self, settings: IngestionSettings) -> None:
        """Only the edges are furniture; a repeated line in the middle is content."""
        pages = [_realistic(n, middle="Repeated warning line.") for n in range(1, 6)]
        cleaned = clean_document(pages, settings)

        assert all("Repeated warning line." in page.text for page in cleaned)

    def test_a_line_repeating_all_over_one_page_is_kept(self, settings: IngestionSettings) -> None:
        """Twenty identical lines on a page are a table, not a running head.

        Without this guard the strip loop walks down the page removing two more
        identical lines each round and eventually deletes the document.
        """
        body = "\n".join(["Dose adjustment required."] * 20)
        pages = [_page(n, body) for n in range(1, 6)]
        cleaned = clean_document(pages, settings)

        assert all("Dose adjustment required." in page.text for page in cleaned)

    def test_stripping_can_be_turned_off(self) -> None:
        pages = [_page(n, f"MMWR Recommendations\nBody {n}.") for n in range(1, 6)]
        cleaned = clean_document(pages, IngestionSettings(strip_running_headers=False))

        assert all("MMWR Recommendations" in page.text for page in cleaned)

    def test_a_second_layer_of_furniture_is_also_stripped(
        self, settings: IngestionSettings
    ) -> None:
        """One pass strips the banner and leaves the rule beneath it on every page."""
        pages = [
            _page(
                n,
                "Publication Banner\nRecommendations and Reports\n" + "\n".join(_body(n)),
                top="Publication Banner",
            )
            for n in range(1, 6)
        ]
        cleaned = clean_document(pages, settings)

        assert all("Publication Banner" not in page.text for page in cleaned)
        assert all("Recommendations and Reports" not in page.text for page in cleaned)
        assert all(_body(page.page_number)[0] in page.text for page in cleaned)

    def test_a_short_document_is_never_stripped(self, settings: IngestionSettings) -> None:
        pages = [_page(1, "Title\nAlpha."), _page(2, "Title\nBeta.")]
        assert find_running_lines(pages) == frozenset()

    def test_layout_bands_count_as_evidence(self, settings: IngestionSettings) -> None:
        """A header the extraction ordered oddly is still positioned at the top."""
        pages = [
            _page(n, f"Body {n}.\nSynthetic Series Header", top="Synthetic Series Header")
            for n in range(1, 6)
        ]
        cleaned = clean_document(pages, settings)

        assert all("Synthetic Series Header" not in page.text for page in cleaned)


class TestPathologicalInput:
    """When detection cannot settle, the safe answer is to change nothing."""

    def test_a_document_of_near_identical_pages_is_left_alone(
        self, settings: IngestionSettings
    ) -> None:
        """Every line repeats on every page, so every line looks like furniture.

        Stripping to convergence here would peel the document two lines at a time until
        nothing was left. The cleaner detects that it is not converging and returns the
        input untouched: a surviving running header is a quality cost, a deleted
        guideline section is a correctness one.
        """
        shared = "\n".join(f"Identical line {i} on every page." for i in range(12))
        pages = [_page(n, shared) for n in range(1, 8)]

        cleaned = clean_document(pages, settings)

        assert [page.text for page in cleaned] == [collapse_whitespace(page.text) for page in pages]

    def test_that_fallback_is_still_idempotent(self, settings: IngestionSettings) -> None:
        """The fallback returns its input, so a second call reaches the same fallback."""
        shared = "\n".join(f"Identical line {i} on every page." for i in range(12))
        pages = [_page(n, shared) for n in range(1, 8)]

        once = clean_document(pages, settings)
        assert clean_document(once, settings) == once


class TestIdempotence:
    def test_cleaning_twice_changes_nothing(self, settings: IngestionSettings) -> None:
        pages = [
            _page(
                n,
                f"MMWR Recommendations\nSection {n}\nA passage with immuno-\n"
                f"compromised split across a line   and  ragged   spacing.\n\n\n"
                f"More body for page {n}.\nPage {n} of 6",
            )
            for n in range(1, 7)
        ]
        once = clean_document(pages, settings)
        twice = clean_document(once, settings)

        assert twice == once

    def test_it_holds_on_a_real_extracted_document(
        self, tmp_path: Path, settings: IngestionSettings
    ) -> None:
        """The property that matters is on parser output, not on hand-built pages."""
        pdf = write_structured_pdf(tmp_path / "structured.pdf", pages=5)
        pages = list(parse(pdf).pages())

        once = clean_document(pages, settings)
        assert clean_document(once, settings) == once

    def test_it_holds_with_stripping_disabled(self, tmp_path: Path) -> None:
        pdf = write_structured_pdf(tmp_path / "structured.pdf", pages=5)
        disabled = IngestionSettings(strip_running_headers=False)

        once = clean_document(list(parse(pdf).pages()), disabled)
        assert clean_document(once, disabled) == once


class TestOnRealExtraction:
    def test_the_running_header_does_not_survive(
        self, tmp_path: Path, settings: IngestionSettings
    ) -> None:
        pdf = write_structured_pdf(tmp_path / "structured.pdf", pages=5)
        cleaned = clean_document(list(parse(pdf).pages()), settings)

        assert all(RUNNING_HEADER not in page.text for page in cleaned)

    def test_the_numbered_footer_does_not_survive(
        self, tmp_path: Path, settings: IngestionSettings
    ) -> None:
        pdf = write_structured_pdf(tmp_path / "structured.pdf", pages=5)
        cleaned = clean_document(list(parse(pdf).pages()), settings)

        assert all("Page 3 of 5" not in page.text for page in cleaned)

    def test_the_body_sections_survive(self, tmp_path: Path, settings: IngestionSettings) -> None:
        """Stripping furniture must not take content with it."""
        pdf = write_structured_pdf(tmp_path / "structured.pdf", pages=5)
        cleaned = clean_document(list(parse(pdf).pages()), settings)

        assert "Section 1. Scope" in cleaned[0].text
        assert "Section 5. Follow-up" in cleaned[4].text

    def test_page_numbers_are_preserved_through_cleaning(
        self, tmp_path: Path, settings: IngestionSettings
    ) -> None:
        pdf = write_structured_pdf(tmp_path / "structured.pdf", pages=5)
        cleaned = clean_document(list(parse(pdf).pages()), settings)

        assert [page.page_number for page in cleaned] == [1, 2, 3, 4, 5]

    def test_the_real_hyphenation_cases_are_handled(
        self, tmp_path: Path, settings: IngestionSettings
    ) -> None:
        pdf = write_structured_pdf(tmp_path / "structured.pdf", pages=5)
        cleaned = clean_document(list(parse(pdf).pages()), settings)

        assert "immunocompromised" in cleaned[1].text
        assert "beta-lactam" in cleaned[1].text
