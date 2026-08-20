"""Chunking: the decision that determines what retrieval can ever return.

Chunk boundaries are the highest-leverage parameter in the whole system. A boundary in
the wrong place splits a recommendation from its dose, or its contraindication, and no
amount of reranking downstream can put them back together -- the passage that answers
the question simply does not exist in the index.

So the assertions here are about properties, not about output the author happened to
observe: determinism, the size envelope, page spans that match reality, and chunk ids
that make re-ingestion overwrite rather than duplicate. The snapshot test is the
exception, and it exists to catch the change nobody intended.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import pytest

from medguard.config import IngestionSettings
from medguard.errors import ChunkingError
from medguard.ingestion.chunker import chunk_document
from medguard.ingestion.cleaner import clean_document
from medguard.ingestion.parser import PageText, parse
from medguard.models import Chunk
from tests.fixtures.make_pdf import write_structured_pdf

SNAPSHOT = Path(__file__).resolve().parents[1] / "fixtures" / "snapshots" / "structured_chunks.json"

#: Deliberately tighter than the shipped defaults. At 1600 characters each page of the
#: reference fixture becomes exactly one chunk, so overlap, page spanning and short-chunk
#: merging would all be untested -- the snapshot would lock in behaviour that never runs.
SNAPSHOT_SETTINGS = IngestionSettings(
    chunk_size=400,
    chunk_overlap=100,
    chunk_min_chars=80,
    chunk_max_chars=500,
)


@pytest.fixture
def settings() -> IngestionSettings:
    return IngestionSettings()


@pytest.fixture
def structured_pages(tmp_path: Path) -> list[PageText]:
    pdf = write_structured_pdf(tmp_path / "structured.pdf", pages=5)
    return clean_document(list(parse(pdf).pages()), IngestionSettings())


def _page(number: int, text: str) -> PageText:
    return PageText(page_number=number, text=text)


def _paragraphs(count: int, *, marker: str = "p", chars: int = 300) -> str:
    """`count` paragraphs of roughly `chars` characters, with no repeated text.

    Every sentence is unique, which matters more than it sounds. An earlier version
    padded each paragraph with the same repeated filler sentence, and
    `_shared_suffix_prefix` below then reported overlaps of 400+ characters against a
    150-character budget -- not because the chunker was wrong, but because the tail of
    one chunk genuinely equalled the head of the next in text that repeats itself. A
    self-similar fixture cannot distinguish real overlap from coincidence.
    """
    paragraphs: list[str] = []
    for i in range(count):
        sentences: list[str] = []
        while sum(len(s) for s in sentences) < chars:
            j = len(sentences)
            sentences.append(f"{marker}{i} sentence {j} on assessment and onward referral. ")
        paragraphs.append("".join(sentences).strip())
    return "\n\n".join(paragraphs)


class TestDeterminism:
    def test_the_same_input_gives_byte_identical_output(
        self, structured_pages: list[PageText], settings: IngestionSettings
    ) -> None:
        first = chunk_document("doc", structured_pages, settings)
        second = chunk_document("doc", structured_pages, settings)

        assert [c.model_dump() for c in first] == [c.model_dump() for c in second]

    def test_nothing_depends_on_the_document_id(
        self, structured_pages: list[PageText], settings: IngestionSettings
    ) -> None:
        """Only the ids should change; the boundaries must not."""
        a = chunk_document("alpha", structured_pages, settings)
        b = chunk_document("beta", structured_pages, settings)

        assert [c.text for c in a] == [c.text for c in b]

    def test_it_matches_the_committed_snapshot(self, tmp_path: Path) -> None:
        """The regression net. Any boundary change shows up here as a diff.

        If this fails after a deliberate change, regenerate with
        `python -m tests.fixtures.snapshots.build` and review the diff -- the point is
        that a boundary move is *seen*, not that it is forbidden.
        """
        chunks = _snapshot_chunks(tmp_path)
        expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

        assert [c.chunk_id for c in chunks] == [e["chunk_id"] for e in expected["chunks"]]
        assert [c.text for c in chunks] == [e["text"] for e in expected["chunks"]]
        assert [(c.page_start, c.page_end) for c in chunks] == [
            (e["page_start"], e["page_end"]) for e in expected["chunks"]
        ]

    def test_the_snapshot_records_the_settings_it_was_built_with(self) -> None:
        """A snapshot that does not say how it was made cannot be regenerated."""
        expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        assert expected["settings"] == SNAPSHOT_SETTINGS.model_dump()


def _snapshot_chunks(tmp_path: Path) -> list[Chunk]:
    pdf = write_structured_pdf(tmp_path / "snapshot.pdf", pages=5)
    pages = clean_document(list(parse(pdf).pages()), SNAPSHOT_SETTINGS)
    return chunk_document("snapshot_doc", pages, SNAPSHOT_SETTINGS)


class TestSizeEnvelope:
    def test_no_chunk_exceeds_the_ceiling(self, tmp_path: Path) -> None:
        """Above it the embedding model truncates silently and indexes half a passage."""
        chunks = _snapshot_chunks(tmp_path)
        assert max(len(c.text) for c in chunks) <= SNAPSHOT_SETTINGS.chunk_max_chars

    def test_a_forced_split_happens_when_one_atom_is_too_long(self) -> None:
        """An unbroken 3,000-character table row has to go somewhere."""
        tight = IngestionSettings(
            chunk_size=200, chunk_overlap=0, chunk_min_chars=0, chunk_max_chars=200
        )
        wall = "x" * 3000
        chunks = chunk_document("doc", [_page(1, wall)], tight)

        assert len(chunks) >= 15
        assert all(len(c.text) <= 200 for c in chunks)

    def test_a_short_trailing_chunk_is_merged_into_its_neighbour(self) -> None:
        """ "Table 3. Continued" alone embeds to a near-meaningless vector."""
        loose = IngestionSettings(
            chunk_size=400, chunk_overlap=0, chunk_min_chars=200, chunk_max_chars=2000
        )
        text = _paragraphs(2, chars=350) + "\n\nTable 3. Continued"
        chunks = chunk_document("doc", [_page(1, text)], loose)

        assert all(len(c.text) >= loose.chunk_min_chars for c in chunks)
        assert "Table 3. Continued" in chunks[-1].text

    def test_merging_never_breaches_the_ceiling(self) -> None:
        """The merge is an optimisation; the ceiling is a constraint."""
        strict = IngestionSettings(
            chunk_size=400, chunk_overlap=0, chunk_min_chars=390, chunk_max_chars=420
        )
        chunks = chunk_document("doc", [_page(1, _paragraphs(6, chars=380))], strict)

        assert all(len(c.text) <= 420 for c in chunks)

    def test_a_single_short_document_still_produces_a_chunk(
        self, settings: IngestionSettings
    ) -> None:
        """Nothing to merge with. Dropping it would lose the whole document."""
        chunks = chunk_document("doc", [_page(1, "A very short guideline.")], settings)

        assert len(chunks) == 1
        assert chunks[0].text == "A very short guideline."

    def test_an_empty_document_returns_no_chunks_without_raising(
        self, settings: IngestionSettings
    ) -> None:
        """A scanned PDF has no text layer. That is a fact to report, not an exception."""
        assert chunk_document("doc", [_page(1, "   \n\n  ")], settings) == []

    def test_no_pages_at_all_returns_no_chunks(self, settings: IngestionSettings) -> None:
        assert chunk_document("doc", [], settings) == []


class TestOverlap:
    def test_adjacent_chunks_share_text(self) -> None:
        """A recommendation split across a boundary stays whole in one of the two."""
        s = IngestionSettings(
            chunk_size=400, chunk_overlap=150, chunk_min_chars=50, chunk_max_chars=800
        )
        chunks = chunk_document("doc", [_page(1, _paragraphs(8, chars=120))], s)

        shared = [_shared_suffix_prefix(a.text, b.text) for a, b in pairwise(chunks)]
        assert sum(1 for n in shared if n > 0) >= len(shared) - 1

    def test_overlap_never_exceeds_the_configured_budget(self) -> None:
        s = IngestionSettings(
            chunk_size=400, chunk_overlap=150, chunk_min_chars=50, chunk_max_chars=800
        )
        chunks = chunk_document("doc", [_page(1, _paragraphs(8, chars=120))], s)

        shared = [_shared_suffix_prefix(a.text, b.text) for a, b in pairwise(chunks)]
        assert all(n <= s.chunk_overlap for n in shared)

    def test_zero_overlap_means_no_shared_text(self) -> None:
        s = IngestionSettings(
            chunk_size=400, chunk_overlap=0, chunk_min_chars=50, chunk_max_chars=800
        )
        chunks = chunk_document("doc", [_page(1, _paragraphs(8, chars=120))], s)

        assert all(_shared_suffix_prefix(a.text, b.text) == 0 for a, b in pairwise(chunks))

    def test_a_long_paragraph_does_not_stall_the_packer(self) -> None:
        """With an atom larger than the overlap budget, a naive tail loops forever."""
        s = IngestionSettings(
            chunk_size=300, chunk_overlap=250, chunk_min_chars=0, chunk_max_chars=600
        )
        chunks = chunk_document("doc", [_page(1, _paragraphs(4, chars=290))], s)

        assert 0 < len(chunks) < 100


def _shared_suffix_prefix(left: str, right: str) -> int:
    for size in range(min(len(left), len(right)), 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


class TestPageSpans:
    def test_a_chunk_inside_one_page_has_equal_bounds(
        self, structured_pages: list[PageText], settings: IngestionSettings
    ) -> None:
        chunks = chunk_document("doc", structured_pages, settings)
        assert any(c.page_start == c.page_end for c in chunks)

    def test_a_chunk_crossing_a_page_break_records_both_pages(self, tmp_path: Path) -> None:
        """The reason this chunker exists instead of a library one."""
        chunks = _snapshot_chunks(tmp_path)
        spanning = [c for c in chunks if c.page_start != c.page_end]

        assert spanning, "no chunk spanned a page; the fixture is not exercising this"
        for chunk in spanning:
            assert chunk.page_start is not None
            assert chunk.page_end == chunk.page_start + 1

    def test_pages_are_one_based(self, structured_pages: list[PageText]) -> None:
        chunks = chunk_document("doc", structured_pages, IngestionSettings())
        assert min(c.page_start for c in chunks if c.page_start) == 1

    def test_no_page_is_ever_zero(self, tmp_path: Path) -> None:
        """The database rejects it; the chunker must never produce it."""
        chunks = _snapshot_chunks(tmp_path)
        assert all(c.page_start != 0 and c.page_end != 0 for c in chunks)

    def test_spans_stay_within_the_document(self, structured_pages: list[PageText]) -> None:
        chunks = chunk_document("doc", structured_pages, IngestionSettings())
        assert all(c.page_end is not None and c.page_end <= 5 for c in chunks)

    def test_spans_never_run_backwards(self, tmp_path: Path) -> None:
        chunks = _snapshot_chunks(tmp_path)
        assert all(
            c.page_start is not None and c.page_end is not None and c.page_end >= c.page_start
            for c in chunks
        )

    def test_the_page_span_reflects_the_text_not_the_separator(self) -> None:
        """A chunk starting at a page break belongs to the page whose text it holds."""
        pages = [_page(1, "Alpha content on page one."), _page(2, "Bravo content on page two.")]
        s = IngestionSettings(
            chunk_size=30, chunk_overlap=0, chunk_min_chars=0, chunk_max_chars=200
        )
        chunks = chunk_document("doc", pages, s)

        bravo = next(c for c in chunks if "Bravo" in c.text)
        assert bravo.page_start == 2


class TestChunkIdentity:
    def test_ids_are_position_addressed_and_zero_padded(
        self, structured_pages: list[PageText], settings: IngestionSettings
    ) -> None:
        chunks = chunk_document("cdc_example_2024", structured_pages, settings)
        assert chunks[0].chunk_id == "cdc_example_2024::0000"

    def test_padding_holds_past_a_thousand_chunks(self) -> None:
        """Lexicographic order must keep matching numeric order in a raw table scan."""
        tight = IngestionSettings(
            chunk_size=64, chunk_overlap=0, chunk_min_chars=0, chunk_max_chars=64
        )
        chunks = chunk_document("doc", [_page(1, "y" * 70_000)], tight)

        assert len(chunks) > 1000
        assert chunks[999].chunk_id == "doc::0999"
        assert chunks[1000].chunk_id == "doc::1000"

    def test_indexes_are_contiguous_from_zero(self, tmp_path: Path) -> None:
        """A gap means a chunk was dropped after its id was assigned."""
        chunks = _snapshot_chunks(tmp_path)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_ids_are_unique(self, tmp_path: Path) -> None:
        chunks = _snapshot_chunks(tmp_path)
        assert len({c.chunk_id for c in chunks}) == len(chunks)

    def test_re_chunking_edited_text_reuses_the_ids(self, settings: IngestionSettings) -> None:
        """Position-addressed ids are what make re-ingestion overwrite in place."""
        original = chunk_document("doc", [_page(1, "Give Drug A twice daily.")], settings)
        revised = chunk_document("doc", [_page(1, "Give Drug B twice daily.")], settings)

        assert original[0].chunk_id == revised[0].chunk_id
        assert original[0].text != revised[0].text


class TestInvariantGuard:
    def test_an_oversized_chunk_would_be_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The guard exists because the failure it catches is invisible downstream.

        No valid configuration can reach it -- the splitter bounds atoms at
        ``chunk_size`` and ``chunk_max_chars`` is validated to be at least that -- which
        is exactly why the bug has to be injected. The guard is insurance against a
        future change to the packer, and insurance that is never exercised is not known
        to work. An over-long chunk is truncated at embedding time with no error, so
        half a passage would be indexed and nobody told.
        """
        from medguard.ingestion import chunker

        monkeypatch.setattr(chunker, "_atomise", lambda text, limit: [chunker._Atom(text, 0)])
        monkeypatch.setattr(chunker, "_enforce_ceiling", lambda text, start, s: [(text, start)])
        s = IngestionSettings(
            chunk_size=200, chunk_overlap=0, chunk_min_chars=0, chunk_max_chars=200
        )

        with pytest.raises(ChunkingError, match="ceiling"):
            chunk_document("doc", [_page(1, _paragraphs(6, chars=200))], s)

    def test_an_inverted_page_span_cannot_be_emitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A citation rendering as "pages 43-41" is stopped at the model boundary.

        The chunker does not re-check this itself: `Chunk` refuses to construct with an
        inverted span, so a bug in `_pages_for` fails here rather than reaching storage.
        Asserting it from the outside is what keeps that guarantee from being quietly
        removed along with the validator.
        """
        from medguard.ingestion import chunker

        monkeypatch.setattr(chunker, "_pages_for", lambda spans, start, end: (9, 3))

        with pytest.raises(ValueError, match="page_end"):
            chunk_document("doc", [_page(1, "Some guideline text to chunk.")], IngestionSettings())

    def test_section_is_left_null_rather_than_guessed(
        self, structured_pages: list[PageText], settings: IngestionSettings
    ) -> None:
        """Standing rule: a field that cannot be derived from the source stays null.

        A wrong section label is worse than an absent one, because it is rendered in a
        citation as though it had been read off the document.
        """
        chunks = chunk_document("doc", structured_pages, settings)
        assert all(c.section is None for c in chunks)

    def test_token_count_is_left_null_until_a_tokeniser_exists(
        self, structured_pages: list[PageText], settings: IngestionSettings
    ) -> None:
        chunks = chunk_document("doc", structured_pages, settings)
        assert all(c.token_count is None for c in chunks)


class TestContentPreservation:
    def test_no_text_is_silently_dropped(self, settings: IngestionSettings) -> None:
        """Chunks may overlap, but between them they must cover the document."""
        text = _paragraphs(10, marker="para", chars=200)
        chunks = chunk_document("doc", [_page(1, text)], settings)

        joined = " ".join(c.text for c in chunks)
        for i in range(10):
            assert f"para{i} sentence 0" in joined

    def test_the_body_of_every_page_survives(self, structured_pages: list[PageText]) -> None:
        chunks = chunk_document("doc", structured_pages, IngestionSettings())
        joined = " ".join(c.text for c in chunks)

        for section in ("Section 1. Scope", "Section 3. Assessment", "Section 5. Follow-up"):
            assert section in joined

    def test_chunks_carry_no_leading_or_trailing_whitespace(self, tmp_path: Path) -> None:
        chunks = _snapshot_chunks(tmp_path)
        assert all(c.text == c.text.strip() for c in chunks)
