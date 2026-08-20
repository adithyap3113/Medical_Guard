"""The real corpus, parsed and chunked.

Every other test uses generated fixtures, which are clean by construction: predictable
fonts, no scanned pages, no two-column layouts, no 148-page tables of contents. Real
public-health PDFs are none of those things, and the failures that matter -- a document
that yields no text at all, a page span that runs off the end of the document, a memory
profile that only appears at 148 pages -- are invisible against a fixture.

Deliberately parse-and-chunk only. Embedding the whole corpus is a separate, much longer
operation, and what embedding would add here is already covered by
`test_ingestion_end_to_end.py`.

Marked ``integration`` because it needs the downloaded corpus on disk; it skips cleanly
when the corpus is absent, so a fresh clone does not fail here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import psutil
import pytest

from medguard.config import Settings, get_settings
from medguard.corpus.manifest import load_manifest
from medguard.errors import ParseError
from medguard.ingestion.chunker import chunk_document
from medguard.ingestion.cleaner import clean_document
from medguard.ingestion.parser import parse
from medguard.models import Chunk

pytestmark = pytest.mark.integration

#: Ceiling for the whole parse-and-chunk pass over the corpus. The machine this targets
#: has roughly 0.8 GB available with an editor and Docker running, and ingestion has to
#: coexist with Qdrant and an ONNX session. A parser holding every page of every document
#: would sail past this; one that streams pages does not come close.
PEAK_RSS_LIMIT_MB = 600


@dataclass(frozen=True)
class ParsedDoc:
    """What one real document produced.

    A typed record rather than ``dict[str, object]``: the dict version needs a cast at
    every use, and the casts are exactly where a test quietly stops asserting anything.
    """

    document_id: str
    page_count: int
    chunks: list[Chunk]

    @property
    def chars(self) -> int:
        return sum(len(c.text) for c in self.chunks)


@dataclass(frozen=True)
class CorpusRun:
    """The whole pass, plus the memory it cost."""

    documents: list[ParsedDoc] = field(default_factory=list)
    baseline_rss_mb: float = 0.0
    peak_rss_mb: float = 0.0


@pytest.fixture(scope="module")
def real_settings() -> Settings:
    """Settings pointed at the corpus that is actually committed.

    conftest redirects the corpus at a temporary directory for every test, which is
    correct everywhere else and is exactly what this module needs to undo.
    """
    for name in ("CORPUS_DATA_DIR", "CORPUS_SOURCES_FILE", "CORPUS_MANIFEST_FILE"):
        os.environ.pop(name, None)
    get_settings.cache_clear()
    settings = get_settings()

    if not settings.corpus.manifest_path.is_file():
        pytest.skip("no manifest; run `medguard corpus download` first")
    return settings


@pytest.fixture(scope="module")
def parsed_corpus(real_settings: Settings) -> CorpusRun:
    """Parse and chunk every document present on disk, once, recording peak RSS."""
    manifest = load_manifest(real_settings.corpus.manifest_path)
    process = psutil.Process()
    baseline_mb = process.memory_info().rss / 1024**2
    peak_mb = baseline_mb

    documents: list[ParsedDoc] = []
    for entry in manifest.documents:
        path = real_settings.corpus.raw_dir / entry.local_filename
        if not path.is_file():
            continue

        parsed = parse(path)
        pages = clean_document(list(parsed.pages()), real_settings.ingestion)
        chunks = chunk_document(entry.document_id, pages, real_settings.ingestion)

        peak_mb = max(peak_mb, process.memory_info().rss / 1024**2)
        documents.append(
            ParsedDoc(
                document_id=entry.document_id,
                page_count=parsed.page_count,
                chunks=chunks,
            )
        )

    if not documents:
        pytest.skip("no corpus PDFs on disk; run `medguard corpus download` first")

    return CorpusRun(documents=documents, baseline_rss_mb=baseline_mb, peak_rss_mb=peak_mb)


class TestEveryDocumentParses:
    def test_all_of_them_produced_text(self, parsed_corpus: CorpusRun) -> None:
        """A document yielding nothing is a scanned PDF with no text layer.

        It would ingest "successfully" with zero chunks and answer no question ever asked
        of it, which is the kind of silent hole a corpus accumulates without anyone
        noticing.
        """
        empty = [d.document_id for d in parsed_corpus.documents if not d.chars]
        assert empty == [], f"documents extracted no text: {empty}"

    def test_all_of_them_produced_chunks(self, parsed_corpus: CorpusRun) -> None:
        for doc in parsed_corpus.documents:
            assert doc.chunks, f"{doc.document_id} produced no chunks"

    def test_a_corrupt_document_would_have_been_noticed(self, real_settings: Settings) -> None:
        """The parser rejects a non-PDF, so a clean pass above means the files are real."""
        with pytest.raises(ParseError):
            parse(real_settings.corpus.raw_dir / "__definitely_not_here__.pdf")


class TestPageAttribution:
    def test_every_span_is_inside_its_document(self, parsed_corpus: CorpusRun) -> None:
        """A citation pointing past the last page is worse than one pointing nowhere."""
        for doc in parsed_corpus.documents:
            for chunk in doc.chunks:
                assert chunk.page_start is not None, f"{doc.document_id}: chunk lost its page"
                assert chunk.page_end is not None
                assert 1 <= chunk.page_start <= doc.page_count, chunk.chunk_id
                assert 1 <= chunk.page_end <= doc.page_count, chunk.chunk_id

    def test_no_span_runs_backwards(self, parsed_corpus: CorpusRun) -> None:
        for doc in parsed_corpus.documents:
            for chunk in doc.chunks:
                assert chunk.page_start is not None
                assert chunk.page_end is not None
                assert chunk.page_end >= chunk.page_start

    def test_some_chunks_really_do_span_pages(self, parsed_corpus: CorpusRun) -> None:
        """If none did, the page-span machinery would be untested against real prose."""
        spanning = [
            chunk
            for doc in parsed_corpus.documents
            for chunk in doc.chunks
            if chunk.page_start != chunk.page_end
        ]
        assert spanning


class TestChunkQuality:
    def test_no_chunk_exceeds_the_configured_ceiling(
        self, parsed_corpus: CorpusRun, real_settings: Settings
    ) -> None:
        """Above it the embedder truncates silently and half the passage is unindexed."""
        limit = real_settings.ingestion.chunk_max_chars
        for doc in parsed_corpus.documents:
            for chunk in doc.chunks:
                assert len(chunk.text) <= limit, f"{chunk.chunk_id} is {len(chunk.text)} chars"

    def test_chunk_ids_are_unique_across_the_corpus(self, parsed_corpus: CorpusRun) -> None:
        """Two chunks sharing an id means one silently overwrites the other in Qdrant."""
        ids = [chunk.chunk_id for doc in parsed_corpus.documents for chunk in doc.chunks]
        assert len(set(ids)) == len(ids)

    def test_chunking_is_reproducible_on_a_real_document(
        self, real_settings: Settings, parsed_corpus: CorpusRun
    ) -> None:
        """Determinism is proved on fixtures; confirmed here on a real government PDF."""
        manifest = load_manifest(real_settings.corpus.manifest_path)
        entry = next(
            e
            for e in manifest.documents
            if (real_settings.corpus.raw_dir / e.local_filename).is_file()
        )
        path = real_settings.corpus.raw_dir / entry.local_filename

        def run() -> list[Chunk]:
            pages = clean_document(list(parse(path).pages()), real_settings.ingestion)
            return chunk_document(entry.document_id, pages, real_settings.ingestion)

        assert [c.model_dump() for c in run()] == [c.model_dump() for c in run()]


class TestMemoryProfile:
    def test_peak_rss_stays_under_the_ceiling(self, parsed_corpus: CorpusRun) -> None:
        """The number this project's hardware constraint actually turns on."""
        peak = parsed_corpus.peak_rss_mb
        print(f"\npeak RSS during corpus parse+chunk: {peak:.0f} MB (limit {PEAK_RSS_LIMIT_MB})")
        assert peak < PEAK_RSS_LIMIT_MB

    def test_the_whole_corpus_is_reported(self, parsed_corpus: CorpusRun) -> None:
        docs = parsed_corpus.documents
        total_chunks = sum(len(d.chunks) for d in docs)
        total_pages = sum(d.page_count for d in docs)
        print(f"\n{len(docs)} documents, {total_pages} pages, {total_chunks} chunks")
        assert docs
