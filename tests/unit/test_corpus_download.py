"""Download safety.

These tests exist because of a specific Phase 1 defect: the old downloader streamed
directly to the destination path and checked the ``%PDF-`` magic bytes only after the
write finished. An interrupted transfer left a truncated file at the destination, which
the next run's size-based "already present" check then accepted as valid.

So the properties pinned down here are: the destination never holds a partial file, an
HTML page served under a .pdf URL is rejected, a failure leaves no ``.part`` behind, and
retries happen only where retrying can plausibly help.

Everything runs against ``httpx.MockTransport``. No test in this file touches a network.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from medguard.config import Settings
from medguard.corpus.download import (
    DownloadStatus,
    download_corpus,
    download_document,
)
from medguard.corpus.sources import SourceDocument, SourceList
from medguard.errors import SourceUnavailableError

URL = "https://example.invalid/a.pdf"


def _source(**overrides: object) -> SourceDocument:
    base: dict[str, object] = {
        "document_id": "cdc_example_2024",
        "title": "An Example Guideline, 2024",
        "source_url": URL,
        "source_organization": "CDC",
        "document_type": "guideline",
        "local_filename": "cdc_example_2024.pdf",
        "license_or_usage_note": "Public domain.",
    }
    base.update(overrides)
    return SourceDocument.model_validate(base)


Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


class TestSuccessfulDownload:
    def test_writes_the_file_and_reports_its_hash(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        import hashlib

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=pdf_bytes)

        with _client(handler) as client:
            outcome = download_document(_source(), settings, client=client)

        assert outcome.status is DownloadStatus.DOWNLOADED
        assert outcome.sha256 == hashlib.sha256(pdf_bytes).hexdigest()
        assert outcome.file_size_bytes == len(pdf_bytes)

        dest = settings.corpus.raw_dir / "cdc_example_2024.pdf"
        assert dest.read_bytes() == pdf_bytes

    def test_no_part_file_survives_a_success(self, settings: Settings, pdf_bytes: bytes) -> None:
        with _client(lambda _r: httpx.Response(200, content=pdf_bytes)) as client:
            download_document(_source(), settings, client=client)
        assert list(settings.corpus.raw_dir.glob("*.part")) == []

    def test_hash_is_reproducible_over_raw_bytes(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        """Anyone with sha256sum must be able to reproduce the manifest hash."""
        from medguard.corpus.manifest import sha256_file

        with _client(lambda _r: httpx.Response(200, content=pdf_bytes)) as client:
            outcome = download_document(_source(), settings, client=client)

        on_disk = sha256_file(settings.corpus.raw_dir / "cdc_example_2024.pdf")
        assert outcome.sha256 == on_disk


class TestContentRejection:
    def test_html_served_under_a_pdf_url_is_rejected(self, settings: Settings) -> None:
        """WHO IRIS and several WAFs return HTTP 200 with an HTML page.

        Content-type is not proof; the magic bytes are.
        """
        html = b"<!DOCTYPE html><html><body>Please accept cookies</body></html>"

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=html, headers={"content-type": "application/pdf"})

        with _client(handler) as client:
            outcome = download_document(_source(), settings, client=client)

        assert outcome.status is DownloadStatus.FAILED
        assert outcome.error is not None
        assert "not a PDF" in outcome.error
        assert not (settings.corpus.raw_dir / "cdc_example_2024.pdf").exists()

    def test_a_rejected_response_leaves_no_part_file(self, settings: Settings) -> None:
        with _client(lambda _r: httpx.Response(200, content=b"<html>nope</html>")) as client:
            download_document(_source(), settings, client=client)
        assert list(settings.corpus.raw_dir.glob("*.part")) == []

    def test_a_pdf_below_the_size_floor_is_rejected(self, settings: Settings) -> None:
        """A 200-byte 'PDF' is an error page or a truncated transfer, not a document."""
        with _client(lambda _r: httpx.Response(200, content=b"%PDF-1.7\nshort")) as client:
            outcome = download_document(_source(), settings, client=client)

        assert outcome.status is DownloadStatus.FAILED
        assert outcome.error is not None
        assert "below the" in outcome.error

    def test_http_404_is_reported_not_raised(self, settings: Settings) -> None:
        """One dead URL must not abort acquisition of the other eleven documents."""
        with _client(lambda _r: httpx.Response(404)) as client:
            outcome = download_document(_source(), settings, client=client)
        assert outcome.status is DownloadStatus.FAILED
        assert outcome.error is not None
        assert "404" in outcome.error


class TestTruncationHandling:
    def test_the_destination_never_holds_a_partial_file(self, settings: Settings) -> None:
        """The Phase 1 defect, pinned down.

        A stream that dies mid-transfer must leave the destination absent, not a
        truncated file that the next run's presence check would accept.
        """

        def handler(_request: httpx.Request) -> httpx.Response:
            def broken_stream() -> object:
                yield b"%PDF-1.7\n" + b"x" * 50_000
                raise httpx.ReadError("connection reset mid-transfer")

            return httpx.Response(200, content=broken_stream())  # type: ignore[arg-type]

        with _client(handler) as client:
            outcome = download_document(_source(), settings, client=client)

        assert outcome.status is DownloadStatus.FAILED
        assert not (settings.corpus.raw_dir / "cdc_example_2024.pdf").exists()
        assert list(settings.corpus.raw_dir.glob("*.part")) == []

    def test_an_existing_truncated_file_is_not_trusted_by_size_alone(
        self,
        settings: Settings,
        pdf_bytes: bytes,
    ) -> None:
        """A stale file that is not a PDF is re-downloaded rather than accepted."""
        dest = settings.corpus.raw_dir / "cdc_example_2024.pdf"
        dest.write_bytes(b"<html>" + b"padding" * 5_000)  # large, but not a PDF

        with _client(lambda _r: httpx.Response(200, content=pdf_bytes)) as client:
            outcome = download_document(_source(), settings, client=client)

        assert outcome.status is DownloadStatus.DOWNLOADED
        assert dest.read_bytes() == pdf_bytes


class TestSkipAndForce:
    def test_a_valid_existing_file_is_not_re_downloaded(
        self, settings: Settings, pdf_bytes: bytes
    ) -> None:
        dest = settings.corpus.raw_dir / "cdc_example_2024.pdf"
        dest.write_bytes(pdf_bytes)

        def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("must not fetch a document that is already present")

        with _client(handler) as client:
            outcome = download_document(_source(), settings, client=client)

        assert outcome.status is DownloadStatus.PRESENT
        assert outcome.sha256 is not None

    def test_force_re_downloads(self, settings: Settings, pdf_bytes: bytes) -> None:
        dest = settings.corpus.raw_dir / "cdc_example_2024.pdf"
        dest.write_bytes(pdf_bytes)
        replacement = b"%PDF-1.7\n" + b"replaced content\n" * 900

        with _client(lambda _r: httpx.Response(200, content=replacement)) as client:
            outcome = download_document(_source(), settings, client=client, force=True)

        assert outcome.status is DownloadStatus.DOWNLOADED
        assert dest.read_bytes() == replacement

    def test_retired_entries_are_never_fetched(self, settings: Settings) -> None:
        """A retired URL is known-dead; requesting it wastes a timeout on every run."""

        def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("must not fetch a retired document")

        source = _source(retired=True, retired_reason="The publisher withdrew the URL.")
        with _client(handler) as client:
            outcome = download_document(source, settings, client=client)

        assert outcome.status is DownloadStatus.RETIRED
        assert outcome.error == "The publisher withdrew the URL."


class TestRetryBehaviour:
    def test_a_transient_503_is_retried_and_can_succeed(
        self,
        settings: Settings,
        pdf_bytes: bytes,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("CORPUS_RETRY_INITIAL_BACKOFF_S", "0.01")
        monkeypatch.setenv("CORPUS_RETRY_MAX_BACKOFF_S", "0.02")
        from medguard.config import get_settings

        get_settings.cache_clear()
        fresh = get_settings()

        attempts = {"n": 0}

        def handler(_request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] == 1:
                return httpx.Response(503)
            return httpx.Response(200, content=pdf_bytes)

        with _client(handler) as client:
            outcome = download_document(_source(), fresh, client=client)

        assert outcome.status is DownloadStatus.DOWNLOADED
        assert attempts["n"] == 2

    def test_a_403_is_not_retried(self, settings: Settings) -> None:
        """A WAF block is a stable answer. Retrying turns a fast failure into a slow one."""
        attempts = {"n": 0}

        def handler(_request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(403, content=b"<HTML>blocked</HTML>")

        with _client(handler) as client:
            outcome = download_document(_source(), settings, client=client)

        assert outcome.status is DownloadStatus.FAILED
        assert attempts["n"] == 1

    def test_retries_are_bounded(
        self,
        settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("CORPUS_RETRY_INITIAL_BACKOFF_S", "0.01")
        monkeypatch.setenv("CORPUS_RETRY_MAX_BACKOFF_S", "0.02")
        monkeypatch.setenv("CORPUS_MAX_RETRIES", "2")
        from medguard.config import get_settings

        get_settings.cache_clear()
        fresh = get_settings()

        attempts = {"n": 0}

        def handler(_request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(503)

        with _client(handler) as client:
            outcome = download_document(_source(), fresh, client=client)

        assert outcome.status is DownloadStatus.FAILED
        # max_retries counts retries *after* the first attempt.
        assert attempts["n"] == 3


class TestDownloadCorpus:
    def test_one_failure_does_not_stop_the_others(
        self,
        settings: Settings,
        pdf_bytes: bytes,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sources = SourceList(
            documents=[
                _source(),
                _source(
                    document_id="cdc_broken_2024",
                    local_filename="cdc_broken_2024.pdf",
                    source_url="https://example.invalid/broken.pdf",
                ),
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            if "broken" in str(request.url):
                return httpx.Response(404)
            return httpx.Response(200, content=pdf_bytes)

        # download_corpus builds its own client so every document shares one connection
        # pool. Substituting the constructor is how the mock transport reaches it.
        # The real constructor is captured first: `medguard.corpus.download.httpx` is
        # the same module object as `httpx`, so calling httpx.Client() after patching
        # would re-enter the patch and recurse forever.
        real_client = httpx.Client

        def patched(**_kwargs: object) -> httpx.Client:
            return real_client(transport=httpx.MockTransport(handler), follow_redirects=True)

        monkeypatch.setattr("medguard.corpus.download.httpx.Client", patched)
        outcomes = download_corpus(sources, settings)

        statuses = {o.document_id: o.status for o in outcomes}
        assert statuses["cdc_example_2024"] is DownloadStatus.DOWNLOADED
        assert statuses["cdc_broken_2024"] is DownloadStatus.FAILED

    def test_unknown_document_id_in_only_is_an_error(self, settings: Settings) -> None:
        """A typo'd --only must fail loudly rather than silently download nothing."""
        sources = SourceList(documents=[_source()])
        with pytest.raises(SourceUnavailableError, match="unknown document ids"):
            download_corpus(sources, settings, only=["no_such_document"])
