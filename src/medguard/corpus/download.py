"""Fetching source PDFs safely.

The failure this module exists to prevent
-----------------------------------------
The Phase 1 downloader streamed straight to the destination path and validated the
``%PDF-`` magic bytes only after the write completed. Interrupt it -- Ctrl-C, a dropped
connection, a full disk -- and a truncated file is left at the destination. The next
run's "already present, skip" check saw a file above its size floor and reported
success. A corrupt PDF then flowed into ingestion, and the symptom surfaced three
phases later as unexplained retrieval gaps.

So the contract here is:

1. Stream to ``<name>.pdf.part``; the destination path never holds a partial file.
2. Validate the magic bytes on the *first* bytes received, so an HTML error page is
   rejected in one chunk rather than after a full download.
3. Validate the total size after the stream closes.
4. Hash while streaming -- one pass over the bytes, not two.
5. ``Path.replace`` onto the destination only after all of the above pass. That call is
   atomic on both POSIX and Windows.
6. Delete the ``.part`` file on *any* exception, including ``KeyboardInterrupt``. This
   uses ``try/finally`` with a success flag rather than ``except``, so cleanup also
   covers exceptions that are not ``Exception`` subclasses.

Retries are bounded and only cover failures that retrying can plausibly fix. An HTTP
403 from a WAF is not one of them: the same request will keep failing, and retrying it
turns a fast clear failure into a slow one.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from medguard.config import CorpusSettings, Settings, get_settings
from medguard.corpus.sources import SourceDocument, SourceList
from medguard.errors import SourceUnavailableError
from medguard.logging import get_logger

__all__ = [
    "DownloadOutcome",
    "DownloadStatus",
    "download_corpus",
    "download_document",
]

log = get_logger(__name__)

_PDF_MAGIC = b"%PDF-"
_CHUNK_BYTES = 64 * 1024

#: Status codes worth a second attempt: transient overload, throttling, gateway blips.
#: 403 and 404 are deliberately absent -- those are stable answers, not hiccups.
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class DownloadStatus(StrEnum):
    """What happened to one document."""

    DOWNLOADED = "downloaded"
    PRESENT = "present"
    RETIRED = "retired"
    FAILED = "failed"


class DownloadOutcome(BaseModel):
    """Result for one document. Carries the hash so the manifest can be built without
    re-reading every file from disk."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    local_filename: str
    status: DownloadStatus
    file_size_bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = None
    error: str | None = None
    elapsed_s: float = Field(default=0.0, ge=0)

    @property
    def ok(self) -> bool:
        """True when the document is usable, however it got there."""
        return self.status in (DownloadStatus.DOWNLOADED, DownloadStatus.PRESENT)


class _ContentRejected(Exception):
    """The response arrived but is not the document we asked for. Not retryable."""


def _browser_headers(settings: CorpusSettings) -> dict[str, str]:
    """Headers that reach public government PDFs.

    Not cosmetic: cdc.gov returns HTTP 403 with a 429-byte HTML block page to a request
    carrying only a User-Agent and Accept. The full navigation header set below is what
    the same URLs answer with ``application/pdf``. Verified against both
    ``www.cdc.gov`` and ``stacks.cdc.gov`` before this list was fixed.
    """
    return {
        "User-Agent": settings.user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
    }


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    # TransportError covers timeouts, connection resets and DNS failures.
    return isinstance(exc, (httpx.TransportError, httpx.StreamError))


def sha256_bytes_of(path: Path) -> tuple[int, str]:
    """Return ``(size, sha256)`` for an existing file, streaming rather than slurping."""
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            hasher.update(chunk)
            size += len(chunk)
    return size, hasher.hexdigest()


def _looks_like_a_pdf(path: Path, *, min_bytes: int) -> bool:
    try:
        if path.stat().st_size < min_bytes:
            return False
        with path.open("rb") as handle:
            return handle.read(len(_PDF_MAGIC)) == _PDF_MAGIC
    except OSError:
        return False


def _stream_to_part(
    client: httpx.Client,
    url: str,
    part: Path,
    *,
    min_bytes: int,
    headers: dict[str, str],
) -> tuple[int, str]:
    """Download ``url`` into ``part``, validating and hashing as bytes arrive.

    Returns ``(size, sha256)``. Raises ``_ContentRejected`` when the response is not a
    usable PDF, and lets httpx exceptions propagate for the retry layer to classify.
    """
    hasher = hashlib.sha256()
    written = 0
    head = b""

    with client.stream("GET", url, headers=headers) as response:
        response.raise_for_status()
        with part.open("wb") as handle:
            for chunk in response.iter_bytes(chunk_size=_CHUNK_BYTES):
                if len(head) < len(_PDF_MAGIC):
                    head += chunk[: len(_PDF_MAGIC) - len(head)]
                    if len(head) == len(_PDF_MAGIC) and head != _PDF_MAGIC:
                        # An HTML consent page or WAF block. Stop now rather than
                        # spending bandwidth on something we will discard.
                        msg = (
                            f"response is not a PDF (starts with {head!r}); "
                            f"content-type {response.headers.get('content-type', '?')!r}"
                        )
                        raise _ContentRejected(msg)
                handle.write(chunk)
                hasher.update(chunk)
                written += len(chunk)

    if head != _PDF_MAGIC:
        msg = f"response is not a PDF (starts with {head!r})"
        raise _ContentRejected(msg)
    if written < min_bytes:
        msg = f"downloaded {written} bytes, below the {min_bytes}-byte floor"
        raise _ContentRejected(msg)
    return written, hasher.hexdigest()


def download_document(
    document: SourceDocument,
    settings: Settings | None = None,
    *,
    client: httpx.Client | None = None,
    force: bool = False,
) -> DownloadOutcome:
    """Fetch one document. Never raises -- failures are reported, not thrown.

    Collecting failures rather than raising lets a run of twelve documents report all
    of its problems once, instead of stopping at the first and hiding the rest.
    """
    s = (settings if settings is not None else get_settings()).corpus
    started = time.perf_counter()
    dest = s.raw_dir / document.local_filename
    part = dest.with_suffix(dest.suffix + ".part")

    def elapsed() -> float:
        return time.perf_counter() - started

    if document.retired:
        return DownloadOutcome(
            document_id=document.document_id,
            local_filename=document.local_filename,
            status=DownloadStatus.RETIRED,
            error=document.retired_reason,
            elapsed_s=elapsed(),
        )

    if not force and _looks_like_a_pdf(dest, min_bytes=s.min_pdf_bytes):
        size, digest = sha256_bytes_of(dest)
        return DownloadOutcome(
            document_id=document.document_id,
            local_filename=document.local_filename,
            status=DownloadStatus.PRESENT,
            file_size_bytes=size,
            sha256=digest,
            elapsed_s=elapsed(),
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    owned_client = client is None
    http = client or httpx.Client(
        follow_redirects=True,
        timeout=s.download_timeout_s,
        headers=_browser_headers(s),
    )

    succeeded = False
    try:
        retrying = Retrying(
            stop=stop_after_attempt(s.max_retries + 1),
            wait=wait_exponential(multiplier=s.retry_initial_backoff_s, max=s.retry_max_backoff_s),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        )
        size, digest = retrying(
            _stream_to_part,
            http,
            document.source_url,
            part,
            min_bytes=s.min_pdf_bytes,
            headers=_browser_headers(s),
        )
        # Atomic on POSIX and Windows alike: the destination either holds the previous
        # file or the complete new one, never a partial write.
        part.replace(dest)
        succeeded = True
    except (_ContentRejected, httpx.HTTPError, OSError) as exc:
        log.warning(
            "corpus.download_failed",
            document_id=document.document_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        return DownloadOutcome(
            document_id=document.document_id,
            local_filename=document.local_filename,
            status=DownloadStatus.FAILED,
            error=f"{type(exc).__name__}: {exc}",
            elapsed_s=elapsed(),
        )
    finally:
        # Covers KeyboardInterrupt and every other BaseException as well, which an
        # `except` clause listing Exception subclasses would miss.
        if not succeeded:
            part.unlink(missing_ok=True)
        if owned_client:
            http.close()

    log.info(
        "corpus.downloaded",
        document_id=document.document_id,
        bytes=size,
        elapsed_s=round(elapsed(), 2),
    )
    return DownloadOutcome(
        document_id=document.document_id,
        local_filename=document.local_filename,
        status=DownloadStatus.DOWNLOADED,
        file_size_bytes=size,
        sha256=digest,
        elapsed_s=elapsed(),
    )


def download_corpus(
    sources: SourceList,
    settings: Settings | None = None,
    *,
    force: bool = False,
    only: Iterable[str] | None = None,
) -> list[DownloadOutcome]:
    """Fetch every document in ``sources``, sharing one HTTP client.

    ``only`` restricts the run to specific document ids, which makes re-fetching one
    broken document cheap instead of re-walking the whole corpus.
    """
    s = settings if settings is not None else get_settings()
    wanted = set(only) if only is not None else None
    selected: Sequence[SourceDocument] = [
        d for d in sources.documents if wanted is None or d.document_id in wanted
    ]

    if wanted is not None:
        missing = sorted(wanted - {d.document_id for d in selected})
        if missing:
            msg = f"unknown document ids: {missing}"
            raise SourceUnavailableError(msg)

    s.corpus.raw_dir.mkdir(parents=True, exist_ok=True)
    outcomes: list[DownloadOutcome] = []
    with httpx.Client(
        follow_redirects=True,
        timeout=s.corpus.download_timeout_s,
        headers=_browser_headers(s.corpus),
    ) as client:
        for document in selected:
            outcomes.append(download_document(document, s, client=client, force=force))
    return outcomes
