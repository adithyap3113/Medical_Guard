"""Application-wide exception taxonomy.

Why this module exists
----------------------
Every external dependency (Gemini, Qdrant, Langfuse, the database) raises its own
vendor-specific exception types. If those propagate into the retrieval, API and CLI
layers, then swapping a provider means editing every ``except`` clause in the
codebase, and the HTTP layer ends up leaking vendor internals to callers.

So the rule is: **adapters translate, the rest of the app catches these types.**
``providers/gemini.py`` is the only module allowed to import ``google.genai.errors``;
it converts them into the ``ProviderError`` hierarchy below.

Each error carries a ``user_message`` that is safe to show to an end user, separate
from the technical detail we keep for logs. Vendor error payloads can echo request
content back, so they never become the user-facing string.
"""

from __future__ import annotations

__all__ = [
    "ChunkingError",
    "CollectionMismatchError",
    "ConfigurationError",
    "CorpusError",
    "EmbeddingError",
    "IngestionError",
    "IntegrityError",
    "ManifestError",
    "MedGuardError",
    "ParseError",
    "ProviderAuthError",
    "ProviderBlockedError",
    "ProviderError",
    "ProviderModelNotFoundError",
    "ProviderQuotaError",
    "ProviderRateLimitError",
    "ProviderResponseError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "SourceUnavailableError",
    "VectorStoreError",
]


class MedGuardError(Exception):
    """Base class for every error MedGuard raises deliberately.

    Catching ``MedGuardError`` catches "something we anticipated"; anything else
    escaping is a genuine bug and should not be swallowed.
    """

    #: Shown to end users. Must never contain secrets or raw vendor payloads.
    user_message = "An internal error occurred."

    def __init__(self, detail: str, *, user_message: str | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        if user_message is not None:
            self.user_message = user_message


class ConfigurationError(MedGuardError):
    """Settings are missing, malformed, or mutually inconsistent.

    Raised at startup rather than at first use, so a misconfigured deployment
    fails immediately instead of halfway through a user request.
    """

    user_message = "The service is not configured correctly."


# ---------------------------------------------------------------------------
# LLM provider errors
# ---------------------------------------------------------------------------


class ProviderError(MedGuardError):
    """Base for all LLM provider failures, vendor-neutral by construction."""

    user_message = "The language model is temporarily unavailable."

    #: Whether retrying the identical request could plausibly succeed. Used by
    #: callers to decide between backing off and failing fast.
    retryable = False

    def __init__(
        self,
        detail: str,
        *,
        user_message: str | None = None,
        status_code: int | None = None,
        provider: str = "gemini",
    ) -> None:
        super().__init__(detail, user_message=user_message)
        self.status_code = status_code
        self.provider = provider


class ProviderAuthError(ProviderError):
    """Missing, invalid, or unauthorised API key (HTTP 401 / 403).

    Never retryable: the same key will keep failing. Surfacing this distinctly
    from a rate limit matters, because the fix is human, not temporal.
    """

    user_message = "The language model is not correctly authenticated."
    retryable = False


class ProviderRateLimitError(ProviderError):
    """Requests-per-minute throttling (HTTP 429, transient).

    Distinct from :class:`ProviderQuotaError`: this clears on its own within
    seconds, so backing off is the correct response.
    """

    user_message = "The service is busy. Please try again in a moment."
    retryable = True

    def __init__(
        self,
        detail: str,
        *,
        retry_after_s: float | None = None,
        user_message: str | None = None,
        status_code: int | None = None,
        provider: str = "gemini",
    ) -> None:
        super().__init__(
            detail, user_message=user_message, status_code=status_code, provider=provider
        )
        self.retry_after_s = retry_after_s


class ProviderQuotaError(ProviderError):
    """Hard quota exhaustion, e.g. free-tier requests-per-day (also HTTP 429).

    Deliberately *not* retryable even though it shares a status code with rate
    limiting: retrying a daily-quota failure just burns the remaining budget and
    delays the honest answer. Telling these apart requires reading the error body.
    """

    user_message = "The daily language-model quota has been exhausted. Try again later."
    retryable = False


class ProviderTimeoutError(ProviderError):
    """The request exceeded the configured deadline."""

    user_message = "The language model took too long to respond."
    retryable = True


class ProviderUnavailableError(ProviderError):
    """Upstream 5xx, connection reset, or model overload."""

    user_message = "The language model is temporarily unavailable."
    retryable = True


class ProviderBlockedError(ProviderError):
    """Generation stopped for safety/recitation reasons, or returned no candidate.

    This is a real case for a medical corpus: clinical phrasing around self-harm,
    overdose or controlled substances can trip safety filters. It is not a bug and
    not retryable, and it must be reported honestly rather than silently returned
    as an empty answer.
    """

    user_message = (
        "The response was withheld by the model's safety filters. Try rephrasing your question."
    )
    retryable = False

    def __init__(
        self,
        detail: str,
        *,
        finish_reason: str | None = None,
        user_message: str | None = None,
        status_code: int | None = None,
        provider: str = "gemini",
    ) -> None:
        super().__init__(
            detail, user_message=user_message, status_code=status_code, provider=provider
        )
        self.finish_reason = finish_reason


class ProviderModelNotFoundError(ProviderError):
    """The requested model does not exist or is not enabled for this API project.

    Kept separate from a generic 4xx because it has a specific recovery: fall back to
    ``GEMINI_FALLBACK_MODEL``. It is also the concrete form of "do not assume a model
    advertised as free is available to your project" -- the only way to know is to ask.
    """

    user_message = "The configured language model is not available."
    retryable = False


class ProviderResponseError(ProviderError):
    """The call succeeded at the HTTP level but the payload was unusable."""

    user_message = "The language model returned an unusable response."
    retryable = True


# ---------------------------------------------------------------------------
# Storage errors
# ---------------------------------------------------------------------------


class VectorStoreError(MedGuardError):
    """The vector database is unreachable or rejected an operation."""

    user_message = "The knowledge index is temporarily unavailable."


class CollectionMismatchError(VectorStoreError):
    """The live Qdrant collection does not have the shape this code expects.

    Raised by ``VectorStore.ensure_collection`` when a collection of the right *name*
    exists with the wrong vector configuration -- a different dimension, a different
    distance metric, a missing named vector, or a sparse vector without IDF.

    Deliberately fatal rather than self-healing. The tempting behaviour is to drop and
    recreate, and that silently destroys an index someone may have spent an hour
    building. The other tempting behaviour is to write into it anyway, which is worse:
    vectors from two models in one collection produce retrieval that is confidently,
    unmeasurably wrong. Refusing and saying exactly what differs leaves the decision
    with the person who knows which of the two is the mistake.
    """

    user_message = "The knowledge index is not compatible with this build."


# ---------------------------------------------------------------------------
# Corpus errors
# ---------------------------------------------------------------------------


class CorpusError(MedGuardError):
    """Base for failures acquiring or validating the source-document corpus."""

    user_message = "The document corpus is unavailable or invalid."


class ManifestError(CorpusError):
    """``data/manifest.json`` is missing, malformed, or internally inconsistent.

    Distinct from :class:`IntegrityError`: the manifest itself is wrong here, rather
    than the manifest being right and the files on disk disagreeing with it.
    """

    user_message = "The corpus manifest could not be read."


class SourceUnavailableError(CorpusError):
    """A source document could not be fetched from its publisher.

    Routine rather than exceptional: publishers move URLs, put documents behind
    consent interstitials, and add anti-bot challenges. One unavailable source must
    not abort acquisition of the rest, so this is raised per-document and collected.
    """

    user_message = "A source document could not be downloaded."


class IntegrityError(CorpusError):
    """A file on disk does not match what the manifest records for it.

    Treated as an error rather than a warning because every downstream guarantee --
    reproducible chunk ids, citations that point at the text they claim to -- assumes
    the bytes are the ones that were hashed. Silent drift here corrupts retrieval in a
    way that surfaces much later and is very hard to trace back.
    """

    user_message = "A corpus document failed its integrity check."


# ---------------------------------------------------------------------------
# Ingestion errors
# ---------------------------------------------------------------------------


class IngestionError(MedGuardError):
    """Base for failures turning a source document into indexed chunks.

    Separate from :class:`CorpusError` on a real boundary: corpus errors are about
    *acquiring and trusting bytes*, ingestion errors are about *interpreting* bytes that
    are already trusted. A corpus error means re-download; an ingestion error means the
    file is fine and this code could not handle it.

    Every subclass is caught per-document by the pipeline and recorded on the row, which
    is what stops document 7 from taking documents 8 through 20 down with it.
    """

    user_message = "A source document could not be processed."


class ParseError(IngestionError):
    """A PDF could not be opened or its text could not be extracted.

    Routine, not exceptional. Public-health PDFs include scanned-image documents with no
    text layer, files truncated by a proxy, and the occasional structurally damaged
    object table. Each of those is one unusable document, not a broken pipeline.

    Always carries the document context. PyMuPDF's own messages are frequently a bare
    ``"cannot open document"`` or an mupdf error string with no filename in it, and an
    ingestion log that says which of twenty documents failed is the difference between
    a two-minute fix and an afternoon.
    """

    user_message = "A source document could not be read."


class ChunkingError(IngestionError):
    """Chunking produced something structurally impossible.

    Not raised for a document that simply yields no chunks -- an empty document is a
    legitimate, reportable outcome. This is for invariant violations: a chunk longer
    than the configured maximum, a page span running backwards, an index that skips.
    Those mean the chunker has a bug, and the pipeline should stop persisting its output
    rather than write a corrupt index that looks fine.
    """

    user_message = "A source document could not be divided into passages."


class EmbeddingError(IngestionError):
    """The embedding model failed, or returned something of the wrong shape.

    The shape check is the important half. A model that silently returns 512-dimensional
    vectors into a 384-dimensional collection, or that returns fewer vectors than the
    texts it was given, corrupts the index in a way no downstream component can detect:
    retrieval keeps working and simply returns the wrong passages.
    """

    user_message = "The embedding model is unavailable."
