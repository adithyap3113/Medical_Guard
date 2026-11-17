"""Typed application configuration.

Design decisions worth defending
--------------------------------
1. **One settings object, built once.** ``get_settings()`` is cached, so the whole
   process sees identical configuration. Modules never read ``os.environ`` directly;
   if they did, config would be untyped, unvalidated, and untestable.

2. **Grouped sub-settings with their own env prefixes** rather than one flat blob.
   ``GEMINI_API_KEY``, ``QDRANT_URL`` and ``LANGFUSE_SECRET_KEY`` are the names the
   upstream vendors document, so using them avoids a translation layer that people
   get wrong. Grouping keeps ``settings.gemini.model`` readable at the call site.

3. **Secrets are ``SecretStr``, never ``str``.** Pydantic's ``SecretStr`` renders as
   ``'**********'`` in ``repr()``, tracebacks, and logs. Reading the real value
   requires an explicit ``.get_secret_value()`` call, which is greppable in review.
   This is the type system enforcing the "do not log secrets" rule, rather than a
   convention people are asked to remember.

4. **Validation happens at construction.** A bad temperature or an unreachable-looking
   URL fails at startup with a clear message, not on the first user request.

5. **The Gemini model is never hard-coded at a call site.** It is read from
   ``GEMINI_MODEL`` with a default that can be changed without touching code. The
   default below is a starting point, not a guarantee: run ``medguard doctor`` to
   confirm the model is actually available to *your* API project.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "AppSettings",
    "CorpusSettings",
    "EmbeddingSettings",
    "GeminiSettings",
    "IngestionSettings",
    "LangfuseSettings",
    "QdrantSettings",
    "RetrievalSettings",
    "Settings",
    "get_settings",
    "project_root",
]


@functools.lru_cache(maxsize=1)
def project_root() -> Path:
    """Locate the repository root by walking up to the directory holding pyproject.toml.

    Resolving ``.env`` relative to the project root rather than the current working
    directory means ``medguard doctor`` behaves identically no matter where it is
    invoked from -- a common source of "works on my machine" confusion.
    """
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[2]


def _env_file() -> Path:
    return project_root() / ".env"


_BASE_CONFIG = SettingsConfigDict(
    env_file_encoding="utf-8",
    extra="ignore",
    case_sensitive=False,
    # `model` is a legitimate field name here; silence pydantic's `model_` guard.
    protected_namespaces=(),
)


class GeminiSettings(BaseSettings):
    """Gemini generation provider configuration (env prefix ``GEMINI_``)."""

    model_config = SettingsConfigDict(
        **_BASE_CONFIG,
        env_prefix="GEMINI_",
        env_file=_env_file(),
    )

    api_key: SecretStr | None = Field(
        default=None,
        description="Gemini API key. Absent is valid: the app boots and reports the "
        "provider as unconfigured rather than crashing.",
    )
    model: str = Field(
        default="gemini-2.5-flash-lite",
        description="Primary generation model. Verify availability with `medguard doctor`.",
    )
    fallback_model: str = Field(
        default="gemini-2.0-flash-lite",
        description="Used when the primary model is unavailable to this API project.",
    )
    timeout_s: float = Field(default=60.0, gt=0, le=600)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_initial_backoff_s: float = Field(default=1.0, gt=0, le=60)
    retry_max_backoff_s: float = Field(default=30.0, gt=0, le=300)
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Low by default: grounded medical answers should follow the "
        "retrieved sources, not improvise around them.",
    )
    max_output_tokens: int = Field(default=2048, ge=64, le=65_536)

    @property
    def configured(self) -> bool:
        """True when an API key is present. Does not imply the key is *valid*."""
        key = self.api_key.get_secret_value().strip() if self.api_key else ""
        return bool(key)

    @property
    def timeout_ms(self) -> int:
        """Gemini's ``HttpOptions.timeout`` is specified in milliseconds."""
        return int(self.timeout_s * 1000)

    @model_validator(mode="after")
    def _check_backoff_ordering(self) -> GeminiSettings:
        if self.retry_max_backoff_s < self.retry_initial_backoff_s:
            msg = (
                f"GEMINI_RETRY_MAX_BACKOFF_S ({self.retry_max_backoff_s}) must be >= "
                f"GEMINI_RETRY_INITIAL_BACKOFF_S ({self.retry_initial_backoff_s})"
            )
            raise ValueError(msg)
        return self


class QdrantSettings(BaseSettings):
    """Vector database configuration (env prefix ``QDRANT_``)."""

    model_config = SettingsConfigDict(
        **_BASE_CONFIG,
        env_prefix="QDRANT_",
        env_file=_env_file(),
    )

    url: str = Field(default="http://localhost:6333")
    api_key: SecretStr | None = Field(
        default=None, description="Only needed for Qdrant Cloud; local Docker needs none."
    )
    collection: str = Field(
        default="medguard",
        description="Base collection name. The effective name is suffixed with the "
        "embedding model and dimension so different models never collide.",
    )
    timeout_s: int = Field(default=30, gt=0, le=300)
    prefer_grpc: bool = Field(
        default=False,
        description="HTTP by default: easier to debug and inspect. gRPC is faster for "
        "bulk upserts and can be enabled during Phase 3 ingestion.",
    )


class LangfuseSettings(BaseSettings):
    """Observability configuration (env prefix ``LANGFUSE_``).

    Note: Langfuse SDK v3+ renamed ``LANGFUSE_HOST`` to ``LANGFUSE_BASE_URL``. Both are
    accepted here so older ``.env`` files keep working.
    """

    model_config = SettingsConfigDict(
        **_BASE_CONFIG,
        env_prefix="LANGFUSE_",
        env_file=_env_file(),
    )

    public_key: SecretStr | None = None
    secret_key: SecretStr | None = None
    base_url: str = Field(default="https://cloud.langfuse.com")
    host: str | None = Field(default=None, description="Deprecated alias for base_url.")
    enabled: bool = Field(
        default=True,
        description="Master switch. Set false to disable tracing even when keys are "
        "present -- useful for offline work and for tests.",
    )

    @property
    def effective_base_url(self) -> str:
        return self.host or self.base_url

    @property
    def configured(self) -> bool:
        return bool(self.enabled and self.public_key and self.secret_key)


class EmbeddingSettings(BaseSettings):
    """Embedding configuration (env prefix ``EMBEDDING_``).

    Declared in Phase 1 but not yet *used*: no model is loaded or downloaded until
    Phase 3B. It lives here now so the collection-naming scheme below is fixed before
    any data is indexed -- changing it later would invalidate an existing index.
    """

    model_config = SettingsConfigDict(
        **_BASE_CONFIG,
        env_prefix="EMBEDDING_",
        env_file=_env_file(),
    )

    model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Dense embedding model, run via ONNX (fastembed). Chosen for a "
        "CPU-only 8 GB machine; swapping to bge-base or BGE-M3 on better hardware "
        "is a config change, not a code change.",
    )
    dimension: int = Field(default=384, gt=0, le=8192)
    sparse_model: str = Field(
        default="Qdrant/bm25",
        description="Sparse retrieval model. BM25 handles exact drug names, dosages "
        "and abbreviations that dense vectors tend to smooth over.",
    )
    batch_size: int = Field(
        default=8,
        gt=0,
        le=512,
        description=(
            "Texts embedded per forward pass. Lowered from 32 in Phase 3A after doing "
            "the arithmetic rather than copying a common default. Peak transient "
            "allocation for BGE-small attention is roughly "
            "batch x 12 heads x 512 tokens x 512 tokens x 4 bytes: at 32 that is about "
            "402 MB in one spike, on a machine measured at 0.84 GB available. At 8 it "
            "is about 100 MB. Raising this trades a machine that finishes ingestion for "
            "one that is killed partway through -- which leaves a half-populated "
            "collection, not a clean failure. Raise it only after re-measuring."
        ),
    )


class IngestionSettings(BaseSettings):
    """Parse-and-chunk configuration (env prefix ``INGESTION_``).

    Declared in Phase 3A, used by the ingestion pipeline from Phase 3B. They live here
    rather than as literals in the chunker because the chunking
    parameters are the single most consequential retrieval decision in the project and
    they deserve to be argued about in a config file rather than discovered as literals
    buried in a chunker.

    Sizes are in characters, not tokens, deliberately: the chunker runs before any
    tokeniser is loaded, and a character budget is checkable without one. BGE-small
    truncates at 512 tokens, which for English prose is roughly 2,000 characters --
    hence ``chunk_max_chars``. A chunk above that is not an error, it is *silently
    truncated at embedding time*, so half the text is indexed and nobody is told.
    """

    model_config = SettingsConfigDict(
        **_BASE_CONFIG,
        env_prefix="INGESTION_",
        env_file=_env_file(),
    )

    chunk_size: int = Field(
        default=1600,
        gt=0,
        le=8192,
        description="Target chunk length in characters. Sized to sit under the "
        "embedding model's 512-token ceiling with headroom for the dense prose and "
        "long drug names typical of clinical guidance.",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=4096,
        description="Characters repeated between adjacent chunks, so a recommendation "
        "split across a boundary is still wholly present in one of them.",
    )
    chunk_min_chars: int = Field(
        default=200,
        ge=0,
        description="Chunks shorter than this are merged into a neighbour. A 30-character "
        "fragment ('Table 3. Continued') embeds to a near-meaningless vector that "
        "still competes for a retrieval slot.",
    )
    chunk_max_chars: int = Field(
        default=2000,
        gt=0,
        le=16_384,
        description="Hard ceiling. Above this the embedding model truncates silently, "
        "so the chunker must split rather than let that happen unannounced.",
    )
    strip_running_headers: bool = Field(
        default=True,
        description="Remove repeated page headers and footers before chunking. Left in, "
        "'MMWR / Recommendations and Reports' appears on all 60 pages and becomes the "
        "highest-IDF-scoring text in the document for no useful reason.",
    )
    dry_run_sample_chunks: int = Field(
        default=5,
        ge=0,
        le=100,
        description="How many chunks `medguard ingest --dry-run` prints for inspection. "
        "Reading actual chunk boundaries is the only reliable way to catch a chunker "
        "that is technically working and semantically useless.",
    )
    batch_size: int = Field(
        default=8,
        gt=0,
        le=512,
        description="Chunks embedded and upserted per batch. Matches the embedding "
        "batch size for the same memory reason -- see EmbeddingSettings.batch_size -- "
        "and the pipeline halves it automatically when available memory drops below "
        "300 MB, abandoning the document below 150 MB rather than being killed "
        "partway through and leaving a half-populated collection.",
    )

    @model_validator(mode="after")
    def _check_size_ordering(self) -> IngestionSettings:
        if self.chunk_overlap >= self.chunk_size:
            # Overlap >= size means each chunk re-emits its predecessor entirely and
            # the chunker either never advances or duplicates the whole document.
            msg = (
                f"INGESTION_CHUNK_OVERLAP ({self.chunk_overlap}) must be < "
                f"INGESTION_CHUNK_SIZE ({self.chunk_size})"
            )
            raise ValueError(msg)
        if self.chunk_max_chars < self.chunk_size:
            msg = (
                f"INGESTION_CHUNK_MAX_CHARS ({self.chunk_max_chars}) must be >= "
                f"INGESTION_CHUNK_SIZE ({self.chunk_size})"
            )
            raise ValueError(msg)
        if self.chunk_min_chars > self.chunk_size:
            msg = (
                f"INGESTION_CHUNK_MIN_CHARS ({self.chunk_min_chars}) must be <= "
                f"INGESTION_CHUNK_SIZE ({self.chunk_size})"
            )
            raise ValueError(msg)
        return self


class CorpusSettings(BaseSettings):
    """Source-document corpus configuration (env prefix ``CORPUS_``).

    Paths are resolved against the project root when relative, so ``medguard corpus
    verify`` behaves the same from any working directory -- the same reasoning as
    :func:`project_root` applies to ``.env``.
    """

    model_config = SettingsConfigDict(
        **_BASE_CONFIG,
        env_prefix="CORPUS_",
        env_file=_env_file(),
    )

    data_dir: Path = Field(
        default=Path("data/raw"), description="Where downloaded source PDFs are stored."
    )
    sources_file: Path = Field(
        default=Path("data/sources.yaml"),
        description="Editable corpus definition. Deliberately not hard-coded in Python "
        "so the corpus can change without a code review.",
    )
    manifest_file: Path = Field(
        default=Path("data/manifest.json"),
        description="Generated provenance record. Committed; the PDFs are not.",
    )
    user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        ),
        description="Several publishers (cdc.gov among them) return HTTP 403 to a "
        "default client string. A real browser UA is required to reach public PDFs.",
    )
    download_timeout_s: float = Field(default=180.0, gt=0, le=1800)
    max_retries: int = Field(
        default=2, ge=0, le=10, description="Retries after the first attempt, not total attempts."
    )
    retry_initial_backoff_s: float = Field(default=2.0, gt=0, le=60)
    retry_max_backoff_s: float = Field(default=30.0, gt=0, le=300)
    min_pdf_bytes: int = Field(
        default=10_000,
        ge=0,
        description="Floor below which a 'PDF' is assumed to be an error page or a "
        "truncated transfer rather than a document.",
    )

    def resolved(self, value: Path) -> Path:
        """Anchor a relative configured path to the project root."""
        return value if value.is_absolute() else project_root() / value

    @property
    def raw_dir(self) -> Path:
        return self.resolved(self.data_dir)

    @property
    def sources_path(self) -> Path:
        return self.resolved(self.sources_file)

    @property
    def manifest_path(self) -> Path:
        return self.resolved(self.manifest_file)

    @model_validator(mode="after")
    def _check_backoff_ordering(self) -> CorpusSettings:
        if self.retry_max_backoff_s < self.retry_initial_backoff_s:
            msg = (
                f"CORPUS_RETRY_MAX_BACKOFF_S ({self.retry_max_backoff_s}) must be >= "
                f"CORPUS_RETRY_INITIAL_BACKOFF_S ({self.retry_initial_backoff_s})"
            )
            raise ValueError(msg)
        return self


class RetrievalSettings(BaseSettings):
    """Read-path configuration (env prefix ``RETRIEVAL_``).

    Every value here is a knob the evaluation harness sweeps, which is why they are
    configuration rather than call-site defaults: an experiment that has to edit code
    to change ``fetch_k`` is an experiment nobody runs a second time.

    ``fetch_k`` is separate from ``top_k`` because they answer different questions.
    ``top_k`` is how many results a caller wants; ``fetch_k`` is how deep each retrieval
    leg reads before anything is truncated. Keeping them equal is the most common way to
    get a weak hybrid: fusing two ten-item lists has almost nothing to fuse, and a chunk
    ranked eleventh by both retrievers can never be recovered by any amount of
    re-ranking downstream. Over-fetching is also what makes recall@50 measurable at all.
    """

    model_config = SettingsConfigDict(
        **_BASE_CONFIG,
        env_prefix="RETRIEVAL_",
        env_file=_env_file(),
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Chunks returned to the caller. Capped because an unbounded value "
        "turns one request into a full-collection scan.",
    )
    fetch_k: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Depth read from each retrieval leg before truncation to top_k. "
        "Must be >= top_k; see the class docstring for why they are not the same knob.",
    )
    default_retriever: Literal["dense", "sparse", "hybrid"] = Field(
        default="dense",
        description="Retriever used when a caller does not name one. Dense is the "
        "Phase 4A default because it is the baseline every later number is compared "
        "against. `hybrid` is accepted here but only implemented from Phase 4B.",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        le=500,
        description="Reciprocal Rank Fusion smoothing parameter. Standard default is 60.",
    )

    @model_validator(mode="after")
    def _check_depth_ordering(self) -> RetrievalSettings:
        if self.fetch_k < self.top_k:
            msg = (
                f"RETRIEVAL_FETCH_K ({self.fetch_k}) must be >= RETRIEVAL_TOP_K "
                f"({self.top_k}); fetching less than is returned truncates results "
                f"before the caller ever sees them"
            )
            raise ValueError(msg)
        return self


class AppSettings(BaseSettings):
    """Process-level settings that have no vendor prefix."""

    model_config = SettingsConfigDict(
        **_BASE_CONFIG,
        env_file=_env_file(),
    )

    environment: Literal["local", "dev", "prod"] = Field(default="local")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    log_format: Literal["json", "console"] = Field(
        default="console",
        description="`console` is human-readable for local work; `json` is what a log "
        "aggregator can parse. Deployment overrides this to `json`.",
    )
    database_url: str = Field(
        default="sqlite:///./medguard.db",
        description="SQLAlchemy URL. SQLite by default (zero RAM, zero setup); point "
        "at postgresql+psycopg://... to use the optional Postgres profile.",
    )
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000, gt=0, lt=65_536)

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper(cls, v: object) -> object:
        return v.upper() if isinstance(v, str) else v

    @field_validator("database_url")
    @classmethod
    def _known_scheme(cls, v: str) -> str:
        allowed = ("sqlite:", "postgresql:", "postgresql+psycopg:", "postgresql+psycopg2:")
        if not v.startswith(allowed):
            msg = f"DATABASE_URL must start with one of {allowed}; got {v.split(':')[0]!r}"
            raise ValueError(msg)
        return v


class Settings(BaseSettings):
    """The single composed configuration object for the whole application."""

    model_config = SettingsConfigDict(
        **_BASE_CONFIG,
        env_file=_env_file(),
    )

    app: AppSettings = Field(default_factory=AppSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    corpus: CorpusSettings = Field(default_factory=CorpusSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)

    @property
    def collection_name(self) -> str:
        """Namespace the Qdrant collection by embedding model and dimension.

        Two embedding models produce incompatible vector spaces. Writing both into
        one collection silently corrupts retrieval quality in a way that is very
        hard to notice. Encoding the model in the name makes the mistake impossible
        and makes A/B comparisons between models cheap.
        """
        slug = self.embedding.model.split("/")[-1].replace(".", "_").replace("-", "_").lower()
        return f"{self.qdrant.collection}__{slug}__{self.embedding.dimension}"

    def redacted(self) -> dict[str, Any]:
        """Configuration rendered safely for logs, ``doctor`` output, and bug reports.

        Every ``SecretStr`` becomes a presence indicator (``"<set>"`` / ``None``) --
        never a prefix, suffix, or length, since those leak usable information about
        a key. This is the only sanctioned way to display configuration.
        """

        def scrub(value: Any) -> Any:
            if isinstance(value, SecretStr):
                return "<set>" if value.get_secret_value() else None
            if isinstance(value, dict):
                return {k: scrub(v) for k, v in value.items()}
            return value

        dumped = self.model_dump()
        return {group: scrub(values) for group, values in dumped.items()}


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that configuration is read and validated exactly once. Tests that need
    different values should call ``get_settings.cache_clear()`` after monkeypatching
    the environment -- see ``tests/conftest.py``.
    """
    return Settings()
