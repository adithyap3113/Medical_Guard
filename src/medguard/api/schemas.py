"""HTTP request/response schemas.

Separate from :mod:`medguard.models` on purpose. Domain models describe what the
system *is*; these describe the public wire contract. Serving domain models directly
would mean any internal refactor becomes a breaking API change, and internal fields
would leak to callers by default rather than by decision.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from medguard.models import MEDICAL_DISCLAIMER, DocumentType, QueryStrategy, SourceOrganization

__all__ = [
    "CitationResponse",
    "ComponentStatus",
    "ErrorResponse",
    "HealthResponse",
    "QueryRequest",
    "QueryResponse",
    "ReadinessResponse",
    "RetrievalMetadataResponse",
]


class HealthResponse(BaseModel):
    """Liveness: is the process running?

    Deliberately checks nothing external. A liveness probe that calls a database will
    restart a perfectly healthy process during a database blip, which turns a small
    dependency outage into a full outage.
    """

    model_config = ConfigDict(frozen=True)

    status: str = Field(default="ok", examples=["ok"])
    service: str = Field(default="medguard")
    version: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ComponentStatus(BaseModel):
    """Per-dependency readiness detail."""

    model_config = ConfigDict(frozen=True)

    name: str
    status: str = Field(description="ok | warn | fail | skip")
    detail: str = ""
    latency_ms: float | None = None


class ReadinessResponse(BaseModel):
    """Readiness: can this instance serve requests?

    Unlike liveness, this *does* check dependencies, and returns HTTP 503 when a
    critical one is down so a load balancer stops sending traffic here.
    """

    model_config = ConfigDict(frozen=True)

    ready: bool
    version: str
    components: list[ComponentStatus] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ErrorResponse(BaseModel):
    """Uniform error envelope.

    Carries only the safe ``user_message`` from the error taxonomy, never the internal
    detail or a vendor payload, because upstream error bodies can echo request content
    and reveal infrastructure. The internal detail goes to the logs with the same
    ``trace_id``, so an operator can still join the two.
    """

    model_config = ConfigDict(frozen=True)

    error: str = Field(description="Machine-readable error type.")
    message: str = Field(description="Safe, human-readable explanation.")
    trace_id: str | None = None
    disclaimer: str = Field(default=MEDICAL_DISCLAIMER)


class QueryRequest(BaseModel):
    """Incoming clinical question request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str = Field(
        min_length=1,
        max_length=2048,
        description="The medical or clinical question to query against the guideline corpus.",
        examples=["What are the recommended first-line treatments for hypertension?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of evidence chunks to retrieve and consider.",
    )
    strategy: QueryStrategy = Field(
        default=QueryStrategy.BASIC,
        description="Query processing strategy (basic, rewrite, multi_query, decomposition, hyde).",
    )
    retriever: str = Field(
        default="hybrid",
        description="Retriever type: 'hybrid' (dense + sparse RRF), 'dense', or 'sparse'.",
    )
    source_organization: list[SourceOrganization] | None = Field(
        default=None,
        description="Optional filter by source organization (e.g. CDC, WHO, ICMR).",
    )
    document_type: list[DocumentType] | None = Field(
        default=None,
        description="Optional filter by document type (e.g. guideline, reference).",
    )

    @field_validator("question")
    @classmethod
    def _validate_question(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("question cannot be empty or contain only whitespace")
        if len(stripped) > 2048:
            raise ValueError("question is too long (maximum 2048 characters)")
        return stripped

    @field_validator("retriever")
    @classmethod
    def _validate_retriever(cls, v: str) -> str:
        allowed = ("hybrid", "dense", "sparse")
        if v not in allowed:
            raise ValueError(f"invalid retriever {v!r}; expected one of {allowed}")
        return v


class CitationResponse(BaseModel):
    """Source provenance and citation details for an answer claim."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(description="Unique identifier of the source document.")
    source_title: str | None = Field(default=None, description="Title of the source guideline.")
    page_start: int | None = Field(
        default=None, ge=1, description="Starting page in original document."
    )
    page_end: int | None = Field(
        default=None, ge=1, description="Ending page in original document."
    )
    quote: str | None = Field(default=None, description="Supporting verbatim quote if applicable.")
    supported: bool | None = Field(
        default=None, description="Validation verdict against source text."
    )
    validation_status: str | None = Field(
        default=None,
        description="Validation status (e.g. quote_verified, source_valid, unsupported).",
    )
    validation_note: str | None = Field(
        default=None, description="Technical detail of validation outcome."
    )


class RetrievalMetadataResponse(BaseModel):
    """Metadata regarding the retrieval stage."""

    model_config = ConfigDict(frozen=True)

    strategy: str = Field(description="Query strategy used.")
    retriever: str = Field(description="Underlying retrieval kind.")
    retrieved_chunks: int = Field(ge=0, description="Number of candidate chunks retrieved.")
    expanded_queries: list[str] = Field(default_factory=list, description="Sub-queries generated.")
    latency_ms: float | None = Field(default=None, ge=0)


class QueryResponse(BaseModel):
    """Complete structured response for a clinical question."""

    model_config = ConfigDict(frozen=True)

    answer: str = Field(description="Context-grounded medical answer.")
    citations: list[CitationResponse] = Field(
        default_factory=list, description="Citations supporting statements in the answer."
    )
    retrieval: RetrievalMetadataResponse = Field(description="Details about retrieval execution.")
    abstained: bool = Field(
        default=False,
        description="True if MedGuard declined to answer due to insufficient evidence.",
    )
    medical_disclaimer: str = Field(
        default=MEDICAL_DISCLAIMER, description="Mandatory medical safety disclaimer."
    )
    trace_id: str | None = Field(default=None, description="Request correlation / tracing ID.")
