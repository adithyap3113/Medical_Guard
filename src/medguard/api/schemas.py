"""HTTP request/response schemas.

Separate from :mod:`medguard.models` on purpose. Domain models describe what the
system *is*; these describe the public wire contract. Serving domain models directly
would mean any internal refactor becomes a breaking API change, and internal fields
would leak to callers by default rather than by decision.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from medguard.models import MEDICAL_DISCLAIMER

__all__ = [
    "ComponentStatus",
    "ErrorResponse",
    "HealthResponse",
    "ReadinessResponse",
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
