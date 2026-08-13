"""Liveness and readiness endpoints.

The distinction is operational, not cosmetic:

* ``/health`` answers "is this process alive?" -- no dependency calls, always cheap.
  An orchestrator uses it to decide whether to *restart* the container.
* ``/ready`` answers "can this instance serve traffic?" -- it checks dependencies and
  returns 503 when a critical one is down. An orchestrator uses it to decide whether
  to *route* to the container.

Conflating them causes a classic outage: a transient database failure trips liveness,
every replica restarts simultaneously, and the outage outlives its cause.
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from medguard import __version__
from medguard.api.schemas import ComponentStatus, HealthResponse, ReadinessResponse
from medguard.config import get_settings
from medguard.diagnostics import CheckStatus, run_diagnostics

__all__ = ["router"]

router = APIRouter(tags=["health"])

#: Connectivity timeout for readiness checks. Short enough that the probe answers
#: well inside a typical 5-10s orchestrator timeout even when a dependency is down.
_PROBE_TIMEOUT_S = 2


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    responses={200: {"description": "The process is running."}},
)
def health() -> HealthResponse:
    """Return 200 whenever the process can serve HTTP. Checks no dependencies."""
    return HealthResponse(version=__version__)


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    responses={
        200: {"description": "All critical dependencies are reachable."},
        503: {"description": "A critical dependency is unavailable."},
    },
)
def ready(response: Response) -> ReadinessResponse:
    """Check dependencies and set 503 when a critical one has failed.

    Hardware probes and the Gemini smoke test are excluded, and connectivity checks
    get a short timeout: a readiness probe runs every few seconds, so it must be
    cheap, must not consume API quota, and must not block on an unreachable
    dependency for longer than the orchestrator's own probe timeout.
    """
    report = run_diagnostics(
        get_settings(),
        smoke_test=False,
        include_hardware=False,
        network_timeout_s=_PROBE_TIMEOUT_S,
    )

    if not report.ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        ready=report.ok,
        version=__version__,
        components=[
            ComponentStatus(
                name=check.name,
                status=CheckStatus(check.status).value,
                detail=check.detail,
                latency_ms=check.latency_ms,
            )
            for check in report.checks
        ],
    )
