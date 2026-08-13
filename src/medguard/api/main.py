"""FastAPI application factory.

Phase 1 exposes health endpoints only. The query endpoint arrives once there is
retrieval to serve -- an endpoint that returns fabricated answers while the pipeline
is unbuilt is worse than a missing endpoint.

What is wired now, and why it belongs at the foundation rather than bolted on later:

* a **lifespan** that configures redaction-aware logging before the first request and
  flushes traces on shutdown;
* an **exception handler** that converts the error taxonomy into a uniform envelope,
  so no vendor payload ever reaches a client;
* the **medical disclaimer** in the OpenAPI description, so it is part of the API
  contract rather than something the UI is trusted to remember.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from medguard import __version__
from medguard.api.routes import health as health_routes
from medguard.api.schemas import ErrorResponse
from medguard.config import Settings, get_settings
from medguard.errors import MedGuardError, ProviderError
from medguard.logging import configure_logging, get_logger
from medguard.models import MEDICAL_DISCLAIMER
from medguard.observability.tracing import get_tracer

__all__ = ["create_app"]

log = get_logger(__name__)

_DESCRIPTION = f"""
Retrieval-augmented question answering over an indexed corpus of published medical
reference documents.

**{MEDICAL_DISCLAIMER}**

MedGuard reports what its indexed sources say, with citations. It does not diagnose,
has not been clinically validated, and is not a medical device.
"""


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    configure_logging(
        level=settings.app.log_level,
        fmt=settings.app.log_format,
        secrets=[
            settings.gemini.api_key,
            settings.qdrant.api_key,
            settings.langfuse.public_key,
            settings.langfuse.secret_key,
        ],
    )
    tracer = get_tracer()
    app.state.tracer = tracer
    log.info(
        "api.startup",
        version=__version__,
        environment=settings.app.environment,
        tracing=tracer.backend,
        collection=settings.collection_name,
    )
    try:
        yield
    finally:
        # Short-lived processes lose buffered traces without an explicit flush.
        tracer.flush()
        tracer.shutdown()
        log.info("api.shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Accepts injected settings so tests can vary them."""
    resolved = settings if settings is not None else get_settings()

    app = FastAPI(
        title="MedGuard",
        version=__version__,
        description=_DESCRIPTION,
        lifespan=_lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    app.state.settings = resolved

    @app.exception_handler(MedGuardError)
    async def _handle_medguard_error(request: Request, exc: MedGuardError) -> JSONResponse:
        """Return the safe user message; keep the technical detail in the logs.

        The correlation id links the two, so an operator can find the full context
        without the client ever seeing an upstream payload.
        """
        correlation_id = str(uuid.uuid4())
        log.warning(
            "api.handled_error",
            error_type=type(exc).__name__,
            detail=exc.detail,
            path=request.url.path,
            trace_id=correlation_id,
        )
        status_code = 503 if isinstance(exc, ProviderError) else 500
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(
                error=type(exc).__name__,
                message=exc.user_message,
                trace_id=correlation_id,
            ).model_dump(mode="json"),
        )

    app.include_router(health_routes.router)
    return app
