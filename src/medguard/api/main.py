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

import urllib.parse
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import SecretStr
from starlette.exceptions import HTTPException as StarletteHTTPException

from medguard import __version__
from medguard.api.routes import health as health_routes
from medguard.api.routes import query as query_routes
from medguard.api.schemas import ErrorResponse
from medguard.config import Settings, get_settings
from medguard.errors import (
    CollectionNotFoundError,
    MedGuardError,
    ProviderError,
    VectorStoreTimeoutError,
    VectorStoreUnavailableError,
)
from medguard.logging import configure_logging, get_logger
from medguard.models import MEDICAL_DISCLAIMER
from medguard.observability.tracing import get_tracer
from medguard.storage.vector_store import VectorStore

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
    secrets: list[str | SecretStr | None] = [
        settings.gemini.api_key,
        settings.qdrant.api_key,
        settings.langfuse.public_key,
        settings.langfuse.secret_key,
    ]
    if settings.app.database_url:
        parsed_db = urllib.parse.urlparse(settings.app.database_url)
        if parsed_db.password:
            secrets.append(parsed_db.password)
    configure_logging(
        level=settings.app.log_level,
        fmt=settings.app.log_format,
        secrets=secrets,
    )
    tracer = get_tracer()
    app.state.tracer = tracer

    # Shared application-scoped vector store resource
    if not hasattr(app.state, "vector_store") or app.state.vector_store is None:
        app.state.vector_store = VectorStore(settings)

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
        # Close shared vector store resources cleanly
        if hasattr(app.state, "vector_store") and app.state.vector_store is not None:
            try:
                app.state.vector_store.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("api.vector_store_close_error", error=str(exc))
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
        if isinstance(
            exc,
            (
                ProviderError,
                VectorStoreUnavailableError,
                VectorStoreTimeoutError,
                CollectionNotFoundError,
            ),
        ):
            status_code = 503
        else:
            status_code = 500
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(
                error=type(exc).__name__,
                message=exc.user_message,
                trace_id=correlation_id,
            ).model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        correlation_id = str(uuid.uuid4())
        log.warning(
            "api.validation_error",
            path=request.url.path,
            errors=str(exc.errors()),
            trace_id=correlation_id,
        )
        messages: list[str] = []
        for err in exc.errors():
            loc = " -> ".join(str(el) for el in err.get("loc", []) if el != "body")
            msg = err.get("msg", "Invalid value")
            messages.append(f"{loc}: {msg}" if loc else msg)
        safe_message = "; ".join(messages) or "Invalid request parameters."
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="ValidationError",
                message=safe_message,
                trace_id=correlation_id,
            ).model_dump(mode="json"),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        correlation_id = str(uuid.uuid4())
        log.warning(
            "api.http_error",
            status_code=exc.status_code,
            detail=str(exc.detail),
            path=request.url.path,
            trace_id=correlation_id,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error="HTTPError",
                message=str(exc.detail),
                trace_id=correlation_id,
            ).model_dump(mode="json"),
        )

    app.include_router(health_routes.router)
    app.include_router(query_routes.router)
    return app
