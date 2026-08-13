"""Tracing abstraction over Langfuse, with a no-op fallback.

The governing rule: **observability must never be able to break the thing it observes.**
A tracing backend is a non-essential dependency, so every failure mode -- missing
credentials, the package not installed, the network down, Langfuse returning 500, an
SDK API change -- degrades to a no-op and lets the request proceed. A RAG system that
returns 500 because a metrics sink is unreachable is a worse system than one with no
metrics at all.

That is enforced structurally: callers only ever touch :class:`Tracer`, whose methods
cannot raise. The two implementations, :class:`NoOpTracer` and
:class:`LangfuseTracer`, are interchangeable, so the code path in the application is
identical whether or not tracing is on. There is no ``if tracing_enabled:`` scattered
through the pipeline.

Secrets: Langfuse keys are read from :mod:`medguard.config` as ``SecretStr`` and are
never logged. Note the privacy consequence of using Langfuse *Cloud* -- traces include
queries and retrieved document text, and they leave this machine. That is acceptable
for a corpus of published guidance and no PHI. It would need revisiting before
pointing MedGuard at private clinical records.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from medguard.config import LangfuseSettings, get_settings
from medguard.logging import get_logger

__all__ = [
    "LangfuseTracer",
    "NoOpTracer",
    "Span",
    "Tracer",
    "TracingHealth",
    "get_tracer",
]

log = get_logger(__name__)


class TracingHealth(BaseModel):
    """Diagnostic snapshot of the tracing backend."""

    model_config = ConfigDict(frozen=True)

    backend: str = Field(description="'langfuse' or 'noop'.")
    configured: bool = False
    reachable: bool | None = Field(
        default=None, description="None means 'not checked' (e.g. tracing disabled)."
    )
    base_url: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return not self.configured or self.reachable is not False


class Span:
    """Handle to an in-flight span.

    Wraps whatever the backend returned, or ``None`` for the no-op case, so callers
    can set attributes unconditionally without null checks at every site.
    """

    __slots__ = ("_raw",)

    def __init__(self, raw: Any = None) -> None:
        self._raw = raw

    @property
    def trace_id(self) -> str | None:
        """Trace id for correlating an API response with its trace. ``None`` if absent."""
        if self._raw is None:
            return None
        value = getattr(self._raw, "trace_id", None)
        return str(value) if value else None

    def update(self, **attributes: Any) -> None:
        """Attach attributes to the span. Silently ignored when tracing is off."""
        if self._raw is None:
            return
        with contextlib.suppress(Exception):
            self._raw.update(**attributes)

    def score(self, name: str, value: float, comment: str | None = None) -> None:
        """Attach a numeric score, used by the Phase 12 evaluation runs."""
        if self._raw is None:
            return
        with contextlib.suppress(Exception):
            self._raw.score(name=name, value=value, comment=comment)


class Tracer:
    """Base tracer. The default implementation is already a safe no-op."""

    backend = "noop"
    configured = False

    @contextlib.contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        """Open a span as a context manager. Never raises, never suppresses."""
        yield Span(None)

    @contextlib.contextmanager
    def generation(self, name: str, **attributes: Any) -> Iterator[Span]:
        """Open a span typed as an LLM generation (token/cost accounting in Langfuse)."""
        yield Span(None)

    def flush(self) -> None:
        """Force-send buffered events. Needed before a short-lived CLI process exits."""

    def shutdown(self) -> None:
        """Release resources."""

    def health(self) -> TracingHealth:
        return TracingHealth(backend=self.backend, configured=False)


class NoOpTracer(Tracer):
    """Explicit no-op, used when tracing is unconfigured, disabled, or broken."""


class LangfuseTracer(Tracer):
    """Langfuse-backed tracer (SDK v3/v4).

    Every SDK interaction is wrapped: if the client cannot be built or a call fails,
    the tracer marks itself degraded and behaves like :class:`NoOpTracer` from then on,
    logging once rather than on every request.
    """

    backend = "langfuse"

    def __init__(self, settings: LangfuseSettings) -> None:
        self._s = settings
        self._client: Any = None
        self._degraded = False
        self.configured = settings.configured

        if not self.configured:
            self._degraded = True
            return

        try:
            from langfuse import Langfuse

            assert settings.public_key is not None  # noqa: S101 - guarded by `configured`
            assert settings.secret_key is not None  # noqa: S101
            self._client = Langfuse(
                public_key=settings.public_key.get_secret_value(),
                secret_key=settings.secret_key.get_secret_value(),
                base_url=settings.effective_base_url,
                tracing_enabled=settings.enabled,
            )
        except ImportError:
            # The package is an optional part of the stack; running without it is fine.
            log.info("tracing.langfuse_not_installed", action="falling back to no-op")
            self._degraded = True
        except Exception as exc:  # noqa: BLE001
            log.warning("tracing.init_failed", error=f"{type(exc).__name__}: {exc}")
            self._degraded = True

    @property
    def _active(self) -> bool:
        return self._client is not None and not self._degraded

    def _degrade(self, exc: Exception) -> None:
        """Disable tracing permanently after a failure, logging exactly once."""
        if not self._degraded:
            self._degraded = True
            log.warning(
                "tracing.degraded",
                error=f"{type(exc).__name__}: {exc}",
                action="tracing disabled for this process; application unaffected",
            )

    @contextlib.contextmanager
    def _observation(self, name: str, as_type: str, attributes: dict[str, Any]) -> Iterator[Span]:
        if not self._active:
            yield Span(None)
            return
        try:
            # Langfuse v4 exposes only `start_as_current_observation(as_type=...)`;
            # the `start_as_current_span` helper documented for v3 is not present.
            cm = self._client.start_as_current_observation(as_type=as_type, name=name, **attributes)
        except Exception as exc:  # noqa: BLE001
            self._degrade(exc)
            yield Span(None)
            return

        try:
            with cm as raw:
                yield Span(raw)
        except Exception:
            # A failure inside the caller's block belongs to the caller: re-raise it
            # unchanged. Only tracing-layer failures are swallowed.
            raise

    @contextlib.contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        with self._observation(name, "span", attributes) as s:
            yield s

    @contextlib.contextmanager
    def generation(self, name: str, **attributes: Any) -> Iterator[Span]:
        with self._observation(name, "generation", attributes) as s:
            yield s

    def flush(self) -> None:
        if self._active:
            try:
                self._client.flush()
            except Exception as exc:  # noqa: BLE001
                self._degrade(exc)

    def shutdown(self) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):
                self._client.shutdown()
            self._client = None

    def health(self) -> TracingHealth:
        """Verify credentials against Langfuse. Never raises."""
        if not self.configured:
            return TracingHealth(
                backend=self.backend,
                configured=False,
                base_url=self._s.effective_base_url,
                error="LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set.",
            )
        if self._client is None:
            return TracingHealth(
                backend=self.backend,
                configured=True,
                reachable=False,
                base_url=self._s.effective_base_url,
                error="Langfuse client could not be constructed.",
            )
        try:
            reachable = bool(self._client.auth_check())
            return TracingHealth(
                backend=self.backend,
                configured=True,
                reachable=reachable,
                base_url=self._s.effective_base_url,
                error=None if reachable else "Langfuse rejected these credentials.",
            )
        except Exception as exc:  # noqa: BLE001
            return TracingHealth(
                backend=self.backend,
                configured=True,
                reachable=False,
                base_url=self._s.effective_base_url,
                error=f"{type(exc).__name__}: {exc}",
            )


@functools.lru_cache(maxsize=1)
def get_tracer() -> Tracer:
    """Return the process-wide tracer, choosing the backend from configuration.

    Cached: one client, one background flush thread. Call ``get_tracer.cache_clear()``
    in tests after changing the environment.
    """
    langfuse_settings = get_settings().langfuse
    if not langfuse_settings.configured:
        return NoOpTracer()
    return LangfuseTracer(langfuse_settings)
