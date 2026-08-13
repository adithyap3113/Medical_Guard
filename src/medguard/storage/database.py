"""Relational store adapter.

SQLite is the default because it needs no server, no container and no RAM, which
matters on this machine. PostgreSQL is one environment variable away:

    DATABASE_URL=postgresql+psycopg://medguard:medguard@localhost:5432/medguard

Both paths go through the same SQLAlchemy engine, so the application code cannot tell
the difference and nothing has to be rewritten to switch. Phase 1 defines the engine
and a health check; ORM tables arrive with ingestion in Phase 3, when there is
something to store.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.pool import StaticPool

from medguard.config import Settings, get_settings

__all__ = ["Database", "DatabaseHealth"]


class DatabaseHealth(BaseModel):
    """Diagnostic snapshot of the relational store, with the URL always redacted."""

    model_config = ConfigDict(frozen=True)

    reachable: bool = False
    dialect: str
    url: str = Field(description="Sanitised: any password component is removed.")
    latency_ms: float | None = Field(default=None, ge=0)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.reachable


class Database:
    """Owns the SQLAlchemy engine and exposes a connectivity check."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._s = settings if settings is not None else get_settings()
        self._engine: Engine | None = None

    @property
    def url(self) -> str:
        return self._s.app.database_url

    @property
    def dialect(self) -> str:
        return self.url.split(":", 1)[0]

    def safe_url(self) -> str:
        """The URL with any embedded password stripped, safe to log or display."""
        return str(create_engine(self.url).url.render_as_string(hide_password=True))

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            if self.url.startswith("sqlite"):
                # SQLite needs explicit settings to behave under a threaded server:
                # allow cross-thread use, and keep a single shared connection for
                # in-memory databases so each thread does not get its own empty one.
                is_memory = ":memory:" in self.url
                self._engine = create_engine(
                    self.url,
                    connect_args={"check_same_thread": False},
                    poolclass=StaticPool if is_memory else None,
                )
            else:
                self._engine = create_engine(self.url, pool_pre_ping=True)
        return self._engine

    def health(self) -> DatabaseHealth:
        """Run ``SELECT 1``. Never raises."""
        started = time.perf_counter()
        try:
            safe = self.safe_url()
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as exc:  # noqa: BLE001 - any failure means "unreachable"
            return DatabaseHealth(
                reachable=False,
                dialect=self.dialect,
                url=self.url.split("://")[0] + "://…",
                latency_ms=(time.perf_counter() - started) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )
        return DatabaseHealth(
            reachable=True,
            dialect=self.dialect,
            url=safe,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def dispose(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
