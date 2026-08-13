"""Relational store adapter.

SQLite is the default because it needs no server, no container and no RAM, which
matters on this machine. PostgreSQL is one environment variable away:

    DATABASE_URL=postgresql+psycopg://medguard:medguard@localhost:5432/medguard

Both paths go through the same SQLAlchemy engine, so the application code cannot tell
the difference and nothing has to be rewritten to switch. Phase 1 defined the engine
and a health check; Phase 2 adds the declarative metadata and a session factory, so
Alembic has something to migrate and ingestion has somewhere to write.

``Base`` is re-exported here rather than defined here. The tables live in
:mod:`medguard.storage.schema`, but ``migrations/env.py`` and application code both
want one obvious import path for the metadata, and importing this module guarantees
the table definitions have been registered on it.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Engine, MetaData, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from medguard.config import Settings, get_settings
from medguard.storage.schema import Base

__all__ = ["Base", "Database", "DatabaseHealth", "metadata"]

#: What Alembic autogenerate compares the live database against.
metadata: MetaData = Base.metadata


def _enable_sqlite_foreign_keys(engine: Engine) -> None:
    """Turn on foreign-key enforcement for every SQLite connection.

    SQLite ignores foreign keys unless ``PRAGMA foreign_keys=ON`` is issued, and the
    pragma is per-connection rather than per-database. Without this, the ``ON DELETE
    CASCADE`` between chunks and documents is inert: deleting a document leaves its
    chunks behind as orphans that still match retrieval queries. The schema would look
    correct while enforcing nothing -- the worst combination -- and PostgreSQL would
    behave differently from SQLite for the same code.
    """

    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()


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
        self._session_factory: sessionmaker[Session] | None = None

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
                _enable_sqlite_foreign_keys(self._engine)
            else:
                self._engine = create_engine(self.url, pool_pre_ping=True)
        return self._engine

    @property
    def session_factory(self) -> sessionmaker[Session]:
        """Configured session factory, built once per :class:`Database`.

        ``expire_on_commit=False`` because the common pattern here is "write rows, then
        report on them". With the default, every attribute access after ``commit()``
        issues a fresh SELECT, which turns a summary line into N queries.
        """
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine, expire_on_commit=False, future=True
            )
        return self._session_factory

    @contextmanager
    def session(self) -> Iterator[Session]:
        """A transactional scope: commit on success, roll back on any exception.

        Callers get a session that cannot leak a half-applied write, which matters
        for ingestion where one document failing must not leave its chunks behind.
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    def create_all(self) -> None:
        """Create every table directly, bypassing migrations.

        For tests and throwaway databases only. Real databases are built by
        ``medguard db upgrade`` so that their schema has a recorded version; a
        database created this way has no Alembic stamp and will confuse the next
        migration.
        """
        Base.metadata.create_all(self.engine)

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
        self._session_factory = None
