"""Migration correctness.

Two things are being protected here.

**Upgrade/downgrade actually round-trips.** A downgrade nobody ever runs is a downgrade
that does not work, and the moment you need one is the moment you cannot test it.

**``render_as_batch`` stays on.** SQLite implements almost none of ``ALTER TABLE``, so
without batch mode the first migration that alters a column fails outright. It has to
be enabled before the migrations that need it exist, which makes it exactly the kind of
setting that gets silently dropped during a refactor -- hence a test.
"""

from __future__ import annotations

import pytest
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect

from medguard.config import Settings, project_root


@pytest.fixture
def alembic_config(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Config:
    """Alembic pointed at the isolated per-test SQLite database."""
    root = project_root()
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "migrations"))
    # migrations/env.py reads the URL from settings, which conftest has already
    # redirected at a temporary file.
    monkeypatch.setenv("DATABASE_URL", settings.app.database_url)
    return config


class TestMigrationHistory:
    def test_there_is_exactly_one_head(self) -> None:
        """Two heads mean a branched history that `upgrade head` cannot resolve."""
        root = project_root()
        config = Config(str(root / "alembic.ini"))
        config.set_main_option("script_location", str(root / "migrations"))
        assert len(ScriptDirectory.from_config(config).get_heads()) == 1

    def test_every_revision_has_a_message(self) -> None:
        """An unnamed revision is unreadable exactly when it matters -- during rollback."""
        root = project_root()
        config = Config(str(root / "alembic.ini"))
        config.set_main_option("script_location", str(root / "migrations"))
        for script in ScriptDirectory.from_config(config).walk_revisions():
            assert script.doc and script.doc.strip()


class TestUpgradeDowngrade:
    def test_upgrade_head_creates_both_tables(
        self, alembic_config: Config, settings: Settings
    ) -> None:
        command.upgrade(alembic_config, "head")

        engine = create_engine(settings.app.database_url)
        try:
            tables = set(inspect(engine).get_table_names())
        finally:
            engine.dispose()

        assert {"documents", "chunks"} <= tables

    def test_the_database_is_stamped_at_head(
        self, alembic_config: Config, settings: Settings
    ) -> None:
        command.upgrade(alembic_config, "head")
        head = ScriptDirectory.from_config(alembic_config).get_current_head()

        engine = create_engine(settings.app.database_url)
        try:
            with engine.connect() as conn:
                assert MigrationContext.configure(conn).get_current_revision() == head
        finally:
            engine.dispose()

    def test_downgrade_base_removes_both_tables(
        self, alembic_config: Config, settings: Settings
    ) -> None:
        command.upgrade(alembic_config, "head")
        command.downgrade(alembic_config, "base")

        engine = create_engine(settings.app.database_url)
        try:
            tables = set(inspect(engine).get_table_names())
        finally:
            engine.dispose()

        assert "documents" not in tables
        assert "chunks" not in tables

    def test_re_upgrading_is_idempotent(self, alembic_config: Config, settings: Settings) -> None:
        """`upgrade head` must be safe to run on an already-current database."""
        command.upgrade(alembic_config, "head")
        command.upgrade(alembic_config, "head")

        engine = create_engine(settings.app.database_url)
        try:
            tables = set(inspect(engine).get_table_names())
        finally:
            engine.dispose()

        assert {"documents", "chunks"} <= tables

    def test_a_full_cycle_round_trips(self, alembic_config: Config, settings: Settings) -> None:
        command.upgrade(alembic_config, "head")
        command.downgrade(alembic_config, "base")
        command.upgrade(alembic_config, "head")

        engine = create_engine(settings.app.database_url)
        try:
            tables = set(inspect(engine).get_table_names())
        finally:
            engine.dispose()

        assert {"documents", "chunks"} <= tables


class TestSchemaShape:
    def test_the_migration_matches_the_orm_models(
        self, alembic_config: Config, settings: Settings
    ) -> None:
        """Guards against hand-editing a model and forgetting the migration.

        Autogenerate comparing the models against a fully-migrated database must find
        nothing left to do.
        """
        from alembic.autogenerate import compare_metadata
        from alembic.migration import MigrationContext as MC

        from medguard.storage.database import metadata

        command.upgrade(alembic_config, "head")
        engine = create_engine(settings.app.database_url)
        try:
            with engine.connect() as conn:
                diff = compare_metadata(MC.configure(conn), metadata)
        finally:
            engine.dispose()

        assert diff == [], f"models and migrations disagree: {diff}"

    def test_constraints_are_named(self, alembic_config: Config, settings: Settings) -> None:
        """Batch migrations on SQLite cannot drop a constraint they cannot name."""
        command.upgrade(alembic_config, "head")

        engine = create_engine(settings.app.database_url)
        try:
            inspector = inspect(engine)
            unique = inspector.get_unique_constraints("chunks")
            foreign = inspector.get_foreign_keys("chunks")
        finally:
            engine.dispose()

        assert any(c["name"] == "uq_chunks_document_id_chunk_index" for c in unique)
        assert all(fk["name"] for fk in foreign)


class TestPageRangeMigration:
    """The second migration: `chunks.page` becomes `page_start` / `page_end`.

    Column-altering migrations are exactly what `render_as_batch` exists for, so this
    is also the first real exercise of that setting rather than a claim about it.
    """

    def test_head_has_the_page_range_columns(
        self, alembic_config: Config, settings: Settings
    ) -> None:
        command.upgrade(alembic_config, "head")

        engine = create_engine(settings.app.database_url)
        try:
            columns = {c["name"] for c in inspect(engine).get_columns("chunks")}
        finally:
            engine.dispose()

        assert {"page_start", "page_end"} <= columns
        assert "page" not in columns

    def test_downgrading_restores_the_single_page_column(
        self, alembic_config: Config, settings: Settings
    ) -> None:
        command.upgrade(alembic_config, "head")
        command.downgrade(alembic_config, "-1")

        engine = create_engine(settings.app.database_url)
        try:
            columns = {c["name"] for c in inspect(engine).get_columns("chunks")}
        finally:
            engine.dispose()

        assert "page" in columns
        assert "page_start" not in columns
        assert "page_end" not in columns

    def test_the_check_constraint_exists_at_head(
        self, alembic_config: Config, settings: Settings
    ) -> None:
        command.upgrade(alembic_config, "head")

        engine = create_engine(settings.app.database_url)
        try:
            names = {c["name"] for c in inspect(engine).get_check_constraints("chunks")}
        finally:
            engine.dispose()

        assert "ck_chunks_page_range_ordered" in names

    def test_existing_data_survives_the_upgrade(
        self, alembic_config: Config, settings: Settings
    ) -> None:
        """A migration that only works on an empty table will be run once and be wrong.

        The `chunks` table happens to be empty today, which is exactly why this is
        asserted now rather than discovered later.
        """
        from sqlalchemy import text

        command.upgrade(alembic_config, "cc54fe498aaa")

        engine = create_engine(settings.app.database_url)
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO documents (document_id,title,source_url,"
                        "source_organization,document_type,local_filename,sha256,"
                        "file_size_bytes,retrieved_at,ingestion_status,"
                        "license_or_usage_note,created_at,updated_at) VALUES "
                        "('d','t','u','CDC','GUIDELINE','d.pdf',:h,1,'2026-01-01',"
                        "'pending','x','2026-01-01','2026-01-01')"
                    ),
                    {"h": "a" * 64},
                )
                conn.execute(
                    text(
                        "INSERT INTO chunks (chunk_id,document_id,chunk_index,page,text,"
                        "char_count,chunk_metadata,created_at) VALUES "
                        "('d::0000','d',0,41,'x',1,'{}','2026-01-01')"
                    )
                )
        finally:
            engine.dispose()

        command.upgrade(alembic_config, "head")

        engine = create_engine(settings.app.database_url)
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text("SELECT page_start, page_end FROM chunks WHERE chunk_id = 'd::0000'")
                ).one()
        finally:
            engine.dispose()

        # A single recorded page becomes a one-page range, which is what it meant.
        assert tuple(row) == (41, 41)

    def test_downgrading_keeps_the_first_page_rather_than_dropping_it(
        self, alembic_config: Config, settings: Settings
    ) -> None:
        """Lossy by necessity -- `page_end` has nowhere to go -- but not destructive."""
        from sqlalchemy import text

        command.upgrade(alembic_config, "head")

        engine = create_engine(settings.app.database_url)
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO documents (document_id,title,source_url,"
                        "source_organization,document_type,local_filename,sha256,"
                        "file_size_bytes,retrieved_at,ingestion_status,"
                        "license_or_usage_note,created_at,updated_at) VALUES "
                        "('d','t','u','CDC','GUIDELINE','d.pdf',:h,1,'2026-01-01',"
                        "'pending','x','2026-01-01','2026-01-01')"
                    ),
                    {"h": "a" * 64},
                )
                conn.execute(
                    text(
                        "INSERT INTO chunks (chunk_id,document_id,chunk_index,page_start,"
                        "page_end,text,char_count,chunk_metadata,created_at) VALUES "
                        "('d::0000','d',0,41,43,'x',1,'{}','2026-01-01')"
                    )
                )
        finally:
            engine.dispose()

        command.downgrade(alembic_config, "-1")

        engine = create_engine(settings.app.database_url)
        try:
            with engine.connect() as conn:
                page = conn.execute(
                    text("SELECT page FROM chunks WHERE chunk_id = 'd::0000'")
                ).scalar_one()
        finally:
            engine.dispose()

        assert page == 41


class TestBatchMode:
    def test_render_as_batch_is_enabled(self) -> None:
        """SQLite has no real ALTER TABLE.

        Retrofitting batch mode after migrations exist means rewriting history, so this
        asserts it is configured rather than trusting that nobody removes it.
        """
        env = (project_root() / "migrations" / "env.py").read_text(encoding="utf-8")
        # Counted as code lines, not substring occurrences: the module docstring
        # explains the setting, and a prose mention must not satisfy this assertion.
        settings_lines = [
            line for line in env.splitlines() if line.strip() == "render_as_batch=True,"
        ]
        # Once for offline mode, once for online. Both paths need it.
        assert len(settings_lines) == 2

    def test_the_baseline_migration_uses_batch_operations(self) -> None:
        versions = sorted((project_root() / "migrations" / "versions").glob("*.py"))
        assert versions, "no migration files found"
        assert any("batch_alter_table" in p.read_text(encoding="utf-8") for p in versions)


class TestEnvConfiguration:
    def test_the_url_comes_from_settings_not_the_ini_file(self) -> None:
        """Two sources of truth for the database URL is how you migrate the wrong one."""
        ini = (project_root() / "alembic.ini").read_text(encoding="utf-8")
        active = [line for line in ini.splitlines() if line.strip().startswith("sqlalchemy.url")]
        assert active == []

        env = (project_root() / "migrations" / "env.py").read_text(encoding="utf-8")
        assert "get_settings()" in env
