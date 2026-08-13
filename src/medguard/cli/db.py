"""``medguard db`` -- schema migrations.

A thin wrapper over Alembic's Python API rather than a shell-out to the ``alembic``
binary. Two reasons: the wrapper resolves ``alembic.ini`` relative to the project root,
so the commands work from any directory; and going through the API keeps the migration
environment inside this process, where ``DATABASE_URL`` has already been resolved by
:func:`medguard.config.get_settings`.

``revision`` refuses to run without a message. Alembic will happily generate
``a1b2c3d4e5f6_.py``, and a history of unnamed revisions is unreadable exactly when it
matters most -- during an incident, deciding what to roll back to.

Alembic is imported inside each command rather than at module scope. It pulls in
SQLAlchemy's DDL machinery, and paying for that on ``medguard --help`` is the kind of
slow CLI that makes people stop using the CLI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import typer

from medguard.cli.bootstrap import bootstrap, console, err_console

if TYPE_CHECKING:  # pragma: no cover - import exists for typing only
    from alembic.config import Config

__all__ = ["db_app"]

db_app = typer.Typer(
    name="db",
    help="Database schema migrations (Alembic).",
    no_args_is_help=True,
)


def _alembic_config() -> Config:
    """Build an Alembic ``Config`` anchored at the project root."""
    from alembic.config import Config

    from medguard.config import project_root

    root = project_root()
    ini = root / "alembic.ini"
    if not ini.is_file():
        err_console.print(f"[red]alembic.ini not found at {ini}[/]")
        raise typer.Exit(code=1)

    config = Config(str(ini))
    config.set_main_option("script_location", str(root / "migrations"))
    return config


@db_app.command("upgrade")
def upgrade(
    revision: Annotated[
        str, typer.Argument(help="Target revision. 'head' applies everything.")
    ] = "head",
) -> None:
    """Apply migrations up to a revision."""
    bootstrap()
    from alembic import command

    from medguard.storage.database import Database

    config = _alembic_config()

    db = Database()
    console.print(f"Upgrading [bold]{db.safe_url()}[/] → {revision}")
    db.dispose()

    command.upgrade(config, revision)
    console.print("[green]Done.[/]")


@db_app.command("downgrade")
def downgrade(
    revision: Annotated[str, typer.Argument(help="Target revision. 'base' rolls back all.")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")] = False,
) -> None:
    """Roll migrations back to a revision. Destructive: this drops tables."""
    bootstrap()
    from alembic import command

    from medguard.storage.database import Database

    config = _alembic_config()

    db = Database()
    target = db.safe_url()
    db.dispose()

    if not yes and not typer.confirm(
        f"Downgrade {target} to {revision!r}? This drops tables and their contents.",
        default=False,
    ):
        console.print("Aborted.")
        raise typer.Exit(code=1)

    command.downgrade(config, revision)
    console.print("[green]Done.[/]")


@db_app.command("current")
def current() -> None:
    """Show the revision the database is currently stamped at."""
    bootstrap()
    from alembic import command

    command.current(_alembic_config(), verbose=True)


@db_app.command("history")
def history() -> None:
    """Show the migration history."""
    bootstrap()
    from alembic import command

    command.history(_alembic_config(), verbose=True)


@db_app.command("revision")
def revision(
    message: Annotated[
        str,
        typer.Option("--message", "-m", help="What this migration does. Required."),
    ],
    autogenerate: Annotated[
        bool,
        typer.Option(
            "--autogenerate/--empty",
            help="Diff the models against the database instead of writing an empty file.",
        ),
    ] = True,
) -> None:
    """Create a new migration.

    Always review the generated file: autogenerate does not detect table or column
    renames and renders them as a drop plus an add, which loses the data.
    """
    bootstrap()
    from alembic import command

    if not message.strip():
        err_console.print("[red]--message must not be empty.[/]")
        raise typer.Exit(code=1)

    command.revision(_alembic_config(), message=message, autogenerate=autogenerate)
    console.print("[yellow]Review the generated migration before committing it.[/]")
