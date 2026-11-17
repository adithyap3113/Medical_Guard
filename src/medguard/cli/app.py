"""``medguard`` command-line entry point.

``doctor``, ``serve``, ``corpus``, ``db``, ``ingest`` and ``eval`` are implemented.
``query`` is declared with its real signature but raises ``NotImplementedError``.

That is deliberate. The alternative -- omitting it until its phase -- hides the shape of
the system, while a stub that silently returns nothing is worse than one that fails
loudly. The stub states which phase implements it, so an accidental call produces an
explanation rather than a confusing no-op.
"""

from __future__ import annotations

from typing import Annotated

import typer

from medguard import __version__
from medguard.cli.bootstrap import bootstrap as _bootstrap
from medguard.cli.bootstrap import console
from medguard.cli.corpus import corpus_app
from medguard.cli.db import db_app
from medguard.cli.eval import eval_
from medguard.cli.ingest import ingest
from medguard.cli.query import query_command
from medguard.config import get_settings

__all__ = ["app"]

app = typer.Typer(
    name="medguard",
    help="MedGuard — medical knowledge RAG and evidence retrieval system.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)

app.add_typer(corpus_app)
app.add_typer(db_app)

# Registered as functions rather than decorated in place, so that the heavy imports
# their bodies need stay out of `medguard --help`.
app.command(name="eval")(eval_)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"medguard {__version__}")
        raise typer.Exit


@app.callback()
def main(
    _version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """MedGuard CLI."""
    _bootstrap()


@app.command()
def doctor(
    smoke_test: Annotated[
        bool,
        typer.Option(
            "--smoke-test/--no-smoke-test",
            help="Spend one real Gemini call to prove the key can generate text. "
            "Off by default because it consumes free-tier quota.",
        ),
    ] = False,
    show_config: Annotated[
        bool,
        typer.Option("--show-config", help="Also print effective settings, secrets redacted."),
    ] = False,
    skip_hardware: Annotated[
        bool, typer.Option("--skip-hardware", help="Skip CPU/RAM/GPU/Docker probes.")
    ] = False,
) -> None:
    """Check the environment: hardware, Docker, Qdrant, database, Gemini, Langfuse.

    This is the Phase 1 acceptance test. Exits non-zero only if a *critical* check
    fails; missing optional services produce warnings, because an unconfigured API key
    is a legitimate state, not a broken install.
    """
    _bootstrap()
    from medguard.cli.doctor import render_report, render_settings
    from medguard.diagnostics import run_diagnostics

    settings = get_settings()
    report = run_diagnostics(settings, smoke_test=smoke_test, include_hardware=not skip_hardware)
    render_report(console, report)

    if show_config:
        console.print()
        render_settings(console, settings)

    raise typer.Exit(code=report.exit_code())


@app.command()
def serve(
    host: Annotated[str | None, typer.Option(help="Bind address. Defaults to API_HOST.")] = None,
    port: Annotated[int | None, typer.Option(help="Port. Defaults to API_PORT.")] = None,
    reload: Annotated[bool, typer.Option(help="Auto-reload on code changes.")] = False,
) -> None:
    """Run the FastAPI application."""
    _bootstrap()
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "medguard.api.main:create_app",
        factory=True,
        host=host or settings.app.api_host,
        port=port or settings.app.api_port,
        reload=reload,
        log_config=None,  # keep our structlog pipeline, including redaction
    )


# Registered from its own module: the command body imports the pipeline, the embedders
# and ONNX Runtime, and none of that should load for `medguard --help`.
app.command()(ingest)
app.command(name="query")(query_command)


if __name__ == "__main__":  # pragma: no cover
    app()
