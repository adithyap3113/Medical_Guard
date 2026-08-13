"""``medguard`` command-line entry point.

Phases 1-2 implement ``doctor``, ``serve``, ``corpus`` and ``db``. ``ingest``, ``query``
and ``eval`` are declared with their real signatures but raise ``NotImplementedError``.

That is deliberate. The alternative -- omitting them until their phase -- hides the
shape of the system, while a stub that silently returns nothing is worse than one that
fails loudly. These stubs state which phase implements them, so an accidental call
produces an explanation rather than a confusing no-op.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from medguard import __version__
from medguard.cli.bootstrap import bootstrap as _bootstrap
from medguard.cli.bootstrap import console, err_console
from medguard.cli.corpus import corpus_app
from medguard.cli.db import db_app
from medguard.config import get_settings
from medguard.models import MEDICAL_DISCLAIMER, QueryStrategy

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


# ---------------------------------------------------------------------------
# Declared now, implemented in later phases.
# ---------------------------------------------------------------------------


@app.command()
def ingest(
    source: Annotated[Path, typer.Argument(help="PDF file or directory of PDFs to ingest.")] = Path(
        "data/raw"
    ),
    recreate: Annotated[
        bool, typer.Option(help="Drop and rebuild the collection before ingesting.")
    ] = False,
) -> None:
    """[dim]Phase 3[/] — Parse, chunk, embed and index documents."""
    _bootstrap()
    err_console.print(
        f"[yellow]Not implemented yet.[/] Ingestion lands in Phase 3 (would ingest: {source})."
    )
    raise NotImplementedError("Ingestion is implemented in Phase 3.")


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="The question to answer.")],
    strategy: Annotated[
        QueryStrategy, typer.Option(help="Query-processing strategy.")
    ] = QueryStrategy.BASIC,
    top_k: Annotated[int, typer.Option(min=1, max=100, help="Chunks to retrieve.")] = 5,
) -> None:
    """[dim]Phase 5-11[/] — Answer a question with cited evidence."""
    _bootstrap()
    err_console.print(
        f"[yellow]Not implemented yet.[/] Retrieval and grounded generation land in "
        f"Phases 5-11 (question: {question!r}, strategy: {strategy.value}, top_k: {top_k})."
    )
    err_console.print(f"[dim]{MEDICAL_DISCLAIMER}[/]")
    raise NotImplementedError("Querying is implemented in Phases 5-11.")


@app.command(name="eval")
def eval_(
    dataset: Annotated[Path, typer.Option(help="Evaluation dataset.")] = Path(
        "data/eval/questions.jsonl"
    ),
    limit: Annotated[int | None, typer.Option(help="Evaluate only the first N questions.")] = None,
) -> None:
    """[dim]Phase 12[/] — Run retrieval and generation evaluation."""
    _bootstrap()
    err_console.print(
        f"[yellow]Not implemented yet.[/] Evaluation lands in Phase 12 "
        f"(dataset: {dataset}, limit: {limit})."
    )
    raise NotImplementedError("Evaluation is implemented in Phase 12.")


if __name__ == "__main__":  # pragma: no cover
    app()
