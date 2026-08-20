"""``medguard ingest`` — the command that fills the index.

Kept out of ``cli/app.py`` for the same reason as ``corpus`` and ``db``: the body needs
the pipeline, the pipeline needs the embedders, and importing those at module scope
would make ``medguard --help`` load ONNX Runtime. The heavy imports happen inside the
command body; Rich and Typer are already loaded by the time any command runs, so those
stay at the top where they can be typed properly.

Registered as a top-level command rather than a sub-app. ``ingest`` is one verb with one
meaning; ``corpus`` and ``db`` are sub-apps because each is a family of related verbs.

Exit codes are the contract with CI: 0 when everything was ingested or skipped, 1 when
any document failed. A run that indexes eighteen of twenty documents has not succeeded,
and a pipeline reporting success over a partial corpus is worse than one that fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from medguard.cli.bootstrap import bootstrap as _bootstrap
from medguard.cli.bootstrap import console, err_console

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from medguard.config import IngestionSettings
    from medguard.ingestion.pipeline import IngestionSummary
    from medguard.storage.schema import DocumentRow

__all__ = ["ingest"]

#: Below this much available memory, ingestion refuses to start. The embedder needs
#: headroom for the ONNX session plus the transient attention allocation, and a process
#: killed by the OS partway through leaves a half-populated collection rather than a
#: clean failure. Overridable, because the person at the keyboard may know better than a
#: threshold does.
PREFLIGHT_RAM_MB = 300

#: Set to bypass the preflight. An environment variable rather than a flag: it is an
#: override for one machine, not a routine choice, and it should look like one.
FORCE_ENV = "MEDGUARD_INGEST_FORCE"

#: How much of a sample chunk to print in a dry run.
_PREVIEW_CHARS = 280


class DryRunEmbedder:
    """Satisfies the embedder protocols; raises if anything actually calls them.

    A dry run answers a question about chunk boundaries, and must not download 130 MB of
    model weights to do it. Returning an object that refuses -- rather than ``None`` --
    keeps the pipeline's type contract intact and turns "the dry-run path touched a
    model" into a loud failure rather than a slow one.
    """

    def __init__(self, kind: str) -> None:
        self.model_id = f"<dry-run {kind}>"
        self.dimension = 0
        self._kind = kind

    def _refuse(self) -> None:
        msg = f"the {self._kind} embedder must not be used during a dry run"
        raise AssertionError(msg)

    def embed_documents(self, texts: list[str]) -> Any:
        self._refuse()

    def embed_query(self, text: str) -> Any:
        self._refuse()


class DryRunVectorStore:
    """Read-only stand-in for the vector store. Every write raises.

    The pipeline already guards its own writes behind ``dry_run``. This exists so that
    "a dry run writes nothing to Qdrant" is a property the tests can *prove* from the
    outside, rather than a claim the code makes about itself.
    """

    def count_for_document(self, document_id: str) -> int:
        return 0

    def collection_points_count(self) -> int:
        return 0

    def _refuse(self, operation: str) -> None:
        msg = f"{operation} must not be called during a dry run"
        raise AssertionError(msg)

    def ensure_collection(self) -> None:
        self._refuse("ensure_collection")

    def upsert_chunks(self, chunks: list[Any]) -> None:
        self._refuse("upsert_chunks")

    def delete_by_document(self, document_id: str) -> int:
        self._refuse("delete_by_document")
        raise AssertionError  # pragma: no cover - _refuse always raises


def ingest(
    source: Annotated[
        Path | None,
        typer.Argument(
            help="Directory holding the corpus PDFs. Defaults to CORPUS_DATA_DIR.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Parse and chunk only. Loads no model, writes nothing to either store.",
        ),
    ] = False,
    only: Annotated[str | None, typer.Option("--only", help="Ingest a single document_id.")] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", min=1, help="Ingest at most this many documents.")
    ] = None,
    force_recreate: Annotated[
        bool,
        typer.Option(
            "--force-recreate",
            help="DESTRUCTIVE: drop the collection and re-embed everything from scratch.",
        ),
    ] = False,
) -> None:
    """Parse, chunk, embed and index the corpus into Qdrant.

    Idempotent. A document is skipped when it is already indexed, its file still hashes
    to what the manifest recorded, and Qdrant holds exactly as many points for it as the
    database has chunks. Re-running over an unchanged corpus does no work.
    """
    _bootstrap()

    import os
    import time

    from medguard.config import get_settings
    from medguard.diagnostics import available_ram_mb
    from medguard.errors import MedGuardError
    from medguard.ingestion.pipeline import IngestionPipeline, IngestionSummary
    from medguard.storage.database import Database
    from medguard.storage.vector_store import VectorStore

    settings = get_settings()
    # No argument means "wherever the corpus is configured to live". A literal
    # `data/raw` default would resolve against the project root and quietly ignore
    # CORPUS_DATA_DIR, so setting that variable would appear to do nothing -- the
    # command would report every document missing from a directory nobody named.
    raw_dir = settings.corpus.raw_dir if source is None else settings.corpus.resolved(source)

    manifest_path = settings.corpus.manifest_path
    if not manifest_path.is_file():
        err_console.print(
            f"[red]No manifest at {manifest_path}.[/] Run `medguard corpus download` first."
        )
        raise typer.Exit(code=1)

    if not dry_run and not os.environ.get(FORCE_ENV):
        available = available_ram_mb()
        if available < PREFLIGHT_RAM_MB:
            err_console.print(
                f"[red]Available RAM below {PREFLIGHT_RAM_MB} MB[/] "
                f"({available:.0f} MB free). Free memory or run with --dry-run to test "
                f"parsing and chunking only. Override with {FORCE_ENV}=1."
            )
            raise typer.Exit(code=1)

    if force_recreate and not dry_run:
        console.print(
            "[yellow]--force-recreate[/] deletes every point in "
            f"[bold]{settings.collection_name}[/] and re-embeds the whole corpus."
        )
        if not typer.confirm("Drop the collection and rebuild it?"):
            console.print("[dim]Nothing was changed.[/]")
            raise typer.Exit(code=0)

    database = Database(settings)
    real_store = VectorStore(settings)
    exit_code = 0

    try:
        store: Any = DryRunVectorStore() if dry_run else real_store
        if not dry_run:
            try:
                if force_recreate:
                    real_store.recreate_collection()
                else:
                    real_store.ensure_collection()
            except MedGuardError as exc:
                err_console.print(f"[red]{exc.detail}[/]")
                raise typer.Exit(code=1) from exc

        dense, sparse = _embedders(dry_run=dry_run)

        with database.session() as session:
            pipeline = IngestionPipeline(
                session=session,
                vector_store=store,
                dense_embedder=dense,
                sparse_embedder=sparse,
                settings=settings.ingestion,
                raw_dir=raw_dir,
            )

            added = pipeline.sync_documents_from_manifest(manifest_path)
            console.print(f"Corpus: [bold]{added}[/] new document(s) recorded from {raw_dir}")

            documents = _selected(session, only=only, limit=limit)
            if not documents:
                console.print("[yellow]Nothing to ingest.[/]")
                raise typer.Exit(code=0)

            started = time.perf_counter()
            results = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("Ingesting", total=len(documents))
                for document in documents:
                    progress.update(task, description=f"Ingesting {document.document_id}")
                    results.append(pipeline.ingest_document(document, dry_run=dry_run))
                    progress.advance(task)

            summary = IngestionSummary(
                results=results,
                duration_s=time.perf_counter() - started,
                dry_run=dry_run,
            )
            _render(console, summary)
            if dry_run:
                _render_samples(console, session, summary, settings.ingestion, raw_dir)
            exit_code = summary.exit_code()
    finally:
        real_store.close()
        database.dispose()

    raise typer.Exit(code=exit_code)


def _embedders(*, dry_run: bool) -> tuple[Any, Any]:
    """The embedder pair, or a pair that refuses to be used."""
    if dry_run:
        return DryRunEmbedder("dense"), DryRunEmbedder("sparse")

    from medguard.embeddings import get_dense_embedder, get_sparse_embedder

    return get_dense_embedder(), get_sparse_embedder()


def _selected(session: Session, *, only: str | None, limit: int | None) -> list[DocumentRow]:
    from sqlalchemy import select

    from medguard.storage.schema import DocumentRow as Row

    statement = select(Row).order_by(Row.document_id)
    if only:
        statement = statement.where(Row.document_id == only)
    documents = list(session.scalars(statement).all())
    return documents[:limit] if limit is not None else documents


def _render(out: Console, summary: IngestionSummary) -> None:
    """Print the per-document outcome table and the one-line summary."""
    from medguard.ingestion.pipeline import IngestionOutcome

    table = Table(title="Ingestion", show_lines=False)
    table.add_column("Document", overflow="fold")
    table.add_column("Outcome")
    table.add_column("Pages", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Note", overflow="fold")

    colour = {
        IngestionOutcome.INGESTED: "green",
        IngestionOutcome.SKIPPED: "dim",
        IngestionOutcome.FAILED: "red",
    }
    for result in summary.results:
        table.add_row(
            result.document_id,
            f"[{colour[result.outcome]}]{result.outcome.value}[/]",
            str(result.page_count if result.page_count is not None else "-"),
            str(result.chunk_count),
            result.reason or "",
        )

    out.print(table)
    out.print(
        f"[green]{len(summary.ingested)} ingested[/] · "
        f"[dim]{len(summary.skipped)} skipped[/] · "
        f"[red]{len(summary.failed)} failed[/] · "
        f"{summary.total_chunks} chunks · {summary.duration_s:.2f}s"
        + (" [yellow](dry run — nothing written)[/]" if summary.dry_run else "")
    )


def _render_samples(
    out: Console,
    session: Session,
    summary: IngestionSummary,
    settings: IngestionSettings,
    raw_dir: Path,
) -> None:
    """Show real chunk boundaries from the first document of a dry run.

    Counts do not tell you whether the chunker is working. A chunker that splits every
    400 characters mid-sentence produces exactly the same chunk count as one that splits
    on paragraphs, and reading the text either side of a boundary is the only reliable
    way to tell them apart -- which is the entire point of having a dry run.
    """
    first = next((r for r in summary.results if r.chunk_count), None)
    if first is None or settings.dry_run_sample_chunks <= 0:
        return

    from medguard.ingestion.chunker import chunk_document
    from medguard.ingestion.cleaner import clean_document
    from medguard.ingestion.parser import parse
    from medguard.storage.schema import DocumentRow as Row

    row = session.get(Row, first.document_id)
    if row is None:  # pragma: no cover - it was selected from this table a moment ago
        return

    pages = clean_document(list(parse(raw_dir / row.local_filename).pages()), settings)
    chunks = chunk_document(first.document_id, pages, settings)
    if not chunks:  # pragma: no cover - the result said it had chunks
        return

    out.print(f"\n[bold]Sample chunks from {first.document_id}[/]")
    for chunk in chunks[: settings.dry_run_sample_chunks]:
        label = (
            f"p{chunk.page_start}"
            if chunk.page_start == chunk.page_end
            else f"pp{chunk.page_start}-{chunk.page_end}"
        )
        tail = "…" if len(chunk.text) > _PREVIEW_CHARS else ""
        out.print(
            f"[dim]{chunk.chunk_id} · {label} · {len(chunk.text)} chars[/]\n"
            f"{chunk.text[:_PREVIEW_CHARS]}{tail}\n"
        )
