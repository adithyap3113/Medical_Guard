"""``medguard query`` CLI command."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.panel import Panel
from rich.table import Table

from medguard.cli.bootstrap import bootstrap as _bootstrap
from medguard.cli.bootstrap import console, err_console
from medguard.config import get_settings
from medguard.errors import MedGuardError, VectorStoreError
from medguard.generation.generator import AnswerGenerator
from medguard.models import DocumentType, QueryStrategy, SourceOrganization
from medguard.providers.gemini import GeminiProvider
from medguard.retrieval import RetrievalFilter, get_retriever
from medguard.retrieval.strategies import QueryExecutor
from medguard.storage.vector_store import VectorStore

__all__ = ["query_command"]


def query_command(
    question: Annotated[str, typer.Argument(help="The clinical question to ask MedGuard.")],
    strategy: Annotated[
        QueryStrategy, typer.Option("--strategy", "-s", help="Query strategy to apply.")
    ] = QueryStrategy.BASIC,
    top_k: Annotated[
        int,
        typer.Option("--top-k", "-k", min=1, max=50, help="Number of evidence chunks to retrieve."),
    ] = 5,
    retriever: Annotated[
        str,
        typer.Option("--retriever", "-r", help="Retriever leg: 'hybrid', 'dense', or 'sparse'."),
    ] = "hybrid",
    source_org: Annotated[
        list[SourceOrganization] | None,
        typer.Option("--source-org", help="Filter by source organization (e.g. CDC, WHO)."),
    ] = None,
    doc_type: Annotated[
        list[DocumentType] | None,
        typer.Option("--doc-type", help="Filter by document type (e.g. guideline, reference)."),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Output raw structured JSON instead of Rich panels.")
    ] = False,
) -> None:
    """Answer a medical or clinical question grounded in the guideline corpus."""
    _bootstrap()

    stripped_question = question.strip()
    if not stripped_question:
        err_console.print("[red]Error:[/] Question cannot be empty or contain only whitespace.")
        raise typer.Exit(code=2)

    settings = get_settings()

    retrieval_filter: RetrievalFilter | None = None
    if source_org or doc_type:
        retrieval_filter = RetrievalFilter(
            source_organization=source_org,
            document_type=doc_type,
        )

    # CLI owns its vector store lifecycle independently
    store = VectorStore(settings)
    try:
        try:
            retriever_inst = get_retriever(retriever, store=store, settings=settings)
        except ValueError as exc:
            err_console.print(f"[red]Invalid retriever {retriever!r}:[/] {exc}")
            raise typer.Exit(code=1) from exc

        provider: GeminiProvider | None = None
        if settings.gemini.configured:
            provider = GeminiProvider(settings.gemini)

        with console.status(
            "[bold cyan]Retrieving medical evidence and generating grounded response..."
        ):
            try:
                executor = QueryExecutor(
                    retriever_inst, provider=provider, settings=settings.retrieval
                )
                retrieval_res = executor.execute(
                    stripped_question,
                    strategy=strategy,
                    top_k=top_k,
                    retrieval_filter=retrieval_filter,
                )
            except VectorStoreError as exc:
                err_console.print(f"[red]Knowledge index error:[/] {exc.user_message}")
                err_console.print(f"[dim]Technical details: {exc.detail}[/]")
                raise typer.Exit(code=1) from exc
            except MedGuardError as exc:
                err_console.print(f"[red]Retrieval failed:[/] {exc.user_message}")
                err_console.print(f"[dim]Technical details: {exc.detail}[/]")
                raise typer.Exit(code=1) from exc

            if provider is not None:
                try:
                    generator = AnswerGenerator(provider, settings=settings)
                    answer = generator.generate(stripped_question, retrieval_res)
                except MedGuardError as exc:
                    err_console.print(f"[red]Generation failed:[/] {exc.user_message}")
                    err_console.print(f"[dim]Technical details: {exc.detail}[/]")
                    raise typer.Exit(code=1) from exc
            else:
                if not retrieval_res.candidates:
                    ans_text = (
                        "No relevant guidelines were found in the index for this query. "
                        "(GEMINI_API_KEY unconfigured)."
                    )
                    abstained = True
                else:
                    top_sources = ", ".join(
                        f"{c.chunk.document_id} (p. {c.chunk.page_start})"
                        for c in retrieval_res.candidates[:3]
                    )
                    ans_text = (
                        f"Retrieved {len(retrieval_res.candidates)} chunk(s) from {top_sources}. "
                        "(Note: GEMINI_API_KEY is not configured for text generation; configure it in .env)."
                    )
                    abstained = False

                from medguard.models import MEDICAL_DISCLAIMER, Answer

                answer = Answer(
                    question=stripped_question,
                    text=ans_text,
                    citations=[],
                    retrieval=retrieval_res,
                    strategy=strategy,
                    abstained=abstained,
                    disclaimer=MEDICAL_DISCLAIMER,
                )
    finally:
        store.close()

    if json_output:
        console.print_json(data=answer.model_dump(mode="json"))
        return

    # Render formatted Rich interface
    console.print()
    panel_title = (
        f"[bold green]MedGuard Response[/] [dim]({retrieval_res.strategy.value} | {retriever})[/]"
    )
    if answer.abstained:
        panel_title = f"[bold yellow]MedGuard Response (Abstained)[/] [dim]({retrieval_res.strategy.value})[/]"

    console.print(
        Panel(
            answer.text,
            title=panel_title,
            border_style="cyan" if not answer.abstained else "yellow",
        )
    )

    if answer.citations:
        table = Table(title="[bold]Supporting Evidence & Citations[/]", border_style="dim")
        table.add_column("Marker", style="cyan", width=8)
        table.add_column("Document", style="green")
        table.add_column("Pages", style="magenta", width=12)
        table.add_column("Status", style="bold", width=14)

        for cit in answer.citations:
            if cit.validation_status == "quote_verified" or cit.supported is True:
                badge = "[green]Quote Verified[/]"
            elif cit.validation_status == "source_valid":
                badge = "[yellow]Source Valid[/]"
            else:
                badge = "[red]Unsupported[/]"
            pages = (
                f"{cit.page_start}-{cit.page_end}"
                if cit.page_start and cit.page_end and cit.page_start != cit.page_end
                else str(cit.page_start or "-")
            )
            table.add_row(cit.marker, cit.document_title or cit.document_id, pages, badge)

        console.print()
        console.print(table)

    console.print()
    console.print(f"[dim italic]{answer.disclaimer}[/]")
    console.print()
