"""``medguard eval`` — run retrieval over the golden set and write a report.

Kept out of ``cli/app.py`` for the same reason as ``ingest``: the body needs the
retrievers, the retrievers need the embedders, and importing those at module scope would
make ``medguard --help`` load ONNX Runtime.

Exit codes are the contract with CI: 0 when the run produced a report, 1 when it could
not -- an unreachable Qdrant, an invalid golden set, ground truth authored against
different chunking. Every one of those produces plausible-looking numbers if it is
allowed through, so none of them is a warning.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.table import Table

from medguard.cli.bootstrap import bootstrap as _bootstrap
from medguard.cli.bootstrap import console, err_console

if TYPE_CHECKING:
    from medguard.evaluation.runner import MetricsReport, RetrievalRun

__all__ = ["eval_"]

_RETRIEVERS = ("dense", "sparse", "hybrid")


def eval_(
    dataset: Annotated[Path, typer.Option(help="Golden set JSONL.")] = Path(
        "evals/golden_set_v1.jsonl"
    ),
    retriever: Annotated[
        str, typer.Option(help="Comma-separated: dense, sparse, hybrid, or combinations.")
    ] = "dense,sparse,hybrid",
    limit: Annotated[int | None, typer.Option(help="Evaluate only the first N questions.")] = None,
    depth: Annotated[int, typer.Option(min=1, max=200, help="Retrieval depth per query.")] = 50,
    allow_unreviewed: Annotated[
        bool,
        typer.Option(
            "--allow-unreviewed",
            help="Score LLM-drafted questions no human has signed off on. Provisional only.",
        ),
    ] = False,
    write: Annotated[
        bool, typer.Option(help="Write a report directory under evals/reports/.")
    ] = True,
) -> None:
    """Measure retrieval quality against the golden set."""
    _bootstrap()

    from medguard.config import get_settings
    from medguard.errors import EvaluationError, MedGuardError
    from medguard.evaluation.golden_set import (
        load_golden_set,
        load_meta,
        review_status,
        verify_provenance,
    )
    from medguard.evaluation.report import write_report
    from medguard.evaluation.runner import compute_metrics, run_retriever_over_golden_set
    from medguard.retrieval import get_retriever

    settings = get_settings()

    try:
        questions = load_golden_set(dataset, require_review=not allow_unreviewed)
        meta = load_meta(dataset)
        # Before a single query runs. A mismatch here means every chunk id in the set
        # addresses different text, and the run would report confident nonsense.
        verify_provenance(meta, settings.ingestion)
    except EvaluationError as exc:
        err_console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    reviewed, pending = review_status(questions)
    if not reviewed:
        err_console.print(
            f"[yellow]PROVISIONAL:[/] {len(pending)} of {len(questions)} questions are "
            f"LLM-drafted and unreviewed. These numbers are not publishable until a human "
            f"has checked each one against the checklist in evals/README.md and recorded "
            f"their initials in reviewed_by."
        )

    if limit is not None:
        questions = questions[:limit]

    kinds = [k.strip() for k in retriever.split(",") if k.strip()]
    unknown = [k for k in kinds if k not in _RETRIEVERS]
    if unknown:
        err_console.print(
            f"[red]Unknown retriever(s) {unknown}.[/] Supported retrievers: {list(_RETRIEVERS)}."
        )
        raise typer.Exit(1)

    console.print(
        f"Evaluating [bold]{len(questions)}[/] questions "
        f"({sum(q.is_scored for q in questions)} scored) against "
        f"[bold]{settings.collection_name}[/] at depth {depth}."
    )

    reports: list[MetricsReport] = []
    runs_by_retriever: dict[str, list[RetrievalRun]] = {}
    try:
        for kind in kinds:
            engine = get_retriever(kind)
            with console.status(f"Running {kind} retrieval…"):
                runs = run_retriever_over_golden_set(engine, questions, depth=depth)
            runs_by_retriever[kind] = runs
            reports.append(compute_metrics(runs, questions, retriever=kind, depth=depth))
    except MedGuardError as exc:
        err_console.print(f"[red]Evaluation failed:[/] {exc}")
        raise typer.Exit(1) from exc

    _render(reports)

    if write:
        out = write_report(
            reports,
            runs_by_retriever,
            golden_set=questions,
            meta=meta,
            settings=settings,
            reports_dir=None,
        )
        console.print(f"\nReport written to [bold]{out}[/]")


def _render(reports: list[MetricsReport]) -> None:
    """A terminal summary. The report file carries the full detail."""
    first = reports[0]
    table = Table(title=f"Retrieval metrics — {first.n_scored} scored questions")
    table.add_column("Retriever", style="bold")
    for column in ("P@5", "R@5", "R@10", "R@50", "nDCG@10", "MRR", "search p95"):
        table.add_column(column, justify="right")

    def cell(value: float) -> str:
        return "n/a" if math.isnan(value) else f"{value:.3f}"

    for report in reports:
        table.add_row(
            report.retriever,
            cell(report.precision_at_k["5"]),
            cell(report.recall_at_k["5"]),
            cell(report.recall_at_k["10"]),
            cell(report.recall_at_k["50"]),
            cell(report.ndcg_at_k["10"]),
            cell(report.mrr),
            f"{report.latency.search_p95_ms:.1f} ms",
        )
    console.print(table)

    if first.n_unscored:
        console.print(
            f"[dim]{first.n_unscored} question(s) excluded from ranking metrics "
            f"(no relevant chunk by design): {', '.join(first.unscored_question_ids)}[/]"
        )
    for report in reports:
        console.print(f"[dim]{report.retriever}: {report.headroom.verdict}[/]")
