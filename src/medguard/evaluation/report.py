"""Write an evaluation run to disk: markdown to read, JSON to re-score, provenance to trust.

Files, not a database table. The access pattern is "read the run I just produced" and
"diff this run against last month's", which a directory of JSON serves and a table with
a migration does not.

``provenance.json`` is the part that makes the numbers mean anything. A metric without
the commit, the corpus hash, the embedding model and the chunking settings that produced
it is not reproducible, and an irreproducible number is one question away from falling
apart. With it, "nDCG@10 of 0.71 over 14 scored questions, commit abc1234, corpus
manifest 3e1569..." is a claim that survives being checked.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medguard.config import Settings, get_settings
from medguard.evaluation.golden_set import GoldenQuestion, GoldenSetMeta, review_status
from medguard.evaluation.runner import MetricsReport, RetrievalRun
from medguard.logging import get_logger

__all__ = ["REPORTS_DIR", "collect_provenance", "render_markdown", "write_report"]

log = get_logger(__name__)

REPORTS_DIR = Path("evals/reports")
SUMMARY_PATH = REPORTS_DIR / "summary.jsonl"


def _git(*args: str) -> str | None:
    """Run a git command, returning None if git is absent or the call fails."""
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
            ["git", *args],  # noqa: S607 - git resolved from PATH by design
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _sha256_of(path: Path) -> str | None:
    import hashlib

    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_provenance(
    *,
    golden_set: Sequence[GoldenQuestion],
    meta: GoldenSetMeta,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Everything needed to reproduce this run, gathered at the moment it happened."""
    settings = settings or get_settings()
    reviewed, pending = review_status(golden_set)
    dirty = _git("status", "--porcelain")
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # A dirty tree means the commit does not describe the code that ran. Recorded
        # rather than blocked: local experiments are legitimate, quoting one as a
        # published figure is not.
        "git_dirty": bool(dirty) if dirty is not None else None,
        "corpus_manifest_sha256": _sha256_of(settings.corpus.manifest_path),
        "collection": settings.collection_name,
        "embedding_model": settings.embedding.model,
        "embedding_dimension": settings.embedding.dimension,
        "sparse_model": settings.embedding.sparse_model,
        "ingestion": {
            "chunk_size": settings.ingestion.chunk_size,
            "chunk_overlap": settings.ingestion.chunk_overlap,
            "chunk_min_chars": settings.ingestion.chunk_min_chars,
            "chunk_max_chars": settings.ingestion.chunk_max_chars,
            "strip_running_headers": settings.ingestion.strip_running_headers,
        },
        "retrieval": {
            "top_k": settings.retrieval.top_k,
            "fetch_k": settings.retrieval.fetch_k,
            "default_retriever": settings.retrieval.default_retriever,
        },
        "golden_set": {
            # Whether a human has signed off on the ground truth. Recorded here rather
            # than only printed, because the report file outlives the terminal session
            # and is the thing a reader will judge the numbers by.
            "reviewed": reviewed,
            "pending_review": pending,
            "version": meta.version,
            "questions": len(golden_set),
            "scored": sum(1 for q in golden_set if q.is_scored),
            "authored_at": meta.authored_at,
        },
        # Stated explicitly rather than omitted: a reader should not have to infer from
        # absence that no LLM touched this run.
        "gemini_called": False,
        "phase": "4A",
    }


def _fmt(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.3f}"


def render_markdown(reports: Sequence[MetricsReport], provenance: dict[str, Any]) -> str:
    """The human-readable report. Caveats above the table, never in a footnote."""
    lines: list[str] = []
    generated = provenance["generated_at"]
    lines.append("# MedGuard retrieval evaluation")
    lines.append("")
    gs_meta = provenance["golden_set"]
    if not gs_meta.get("reviewed", True):
        pending_ids = gs_meta.get("pending_review") or []
        lines.append(
            f"> **PROVISIONAL — not a publishable figure.** {len(pending_ids)} of "
            f"{gs_meta['questions']} golden-set questions were drafted with LLM "
            f"assistance and no human has reviewed them. Generated questions tend to "
            f"echo their source chunk's wording, which the lexical retriever then "
            f"matches trivially -- a failure these metrics cannot show. Until each "
            f"question is checked against the checklist in `evals/README.md`, the "
            f"numbers below describe an unvalidated instrument."
        )
        lines.append("")
    lines.append(f"- Generated: `{generated}`")
    lines.append(
        f"- Commit: `{provenance.get('git_commit') or 'unknown'}`"
        + ("  **(dirty tree — not a publishable figure)**" if provenance.get("git_dirty") else "")
    )
    lines.append(f"- Collection: `{provenance['collection']}`")
    lines.append(
        f"- Embeddings: `{provenance['embedding_model']}` "
        f"({provenance['embedding_dimension']}d) + `{provenance['sparse_model']}`"
    )
    gs = provenance["golden_set"]
    lines.append(
        f"- Golden set: `{gs['version']}` — {gs['questions']} questions, {gs['scored']} scored"
    )
    lines.append("")

    first = reports[0]
    lines.append("## How to read this")
    lines.append("")
    lines.append(
        f"Ranking metrics are averaged over the **{first.n_scored} questions that have at "
        f"least one relevant chunk**. The remaining {first.n_unscored} "
        f"({', '.join(first.unscored_question_ids) or 'none'}) are unsupported, "
        f"adversarial or ambiguous by design: precision, recall, MRR and nDCG are all "
        f"mathematically undefined with an empty relevant set, so averaging them in "
        f"would be inventing a number. They are reported separately below and scored "
        f"properly in Phase 6, once there is an answer to abstain from."
    )
    lines.append("")
    lines.append(
        "Latency excludes a warm-up call: the first query builds a ~130 MB ONNX session "
        "that no subsequent query pays for. Embed and search are reported separately "
        "because on a CPU-only machine the embed can exceed the search it feeds."
    )
    lines.append("")

    lines.append("## Ranking metrics")
    lines.append("")
    header = (
        "| Retriever | "
        + " | ".join(
            [f"P@{k}" for k in (1, 3, 5, 10)]
            + [f"R@{k}" for k in (1, 3, 5, 10, 20, 50)]
            + [f"nDCG@{k}" for k in (1, 3, 5, 10)]
            + ["MRR"]
        )
        + " |"
    )
    lines.append(header)
    lines.append("|" + "---|" * (header.count("|") - 1))
    for report in reports:
        cells = (
            [_fmt(report.precision_at_k[str(k)]) for k in (1, 3, 5, 10)]
            + [_fmt(report.recall_at_k[str(k)]) for k in (1, 3, 5, 10, 20, 50)]
            + [_fmt(report.ndcg_at_k[str(k)]) for k in (1, 3, 5, 10)]
            + [_fmt(report.mrr)]
        )
        lines.append(f"| `{report.retriever}` | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Latency")
    lines.append("")
    lines.append(
        "| Retriever | embed p50 | embed p95 | search p50 | search p95 | total p50 | total p95 |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for report in reports:
        latency = report.latency
        lines.append(
            f"| `{report.retriever}` | {_fmt(latency.embed_p50_ms)} ms | "
            f"{_fmt(latency.embed_p95_ms)} ms | {_fmt(latency.search_p50_ms)} ms | "
            f"{_fmt(latency.search_p95_ms)} ms | {_fmt(latency.total_p50_ms)} ms | "
            f"{_fmt(latency.total_p95_ms)} ms |"
        )
    lines.append("")

    lines.append("## Reranker headroom (Phase 5 input)")
    lines.append("")
    for report in reports:
        head = report.headroom
        lines.append(f"**`{report.retriever}`** — {head.verdict}")
        lines.append("")
        lines.append(
            f"- recall@10 {_fmt(head.recall_at_10)} → recall@50 {_fmt(head.recall_at_50)} "
            f"(delta {_fmt(head.delta)})"
        )
        rank = head.mean_first_hit_rank_beyond_10
        lines.append(
            f"- {head.questions_with_first_hit_beyond_10} question(s) whose first relevant "
            f"chunk ranks below 10" + (f", mean rank {rank:.1f}" if rank is not None else "")
        )
        lines.append("")

    lines.append("## nDCG@10 by question type")
    lines.append("")
    types = sorted({t for report in reports for t in report.per_type_ndcg_at_10})
    if types:
        lines.append("| Retriever | " + " | ".join(types) + " |")
        lines.append("|" + "---|" * (len(types) + 1))
        for report in reports:
            cells = [_fmt(report.per_type_ndcg_at_10.get(t, math.nan)) for t in types]
            lines.append(f"| `{report.retriever}` | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(
            "Per-type figures on a 20-question worked example rest on two to four "
            "questions each. They indicate direction and nothing more; the full "
            "75-question set is what makes them worth quoting."
        )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "This measures **retrieval only** — which chunks come back for a query. It says "
        "nothing about answer quality, and MedGuard is not a diagnostic system."
    )
    lines.append("")
    return "\n".join(lines)


def write_report(
    reports: Sequence[MetricsReport],
    runs_by_retriever: dict[str, list[RetrievalRun]],
    *,
    golden_set: Sequence[GoldenQuestion],
    meta: GoldenSetMeta,
    settings: Settings | None = None,
    reports_dir: Path | None = None,
) -> Path:
    """Write ``report.md``, ``results.json`` and ``provenance.json``; return the directory."""
    if not reports:
        raise ValueError("no metrics to report")

    base = reports_dir or REPORTS_DIR
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    out = base / stamp
    out.mkdir(parents=True, exist_ok=True)

    provenance = collect_provenance(golden_set=golden_set, meta=meta, settings=settings)

    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "results.json").write_text(
        json.dumps(
            {
                "provenance": provenance,
                "metrics": [json.loads(r.model_dump_json()) for r in reports],
                "runs": {
                    name: [json.loads(run.model_dump_json()) for run in runs]
                    for name, runs in runs_by_retriever.items()
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "report.md").write_text(render_markdown(reports, provenance), encoding="utf-8")

    base.mkdir(parents=True, exist_ok=True)
    summary = base / SUMMARY_PATH.name
    with summary.open("a", encoding="utf-8") as handle:
        for report in reports:
            handle.write(
                json.dumps(
                    {
                        "run": stamp,
                        "retriever": report.retriever,
                        "golden_set": meta.version,
                        "golden_set_reviewed": provenance["golden_set"]["reviewed"],
                        "n_scored": report.n_scored,
                        "ndcg_at_10": report.ndcg_at_k["10"],
                        "recall_at_10": report.recall_at_k["10"],
                        "recall_at_50": report.recall_at_k["50"],
                        "mrr": report.mrr,
                        "search_p95_ms": report.latency.search_p95_ms,
                        "git_commit": provenance.get("git_commit"),
                        "git_dirty": provenance.get("git_dirty"),
                    }
                )
                + "\n"
            )

    log.info("evaluation.report_written", path=str(out), retrievers=len(reports))
    return out
