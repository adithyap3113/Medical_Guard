"""Run a retriever over the golden set and turn the results into metrics.

Two stages kept deliberately separate. :func:`run_retriever_over_golden_set` touches
Qdrant and produces raw per-question records; :func:`compute_metrics` is pure arithmetic
over those records. That split is what makes a run re-scorable: adding a metric later
means recomputing from ``results.json`` rather than re-running retrieval, and it means
the metric code is testable without a vector store.

Retrieval depth is 50 regardless of the ``top_k`` a caller would use in production,
because recall@50 versus recall@10 is the measurement that says whether a Phase 5
reranker has anything to rescue. Truncation to a smaller k happens in the metrics, not
in the retrieval.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from medguard.errors import EvaluationError
from medguard.evaluation.golden_set import GoldenQuestion
from medguard.evaluation.metrics import (
    mean_ignoring_undefined,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_position,
)
from medguard.logging import get_logger
from medguard.retrieval.base import Retriever

__all__ = [
    "EVAL_DEPTH",
    "K_VALUES",
    "LatencySummary",
    "MetricsReport",
    "RerankerHeadroom",
    "RetrievalRun",
    "compute_metrics",
    "percentile",
    "run_retriever_over_golden_set",
]

log = get_logger(__name__)

#: Cut-offs reported for every metric. 1 and 3 are what a user actually reads; 10 is the
#: conventional headline; 5 is the production default.
K_VALUES: tuple[int, ...] = (1, 3, 5, 10)

#: Depth every evaluation run retrieves to. Recall is additionally reported at 20 and 50
#: from the same fetch, which costs nothing extra and is the reranker-headroom signal.
EVAL_DEPTH = 50

_RECALL_K_VALUES: tuple[int, ...] = (*K_VALUES, 20, 50)

#: Sent once before measurement starts. The first call builds a ~130 MB ONNX session,
#: and letting that land in the sample would inflate p95 by an order of magnitude and
#: describe a cost no real query pays twice.
_WARMUP_QUERY = "warm up the embedding session"


class RetrievalRun(BaseModel):
    """What one retriever did on one question. The raw record; no metrics."""

    model_config = ConfigDict(frozen=True)

    question_id: str
    retrieved_chunk_ids: list[str] = Field(description="Rank order, best first.")
    per_leg_scores: dict[str, list[float]] = Field(
        default_factory=dict,
        description="Raw score per leg, parallel to retrieved_chunk_ids. One key in "
        "Phase 4A; two once fusion exists, which is how the sparse leg's contribution "
        "becomes answerable per result rather than only in aggregate.",
    )
    latency_embed_ms: float
    latency_search_ms: float

    @property
    def latency_total_ms(self) -> float:
        return self.latency_embed_ms + self.latency_search_ms


class LatencySummary(BaseModel):
    """Percentiles, never means: one cold call destroys a mean and says nothing."""

    model_config = ConfigDict(frozen=True)

    n_measured: int
    warmup_excluded: bool = True
    embed_p50_ms: float
    embed_p95_ms: float
    search_p50_ms: float
    search_p95_ms: float
    total_p50_ms: float
    total_p95_ms: float


class RerankerHeadroom(BaseModel):
    """How much a Phase 5 reranker could possibly recover.

    If ``recall@50`` barely exceeds ``recall@10``, the relevant chunks a reranker would
    reorder are not being retrieved in the first place, and reranking cannot help. That
    is a negative result worth reporting rather than discovering after building one.
    """

    model_config = ConfigDict(frozen=True)

    recall_at_10: float
    recall_at_50: float
    delta: float
    questions_with_first_hit_beyond_10: int
    mean_first_hit_rank_beyond_10: float | None
    verdict: str


class MetricsReport(BaseModel):
    """Aggregate metrics for one retriever over one golden set."""

    model_config = ConfigDict(frozen=True)

    retriever: str
    depth: int
    n_questions: int
    n_scored: int = Field(description="Questions with at least one relevant chunk.")
    n_unscored: int = Field(description="Excluded from ranking metrics; see below.")
    unscored_question_ids: list[str]
    precision_at_k: dict[str, float]
    recall_at_k: dict[str, float]
    ndcg_at_k: dict[str, float]
    mrr: float
    latency: LatencySummary
    headroom: RerankerHeadroom
    per_type_ndcg_at_10: dict[str, float] = Field(default_factory=dict)


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile. ``nan`` for an empty sample rather than 0.0."""
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = max(0, math.ceil(p / 100 * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def run_retriever_over_golden_set(
    retriever: Retriever,
    golden_set: Sequence[GoldenQuestion],
    *,
    depth: int = EVAL_DEPTH,
) -> list[RetrievalRun]:
    """Retrieve for every question, recording ranked ids and split latency.

    Every question is run, including the ones with no relevant chunk: what a retriever
    returns for an out-of-corpus question is a real property worth recording, even
    though no ranking metric is defined for it.
    """
    if not golden_set:
        raise EvaluationError("golden set is empty; nothing to evaluate")

    try:
        retriever.retrieve(_WARMUP_QUERY, top_k=1)
    except Exception as exc:  # noqa: BLE001 - a warm-up failure is not yet a run failure
        log.warning("evaluation.warmup_failed", error=f"{type(exc).__name__}: {exc}")

    runs: list[RetrievalRun] = []
    for question in golden_set:
        started = time.perf_counter()
        candidates = retriever.retrieve(question.question_text, top_k=depth)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        embed_ms = retriever.last_embed_ms
        search_ms = retriever.last_search_ms
        if embed_ms + search_ms == 0.0:  # pragma: no cover - defensive
            search_ms = elapsed_ms

        leg = str(retriever.kind)
        runs.append(
            RetrievalRun(
                question_id=question.question_id,
                retrieved_chunk_ids=[c.chunk_id for c in candidates],
                per_leg_scores={leg: [c.score for c in candidates]},
                latency_embed_ms=embed_ms,
                latency_search_ms=search_ms,
            )
        )
        log.debug(
            "evaluation.question_done",
            question_id=question.question_id,
            hits=len(candidates),
            total_ms=round(elapsed_ms, 2),
        )

    log.info("evaluation.run_complete", retriever=str(retriever.kind), questions=len(runs))
    return runs


def compute_metrics(
    runs: Sequence[RetrievalRun],
    golden_set: Sequence[GoldenQuestion],
    *,
    retriever: str,
    depth: int = EVAL_DEPTH,
) -> MetricsReport:
    """Score raw runs. Pure arithmetic -- no Qdrant, no embedder, no clock."""
    by_id = {q.question_id: q for q in golden_set}
    missing = [r.question_id for r in runs if r.question_id not in by_id]
    if missing:
        raise EvaluationError(f"runs reference questions absent from the golden set: {missing}")

    scored = [(r, by_id[r.question_id]) for r in runs if by_id[r.question_id].is_scored]
    unscored = [r.question_id for r in runs if not by_id[r.question_id].is_scored]

    def aggregate(fn: object, k: int) -> float:
        values = [
            fn(run.retrieved_chunk_ids, set(question.relevant_chunk_ids), k)  # type: ignore[operator]
            for run, question in scored
        ]
        mean, _ = mean_ignoring_undefined(values)
        return mean

    precision = {str(k): aggregate(precision_at_k, k) for k in K_VALUES}
    recall = {str(k): aggregate(recall_at_k, k) for k in _RECALL_K_VALUES}
    ndcg = {str(k): aggregate(ndcg_at_k, k) for k in K_VALUES}
    mean_mrr, _ = mean_ignoring_undefined(
        [mrr(run.retrieved_chunk_ids, set(q.relevant_chunk_ids)) for run, q in scored]
    )

    latency = LatencySummary(
        n_measured=len(runs),
        embed_p50_ms=percentile([r.latency_embed_ms for r in runs], 50),
        embed_p95_ms=percentile([r.latency_embed_ms for r in runs], 95),
        search_p50_ms=percentile([r.latency_search_ms for r in runs], 50),
        search_p95_ms=percentile([r.latency_search_ms for r in runs], 95),
        total_p50_ms=percentile([r.latency_total_ms for r in runs], 50),
        total_p95_ms=percentile([r.latency_total_ms for r in runs], 95),
    )

    deep_positions = [
        position
        for run, question in scored
        if (
            position := reciprocal_rank_position(
                run.retrieved_chunk_ids, set(question.relevant_chunk_ids)
            )
        )
        is not None
        and position > 10
    ]
    r10, r50 = recall["10"], recall["50"]
    delta = r50 - r10 if not (math.isnan(r10) or math.isnan(r50)) else math.nan
    headroom = RerankerHeadroom(
        recall_at_10=r10,
        recall_at_50=r50,
        delta=delta,
        questions_with_first_hit_beyond_10=len(deep_positions),
        mean_first_hit_rank_beyond_10=(
            statistics.fmean(deep_positions) if deep_positions else None
        ),
        verdict=_headroom_verdict(delta, len(deep_positions)),
    )

    per_type: dict[str, float] = {}
    for question_type in sorted({q.question_type for _, q in scored}):
        subset = [(r, q) for r, q in scored if q.question_type == question_type]
        values = [
            ndcg_at_k(run.retrieved_chunk_ids, set(question.relevant_chunk_ids), 10)
            for run, question in subset
        ]
        mean, _ = mean_ignoring_undefined(values)
        per_type[str(question_type)] = mean

    return MetricsReport(
        retriever=retriever,
        depth=depth,
        n_questions=len(runs),
        n_scored=len(scored),
        n_unscored=len(unscored),
        unscored_question_ids=unscored,
        precision_at_k=precision,
        recall_at_k=recall,
        ndcg_at_k=ndcg,
        mrr=mean_mrr,
        latency=latency,
        headroom=headroom,
        per_type_ndcg_at_10=per_type,
    )


def _headroom_verdict(delta: float, deep_hits: int) -> str:
    if math.isnan(delta):
        return "undefined: no scored questions"
    if delta < 0.02 and deep_hits == 0:
        return (
            "no measurable headroom: relevant chunks found by depth 50 are already "
            "inside the top 10, so a reranker has nothing to reorder"
        )
    if delta < 0.05:
        return (
            f"marginal headroom ({delta:.3f} recall between k=10 and k=50); a reranker "
            f"would be defensible only if it is cheap"
        )
    return (
        f"real headroom ({delta:.3f} recall between k=10 and k=50, {deep_hits} question(s) "
        f"whose first relevant chunk sits below rank 10): reranking has material upside"
    )
