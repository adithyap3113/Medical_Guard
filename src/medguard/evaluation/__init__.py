"""Retrieval evaluation: the golden set, the metrics, the runner, the report.

Phase 4A builds the instrument before there is anything interesting to measure with it,
deliberately. A retrieval improvement that cannot be measured is an opinion, and the
order in which those two things get built decides whether the rest of the project
reports numbers or adjectives.

Phase 4A scores retrieval only -- which chunks come back for a query. Answer quality,
faithfulness and citation correctness are Phase 6, and nothing here should be read as
saying anything about them.
"""

from __future__ import annotations

from medguard.evaluation.golden_set import (
    GOLDEN_SET_PATH,
    GoldenQuestion,
    GoldenSetMeta,
    QuestionType,
    load_golden_set,
    load_meta,
    verify_provenance,
)
from medguard.evaluation.runner import (
    MetricsReport,
    RetrievalRun,
    compute_metrics,
    run_retriever_over_golden_set,
)

__all__ = [
    "GOLDEN_SET_PATH",
    "GoldenQuestion",
    "GoldenSetMeta",
    "MetricsReport",
    "QuestionType",
    "RetrievalRun",
    "compute_metrics",
    "load_golden_set",
    "load_meta",
    "run_retriever_over_golden_set",
    "verify_provenance",
]
