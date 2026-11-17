"""Retrieval metrics, hand-implemented.

Why hand-implemented rather than ``ranx``
-----------------------------------------
Four reasons, and the fourth is the real one.

1. ``ranx`` pulls ``numba``, which pins LLVM and NumPy versions, into a tree already
   balancing fastembed/ONNX and qdrant-client on a CPU-only 8 GB machine. That is a
   real resolution risk taken on for arithmetic over lists of at most fifty strings.
2. Relevance here is binary, so nDCG's gain function ``2**rel - 1`` collapses to ``rel``
   and the "hard" part of the metric is not hard.
3. The genuinely subtle part -- ideal DCG normalising over ``min(|relevant|, k)`` rather
   than ``|relevant|`` -- is invisible inside a library and consequential outside one.
   Getting it wrong means nDCG@5 for a question with eight relevant chunks can never
   reach 1.0, and every reported figure is quietly depressed.
4. Fourteen of the planned seventy-five golden questions have *no* relevant chunk by
   design: out-of-corpus, adversarial, unanswerable. All four metrics are mathematically
   undefined there, every library makes a silent choice about it, and that choice moves
   the headline number by up to a fifth. Owning the choice is not optional.

The choice made here: those questions return ``nan`` from every metric, and
:func:`mean_ignoring_undefined` excludes them while reporting how many it excluded. They
are counted and reported separately as an abstention signal, which Phase 6 can score
properly once there is an answer to abstain from. A question that *has* relevant chunks
and retrieves none of them scores ``0.0`` -- a real miss, not an undefined one.

Everything here is a pure function over a ranked list of chunk ids and a set of relevant
chunk ids. No I/O, no Qdrant, no embedder, so the whole module is testable offline
against values worked out by hand.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

from medguard.errors import EvaluationError

__all__ = [
    "abstention_accuracy",
    "answer_similarity_score",
    "citation_correctness",
    "dcg",
    "faithfulness_score",
    "hit_rate_at_k",
    "mean_ignoring_undefined",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank_position",
]


def _validate(retrieved: Sequence[str], k: int | None = None) -> None:
    if k is not None and k < 1:
        raise EvaluationError(f"k must be >= 1, got {k}")
    if len(set(retrieved)) != len(retrieved):
        # A ranked list containing the same chunk twice inflates precision and is always
        # a bug in the retriever or the fusion, never a property of the corpus. Phase 4B
        # merges two ranked lists, which is exactly where this would first appear.
        raise EvaluationError("retrieved list contains duplicate chunk ids")


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top ``k`` that is relevant.

    Divided by ``k``, not by the number actually retrieved. A retriever that returns
    three results when asked for ten has not earned the precision of one that returns
    ten good ones, and the alternative convention hides short result lists.
    """
    _validate(retrieved, k)
    if not relevant:
        return math.nan
    hits = sum(1 for cid in retrieved[:k] if cid in relevant)
    return hits / k


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of all relevant chunks found within the top ``k``."""
    _validate(retrieved, k)
    if not relevant:
        return math.nan
    hits = sum(1 for cid in retrieved[:k] if cid in relevant)
    return hits / len(relevant)


def hit_rate_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary indicator: 1.0 if at least one relevant chunk is retrieved within top ``k``, else 0.0."""
    _validate(retrieved, k)
    if not relevant:
        return math.nan
    return 1.0 if any(cid in relevant for cid in retrieved[:k]) else 0.0


def reciprocal_rank_position(retrieved: Sequence[str], relevant: set[str]) -> int | None:
    """1-based position of the first relevant chunk, or ``None`` if there is none.

    Exposed separately from :func:`mrr` because the position itself is the Phase 5
    signal: a first hit sitting at rank 23 is a question a reranker could rescue, and
    the reciprocal alone (0.043) rounds that story away.
    """
    _validate(retrieved)
    for position, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return position
    return None


def mrr(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Reciprocal of the rank of the first relevant chunk; 0.0 if none was retrieved."""
    if not relevant:
        return math.nan
    position = reciprocal_rank_position(retrieved, relevant)
    return 0.0 if position is None else 1.0 / position


def dcg(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Discounted cumulative gain over binary relevance.

    ``gain = 2**rel - 1`` with ``rel`` in ``{0, 1}`` is just ``rel``, so each hit
    contributes ``1 / log2(position + 1)``.
    """
    return sum(
        1.0 / math.log2(position + 1)
        for position, chunk_id in enumerate(retrieved[:k], start=1)
        if chunk_id in relevant
    )


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Normalised DCG at ``k``.

    The ideal ranking puts relevant chunks in the first ``min(len(relevant), k)``
    positions -- *not* the first ``len(relevant)``. When a question has more relevant
    chunks than ``k``, no ranking can retrieve them all within ``k``, so normalising by
    the unreachable ideal would cap the metric below 1.0 for reasons that have nothing
    to do with retrieval quality.
    """
    _validate(retrieved, k)
    if not relevant:
        return math.nan
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(relevant), k) + 1))
    if ideal == 0.0:  # pragma: no cover - unreachable while relevant is non-empty
        return 0.0
    return dcg(retrieved, relevant, k) / ideal


def mean_ignoring_undefined(values: Iterable[float]) -> tuple[float, int]:
    """Mean over the defined values, plus how many were skipped.

    Returns the count alongside the mean so a report can never quote an average without
    saying how many questions it covers. ``(nan, n)`` when every value was undefined --
    an honest "no scored questions" rather than a zero that reads as a bad score.
    """
    values = list(values)
    defined = [v for v in values if not math.isnan(v)]
    skipped = len(values) - len(defined)
    if not defined:
        return math.nan, skipped
    return sum(defined) / len(defined), skipped


def citation_correctness(citations: Sequence[object]) -> float:
    """Fraction of emitted citations verified as supported against source documents.

    Returns 1.0 if no citations were needed/emitted, or the proportion of citations
    where ``supported is True``.
    """
    if not citations:
        return 1.0
    supported_count = sum(1 for c in citations if getattr(c, "supported", False) is True)
    return supported_count / len(citations)


def faithfulness_score(answer_text: str, context_text: str) -> float:
    """Lexical token overlap heuristic between answer assertions and retrieved grounding context.

    Evaluates groundedness locally without requiring an external LLM-as-a-judge API.
    Computes content-word containment (excluding short stopwords).
    """
    if not answer_text.strip():
        return 0.0
    words = [w.lower() for w in re.findall(r"\b[A-Za-z0-9_\-\.]{3,}\b", answer_text)]
    if not words:
        return 0.0
    context_lower = context_text.lower()
    found = sum(1 for w in words if w in context_lower)
    return found / len(words)


def abstention_accuracy(is_unanswerable: bool, abstained: bool) -> float:
    """Binary score indicating whether the system abstained correctly.

    Returns 1.0 if the system abstained when the question was unanswerable,
    or answered when the question was answerable; otherwise 0.0.
    """
    return 1.0 if is_unanswerable == abstained else 0.0


def answer_similarity_score(answer: str, reference_answer: str | None) -> float:
    """Heuristic token overlap against reference answer if available.

    Returns math.nan when reference_answer is missing or empty, ensuring missing
    ground truth never fabricates a false 0.0 or 1.0 score.
    Note: Semantic entailment / ROUGE / LLM-as-a-judge evaluation is not yet implemented.
    """
    if reference_answer is None or not reference_answer.strip():
        return math.nan
    ans_words = set(re.findall(r"\b\w{3,}\b", answer.lower()))
    ref_words = set(re.findall(r"\b\w{3,}\b", reference_answer.lower()))
    if not ref_words:
        return math.nan
    return len(ans_words & ref_words) / len(ref_words)
