"""Retrieval metrics, checked against values worked out by hand.

These are the numbers every retrieval claim in the project rests on, so the expected
values below are computed on paper in the comments rather than by running the code and
recording whatever came out -- which would test only that the implementation is stable,
not that it is right.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from medguard.errors import EvaluationError
from medguard.evaluation.metrics import (
    abstention_accuracy,
    answer_similarity_score,
    citation_correctness,
    dcg,
    faithfulness_score,
    hit_rate_at_k,
    mean_ignoring_undefined,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_position,
)

# Ranked ids, best first, with relevant chunks at 1-based positions 2 and 4.
RANKED = ["a", "b", "c", "d", "e"]
RELEVANT = {"b", "d"}


class TestPrecisionAtK:
    def test_nothing_relevant_in_the_top_one(self) -> None:
        # top 1 = [a]; 0 hits / 1 = 0.0
        assert precision_at_k(RANKED, RELEVANT, 1) == 0.0

    def test_one_hit_in_the_top_two(self) -> None:
        # top 2 = [a, b]; 1 hit / 2 = 0.5
        assert precision_at_k(RANKED, RELEVANT, 2) == 0.5

    def test_both_hits_in_the_top_five(self) -> None:
        # top 5 = all; 2 hits / 5 = 0.4
        assert precision_at_k(RANKED, RELEVANT, 5) == pytest.approx(0.4)

    def test_it_divides_by_k_not_by_the_number_retrieved(self) -> None:
        """A retriever returning three when asked for ten has not earned the precision
        of one that returned ten good ones."""
        # top 10 over a 5-item list: 2 hits / 10 = 0.2, not 2/5.
        assert precision_at_k(RANKED, RELEVANT, 10) == pytest.approx(0.2)


class TestRecallAtK:
    def test_half_the_relevant_set_by_rank_two(self) -> None:
        # 1 of 2 relevant found = 0.5
        assert recall_at_k(RANKED, RELEVANT, 2) == 0.5

    def test_all_of_it_by_rank_four(self) -> None:
        assert recall_at_k(RANKED, RELEVANT, 4) == 1.0

    def test_it_cannot_exceed_one(self) -> None:
        assert recall_at_k(RANKED, RELEVANT, 50) == 1.0


class TestMRR:
    def test_first_relevant_at_rank_two(self) -> None:
        assert mrr(RANKED, RELEVANT) == 0.5

    def test_first_relevant_at_rank_one(self) -> None:
        assert mrr(RANKED, {"a"}) == 1.0

    def test_nothing_relevant_retrieved_is_a_real_zero(self) -> None:
        """Distinct from undefined: the question had an answer and retrieval missed it."""
        assert mrr(RANKED, {"zzz"}) == 0.0

    def test_the_position_is_available_separately(self) -> None:
        assert reciprocal_rank_position(RANKED, RELEVANT) == 2
        assert reciprocal_rank_position(RANKED, {"zzz"}) is None


class TestNDCG:
    def test_a_perfect_ranking_scores_one(self) -> None:
        assert ndcg_at_k(["b", "d", "a"], RELEVANT, 3) == pytest.approx(1.0)

    def test_hand_computed_value(self) -> None:
        # DCG@2 over [a, b] = 0 + 1/log2(3)          = 0.63093
        # IDCG@2           = 1/log2(2) + 1/log2(3)   = 1.63093
        # nDCG@2           = 0.63093 / 1.63093       = 0.38685
        assert ndcg_at_k(RANKED, RELEVANT, 2) == pytest.approx(0.38685, abs=1e-5)

    def test_ideal_normalises_over_min_of_relevant_and_k(self) -> None:
        """The subtle one. Four relevant chunks, k=2: no ranking can retrieve all four
        within two, so the ideal must be the best *achievable* ranking at k, not the
        unreachable one. Normalising over |relevant| would cap this below 1.0 for
        reasons that have nothing to do with retrieval quality."""
        relevant = {"a", "b", "c", "d"}
        assert ndcg_at_k(["a", "b", "z"], relevant, 2) == pytest.approx(1.0)

    def test_a_complete_miss_is_zero(self) -> None:
        assert ndcg_at_k(RANKED, {"zzz"}, 5) == 0.0

    def test_dcg_is_position_discounted(self) -> None:
        # A hit at rank 1 is worth more than the same hit at rank 3.
        assert dcg(["b"], {"b"}, 1) > dcg(["x", "y", "b"], {"b"}, 3)


class TestUndefinedCases:
    """An empty relevant set makes all four metrics undefined, and that must be
    representable. Silently scoring these questions zero would move the headline by up
    to a fifth of its value on a set that is one-fifth unanswerable by design."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda: precision_at_k(RANKED, set(), 5),
            lambda: recall_at_k(RANKED, set(), 5),
            lambda: mrr(RANKED, set()),
            lambda: ndcg_at_k(RANKED, set(), 5),
        ],
    )
    def test_empty_relevant_set_is_nan_not_zero(self, call: Callable[[], float]) -> None:
        assert math.isnan(call())

    def test_the_mean_skips_them_and_says_how_many(self) -> None:
        mean, skipped = mean_ignoring_undefined([1.0, 0.0, math.nan, math.nan])
        assert mean == 0.5
        assert skipped == 2

    def test_all_undefined_reports_nan_rather_than_zero(self) -> None:
        mean, skipped = mean_ignoring_undefined([math.nan, math.nan])
        assert math.isnan(mean)
        assert skipped == 2

    def test_an_empty_sample_is_nan(self) -> None:
        mean, skipped = mean_ignoring_undefined([])
        assert math.isnan(mean)
        assert skipped == 0


class TestInputValidation:
    def test_duplicate_ids_are_refused(self) -> None:
        """Always a retriever or fusion bug, never a corpus property -- and it inflates
        precision. Phase 4B merges two ranked lists, which is where it would appear."""
        with pytest.raises(EvaluationError, match="duplicate"):
            precision_at_k(["a", "a", "b"], {"a"}, 3)

    def test_k_must_be_positive(self) -> None:
        with pytest.raises(EvaluationError, match="k must be"):
            precision_at_k(RANKED, RELEVANT, 0)

    def test_an_empty_ranked_list_scores_zero(self) -> None:
        assert precision_at_k([], RELEVANT, 5) == 0.0
        assert recall_at_k([], RELEVANT, 5) == 0.0


class TestAdditionalMetrics:
    def test_hit_rate_at_k(self) -> None:
        assert hit_rate_at_k(RANKED, RELEVANT, 1) == 0.0
        assert hit_rate_at_k(RANKED, RELEVANT, 2) == 1.0
        assert hit_rate_at_k(RANKED, RELEVANT, 5) == 1.0
        assert math.isnan(hit_rate_at_k(RANKED, set(), 5))

    def test_citation_correctness(self) -> None:
        from unittest.mock import Mock

        c1 = Mock(supported=True)
        c2 = Mock(supported=False)
        assert citation_correctness([]) == 1.0
        assert citation_correctness([c1, c1]) == 1.0
        assert citation_correctness([c1, c2]) == 0.5
        assert citation_correctness([c2, c2]) == 0.0

    def test_faithfulness_score(self) -> None:
        context = "Hypertension is defined as systolic blood pressure of 140 mmHg or greater."
        good_answer = "Hypertension is systolic blood pressure of 140 mmHg."
        bad_answer = "Diabetes mellitus requires insulin therapy."

        assert faithfulness_score(good_answer, context) > 0.8
        assert faithfulness_score(bad_answer, context) < 0.3
        assert faithfulness_score("", context) == 0.0

        assert mrr([], RELEVANT) == 0.0

    def test_abstention_accuracy(self) -> None:
        assert abstention_accuracy(is_unanswerable=True, abstained=True) == 1.0
        assert abstention_accuracy(is_unanswerable=False, abstained=False) == 1.0
        assert abstention_accuracy(is_unanswerable=True, abstained=False) == 0.0
        assert abstention_accuracy(is_unanswerable=False, abstained=True) == 0.0

    def test_answer_similarity_score_handles_missing_reference(self) -> None:
        assert math.isnan(answer_similarity_score("Some answer", None))
        assert math.isnan(answer_similarity_score("Some answer", "   "))
        assert (
            answer_similarity_score("blood pressure guidelines", "blood pressure guidelines") == 1.0
        )
