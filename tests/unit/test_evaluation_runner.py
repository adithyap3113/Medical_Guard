"""The evaluation runner and report writer, offline.

The runner is driven with a scripted retriever rather than a real one: what is under
test is how raw runs become metrics, and a stub makes the ranked lists exact instead of
approximately right, so the expected metric values can be computed on paper.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from medguard.config import Settings
from medguard.errors import EvaluationError
from medguard.evaluation.golden_set import GoldenQuestion, GoldenSetMeta
from medguard.evaluation.report import collect_provenance, render_markdown, write_report
from medguard.evaluation.runner import (
    MetricsReport,
    RetrievalRun,
    compute_metrics,
    percentile,
    run_retriever_over_golden_set,
)
from medguard.models import Candidate, Chunk, RetrieverKind
from medguard.retrieval.base import Retriever

META = GoldenSetMeta.model_validate(
    {
        "version": "v-test",
        "authored_at": "2026-01-01",
        "chunk_id_scheme": "{document_id}::{chunk_index:04d}",
        "chunk_size": 1600,
        "chunk_overlap": 200,
        "chunk_min_chars": 200,
        "chunk_max_chars": 2000,
        "strip_running_headers": True,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dimension": 384,
    }
)


def question(
    qid: str, *, chunks: list[str], qtype: str = "factual", drafted: bool = False
) -> GoldenQuestion:
    return GoldenQuestion.model_validate(
        {
            "question_id": qid,
            "question_text": f"A question numbered {qid}?",
            "question_type": qtype,
            "relevant_chunk_ids": chunks,
            "relevant_document_ids": sorted({c.split("::")[0] for c in chunks}),
            "notes": "Fixture question used by the offline runner tests.",
            "created_by": "llm_assisted" if drafted else "human",
            "reviewed_by": None,
            **({} if chunks else {"expected_behavior": "Decline."}),
        }
    )


class ScriptedRetriever:
    """Returns a fixed ranked list per query. Satisfies the Retriever protocol."""

    def __init__(self, script: dict[str, list[str]], kind: RetrieverKind = RetrieverKind.DENSE):
        self._script = script
        self._kind = kind
        self.last_embed_ms = 1.0
        self.last_search_ms = 2.0
        self.calls: list[str] = []

    @property
    def kind(self) -> RetrieverKind:
        return self._kind

    def retrieve(self, query, *, top_k, retrieval_filter=None):
        self.calls.append(query)
        ids = self._script.get(query, [])[:top_k]
        return [
            Candidate(
                chunk=Chunk(
                    chunk_id=cid, document_id=cid.split("::")[0], text="text", chunk_index=index
                ),
                score=1.0 - index / 100,
                rank=index,
                retriever=self._kind,
                scores={str(self._kind): 1.0 - index / 100},
            )
            for index, cid in enumerate(ids)
        ]


class TestPercentile:
    def test_nearest_rank(self) -> None:
        assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
        assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10

    def test_a_single_sample_is_itself(self) -> None:
        assert percentile([42.0], 95) == 42.0

    def test_an_empty_sample_is_nan_not_zero(self) -> None:
        """Zero would read as "instant"; nan reads as "not measured"."""
        assert math.isnan(percentile([], 95))


class TestRunner:
    def test_it_asks_the_retriever_for_every_question(self) -> None:
        questions = [question("gq_001", chunks=["d::0001"]), question("gq_002", chunks=["d::0002"])]
        retriever = ScriptedRetriever({q.question_text: ["d::0001"] for q in questions})
        runs = run_retriever_over_golden_set(retriever, questions, depth=10)
        assert [r.question_id for r in runs] == ["gq_001", "gq_002"]

    def test_a_warmup_query_precedes_measurement_and_is_not_recorded(self) -> None:
        """The first call builds a ~130 MB ONNX session; letting it into the sample would
        inflate p95 by an order of magnitude and describe a cost no query pays twice."""
        questions = [question("gq_001", chunks=["d::0001"])]
        retriever = ScriptedRetriever({})
        runs = run_retriever_over_golden_set(retriever, questions, depth=10)
        assert len(retriever.calls) == 2  # warm-up + one question
        assert len(runs) == 1

    def test_unanswerable_questions_are_still_run(self) -> None:
        """What a retriever returns for an out-of-corpus question is a real property,
        even though no ranking metric is defined for it."""
        questions = [question("gq_001", chunks=[], qtype="unsupported")]
        runs = run_retriever_over_golden_set(ScriptedRetriever({}), questions, depth=10)
        assert len(runs) == 1

    def test_an_empty_golden_set_is_refused(self) -> None:
        with pytest.raises(EvaluationError, match="nothing to evaluate"):
            run_retriever_over_golden_set(ScriptedRetriever({}), [], depth=10)

    def test_timings_are_carried_through(self) -> None:
        questions = [question("gq_001", chunks=["d::0001"])]
        runs = run_retriever_over_golden_set(ScriptedRetriever({}), questions, depth=10)
        assert runs[0].latency_embed_ms == 1.0
        assert runs[0].latency_total_ms == 3.0


class TestComputeMetrics:
    def _runs(self) -> tuple[list[RetrievalRun], list[GoldenQuestion]]:
        questions = [
            question("gq_001", chunks=["d::0001"]),
            question("gq_002", chunks=["d::0002"]),
            question("gq_003", chunks=[], qtype="unsupported"),
        ]
        runs = [
            # gq_001: relevant chunk at rank 1  -> P@1 1.0, RR 1.0
            RetrievalRun(
                question_id="gq_001",
                retrieved_chunk_ids=["d::0001", "d::9999"],
                latency_embed_ms=1.0,
                latency_search_ms=2.0,
            ),
            # gq_002: relevant chunk at rank 2  -> P@1 0.0, RR 0.5
            RetrievalRun(
                question_id="gq_002",
                retrieved_chunk_ids=["d::9999", "d::0002"],
                latency_embed_ms=3.0,
                latency_search_ms=4.0,
            ),
            # gq_003: no ground truth -> excluded from every ranking metric
            RetrievalRun(
                question_id="gq_003",
                retrieved_chunk_ids=["d::9999"],
                latency_embed_ms=5.0,
                latency_search_ms=6.0,
            ),
        ]
        return runs, questions

    def test_unscored_questions_are_excluded_and_named(self) -> None:
        runs, questions = self._runs()
        report = compute_metrics(runs, questions, retriever="dense")
        assert report.n_questions == 3
        assert report.n_scored == 2
        assert report.n_unscored == 1
        assert report.unscored_question_ids == ["gq_003"]

    def test_aggregates_match_hand_computation(self) -> None:
        runs, questions = self._runs()
        report = compute_metrics(runs, questions, retriever="dense")
        # P@1 over the two scored questions = mean(1.0, 0.0) = 0.5
        assert report.precision_at_k["1"] == pytest.approx(0.5)
        # MRR = mean(1/1, 1/2) = 0.75
        assert report.mrr == pytest.approx(0.75)
        # Both found by rank 2, so recall@3 is 1.0 for each.
        assert report.recall_at_k["3"] == pytest.approx(1.0)

    def test_latency_covers_every_question_including_unscored_ones(self) -> None:
        """Latency is a property of the query, not of whether it had an answer."""
        runs, questions = self._runs()
        report = compute_metrics(runs, questions, retriever="dense")
        assert report.latency.n_measured == 3
        assert report.latency.warmup_excluded

    def test_a_run_for_an_unknown_question_is_refused(self) -> None:
        _, questions = self._runs()
        stray = [
            RetrievalRun(
                question_id="gq_999",
                retrieved_chunk_ids=[],
                latency_embed_ms=0.0,
                latency_search_ms=0.0,
            )
        ]
        with pytest.raises(EvaluationError, match="absent from the golden set"):
            compute_metrics(stray, questions, retriever="dense")

    def test_per_type_breakdown_only_covers_scored_types(self) -> None:
        runs, questions = self._runs()
        report = compute_metrics(runs, questions, retriever="dense")
        assert set(report.per_type_ndcg_at_10) == {"factual"}


class TestRerankerHeadroom:
    def test_no_headroom_when_everything_is_already_in_the_top_ten(self) -> None:
        questions = [question("gq_001", chunks=["d::0001"])]
        runs = [
            RetrievalRun(
                question_id="gq_001",
                retrieved_chunk_ids=["d::0001"],
                latency_embed_ms=1.0,
                latency_search_ms=1.0,
            )
        ]
        report = compute_metrics(runs, questions, retriever="dense")
        assert report.headroom.delta == pytest.approx(0.0)
        assert "no measurable headroom" in report.headroom.verdict

    def test_headroom_when_the_first_hit_sits_deep(self) -> None:
        """recall@50 far above recall@10 is precisely what a reranker can rescue."""
        questions = [question("gq_001", chunks=["d::0042"])]
        retrieved = [f"d::{i:04d}" for i in range(41)] + ["d::0042"]
        runs = [
            RetrievalRun(
                question_id="gq_001",
                retrieved_chunk_ids=retrieved,
                latency_embed_ms=1.0,
                latency_search_ms=1.0,
            )
        ]
        report = compute_metrics(runs, questions, retriever="dense")
        assert report.headroom.recall_at_10 == 0.0
        assert report.headroom.recall_at_50 == 1.0
        assert report.headroom.questions_with_first_hit_beyond_10 == 1
        assert report.headroom.mean_first_hit_rank_beyond_10 == 42
        assert "real headroom" in report.headroom.verdict


class TestReport:
    def _report(self) -> tuple[MetricsReport, list[RetrievalRun], list[GoldenQuestion]]:
        questions = [
            question("gq_001", chunks=["d::0001"]),
            question("gq_002", chunks=[], qtype="unsupported"),
        ]
        runs = [
            RetrievalRun(
                question_id="gq_001",
                retrieved_chunk_ids=["d::0001"],
                per_leg_scores={"dense": [0.9]},
                latency_embed_ms=1.0,
                latency_search_ms=2.0,
            ),
            RetrievalRun(
                question_id="gq_002",
                retrieved_chunk_ids=[],
                per_leg_scores={"dense": []},
                latency_embed_ms=1.0,
                latency_search_ms=2.0,
            ),
        ]
        return compute_metrics(runs, questions, retriever="dense"), runs, questions

    def test_provenance_records_what_reproduces_the_run(self, settings: Settings) -> None:
        metrics, _, questions = self._report()
        assert metrics
        provenance = collect_provenance(golden_set=questions, meta=META, settings=settings)
        for key in ("collection", "embedding_model", "ingestion", "retrieval", "golden_set"):
            assert key in provenance
        assert provenance["ingestion"]["chunk_size"] == settings.ingestion.chunk_size
        # Stated rather than omitted: a reader should not infer from absence.
        assert provenance["gemini_called"] is False

    def test_the_markdown_states_the_scored_count_above_the_table(self, settings: Settings) -> None:
        """The caveat belongs where a reader meets the number, not in a footnote."""
        metrics, _, questions = self._report()
        provenance = collect_provenance(golden_set=questions, meta=META, settings=settings)
        markdown = render_markdown([metrics], provenance)
        assert "1 questions that have at least one relevant chunk" in markdown
        assert "gq_002" in markdown
        assert markdown.index("mathematically undefined") < markdown.index("| Retriever |")

    def test_an_unreviewed_golden_set_is_stamped_provisional(self, settings: Settings) -> None:
        """In the file, not only in the terminal. The report outlives the session, and a
        reader a month later has no way to know the ground truth was never checked."""
        metrics, _, _ = self._report()
        drafted = [question("gq_050", chunks=["d::0001"], drafted=True)]
        provenance = collect_provenance(golden_set=drafted, meta=META, settings=settings)
        assert provenance["golden_set"]["reviewed"] is False
        assert provenance["golden_set"]["pending_review"] == ["gq_050"]
        assert "PROVISIONAL" in render_markdown([metrics], provenance)

    def test_a_reviewed_golden_set_carries_no_banner(self, settings: Settings) -> None:
        metrics, _, questions = self._report()
        # The fixture questions are human-authored, so nothing is pending review.
        provenance = collect_provenance(golden_set=questions, meta=META, settings=settings)
        assert provenance["golden_set"]["reviewed"] is True
        assert "PROVISIONAL" not in render_markdown([metrics], provenance)

    def test_it_never_claims_to_measure_answer_quality(self, settings: Settings) -> None:
        metrics, _, questions = self._report()
        provenance = collect_provenance(golden_set=questions, meta=META, settings=settings)
        markdown = render_markdown([metrics], provenance)
        assert "retrieval only" in markdown
        assert "not a diagnostic system" in markdown

    def test_writing_produces_three_files_and_a_summary_line(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        metrics, runs, questions = self._report()
        out = write_report(
            [metrics],
            {"dense": runs},
            golden_set=questions,
            meta=META,
            settings=settings,
            reports_dir=tmp_path,
        )
        assert (out / "report.md").is_file()
        assert (out / "provenance.json").is_file()

        results = json.loads((out / "results.json").read_text(encoding="utf-8"))
        # Raw per-question records are kept so a metric added later can be recomputed
        # without re-running retrieval.
        assert results["runs"]["dense"][0]["question_id"] == "gq_001"

        summary = (tmp_path / "summary.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert json.loads(summary[-1])["retriever"] == "dense"

    def test_writing_nothing_is_refused(self, settings: Settings, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no metrics"):
            write_report([], {}, golden_set=[], meta=META, settings=settings, reports_dir=tmp_path)


def test_the_scripted_retriever_satisfies_the_protocol() -> None:
    assert isinstance(ScriptedRetriever({}), Retriever)
