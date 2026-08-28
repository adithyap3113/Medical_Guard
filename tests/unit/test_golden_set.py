"""Golden-set loading, and the guards that keep a metric honest.

The committed set is loaded here too: it is the artefact every retrieval number is
derived from, so a change that breaks its schema should fail in the offline suite rather
than an hour later in an evaluation run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from medguard.config import IngestionSettings
from medguard.errors import EvaluationError
from medguard.evaluation.golden_set import (
    GOLDEN_SET_PATH,
    GoldenQuestion,
    GoldenSetMeta,
    QuestionType,
    load_golden_set,
    load_meta,
    meta_path_for,
    review_status,
    verify_provenance,
)

VALID = {
    "question_id": "gq_001",
    "question_text": "What is the recommended screening interval?",
    "question_type": "factual",
    "relevant_chunk_ids": ["doc_a::0001"],
    "relevant_document_ids": ["doc_a"],
    "notes": "Chunk states the interval directly.",
    "created_by": "llm_assisted",
    "reviewed_by": "XY 2026-01-01",
}


def write_set(tmp_path: Path, *questions: dict[str, Any]) -> Path:
    path = tmp_path / "golden_set_test.jsonl"
    path.write_text("\n".join(json.dumps(q) for q in questions) + "\n", encoding="utf-8")
    return path


class TestSchema:
    def test_a_valid_question_loads(self) -> None:
        question = GoldenQuestion.model_validate(VALID)
        assert question.question_type is QuestionType.FACTUAL
        assert question.is_scored

    def test_question_ids_follow_one_format(self) -> None:
        with pytest.raises(ValidationError):
            GoldenQuestion.model_validate({**VALID, "question_id": "q1"})

    def test_unknown_fields_are_refused(self) -> None:
        """extra='forbid' so a typo'd field name is an error, not a silently ignored
        label that the author believes is being used."""
        with pytest.raises(ValidationError):
            GoldenQuestion.model_validate({**VALID, "relevant_chunks": ["x"]})

    def test_a_scored_type_must_supply_ground_truth(self) -> None:
        with pytest.raises(ValidationError, match="promises ground truth"):
            GoldenQuestion.model_validate({**VALID, "relevant_chunk_ids": []})

    def test_an_unanswerable_type_may_have_none(self) -> None:
        question = GoldenQuestion.model_validate(
            {
                **VALID,
                "question_type": "unsupported",
                "relevant_chunk_ids": [],
                "relevant_document_ids": [],
                "expected_behavior": "Say the corpus does not cover this.",
            }
        )
        assert not question.is_scored

    def test_an_unanswerable_question_must_say_what_correct_looks_like(self) -> None:
        with pytest.raises(ValidationError, match="expected_behavior"):
            GoldenQuestion.model_validate(
                {
                    **VALID,
                    "question_type": "unsupported",
                    "relevant_chunk_ids": [],
                    "relevant_document_ids": [],
                }
            )

    def test_duplicate_ground_truth_ids_are_refused(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            GoldenQuestion.model_validate(
                {**VALID, "relevant_chunk_ids": ["doc_a::0001", "doc_a::0001"]}
            )

    def test_chunk_ids_must_agree_with_the_declared_documents(self) -> None:
        """The coarse label is the fallback when re-chunking invalidates the fine one,
        so the two disagreeing makes the fallback wrong as well."""
        with pytest.raises(ValidationError, match="not listed in relevant_document_ids"):
            GoldenQuestion.model_validate(
                {**VALID, "relevant_chunk_ids": ["doc_b::0001"], "relevant_document_ids": ["doc_a"]}
            )


class TestReviewPolicy:
    def test_an_unreviewed_generated_question_blocks_a_run(self, tmp_path: Path) -> None:
        path = write_set(tmp_path, {**VALID, "reviewed_by": None})
        with pytest.raises(EvaluationError, match="no reviewed_by"):
            load_golden_set(path)

    def test_it_can_be_overridden_explicitly_for_a_provisional_run(self, tmp_path: Path) -> None:
        path = write_set(tmp_path, {**VALID, "reviewed_by": None})
        questions = load_golden_set(path, require_review=False)
        assert len(questions) == 1
        assert questions[0].needs_review

    def test_review_status_names_what_is_outstanding(self) -> None:
        drafted = GoldenQuestion.model_validate({**VALID, "reviewed_by": None})
        signed = GoldenQuestion.model_validate(VALID)
        assert review_status([signed]) == (True, [])
        assert review_status([drafted, signed]) == (False, ["gq_001"])

    def test_a_human_authored_question_needs_no_reviewer(self) -> None:
        question = GoldenQuestion.model_validate(
            {**VALID, "created_by": "human", "reviewed_by": None}
        )
        assert not question.needs_review


class TestLoading:
    def test_line_numbers_appear_in_parse_errors(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.jsonl"
        path.write_text(json.dumps(VALID) + "\n{not json}\n", encoding="utf-8")
        with pytest.raises(EvaluationError, match=r":2:"):
            load_golden_set(path)

    def test_duplicate_question_ids_are_refused(self, tmp_path: Path) -> None:
        path = write_set(tmp_path, VALID, VALID)
        with pytest.raises(EvaluationError, match="duplicate question_id"):
            load_golden_set(path)

    def test_blank_lines_and_comments_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "commented.jsonl"
        path.write_text(f"// a note\n\n{json.dumps(VALID)}\n", encoding="utf-8")
        assert len(load_golden_set(path)) == 1

    def test_a_missing_file_explains_itself(self, tmp_path: Path) -> None:
        with pytest.raises(EvaluationError, match="golden set not found"):
            load_golden_set(tmp_path / "absent.jsonl")

    def test_an_empty_file_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("\n", encoding="utf-8")
        with pytest.raises(EvaluationError, match="no questions"):
            load_golden_set(path)


META = {
    "version": "v1",
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


class TestProvenanceTripwire:
    """The difference between "our numbers are wrong" and "we know they are wrong"."""

    def test_matching_settings_pass_silently(self) -> None:
        verify_provenance(GoldenSetMeta.model_validate(META), IngestionSettings())

    def test_a_changed_chunk_size_invalidates_the_set(self) -> None:
        meta = GoldenSetMeta.model_validate({**META, "chunk_size": 900})
        with pytest.raises(EvaluationError, match="GOLDEN SET INVALIDATED") as exc:
            verify_provenance(meta, IngestionSettings())
        message = str(exc.value)
        # The message has to be unmissable: both values, and why it matters.
        assert "CHUNK_SIZE=900" in message
        assert "CHUNK_SIZE=1600" in message
        assert "position-addressed" in message

    def test_a_changed_header_policy_also_invalidates_it(self) -> None:
        """Header stripping changes page text, which moves every chunk boundary after
        it -- the same silent failure by a different route."""
        meta = GoldenSetMeta.model_validate({**META, "strip_running_headers": False})
        with pytest.raises(EvaluationError, match="GOLDEN SET INVALIDATED"):
            verify_provenance(meta, IngestionSettings())

    def test_the_sidecar_path_is_derived_from_the_set(self) -> None:
        assert meta_path_for(Path("evals/golden_set_v1.jsonl")).name == "golden_set_v1.meta.json"

    def test_a_missing_sidecar_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(EvaluationError, match="provenance missing"):
            load_meta(tmp_path / "nope.jsonl")


class TestTheCommittedSet:
    """The shipped artefact, validated offline."""

    def test_it_loads(self) -> None:
        questions = load_golden_set(GOLDEN_SET_PATH, require_review=False)
        assert len(questions) >= 20

    def test_every_question_id_is_unique(self) -> None:
        questions = load_golden_set(GOLDEN_SET_PATH, require_review=False)
        assert len({q.question_id for q in questions}) == len(questions)

    def test_all_seven_types_are_represented(self) -> None:
        questions = load_golden_set(GOLDEN_SET_PATH, require_review=False)
        assert {q.question_type for q in questions} == set(QuestionType)

    def test_unanswerable_questions_state_expected_behaviour(self) -> None:
        for question in load_golden_set(GOLDEN_SET_PATH, require_review=False):
            if not question.is_scored:
                assert question.expected_behavior, question.question_id

    def test_every_question_carries_authoring_notes(self) -> None:
        """Notes are what let the next reader tell a retrieval miss from a bad label."""
        for question in load_golden_set(GOLDEN_SET_PATH, require_review=False):
            assert len(question.notes) > 40, question.question_id

    def test_its_provenance_matches_the_configured_chunking(self) -> None:
        verify_provenance(load_meta(GOLDEN_SET_PATH), IngestionSettings())

    def test_more_than_one_publisher_is_covered(self) -> None:
        questions = load_golden_set(GOLDEN_SET_PATH, require_review=False)
        documents = {d for q in questions for d in q.relevant_document_ids}
        prefixes = {d.split("_")[0] for d in documents}
        assert len(prefixes) >= 3, prefixes
