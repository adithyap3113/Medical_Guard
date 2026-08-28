"""The golden set: questions with human-reviewed ground truth, and its provenance.

A retrieval metric is only ever as good as the labels it is scored against, so the
loader here is deliberately strict. Three things it refuses to do:

* load a question drafted by an LLM that no human has signed off on, unless the caller
  explicitly asks for a provisional run and accepts that its numbers are not
  publishable;
* load a question whose type promises relevant chunks and supplies none, or promises
  none and supplies some;
* let a run proceed when the chunk ids were authored against different chunking
  settings than the ones currently configured.

The third is the one that matters most and is the least obvious. Chunk ids are
position-addressed -- ``{document_id}::{chunk_index:04d}`` -- so changing
``INGESTION_CHUNK_SIZE`` and re-ingesting leaves every id in this file pointing at
*different text*. Nothing errors. The file still parses, the ids still resolve, the
metrics still compute, and every number is wrong. That is why the sidecar exists and why
:func:`verify_provenance` raises rather than warns.

``relevant_document_ids`` is carried alongside the chunk ids for the same reason: it
survives re-chunking, so a golden set invalidated by a settings change still holds a
usable, coarser signal while it is being re-labelled.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from medguard.config import IngestionSettings
from medguard.errors import EvaluationError
from medguard.logging import get_logger

__all__ = [
    "GOLDEN_SET_PATH",
    "GoldenQuestion",
    "GoldenSetMeta",
    "QuestionType",
    "load_golden_set",
    "load_meta",
    "meta_path_for",
    "review_status",
    "verify_provenance",
]

log = get_logger(__name__)

#: Default location. Committed to git as JSONL so that a change to a single question is
#: a one-line diff reviewable in a pull request -- the property that makes a golden set
#: auditable, and the reason this is not a SQLite table.
GOLDEN_SET_PATH = Path("evals/golden_set_v1.jsonl")


class QuestionType(StrEnum):
    """What a question is testing, which governs how its result should be read.

    The split is not decorative. The first four types have ground truth in the corpus
    and are scored with ranking metrics; the last three either have no correct chunk or
    have a contested one, and averaging them together with the rest produces a number
    that means nothing in either direction.
    """

    FACTUAL = "factual"
    TERMINOLOGY = "terminology"
    MULTI_DOCUMENT = "multi_document"
    COMPLEX = "complex"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"
    ADVERSARIAL = "adversarial"


#: Types for which an empty ``relevant_chunk_ids`` is the correct answer rather than an
#: unfinished label. Anything else with no ground truth is a question someone forgot to
#: finish, and loading it would silently shrink the scored set.
_MAY_BE_UNANSWERABLE = frozenset(
    {QuestionType.UNSUPPORTED, QuestionType.ADVERSARIAL, QuestionType.AMBIGUOUS}
)


class GoldenQuestion(BaseModel):
    """One evaluation question and its human-reviewed ground truth."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str = Field(pattern=r"^gq_\d{3}$", description="Stable; never reused.")
    question_text: str = Field(min_length=8)
    question_type: QuestionType
    relevant_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Position-addressed chunk ids. Empty is legal only for the "
        "unanswerable types -- see _MAY_BE_UNANSWERABLE.",
    )
    relevant_document_ids: list[str] = Field(
        default_factory=list,
        description="Coarser ground truth that survives re-chunking.",
    )
    notes: str = Field(
        min_length=1,
        description="Why these chunks, and what makes this question its type. Read by "
        "the next person deciding whether a retrieval miss is a bug or a bad label.",
    )
    created_by: str = Field(pattern=r"^(human|llm_assisted)$")
    reviewed_by: str | None = Field(
        default=None,
        description="Reviewer initials and ISO date. Required for llm_assisted.",
    )
    expected_behavior: str | None = Field(
        default=None,
        description="For questions with no relevant chunk: what a correct system should "
        "do instead. Not scored in Phase 4A; the contract Phase 6 will score against.",
    )

    @model_validator(mode="after")
    def _check_ground_truth(self) -> GoldenQuestion:
        if not self.relevant_chunk_ids and self.question_type not in _MAY_BE_UNANSWERABLE:
            raise ValueError(
                f"{self.question_id}: question_type={self.question_type.value!r} promises "
                f"ground truth but relevant_chunk_ids is empty"
            )
        if not self.relevant_chunk_ids and not self.expected_behavior:
            raise ValueError(
                f"{self.question_id}: a question with no relevant chunk must say what "
                f"correct behaviour looks like via expected_behavior"
            )
        if len(set(self.relevant_chunk_ids)) != len(self.relevant_chunk_ids):
            raise ValueError(f"{self.question_id}: duplicate ids in relevant_chunk_ids")

        implied = {cid.split("::")[0] for cid in self.relevant_chunk_ids}
        declared = set(self.relevant_document_ids)
        if implied - declared:
            raise ValueError(
                f"{self.question_id}: relevant_chunk_ids reference document(s) "
                f"{sorted(implied - declared)} not listed in relevant_document_ids"
            )
        return self

    @property
    def is_scored(self) -> bool:
        """Whether ranking metrics are defined for this question."""
        return bool(self.relevant_chunk_ids)

    @property
    def needs_review(self) -> bool:
        """An LLM-drafted question no human has signed off on yet.

        Kept as a property of the data rather than a validation error, so that the
        honest state -- "drafted, not yet reviewed" -- is representable in the file. The
        *policy* about whether such a question may be scored lives in
        :func:`load_golden_set`, where it can be overridden explicitly and loudly rather
        than worked around by writing a reviewer's initials that nobody typed.
        """
        return self.created_by == "llm_assisted" and not self.reviewed_by


class GoldenSetMeta(BaseModel):
    """The chunk-id basis this golden set was authored against.

    Everything recorded here can invalidate the chunk ids in the set without changing
    the file, which is exactly why it is recorded rather than assumed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    authored_at: str
    chunk_id_scheme: str = Field(
        description="Documented so a future reader knows ids are position-addressed."
    )
    chunk_size: int
    chunk_overlap: int
    chunk_min_chars: int
    chunk_max_chars: int
    strip_running_headers: bool
    embedding_model: str
    embedding_dimension: int
    corpus_manifest_sha256: str | None = None
    notes: str = ""


def meta_path_for(path: Path) -> Path:
    """``evals/golden_set_v1.jsonl`` -> ``evals/golden_set_v1.meta.json``."""
    return path.with_suffix("").with_suffix(".meta.json")


def load_golden_set(
    path: Path | None = None,
    *,
    require_review: bool = True,
) -> list[GoldenQuestion]:
    """Parse the JSONL golden set, refusing anything that would corrupt a metric.

    Errors name the line number, because a validation failure on line 47 of a file with
    no line numbers in the message is a genuinely annoying thing to debug.

    ``require_review`` defaults to True: an LLM-drafted question that no human has
    checked must not silently reach a published number, because the characteristic
    failure of generated questions -- echoing the source chunk's wording, so BM25 wins
    trivially and the set measures lexical overlap rather than retrieval -- is invisible
    in the metrics and obvious to a reader. Passing False is the supported way to run a
    provisional baseline over a freshly drafted set; every caller that does so is
    expected to mark the resulting numbers provisional.
    """
    path = path or GOLDEN_SET_PATH
    if not path.is_file():
        raise EvaluationError(
            f"golden set not found at {path}. Phase 4A ships a 20-question worked "
            f"example; see evals/README.md for how the remaining questions are authored"
        )

    questions: list[GoldenQuestion] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"{path}:{lineno}: not valid JSON: {exc}") from exc
        try:
            question = GoldenQuestion.model_validate(payload)
        except ValidationError as exc:
            raise EvaluationError(f"{path}:{lineno}: invalid question: {exc}") from exc
        if question.question_id in seen:
            raise EvaluationError(f"{path}:{lineno}: duplicate question_id {question.question_id}")
        seen.add(question.question_id)
        questions.append(question)

    if not questions:
        raise EvaluationError(f"{path} contains no questions")

    unreviewed = [q.question_id for q in questions if q.needs_review]
    if unreviewed:
        if require_review:
            raise EvaluationError(
                f"{path}: {len(unreviewed)} of {len(questions)} questions were drafted "
                f"with LLM assistance and have no reviewed_by: {', '.join(unreviewed)}.\n\n"
                f"Generated questions fail in a way metrics cannot show: they tend to "
                f"reuse the source chunk's exact wording, so the lexical retriever "
                f"matches them trivially and the set ends up measuring word overlap "
                f"rather than retrieval quality. A human has to read each one against "
                f"the checklist in evals/README.md and record their initials.\n\n"
                f"To run a provisional baseline over an unreviewed draft, pass "
                f"require_review=False (CLI: --allow-unreviewed). Those numbers are not "
                f"publishable."
            )
        log.warning(
            "evaluation.golden_set.unreviewed",
            count=len(unreviewed),
            total=len(questions),
            question_ids=unreviewed,
        )
    return questions


def review_status(questions: Sequence[GoldenQuestion]) -> tuple[bool, list[str]]:
    """``(fully_reviewed, ids_still_needing_review)``, for provenance and report banners."""
    pending = [q.question_id for q in questions if q.needs_review]
    return (not pending), pending


def load_meta(path: Path | None = None) -> GoldenSetMeta:
    """Load the provenance sidecar for a golden set."""
    meta_file = meta_path_for(path or GOLDEN_SET_PATH)
    if not meta_file.is_file():
        raise EvaluationError(
            f"golden set provenance missing at {meta_file}. Without it there is no way "
            f"to tell whether the chunk ids in the set still address the same text"
        )
    try:
        return GoldenSetMeta.model_validate_json(meta_file.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise EvaluationError(f"{meta_file}: invalid provenance: {exc}") from exc


def verify_provenance(meta: GoldenSetMeta, ingestion: IngestionSettings) -> None:
    """Fail loudly if the live chunking settings differ from the authored-against ones.

    This is the tripwire. It is the difference between "our numbers are wrong" and "we
    know our numbers are wrong", and it only earns that if the message is impossible to
    misread -- hence the essay rather than an assertion.
    """
    fields = ("chunk_size", "chunk_overlap", "chunk_min_chars", "chunk_max_chars")
    drift = [(f, getattr(meta, f), getattr(ingestion, f)) for f in fields]
    drift = [(f, a, b) for f, a, b in drift if a != b]
    if meta.strip_running_headers != ingestion.strip_running_headers:
        drift.append(
            ("strip_running_headers", meta.strip_running_headers, ingestion.strip_running_headers)
        )
    if not drift:
        return

    authored = " ".join(f"{f.upper()}={a}" for f, a, _ in drift)
    current = " ".join(f"{f.upper()}={b}" for f, _, b in drift)
    raise EvaluationError(
        f"GOLDEN SET INVALIDATED BY A CHUNKING CHANGE.\n\n"
        f"Golden set {meta.version} was authored against chunks produced with "
        f"{authored}.\nCurrent ingestion settings are {current}.\n\n"
        f"Chunk ids are position-addressed ({meta.chunk_id_scheme}), so every "
        f"relevant_chunk_id in the set now points at different text. The ids still "
        f"resolve and the metrics will still compute -- they will simply be wrong, with "
        f"nothing anywhere reporting it.\n\n"
        f"Either revert the ingestion settings and re-index, or re-label the ground "
        f"truth against the new chunking and publish it as a new golden set version. "
        f"Do not edit {meta.version} in place: numbers have been published against it."
    )
