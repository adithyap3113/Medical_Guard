"""Query processing strategies for medical knowledge retrieval.

Implements the five strategies defined in :class:`~medguard.models.QueryStrategy`:
- ``BASIC``: Direct single retrieval using the original query.
- ``REWRITE``: Rewrites layperson queries into precise clinical search phrasing.
- ``MULTI_QUERY``: Generates multi-perspective queries to maximize guideline recall.
- ``DECOMPOSITION``: Breaks complex or multi-part clinical questions into sub-queries.
- ``HYDE``: Generates a hypothetical clinical guidance passage for retrieval.

All strategies that rely on LLM generation use the configured LLMProvider (Gemini)
and fall back gracefully to ``BASIC`` retrieval if the provider is unconfigured,
unreachable, rate-limited, or returns an empty response.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from medguard.config import RetrievalSettings, get_settings
from medguard.errors import ProviderError, RetrievalError
from medguard.logging import get_logger
from medguard.models import Candidate, QueryStrategy, RetrievalResult
from medguard.retrieval.base import RetrievalFilter, Retriever
from medguard.retrieval.hybrid import reciprocal_rank_fusion

if TYPE_CHECKING:
    from medguard.providers.base import LLMProvider

__all__ = ["QueryExecutor", "execute_strategy"]

log = get_logger(__name__)

_REWRITE_SYSTEM_PROMPT = """You are a clinical query optimization assistant.
Your task is to rewrite user questions into search queries optimized for finding relevant evidence in published medical clinical guidelines (e.g., CDC, WHO, ICMR, NIH).
Rules:
1. Translate colloquial phrases into standard medical terminology (e.g., "high blood pressure" -> "hypertension", "sugar levels" -> "glycated hemoglobin HbA1c").
2. Preserve all specific medications, dosages, units, patient age groups, and clinical contraindications.
3. Do NOT answer the question. Output ONLY the single rewritten search query string.
"""

_MULTI_QUERY_SYSTEM_PROMPT = """You are a clinical information retrieval specialist.
Your task is to generate 3 alternative formulations of a medical query to retrieve clinical guidelines from different relevant angles:
1. One using formal diagnostic/clinical terminology.
2. One focusing on standard therapeutic/pharmacological phrasing.
3. One focusing on guideline recommendation/protocol phrasing.

Rules:
- Output exactly 3 queries, one per line.
- Do NOT number them or use bullets.
- Preserve all medical constraints and drug names.
"""

_DECOMPOSITION_SYSTEM_PROMPT = """You are a medical question decomposition assistant.
Your task is to break down a complex or multi-part medical question into 2 or 3 distinct sub-questions that can each be searched independently in clinical guidelines.
Rules:
- Output each sub-question on its own line.
- Do NOT number them or use bullets.
- If the question is simple and cannot be decomposed, return the original question on a single line.
"""

_HYDE_SYSTEM_PROMPT = """You are a medical guideline authoring assistant.
Given a clinical question, write a short, hypothetical excerpt (2-3 sentences) from an official clinical practice guideline that would directly answer this question.
Use authoritative clinical guideline tone, standard medical units, and formal terminology.
Do NOT include preambles or explanations; output only the hypothetical passage.
"""


def _parse_lines(text: str, max_lines: int = 5) -> list[str]:
    """Extract cleaned non-empty lines from LLM response, stripping bullets and numbers."""
    cleaned: list[str] = []
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        # Strip common markdown bullets and numbered prefixes like "1. ", "- ", "* "
        line = re.sub(r"^(?:\d+[\.\)]|\*|-|\+)\s*", "", line).strip()
        # Strip quotes if wrapped
        if line.startswith(('"', "'")) and line.endswith(('"', "'")) and len(line) > 1:
            line = line[1:-1].strip()
        if line and line not in cleaned:
            cleaned.append(line)
        if len(cleaned) >= max_lines:
            break
    return cleaned


class QueryExecutor:
    """Executes retrieval under a chosen QueryStrategy."""

    def __init__(
        self,
        retriever: Retriever,
        provider: LLMProvider | None = None,
        settings: RetrievalSettings | None = None,
    ) -> None:
        self._retriever = retriever
        self._provider = provider
        self._settings = settings if settings is not None else get_settings().retrieval

    def execute(
        self,
        query: str,
        *,
        strategy: QueryStrategy = QueryStrategy.BASIC,
        top_k: int = 5,
        retrieval_filter: RetrievalFilter | None = None,
    ) -> RetrievalResult:
        """Run retrieval under the requested strategy."""
        query_str = query.strip()
        if not query_str:
            raise RetrievalError("query is empty; there is nothing to retrieve")
        if top_k < 1:
            raise RetrievalError(f"top_k must be >= 1, got {top_k}")

        started = time.perf_counter()

        if strategy == QueryStrategy.BASIC or self._provider is None:
            return self._execute_basic(
                query_str, top_k=top_k, retrieval_filter=retrieval_filter, started=started
            )

        if strategy == QueryStrategy.REWRITE:
            return self._execute_rewrite(
                query_str, top_k=top_k, retrieval_filter=retrieval_filter, started=started
            )

        if strategy == QueryStrategy.MULTI_QUERY:
            return self._execute_multi_query(
                query_str, top_k=top_k, retrieval_filter=retrieval_filter, started=started
            )

        if strategy == QueryStrategy.DECOMPOSITION:
            return self._execute_decomposition(
                query_str, top_k=top_k, retrieval_filter=retrieval_filter, started=started
            )

        return self._execute_hyde(
            query_str, top_k=top_k, retrieval_filter=retrieval_filter, started=started
        )

    # -- Strategy Implementations --------------------------------------------

    def _execute_basic(
        self,
        query: str,
        *,
        top_k: int,
        retrieval_filter: RetrievalFilter | None,
        started: float,
    ) -> RetrievalResult:
        candidates = self._retriever.retrieve(query, top_k=top_k, retrieval_filter=retrieval_filter)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return RetrievalResult(
            query=query,
            candidates=candidates,
            strategy=QueryStrategy.BASIC,
            retriever=self._retriever.kind,
            expanded_queries=[],
            latency_ms=elapsed_ms,
            total_candidates_considered=len(candidates),
        )

    def _execute_rewrite(
        self,
        query: str,
        *,
        top_k: int,
        retrieval_filter: RetrievalFilter | None,
        started: float,
    ) -> RetrievalResult:
        rewritten: str | None = None
        if self._provider is not None:
            try:
                resp = self._provider.generate(
                    user_prompt=f"Question: {query}",
                    system_prompt=_REWRITE_SYSTEM_PROMPT,
                    temperature=0.1,
                    max_output_tokens=128,
                )
                parsed = resp.text.strip().replace("\n", " ")
                if parsed:
                    rewritten = parsed
            except (ProviderError, Exception) as exc:  # noqa: BLE001
                log.warning("strategy.rewrite.failed_fallback_to_basic", error=str(exc))

        search_query = rewritten or query
        candidates = self._retriever.retrieve(
            search_query, top_k=top_k, retrieval_filter=retrieval_filter
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return RetrievalResult(
            query=query,
            candidates=candidates,
            strategy=QueryStrategy.REWRITE,
            retriever=self._retriever.kind,
            expanded_queries=[search_query] if rewritten else [],
            latency_ms=elapsed_ms,
            total_candidates_considered=len(candidates),
        )

    def _execute_multi_query(
        self,
        query: str,
        *,
        top_k: int,
        retrieval_filter: RetrievalFilter | None,
        started: float,
    ) -> RetrievalResult:
        queries = [query]
        if self._provider is not None:
            try:
                resp = self._provider.generate(
                    user_prompt=f"Medical Question: {query}",
                    system_prompt=_MULTI_QUERY_SYSTEM_PROMPT,
                    temperature=0.3,
                    max_output_tokens=256,
                )
                expanded = _parse_lines(resp.text, max_lines=3)
                for q in expanded:
                    if q and q not in queries:
                        queries.append(q)
            except (ProviderError, Exception) as exc:  # noqa: BLE001
                log.warning("strategy.multi_query.failed_fallback_to_basic", error=str(exc))

        # Retrieve for all queries and fuse via RRF
        depth = max(top_k, self._settings.fetch_k)
        candidate_lists: list[list[Candidate]] = []
        for q in queries:
            try:
                candidate_lists.append(
                    self._retriever.retrieve(q, top_k=depth, retrieval_filter=retrieval_filter)
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("strategy.multi_query.leg_failed", query=q, error=str(exc))

        if not candidate_lists:
            candidates = self._retriever.retrieve(
                query, top_k=top_k, retrieval_filter=retrieval_filter
            )
        elif len(candidate_lists) == 1:
            candidates = candidate_lists[0][:top_k]
        else:
            # Multi-list RRF fusion
            candidates = self._fuse_multiple_lists(candidate_lists, top_k=top_k)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return RetrievalResult(
            query=query,
            candidates=candidates,
            strategy=QueryStrategy.MULTI_QUERY,
            retriever=self._retriever.kind,
            expanded_queries=queries[1:],
            latency_ms=elapsed_ms,
            total_candidates_considered=sum(len(c) for c in candidate_lists) or len(candidates),
        )

    def _execute_decomposition(
        self,
        query: str,
        *,
        top_k: int,
        retrieval_filter: RetrievalFilter | None,
        started: float,
    ) -> RetrievalResult:
        sub_questions: list[str] = []
        if self._provider is not None:
            try:
                resp = self._provider.generate(
                    user_prompt=f"Complex medical question: {query}",
                    system_prompt=_DECOMPOSITION_SYSTEM_PROMPT,
                    temperature=0.1,
                    max_output_tokens=256,
                )
                sub_questions = _parse_lines(resp.text, max_lines=4)
            except (ProviderError, Exception) as exc:  # noqa: BLE001
                log.warning("strategy.decomposition.failed_fallback_to_basic", error=str(exc))

        if not sub_questions:
            sub_questions = [query]

        # Retrieve for each sub-question
        depth = max(top_k, self._settings.fetch_k)
        candidate_lists: list[list[Candidate]] = []
        for sq in sub_questions:
            try:
                candidate_lists.append(
                    self._retriever.retrieve(sq, top_k=depth, retrieval_filter=retrieval_filter)
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("strategy.decomposition.sub_query_failed", query=sq, error=str(exc))

        if not candidate_lists:
            candidates = self._retriever.retrieve(
                query, top_k=top_k, retrieval_filter=retrieval_filter
            )
        elif len(candidate_lists) == 1:
            candidates = candidate_lists[0][:top_k]
        else:
            candidates = self._fuse_multiple_lists(candidate_lists, top_k=top_k)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return RetrievalResult(
            query=query,
            candidates=candidates,
            strategy=QueryStrategy.DECOMPOSITION,
            retriever=self._retriever.kind,
            expanded_queries=sub_questions,
            latency_ms=elapsed_ms,
            total_candidates_considered=sum(len(c) for c in candidate_lists) or len(candidates),
        )

    def _execute_hyde(
        self,
        query: str,
        *,
        top_k: int,
        retrieval_filter: RetrievalFilter | None,
        started: float,
    ) -> RetrievalResult:
        hypothetical_doc: str | None = None
        if self._provider is not None:
            try:
                resp = self._provider.generate(
                    user_prompt=f"Clinical Question: {query}",
                    system_prompt=_HYDE_SYSTEM_PROMPT,
                    temperature=0.2,
                    max_output_tokens=300,
                )
                parsed = resp.text.strip()
                if parsed:
                    hypothetical_doc = parsed
            except (ProviderError, Exception) as exc:  # noqa: BLE001
                log.warning("strategy.hyde.failed_fallback_to_basic", error=str(exc))

        depth = max(top_k, self._settings.fetch_k)
        orig_hits = self._retriever.retrieve(query, top_k=depth, retrieval_filter=retrieval_filter)

        if hypothetical_doc:
            try:
                hyde_hits = self._retriever.retrieve(
                    hypothetical_doc, top_k=depth, retrieval_filter=retrieval_filter
                )
                candidates = reciprocal_rank_fusion(
                    orig_hits,
                    hyde_hits,
                    rrf_k=getattr(self._settings, "rrf_k", 60),
                    top_k=top_k,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("strategy.hyde.retrieval_failed", error=str(exc))
                candidates = orig_hits[:top_k]
        else:
            candidates = orig_hits[:top_k]

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return RetrievalResult(
            query=query,
            candidates=candidates,
            strategy=QueryStrategy.HYDE,
            retriever=self._retriever.kind,
            expanded_queries=[hypothetical_doc] if hypothetical_doc else [],
            latency_ms=elapsed_ms,
            total_candidates_considered=len(orig_hits),
        )

    def _fuse_multiple_lists(
        self, candidate_lists: list[list[Candidate]], *, top_k: int
    ) -> list[Candidate]:
        """Multi-list RRF fusion across arbitrary number of candidate rankings."""
        rrf_k = getattr(self._settings, "rrf_k", 60)
        rrf_scores: dict[str, float] = {}
        chunks_map: dict[str, Candidate] = {}
        accumulated_scores: dict[str, dict[str, float]] = {}

        for list_idx, cands in enumerate(candidate_lists):
            for rank, cand in enumerate(cands):
                cid = cand.chunk_id
                if cid not in chunks_map:
                    chunks_map[cid] = cand
                    accumulated_scores[cid] = {}

                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))
                accumulated_scores[cid][f"list_{list_idx}"] = cand.score
                if "dense" in cand.scores:
                    accumulated_scores[cid]["dense"] = cand.scores["dense"]
                if "sparse" in cand.scores:
                    accumulated_scores[cid]["sparse"] = cand.scores["sparse"]

        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: (-rrf_scores[cid], cid))
        results: list[Candidate] = []
        for rank, cid in enumerate(sorted_ids[:top_k]):
            orig = chunks_map[cid]
            scores = dict(accumulated_scores[cid])
            scores["rrf"] = rrf_scores[cid]
            results.append(
                Candidate(
                    chunk=orig.chunk,
                    score=rrf_scores[cid],
                    rank=rank,
                    retriever=orig.retriever,
                    scores=scores,
                )
            )
        return results


def execute_strategy(
    query: str,
    *,
    strategy: QueryStrategy = QueryStrategy.BASIC,
    top_k: int = 5,
    retriever_kind: str = "hybrid",
    retrieval_filter: RetrievalFilter | None = None,
    provider: LLMProvider | None = None,
) -> RetrievalResult:
    """Convenience helper to retrieve with a strategy from default settings."""
    from medguard.retrieval import get_retriever

    retriever = get_retriever(retriever_kind)
    executor = QueryExecutor(retriever=retriever, provider=provider)
    return executor.execute(
        query,
        strategy=strategy,
        top_k=top_k,
        retrieval_filter=retrieval_filter,
    )
