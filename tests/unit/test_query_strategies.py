"""Unit tests for QueryExecutor and QueryStrategy implementations."""

from __future__ import annotations

import pytest

from medguard.errors import ProviderError, RetrievalError
from medguard.models import Candidate, Chunk, QueryStrategy, RetrieverKind
from medguard.providers.base import LLMResponse, ProviderHealth
from medguard.retrieval.base import RetrievalFilter
from medguard.retrieval.strategies import QueryExecutor, _parse_lines


def _candidate(chunk_id: str, score: float = 1.0) -> Candidate:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id=chunk_id.split("::")[0],
        chunk_index=0,
        text=f"Text for {chunk_id}",
        page_start=1,
        page_end=1,
    )
    return Candidate(
        chunk=chunk,
        score=score,
        rank=0,
        retriever=RetrieverKind.HYBRID,
        scores={"rrf": score},
    )


class StubRetriever:
    kind = RetrieverKind.HYBRID
    last_embed_ms = 1.0
    last_search_ms = 2.0

    def __init__(self) -> None:
        self.recorded_queries: list[str] = []

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        retrieval_filter: RetrievalFilter | None = None,
    ) -> list[Candidate]:
        self.recorded_queries.append(query)
        return [_candidate(f"{query}::0001", 0.5)]


class StubProvider:
    name = "stub_llm"

    def __init__(self, response_text: str = "", should_raise: bool = False) -> None:
        self.response_text = response_text
        self.should_raise = should_raise
        self.prompts: list[str] = []

    def generate(
        self,
        *,
        user_prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        self.prompts.append(user_prompt)
        if self.should_raise:
            raise ProviderError("Stub generation failed")
        return LLMResponse(
            text=self.response_text,
            model="stub-model",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            latency_ms=15.0,
        )

    def health(self, *, smoke_test: bool = False) -> ProviderHealth:
        return ProviderHealth(
            provider="stub_llm", configured=True, reachable=True, model_available=True
        )

    def list_models(self) -> list[str]:
        return ["stub-model"]


class TestLineParser:
    def test_parses_bullets_numbers_and_quotes(self) -> None:
        raw = '1. First query\n* Second query\n- Third query\n"Fourth query"\n'
        parsed = _parse_lines(raw, max_lines=4)
        assert parsed == ["First query", "Second query", "Third query", "Fourth query"]


class TestQueryExecutor:
    def test_basic_strategy_without_provider(self) -> None:
        retriever = StubRetriever()
        executor = QueryExecutor(retriever=retriever, provider=None)
        res = executor.execute("diabetes treatment", strategy=QueryStrategy.BASIC, top_k=2)

        assert res.strategy is QueryStrategy.BASIC
        assert res.query == "diabetes treatment"
        assert len(res.candidates) == 1
        assert res.expanded_queries == []

    def test_rewrite_strategy_uses_llm_output(self) -> None:
        retriever = StubRetriever()
        provider = StubProvider(response_text="hypertension clinical management guidelines")
        executor = QueryExecutor(retriever=retriever, provider=provider)

        res = executor.execute("high bp", strategy=QueryStrategy.REWRITE, top_k=2)
        assert res.strategy is QueryStrategy.REWRITE
        assert "hypertension clinical management guidelines" in retriever.recorded_queries
        assert res.expanded_queries == ["hypertension clinical management guidelines"]

    def test_rewrite_fallback_on_provider_error(self) -> None:
        retriever = StubRetriever()
        provider = StubProvider(should_raise=True)
        executor = QueryExecutor(retriever=retriever, provider=provider)

        res = executor.execute("high bp", strategy=QueryStrategy.REWRITE, top_k=2)
        assert res.strategy is QueryStrategy.REWRITE
        assert "high bp" in retriever.recorded_queries

    def test_multi_query_strategy(self) -> None:
        retriever = StubRetriever()
        provider = StubProvider(
            response_text="1. asthma management\n2. asthma treatment protocol\n3. pediatric asthma guidelines"
        )
        executor = QueryExecutor(retriever=retriever, provider=provider)

        res = executor.execute("asthma", strategy=QueryStrategy.MULTI_QUERY, top_k=2)
        assert res.strategy is QueryStrategy.MULTI_QUERY
        assert len(retriever.recorded_queries) >= 3
        assert len(res.expanded_queries) >= 2

    def test_decomposition_strategy(self) -> None:
        retriever = StubRetriever()
        provider = StubProvider(
            response_text="Sub-question 1: hypertension first line treatment\nSub-question 2: blood pressure target"
        )
        executor = QueryExecutor(retriever=retriever, provider=provider)

        res = executor.execute(
            "treat hypertension and bp targets?", strategy=QueryStrategy.DECOMPOSITION, top_k=2
        )
        assert res.strategy is QueryStrategy.DECOMPOSITION
        assert len(res.expanded_queries) == 2

    def test_hyde_strategy(self) -> None:
        retriever = StubRetriever()
        provider = StubProvider(
            response_text="According to the clinical guideline, first-line antihypertensive therapy consists of ACE inhibitors or ARBs."
        )
        executor = QueryExecutor(retriever=retriever, provider=provider)

        res = executor.execute(
            "What is first line therapy for HTN?", strategy=QueryStrategy.HYDE, top_k=2
        )
        assert res.strategy is QueryStrategy.HYDE
        assert len(res.expanded_queries) == 1
        assert "According to the clinical guideline" in res.expanded_queries[0]

    def test_empty_query_raises(self) -> None:
        retriever = StubRetriever()
        executor = QueryExecutor(retriever=retriever)
        with pytest.raises(RetrievalError, match="query is empty"):
            executor.execute("   ")

    def test_invalid_top_k_raises(self) -> None:
        retriever = StubRetriever()
        executor = QueryExecutor(retriever=retriever)
        with pytest.raises(RetrievalError, match="top_k must be >= 1"):
            executor.execute("test", top_k=0)
