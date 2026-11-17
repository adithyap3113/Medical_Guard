"""Query endpoint for medical knowledge retrieval and grounded answer generation."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request

from medguard.api.schemas import (
    CitationResponse,
    ErrorResponse,
    QueryRequest,
    QueryResponse,
    RetrievalMetadataResponse,
)
from medguard.config import Settings, get_settings
from medguard.errors import RetrievalError
from medguard.generation.generator import AnswerGenerator
from medguard.logging import get_logger
from medguard.models import MEDICAL_DISCLAIMER, Answer
from medguard.providers.gemini import GeminiProvider
from medguard.retrieval import RetrievalFilter, get_retriever
from medguard.retrieval.strategies import QueryExecutor

__all__ = ["router"]

log = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query medical guidelines with grounded evidence",
    responses={
        200: {"description": "Grounded answer with citations and retrieval metadata."},
        400: {"model": ErrorResponse, "description": "Invalid query parameters."},
        500: {"model": ErrorResponse, "description": "Internal retrieval or processing error."},
        503: {
            "model": ErrorResponse,
            "description": "LLM generation provider error or unavailable.",
        },
    },
)
def query_guidelines(body: QueryRequest, request: Request) -> QueryResponse:
    """Execute end-to-end RAG pipeline over the indexed medical guideline corpus.

    1. Formulates and executes search across dense and sparse vectors via RRF.
    2. Applies requested QueryStrategy (e.g., query rewriting, decomposition, HyDE).
    3. Prompts Gemini with strict clinical grounding and prompt-injection safeguards.
    4. Validates all generated citations against retrieved document chunks and page spans.
    5. Returns grounded answer, verified citations, and retrieval diagnostics.
    """
    settings: Settings = getattr(request.app.state, "settings", None) or get_settings()
    trace_id = str(uuid.uuid4())

    log.info(
        "api.query.received",
        question=body.question[:100],
        strategy=body.strategy.value,
        top_k=body.top_k,
        retriever=body.retriever,
        trace_id=trace_id,
    )

    # 1. Build optional metadata filter
    retrieval_filter: RetrievalFilter | None = None
    if body.source_organization or body.document_type:
        retrieval_filter = RetrievalFilter(
            source_organization=body.source_organization,
            document_type=body.document_type,
        )

    # 2. Get retriever
    try:
        custom_retriever = getattr(request.app.state, "retriever", None)
        if custom_retriever is not None:
            retriever = custom_retriever
        else:
            custom_store = getattr(request.app.state, "vector_store", None)
            retriever = get_retriever(body.retriever, store=custom_store, settings=settings)
    except ValueError as exc:
        raise RetrievalError(str(exc), user_message=f"Invalid retriever: {body.retriever}") from exc

    # 3. Setup Gemini provider if configured
    provider: GeminiProvider | None = None
    if settings.gemini.configured:
        provider = GeminiProvider(settings.gemini)

    # 4. Execute retrieval strategy
    executor = QueryExecutor(retriever, provider=provider, settings=settings.retrieval)
    retrieval_result = executor.execute(
        body.question,
        strategy=body.strategy,
        top_k=body.top_k,
        retrieval_filter=retrieval_filter,
    )

    # 5. Generate grounded answer
    if provider is not None:
        generator = AnswerGenerator(provider, settings=settings)
        answer = generator.generate(body.question, retrieval_result, trace_id=trace_id)
    else:
        # Safe fallback when no Gemini API key is configured
        if not retrieval_result.candidates:
            ans_text = (
                "No relevant guidelines were found in the index for this query. "
                "Additionally, GEMINI_API_KEY is not configured for text generation."
            )
            abstained = True
        else:
            top_sources = ", ".join(
                f"{c.chunk.document_id} (page {c.chunk.page_start})"
                for c in retrieval_result.candidates[:3]
            )
            ans_text = (
                f"Retrieved {len(retrieval_result.candidates)} guideline passage(s) "
                f"from {top_sources}. (LLM generation is currently unconfigured; "
                "set GEMINI_API_KEY to generate synthesized answers)."
            )
            abstained = False

        answer = Answer(
            question=body.question,
            text=ans_text,
            citations=[],
            retrieval=retrieval_result,
            strategy=body.strategy,
            abstained=abstained,
            trace_id=trace_id,
            disclaimer=MEDICAL_DISCLAIMER,
        )

    return QueryResponse(
        answer=answer.text,
        citations=[
            CitationResponse(
                document_id=c.document_id,
                source_title=c.document_title,
                page_start=c.page_start,
                page_end=c.page_end,
                quote=c.quote,
                supported=c.supported,
                validation_status=c.validation_status,
                validation_note=c.validation_note,
            )
            for c in answer.citations
        ],
        retrieval=RetrievalMetadataResponse(
            strategy=retrieval_result.strategy.value,
            retriever=str(retrieval_result.retriever),
            retrieved_chunks=len(retrieval_result.candidates),
            expanded_queries=retrieval_result.expanded_queries,
            latency_ms=retrieval_result.latency_ms,
        ),
        abstained=answer.abstained,
        medical_disclaimer=answer.disclaimer,
        trace_id=trace_id,
    )
