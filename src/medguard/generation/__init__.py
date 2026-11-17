"""Grounded answer generation and citation verification package."""

from __future__ import annotations

from medguard.generation.citation_validator import CitationValidator, validate_citations
from medguard.generation.context import build_grounding_context, sanitize_for_prompt
from medguard.generation.generator import AnswerGenerator, generate_grounded_answer

__all__ = [
    "AnswerGenerator",
    "CitationValidator",
    "build_grounding_context",
    "generate_grounded_answer",
    "sanitize_for_prompt",
    "validate_citations",
]
