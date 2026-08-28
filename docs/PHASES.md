# Phase numbering

This file is the canonical phase list. Where a docstring elsewhere in the codebase names
a different number, **this file wins**.

That situation exists on purpose rather than by neglect. The original plan ran to sixteen
phases with retrieval, generation and evaluation spread across most of them. Phase 4
collapsed that: building the evaluation harness *before* the retrieval improvements it
measures turned out to be the only ordering that produces numbers rather than opinions,
and doing so pulled work from six later phases into one. Roughly nine docstrings still
carry the old numbers — `models.py` says "Phase 12" of the evaluation contracts,
`embeddings/base.py` says "Phase 7" of the sparse leg, and so on.

Those docstrings were left alone deliberately. Each one explains *why* a decision was
made, and that reasoning is still correct even where the number attached to it is not.
Rewriting nine of them to renumber a label would churn the parts of the code most worth
reading, for no gain a reader could not get from this page.

## Phases

| Phase | Scope | Status |
|---|---|---|
| 0 | Repository audit | Done |
| 1 | Foundation: config, logging, domain models, provider, CLI, API, diagnostics | Done |
| 2 | Corpus acquisition, provenance manifest, database schema, migrations | Done |
| 3A | Corpus expansion, page-range schema, backfill, ingestion config | Done |
| 3B | Ingestion: PDF parsing, cleaning, chunking, embeddings, Qdrant indexing | Done |
| **4A** | **Retrieval evaluation harness, golden set, dense + sparse baselines** | **Done** |
| 4B | Hybrid retrieval with RRF fusion, query rewriting, `POST /query`, Langfuse spans | Not started |
| 5 | Cross-encoder reranking; HyDE, multi-query, decomposition | Not started |
| 6 | Grounded generation, citation validation, RAGAS answer metrics, streaming | Not started |
| 7 | User-facing interface | Not started |
| 8+ | Security hardening, deployment | Not started |

## What Phase 4A delivered

- `RetrievalSettings`, with `fetch_k` separate from `top_k`.
- `VectorStore.search_dense` / `search_sparse`, and a typed metadata filter.
- `medguard.retrieval`: the `Retriever` protocol, `RetrievalFilter`, and the dense and
  sparse retrievers, both returning the existing `Candidate` type.
- `medguard.evaluation`: golden-set schema and loader, four hand-implemented metrics,
  a runner, and a report writer that records full provenance.
- `medguard eval`, and `scripts/find_chunks.py` for labelling ground truth.
- A 20-question worked example at `evals/golden_set_v1.jsonl` — LLM-drafted, **not yet
  human-reviewed** — and the first recorded dense and sparse baselines.

## What Phase 4A deliberately did not do

- No hybrid retrieval or RRF fusion, no query rewriting, no `POST /query`, no Langfuse
  instrumentation. All Phase 4B.
- No reranking, and no HyDE, multi-query or decomposition. Phase 5.
- No answer generation and no answer-quality metrics. Phase 6.
- No publication-date filter: 9 of 20 documents state no date, and a Qdrant range
  condition drops all of them silently. See `evals/README.md`.
- `check_index` was left non-critical in `medguard doctor`. It becomes critical in 4B,
  when a drifted index starts serving user-facing queries rather than only evaluation
  runs.
