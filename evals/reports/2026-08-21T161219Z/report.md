# MedGuard retrieval evaluation

> **PROVISIONAL — not a publishable figure.** 20 of 20 golden-set questions were drafted with LLM assistance and no human has reviewed them. Generated questions tend to echo their source chunk's wording, which the lexical retriever then matches trivially -- a failure these metrics cannot show. Until each question is checked against the checklist in `evals/README.md`, the numbers below describe an unvalidated instrument.

- Generated: `2026-08-21T16:12:19.573917+00:00`
- Commit: `52f6b99af3855357c486b4d172a9f64ea0940ce1`  **(dirty tree — not a publishable figure)**
- Collection: `medguard__bge_small_en_v1_5__384`
- Embeddings: `BAAI/bge-small-en-v1.5` (384d) + `Qdrant/bm25`
- Golden set: `v1` — 20 questions, 14 scored

## How to read this

Ranking metrics are averaged over the **14 questions that have at least one relevant chunk**. The remaining 6 (gq_015, gq_016, gq_017, gq_018, gq_019, gq_020) are unsupported, adversarial or ambiguous by design: precision, recall, MRR and nDCG are all mathematically undefined with an empty relevant set, so averaging them in would be inventing a number. They are reported separately below and scored properly in Phase 6, once there is an answer to abstain from.

Latency excludes a warm-up call: the first query builds a ~130 MB ONNX session that no subsequent query pays for. Embed and search are reported separately because on a CPU-only machine the embed can exceed the search it feeds.

## Ranking metrics

| Retriever | P@1 | P@3 | P@5 | P@10 | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `dense` | 0.286 | 0.214 | 0.214 | 0.121 | 0.167 | 0.393 | 0.601 | 0.649 | 0.774 | 0.869 | 0.286 | 0.328 | 0.438 | 0.461 | 0.472 |
| `sparse` | 0.357 | 0.190 | 0.143 | 0.100 | 0.190 | 0.286 | 0.381 | 0.571 | 0.714 | 0.905 | 0.357 | 0.297 | 0.342 | 0.408 | 0.468 |

## Latency

| Retriever | embed p50 | embed p95 | search p50 | search p95 | total p50 | total p95 |
|---|---|---|---|---|---|---|
| `dense` | 22.937 ms | 68.129 ms | 22.938 ms | 66.035 ms | 51.836 ms | 121.445 ms |
| `sparse` | 0.249 ms | 0.341 ms | 16.908 ms | 42.971 ms | 17.151 ms | 43.138 ms |

## Reranker headroom (Phase 5 input)

**`dense`** — real headroom (0.220 recall between k=10 and k=50, 2 question(s) whose first relevant chunk sits below rank 10): reranking has material upside

- recall@10 0.649 → recall@50 0.869 (delta 0.220)
- 2 question(s) whose first relevant chunk ranks below 10, mean rank 28.5

**`sparse`** — real headroom (0.333 recall between k=10 and k=50, 2 question(s) whose first relevant chunk sits below rank 10): reranking has material upside

- recall@10 0.571 → recall@50 0.905 (delta 0.333)
- 2 question(s) whose first relevant chunk ranks below 10, mean rank 16.0

## nDCG@10 by question type

| Retriever | complex | factual | multi_document | terminology |
|---|---|---|---|---|
| `dense` | 0.333 | 0.608 | 0.099 | 0.704 |
| `sparse` | 0.224 | 0.467 | 0.257 | 0.643 |

Per-type figures on a 20-question worked example rest on two to four questions each. They indicate direction and nothing more; the full 75-question set is what makes them worth quoting.

---

This measures **retrieval only** — which chunks come back for a query. It says nothing about answer quality, and MedGuard is not a diagnostic system.
