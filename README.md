# MedGuard

Medical knowledge RAG and evidence retrieval system — hybrid retrieval, reranking,
measured evaluation, and observability over a corpus of published medical reference
documents.

> **This is not a medical diagnosis system.** MedGuard reports what its indexed
> sources say, with citations. It does not diagnose, has not been clinically
> validated, and is not a medical device. Answers are informational only and are not
> a substitute for consultation with a qualified healthcare professional. Always
> verify against the cited sources.

**Status: Phase 1 of 16 complete** — foundation only. There is no retrieval, no
chunking, no embeddings, and no generation pipeline yet. See
[Roadmap](#roadmap) for what exists and what does not.

---

## Quickstart

```bash
# 1. Install (Python 3.11+, uv)
uv sync

# 2. Configure
cp .env.example .env        # then add GEMINI_API_KEY — free at https://aistudio.google.com/apikey

# 3. Start the vector database (optional in Phase 1; required from Phase 3)
docker compose -f docker/docker-compose.yml up -d

# 4. Verify the environment
uv run medguard doctor

# 5. Serve the API
uv run medguard serve       # http://127.0.0.1:8000/docs
```

`medguard doctor` is the acceptance test. It reports Python, CPU, RAM, GPU, Docker,
Qdrant, the database, Langfuse, and — importantly — **asks the Gemini API which models
your specific key can actually use**, rather than trusting documentation. Free-tier
model availability varies by API project and changes over time.

Add `--smoke-test` to spend one real generation call proving the key can generate, not
merely list. It is off by default because it consumes free-tier quota.

```bash
uv run medguard doctor --smoke-test --show-config
```

`--show-config` prints effective settings with every secret replaced by `<set>` — no
prefix, no suffix, no length, since those still leak usable information.

---

## Design decisions

The non-obvious choices, and the reasoning behind them. These are the parts worth
challenging.

### ONNX instead of PyTorch

The development machine is CPU-only (AMD Ryzen 3 7320U, 7.3 GB RAM, integrated
Radeon 610M with no usable CUDA or ROCm) with roughly 2.5 GB of headroom. PyTorch plus
`sentence-transformers` costs ~2.5 GB on disk and ~1 GB resident before any model
loads. The whole stack then does not fit.

Embeddings therefore run through ONNX Runtime via `fastembed`, and **`torch` never
enters the dependency tree.** `BAAI/bge-small-en-v1.5` (384-dim, ~0.07 GB) handles
dense retrieval; `Qdrant/bm25` handles sparse. Projected total RSS is ~700–850 MB.

BGE-M3 was evaluated and rejected: 568M parameters, ~2.2 GB of weights, against
~2.5 GB of total headroom. The abstraction in
[`embeddings/base.py`](src/medguard/embeddings/base.py) is written so moving to
BGE-M3 or `bge-base` on better hardware is an environment-variable change plus a
re-index — not a refactor.

### BM25 alongside dense retrieval

Dense vectors smooth over exactly the tokens clinical text depends on: drug names,
dosages such as `12.5 mg`, ICD codes, abbreviations. BM25 cannot, because it matches
lexically. Whether this actually helps here is an empirical question answered in
Phase 7, not an assumption — which is why the baseline is measured first.

### Collections namespaced by embedding model

The effective Qdrant collection is `medguard__bge_small_en_v1_5__384`, derived from
the model id and dimension. Two embedding models produce incompatible vector spaces;
writing both into one collection silently degrades retrieval in a way that is very
hard to notice. Encoding the model in the name makes the mistake impossible and makes
A/B comparisons cheap.

### Langfuse Cloud, not self-hosted

Self-hosted Langfuse v3+ requires PostgreSQL, ClickHouse, Redis and MinIO — roughly
3–4 GB of RAM, which would make this machine unusable for development. The free cloud
tier costs zero local memory.

**Privacy consequence:** traces include your queries and retrieved document text, and
they leave this machine. That is acceptable for a corpus of published guidance
containing no PHI. It would need revisiting before pointing MedGuard at private
clinical records. Tracing is entirely optional — unconfigured, it degrades to a no-op
and the application is unaffected.

### SQLite by default, PostgreSQL by configuration

One `DATABASE_URL` drives both. SQLite needs no server and no RAM; PostgreSQL is one
environment variable and `--profile postgres` away, and is what the deployment config
uses. Application code cannot tell the difference.

---

## Security

| Concern | Approach |
|---|---|
| Secrets in logs | Redaction enforced at the log **sink**, not the call site. Three independent defences: sensitive key names, credential-shaped value patterns, and registered literal values. stdlib logging is routed through the same pipeline, so third-party SDKs cannot emit an unredacted line. |
| Secrets in config | All credentials are `SecretStr`. Reading the real value needs an explicit `.get_secret_value()`, which is greppable in review. |
| Secrets in errors | Vendor error payloads (which can echo request content) never reach a client. The API returns a safe `user_message` plus a correlation id; technical detail goes to the logs only. |
| Committing secrets | `.env` is gitignored; `.env.example` holds placeholders only. |

Prompt-injection detection, input/output validation, and API authentication arrive in
Phase 14.

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Repository audit | Done |
| 1 | Foundation: config, logging, domain models, provider, CLI, API, diagnostics | **Done** |
| 2 | Infrastructure | Compose file written; Qdrant not yet exercised |
| 3 | Ingestion: PDF parsing, chunking, corpus manifest | Not started |
| 4 | Dense retrieval | Not started |
| 5 | Baseline RAG | Not started |
| 6–8 | Sparse retrieval, hybrid RRF fusion, cross-encoder reranking | Not started |
| 9 | Query strategies: rewrite, HyDE, multi-query, decomposition | Not started |
| 10–11 | Grounded generation, citations with validation | Not started |
| 12–13 | Evaluation (100 questions, retrieval + RAGAS metrics), failure analysis | Not started |
| 14–16 | Security, testing, deployment | Not started |

**No performance or quality metrics are claimed anywhere in this repository.** None
have been measured yet. Numbers will appear only after the Phase 12 evaluation
harness produces them, and every claim will be reproducible from a recorded run.

---

## Development

```bash
uv run pytest              # 154 tests, no network access required
uv run ruff check .        # lint
uv run ruff format .       # format
uv run mypy                # strict type checking
```

The test suite runs fully offline. `tests/conftest.py` strips every application
environment variable before each test, so results never depend on whether the
developer has real credentials configured, and no test can accidentally spend API
quota.

### Layout

```
src/medguard/
  config.py            typed settings, one cached object, secrets as SecretStr
  logging.py           structlog + mandatory secret redaction
  models.py            domain types: Chunk, Candidate, Citation, Answer, …
  errors.py            vendor-neutral exception taxonomy
  diagnostics.py       the checks behind `doctor` and /ready
  providers/           LLM provider contract + the Gemini adapter
  embeddings/          embedding contracts (no implementation until Phase 4)
  storage/             Qdrant and SQLAlchemy adapters
  observability/       tracing abstraction with a no-op fallback
  cli/                 Typer application
  api/                 FastAPI application factory and routes
```

Only `providers/gemini.py` may import `google.genai`. Everything else depends on the
vendor-neutral contract in `providers/base.py`.

---

## Licence and sources

Source documents are published guidance from organisations such as the WHO. Each
document's provenance, licence and content hash are recorded at ingestion. PDFs are
not committed to this repository; `scripts/download_sample_docs.py` fetches them.
