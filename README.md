# MedGuard

Medical knowledge RAG and evidence retrieval system — hybrid retrieval, reranking,
measured evaluation, and observability over a corpus of published medical reference
documents.

> **This is not a medical diagnosis system.** MedGuard reports what its indexed
> sources say, with citations. It does not diagnose, has not been clinically
> validated, and is not a medical device. Answers are informational only and are not
> a substitute for consultation with a qualified healthcare professional. Always
> verify against the cited sources.

**Status: Phase 2 of 16 complete** — foundation, corpus and schema. There is a corpus of
13 clinical guideline PDFs with recorded provenance, and a versioned database schema.
There is still no parsing, chunking, embedding, retrieval, or generation. See
[Roadmap](#roadmap) for what exists and what does not.

---

## Quickstart

```bash
# 1. Install (Python 3.11+, uv)
uv sync

# 2. Configure
cp .env.example .env        # then add GEMINI_API_KEY — free at https://aistudio.google.com/apikey

# 3. Start the vector database (needed by `medguard ingest`)
docker compose -f docker/docker-compose.yml up -d

# 4. Create the database schema
uv run medguard db upgrade

# 5. Fetch the source corpus (~36 MB of public-domain CDC guidelines)
uv run medguard corpus download
uv run medguard corpus verify

# 6. Verify the environment
uv run medguard doctor

# 7. Serve the API
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

### The corpus is committed as a manifest, not as PDFs

`data/manifest.json` is in git; the 36 MB of PDFs are not. The manifest records each
document's title, publisher, URL, licence terms, SHA-256 and size, so a clone knows
exactly what the system was built on and can prove its local copies match — without
carrying government PDFs in git history forever.

The corpus itself is defined in `data/sources.yaml`, which is data rather than code:
changing the corpus does not mean editing Python.

---

## Corpus

20 documents, 59 MB, across four publishers — 19 active plus one retired document whose file is kept (`data/sources.yaml` carries 21 entries; the second retired one had its file deleted, and `data/raw/SOURCES.md` says why):

| Publisher | Documents | Reuse terms |
|---|---:|---|
| CDC | 13 | Quoted verbatim (public domain) |
| NIH (NHLBI) | 3 | **Not published by the source** — recorded as unstated |
| ICMR | 3 (+1 retired) | Quoted verbatim (free reuse with attribution) |
| WHO | 1 (retired) | Not verifiable; source page now gated |

Four publishers rather than one is a deliberate correction. A single-publisher corpus
makes "which organisation issued this recommendation?" a question with one answer, and
nothing downstream that depends on provenance can be demonstrated at all.

```bash
uv run medguard corpus download    # fetch what data/sources.yaml defines
uv run medguard corpus verify      # check every file against the manifest
uv run medguard corpus verify -v   # also report what the database knows and it does not
uv run medguard corpus sync        # record a file that exists but was never logged
uv run medguard corpus sync --backfill-page-counts   # promote values into null fields
uv run medguard ingest             # parse, chunk, embed and index the corpus
```

`verify` exits non-zero on any integrity error, so it works as a CI gate. A hash
mismatch, a missing file and an unrecorded file are all *errors*; a `null`
`publication_date` is a *warning*, because it is a correct record of an absent fact.
Run `corpus download` before `verify` on a fresh clone — the PDFs are not committed.

### Provenance rules

Four rules, and they are enforced rather than encouraged:

1. **Never invent metadata.** A field the publisher does not state is `null`, and
   [`data/raw/SOURCES.md`](data/raw/SOURCES.md) says why for each one. Nine documents
   have a null `publication_date`, because their source states a year or a catalogue
   placeholder like `2019/01/01` rather than a real date; recording that as fact would
   be indistinguishable from data. Where a publisher states no licence at all, the
   manifest says exactly that rather than borrowing a plausible one.
2. **Every URL is verified before it is added.** Content-type is not proof — WHO IRIS
   returns HTTP 200 with `content-type: application/pdf` and an HTML consent page in the
   body. The check is the `%PDF-` magic bytes, and it runs on the first bytes received.
3. **Licence terms are quoted verbatim**, with the page they were read from.
4. **Only `corpus download` and `corpus sync` may write the manifest.** Ingestion never
   does. It writes what it learns to the `documents` table, and a human promotes it with
   an explicit `corpus sync --backfill-*` that shows the diff and asks first. A pipeline
   that rewrites its own provenance record has destroyed the only independent check on
   itself. This is enforced by a test, not by convention.

`page_count` is filled in by `medguard ingest`, which is the first thing in the
project able to count pages. It stays `null` in the committed manifest until a human
promotes it with `corpus sync --backfill-page-counts`, for the reason in rule 4: the
pipeline is not allowed to write its own provenance record.

### What is in it, and what is not

The corpus is deliberately **guideline-shaped** — consensus recommendation sets, not
primary research. A corpus containing single trials will answer "what is the first-line
treatment for X" from one study with n=40, which is a retrieval-quality problem that no
amount of reranking fixes.

Publisher diversity is limited, and the limit is imposed rather than chosen. NICE and
NIH's `clinicalinfo.hiv.gov` return HTTP 403 to scripted requests; WHO IRIS returns a
consent interstitial; and PubMed Central has added a JavaScript proof-of-work challenge
in front of its PDFs. Phase 3A added NHLBI and ICMR, taking the corpus to four
publishers and one non-US source. [`data/raw/SOURCES.md`](data/raw/SOURCES.md) records
each of those findings with its evidence, along with every substitution made.

---

## Ingestion

```bash
uv run medguard ingest                       # parse, chunk, embed, index
uv run medguard ingest --dry-run             # parse and chunk only; loads no model
uv run medguard ingest --only cdc_anthrax_2023
uv run medguard ingest --limit 3
uv run medguard ingest --force-recreate      # DESTRUCTIVE: drop and rebuild the index
```

Four stages, three of them pure functions of their input:

| Stage | Module | Pure? |
|---|---|---|
| PDF → pages of text, 1-based page numbers | `ingestion/parser.py` | no (reads files) |
| de-hyphenation, whitespace, running headers | `ingestion/cleaner.py` | yes |
| recursive character split, page spans preserved | `ingestion/chunker.py` | yes |
| SQL rows + Qdrant points, per document | `ingestion/pipeline.py` | no (owns all state) |

**It is idempotent, and that claim has three parts.** A document is skipped only when
all of these hold: its status is `indexed`; the file on disk still hashes to what was
*indexed* (`documents.indexed_sha256`, which is deliberately a different column from the
manifest's `sha256`); and Qdrant holds exactly as many points for it as the database has
chunks. Dropping any one clause produces a system that reports success over an index
that cannot answer anything — which is the worst outcome available, because it looks
fine. If the collection is lost, the chunk rows in SQL are re-embedded without
re-parsing the PDFs.

**One document failing does not stop the run.** Each document is its own transaction and
its own error boundary; a failure records the reason on the row, removes any points that
document had, and moves on. The command exits non-zero if anything failed, because a
run that indexed eighteen of twenty documents has not succeeded.

**A dry run writes nothing** — no database rows, no Qdrant points, and no embedding
model is even loaded. It prints real page and chunk counts plus the first few chunks
verbatim, because counts cannot distinguish a chunker that splits on paragraphs from one
that cuts mid-sentence, and reading the text either side of a boundary can.

**Memory is the binding constraint on this hardware.** Ingestion refuses to start below
300 MB available, halves its batch size under pressure, and abandons a document below
150 MB rather than being killed by the OS partway through — which would leave a
half-populated collection and no error anywhere. Parsing and chunking the whole 20-
document corpus peaks at 378 MB RSS.

---

## Database

SQLite by default, PostgreSQL by configuration, one schema for both.

```bash
uv run medguard db upgrade              # apply migrations
uv run medguard db current              # what revision is this database at?
uv run medguard db history              # the migration history
uv run medguard db revision -m "..."    # new migration (a message is mandatory)
uv run medguard db downgrade base       # roll back (prompts first — this drops tables)
```

Two tables: `documents` (one row per corpus document, plus ingestion state) and
`chunks` (one row per retrievable passage, written by `medguard ingest`). `medguard doctor` reports when the database's schema
version is behind the code, which turns an otherwise baffling `no such column` deep
inside a request into one line of diagnostic output.

Migrations read `DATABASE_URL` from application settings rather than from `alembic.ini`,
so the app and its migrations cannot end up pointed at different databases.
`render_as_batch=True` is set from the first migration onward — SQLite implements almost
none of `ALTER TABLE`, and retrofitting batch mode later means rewriting history.

---

## Qdrant

```bash
docker compose -f docker/docker-compose.yml up -d qdrant
uv run python scripts/verify_qdrant_persistence.py
```

The persistence script writes known vectors to a throwaway collection, restarts the
container, and asserts the vectors came back. This is worth testing because a Qdrant
container with a broken volume mount behaves perfectly until its first restart, then
returns empty — and the symptom looks like a retrieval bug rather than a storage one.

It never runs `docker compose down -v`, and never touches the real collection.

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
| 2 | Corpus acquisition, provenance manifest, database schema, migrations | **Done** — 14 documents, Qdrant persistence verified |
| 3A | Corpus expansion, page-range schema, backfill, ingestion config | **Done** — 20 documents, 4 publishers |
| 3B | Ingestion: PDF parsing, cleaning, chunking, embeddings, Qdrant indexing | **Done** — 20 documents, 2,081 pages, 5,185 chunks |
| 4A | Retrieval evaluation harness, golden set, dense + sparse baselines | **Done** — 20-question worked example, dense and sparse measured |
| 4B | Hybrid retrieval with RRF fusion, query rewriting, `POST /query`, Langfuse traces | Not started |
| 5 | Cross-encoder reranking, HyDE, multi-query, decomposition | Not started |
| 6 | Grounded generation, citation validation, RAGAS answer metrics, streaming | Not started |
| 7 | User-facing interface | Not started |
| 8+ | Security, deployment | Not started |

Phase 4A moved this project from claiming nothing to measuring something. Retrieval
metrics now exist and live in `evals/reports/`, each run recorded with the commit,
corpus hash, chunking settings and embedding model that produced it.

Two things are still true and worth stating plainly. The golden set is a **20-question
worked example that no human has reviewed yet**, so `medguard eval` marks every run over
it provisional and refuses to score it without `--allow-unreviewed`; the numbers
describe an instrument that has not been validated. And nothing here measures **answer**
quality — retrieval metrics say which chunks come back, not whether an answer built from
them would be right. That arrives in Phase 6. See `evals/README.md` for the methodology
and its limitations.

---

## Development

```bash
uv run pytest -m "not integration"   # offline suite, no network access required
uv run pytest -m integration         # needs Docker and a running Qdrant
uv run ruff check .                  # lint
uv run ruff format .                 # format
uv run mypy                          # strict type checking
```

The unit suite runs fully offline. `tests/conftest.py` strips every application
environment variable before each test, so results never depend on whether the
developer has real credentials configured, and no test can accidentally spend API
quota. Corpus tests are redirected at a temporary directory, so they cannot read or
rewrite the real corpus and its committed manifest. Download behaviour is tested
against `httpx.MockTransport`.

### Layout

```
src/medguard/
  config.py            typed settings, one cached object, secrets as SecretStr
  logging.py           structlog + mandatory secret redaction
  models.py            domain types: Chunk, Candidate, Citation, Answer, …
  errors.py            vendor-neutral exception taxonomy
  diagnostics.py       the checks behind `doctor` and /ready
  corpus/              source list, safe download, provenance manifest
  providers/           LLM provider contract + the Gemini adapter
  constants.py         values that must never change once data exists
  embeddings/          embedding contracts + the fastembed (ONNX) implementations
  ingestion/           parser, cleaner, chunker, and the ingestion pipeline
  storage/             Qdrant adapter, SQLAlchemy engine, ORM schema
  observability/       tracing abstraction with a no-op fallback
  cli/                 Typer application
  api/                 FastAPI application factory and routes

data/
  sources.yaml         the corpus definition — data, not code
  manifest.json        generated provenance record (committed)
  raw/SOURCES.md       what was tried, what failed, why fields are null
  raw/*.pdf            the documents themselves (not committed)

migrations/            Alembic environment and revision history
```

Only `providers/gemini.py` may import `google.genai`. Everything else depends on the
vendor-neutral contract in `providers/base.py`. The same rule holds for the two
other vendor surfaces: `ingestion/parser.py` is the only module that imports
PyMuPDF, and `embeddings/fastembed_*.py` the only ones that import fastembed.

---

## Licence and sources

**Dependency note.** PyMuPDF, used for PDF text extraction, is AGPL-3.0 (or a
commercial licence from Artifex). It is the only copyleft dependency in the tree.
That is fine for this project, which is open and distributes no binary, but it is
the one dependency whose terms would matter to anyone reusing this commercially.
Everything else is permissively licensed.

Source documents are published clinical guidance from CDC, NIH/NHLBI, ICMR and WHO.
Each document's provenance, licence terms, content hash and size are recorded in
`data/manifest.json`, which is committed. The PDFs are not; fetch them with
`medguard corpus download`.

Usage terms are quoted verbatim from each publisher, with the page they were read from
— except for the three NHLBI documents, where the publisher states none and the manifest
records that absence rather than assuming one. MedGuard cites every source document it
draws on, which satisfies the CDC and ICMR attribution requests. Full detail, including
every source that was tried and rejected, is in
[`data/raw/SOURCES.md`](data/raw/SOURCES.md).
