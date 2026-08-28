# MedGuard evaluation

Everything here exists so that retrieval claims are numbers from a recorded run rather
than adjectives. It measures **retrieval only** — which chunks come back for a query.
Answer quality, faithfulness and citation correctness are Phase 6, and nothing in this
directory says anything about them. MedGuard is not a diagnostic system.

```
evals/
  golden_set_v1.jsonl        questions + human-reviewable ground truth
  golden_set_v1.meta.json    the chunk-id basis the set was authored against
  reports/<timestamp>/       report.md, results.json, provenance.json
  reports/summary.jsonl      one line per run, for trends
  reports/scratch/           parameter sweeps (gitignored)
```

Run it:

```bash
medguard eval                          # dense + sparse over the golden set
medguard eval --retriever dense        # one leg
medguard eval --limit 5 --no-write     # quick check, no report directory
```

## Status: the v1 set is a worked example, and it is unreviewed

`golden_set_v1.jsonl` currently holds **20 questions, all LLM-drafted, none human
reviewed**. `medguard eval` refuses to score it without `--allow-unreviewed`, and every
run over it is stamped provisional in both the terminal output and `provenance.json`.

That is deliberate and it is not a formality. The characteristic failure of a generated
question is that it echoes its source chunk's wording, so the lexical retriever matches
it trivially and the set ends up measuring word overlap rather than retrieval quality.
Nothing in the metrics shows this. A human reading the question next to its chunk sees
it immediately. Until that reading happens, the numbers describe an unvalidated
instrument.

The target is 75 questions. The remaining 55 are authored against the methodology below.

## Why 75, and not 50 or 100

For a proportion-like metric near 0.5 the 95% confidence half-width is roughly
`1.96·√(0.25/n)`: **±14 points at n=50, ±11 at n=75, ±10 at n=100**. Going 50 → 75 buys
three points; 75 → 100 buys 1.4 for another two hours of annotation. That is where the
curve flattens.

The unpaired figure also understates the design. Every comparison here is **paired** —
dense, sparse and hybrid run the same questions — so what is being tested is the
per-question delta, not two independent means. A paired bootstrap over 75 deltas detects
a shift of about 0.04–0.05 in nDCG@10, comfortably inside the range worth claiming. The
±11 is the interval on the absolute number; the paired interval is what licenses
"hybrid beat dense".

Seven question types over 75 also gives roughly ten per type, which is the minimum for a
directional per-type read. At 50 it is seven, which is noise one would be tempted to
over-read.

## Question types

| Type | What it tests | Ground truth |
|---|---|---|
| `factual` | A stated fact in one place | Required |
| `terminology` | A definition or a table lookup | Required |
| `multi_document` | An answer spanning two or more documents | Required |
| `complex` | Joining two facts, or reasoning across them | Required |
| `ambiguous` | Under-specified queries a real search box receives | Optional |
| `unsupported` | Plausible, and genuinely not in the corpus | Empty |
| `adversarial` | False premises and prompt injection | Empty |

Target distribution for 75: factual 20, terminology 10, multi-document 12, complex 12,
ambiguous 7, unsupported 8, adversarial 6.

The last three types have no correct chunk, so precision, recall, MRR and nDCG are all
**mathematically undefined** for them. They are excluded from every ranking metric —
`n` is stated on every table — and are the abstention signal Phase 6 will score once
there is an answer to abstain from. Averaging them in as zeros would move the headline
by up to a fifth and mean nothing in either direction.

## Who authors what

| Type | Author | Why |
|---|---|---|
| `unsupported`, `adversarial` | Human | An LLM drafting *from a chunk* cannot write a question that is out-of-corpus by construction. The technique defeats the purpose of the type. |
| `multi_document` | Human | Requires holding two documents in mind at once, which chunk-conditioned drafting structurally cannot do. |
| `ambiguous` | Human | Ambiguity is a property of the question against the whole corpus, not against one chunk. |
| `factual`, `terminology`, `complex` | LLM-drafted, human-reviewed | The seed chunk is relevant by construction, so ground truth starts out known. |

The v1 worked example does not yet follow this split — all twenty were LLM-drafted,
including the six unanswerable ones (written by verifying absence against the corpus
rather than by drafting from a chunk). The full set should follow it.

Drafting uses Gemini at `temperature=0` with the prompt versioned by a module constant,
so a report can name the prompt that produced a question. The system prompt forbids
yes/no questions, requires clinical phrasing, forbids reusing distinctive multi-word
phrases from the passage, and requires an empty result rather than an invented question
when the passage is a table of contents or a reference list.

## Human review checklist

Every LLM-drafted question, no exceptions. Record initials and ISO date in
`reviewed_by`; `medguard eval` refuses to score anything still `null`.

1. **Answerable from the cited chunks alone.** Read the chunks. Do not trust the note.
2. **No rare multi-word span shared verbatim with the source chunk.** Paraphrase into
   the way a clinician would actually ask.
3. **The chunk set is complete, not just the seed.** Search for other chunks that also
   answer it. A question answered in three chunks but labelled with one measures
   precision against a lie and understates recall systematically.
4. **The type label is correct.**
5. **Not a near-duplicate** of an existing question.
6. **It reads like something a person would ask.**
7. **Initials and date recorded.**

## Finding ground truth

```bash
uv run python scripts/find_chunks.py "when should PrEP be offered"
```

The script pools candidates from three channels that fail differently — dense (semantic,
misses rare tokens), BM25 (lexical, misses paraphrase) and a SQL `LIKE` scan (literal,
misses everything else) — and prints the union with page numbers and previews. This is
TREC-style **pooled relevance assessment**.

Two things about it have to be said out loud rather than discovered later.

**The circularity is real.** Labelling ground truth from your own retriever's output
biases the labels toward what that retriever already finds, which inflates every metric
computed afterwards. Pooling three channels at depth mitigates it; it does not remove
it. The other half of the mitigation is human: open the document around a candidate,
because the *adjacent* chunk is often the better answer and no channel will suggest it.

**Recall is measured against the pool, not against all 5,185 chunks.** A relevant chunk
that no channel surfaced at depth 30 is invisible to the metric. Absolute recall is
therefore an upper bound. The *paired* comparison between retrievers stays valid,
because every retriever is scored against the same pool.

## Corpus properties worth knowing before you label

- **Bibliographies rank well and answer nothing.** Several documents devote a third of
  their pages to references, and those chunks are dense in topic terms, so BM25 ranks
  them highly. Never label one relevant.
- **Tables of contents behave the same way**, especially in the 440-page asthma
  guideline.
- **Nine of twenty documents state no publication date.** Retrieval therefore exposes
  no date filter at all — see below.
- **`cdc_antibiotic_resistance_2019` is the corpus's only `document_type: reference`.**
  It needs questions of its own, or the `document_type` filter has nothing to select.

## The tripwire: chunk ids are position-addressed

Chunk ids are `{document_id}::{chunk_index:04d}`. They encode a *position*, not content.
Change `INGESTION_CHUNK_SIZE` and re-ingest, and every `relevant_chunk_id` in the golden
set points at different text. The file still parses. The ids still resolve. The metrics
still compute. Every number is wrong, and nothing anywhere reports it.

`golden_set_v1.meta.json` records the chunking settings the ids were authored against,
and `medguard eval` checks them before running a single query. `relevant_document_ids`
is carried alongside for the same reason: it survives re-chunking, so an invalidated set
still holds a coarser usable signal while it is being re-labelled.

If the check fires, either revert the ingestion settings and re-index, or re-label
against the new chunking and publish a **new version**. Do not edit `v1` in place once
numbers have been published against it.

## What is not filtered, and why

There is no publication-date filter. Measured against the live 5,185-point collection:

- the `publication_date` payload index reports `points: 2635`; the other three report
  5185;
- a range condition of "has any date at all" counts 2,635, not 5,185;
- `is_null` counts exactly 2,550 — 49.2% of the corpus, including the largest document
  and the only `reference`-typed one.

Qdrant's `DatetimeRange` matches only points where the field is present, so any date
filter silently removes half the corpus. Date-only and RFC 3339 strings behave
identically, so the string format is not the problem. A filter that quietly halves the
searchable corpus is worse than no filter. Phase 4B revisits it; the fix is either
backfilling the dates or a null-tolerant condition, and that is a corpus decision.

## Reproducibility

Every run writes `provenance.json`: git commit and dirty flag, corpus manifest SHA-256,
embedding models and dimension, collection name, chunking settings, retrieval settings,
golden set version and question count, and whether any LLM was called. A metric without
those is not reproducible, and an irreproducible number is one question away from
falling apart.

A run made on a dirty tree is labelled as such in `report.md` and is not a publishable
figure.
