# Corpus sources — provenance and substitution record

This file records **how the corpus was chosen**: what was tried, what worked, what did
not, and every field that is `null` in `data/manifest.json` because the publisher does
not state it.

The PDFs themselves are not committed (see `.gitignore`). `data/manifest.json` is, and
`medguard corpus verify` proves that the files on any given machine are the ones it
describes.

**Every URL below was fetched and confirmed to return `%PDF-` magic bytes** — the CDC
and WHO entries on 2026-08-13 during Phase 2, the NIH and ICMR entries on 2026-08-13
during Phase 3A. Content-type was not treated as proof: several publishers serve an HTML
consent page or WAF block under a `.pdf` URL with HTTP 200 and
`content-type: application/pdf`.

---

## Corpus composition

20 entries, 19 downloadable, 1 retired, across 4 publishers.

| Publisher | Documents | Reuse terms |
|---|---:|---|
| CDC | 13 | Stated verbatim by the publisher (public domain) |
| NIH (NHLBI) | 3 | **Not published by the source.** Recorded as unstated |
| ICMR | 3 | Stated verbatim by the publisher (free reuse with attribution) |
| WHO | 1 (retired) | Not verifiable; source page gated |

| document_type | Count |
|---|---:|
| `guideline` | 19 |
| `reference` | 1 |
| `systematic_review` | 0 |
| `review` | 0 |
| `protocol` | 0 |

The corpus is deliberately **guideline-shaped**. Clinical guidelines carry consensus
positions with graded recommendations; a single primary study does not. A corpus that
drifts toward primary research will confidently answer "what is the first-line treatment
for X" from one trial with n=40 — a retrieval-quality problem that no amount of
reranking fixes.

`cdc_antibiotic_resistance_2019` is classified `reference`, not `guideline`: it is a
surveillance and burden report, not a recommendation set. It is included so the corpus
is not uniformly one document type.

The empty `systematic_review` and `review` rows are a direct consequence of PMC being
unreachable, since that was the intended source for both.

---

## Source selection

The intended priority order was **CDC > NICE > NIH/NIAID > WHO > PMC**. Each was probed
directly.

| Source | Result | Evidence |
|---|---|---|
| **CDC** (`www.cdc.gov`, `stacks.cdc.gov`) | **Used.** 31 MMWR Recommendations & Reports PDFs found reachable; 11 selected, plus 2 from CDC Stacks. | HTTP 200, `application/pdf`, `%PDF-` confirmed for every selected URL. |
| **NIH — NHLBI** | **Used in Phase 3A**, after being rejected in Phase 2. See the reversal below. | 3 of the probed NHLBI guideline PDFs returned valid `%PDF-` content and are now in the corpus. |
| **ICMR** (`www.icmr.gov.in`) | **Used in Phase 3A.** 3 documents. | HTTP 200, `%PDF-` confirmed 2026-08-13 for all three URLs. |
| **NICE** | **Rejected — unreachable.** | `https://www.nice.org.uk/guidance/ng136/resources/...-pdf-66141722710213` and three other NICE guideline PDFs all returned **HTTP 403** with a 3,357-byte HTML block page, both with a minimal header set and with a full browser navigation header set. |
| **NIH — clinicalinfo.hiv.gov** | **Rejected — unreachable.** | The adult ARV and opportunistic-infection guideline PDFs returned **HTTP 403** (HTML) even with full browser headers. |
| **WHO IRIS** | **Rejected — unreachable.** | Every `iris.who.int/bitstream/...` URL tried returns **HTTP 200 with a 432-byte HTML consent interstitial** (`<!DOC`), not the PDF. This includes the URL that worked during Phase 1. |
| **PMC** | **Rejected — anti-bot challenge.** | `pmc.ncbi.nlm.nih.gov/articles/PMC…/pdf/…` returns HTTP 200 with an HTML page titled *"Preparing to download …"* containing a JavaScript **proof-of-work challenge** (`POW_CHALLENGE`). Obtaining the PDF requires executing that JavaScript. |

**Note on the Phase 2 brief.** It stated that PMC "serve[s] direct PDFs without anti-bot
walls" and that CDC was the preferred primary. Only the second half held. PMC has since
added a JS proof-of-work gate, and CDC — which the brief expected to work plainly — in
fact returns HTTP 403 to any request without a full browser navigation header set. Both
findings are empirical, reproducible, and are why `CorpusSettings.user_agent` and
`_browser_headers()` exist in `src/medguard/corpus/download.py`.

### The header finding

`cdc.gov` returned **HTTP 403** with a 429-byte HTML block page to a request carrying
only `User-Agent` and `Accept`. The same URLs returned `application/pdf` once
`Accept-Language`, `Sec-Fetch-Dest`, `Sec-Fetch-Mode`, `Sec-Fetch-Site`,
`Sec-Fetch-User` and `Upgrade-Insecure-Requests` were added. This is why the downloader
sends a full navigation header set rather than a bare User-Agent.

Neither NHLBI nor ICMR needed it — both serve their PDFs to a plain request — but they
go through the same client, so they get the same headers.

---

## The NHLBI reversal: rejected in Phase 2, admitted in Phase 3A

Phase 2 excluded five reachable NHLBI documents on metadata grounds, and the finding
that drove it has **not changed**:

* NHLBI publishes **no document-specific usage statement** on any reachable page.
  `nhlbi.nih.gov/copyright`, `/policies` and `/about/policies-and-notices` all return
  404, and `nih.gov/web-policies-notices` returns 403. Re-checked on 2026-08-13 against
  all three landing pages now in the corpus: none states reuse terms.
* NHLBI's landing pages state a **year**, never a full publication date.

What changed is the judgement about which cost is higher. Phase 2 weighed three
unverifiable fields against publisher diversity and chose verifiability. That left 13 of
14 documents from a single publisher, which has its own cost: with one publisher,
"which organisation issued this recommendation?" has one answer, and no retrieval or
citation behaviour that depends on provenance can be demonstrated at all — not to an
interviewer, and not to the evaluation harness in Phase 12.

The resolution is **not** to invent terms. `license_or_usage_note` for the three NHLBI
documents reads:

> NIH-authored publication; specific reuse terms not published. NHLBI states no
> document-specific usage terms on this document's landing page, and no reachable NHLBI
> policy page states any. Recorded as unstated rather than assumed: US Government works
> are generally public domain, but that is an inference about the publisher, not a term
> this publisher published.

That is a true statement about what is known. Attaching NLM's statement — a *different*
institute's page, which is what Phase 2 correctly refused to do — would have been a
false one. The rule "never invent metadata" is satisfied by recording the absence, not
by excluding the document.

`publication_date` is `null` for all three, for the same reason it is null for two CDC
Stacks documents: a year is not a publication date.

### Which NHLBI documents, and which were left out

| Document | In corpus | Reason |
|---|---|---|
| Sickle cell disease, Expert Panel Report 2014 | ✅ | Current; distinct clinical area |
| Integrated cardiovascular guidelines for children and adolescents | ✅ | Current enough; paediatric coverage nothing else in the corpus provides |
| Asthma EPR-3 | ✅ | Still the reference NHLBI asthma document, and explicitly versioned |
| **JNC 7** (hypertension, 2003) | ❌ | **Superseded and too old.** Its successor guidance has changed first-line recommendations. In a corpus queried for current practice, it is a source of confidently wrong answers |
| **Obesity guidelines (1998)** | ❌ | **Too old for a 2026 medical corpus.** 28-year-old clinical guidance retrieves as readily as current guidance and the retriever cannot tell the difference |

The currency exclusions are a deliberate, defensible position: **age is a corpus-quality
property, not a metadata field to record and ignore.** A retrieval system has no way to
prefer 2024 guidance over 2003 guidance unless the corpus curator declines to index the
2003 guidance, or unless recency is modelled explicitly — which it is not, in this
project, at this phase.

### One substitution within NHLBI

The paediatric cardiovascular entry was specified as `peds_guidelines_sum.pdf` (the
summary report). That filename returns **HTTP 404** at every path tried on 2026-08-13;
only `peds_guidelines_full.pdf` resolves. The full report is used instead, and is the
better corpus document regardless: it carries the graded evidence statements that the
summary condenses away.

---

## ICMR

The one publisher in this corpus that states its reuse terms in the affirmative, and the
only non-US publisher. Both matter. A corpus of exclusively US federal guidance encodes
US clinical practice as though it were the whole of medicine — which is a real
limitation for a system that answers medical questions, and one worth being able to
point at.

### Licence terms, quoted verbatim

From <https://www.icmr.gov.in/copyright-policy>:

> Material featured on this Website may be reproduced free of charge. However, the
> material has to be reproduced accurately and not to be used in a derogatory manner or
> in a misleading context. Wherever the material is being published or issued to others,
> the source must be prominently acknowledged. However, the permission to reproduce this
> material shall not extend to any material which is identified as being copyright of a
> third party. Authorization to reproduce such material must be obtained from the
> departments/copyright holders concerned.

MedGuard cites every source document it draws on, which satisfies the acknowledgement
requirement.

### Titles, and the year prefix

ICMR's guidelines listing page prefixes each entry with its publication year. The exact
listing text, and the title recorded in the manifest:

| Listing text | Recorded title |
|---|---|
| `2022-Treatment Guidelines For Antimicrobial Use in Common Syndromes` | Treatment Guidelines for Antimicrobial Use in Common Syndromes |
| `Guidelines for Management of Type 2 Diabetes` | Guidelines for Management of Type 2 Diabetes |
| `2017-Hospital Infection Control Guidelines` | Hospital Infection Control Guidelines |

The `YYYY-` prefix is a listing convention rather than part of the title, so it is
dropped; the year is recorded in `version_or_revision`. The listing text is reproduced
above so the decision is checkable rather than merely asserted. **This is a judgement
call** — a reviewer who disagrees can restore the prefix from the table above.

### Why the 2019 AMR guidelines are not in the corpus

ICMR publishes both a 2019 and a 2022 edition of the antimicrobial treatment guidelines.
Only 2022 is indexed.

Two editions of the same live clinical guidance in one corpus means a query about
first-line antimicrobial therapy retrieves **contradictory recommendations with equal
confidence**, and the system has no principled way to prefer the newer one: both are
`guideline`, both are ICMR, and neither carries a full publication date the retriever
could rank on. The likely outcome is an answer citing the superseded dosing.

Version handling is a genuine capability and it is not being skipped — it is tested
against **synthetic fixtures** (`tests/fixtures/make_pdf.py`, two generated PDFs sharing
a `document_id_root` of `test_amr` with different `version_or_revision` values and
deliberately conflicting recommendations). That exercises the mechanism without putting
contradictory live medical guidance in front of a user.

Those fixtures are generated at test time and never enter `data/raw/`,
`data/sources.yaml` or `data/manifest.json`.

---

## Substitutions and removals

| Intended | Actual | Reason |
|---|---|---|
| `who_malaria_2023` (WHO IRIS) | Dropped, not replaced 1:1 | IRIS consent interstitial. CDC coverage of infectious-disease guidance (anthrax, botulism, TB, hepatitis B/C, HIV PrEP/PEP) substitutes for the topic area. |
| `who_essential_medicines_2023` (WHO IRIS) | Dropped, not replaced 1:1 | IRIS consent interstitial. |
| `who_hypertension_2021` | **Retired, file kept** | See below. |
| NICE hypertension / diabetes / asthma / depression | Dropped | HTTP 403 from NICE's WAF. |
| PMC systematic reviews | Dropped | JavaScript proof-of-work challenge. No `systematic_review` or `review` document made it into the corpus as a result. |
| NHLBI `peds_guidelines_sum.pdf` | `peds_guidelines_full.pdf` | The summary filename 404s; the full report is the better document anyway. |

### `who_hypertension_2021` — retired rather than deleted

The Phase 1 corpus contained one document, `who_hypertension_2021.pdf`, downloaded when
the IRIS URL still resolved. That URL is now gated.

The brief said to remove unreachable URLs from `sources.yaml`. Removing the entry
outright would have left the PDF on disk with **no provenance record at all**, and
`medguard corpus verify` would have reported it as an orphan on every run, forever.

So the entry is kept with `retired: true` and a `retired_reason`. The downloader skips
it; the manifest still records its hash and origin. `retired` means *"do not attempt to
download"*, not *"not part of the corpus"*.

**Two fields on this document are explicitly not verified**, and say so in the manifest:

* `license_or_usage_note` begins **"NOT VERIFIED."** WHO IRIS titles are generally
  CC BY-NC-SA 3.0 IGO, but that could not be confirmed for this document: the source page
  is gated, and there is no PDF parser with which to read the terms off the document
  itself. It is recorded as unverified rather than asserted.
* `title` is as recorded in Phase 1 and was not independently re-verified, for the same
  reason.

Both should be re-checked when a PDF parser lands in Phase 3B.

---

## Null fields, and why

Under "never invent metadata", a field the publisher does not state is `null`. Each
`null` in the manifest is accounted for here.

### `publication_date` is null for nine documents

| Document | Reason |
|---|---|
| `cdc_hiv_prep_2021` | CDC Stacks records `2021/12/01`. The day component is a catalogue placeholder, not a stated publication date. The year appears in the title. |
| `cdc_antibiotic_resistance_2019` | CDC Stacks records `2019/01/01`, a year-only placeholder rendered as a date. Same treatment. |
| `who_hypertension_2021` | Source page gated; no date could be read. |
| `nhlbi_sickle_cell_2014` | Landing page states the year 2014 (published September 2014); no full date. |
| `nhlbi_pediatric_cv_2011` | Landing page states January 2012 for a 2011 expert panel. The two disagree and neither is a full date. |
| `nhlbi_asthma_epr3_2007` | Landing page gives 2007 for the guidelines and September 2012 for the posted file. |
| `icmr_amr_treatment_2022` | ICMR's listing states a year only. |
| `icmr_type2_diabetes_2018` | ICMR's listing states a year only. |
| `icmr_hospital_infection_control_2017` | ICMR's listing states a year only. |

The other eleven documents have full, authoritative dates taken from each report's DOI
record at `api.crossref.org` (for example `10.15585/mmwr.rr7103a1` → `2022-11-04`).

`medguard corpus verify` reports each of these as a **warning**, not an error. That is
deliberate: a null date here is a correct record of an absent fact, and the warning
exists so the count stays visible rather than being forgotten.

Reading a date off a document's title page is a Phase 3B capability. When it exists, a
discovered date is written to the `documents` table and promoted into the manifest by an
explicit `medguard corpus sync --backfill-publication-dates` — never by ingestion
writing to the manifest directly.

### `page_count` is null for every document

Counting pages requires a PDF parser. PyMuPDF is a Phase 3B dependency and is
deliberately absent from Phase 3A, including from test infrastructure — the synthetic
fixtures are generated with `reportlab`, which writes PDFs and cannot read them.

The field exists in the schema so it stays stable across the phase boundary; Phase 3B
fills it, via the same database-then-backfill route as `publication_date`. `null` means
"not counted yet", never "zero pages".

### `title` is taken from the publisher, not from PDF metadata

The manifest schema suggested taking `title` from PDF metadata where extractable. There
is no PDF parser, so instead each title is the one the **publisher states** — read from
the MMWR landing page's `citation_title` meta tag, the CrossRef DOI record, the CDC
Stacks catalogue entry, the NHLBI document landing page, or the ICMR guidelines listing.

This is arguably *better* than the PDF's own `/Title` field, which in government PDFs is
frequently the production filename rather than the publication title.

**This is now a guardrail rather than a temporary state.** `DocumentRow.title` in
`src/medguard/storage/schema.py` carries a comment stating that a Phase 3B extracted
title must never overwrite a publisher-sourced one. The citations shown to a clinician
are rendered from that column, and replacing verified titles with build artefacts such
as `rr7103a1-H` would degrade them silently.

---

## Licence terms by publisher

**MMWR documents** (11), from <https://www.cdc.gov/mmwr/about.html>:

> All material in the MMWR Series is in the public domain and may be used and reprinted
> without special permission; citation as to source, however, is appreciated.

**CDC Stacks documents** (2), from <https://www.cdc.gov/other/agencymaterials.html>:

> Most of the information on the CDC and ATSDR websites is not subject to copyright, is
> in the public domain, and may be freely used or reproduced without obtaining copyright
> permission. Attribution to the agency that developed the material must be provided in
> your use of the materials.

**ICMR documents** (3), from <https://www.icmr.gov.in/copyright-policy> — quoted in full
in the ICMR section above.

**NHLBI documents** (3) — no terms published by the source. Recorded as unstated; see
the NHLBI section above.

**WHO document** (1, retired) — not verifiable; recorded as `NOT VERIFIED`.

MedGuard cites every source document it draws on, which satisfies every attribution
request above.

---

## Reproducing this

```bash
medguard corpus download   # fetch everything in data/sources.yaml
medguard corpus verify     # check every file against data/manifest.json
medguard corpus verify -v  # also report values the database has that the manifest lacks
```

`verify` exits non-zero on any integrity error, so it works as a CI gate. Hashes are
plain SHA-256 over raw file bytes and are reproducible with `sha256sum`.
