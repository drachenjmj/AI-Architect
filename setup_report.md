# Setup Report & RAG Explanation

## Part 1 – Setup Struggles (Short Report)

**1. Wrong embedding model (404 Not Found)**
The originally intended model `text-embedding-004` was not even enabled for the API key in use. Solution: query the actually available models via `ListModels` → switch to `gemini-embedding-001` and later `gemini-embedding-2`.

**2. Free-tier quota of the API key (429 Resource Exhausted)**
The key runs in the free tier with a metered, highly contended limit (initially 100 requests/min, later 1000/day per model). A single bulk build failed immediately with `429`. Solution: batch insertion with automatic retry + backoff instead of a single `from_documents` call.

**3. Duplicates in the vector store (642 instead of 321 vectors)**
`Rag_Setup.ipynb` was not idempotent – every additional run inserted the chunks again. Solution: call `delete_collection()` before inserting.

**4. Windows file lock**
The first idempotency approach (`shutil.rmtree`) failed with `PermissionError` because Chroma exclusively locks the index file `data_level0.bin`. Solution: deletion via the Chroma API (`delete_collection()`) instead of via the file system.

**5. Transient network errors (502 Bad Gateway / DNS outages)**
The environment showed unstable networking at times. Solution: extend the retry logic to all transient errors (429, 5xx, DNS/timeout).

**6. Duplicate code maintenance (notebook vs. app)**
Logic existed in parallel in `agent.ipynb` and `app.py`. Solution: extraction into a shared module `architect.py` (Single Source of Truth) – changes only in one place.

**7. Daily quota exhaustion from tests**
The many rebuilds had used up `gemini-embedding-001` for the day (1000/day). Solution: switch to `gemini-embedding-2` (separate quota, same vector dimension) – consistent across build and query directions.

**8. Outdated / deprecation warnings**
`google.generativeai` and `langchain-community` are deprecated; `langchain-chroma` was not installed. Purely cosmetic, no functional impact.

---

## Part 2 – RAG explained simply (for someone with coding basics)

### The core idea in one sentence
RAG (Retrieval-Augmented Generation) is an "open-book exam" for a language model – instead of answering only from memory, it may first look something up in its own knowledge store.

### Why do you need it?
A normal LLM (like Gemini) knows a lot, but not your company-specific documents – and it cannot be up to date. RAG pulls exactly the relevant passages from your PDFs into the model before it answers.

### The flow – like a pipeline with 5 stages

```
PDFs ──► 1. Shredding ──► 2. Translating ──► 3. Storing
                                                    │
              5. Answering ◄── 4. Searching ◄───────┘
```

1. **Shredding (chunking):** A long PDF is cut into small, handy pieces (~1000 characters) – like a book into paragraphs.
2. **Translating (embedding):** Each piece becomes a vector = a list of numbers that captures the content/meaning. Similar texts get similar numbers – practically GPS coordinates in meaning space.
3. **Storing (vector store / Chroma):** All vectors end up in a database that is particularly fast at "finding the most similar pieces."
4. **Searching (retrieval):** For a question, the question itself is translated into a vector – and the DB returns the 3 most similar text pieces.
5. **Answering (generation):** These pieces are sent to Gemini together with the question: "Answer this based on these sources." The model formulates the answer and can cite the source.

### The library metaphor
Imagine asking a librarian (vector store): "What is good about microservices?" He picks out the most relevant book pages and hands them to an expert (LLM). The expert reads them through and explains it to you in his own words – with source citations. That is exactly what `search_patterns()` + Gemini do in our code.

### Concretely in our project (final state after the curated rebuild)

**Three knowledge boxes.** The knowledge base is intentionally split into three sources of grounding:

- **Box 1 — general architecture knowledge** (440 vectors): `architecture_patterns_v2.md` (curated pattern reference), `microservices-on-aws.pdf`, `wellarchitected-serverless-applications-lens.pdf`.
- **Box 2 — curated e-commerce domain knowledge** (63 vectors): 8 curated Markdown files covering service boundaries, migration, Saga/compensation, checkout flow, database-per-service, polyglot persistence, search scaling, inventory concurrency/reservation, and payment idempotency.
- **Box 3 — live grounded web fallback:** when the internal KB produces no usable result, `architect.py` queries Gemini with Google-Search grounding (`WEB_SEARCH_MODEL = "gemini-2.5-flash"`, separate from the main model) and returns only chunks with real grounding evidence.

**Measured facts (2026-08-21 rebuild):**

- 503 vectors total (440 Box 1, 63 Box 2); chunk size 1000, overlap 200; embeddings `models/gemini-embedding-2`.
- Retrieval: raw Chroma top-k with `DISTANCE_THRESHOLD = 0.65`; chunks carry `{content, source, page, box, distance}` (Box-3 chunks: `box=3`, `distance=null`).
- Validation on 10 e-commerce queries: 9 STRONG / 1 GOOD / 0 WEAK / 0 MISS; best-distance 0.3170–0.5487 (mean 0.4000). On 6 out-of-domain negatives: best-distance 0.6914–0.7944 (mean 0.7448) — no overlap with the positives, so 0.65 sits inside a clean separation gap.
- Every fallback outcome is logged with an explicit status (`web_fallback`, `web_fallback_empty`, `web_fallback_error`, `web_fallback_disabled`); `kb_gap_report.py` groups any of them into a frequency table of internal-KB gaps ("consider adding to KB box 1/2").
- Raw upstream sources (full PDFs, v1 pattern file) are preserved in `Rag Database/raw_source_archive/`, outside the ingestion globs — the active index contains only curated content.
- Offline test suite: 302 passed.

**Design rationale.** The active knowledge base is intentionally *curated* rather than "ingest everything available": only high-signal sections were extracted per source, because the current retrieval path is a shared collection with top-k search and no domain-specific router or reranker — precision comes from corpus quality, not ranking tricks. Raw material that did not fit that bar is retained on disk but not indexed.

> The current active knowledge base is intentionally curated for retrieval precision. Broader raw knowledge is retained but not all of it is indexed because the current retrieval path has no domain-specific router or reranker. Future extensions can introduce additional domain/technology knowledge packs and metadata-aware routing.

**Future work.** Additional domain/technology knowledge packs (curation-first), metadata-aware routing or box filtering, and reranking of raw candidates before the threshold cut.
