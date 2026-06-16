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

### Concretely in our project
- Knowledge base: 2 AWS architecture PDFs → 321 chunks → Chroma DB (`./chroma_db`)
- When the user asks an architecture question, `search_patterns()` fetches the top-3 passages, and the agent answers in a well-founded way instead of from the top of its head.
