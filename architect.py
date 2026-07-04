"""architect.py – Single Source of Truth for the AI Solution Architect.

This module contains all of the domain logic: the Gemini model, the RAG/Chroma
knowledge base, and the SQLite memory. Both app.py (Streamlit UI) and agent.ipynb
(terminal demo) import exclusively from here – changes only need to be made in this
one place.
"""
import os
import sqlite3
import sys
import time
from datetime import datetime

from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from rag_logger import RagLogger


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
MODEL_NAME = "gemini-2.5-flash"
CHROMA_DIR = "./chroma_db"
DB_PATH = "architect.db"

# None = inactive. Chroma returns a DISTANCE (lower = better); set to a float
# once a value has been derived from rag_log.jsonl to drop weak matches.
DISTANCE_THRESHOLD = 0.65

# Kill-switch for the web-search fallback: when True an empty knowledge-base
# retrieval is retried via Gemini Google-Search grounding so the architect can
# still answer. Set to False to restore the original behaviour (return []).
WEB_FALLBACK_ENABLED = True

ARCHITECTURE_KEYWORDS = [
    "architecture", "pattern", "microservice", "monolith",
    "event", "scal", "design", "structure", "system",
]

SYSTEM_PROMPT = """You are a world-class AI Solution Architect, designed for high-caliber IT architecture consulting within the BCG Platinion use case. Your task is to transform vague business requirements into precise, scalable, and production-ready system architectures.

You follow a strict two-phase approach:

PHASE 1: CONTEXT CAPTURE & REQUIREMENTS CLARIFICATION
Before you recommend any technology or draw a diagram/blueprint, you must first fully secure all boundary conditions. When the user expresses a vague idea, ask specifically about the following 5 core criteria if they have not yet been mentioned:
1. Cloud provider preference (AWS, Azure, GCP, On-Premise, or Hybrid)
2. Budget range (Low, Medium, High, or concrete figures)
3. Scaling requirements (expected user numbers, concurrent requests, data volume, peak events)
4. Compliance & security requirements (GDPR, PCI-DSS, encryption, EU data residency)
5. Existing system landscape (Brownfield integration vs. Greenfield build)

As soon as these parameters are clear or defined by plausible assumptions, create a "Context Record" as your first output (a frozen snapshot of all conditions and open questions).

PHASE 2: ARCHITECTURE DESIGN & DELIVERABLE ARTIFACTS
Only after context clarification do you generate the final architecture. Use the results from the RAG message (search_patterns) for this. Your output MUST strictly follow this structure in clean Markdown:

### 1. Context Record
- Summary of all validated constraints, user groups, and integration requirements.

### 2. Architecture Blueprint
You must provide two clearly separated views:
- **Stakeholder View:** An understandable description in clear business language. What does the system do? How do the data flow from a business perspective?
- **Technical View:** An in-depth technical specification for software engineers. Which services are used? What do the data model, module responsibilities, and integration structure look like?

### 3. Component Description
- Detailed listing and justification of each individual selected component or pipeline stage.

### 4. Architecture Decision Records (ADRs)
Create a short ADR for every significant design decision (e.g., microservices instead of monolith, a specific NoSQL database, etc.) in the following format:
- **Title:** ADR-[number]: [decision]
- **Context:** What technical challenge existed?
- **Considered options:** What alternatives were there?
- **Chosen option:** What will be implemented and why?
- **Trade-offs:** Which disadvantages or compromises (e.g., latency vs. consistency, cost vs. autonomy) are consciously accepted?

STYLE GUIDELINES:
- Always answer in English. Be precise, structured, and use professional consultant phrasing.
- Use Markdown tables and lists for comparisons and component overviews.
- Back up your recommendations with hard facts from the retrieved RAG documents and cite the corresponding sources/pages when you read them from the context.
"""


# ──────────────────────────────────────────────
# API KEY (internal)
# ──────────────────────────────────────────────
def _load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env!")
    return api_key


# ──────────────────────────────────────────────
# GEMINI MODEL (singleton – once per process)
# ──────────────────────────────────────────────
_model = None


def get_model():
    global _model
    if _model is None:
        genai.configure(api_key=_load_api_key())
        _model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=SYSTEM_PROMPT,
        )
    return _model


# ──────────────────────────────────────────────
# RAG: EMBEDDINGS + CHROMA VECTOR STORE (singleton)
# ──────────────────────────────────────────────
_vectorstore = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        os.environ["GOOGLE_API_KEY"] = _load_api_key()
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
        _vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
    return _vectorstore


_rag_logger = RagLogger()


# ──────────────────────────────────────────────
# WEB-SEARCH FALLBACK (Gemini Google-Search grounding)
# ──────────────────────────────────────────────
_web_client = None


def _get_web_client():
    """Lazy singleton for the new-SDK Client used for Google-Search grounding.

    Uses the google-genai package (not the deprecated google.generativeai used by
    the rest of the file) because the `google_search` grounding tool is only
    available there. Import is local so the module-level dependency stays isolated.
    """
    global _web_client
    if _web_client is None:
        from google import genai as genai_new
        _web_client = genai_new.Client(api_key=_load_api_key())
    return _web_client


def web_search_fallback(query: str) -> list[dict]:
    """Query Gemini with Google-Search grounding and return KB-shaped chunks.

    Used when the Chroma knowledge base has no usable match. Returns dicts of
    {content, source, page: 0, box: 3, distance: None}, one per grounded
    source URL. Never raises — on any failure it logs to stderr and returns [].
    """
    try:
        from google.genai import types
        client = _get_web_client()
        prompt = (
            f"Find current architecture best practices for: {query}. "
            "Summarize the top findings with source URLs."
        )
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            ),
        )
        return _extract_grounding_entries(response)[:5]
    except Exception as e:
        print(f"web_search_fallback: failed for '{query}': {e}", file=sys.stderr)
        return []


def _resolve_url(url, timeout=3):
    """Follow redirects to the final URL using only the stdlib.

    Tries HEAD first and retries with GET on HTTP 405. Returns the original URL
    on any error or timeout. Never raises.
    """
    import urllib.error
    import urllib.request

    def _final(method):
        req = urllib.request.Request(url, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.geturl()

    try:
        return _final("HEAD")
    except urllib.error.HTTPError as e:
        if e.code == 405:
            try:
                return _final("GET")
            except Exception:
                return url
        return url
    except Exception:
        return url


def _extract_grounding_entries(response) -> list[dict]:
    """Turn a grounded generate_content response into KB-shaped chunk dicts.

    Prefers per-segment grounding_supports (each snippet -> its source URL);
    falls back to the full summary text when supports are unavailable.
    """
    try:
        text = response.text or ""
    except (AttributeError, ValueError):
        text = ""

    gm = None
    try:
        gm = response.candidates[0].grounding_metadata
    except (AttributeError, IndexError, TypeError):
        pass

    def _fmt_source(title, uri):
        if title and uri:
            return f"{title} ({uri})"
        return uri or "web_search"

    _resolved_cache = {}
    _resolve_calls = 0

    def _resolved(uri):
        nonlocal _resolve_calls
        if not uri:
            return uri
        if uri in _resolved_cache:
            return _resolved_cache[uri]
        if "vertexaisearch" not in uri:
            return uri
        if _resolve_calls >= 5:
            return uri
        _resolve_calls += 1
        final = _resolve_url(uri)
        _resolved_cache[uri] = final
        return final

    urls = []
    sources = []
    if gm is not None:
        for chunk in getattr(gm, "grounding_chunks", []) or []:
            web = getattr(chunk, "web", None)
            uri = getattr(web, "uri", None) if web else None
            title = getattr(web, "title", None) if web else None
            if uri:
                r_uri = _resolved(uri)
                urls.append(r_uri)
                sources.append(_fmt_source(title, r_uri))
    urls = list(dict.fromkeys(urls))
    sources = list(dict.fromkeys(sources))

    entries: list[dict] = []
    if gm is not None:
        chunks_meta = list(getattr(gm, "grounding_chunks", []) or [])
        for support in getattr(gm, "grounding_supports", []) or []:
            seg = getattr(support, "segment", None)
            seg_text = getattr(seg, "text", None) if seg else None
            if not seg_text:
                continue
            indices = getattr(support, "grounding_chunk_indices", []) or []
            source = None
            for idx in indices:
                try:
                    web = chunks_meta[idx].web
                    uri = getattr(web, "uri", None)
                    title = getattr(web, "title", None)
                except (IndexError, AttributeError):
                    continue
                if uri:
                    source = _fmt_source(title, _resolved(uri))
                    break
            if source is None and sources:
                source = sources[0]
            entries.append({
                "content": seg_text,
                "source": source or "web_search",
                "page": 0,
                "box": 3,
                "distance": None,
            })

    if not entries and text:
        source = sources[0] if sources else "web_search"
        body = text
        if len(urls) > 1:
            body += "\n\nSources:\n" + "\n".join(f"- {u}" for u in urls)
        entries.append({
            "content": body,
            "source": source,
            "page": 0,
            "box": 3,
            "distance": None,
        })
    return entries


def _apply_web_fallback(query: str) -> list[dict]:
    """Run the web-search fallback when the knowledge base returned nothing.

    Honours WEB_FALLBACK_ENABLED and logs the result as status "web_fallback".
    Returns [] when disabled or when the fallback yields nothing, so callers
    behave exactly as before the fallback existed.
    """
    if not WEB_FALLBACK_ENABLED:
        return []
    start = time.perf_counter()
    web_chunks = web_search_fallback(query)
    duration_ms = (time.perf_counter() - start) * 1000
    _rag_logger.log_query(
        query=query,
        duration_ms=duration_ms,
        chunks_returned=len(web_chunks),
        best_distance=None,
        source_files=sorted({c["source"] for c in web_chunks}),
        status="web_fallback",
    )
    return web_chunks


def retrieve_chunks(query: str, k: int = 3) -> tuple[list[dict], str]:
    """Retrieve up to k matching chunks from the Chroma vector store.

    Returns a (chunks, origin) tuple. chunks is a list of dicts
    [{content, source, page, box, distance}], sorted by distance ascending
    (lower = better). Chunks whose distance exceeds DISTANCE_THRESHOLD are
    discarded. Every call is logged via RagLogger. origin is "kb" when the
    knowledge base produced hits, "web" when the live web-search fallback was
    used, or "none" when nothing was found.
    """
    vs = get_vectorstore()

    start = time.perf_counter()
    results = vs.similarity_search_with_score(query, k=k)
    duration_ms = (time.perf_counter() - start) * 1000

    if not results:
        _rag_logger.log_query(
            query=query,
            duration_ms=duration_ms,
            chunks_returned=0,
            best_distance=None,
            source_files=[],
            status="no_results",
        )
        chunks = _apply_web_fallback(query)
        return chunks, ("web" if chunks else "none")

    best_distance = results[0][1]

    if DISTANCE_THRESHOLD is not None:
        filtered = [(doc, dist) for doc, dist in results if dist <= DISTANCE_THRESHOLD]
    else:
        filtered = list(results)

    if not filtered:
        source_files = sorted({doc.metadata.get("source", "unknown") for doc, _ in results})
        _rag_logger.log_query(
            query=query,
            duration_ms=duration_ms,
            chunks_returned=0,
            best_distance=best_distance,
            source_files=source_files,
            status="below_threshold",
        )
        chunks = _apply_web_fallback(query)
        return chunks, ("web" if chunks else "none")

    chunks = []
    source_files = []
    for doc, dist in filtered:
        source = doc.metadata.get("source", "unknown")
        chunks.append({
            "content": doc.page_content,
            "source": source,
            "page": doc.metadata.get("page", 0),
            "box": doc.metadata.get("box", 1),
            "distance": dist,
        })
        source_files.append(source)

    _rag_logger.log_query(
        query=query,
        duration_ms=duration_ms,
        chunks_returned=len(chunks),
        best_distance=chunks[0]["distance"],
        source_files=sorted(set(source_files)),
        status="success",
    )
    return chunks, "kb"


def search_patterns(query: str, vectorstore=None) -> str:
    """Searches the Chroma vector store for matching architecture documents
    and returns the contents of the top-3 hits as a contiguous string.

    Delegates retrieval (similarity_search_with_score, DISTANCE_THRESHOLD
    filtering, RagLogger logging) to retrieve_chunks(). The *vectorstore*
    parameter is deprecated, ignored — retained for signature stability with
    existing callers.
    """
    chunks, origin = retrieve_chunks(query, k=3)
    if not chunks:
        return "No relevant information found in the knowledge base."

    prefix = ""
    if origin == "web":
        prefix = (
            "[Note: No sufficient match in the internal knowledge base — "
            "the following results come from a live web search.]\n\n"
        )

    combined = []
    for i, c in enumerate(chunks, 1):
        combined.append(
            f"[Hit {i} | Source: {c['source']} | Page: {c['page']}]\n{c['content']}"
        )
    return prefix + "\n\n---\n\n".join(combined)


def build_enriched_input(user_input: str, vectorstore=None) -> str:
    """Appends the RAG context to architecture-related questions.

    The *vectorstore* parameter is deprecated, ignored — retained for signature
    stability with existing callers.
    """
    if any(kw in user_input.lower() for kw in ARCHITECTURE_KEYWORDS):
        context = search_patterns(user_input, vectorstore)
        return f"{user_input}\n\n[Pattern search for '{user_input}']:\n{context}"
    return user_input


# ──────────────────────────────────────────────
# SQLITE MEMORY
# ──────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def save_message(conn, role: str, content: str):
    conn.execute(
        "INSERT INTO conversations (role, content, timestamp) VALUES (?, ?, ?)",
        (role, content, datetime.now().isoformat()),
    )
    conn.commit()


def load_history(conn, limit: int = 10) -> list:
    rows = conn.execute(
        "SELECT role, content FROM conversations ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    rows = list(reversed(rows))
    return [{"role": r, "parts": [c]} for r, c in rows]


def get_all_messages(conn) -> list:
    return conn.execute(
        "SELECT id, role, content, timestamp FROM conversations ORDER BY id ASC"
    ).fetchall()


# ──────────────────────────────────────────────
# FULL MESSAGE FLOW (for terminal demo)
# ──────────────────────────────────────────────
def send_message(conn, model, user_input: str):
    """Saves the question, enriches it via RAG, queries Gemini, saves the
    response, and returns (answer, input_tokens, output_tokens)."""
    save_message(conn, "user", user_input)
    chat = model.start_chat(history=load_history(conn))
    enriched = build_enriched_input(user_input)
    response = chat.send_message(enriched)
    answer = response.text
    save_message(conn, "model", answer)

    meta = response.usage_metadata
    in_tok = getattr(meta, "prompt_token_count", 0) or 0
    out_tok = getattr(meta, "candidates_token_count", 0) or 0
    return answer, in_tok, out_tok
