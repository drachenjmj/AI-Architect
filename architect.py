"""architect.py – Single Source of Truth for the AI Solution Architect.

This module contains all of the domain logic: the Gemini model, the RAG/Chroma
knowledge base, and the SQLite memory. Both app.py (Streamlit UI) and agent.ipynb
(terminal demo) import exclusively from here – changes only need to be made in this
one place.
"""
import os
import sqlite3
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
DISTANCE_THRESHOLD = None

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


def search_patterns(query: str, vectorstore=None) -> str:
    """Searches the Chroma vector store for matching architecture documents
    and returns the contents of the top-3 hits as a contiguous string.

    Uses similarity_search_with_score so the Chroma DISTANCE (lower = better)
    is available for rule-based quality control: chunks whose distance exceeds
    DISTANCE_THRESHOLD are discarded. Every call is logged to rag_log.jsonl.
    """
    vs = vectorstore if vectorstore is not None else get_vectorstore()

    start = time.perf_counter()
    results = vs.similarity_search_with_score(query, k=3)
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
        return "No relevant information found in the knowledge base."

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
        return "No relevant information found in the knowledge base."

    combined = []
    source_files = []
    for i, (doc, dist) in enumerate(filtered, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        source_files.append(source)
        combined.append(
            f"[Hit {i} | Source: {source} | Page: {page}]\n{doc.page_content}"
        )

    _rag_logger.log_query(
        query=query,
        duration_ms=duration_ms,
        chunks_returned=len(filtered),
        best_distance=filtered[0][1],
        source_files=sorted(set(source_files)),
        status="success",
    )
    return "\n\n---\n\n".join(combined)


def build_enriched_input(user_input: str, vectorstore=None) -> str:
    """Appends the RAG context to architecture-related questions."""
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
