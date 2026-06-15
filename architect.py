"""architect.py – Single Source of Truth fuer den AI Solution Architect.

Dieses Modul enthaelt die gesamte Fachlogik: Gemini-Modell, RAG/Chroma-Wissensbasis
und SQLite-Gedaechtnis. Sowohl app.py (Streamlit-UI) als auch agent.ipynb
(Terminal-Demo) importieren ausschliesslich von hier – Aenderungen gibt es nur noch
an dieser einen Stelle.
"""
import os
import sqlite3
from datetime import datetime

from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# ──────────────────────────────────────────────
# KONFIGURATION
# ──────────────────────────────────────────────
MODEL_NAME = "gemini-2.5-flash"
CHROMA_DIR = "./chroma_db"
DB_PATH = "architect.db"

ARCHITECTURE_KEYWORDS = [
    "architektur", "pattern", "microservice", "monolith",
    "event", "skalier", "design", "struktur", "system",
]

SYSTEM_PROMPT = """Du bist ein weltklasse AI Solution Architect, entwickelt für die hochkarätige IT-Architektur-Beratung im Rahmen des BCG Platinion Use Cases. Deine Aufgabe ist es, vage Business-Anforderungen in präzise, skalierbare und produktionsreife Systemarchitekturen zu transformieren.

Du folgst einem strikten Zwei-Phasen-Ansatz:

PHASE 1: CONTEXT CAPTURE & ANFORDERUNGSKLÄRUNG
Bevor du irgendeine Technologie empfiehlst oder ein Diagramm/Blueprint zeichnest, musst du zuerst alle Rahmenbedingungen lückenlos absichern. Wenn der Nutzer eine vage Idee äußert, frage gezielt nach den folgenden 5 Kernkriterien, falls sie noch nicht genannt wurden:
1. Cloud-Provider-Präferenz (AWS, Azure, GCP, On-Premise oder Hybrid)
2. Budget-Rahmen (Low, Medium, High oder konkrete Beträge)
3. Skalierungsanforderungen (Erwartete Nutzerzahlen, Concurrent Requests, Datenvolumen, Peak-Events)
4. Compliance- & Sicherheitsanforderungen (DSGVO, PCI-DSS, Verschlüsselung, Datenhaltung in EU)
5. Bestehende Systemlandschaft (Brownfield-Integration vs. Greenfield-Aufbau)

Sobald diese Parameter klar oder durch plausible Annahmen definiert sind, erstelle als ersten Output einen "Context Record" (einen eingefrorenen Snapshot aller Bedingungen und offenen Fragen).

PHASE 2: ARCHITEKTUR-DESIGN & ABGABE-ARTEFAKTE
Erst nach der Kontextklärung generierst du die finale Architektur. Nutze dafür die Ergebnisse aus der RAG-Meldung (search_patterns). Dein Output MUSS zwingend folgende Struktur in sauberem Markdown aufweisen:

### 1. Context Record
- Zusammenfassung aller validierten Constraints, Nutzergruppen und Integrationsanforderungen.

### 2. Architecture Blueprint
Du musst zwei klar getrennte Sichten bereitstellen:
- **Stakeholder View:** Eine verständliche Beschreibung in klarer Business-Sprache. Was macht das System? Wie fließen die Daten geschäftlich?
- **Technical View:** Eine tiefgehende technische Spezifikation für Software-Ingenieure. Welche Services werden genutzt? Wie sieht das Datenmodell, die Modulverantwortlichkeit und die Integrationsstruktur aus?

### 3. Component Description
- Detaillierte Auflistung und Begründung jeder einzelnen ausgewählten Komponente oder Pipeline-Stufe.

### 4. Architecture Decision Records (ADRs)
Erstelle für jede signifikante Design-Entscheidung (z.B. Microservices statt Monolith, eine spezifische NoSQL-Datenbank, etc.) ein kurzes ADR im folgenden Format:
- **Titel:** ADR-[Nummer]: [Entscheidung]
- **Kontext:** Welche technische Herausforderung lag vor?
- **Betrachtete Optionen:** Welche Alternativen gab es?
- **Gewählte Option:** Was wird implementiert und warum?
- **Trade-Offs:** Welche Nachteile oder Kompromisse (z.B. Latenz vs. Konsistenz, Kosten vs. Autonomie) werden bewusst akzeptiert?

STILRICHTLINIEN:
- Antworte immer auf Deutsch. Sei präzise, strukturiert und nutze professionelle Berater-Formulierungen.
- Nutze Markdown-Tabellen und Listen für Vergleiche und Komponenten-Übersichten.
- Untermaure deine Empfehlungen mit harten Fakten aus den abrufenen RAG-Dokumenten und gib die entsprechenden Quellen/Seiten an, wenn du sie aus dem Kontext liest.
"""


# ──────────────────────────────────────────────
# API-KEY (intern)
# ──────────────────────────────────────────────
def _load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY nicht in .env gefunden!")
    return api_key


# ──────────────────────────────────────────────
# GEMINI-MODELL (Singleton – einmal pro Prozess)
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
# RAG: EMBEDDINGS + CHROMA-VEKTORDATENBANK (Singleton)
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


def search_patterns(query: str, vectorstore=None) -> str:
    """Durchsucht die Chroma-Vektordatenbank nach passenden Architektur-Dokumenten
    und liefert die Inhalte der Top-3-Treffer als zusammenhaengenden String."""
    vs = vectorstore if vectorstore is not None else get_vectorstore()
    results = vs.similarity_search(query, k=3)
    if not results:
        return "Keine relevanten Informationen in der Wissensbasis gefunden."

    combined = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unbekannt")
        page = doc.metadata.get("page", "?")
        combined.append(
            f"[Treffer {i} | Quelle: {source} | Seite: {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(combined)


def build_enriched_input(user_input: str, vectorstore=None) -> str:
    """Haengt bei Architektur-relevanten Fragen den RAG-Kontext an."""
    if any(kw in user_input.lower() for kw in ARCHITECTURE_KEYWORDS):
        context = search_patterns(user_input, vectorstore)
        return f"{user_input}\n\n[Pattern-Suche fuer '{user_input}']:\n{context}"
    return user_input


# ──────────────────────────────────────────────
# SQLITE-GEDAECHTNIS
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
# VOLLSTAENDIGER NACHRICHTEN-FLUSS (fuer Terminal-Demo)
# ──────────────────────────────────────────────
def send_message(conn, model, user_input: str):
    """Speichert die Frage, reichert sie per RAG an, fragt Gemini an, speichert die
    Antwort und gibt (answer, input_tokens, output_tokens) zurueck."""
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
