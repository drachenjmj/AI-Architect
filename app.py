import os
import sqlite3
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st

# ──────────────────────────────────────────────
# PAGE CONFIG (muss erster Streamlit-Befehl sein)
# ──────────────────────────────────────────────
st.set_page_config(page_title="AI Architect", page_icon="🏗️", layout="wide")

# ──────────────────────────────────────────────
# 1) SETUP
# ──────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY nicht in .env gefunden!")
    st.stop()

genai.configure(api_key=api_key)
MODEL_NAME = "gemini-2.5-flash"

# ──────────────────────────────────────────────
# SYSTEM PROMPT (vor get_model definiert, da es dort referenziert wird)
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """Du bist ein AI Solution Architect. Deine Aufgabe ist es, vage Business-Anforderungen zu verstehen, Rueckfragen zu stellen um Unklaerheiten zu beseitigen, und dann eine passende Architektur vorzuschlagen.

Bevor du eine Architektur empfiehlst, frage IMMER zuerst nach:
1. Cloud-Provider-Präferenz (AWS, Azure, GCP, On-Premise)
2. Budget-Rahmen
3. Skalierungsanforderungen (Nutzerzahlen, Traffic)
4. Compliance-Anforderungen (DSGVO, HIPAA, etc.)
5. Bestehende Systeme und Integrationen

Nutze das search_patterns Tool wenn du Architektur-Patterns nachschlagen musst.
Antworte immer auf Deutsch. Sei praegise und strukturiert.

Verfuegbare Architektur-Patterns in deiner Wissensbasis:
- microservices
- monolith
- event-driven

Du kannst die Funktion search_patterns(query) nutzen, um Details zu diesen Patterns abzurufen.
Die Ergebnisse werden dir als JSON-String zur Verfuegung gestellt.
"""

@st.cache_resource
def get_model():
    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_PROMPT,
    )

def get_db():
    conn = sqlite3.connect("architect.db", check_same_thread=False)
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

# ──────────────────────────────────────────────
# 5) TOOLS – Wissensbasis & Suchfunktion
# ──────────────────────────────────────────────
ARCHITECTURE_PATTERNS = {
    "microservices": {
        "beschreibung": "Die Anwendung wird in kleine, unabhaengige Services aufgeteilt, die jeweils eine Geschaeftsfaehigkeit kapseln und ueber APIs kommunizieren.",
        "vorteile": [
            "Unabhaengige Deployierung einzelner Services",
            "Technologie-Freiheit pro Service",
            "Horizontale Skalierung einzelner Komponenten",
            "Fehlertoleranz durch Isolation",
        ],
        "nachteile": [
            "Hohe Komplexitaet bei verteilten Systemen",
            "Netzwerk-Latenzen zwischen Services",
            "Erschwertes Debugging und Tracing",
            "Benötigt DevOps-Reife (CI/CD, Container-Orchestrierung)",
        ],
        "use_case": "Grosse Plattformen mit vielen Teams, die unabhaengig arbeiten muessen (z.B. E-Commerce, Streaming-Dienste, SaaS-Produkte).",
    },
    "monolith": {
        "beschreibung": "Die gesamte Anwendung wird als ein einzigen deployierbaren Einheit gebaut. Alle Module teilen sich denselben Prozess und dieselbe Datenbank.",
        "vorteile": [
            "Einfache Entwicklung und Deployment",
            "Keine Netzwerk-Kommunikation zwischen Modulen",
            "Einfaches Debugging und Testing",
            "Geringe Infrastruktur-Kosten",
        ],
        "nachteile": [
            "Schwierig zu skalieren einzelner Komponenten",
            "Enge Kopplung fuehrt zu Regressionen",
            "Technologie-Wechsel nur fuer die gesamte App moeglich",
            "Deployment-Zyklen werden mit wachsendem Team langsamer",
        ],
        "use_case": "Startups, MVPs, interne Tools, kleine Teams mit begrenzten Ressourcen und klar abgegrenztem Domain-Bereich.",
    },
    "event-driven": {
        "beschreibung": "Komponenten kommunizieren asynchron ueber Events. Produzenten senden Events ohne Kenntnis der Konsumenten. Ein Message-Broker (z.B. Kafka, RabbitMQ) vermittelt.",
        "vorteile": [
            "Starke Entkopplung zwischen Services",
            "Asynchrone Verarbeitung fuer bessere Performance",
            "Einfache Erweiterbarkeit durch neue Konsumenten",
            "Event-Sourcing fuer vollstaendige Audit-Trails",
        ],
        "nachteile": [
            "Eventual Consistency statt sofortiger Konsistenz",
            "Komplexe Fehlerbehandlung und Dead-Letter-Queues",
            "Schwierig zu testen und zu debuggen",
            "Benötigt zuverlaessige Message-Broker-Infrastruktur",
        ],
        "use_case": "Echtzeit-Datenverarbeitung, IoT-Plattformen, Notification-Systeme, und Anwendungen mit asynchronen Workflows (z.B. Bestellprozesse mit Lagerbestand-Updates).",
    },
}


def search_patterns(query: str) -> str:
    """Durchsucht die Wissensbasis nach passenden Architektur-Patterns."""
    query_lower = query.lower()
    results = []
    for name, pattern in ARCHITECTURE_PATTERNS.items():
        keywords = [name] + pattern["vorteile"] + pattern["nachteile"]
        if any(kw.lower() in query_lower for kw in keywords) or name in query_lower:
            results.append({"name": name, **pattern})
    if not results:
        results = [{"name": k, **v} for k, v in ARCHITECTURE_PATTERNS.items()]
    return json.dumps(results, indent=2, ensure_ascii=False)


# Keyword-Heuristik fuer automatische Pattern-Suche
ARCHITECTURE_KEYWORDS = [
    "architektur", "pattern", "microservice", "monolith",
    "event", "skalier", "design", "struktur", "system",
]


def build_enriched_input(user_input: str) -> str:
    """Haengt bei Architektur-relevanten Fragen Pattern-Kontext an."""
    if any(kw in user_input.lower() for kw in ARCHITECTURE_KEYWORDS):
        patterns = search_patterns(user_input)
        return f"{user_input}\n\n[Pattern-Suche fuer '{user_input}']:\n{patterns}"
    return user_input


# ──────────────────────────────────────────────
# DB-HILFSFUNKTIONEN
# ──────────────────────────────────────────────
def save_message(conn, role: str, content: str):
    timestamp = datetime.now().isoformat()
    conn.execute(
        "INSERT INTO conversations (role, content, timestamp) VALUES (?, ?, ?)",
        (role, content, timestamp),
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
# SESSION STATE INITIALISIERUNG
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_tokens" not in st.session_state:
    st.session_state.input_tokens = 0
if "output_tokens" not in st.session_state:
    st.session_state.output_tokens = 0

# ──────────────────────────────────────────────
# SEITENLAYOUT
# ──────────────────────────────────────────────
st.title("🏗️ AI Architect Agent")
st.caption("BCG Platinion x UoC – Theme 4")

conn = get_db()
model = get_model()

# ──────────────────────────────────────────────
# 3) SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    # a) Model Info
    st.header("Model Info")
    st.info(f"**Modell:** `{MODEL_NAME}`")

    st.divider()

    # b) Database
    st.header("Database")
    if st.button("Show conversations", use_container_width=True):
        rows = conn.execute(
            "SELECT timestamp, role, content FROM conversations ORDER BY id DESC LIMIT 20"
        ).fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=["Timestamp", "Role", "Content"])
            df["Role"] = df["Role"].map({"user": "👤 User", "model": "🤖 Agent"})
            df["Content"] = df["Content"].str[:120] + "..."
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.warning("Keine Nachrichten vorhanden.")

    st.divider()

    # c) Token Usage
    st.header("Token Usage")
    col1, col2 = st.columns(2)
    col1.metric("Input", f"{st.session_state.input_tokens:,}")
    col2.metric("Output", f"{st.session_state.output_tokens:,}")

    st.divider()

    # Footer/Actions
    st.header("Actions")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        conn.execute("DELETE FROM conversations")
        conn.commit()
        st.session_state.messages = []
        st.session_state.input_tokens = 0
        st.session_state.output_tokens = 0
        st.rerun()

    all_rows = get_all_messages(conn)
    if all_rows:
        md_lines = [f"# AI Architect – Chat Export ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n"]
        for _, role, content, ts in all_rows:
            label = "**User**" if role == "user" else "**Architect**"
            md_lines.append(f"### {label} _({ts[:19]})_\n\n{content}\n")
        md_content = "\n---\n".join(md_lines)
        st.download_button(
            "📄 Export chat",
            data=md_content,
            file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

conn.close()

# ──────────────────────────────────────────────
# 6) CHAT INTERFACE
# ──────────────────────────────────────────────

# Bestehende DB-Nachrichten beim Start laden (damit Refresh den Verlauf zeigt)
conn = get_db()
if not st.session_state.messages:
    db_rows = conn.execute(
        "SELECT role, content FROM conversations ORDER BY id ASC"
    ).fetchall()
    for role, content in db_rows:
        st.session_state.messages.append({"role": role, "content": content})
conn.close()

# Chat-Verlauf anzeigen (Streamlit kennt nur "user" und "assistant")
def to_streamlit_role(role: str) -> str:
    return "assistant" if role == "model" else role

for msg in st.session_state.messages:
    with st.chat_message(to_streamlit_role(msg["role"])):
        st.markdown(msg["content"])

# Chat-Input
if prompt := st.chat_input("Beschreibe dein Projekt oder stelle eine Frage..."):
    conn = get_db()

    # User-Nachricht speichern & anzeigen
    save_message(conn, "user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent-Antwort generieren
    with st.chat_message("assistant"):
        with st.spinner("Architect denkt nach..."):
            try:
                history = load_history(conn)
                chat = model.start_chat(history=history)
                enriched = build_enriched_input(prompt)
                response = chat.send_message(enriched)
                answer = response.text

                # Token-Usage kumulieren
                meta = response.usage_metadata
                if meta:
                    st.session_state.input_tokens += getattr(meta, "prompt_token_count", 0) or 0
                    st.session_state.output_tokens += getattr(meta, "candidates_token_count", 0) or 0

                # Antwort speichern & anzeigen
                save_message(conn, "model", answer)
                st.session_state.messages.append({"role": "model", "content": answer})
                st.markdown(answer)

            except Exception as e:
                st.error(f"Fehler: {e}")

    conn.close()
