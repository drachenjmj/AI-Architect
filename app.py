import pandas as pd
from datetime import datetime
import streamlit as st

from architect import (
    MODEL_NAME,
    get_model,
    get_db,
    get_all_messages,
    save_message,
    load_history,
    build_enriched_input,
)

# ──────────────────────────────────────────────
# PAGE CONFIG (must be the first Streamlit command)
# ──────────────────────────────────────────────
st.set_page_config(page_title="AI Architect", page_icon="🏗️", layout="wide")

# ──────────────────────────────────────────────
# 1) SETUP – logic comes from architect.py
# ──────────────────────────────────────────────
model = get_model()

# ──────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_tokens" not in st.session_state:
    st.session_state.input_tokens = 0
if "output_tokens" not in st.session_state:
    st.session_state.output_tokens = 0

# ──────────────────────────────────────────────
# PAGE LAYOUT
# ──────────────────────────────────────────────
st.title("🏗️ AI Architect Agent")
st.caption("BCG Platinion x UoC – Theme 4")

# ──────────────────────────────────────────────
# 2) SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    # a) Model Info
    st.header("Model Info")
    st.info(f"**Model:** `{MODEL_NAME}`")

    st.divider()

    # b) Database
    st.header("Database")
    conn = get_db()
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
            st.warning("No messages available.")
    conn.close()

    st.divider()

    # c) Token Usage
    st.header("Token Usage")
    col1, col2 = st.columns(2)
    col1.metric("Input", f"{st.session_state.input_tokens:,}")
    col2.metric("Output", f"{st.session_state.output_tokens:,}")

    st.divider()

    # d) Actions
    st.header("Actions")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        conn = get_db()
        conn.execute("DELETE FROM conversations")
        conn.commit()
        conn.close()
        st.session_state.messages = []
        st.session_state.input_tokens = 0
        st.session_state.output_tokens = 0
        st.rerun()

    conn = get_db()
    all_rows = get_all_messages(conn)
    conn.close()
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

# ──────────────────────────────────────────────
# 3) CHAT INTERFACE
# ──────────────────────────────────────────────
conn = get_db()

# Load existing DB messages at startup (so a refresh shows the history)
if not st.session_state.messages:
    db_rows = conn.execute(
        "SELECT role, content FROM conversations ORDER BY id ASC"
    ).fetchall()
    for role, content in db_rows:
        st.session_state.messages.append({"role": role, "content": content})


# Display chat history (Streamlit only knows "user" and "assistant")
def to_streamlit_role(role: str) -> str:
    return "assistant" if role == "model" else role


for msg in st.session_state.messages:
    with st.chat_message(to_streamlit_role(msg["role"])):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Describe your project or ask a question..."):
    # Save & display user message
    save_message(conn, "user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate agent response
    with st.chat_message("assistant"):
        with st.spinner("Architect is thinking..."):
            try:
                history = load_history(conn)
                chat = model.start_chat(history=history)
                enriched = build_enriched_input(prompt)
                response = chat.send_message(enriched)
                answer = response.text

                # Accumulate token usage
                meta = response.usage_metadata
                if meta:
                    st.session_state.input_tokens += getattr(meta, "prompt_token_count", 0) or 0
                    st.session_state.output_tokens += getattr(meta, "candidates_token_count", 0) or 0

                # Save & display response
                save_message(conn, "model", answer)
                st.session_state.messages.append({"role": "model", "content": answer})
                st.markdown(answer)

            except Exception as e:
                st.error(f"Error: {e}")

conn.close()
