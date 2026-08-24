"""field_discussion.py — context-aware "Ask AI" field discussions (Kush
integration, Kati-owned UI surface).

WHAT THIS IS
------------
At every clarification/ground-truth decision point — the required
clarification questions, the optional round, the Context Record approval
screen, and the pre-run intake form's system description — the human can
open a small, CONTEXT-AWARE discussion about the ONE field they are
currently deciding, before committing an answer. This is the REACT half:
`ui_sections.render_ask_ai` is the DRAW half (see its docstring for why the
split exists and how the two fit together), and `ui.py` is the caller that
wires them, exactly the same three-way split `architecture_chat.py` /
`ui_workspace.py` / `ui.py` already use for the Architecture Chat.

NOT A SECOND ADVISOR CHANNEL
-----------------------------
`pipeline.agents.clarifier.ask_advisor` already exists and is reused for the
LLM call shape (`clarifier.discuss_field` sits right beside it in the same
module and shares its cheap default model) — but `ask_advisor` is scoped to
a WHOLE frozen Context Record or a WHOLE finished design and persists each
turn onto `state.advisory_turns`/`state.history`, because it can only ever
run once one of those objects exists. A field discussion is scoped to ONE
field, can run before a Context Record — or even a run — exists at all
(the pre-run intake form), and is SESSION-ONLY by design: multiplying
`ask_advisor`'s per-turn ledger write across every field on a form would
turn a lightweight scratch pad into run-history noise, and would require a
persistence/schema change this feature does not need (see "STATE AND
PERSISTENCE" below).

NOT ROUTED THROUGH THE ARCHITECTURE LITERATURE RAG
----------------------------------------------------
This is for clarifying user intent and project constraints before design
starts; RAG grounding remains part of architecture decision-making later,
in `pipeline.agents.researcher`/`architect`. `clarifier.discuss_field` never
calls `retrieve_chunks`. Repository facts already extracted by the repo
tooling ARE included when available (via `repo_representation`) — that is
tool output already on the state, not literature retrieval.

STATE AND PERSISTENCE
----------------------
Discussion history lives ONLY in `st.session_state`, keyed by (scope,
field) — nothing here is ever written to `ArchitectState`, a checkpoint, or
`.cache/runs/`. Known limitation, stated plainly: a field discussion does
NOT survive a resumed run's checkpoint reload in a NEW browser session (only
the run's own persisted artifacts do); it DOES survive an ordinary rerun,
and switching to a DIFFERENT run and back within the SAME session, because
history is keyed by `run_id` exactly like `architecture_chat.messages_for`.
The pre-run intake screen (before any run/run_id exists) uses the fixed
`PRE_RUN_SCOPE` sentinel instead — there is only ever one pre-run draft in a
given browser session, and `st.session_state.clear()` on "New run" (see
ui.py) already gives every genuinely new draft a clean slate, the same
mechanism every other pre-run widget already relies on.

NO CROSS-RUN / CROSS-FIELD LEAKAGE
------------------------------------
Every entry is double-keyed: scope (run_id, or `PRE_RUN_SCOPE`) THEN field
key (a stable identity such as "context.business_goal" or
"clarification::<question text>" — see ui_sections.render_ask_ai and ui.py
for how each screen derives one). Two different fields, or the same field
key under two different scopes, can never see each other's turns.
"""
from __future__ import annotations

import streamlit as st

from pipeline.agents import clarifier
from pipeline.state import ArchitectState, RepoRepresentation

# The fixed scope for the pre-run intake form, before any ArchitectState (and
# therefore any run_id) exists. See the module docstring's STATE AND
# PERSISTENCE section for why a single fixed sentinel is sufficient.
PRE_RUN_SCOPE = "__pre_run__"

_STORE_KEY = "field_discussions"


def history_for(scope_id: str, field_key: str) -> list[dict[str, str]]:
    """The turns so far for one field, within one scope. Read/append target
    for both halves (`ui_sections.render_ask_ai` reads it to draw the
    transcript; `ask` below appends to it).

    Creates the (empty) entry on first read rather than requiring a prior
    write, so DRAW code can call this freely for a field nobody has
    discussed yet and just get `[]` back — never a `KeyError`.
    """
    store = st.session_state.setdefault(_STORE_KEY, {})
    by_field = store.setdefault(scope_id, {})
    return by_field.setdefault(field_key, [])


def ask(
    state: ArchitectState | None,
    *,
    scope_id: str,
    field_key: str,
    raw_prompt: str,
    repo_representation: RepoRepresentation | None,
    known_context: dict[str, object],
    field_label: str,
    field_purpose: str,
    current_draft_answer: str,
    message: str,
) -> None:
    """The REACT half of one "Ask AI" turn: call the model, append the turn
    to session history. Nothing else moves — no Context Record write, no
    `state.history`/`advisory_turns` entry (see the module docstring).

    Raises `pipeline.llm.LLMError` on failure — propagated from
    `clarifier.discuss_field` uncaught, by design: the caller (ui.py) reports
    it and leaves the field exactly as it was, the same contract
    `clarifier.ask_advisor` gives its own callers.
    """
    history = history_for(scope_id, field_key)
    response = clarifier.discuss_field(
        state,
        raw_prompt=raw_prompt,
        repo_representation=repo_representation,
        known_context=known_context,
        field_label=field_label,
        field_purpose=field_purpose,
        current_draft_answer=current_draft_answer,
        history=[(turn["question"], turn["reply"]) for turn in history],
        user_message=message,
    )
    history.append(
        {
            "question": message,
            "reply": response.reply,
            "suggested_answer": response.suggested_answer,
        }
    )
