"""ui.py — Streamlit front-end for the AI-Architect pipeline (Kati).

HOW TO RUN
----------
From the AI-Architect/ folder (so the `pipeline` package imports resolve):

    streamlit run ui.py

    streamlit run ui.py -- --demo          # DEV ONLY: load a finished run
    streamlit run ui.py -- --demo capped   # DEV ONLY: one stopped on budget

The `--demo` flag loads a fully-populated state straight into the session so
the finished-run screen can be screenshotted without spending API quota (see
ui_demo.py). It never runs the pipeline and makes no LLM calls.

HOW THIS WORKS (the one mental model you need)
----------------------------------------------
Streamlit re-runs this WHOLE script top-to-bottom on every user interaction.
Normal variables are wiped each time; only `st.session_state` (a dict) survives.
So the app is just:

    REACT  — an event fires (prompt submitted / answers submitted):
             update the ArchitectState, call run_pipeline(), store the result
             back in st.session_state, then st.rerun().
    DRAW   — every run, render whatever the CURRENT state says: the chat so
             far, a question form if paused, the artifacts if done.

There is NO separate message list: the chat is DERIVED from the state object
every time. One source of truth — same principle as the pipeline itself, and
it means the UI would survive a state-on-disk reload for free.

The pipeline pauses by RETURNING with stage == AWAITING_INPUT (see
orchestrator._route). This script is the "caller" from clarifier-design: it
holds the state between calls, writes answers into
`state.clarification_answers`, and calls run_pipeline(state) again to resume.

THE FINISHED-RUN SCREEN (what the DONE branch shows)
----------------------------------------------------
This file stays the event/DRAW spine; the sections themselves live in
ui_sections.py as small DRAW-only functions, each taking the state and reading
nothing else. The DONE branch below is therefore a table of contents, in the
order a viewer should meet them.

The screen has one job: make what the run actually DID visible. Four of the
five agentic behaviours happen inside the pipeline and used to leave no trace
on screen, so each section is labelled with the behaviour it evidences —

    Run trace            -> multi-step reasoning
    Knowledge retrieved  -> retrieval
    Repository analysis  -> tool use
    Review report        -> iteration + quality gate
    the Q&A replay above -> clarification

— followed by the artifacts in full. Nothing on this screen is computed here:
every value already exists on the state object, and the run status, the token
totals and the cost are read from the same fields the pipeline wrote.

Two honesty rules are load-bearing. The status line never claims completion
for a run whose review failed or that stopped on the refine budget; and every
cost is labelled as the list-price equivalent on a free-tier key, never as
money spent (`ui_sections` prints "unknown" rather than a confident $0.0000
when a step used a model we have no verified price for).

RESUMING ACROSS SESSIONS
------------------------
`st.session_state` dies with the browser tab, but the orchestrator checkpoints
every transition to disk (pipeline/persistence.py). So the "Resume run" picker
below is the promise at the top of this docstring being collected: because the
chat is DERIVED from the state object rather than stored separately, loading a
checkpoint back into `st.session_state` replays the whole conversation for
free — including a pending clarifying-question form. Nothing in the DRAW
section needed to change to support it.
"""
from __future__ import annotations

import sys

import streamlit as st

from pipeline.orchestrator import run_pipeline_streaming
from pipeline.persistence import CheckpointError, list_runs, load_state
from pipeline.repo_analysis import is_repo_url
from pipeline.state import ArchitectState, Stage, new_run
from ui_sections import (
    BCG_DARK as _BCG_DARK,
    BCG_GREEN as _BCG_GREEN,
    GREY as _GREY,
    RED as _RED,
    live_step_caption,
    render_adrs,
    render_blueprint,
    render_components,
    render_context_record,
    render_features,
    render_knowledge,
    render_repo_analysis,
    render_review_report,
    render_run_status,
    render_run_trace,
    render_status_strip,
)

# ── page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Architect", page_icon="🏛️", layout="wide")

# The stages shown in the sidebar checklist, in pipeline order.
# AWAITING_INPUT is not listed: it is a pause WITHIN clarifying, not a step.
_CHECKLIST: list[tuple[Stage, str]] = [
    (Stage.INGESTING, "Ingest repo"),
    (Stage.CLARIFYING, "Clarify"),
    (Stage.RESEARCHING, "Research"),
    (Stage.DESIGNING, "Design"),
    (Stage.REVIEWING, "Review"),
    (Stage.DONE, "Done"),
]
# Rank of each stage so we can mark earlier ones as completed.
_ORDER = {stage: i for i, (stage, _) in enumerate(_CHECKLIST)}
# A pause happens inside the clarifying step.
_ORDER[Stage.AWAITING_INPUT] = _ORDER[Stage.CLARIFYING]
_ORDER[Stage.CREATED] = -1
_ORDER[Stage.REFINING] = _ORDER[Stage.REVIEWING]
_ORDER[Stage.FAILED] = -1

# Human-readable label for each stage — drives the live status header in `_run`.
_STAGE_LABELS: dict[Stage, str] = dict(_CHECKLIST)
_STAGE_LABELS[Stage.CREATED] = "Starting"
_STAGE_LABELS[Stage.AWAITING_INPUT] = "Clarifying"
_STAGE_LABELS[Stage.REFINING] = "Refining"
_STAGE_LABELS[Stage.FAILED] = "Failed"


def _run(state: ArchitectState) -> None:
    """REACT half: drive the pipeline, show live backend status, redraw.

    Streams the state after each node so the viewer can follow what the backend
    is doing (stage header + the latest step note). The last streamed state is
    the terminal one — identical to what run_pipeline would return — so we just
    keep it. No pipeline logic is touched; we only consume the stream.

    Each step also prints the RUNNING token/cost total. `input_tokens` and
    `output_tokens` are reducer fields, so every snapshot already carries the
    total so far and nothing has to be accumulated here. That turns the cost
    guardrail into something you watch happen — the number climbing through the
    refine loop — rather than a figure that only appears at the end.
    """
    latest = state
    with st.status("Pipeline running…", expanded=True) as status:
        for snapshot in run_pipeline_streaming(state):
            latest = snapshot
            status.update(label=f"{_STAGE_LABELS.get(snapshot.stage, 'Working')}…")
            if snapshot.history:
                step = snapshot.history[-1]
                st.write(f"**{step.agent}** — {step.note or step.stage_out.value}")
                st.caption(live_step_caption(snapshot))
        st.session_state["state"] = latest
    st.rerun()  # restart the script so the DRAW section shows the new state


# The first option is a sentinel, not a run: it is what keeps "start a new run"
# a ZERO-CLICK path. The selector defaults to it, so a user who ignores the
# picker entirely lands in the intake form exactly as before.
_NEW_RUN_LABEL = "➕ Start a new run"


def _run_label(summary) -> str:
    """One picker line: when it was last touched, where it stopped, what it was about."""
    when = summary.updated_at.replace("T", " ").removesuffix("+00:00").strip()
    where = summary.stage.replace("_", " ").capitalize()
    return f"{when} · {where} · {summary.raw_prompt_excerpt}"


def _resume_picker() -> None:
    """DRAW the "Resume run" selector above the intake form.

    Renders NOTHING when there are no checkpoints on disk, so a first-time user
    never sees it. Selecting a run loads that checkpoint into `st.session_state`
    and reruns; from there the normal DRAW section takes over and replays the
    run — question form included — because the whole UI is derived from state.

    Persistence problems degrade to a working app: an unreadable runs directory
    just means no picker, and a corrupt checkpoint reports itself instead of
    blocking the new-run path.
    """
    try:
        runs = list_runs()
    except Exception as exc:  # noqa: BLE001 — the picker is never worth a crash
        st.warning(f"Could not read saved runs: {exc}")
        return
    if not runs:
        return

    options = {_NEW_RUN_LABEL: None}
    for summary in runs:
        options[_run_label(summary)] = summary.run_id

    choice = st.selectbox(
        "Resume a previous run",
        list(options),
        index=0,  # ← the zero-click default
        help="Every run is checkpointed after each step, so you can close the "
             "tab mid-run and pick up exactly where you left off.",
    )
    run_id = options[choice]
    if run_id is None:
        return  # sentinel selected — fall through to the intake form

    try:
        st.session_state["state"] = load_state(run_id)
    except CheckpointError as exc:
        st.error(f"Could not resume that run: {exc}")
        return
    st.rerun()  # redraw with the loaded state


# ── DEV ONLY: `-- --demo [variant]` loads a finished run, no pipeline ────
# Fills the session with a fully-populated state so the finished-run screen can
# be screenshotted for the slides and the video without spending API quota (our
# KB currently returns nothing, so a live run cannot fill every section). The
# import is lazy: a normal `streamlit run ui.py` never touches ui_demo.
def _demo_variant() -> str | None:
    """The variant named after `--demo` on the command line, or None if absent."""
    argv = sys.argv[1:]
    if "--demo" not in argv:
        return None
    position = argv.index("--demo") + 1
    if position < len(argv) and not argv[position].startswith("-"):
        return argv[position]
    return "pass"


# ── session init ──────────────────────────────────────────────────────────
st.session_state.setdefault("state", None)

_DEMO = _demo_variant()
if _DEMO is not None and st.session_state["state"] is None:
    from ui_demo import build_demo_state

    st.session_state["state"] = build_demo_state(_DEMO)

state: ArchitectState | None = st.session_state["state"]

# ── BCG brand palette ────────────────────────────────────────────────────
# Defined once in ui_sections.py and imported above under these names, so the
# spine and the section renderers cannot drift apart. (Widget theme itself
# lives in .streamlit/config.toml.)

# ── SIDEBAR: stage checklist (Event-agnostic — pure DRAW) ─────────────────
with st.sidebar:
    st.markdown(
        f"<h1 style='color:{_BCG_DARK}; margin-bottom:0.2rem'>AI Architect</h1>",
        unsafe_allow_html=True,
    )
    st.caption("BCG Platinion × AIIM — Architecture Assistant")
    current_rank = _ORDER.get(state.stage, -1) if state else -1
    for stage, label in _CHECKLIST:
        rank = _ORDER[stage]
        if state and state.stage is Stage.FAILED:
            mark, color = "✕", _RED
        elif rank < current_rank or (state and state.stage is Stage.DONE):
            mark, color = "✓", _BCG_DARK
        elif rank == current_rank:
            mark = "⏸" if state.stage is Stage.AWAITING_INPUT else "●"
            color = _BCG_GREEN
        else:
            mark, color = "○", _GREY
        st.markdown(
            f"<span style='color:{color}; font-weight:600'>{mark}</span>&nbsp; "
            f"<span style='color:{color}'>{label}</span>",
            unsafe_allow_html=True,
        )
    st.divider()
    if state is not None and st.button("🔄 New run"):
        st.session_state.clear()  # Event 3: forget everything → fresh start
        st.rerun()

# ── DRAW: replay the conversation from the state object ──────────────────
if state is None:
    _resume_picker()  # above the prompt box; silent when nothing is saved yet
    st.markdown("#### Describe the system you need architected")
    st.caption(
        "Include what you know: domain, scale, cloud, compliance, budget. "
        "Point us at the existing codebase — the Clarifier grounds its questions "
        "in it and only asks about what the repo cannot answer."
    )
    # Structured intake: st.form batches both fields so nothing fires until the
    # user clicks Start (a half-filled form never starts a run). The URL gets
    # its own hard-validated box — the repo is first-class input, not something
    # to bury in the prompt text.
    with st.form("intake"):
        prompt = st.text_area(
            "System description",
            height=160,
            placeholder=(
                "e.g. Our monolithic online shop crashes on peak sale days. "
                "On AWS, medium budget, must stay GDPR-compliant, ~50k peak users."
            ),
        )
        repo_url = st.text_input(
            "GitHub repository URL (optional)",
            placeholder="https://github.com/org/repo",
        )
        st.caption("Leave empty for a greenfield project (no existing code).")
        submitted = st.form_submit_button("Start")
    if submitted:
        clean_url = repo_url.strip()
        if not prompt.strip():
            st.warning("Please describe the system you need architected.")
        elif clean_url and not is_repo_url(clean_url):
            # HARD validation: a non-empty URL must be a well-formed repo URL.
            st.error(
                "That does not look like a repository URL. Use the form "
                "`https://github.com/org/repo` (GitHub, GitLab or Bitbucket), "
                "or leave the field empty for a greenfield project."
            )
        else:
            _run(new_run(prompt.strip(), clean_url))
else:
    # 1. The original request — the ground truth of the run.
    with st.chat_message("user"):
        st.write(state.initial_request.raw_prompt)

    # 2. Q&A rounds already answered (drawn from state, not stored separately).
    for question, answer in state.clarification_answers.items():
        with st.chat_message("assistant"):
            st.write(question)
        with st.chat_message("user"):
            st.write(answer)

    # 3. Paused? → show open questions as a form (Event 2 lives here).
    if state.stage is Stage.AWAITING_INPUT:
        with st.chat_message("assistant"):
            st.write("Before I can design safely, I need a few answers:")
            # st.form batches the inputs: nothing happens until Submit,
            # so a half-filled form never triggers a pipeline run.
            with st.form("clarify_form"):
                answers: dict[str, str] = {}
                for i, q in enumerate(state.clarifying_questions):
                    answers[q] = st.text_input(q, key=f"answer_{i}")
                if st.form_submit_button("Submit answers"):
                    # Merge (not replace): keep earlier rounds so the
                    # Clarifier re-judges with the FULL Q&A history.
                    state.clarification_answers.update(
                        {q: a.strip() for q, a in answers.items() if a.strip()}
                    )
                    _run(state)  # resume: entry router sends us back to clarifier

    # 4. Finished? → show what the run DID, then the design in full.
    #    Every section is a DRAW-only function in ui_sections.py taking this
    #    same state object; the order below is the order a viewer meets them.
    #    Each renderer is null-safe on its own: a section whose source is
    #    missing either omits itself or says why it is empty, so a thin run
    #    (greenfield, no KB results, no review) renders honestly rather than
    #    leaving empty headings behind.
    elif state.stage is Stage.DONE:
        with st.chat_message("assistant"):
            render_run_status(state)    # honest verdict — never a false success
            render_status_strip(state)  # the run's shape at a glance

            # What the system DID — one expander per agentic behaviour.
            render_run_trace(state)      # multi-step reasoning
            render_knowledge(state)      # retrieval
            render_repo_analysis(state)  # tool use
            render_review_report(state)  # iteration + quality gate

            # What the system PRODUCED — the artifacts, in full.
            render_context_record(state)
            render_features(state)
            render_blueprint(state)
            render_adrs(state)
            render_components(state)

    # 5. Failed? → say so plainly, with the recorded errors.
    elif state.stage is Stage.FAILED:
        with st.chat_message("assistant"):
            st.error("Run failed.")
            for err in state.errors:
                st.code(err)

    # 6. Parked mid-flight → offer to carry on.
    #    Only reachable by RESUMING a checkpoint whose process died between
    #    stages (crash, closed tab, killed server). A live run never lands here:
    #    `_run` keeps control until the pipeline reaches a terminal stage or
    #    pauses at AWAITING_INPUT.
    else:
        with st.chat_message("assistant"):
            label = _STAGE_LABELS.get(state.stage, state.stage.value)
            st.info(f"This run was interrupted at **{label}**.")
            if st.button("▶ Continue run"):
                _run(state)
