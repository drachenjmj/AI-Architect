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

The pipeline pauses by RETURNING with stage == AWAITING_HUMAN (see
orchestrator._route). This script is the "caller" from clarifier-design: it
holds the state between calls, resolves whatever `state.pending_decision` says
is owed, and calls run_pipeline(state) again to resume.

TWO PAUSES, ONE STAGE
---------------------
`pending_decision` says which one:

  CLARIFICATION — the clarifier asked questions. Write the answers into
                  `state.clarification_answers` and re-run; the entry route
                  sends the state back to the clarifier.
  CONTEXT_LOCK  — a Context Record is frozen and waiting to be approved. This
                  pause is resolved ENTIRELY HERE, before the graph is entered
                  again, and the orchestrator refuses to be entered while it is
                  still pending. Three resolutions: approve, edit, or ask.

This UI is the caller that sets `require_context_approval=True`, because it is
the only one with a human in front of it. The CLI, the eval harness and the
tests leave it off and keep running unattended (see pipeline/state.py).

Nothing here writes to the `ContextRecord`. Approving, editing and asking all go
through `pipeline.agents.clarifier`, which owns Maheen's schema — the panel in
ui_sections.py only reports what the human asked for.

THE FINISHED-RUN SCREEN (what the DONE branch shows)
----------------------------------------------------
This file stays the event/DRAW spine; the sections themselves live in
ui_sections.py as small DRAW-only functions, and the finished-run screen is
now a multi-view WORKSPACE (see ui_workspace.py): a persistent left-side
navigation in the sidebar picks one view — Overview, Context, Knowledge,
Repository, Design, Review, and the History/Chat placeholders — and only the
selected view renders. The screen had grown past what one page should carry,
and the upcoming History and Chat workstreams would only have added to it.

The views are a table of contents, not a rewrite: each one is a thin
orchestration over the same section renderers as before, every action the
single-page screen had lives in exactly one view, and the intent handling
below (question, feedback, sign-off) is the same code the long page used.

The screen still has one job: make what the run actually DID visible. Four of
the five agentic behaviours happen inside the pipeline and used to leave no
trace on screen, so the sections stay labelled with the behaviour they
evidence —

    Run trace (Overview)  -> multi-step reasoning
    Knowledge view        -> retrieval
    Repository view       -> tool use
    Review view           -> iteration + quality gate
    the Q&A replay above  -> clarification

— followed by the artifacts in full, in their own views. Nothing on this
screen is computed here: every value already exists on the state object, and
the run status, the token totals and the cost are read from the same fields
the pipeline wrote.

Two honesty rules are load-bearing. The status line never claims completion
for a run whose review failed or that stopped on the refine budget; and every
cost is labelled as the list-price equivalent on a free-tier key, never as
money spent (`ui_sections` prints "unknown" rather than a confident $0.0000
when a step used a model we have no verified price for).

FEEDBACK AT DONE — THE THIRD PLACE A HUMAN CAN CHANGE THE RUN
--------------------------------------------------------------
The finished workspace carries TWO boxes, each in the view of the artifact it
acts on: a requirements box in the Context view, and a design box in the
Design view. Which box the text is typed into is what routes the run — there
is no classifier, so the graph stays 100% deterministically routed even with
a human steering it.

BATCHING SURVIVES THE SPLIT. The boxes do not submit directly; each stages
its text into a session-state bundle (`ui_sections.stage_pending_feedback`),
the bundle survives navigation, and ONE submit action in the sidebar panel
(`render_pending_feedback_panel`) sends everything as a single
`user_feedback.submit_feedback` call — one refinement round, however many
views contributed, exactly as when both boxes shared one page. The bundle is
UI-only: unsent drafts live in `st.session_state`, never in the checkpointed
run, and a page refresh discards them (the same as any other unsent widget
value). A refused submission (round budget spent) keeps the bundle; a
successful one clears it before the pipeline re-runs.

Everything the bundle does to the state happens in `pipeline.user_feedback`;
this file only reports a refused round. Same division as the context gate.

CLOSING A RUN OUT — ASK, DIRECT, ACCEPT
----------------------------------------
In the workspace, in ascending cost, and the order is the design:

    ask     — one flash-lite call, in the Design view. Changes nothing,
              spends no round, works at ACCEPTED too. It is FIRST because
              most first reactions to a finished design are questions, and
              with only a directive box on screen the question gets typed as
              an instruction and costs a full refine round to answer.
    direct  — the design box, also in the Design view; it stages into the
              bundle, and the sidebar's submit action spends the round.
    accept  — the sign-off, in the Overview view. `DONE` says the pipeline
              stopped; `ACCEPTED` says a person took the design, and only
              the second is a fact anyone downstream can act on. It is
              terminal, it is never reached by an agent, and it records what
              was accepted DESPITE — every open finding, as a waiver, plus
              anything the person typed and never re-ran (see
              pipeline/sign_off.py).

The finished-run branch renders at DONE **and** at ACCEPTED. After sign-off
the workspace redraws with the boxes disabled and the acceptance on it: the
artifacts do not change, because the whole value of the record is that what
is on screen is what was signed.

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

import html
import sys

import streamlit as st

import architecture_chat
from pipeline.agents import clarifier as clarifier_gate
from pipeline.llm import LLMError
from pipeline import orchestrator
from pipeline.persistence import CheckpointError, list_runs, load_state, save_state
from pipeline.refine_gate import begin_user_round
from pipeline.repo_analysis import is_repo_url
from pipeline.state import ArchitectState, PendingDecision, Stage, new_run
from pipeline import sign_off, user_feedback
from ui_sections import (
    BCG_DARK as _BCG_DARK,
    BCG_GREEN as _BCG_GREEN,
    GREY as _GREY,
    RED as _RED,
    clear_pending_feedback,
    get_pending_feedback,
    live_step_caption,
    render_context_approval,
    render_user_rounds,
)
from ui_workspace import (
    render_pending_feedback_panel,
    render_workspace_view,
    select_workspace_view,
)

# ── page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Architect", page_icon="🏛️", layout="wide")

# The pipeline's stages, named once for the live status header in `_run` and
# the sidebar's compact status line. The full stage-by-stage trace lives in
# the workspace's `Run details` view.
_CHECKLIST: list[tuple[Stage, str]] = [
    (Stage.INGESTING, "Ingest repo"),
    (Stage.CLARIFYING, "Clarify"),
    (Stage.RESEARCHING, "Research"),
    (Stage.DESIGNING, "Design"),
    (Stage.REVIEWING, "Review"),
    (Stage.DONE, "Done"),
]

# Human-readable label for each stage — drives the live status header in `_run`.
_STAGE_LABELS: dict[Stage, str] = dict(_CHECKLIST)
_STAGE_LABELS[Stage.CREATED] = "Starting"
_STAGE_LABELS[Stage.AWAITING_HUMAN] = "Clarifying"
_STAGE_LABELS[Stage.REFINING] = "Refining"
_STAGE_LABELS[Stage.ACCEPTED] = "Accepted"
_STAGE_LABELS[Stage.FAILED] = "Failed"

# The two terminal stages that carry a finished design. Named once because four
# places ask the same question — is there a design on screen? — and DONE alone
# was the answer until a run could be closed out.
_FINISHED = (Stage.DONE, Stage.ACCEPTED)


def _stage_label(state: ArchitectState) -> str:
    """The header line for a state — the stage, refined by what is pending.

    `_STAGE_LABELS` is keyed by stage alone because the sidebar checklist is:
    both pauses happen inside the "Clarify" step and neither is a step of its
    own. The live header has room to be more specific, and "Clarifying…" over a
    finished record that is waiting on a signature would be a small lie.
    """
    if state.pending_decision is PendingDecision.CONTEXT_LOCK:
        return "Awaiting your approval"
    return _STAGE_LABELS.get(state.stage, "Working")


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
        # Called through the module on purpose: `from … import` would bind
        # the function object into THIS module's namespace, where a test (or
        # any other caller) patching `pipeline.orchestrator` could no longer
        # reach it — the offline test suite would silently drive the real
        # orchestrator instead of the stub.
        for snapshot in orchestrator.run_pipeline_streaming(state):
            latest = snapshot
            status.update(label=f"{_stage_label(snapshot)}…")
            if snapshot.history:
                step = snapshot.history[-1]
                st.write(f"**{step.agent}** — {step.note or step.stage_out.value}")
                st.caption(live_step_caption(snapshot))
        st.session_state["state"] = latest
    st.rerun()  # restart the script so the DRAW section shows the new state


def _submit_pending_feedback(state: ArchitectState) -> None:
    """REACT half of the batched feedback round: send the whole bundle once.

    ONE `user_feedback.submit_feedback` call with whatever is pending — a
    correction staged in the Context view and a directive staged in the
    Design view leave as a single refinement round, exactly as they did when
    both boxes shared one page. Every write lives in
    `pipeline.user_feedback`; this function only reports a refusal.

    The bundle is cleared ONLY on acceptance. A refusal (round budget spent)
    records nothing on the run and must not eat the staged text either — the
    warning below is the whole outcome, and the person can still discard it
    from the sidebar or carry it into a new run.
    """
    pending = get_pending_feedback()
    requirements = str(pending.get("requirements", "")).strip()
    design = str(pending.get("design", "")).strip()
    if not requirements and not design:
        return  # unreachable: the panel renders only with pending content

    accepted, why = user_feedback.submit_feedback(
        state, requirements=requirements, design=design
    )
    if not accepted:
        st.warning(
            f"Out of refinement rounds ({why}), so this was not submitted. "
            f"Your pending feedback is kept — discard it from the sidebar, "
            f"or start a new run to take it further."
        )
        return
    clear_pending_feedback()
    _run(state)


def _sign_off(state: ArchitectState, note: str) -> None:
    """REACT half of the sign-off: record the decision, then redraw.

    Every write lives in `pipeline.sign_off` — the stage, the timestamp, the
    waiver, the abandonment of anything still pending. The same division the
    context gate and the feedback boxes already use: ui_sections returns intent,
    ui.py performs it, the pipeline owns what it means.

    NO `_run` HERE, and that is the whole difference from `_submit_feedback`.
    Feedback sends the run back into the graph; a sign-off ends it. There is
    nothing to stream, no node to visit and no token to spend — the state is
    checkpointed and the page simply redraws in its accepted form.
    """
    sign_off.accept_design(state, note=note)
    # The pipeline checkpoints on every graph transition, and this is not one.
    # Saving here is what makes the sign-off survive the tab being closed — an
    # acceptance that only existed in `st.session_state` would be the one fact
    # in the run that a refresh could erase.
    save_state(state)
    st.rerun()


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
        help="Every run is checkpointed after each step, so you can close "
             "the tab mid-run and pick up exactly where you left off.",
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


# ── the CURRENT-RUN switcher ───────────────────────────────────────────────
# History (ui_workspace) is the READ-ONLY browser of saved runs. This control
# is its deliberate counterpart: the ONE place in the workspace that makes a
# different saved run CURRENT. It reuses the resume flow exactly — the same
# `list_runs` summaries the picker above builds on and the same `load_state`
# deserialization — so there is no second loader and no new persistence.
_SWITCH_OPEN_KEY = "switch_run_expander"


def _current_run_line(state: ArchitectState) -> str:
    """One identifying line for the active run: project · repo · short id."""

    record = state.context_record
    project = (
        (record.project_name if record is not None else "")
        or (state.blueprint.project_name if state.blueprint is not None else "")
        or "Architecture run"
    ).strip() or "Architecture run"
    url = (
        state.repo_representation.meta.url
        if state.repo_representation is not None
        else state.initial_request.repo_url
    ).strip()
    repo = url.rstrip("/").rsplit("/", 1)[-1] if url else ""
    parts = [project] + ([repo] if repo else [])
    return (
        f"{html.escape(' · '.join(parts))} · "
        f"<code>{html.escape(state.run_id.rsplit('-', 1)[-1])}</code>"
    )


def _render_run_switcher(state: ArchitectState) -> None:
    """DRAW the current-run control in the sidebar.

    Two-step on purpose: opening the expander only READS the list of saved
    runs, and only the explicit Make-current action replaces
    `st.session_state["state"]` — switching a run is never a side effect of
    browsing (History cannot do it either).

    PENDING FEEDBACK GUARDS THE SWITCH. The staged bundle belongs to the run
    it was typed against; making another run current with the bundle alive
    would let one run's feedback land on a different run's next submission.
    The switch is refused with a warning until the bundle is submitted or
    discarded — the smallest safe rule, and the one that can never misapply
    text.

    The runs directory is scanned only while the expander is OPEN (Streamlit
    keeps an expander's open/closed state under its key), so ordinary
    reruns pay nothing for having this control in the sidebar.
    """
    st.caption("**Current run**")
    st.markdown(_current_run_line(state), unsafe_allow_html=True)

    with st.expander("Switch run", key=_SWITCH_OPEN_KEY):
        if not st.session_state.get(_SWITCH_OPEN_KEY):
            return  # closed: no directory scan, no widgets
        try:
            runs = list_runs()
        except Exception as exc:  # noqa: BLE001 — a control is never worth a crash
            st.warning(f"Could not read saved runs: {exc}")
            return
        others = [summary for summary in runs if summary.run_id != state.run_id]
        if not others:
            st.caption("No other saved runs are available on this machine yet.")
            return

        options = {_run_label(summary): summary.run_id for summary in others}
        choice = st.selectbox("Saved runs", list(options), key="switch_run_choice")
        if st.button(
            "Make current run",
            key="switch_run_apply",
            type="primary",
            use_container_width=True,
        ):
            pending = get_pending_feedback()
            if any(str(value or "").strip() for value in pending.values()):
                st.warning(
                    "You have unsent feedback staged for the CURRENT run. "
                    "Submit or discard it from the sidebar first — pending "
                    "feedback must never land on a different run."
                )
                return
            try:
                st.session_state["state"] = load_state(options[choice])
            except CheckpointError as exc:
                st.error(f"Could not make that run current: {exc}")
                return
            st.rerun()  # redraw everything with the newly-current run


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

# ── SIDEBAR: brand, compact run status, workspace navigation ──────────────
# The old six-row stage checklist restated the pipeline on every screen; the
# workspace nav already says where you are. Status is now ONE line, and the
# full stage-by-stage detail lives in the `Run details` view.
def _sidebar_status_line(state: ArchitectState) -> str:
    """The run's state as one compact sidebar line (HTML)."""

    if state.stage is Stage.FAILED:
        return f"<span style='color:{_RED}; font-weight:600'>✕ Run failed</span>"
    if state.stage is Stage.ACCEPTED:
        return (
            f"<span style='color:{_BCG_DARK}; font-weight:600'>✓ Accepted</span>"
        )
    if state.stage in _FINISHED:
        verdict = "Review passed" if (
            state.review and state.review.overall_status == "pass"
        ) else "Review not passed"
        if state.stopped_on_cap:
            verdict += " · stopped on budget"
        refine = (
            f" · {state.refine_iterations} refinement"
            f"{'s' if state.refine_iterations != 1 else ''}"
            if state.refine_iterations
            else ""
        )
        return (
            f"<span style='color:{_BCG_GREEN}; font-weight:600'>✓ Completed</span>"
            f"<span style='color:{_GREY}'> · {verdict}{refine}</span>"
        )
    return (
        f"<span style='color:{_BCG_GREEN}; font-weight:600'>● {_stage_label(state)}</span>"
    )


with st.sidebar:
    st.markdown(
        f"<h1 style='color:{_BCG_DARK}; margin-bottom:0.2rem'>AI Architect</h1>",
        unsafe_allow_html=True,
    )
    st.caption("BCG Platinion × AIIM — Architecture Assistant")
    if state is not None:
        st.markdown(_sidebar_status_line(state), unsafe_allow_html=True)
        # The current-run control sits directly under the status it names:
        # what is active, and the deliberate way to change it. History stays
        # the read-only browser; THIS is what makes a run current.
        _render_run_switcher(state)
    st.divider()
    # Workspace navigation — only for a finished run. A run mid-flight (or a
    # first visit) keeps the focused linear flow: intake, clarifier, context
    # lock. Nothing mandatory ever waits on navigation. The pending-feedback
    # panel sits under it: the bundle outlives navigation BY DESIGN, so its
    # submit action has to be reachable from every view — which is also why
    # it lives in the sidebar rather than in any one view.
    workspace_view: str | None = None
    request_submit_round = False
    if state is not None and state.stage in _FINISHED:
        workspace_view = select_workspace_view()
        request_submit_round = render_pending_feedback_panel(state)
    if state is not None and st.button("🔄 New run"):
        st.session_state.clear()  # Event 3: forget everything → fresh start
        st.rerun()

# A submit request from the sidebar panel. Handled OUTSIDE the `with
# st.sidebar` block on purpose: `_submit_pending_feedback` → `_run` streams
# the pipeline status widget, and inside the block Streamlit would capture it
# into the sidebar instead of the main area.
if request_submit_round and state is not None:
    _submit_pending_feedback(state)

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
            # The ONE caller with a human attached, so the ONE caller that turns
            # the context lock on. Everything headless leaves it off and keeps
            # auto-approving, which is what it did before this existed.
            _run(new_run(prompt.strip(), clean_url, require_context_approval=True))
else:
    # 1 & 2. The original request and the Q&A replay — the LINEAR flow's
    #    conversation. The finished workspace deliberately does not draw
    #    them: it is an artifact view, and the same data lives there in the
    #    Context view's collapsed "Clarification history" (plus the header).
    #    Drawing both would put the transcript back on top of every view.
    if state.stage not in _FINISHED:
        with st.chat_message("user"):
            st.write(state.initial_request.raw_prompt)

        for question, answer in state.clarification_answers.items():
            with st.chat_message("assistant"):
                st.write(question)
            with st.chat_message("user"):
                st.write(answer)

    # 3a. Paused on questions? → show them as a form (Event 2 lives here).
    if (
        state.stage is Stage.AWAITING_HUMAN
        and state.pending_decision is not PendingDecision.CONTEXT_LOCK
    ):
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
                    # No `begin_user_round` here. Answering the clarifier is the
                    # pipeline working as designed, not the human asking it to
                    # redo work — and charging for it drained the refinement
                    # budget before design started. See state.user_rounds.
                    _run(state)  # entry router sends us back to clarifier

    # 3b. Paused at the context lock? → the veto panel. Resolved ENTIRELY here:
    #     the graph is not entered until the human has accepted, or until an
    #     edit has opened a gap that the clarifier has to re-judge.
    elif state.stage is Stage.AWAITING_HUMAN:
        with st.chat_message("assistant"):
            render_user_rounds(state)
            intent = render_context_approval(state)

            if intent is not None:
                action, payload = intent

                if action == "accept":
                    # Also free: approving RELEASES work that was already
                    # waiting. Only an edit that sends the record back for
                    # re-judging asks for work to be redone, and only that is
                    # charged (below).
                    clarifier_gate.accept_context_lock(state)
                    _run(state)  # entry route: CLARIFYING → researcher

                elif action == "ask":
                    # OUTSIDE the graph: no routing decision, no stage change,
                    # `pending_decision` untouched, no artifact written. The
                    # pause stays open across any number of these. It does cost
                    # tokens, so `ask_advisor` bills it to the trace under its
                    # own agent name.
                    try:
                        clarifier_gate.ask_advisor(state, str(payload))
                    except LLMError as exc:
                        # A failed side question must never cost the human their
                        # gate — report it and leave the pause exactly as it was.
                        st.error(f"Could not answer that: {exc}")
                    st.rerun()

                else:  # "edit"
                    try:
                        reasons = clarifier_gate.submit_context_edits(state, payload)
                    except ValueError as exc:
                        st.error(str(exc))
                        reasons = None

                    if reasons == []:
                        # Filling or replacing a value cannot open a gap, so the
                        # record re-froze deterministically with NO model call.
                        st.rerun()
                    elif reasons:
                        allowed, why = begin_user_round(state)
                        if not allowed:
                            st.warning(
                                f"Out of human rounds ({why}), so the clarifier "
                                f"will not re-judge this. Your edit is kept — "
                                f"approve the record as it stands, or start over."
                            )
                        else:
                            st.info("Re-judging, because " + "; ".join(reasons) + ".")
                            clarifier_gate.open_for_rejudge(state)
                            _run(state)

    # 4. Finished? → the multi-view workspace. The sidebar (drawn above, in
    #    this same rerun) holds the navigation; this branch draws ONLY the
    #    selected view — the whole point of the restructure is that selecting
    #    Knowledge no longer also renders the Blueprint underneath. Each view
    #    is a thin orchestration over the same section renderers as before
    #    (see ui_workspace.py), and every action the single-page screen had
    #    still exists in exactly one view, performed by the exact same code
    #    below as before. No chat bubble wraps the workspace: the finished
    #    screen is an artifact view with its own header, not a transcript.
    elif state.stage in _FINISHED:
        intents = render_workspace_view(state, workspace_view)

        if intents.chat_question is not None:
            # The Architecture Chat's REACT half: ONE grounded answer call
            # through the read-only chat layer (which owns the session
            # transcript and every read-only rule), then redraw. It is not
            # the advisory turn and not feedback: no round, no artifact
            # write, no checkpoint — chat lives in session state only.
            architecture_chat.submit_question(state, intents.chat_question)
            st.rerun()
        elif intents.question is not None:
            # OUTSIDE the graph, exactly like the context-gate advisory
            # turn: no stage change, no `pending_decision`, no round
            # consumed. Works at ACCEPTED too — reading is not changing.
            try:
                clarifier_gate.ask_advisor(
                    state, intents.question, subject="design"
                )
                # Checkpointed HERE because nothing else will. At the context
                # gate the next graph entry saves the turn for free; from a
                # terminal stage there may never BE a next graph entry, so
                # the tokens this just spent would be missing from the run on
                # disk and its totals would stop reconciling.
                save_state(state)
            except LLMError as exc:
                # A failed side question must never cost the human their
                # screen — report it and leave everything as it was.
                st.error(f"Could not answer that: {exc}")
            st.rerun()
        elif intents.sign_off is not None:
            _sign_off(state, intents.sign_off[1])

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
    #    pauses at AWAITING_HUMAN.
    else:
        with st.chat_message("assistant"):
            label = _stage_label(state)
            st.info(f"This run was interrupted at **{label}**.")
            if st.button("▶ Continue run"):
                _run(state)
