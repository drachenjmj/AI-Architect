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

THREE PAUSES, ONE STAGE
------------------------
`pending_decision` says which one:

  CLARIFICATION     — the clarifier asked required questions. Write the
                       answers into `state.clarification_answers` and re-run;
                       the entry route sends the state back to the clarifier.
  OPTIONAL_CONTEXT  — the required round is done; a dedicated, clearly
                       skippable step offers relevant-but-non-blocking
                       fields the required round did not ask about. Resolved
                       ENTIRELY HERE, exactly like CONTEXT_LOCK below (no
                       graph entry) — see `render_optional_context` /
                       `clarifier.resolve_optional_context`. Advances to
                       CONTEXT_LOCK, never back into the graph.
  CONTEXT_LOCK      — a Context Record is frozen and waiting to be approved.
                       This pause is resolved ENTIRELY HERE, before the graph
                       is entered again, and the orchestrator refuses to be
                       entered while it is still pending. Three resolutions:
                       approve, edit, or ask.

This UI is the caller that sets `require_context_approval=True`, because it is
the only one with a human in front of it. The CLI, the eval harness and the
tests leave it off and keep running unattended (see pipeline/state.py).

Nothing here writes to the `ContextRecord`. Approving, editing, the optional
answers, and asking all go through `pipeline.agents.clarifier`, which owns
Maheen's schema — the panels in ui_sections.py only report what the human
asked for.

THE FINISHED-RUN SCREEN (what the DONE branch shows)
----------------------------------------------------
This file stays the event/DRAW spine; the sections themselves live in
ui_sections.py as small DRAW-only functions, and the finished-run screen is
now a multi-view WORKSPACE (see ui_workspace.py): a persistent left-side
navigation in the sidebar picks one view — Overview, Architecture, Review,
Context, Repository, Knowledge, the read-only History browser, and the
Architecture Chat — and only the selected view renders. The screen had
grown past what one page should carry, and the History and Chat
workstreams are exactly what the split was built for.

The views are a table of contents, not a rewrite: each one is a thin
orchestration over the same section renderers as before, every action the
single-page screen had lives in exactly one view, and the intent handling
below (question, feedback, sign-off, chat) is the same code the long page
used.

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

from webapp import architecture_chat
from webapp import field_discussion
from pipeline.agents import clarifier as clarifier_gate
from pipeline.llm import LLMError
from pipeline import orchestrator
from pipeline.persistence import CheckpointError, list_runs, load_state, save_state
from pipeline.refine_gate import begin_user_round
from pipeline.repo_analysis import is_repo_url
from pipeline.state import ArchitectState, PendingDecision, Stage, new_run
from pipeline import reporting, sign_off, user_feedback
from webapp.ui_sections import (
    BCG_DARK as _BCG_DARK,
    BCG_GREEN as _BCG_GREEN,
    GREY as _GREY,
    RED as _RED,
    clear_pending_feedback,
    get_pending_feedback,
    live_step_caption,
    render_ask_ai,
    render_context_approval,
    render_optional_context,
    render_user_rounds,
    set_report_generation_feedback,
)
from webapp.ui_workspace import (
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
    all three pauses happen inside the "Clarify" step and none is a step of
    its own. The live header has room to be more specific, and "Clarifying…"
    over a finished record that is waiting on a signature would be a small lie.
    """
    if state.pending_decision is PendingDecision.CONTEXT_LOCK:
        return "Awaiting your approval"
    if state.pending_decision is PendingDecision.OPTIONAL_CONTEXT:
        return "Optional context"
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


def _handle_field_ask(state: ArchitectState | None, payload: dict) -> None:
    """REACT half of one per-field "Ask AI" turn — shared by every screen
    that renders `ui_sections.render_ask_ai` (required questions, the
    optional round, the Context Record approval screen, and the pre-run
    intake form). `payload` is exactly the kwargs `render_ask_ai`'s callers
    assembled for `field_discussion.ask`.

    `state` may be `None` (the pre-run intake form): `field_discussion.ask`
    passes it straight through to `clarifier.discuss_field`, which is
    documented as safe with `state=None` — see that function's docstring.

    Never enters the graph, never touches the Context Record — this is the
    same "read-only side channel" contract `ask_advisor` gives its callers,
    just session-only instead of billed to the run trace (see
    field_discussion.py's module docstring for why).
    """
    try:
        field_discussion.ask(state, **payload)
    except LLMError as exc:
        st.error(f"Could not reach the assistant: {exc}")
    st.rerun()


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

    THE FINAL REPORT IS GENERATED RIGHT AFTER, NEVER BEFORE. Acceptance is
    the trigger `pipeline.reporting` is documented to require — see
    `build_report`'s guard on `Stage.ACCEPTED` — so the export call sits here
    and nowhere upstream of `accept_design`. A failed export never undoes or
    hides a valid human sign-off: the acceptance above is already
    checkpointed by the time `generate_reports` runs, and any failure is only
    recorded for the report panel to show, via `set_report_generation_feedback`.
    """
    sign_off.accept_design(state, note=note)
    # The pipeline checkpoints on every graph transition, and this is not one.
    # Saving here is what makes the sign-off survive the tab being closed — an
    # acceptance that only existed in `st.session_state` would be the one fact
    # in the run that a refresh could erase.
    save_state(state)
    result = reporting.generate_reports(state)
    set_report_generation_feedback(state.run_id, result)
    st.rerun()


def _regenerate_report(state: ArchitectState) -> None:
    """REACT half of the report (re)generate/retry button.

    Idempotent: `generate_reports` always fully re-renders both files from
    the CURRENT accepted state and atomically replaces this run's own report
    artifacts, so retrying after a partial failure is always safe and never
    leaves a stale mix of an old file and a new one.
    """
    result = reporting.generate_reports(state)
    set_report_generation_feedback(state.run_id, result)
    st.rerun()


# The first option is a sentinel, not a run: it is what keeps "start a new run"
# a ZERO-CLICK path. The selector defaults to it, so a user who ignores the
# picker entirely lands in the intake form exactly as before.
_NEW_RUN_LABEL = "➕ Start a new run"


def _run_label(summary) -> str:
    """Show project name and run date in the run picker."""
    from pipeline.persistence import load_state

    state = load_state(summary.run_id)
    record = state.context_record

    project = (
        (record.project_name if record is not None else "")
        or (state.blueprint.project_name if state.blueprint is not None else "")
        or summary.raw_prompt_excerpt
        or summary.run_id
    ).strip()

    when = summary.updated_at.replace("T", " ").removesuffix("+00:00").strip()

    return f"{project} · {when}"


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

    BUG FIX: `on_change="rerun"` is REQUIRED for `key=` to actually populate
    `st.session_state[key]` at all — with the default `on_change="ignore"`,
    Streamlit never tracks the expanded state, `st.session_state.get(key)`
    is permanently `None`, and this control silently rendered nothing the
    moment a real user opened it (the sidebar "Switch run" bug). Without
    `on_change="rerun"` the check below is not a lazy-render optimization,
    it is a guaranteed early return.
    """
    st.caption("**Current run**")
    st.markdown(_current_run_line(state), unsafe_allow_html=True)

    with st.expander("Switch run", key=_SWITCH_OPEN_KEY, on_change="rerun"):
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

# Set by the New run button (after its session clear) so an EXPLICIT new run
# outranks the `--demo` auto-seeder below: without it, clicking New run under
# `-- --demo` cleared the session and the seeder immediately re-loaded the
# same demo state — the button appeared to do nothing. A plain page refresh
# starts a fresh session, so demo seeding still works exactly as before.
_NEW_RUN_KEY = "new_run_requested"

_DEMO = _demo_variant()
if (
    _DEMO is not None
    and st.session_state["state"] is None
    and not st.session_state.get(_NEW_RUN_KEY)
):
    from webapp.ui_demo import build_demo_state

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
        # Set AFTER the clear, so it survives it: this is what stops the
        # `--demo` seeder from re-loading a finished run over the fresh
        # start the human just asked for (see the note by _NEW_RUN_KEY).
        st.session_state[_NEW_RUN_KEY] = True
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
    # Structured intake, plain (non-form) widgets: nothing still fires until
    # the human clicks "Start" below (a half-filled screen never starts a
    # run) — but NOT an `st.form`, because the description field's own
    # "Ask AI" popover (a plain button — see ui_sections.render_ask_ai) needs
    # the CURRENTLY VISIBLE draft, and a form would only hand it the
    # last-submitted value. The URL gets its own hard-validated box — the
    # repo is first-class input, not something to bury in the prompt text.
    #
    # "Ask AI" is offered on the system description only — the field that
    # actually needs reasoning about — not the repo URL, which has nothing
    # to discuss. No `ArchitectState` exists yet here, so the discussion is
    # scoped by `field_discussion.PRE_RUN_SCOPE` rather than a run_id, and
    # its "project context" IS the draft description itself.
    _INTAKE_DESC_KEY = "intake_system_description"
    _INTAKE_REPO_KEY = "intake_repo_url"
    _pre_run_scope = field_discussion.pre_run_scope()
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown("System description")
    with col2:
        intent = render_ask_ai(
            scope_id=_pre_run_scope,
            field_key="start.system_description",
            field_label="System description",
        )
    if intent is not None:
        action, value = intent
        if action == "apply":
            st.session_state[_INTAKE_DESC_KEY] = value
            st.rerun()
        else:  # "ask"
            draft_desc = st.session_state.get(_INTAKE_DESC_KEY, "")
            draft_repo = st.session_state.get(_INTAKE_REPO_KEY, "").strip()
            _handle_field_ask(
                None,
                {
                    "scope_id": _pre_run_scope,
                    "field_key": "start.system_description",
                    "raw_prompt": draft_desc,
                    "repo_representation": None,  # never inspected pre-run
                    "known_context": {"repo_url": draft_repo} if draft_repo else {},
                    "field_label": "System description",
                    "field_purpose": "",
                    "current_draft_answer": draft_desc,
                    "message": value,
                },
            )

    prompt = st.text_area(
        "System description",
        height=160,
        key=_INTAKE_DESC_KEY,
        label_visibility="collapsed",
        placeholder=(
            "e.g. Our monolithic online shop crashes on peak sale days. "
            "On AWS, medium budget, must stay GDPR-compliant, ~50k peak users."
        ),
    )
    repo_url = st.text_input(
        "GitHub repository URL (optional)",
        key=_INTAKE_REPO_KEY,
        placeholder="https://github.com/org/repo",
    )
    st.caption("Leave empty for a greenfield project (no existing code).")
    submitted = st.button("Start", key="intake_start")
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

    # 3a. Paused on required questions? → show them as a form (Event 2 lives here).
    if (
        state.stage is Stage.AWAITING_HUMAN
        and state.pending_decision is PendingDecision.CLARIFICATION
    ):
        with st.chat_message("assistant"):
            st.write("Before I can design safely, I need a few answers:")
            # NOT `st.form`. A form batches its widgets until one submit —
            # which broke two things at once: the per-question "Ask AI"
            # popover (a plain button, so it has to live outside any
            # wrapping form — see ui_sections.render_ask_ai) would fire on a
            # rerun that still carried the FORM's last-submitted answer
            # rather than whatever the human had just typed and not yet
            # submitted, and the form's own reserved slot painted as one
            # block — every answer input, then the submit button — AFTER
            # every question's label+Ask-AI pair, instead of each question
            # being one interleaved unit. Plain, session-state-backed
            # widgets fix both: they sync on the same blur that opens the
            # popover, and they paint in call order. "Submit answers"
            # (below) stays the only thing that writes into
            # `state.clarification_answers` and re-enters the pipeline.
            answer_keys = [f"answer_{i}" for i in range(len(state.clarifying_questions))]
            for i, q in enumerate(state.clarifying_questions):
                # Question TEXT is the stable field identity here — the
                # loop index alone would wrongly reuse one field's history
                # for a DIFFERENT question if a later round asks something
                # else at the same position (see field_discussion.py).
                field_key = f"clarification::{q}"
                col1, col2 = st.columns([6, 1])
                with col1:
                    st.markdown(q)
                with col2:
                    intent = render_ask_ai(
                        scope_id=state.run_id, field_key=field_key, field_label=q
                    )
                if intent is not None:
                    action, value = intent
                    if action == "apply":
                        st.session_state[f"answer_{i}"] = value
                        st.rerun()
                    else:  # "ask" — other currently-visible draft answers,
                        # not only the prior rounds already on the record.
                        known = dict(state.clarification_answers)
                        known.update(
                            {
                                state.clarifying_questions[j]: st.session_state.get(key, "")
                                for j, key in enumerate(answer_keys)
                                if j != i and st.session_state.get(key, "").strip()
                            }
                        )
                        _handle_field_ask(
                            state,
                            {
                                "scope_id": state.run_id,
                                "field_key": field_key,
                                "raw_prompt": state.initial_request.raw_prompt,
                                "repo_representation": state.repo_representation,
                                "known_context": known,
                                "field_label": q,
                                "field_purpose": "",
                                "current_draft_answer": st.session_state.get(
                                    f"answer_{i}", ""
                                ),
                                "message": value,
                            },
                        )
                st.text_input(q, key=f"answer_{i}", label_visibility="collapsed")
            if st.button("Submit answers", key="clarify_submit"):
                # Merge (not replace): keep earlier rounds so the
                # Clarifier re-judges with the FULL Q&A history.
                state.clarification_answers.update(
                    {
                        q: st.session_state.get(f"answer_{i}", "").strip()
                        for i, q in enumerate(state.clarifying_questions)
                        if st.session_state.get(f"answer_{i}", "").strip()
                    }
                )
                # No `begin_user_round` here. Answering the clarifier is the
                # pipeline working as designed, not the human asking it to
                # redo work — and charging for it drained the refinement
                # budget before design started. See state.user_rounds.
                _run(state)  # entry router sends us back to clarifier

    # 3a-bis. Paused on the OPTIONAL round? → a dedicated, clearly-skippable
    #     step between the required questions and the review screen. Resolved
    #     ENTIRELY here, like the context lock: no graph entry either way, just
    #     a redraw at CONTEXT_LOCK next (see clarifier.resolve_optional_context).
    elif (
        state.stage is Stage.AWAITING_HUMAN
        and state.pending_decision is PendingDecision.OPTIONAL_CONTEXT
    ):
        with st.chat_message("assistant"):
            intent = render_optional_context(state)

            if intent is not None:
                action, payload = intent
                if action == "field_ask":
                    _handle_field_ask(state, payload)
                else:
                    clarifier_gate.resolve_optional_context(
                        state, payload if action == "continue" else None
                    )
                    # No `_run` here — no pipeline work happens on this
                    # transition, only a move to the next caller-resolved
                    # pause. Saved explicitly because nothing else will
                    # before the next graph entry (see `_sign_off`'s note on
                    # the same situation).
                    save_state(state)
                    st.rerun()

    # 3b. Paused at the context lock? → the veto panel. Resolved ENTIRELY here:
    #     the graph is not entered until the human has accepted, or until an
    #     edit has opened a gap that the clarifier has to re-judge.
    elif (
        state.stage is Stage.AWAITING_HUMAN
        and state.pending_decision is PendingDecision.CONTEXT_LOCK
    ):
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

                elif action == "field_ask":
                    _handle_field_ask(state, payload)

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
        elif intents.regenerate_report:
            _regenerate_report(state)

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
