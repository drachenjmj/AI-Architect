"""ui_workspace.py — the multi-view workspace for a finished run (Kati).

WHY THIS FILE EXISTS
--------------------
The finished-run screen used to render every section on one long page: trace,
knowledge, repository, review, the artifacts, and three different action
panels. That was fine when each section was small; it stopped being fine when
the artifacts became rich, and the next workstreams (History, Architecture
Chat) would only have added more to the same page.

This module owns the workspace SPLIT: a persistent left-side navigation
(drawn in the sidebar, next to the stage checklist) and one view per
destination. Only the selected view renders — selecting Knowledge does not
also render the Blueprint below it — which is the entire point of the
restructure.

THE ONE RULE, INHERITED
-----------------------
Same division as ui_sections.py: everything here is DRAW-only. Each view is a
thin orchestration over the existing section renderers, which are reused
unchanged — no renderer is duplicated, no pipeline code is touched, and the
REACT half stays in ui.py. Views RETURN what the human asked for (a question,
a sign-off) and ui.py performs it through the exact same backend paths as
before:

    Architecture view -> render_design_advisory(state)      (read-only question)
    Overview view     -> render_sign_off(state)             (final acceptance)

Feedback is the exception that proves the rule: the boxes STAGE into the
session-state bundle (`ui_sections.stage_pending_feedback`) and never
submit. `render_pending_feedback_panel` — the sidebar panel under the
navigation — previews the bundle and returns True when its one submit action
fires; ui.py then sends the WHOLE bundle through `user_feedback.submit_feedback`
as a single refinement round. That is how the pre-workspace batching
semantics survive a per-view layout.

WHO EACH VIEW IS FOR
--------------------
Overview is the CLIENT view: the recommended architecture, its key decisions
and components, the verdict and the risks — no trace, tokens, cost or stage
machinery. Architecture is the DELIVERABLE in full. Run details is the
machinery the operators need. The sidebar groups them that way (PROJECT /
EVIDENCE / WORKSPACE / TECHNICAL) instead of replaying the pipeline's stage
list, which lives on as one compact status line and the full trace in Run
details.

PRE-RUN FLOW DELIBERATELY UNTOUCHED
-----------------------------------
The workspace exists only for a run at DONE or ACCEPTED. New-run intake,
clarifier questions and the context lock keep the focused linear flow in
ui.py; nothing here renders for them, so no mandatory pipeline step ever
waits on navigation.

HISTORY — READ-ONLY (this phase)
--------------------------------
The History view browses previous runs WITHOUT resuming or mutating them.
Its data layer is run_history.py, which reads the existing `.cache/runs/`
checkpoints and nothing else — no second persistence system. The selection
lives ONLY in UI session state (the selected run id and the loaded display
object); it never touches `st.session_state["state"]`, the current run id,
pending feedback or the checkpoint pointer, so leaving History always lands
exactly where the user left the current run.

The historical detail REUSES the section renderers, minus every action they
carry on the live screen: no requirements correction, no design box, no
advisory ask (it would call an LLM), no sign-off. The read-only wrappers
below compose renderers rather than duplicating them, so a historical
Context Record is drawn by the same code as a live one.

CHAT — READ-ONLY, GROUNDED
--------------------------
The Chat view is a real architecture assistant for the CURRENT run (with
explicitly historical questions answered from the read-only History layer).
Its data and answer layer is architecture_chat.py; this view is DRAW-only
like everything here — it renders the transcript and input, RETURNS a
submitted question as an intent, and ui.py performs the single grounded
answer call. Messages live in Streamlit session state keyed by run id, so
switching runs never mixes conversations, and nothing is ever persisted to
checkpoints or any store.
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime

import streamlit as st

import architecture_chat
import run_history
from pipeline.sign_off import feedback_is_closed
from pipeline.state import ArchitectState, Stage
from ui_sections import (
    clear_pending_feedback,
    get_pending_feedback,
    render_acceptance,
    render_adrs,
    render_architecture_summary,
    render_blueprint,
    render_clarification_history,
    render_components,
    render_component_summary,
    render_context_cards,
    render_design_advisory,
    render_executive_recommendation,
    render_feedback_box,
    render_features,
    render_key_decisions,
    render_key_risks,
    render_knowledge,
    render_blocking_findings,
    render_objections,
    render_repo_analysis,
    render_review_confidence,
    render_review_report,
    render_run_status,
    render_run_trace,
    render_sign_off,
    render_status_strip,
    render_target_architecture,
    render_user_rounds,
    render_why_architecture,
    render_risks_and_tradeoffs,
)

# The sidebar navigation, GROUPED by what the reader wants, not by what the
# pipeline did. Order inside the tuples is reading order; the flattened
# sequence is the app's canonical view list (`WORKSPACE_VIEWS`).
#
# The old flat list mixed the client-facing result (Overview) with pipeline
# vocabulary, and a second block above it replayed the pipeline stages. The
# groups answer the two questions a reader arrives with: what is the
# architecture (PROJECT), what is it built on (EVIDENCE), and only then the
# tooling (WORKSPACE placeholders) and the machinery (TECHNICAL).
#
# `Design` is now `Architecture` in every user-facing label: that view IS
# the deliverable. Backend names (stages, agents, schemas) are untouched —
# this is a UI label, not a rename of the pipeline's concepts.
_NAV_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("PROJECT", ("Overview", "Architecture", "Review")),
    ("EVIDENCE", ("Context", "Repository", "Knowledge")),
    ("WORKSPACE", ("History", "Chat")),
    ("TECHNICAL", ("Run details",)),
)

WORKSPACE_VIEWS: tuple[str, ...] = tuple(
    view for _, views in _NAV_GROUPS for view in views
)

# The History view's whole copy for the empty case — one sentence, because
# "no runs on this machine yet" is a state, not a problem.
HISTORY_EMPTY_MESSAGE = "No saved runs are available on this machine yet."

# Session key for the radio. Streamlit stores the selection under this key,
# which is what makes the navigation persistent across reruns — no extra
# session bookkeeping and no persistence layer.
_NAV_KEY = "workspace_view"


@dataclass
class WorkspaceIntents:
    """What the human asked for from the visible view, if anything.

    At most one view renders per rerun, so at most one of these is ever
    non-None — but they are kept as separate fields (not a union) so ui.py
    can keep the same precedence order the single-page screen had:
    question, then sign-off. (Feedback is NOT here: the boxes stage into
    the pending bundle, and its submission — one round for the whole bundle — is
    driven by the sidebar panel, not by a view. The Chat question is also
    NOT feedback: it is a read-only grounded answer, performed through
    architecture_chat and costing no round.)
    """

    question: str | None = None
    sign_off: tuple[str, str] | None = None
    chat_question: str | None = None


# Which workspace view contributed which kind of pending feedback, for the
# sidebar preview. Submission order ("requirements" first) is the backend's;
# the labels are this UI's.
_PENDING_KIND_VIEWS = {"requirements": "Context", "design": "Architecture"}


def render_pending_feedback_panel(state: ArchitectState) -> bool:
    """DRAW the pending-feedback panel in the sidebar. True = submit requested.

    THE GLOBAL SUBMIT ACTION for all feedback lives here: one button sends
    the whole bundle — Context and Design text together — as a single
    `user_feedback.submit_feedback` call, i.e. ONE refinement round. The
    boxes themselves never submit; they only stage into the bundle this
    panel previews.

    Hidden entirely while nothing is pending, so the sidebar stays clean on a
    fresh run and the panel's appearance is itself the "you have unsent
    feedback" signal. Drawn only for a finished run (the caller's condition).

    Returns True only for the submit action. Discard is handled here — it
    touches nothing but the session-state bundle. When feedback is closed
    (accepted run / round budget spent) the bundle is still shown — the
    person should see what they staged — but no submit is offered.
    """

    pending = get_pending_feedback()
    kinds = [
        kind for kind in ("requirements", "design")
        if str(pending.get(kind) or "").strip()
    ]
    if not kinds:
        return False

    st.divider()
    st.markdown("**Pending feedback round**")
    for kind in kinds:
        one_line = " ".join(str(pending[kind]).split())
        preview = one_line if len(one_line) <= 70 else one_line[:70] + "…"
        st.caption(f"**{_PENDING_KIND_VIEWS[kind]}** — {preview}")

    closed, why = feedback_is_closed(state)
    if closed:
        st.caption(
            f"Feedback is closed for this run ({why}), so this cannot be "
            f"submitted. Start a new run to take it further."
        )
        return False

    st.caption(
        "One submission is one refinement round, however many views "
        "contributed."
    )
    if st.button(
        "Submit feedback round", type="primary", use_container_width=True
    ):
        return True
    if st.button("Discard pending", use_container_width=True):
        clear_pending_feedback()
        st.rerun()
    return False


def select_workspace_view() -> str:
    """DRAW the grouped navigation. Call from inside the sidebar block.

    One full-width `st.button` per destination under small group headers.
    Buttons rather than a radio, on purpose: a radio cannot render
    non-selectable group headers, and four separate radios would each keep
    their own selection with no way to know which one the human just used.
    The selection persists in `st.session_state[_NAV_KEY]`; the current view is
    highlighted with the primary button style.

    THE CLICK IS READ BEFORE ANY BUTTON IS DRAWN. A button's `type=` is
    decided at the moment `st.button` is called, and the click it answers
    is only known AFTER that call returns — so styling from the return
    value painted the ACTIVE button one rerun late (click Chat: this rerun
    still paints Knowledge green; the next interaction fixes it). Streamlit
    commits the clicked button's value to `st.session_state[key]` at the
    START of the rerun the click triggers, so a pre-pass over the keys lets
    this rerun resolve the destination FIRST and then style every button
    against it — content and highlight move in the same interaction, and
    `_NAV_KEY` stays the one source of truth (the peek only accelerates its
    update; without it the old two-step behaviour would simply lag again).
    """

    current = st.session_state.get(_NAV_KEY, "Overview")
    for view in WORKSPACE_VIEWS:
        if st.session_state.get(f"nav_{view}"):
            st.session_state[_NAV_KEY] = current = view
    for title, views in _NAV_GROUPS:
        st.caption(f"**{title}**")
        for view in views:
            st.button(
                view,
                key=f"nav_{view}",
                use_container_width=True,
                type="primary" if view == current else "secondary",
            )
    return current


# ─────────────────────────────────────────────────────────────────────────
# The lightweight visual system
#
# ONE small, scoped <style> block and nothing else: a capped, centered main
# column (Streamlit has no max-width control natively — this is the one rule
# that needs a global-ish selector, and `.block-container` has been the main
# column's class for years), plus the workspace's own `ws-*` classes for the
# header, chips, and cards. No external UI library, no DOM-id selectors, no
# animations — just enough surface for the views to share one look.
# ─────────────────────────────────────────────────────────────────────────
_WORKSPACE_STYLES = """
<style>
.block-container { max-width: 1080px; margin: 0 auto;
                    padding-top: 1.1rem; padding-bottom: 2.5rem; }
.ws-header { display: flex; align-items: baseline; gap: .75rem;
             flex-wrap: wrap; margin-bottom: .15rem; }
.ws-view { font-size: 1.45rem; font-weight: 700; color: #147B58; }
.ws-project { font-size: 1.1rem; font-weight: 600; color: #333333; }
.ws-chips { margin-left: auto; display: flex; gap: .4rem; }
.ws-chip { font-size: .78rem; font-weight: 600; padding: .12rem .6rem;
           border-radius: 999px; border: 1px solid #d5e5dc;
           background: #EFF6F2; color: #147B58; white-space: nowrap; }
.ws-chip-fail { background: #FDECEA; border-color: #f2c4be; color: #C0392B; }
.ws-chip-warn { background: #FFF8E6; border-color: #eedfc0; color: #8a6d1a; }
.ws-sub { color: #6b7a74; font-size: .8rem; margin-bottom: .9rem; }
.ws-card { border: 1px solid #dfe8e3; border-radius: 8px;
           padding: .7rem .9rem; margin-bottom: .75rem; background: #ffffff; }
.ws-card-title { font-size: .72rem; font-weight: 700; color: #147B58;
                 letter-spacing: .08em; text-transform: uppercase;
                 margin-bottom: .3rem; }
.ws-card-meta { font-size: .75rem; color: #6b7a74; }
.ws-card-body { font-size: .9rem; color: #333333; line-height: 1.45; }
</style>
"""


def _workspace_chips(state: ArchitectState) -> str:
    """The 2-3 status facts that belong in the header, as chip HTML."""

    chips: list[str] = []
    review = state.review
    if review is None:
        chips.append('<span class="ws-chip ws-chip-warn">Not reviewed</span>')
    elif review.overall_status == "pass":
        chips.append('<span class="ws-chip">PASS</span>')
    else:
        chips.append('<span class="ws-chip ws-chip-fail">FAIL</span>')
    if state.stopped_on_cap:
        chips.append('<span class="ws-chip ws-chip-warn">Stopped on budget</span>')
    if state.stage is Stage.ACCEPTED:
        chips.append('<span class="ws-chip">Accepted</span>')
    elif state.stage is Stage.DONE:
        chips.append('<span class="ws-chip">Done</span>')
    return "".join(chips)


def _repo_short_name(state: ArchitectState) -> str:
    """The repository's short name for the header's secondary line."""

    url = (
        state.repo_representation.meta.url
        if state.repo_representation is not None
        else state.initial_request.repo_url
    ).strip()
    return url.rstrip("/").rsplit("/", 1)[-1] if url else ""


def render_workspace_header(state: ArchitectState, view: str) -> None:
    """The one-line anchor every workspace view starts with.

    View name, project name, status chips, then a thin secondary line (run
    id, repository). Nothing more: the header orients, the view below it
    carries the content. All values are escaped — they come from the run's
    own artifacts.
    """
    record = state.context_record
    project = (
        (record.project_name if record is not None else "")
        or (state.blueprint.project_name if state.blueprint is not None else "")
        or "Architecture run"
    ).strip() or "Architecture run"

    sub = [f"Run <code>{html.escape(state.run_id)}</code>"]
    repo = _repo_short_name(state)
    if repo:
        sub.append(f"repo <code>{html.escape(repo)}</code>")

    st.markdown(
        _WORKSPACE_STYLES
        + "<div class='ws-header'>"
        + f"<span class='ws-view'>{html.escape(view)}</span>"
        + f"<span class='ws-project'>{html.escape(project)}</span>"
        + f"<span class='ws-chips'>{_workspace_chips(state)}</span>"
        + "</div>"
        + f"<div class='ws-sub'>{' · '.join(sub)}</div>",
        unsafe_allow_html=True,
    )


def _render_overview_view(state: ArchitectState) -> tuple[str, str] | None:
    """The CLIENT-FACING ANSWER, scannable top to bottom.

    Reading order is the order a client asks: what do we recommend and why
    are we here (the pattern and goal lead, so the target is known
    immediately without the large visual), why this design, what was
    decided, what it consists of, what the risks are, whether it passed —
    and only then the large Target architecture visualization, which would
    otherwise dominate the first screenful. The sign-off stays last: it
    acts on the whole picture above it.

    Every section is existing artifact text, assembled deterministically
    (the renderers live in ui_sections beside this note). What is
    deliberately NOT here: run trace, token/cost metrics, stage machinery,
    run identifiers in the body, full ADR bodies, full Component
    descriptions, KB chunks, file trees — operations and evidence belong to
    their own views.

    A Migration Approach section is deliberately absent: nothing in the
    saved artifacts carries an ordered migration sequence (only prose
    mentions of "incremental" work), and inferring one from prose is not
    displaying the run's record. It stays omitted until an artifact states
    a sequence explicitly.
    """

    render_executive_recommendation(state)   # A. what we recommend, and why we're here
    render_why_architecture(state)           # C. grounded reasons, deduped, capped
    render_key_decisions(state)              # E. every ADR, one line each
    render_component_summary(state)          # F. name / type / purpose grid
    render_risks_and_tradeoffs(state)        # G. risks, trade-offs, open findings
    render_review_confidence(state)          # H. verdict + counts, compact
    render_target_architecture(state)        # B. the large visual, below the verdict
    render_acceptance(state)                 # I. the acceptance record, when it exists
    return render_sign_off(state)            # I. the client action, when still valid


def _render_run_details_view(state: ArchitectState) -> None:
    """The machinery: metrics, trace, and refinement budget — operational
    metadata drawn entirely from existing state. Secondary on purpose: a
    client reading the architecture never needs this screen, and an auditor
    finds everything here."""

    render_status_strip(state)  # verdict/refine/tokens/cost/counts/run id
    render_user_rounds(state)   # what refinement has cost so far
    render_run_trace(state)     # multi-step reasoning — the trace, in full


def _render_context_view(state: ArchitectState) -> None:
    """The ground truth the design was built on — context FIRST, chat SECOND.

    Order is the fix for the transcript look: the objective and the locked
    record are the primary objects (cards, not bullets in an expander), the
    derived Features follow, the clarification Q&A becomes a collapsed
    secondary section, and the requirements box — the one action here — is
    demoted to a labelled expander. Its staging semantics are unchanged: the
    expensive path starts only when the sidebar's submit action sends the
    whole bundle, so a correction prepared here can travel with a design
    directive staged in the Design view, as one round.
    """

    render_context_cards(state)
    render_features(state)
    render_clarification_history(state)
    render_feedback_box(state, "requirements")


def _render_knowledge_view(state: ArchitectState) -> None:
    """What the researcher retrieved — the grounding, nothing else."""

    render_knowledge(state)


def _render_repository_view(state: ArchitectState) -> None:
    """The repository analysis the ingestor built, when there is one."""

    render_repo_analysis(state)


def _render_architecture_view(state: ArchitectState) -> str | None:
    """The DELIVERABLE in full — the view the sidebar now names honestly.

    Objections render FIRST: they say the design below deviates from what
    was asked for, and everything after them would read as compliance. The
    feedback box STAGES into the pending bundle like the Context one — same
    bundle, same single round on submit.
    """

    render_objections(state)
    render_blueprint(state)
    render_adrs(state)
    render_components(state)
    # ASK before DIRECT — the cheap read-only question above the box that
    # costs a full refine round. Same order, same semantics as before.
    question = render_design_advisory(state)
    render_feedback_box(state, "design")
    return question


def _render_review_view(state: ArchitectState) -> None:
    """The quality gate: code-owned checks, LLM judgments, blocking issues."""

    render_review_report(state)


# ─────────────────────────────────────────────────────────────────────────
# History — chronological, READ-ONLY browsing of saved runs
#
# Two screens, one session key: the LIST (compact run cards, newest first)
# and the DETAIL of the one selected run (header + six read-only sections in
# tabs). Selection is pure UI state — `history_selected_run_id` plus the
# loaded display object — and clearing it returns to the list without ever
# touching the current run's state.
# ─────────────────────────────────────────────────────────────────────────

# Session keys. The selected id and the loaded display state are the ONLY
# things History writes, and both are UI-only: the current pipeline state,
# its run id, the pending-feedback bundle and the checkpoint pointer are
# never touched (the current run keeps checkpointing itself as it always did).
_HISTORY_RUN_KEY = "history_selected_run_id"
_HISTORY_STATE_KEY = "history_display_state"


def _select_history_run(run_id: str) -> None:
    """on_click for an Open button: remember which run was asked for."""
    st.session_state[_HISTORY_RUN_KEY] = run_id


def _clear_history_selection() -> None:
    """on_click for the back button: return to the list, drop the display copy."""
    st.session_state.pop(_HISTORY_RUN_KEY, None)
    st.session_state.pop(_HISTORY_STATE_KEY, None)


def _history_date(iso: str) -> str:
    """Date AND time from an ISO timestamp; the raw value if it will not parse."""
    try:
        return datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError):
        return str(iso or "")


def _history_stage_label(stage: str) -> str:
    """The saved stage as one compact phrase. Unknown values degrade to
    themselves — never a fabricated status."""
    if not stage:
        return "saved early"
    if stage in ("done", "accepted", "failed"):
        return stage.capitalize()
    return f"stopped at {stage.replace('_', ' ')}"


def _history_status_chips(summary: run_history.HistorySummary) -> str:
    """The status facts a list card / detail header carries, as chip HTML."""

    chips: list[str] = []
    if summary.verdict == "pass":
        chips.append('<span class="ws-chip">PASS</span>')
    elif summary.verdict == "fail":
        chips.append('<span class="ws-chip ws-chip-fail">FAIL</span>')
    else:
        chips.append('<span class="ws-chip ws-chip-warn">No review</span>')
    if summary.accepted:
        chips.append('<span class="ws-chip">Accepted</span>')
    if summary.stage == "failed":
        chips.append('<span class="ws-chip ws-chip-fail">Failed</span>')
    return "".join(chips)


def _history_card_html(
    summary: run_history.HistorySummary, is_current: bool
) -> str:
    """One run as a compact History row card (HTML, all values escaped)."""

    counts = (
        f"{summary.feature_count} features · {summary.adr_count} ADRs · "
        f"{summary.component_count} components · {summary.kb_chunk_count} KB chunks"
    )
    meta = [
        _history_date(summary.updated_at),
        summary.repo_name or "no repository",
        f"run <code>{html.escape(summary.run_id)}</code>",
    ]
    if is_current:
        meta.append('<span class="ws-chip">current run</span>')
    return (
        "<div class='ws-card'>"
        + f"<div class='ws-card-title'>{html.escape(summary.project_label)}"
        + f"<span class='ws-chips' style='margin-left:auto'>"
        + _history_status_chips(summary)
        + "</span></div>"
        + f"<div class='ws-card-meta'>{' · '.join(meta)}</div>"
        + f"<div class='ws-card-body'>{html.escape(_history_stage_label(summary.stage))}"
        + f" · {html.escape(counts)}</div>"
        + "</div>"
    )


def _render_history_list(current: ArchitectState) -> None:
    """Every saved run on this machine, newest first, one card per run."""

    summaries = run_history.list_history_runs()
    if not summaries:
        # Not an error: a fresh machine (or an empty cache) simply has no
        # history yet, and the message says exactly that.
        st.info(HISTORY_EMPTY_MESSAGE)
        return

    for summary in summaries:
        left, right = st.columns([7, 1], gap="small")
        with left:
            st.markdown(
                _history_card_html(summary, summary.run_id == current.run_id),
                unsafe_allow_html=True,
            )
        with right:
            st.button(
                "Open",
                key=f"hist_open_{summary.run_id}",
                on_click=_select_history_run,
                args=(summary.run_id,),
                use_container_width=True,
            )


def _history_display_state(run_id: str) -> ArchitectState | None:
    """The selected run's state, loaded ON DEMAND and cached for the session.

    Full deserialization happens only here — the list pass reads plain JSON
    summaries — and the result is kept in session state so switching tabs
    inside the detail does not re-read the checkpoint. The cache holds ONE
    display object, replaced when a different run is opened; it is display
    state only and can never be mistaken for the active run.
    """
    cached = st.session_state.get(_HISTORY_STATE_KEY)
    if cached is not None and cached[0] == run_id:
        return cached[1]
    try:
        state = run_history.load_history_run(run_id)
    except run_history.HistoryError:
        return None
    st.session_state[_HISTORY_STATE_KEY] = (run_id, state)
    return state


def _history_detail_chips(state: ArchitectState) -> str:
    """Status chips for the detail header, from the loaded state — handles
    every stage a saved run can have stopped at, not just the finished ones
    the live workspace header ever sees."""

    chips: list[str] = []
    review = state.review
    if review is None:
        chips.append('<span class="ws-chip ws-chip-warn">No review</span>')
    elif review.overall_status == "pass":
        chips.append('<span class="ws-chip">PASS</span>')
    else:
        chips.append('<span class="ws-chip ws-chip-fail">FAIL</span>')
    if state.stopped_on_cap:
        chips.append('<span class="ws-chip ws-chip-warn">Stopped on budget</span>')
    if state.stage is Stage.ACCEPTED:
        chips.append('<span class="ws-chip">Accepted</span>')
    elif state.stage is Stage.FAILED:
        chips.append('<span class="ws-chip ws-chip-fail">Failed</span>')
    elif state.stage is not Stage.DONE:
        chips.append(
            f"<span class='ws-chip ws-chip-warn'>Stopped at "
            f"{html.escape(state.stage.value.replace('_', ' '))}</span>"
        )
    return "".join(chips)


def _render_history_detail_header(state: ArchitectState) -> None:
    """The selected run's own header: project, when, repo, run id, status."""

    record = state.context_record
    project = (
        (record.project_name if record is not None else "")
        or (state.blueprint.project_name if state.blueprint is not None else "")
        or "Architecture run"
    ).strip() or "Architecture run"

    last_step = state.history[-1].timestamp if state.history else ""
    sub = [_history_date(last_step) or "date unknown"]
    repo = _repo_short_name(state)
    sub.append(f"repo <code>{html.escape(repo)}</code>" if repo else "no repository")
    sub.append(f"run <code>{html.escape(state.run_id)}</code>")

    st.markdown(
        "<div class='ws-header'>"
        + "<span class='ws-view'>Saved run</span>"
        + f"<span class='ws-project'>{html.escape(project)}</span>"
        + f"<span class='ws-chips'>{_history_detail_chips(state)}</span>"
        + "</div>"
        + f"<div class='ws-sub'>{' · '.join(sub)} · read-only</div>",
        unsafe_allow_html=True,
    )


def _render_historical_overview(state: ArchitectState) -> None:
    """The client cut of a saved run — the live Overview minus the sign-off.

    Same hierarchy the accepted information architecture prescribes: the
    recommended architecture, key decisions, key components, the verdict and
    its open findings, the stated risks. No trace, tokens or cost: those are
    operations, and operations belong to the live Run details view.
    """
    render_run_status(state)             # the honest verdict, unchanged
    render_acceptance(state)             # a signed-off run shows its acceptance
    render_architecture_summary(state)
    render_key_decisions(state)
    render_component_summary(state)
    render_blocking_findings(state)
    render_key_risks(state)
    # NOT render_sign_off: history cannot accept anything.


def _render_historical_context(state: ArchitectState) -> None:
    """The saved ground truth — record, features, clarification Q&A.

    NOT render_feedback_box: a correction would re-open a run that is not
    the user's current run, so the box simply does not exist here.
    """
    render_context_cards(state)
    render_features(state)
    render_clarification_history(state)


def _render_historical_architecture(state: ArchitectState) -> None:
    """The saved deliverable in full.

    NOT render_design_advisory and NOT render_feedback_box: the advisory ask
    would spend an LLM call against a run that is over, and a directive
    would spend a refinement round on it. Neither belongs in history.
    """
    render_objections(state)
    render_blueprint(state)
    render_adrs(state)
    render_components(state)


def _render_history_detail(run_id: str) -> None:
    """One selected run: header, then the six read-only sections as tabs."""

    state = _history_display_state(run_id)
    st.button("← All saved runs", on_click=_clear_history_selection)
    if state is None:
        st.error(
            "That saved run's checkpoints could not be read back from disk. "
            "The rest of the list is unaffected."
        )
        return

    _render_history_detail_header(state)
    st.caption(
        "A saved run, opened read-only — it cannot be resumed, corrected, "
        "refined or signed off from here. Your current run is unchanged."
    )

    overview, context, architecture, review, knowledge, repository = st.tabs(
        ["Overview", "Context", "Architecture", "Review", "Knowledge", "Repository"]
    )
    with overview:
        _render_historical_overview(state)
    with context:
        _render_historical_context(state)
    with architecture:
        _render_historical_architecture(state)
    with review:
        render_review_report(state)
    with knowledge:
        render_knowledge(state)
    with repository:
        render_repo_analysis(state)


def _render_history_view(current: ArchitectState) -> None:
    """DRAW the History destination: the list, or one run's read-only detail."""

    st.caption("Local saved architecture runs.")
    selected = st.session_state.get(_HISTORY_RUN_KEY)
    if selected:
        _render_history_detail(selected)
    else:
        _render_history_list(current)


# ─────────────────────────────────────────────────────────────────────────
# Chat — the read-only, grounded architecture assistant
#
# DRAW-only like every view: the transcript and the input render here, a
# submitted question is RETURNED as an intent, and ui.py performs the one
# grounded answer call through architecture_chat (which owns session state,
# citations and every read-only rule). Rendering this view costs nothing —
# no LLM, no retrieval, no history scan.
# ─────────────────────────────────────────────────────────────────────────

_SCOPE_LABELS = {"current": "Current run", "history": "History", "kb": "KB"}


def _render_chat_sources(sources: list[dict]) -> None:
    """`Sources used` under one assistant answer: the citation ids the
    answer was grounded in, labelled with scope, run/date and KB metadata."""

    if not sources:
        return
    with st.expander(f"Sources used — {len(sources)}"):
        for source in sources:
            meta = [
                _SCOPE_LABELS.get(source.get("scope", ""), "Source"),
                html.escape(str(source.get("label", ""))),
            ]
            if source.get("run_date"):
                meta.append(html.escape(str(source["run_date"])))
            box = source.get("box") or 0
            if box:
                meta.append(f"box {box}")
            distance = source.get("distance")
            if distance is not None:
                meta.append(f"distance {distance:.4f}")
            st.caption(
                f"`[{html.escape(str(source.get('sid', '')))}]` · "
                + " · ".join(meta)
            )


def _render_chat_view(current: ArchitectState) -> str | None:
    """DRAW the Chat destination. Returns a freshly submitted question, if
    any — the REACT half (the one grounded answer call) lives in ui.py."""

    st.markdown("#### Architecture Chat")
    st.caption(
        "Ask questions about the current architecture, its evidence, and "
        "relevant previous runs."
    )

    record = current.context_record
    project = (
        (record.project_name if record is not None else "")
        or (current.blueprint.project_name if current.blueprint is not None else "")
        or "Architecture run"
    ).strip() or "Architecture run"
    st.caption(
        f"Active context: **{html.escape(project)}** · run "
        f"`{html.escape(current.run_id)}` — previous runs are consulted "
        f"only when your question asks about them."
    )

    messages = architecture_chat.messages_for(current.run_id)
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_chat_sources(message.get("sources") or [])

    if st.button("Clear chat", help="Clears this run's conversation only."):
        architecture_chat.clear_chat(current.run_id)
        st.rerun()

    question = st.chat_input("Ask about this architecture…")
    return question.strip() if question and question.strip() else None


def render_workspace_view(
    state: ArchitectState, view: str | None
) -> WorkspaceIntents:
    """DRAW the selected workspace view. Returns the intents it produced.

    Unknown or missing `view` falls back to Overview rather than an error:
    the selector is the only writer of the value and it cannot produce one,
    so the fallback is unreachable in practice — but a workspace that
    crashed over a navigation string would be a poor kind of failure.
    """

    render_workspace_header(state, view or "Overview")
    intents = WorkspaceIntents()

    if view == "Context":
        _render_context_view(state)
    elif view == "Repository":
        _render_repository_view(state)
    elif view == "Knowledge":
        _render_knowledge_view(state)
    elif view == "Architecture":
        intents.question = _render_architecture_view(state)
    elif view == "Review":
        _render_review_view(state)
    elif view == "Run details":
        _render_run_details_view(state)
    elif view == "History":
        # READ-ONLY browsing of saved runs. Produces no intents: nothing in
        # a historical run can be asked of, fed back into, or signed off.
        _render_history_view(state)
    elif view == "Chat":
        # The read-only grounded assistant. Returns the submitted question
        # as an intent; ui.py performs the ONE answer call.
        intents.chat_question = _render_chat_view(state)
    else:  # Overview — the default and the fallback
        intents.sign_off = _render_overview_view(state)

    return intents
