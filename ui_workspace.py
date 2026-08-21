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

PLACEHOLDERS
------------
History and Chat are destinations WITHOUT data logic in this phase. They draw
one line of text each — no filesystem scan, no LLM call, no retrieval — so
the navigation shape can land before the capabilities do.
"""
from __future__ import annotations

import html
from dataclasses import dataclass

import streamlit as st

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
    render_feedback_box,
    render_features,
    render_key_decisions,
    render_key_risks,
    render_knowledge,
    render_blocking_findings,
    render_objections,
    render_repo_analysis,
    render_review_report,
    render_run_status,
    render_run_trace,
    render_sign_off,
    render_status_strip,
    render_user_rounds,
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

HISTORY_PLACEHOLDER = "History will be available in the next iteration."
CHAT_PLACEHOLDER = "Architecture Chat will be available in the next iteration."

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
    question, then sign-off. (Feedback is NOT here: the boxes stage into the
    pending bundle, and its submission — one round for the whole bundle — is
    driven by the sidebar panel, not by a view.)
    """

    question: str | None = None
    sign_off: tuple[str, str] | None = None


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
    A clicked button returns True on exactly the rerun it was clicked, and
    that same rerun can switch the view — no `st.rerun()` round trip. The
    selection persists in `st.session_state[_NAV_KEY]`; the current view is
    highlighted with the primary button style.
    """

    current = st.session_state.get(_NAV_KEY, "Overview")
    for title, views in _NAV_GROUPS:
        st.caption(f"**{title}**")
        for view in views:
            if st.button(
                view,
                key=f"nav_{view}",
                use_container_width=True,
                type="primary" if view == current else "secondary",
            ):
                st.session_state[_NAV_KEY] = current = view
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
    """The CLIENT-FACING RESULT: what architecture, why, what it rests on.

    Answers, in one screenful: the recommended architecture (summary card
    from the Blueprint's own views), the key decisions (ADR titles + one
    line each), the key components (compact grid), the review verdict and
    anything it left open, the design's own stated risks — and the sign-off,
    which acts on this whole picture.

    What is deliberately NOT here: run trace, token/cost metrics, stage
    detail, run identifiers. Those are operations, not results, and they
    live in `Run details`. The header's thin secondary line still carries
    the run id — one line, every view, for audit — but nothing operational
    renders in the body.
    """

    render_run_status(state)             # honest verdict — never a false success
    render_acceptance(state)             # and, separately, whether a human took it
    render_architecture_summary(state)   # pattern + both blueprint views, visible
    render_key_decisions(state)          # ADR titles + one-line decisions
    render_component_summary(state)      # name / type / purpose grid
    render_blocking_findings(state)      # what the review left open, if anything
    render_key_risks(state)              # the design's own stated risks
    return render_sign_off(state)


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
        # Placeholder ONLY: no history scan, no stored-run retrieval.
        st.info(HISTORY_PLACEHOLDER)
    elif view == "Chat":
        # Placeholder ONLY: no LLM call, no chat history, no retrieval.
        st.info(CHAT_PLACEHOLDER)
    else:  # Overview — the default and the fallback
        intents.sign_off = _render_overview_view(state)

    return intents
