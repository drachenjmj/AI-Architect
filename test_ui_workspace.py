"""test_ui_workspace.py — tests for the Phase 1 workspace navigation.

Scope: the NAVIGATION LAYER only — the sidebar selector for a finished run
and the one-view-at-a-time dispatch in ui_workspace.py. The section
renderers themselves are unchanged and already covered by
test_ui_context_record.py; here we prove the restructure did what it exists
to do (only the selected view renders) and did not do what it must not do
(touch the pre-run flow, or trigger any pipeline/API action from the
placeholder views).

Runs via Streamlit's AppTest against the real ui.py: no server, no browser,
no network, no LLM. The finished run is the offline demo state (the same one
`streamlit run ui.py -- --demo` loads), seeded directly into session state,
so nothing is ever checkpointed and no architecture run happens.

NEVER `import ui` IN THIS FILE. ui.py runs Streamlit calls at module level;
importing it outside a script-run (bare mode) executes the intake `st.form`
without a script context, which leaks Streamlit's form-id state into every
later AppTest script thread in the process — the next `st.button` anywhere
then fails with "can't be used in an `st.form()`". ui_sections and
ui_workspace are safe to import (no module-level st calls); ui.py is not.
Monkeypatching therefore targets the SOURCE modules: every action call in
ui.py's DONE branch resolves through module attribute access
(`clarifier_gate.ask_advisor`, `user_feedback.submit_feedback`,
`sign_off.accept_design`), so source-module patches are fully effective.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ui_demo import build_demo_state
from ui_workspace import WORKSPACE_VIEWS

_UI = str(Path(__file__).parent / "ui.py")


def _finished_app() -> AppTest:
    """The app with a completed run loaded, exactly as a resume produces."""

    at = AppTest.from_file(_UI, default_timeout=30)
    at.session_state["state"] = build_demo_state("pass")
    at.run()
    assert not at.exception
    return at


def _expander_labels(at: AppTest) -> list[str]:
    return [e.label for e in at.expander]


def _main_expander_labels(at: AppTest) -> list[str]:
    """Expander labels EXCLUDING the sidebar's Switch-run control, which is
    chrome rather than view content."""
    return [label for label in _expander_labels(at) if label != "Switch run"]


def _has_expander(at: AppTest, needle: str) -> bool:
    return any(needle in label for label in _expander_labels(at))


def _select_view(at: AppTest, view: str) -> AppTest:
    """Navigate by clicking the sidebar's grouped view buttons."""

    next(b for b in at.sidebar.button if b.label == view).click()
    at.run()
    assert not at.exception
    return at


# ── 1. the grouped sidebar navigation ──────────────────────────────────────


def test_sidebar_exposes_the_grouped_destinations():
    at = _finished_app()

    labels = [b.label for b in at.sidebar.button if b.label != "🔄 New run"]
    assert labels == [
        "Overview", "Architecture", "Review",
        "Context", "Repository", "Knowledge",
        "History", "Chat",
        "Run details",
    ]


def test_old_design_navigation_label_is_gone():
    at = _finished_app()

    labels = [b.label for b in at.sidebar.button]
    assert "Design" not in labels
    assert "Architecture" in labels


def test_sidebar_shows_a_compact_status_not_a_stage_checklist():
    at = _finished_app()

    status = " ".join(m.value for m in at.sidebar.markdown)
    assert "✓ Completed" in status
    assert "Review passed" in status
    # The old six-row stage checklist is gone; the one line carries the
    # verdict and the refinement count instead.
    assert "Ingest repo" not in status
    assert "1 refinement" in status


def test_every_destination_switches_cleanly():
    at = _finished_app()
    for view in WORKSPACE_VIEWS:
        _select_view(at, view)  # asserts no exception internally


def _nav_types(at: AppTest) -> dict[str, str]:
    """label -> button type ("primary"/"secondary") for the nav buttons."""

    return {
        b.label: b.proto.type
        for b in at.sidebar.button
        if b.label in WORKSPACE_VIEWS
    }


def test_click_updates_highlight_and_content_in_the_same_interaction():
    """Regression: the green highlight used to lag one click behind the
    content — clicking Chat rendered Chat but kept Knowledge primary until
    the NEXT interaction. Content and active styling must move together."""

    at = _finished_app()
    next(b for b in at.sidebar.button if b.label == "Chat").click()
    at.run()
    assert not at.exception

    types = _nav_types(at)
    assert types["Chat"] == "primary"          # active IMMEDIATELY...
    assert sum(1 for t in types.values() if t == "primary") == 1
    # ...and the content on screen is the same destination as the highlight.
    assert any("Architecture Chat" in m.value for m in at.markdown)

    # A second click keeps exactly one active and moves it again.
    next(b for b in at.sidebar.button if b.label == "Run details").click()
    at.run()
    assert not at.exception
    types = _nav_types(at)
    assert types["Run details"] == "primary"
    assert types["Chat"] == "secondary"
    assert sum(1 for t in types.values() if t == "primary") == 1


# ── 2. Overview: the client-facing result, not the operations log ─────────


def test_overview_is_the_client_facing_architecture_result():
    at = _finished_app()

    md = " | ".join(m.value for m in at.markdown)
    caps = " | ".join(c.value for c in at.caption)
    # The recommendation and the structure it names, directly visible.
    assert "Executive recommendation" in md
    assert "Event-driven services" in md            # the pattern, visible
    assert "Target architecture" in md
    assert "Key decisions" in md
    assert "ADR-1" in md and "ADR-2" in md          # every ADR, one line each
    assert "Key components" in md
    assert "Checkout Service" in md
    assert "REVIEW PASSED" in md                    # compact confidence line
    assert "Risks and trade-offs" in md
    assert any(
        "Accept this design" in b.label for b in at.button
    )                                          # the sign-off lives here
    # Operational machinery is NOT on the client view.
    assert not _has_expander(at, "Run trace")
    assert len(at.metric) == 0                  # no token/cost metric cards
    assert "Tokens" not in md + caps


# ── 3. one view at a time — the reason the restructure exists ──────────────


def test_knowledge_view_renders_knowledge_but_not_design_or_review():
    at = _select_view(_finished_app(), "Knowledge")

    # Compact source cards under a header line (no giant expander).
    assert any("Knowledge retrieved" in m.value for m in at.markdown)
    assert any(
        "architecture_patterns.md" in m.value for m in at.markdown
    )  # a source card is on screen
    # No Design detail, no Review detail, no repository, no trace.
    assert not _has_expander(at, "Blueprint")
    assert not _has_expander(at, "Architecture Decision Records")
    assert not _has_expander(at, "Components")
    assert not _has_expander(at, "Review report")
    assert not _has_expander(at, "Repository analysis")
    assert not _has_expander(at, "Run trace")


def test_architecture_view_renders_the_full_deliverable():
    at = _select_view(_finished_app(), "Architecture")

    assert _has_expander(at, "Blueprint")
    assert _has_expander(at, "Architecture Decision Records")
    assert _has_expander(at, "Components")
    # The read-only question and the change box, unchanged.
    assert any("understand something" in m.value for m in at.markdown)
    assert any(
        "Direct the architect" in (t.label or "") for t in at.text_area
    )
    assert any("Ask" == b.label for b in at.button)
    assert any("Add to pending feedback" == b.label for b in at.button)
    # No knowledge / review / repo detail bleeds in.
    assert not any("Knowledge retrieved" in m.value for m in at.markdown)
    assert not _has_expander(at, "Review report")
    assert not _has_expander(at, "Repository analysis")


def test_review_view_renders_review_content_only():
    at = _select_view(_finished_app(), "Review")

    md = " | ".join(m.value for m in at.markdown)
    # Verdict, issue count, loop status AND the dimension summary are all
    # directly visible — the checks are no longer hidden behind one giant
    # accordion.
    assert any("REVIEW — PASS" in m.value for m in at.markdown)
    assert any("No blocking issues" in c.value for c in at.caption)
    assert "Review dimensions" in md
    assert "Traceability" in md and "Repo grounding" in md  # real check names
    # Only the detailed EVIDENCE stays collapsible, under a clearer name.
    assert any(
        "Detailed reviewer evidence" in label for label in _expander_labels(at)
    )
    assert not _has_expander(at, "Rubric detail")   # the old giant expander is gone
    assert not _has_expander(at, "Blueprint")
    assert not _has_expander(at, "Knowledge retrieved")
    assert not _has_expander(at, "Components")
    assert not _has_expander(at, "Run trace")


def test_context_view_renders_record_features_and_requirements_box():
    at = _select_view(_finished_app(), "Context")

    # The record is the primary object (header + cards), the Q&A is a
    # collapsed secondary section, and the feedback entry is demoted into
    # its own expander — all still present, none removed.
    assert any("Context Record" in m.value for m in at.markdown)
    assert any("Seasonal Shop" in m.value for m in at.markdown)  # project
    assert _has_expander(at, "Clarification history")
    assert any(
        "Correct the context" in label for label in _expander_labels(at)
    )
    assert any(
        "Correct the requirements" in (t.label or "") for t in at.text_area
    )
    assert _has_expander(at, "Features")
    assert not _has_expander(at, "Blueprint")
    assert not _has_expander(at, "Review report")


def test_repository_summary_is_visible_without_expanding():
    at = _select_view(_finished_app(), "Repository")

    md = " | ".join(m.value for m in at.markdown)
    # Summary first, directly visible: stack, overview, partitions.
    assert "Repository analysis" in md
    assert "Python" in md and "Flask" in md        # detected stack
    assert "One Flask deployable" in md            # what it does
    assert "shop/checkout" in md                   # partition roles
    # Deep technical detail stays collapsible — but as SEPARATE expanders,
    # not one collapsed block hiding the whole analysis.
    labels = _expander_labels(at)
    assert any("File tree" in label for label in labels)
    assert any("Import edges" in label for label in labels)
    assert not any(label.strip().startswith("Repository analysis") for label in labels)


# ── 4b. Run details: the operational machinery, secondary but complete ─────


def test_run_details_holds_the_execution_metadata():
    at = _select_view(_finished_app(), "Run details")

    assert len(at.metric) == 6                     # tokens, cost, counts…
    assert any(
        "review" in m.value.lower() for m in at.markdown
    ) or len(at.metric) > 0                        # the strip, incl. verdict
    assert _has_expander(at, "Run trace")          # the full pipeline trace
    # The client-facing artifacts are NOT here.
    assert not _has_expander(at, "Blueprint")
    assert not _has_expander(at, "Components")


# ── 4. the placeholders are inert ──────────────────────────────────────────


@pytest.fixture()
def no_pipeline_actions(monkeypatch):
    """Fail the test if any pipeline/API/storage action fires during a run.

    Patched at the SOURCE modules (see the module docstring for why `ui`
    itself must not be imported here). Every action path reachable from the
    finished-run branch resolves these module attributes at call time.
    """

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003 — any call is the bug
        raise AssertionError("a pipeline/API/storage action was triggered")

    monkeypatch.setattr("pipeline.orchestrator.run_pipeline_streaming", _boom)
    monkeypatch.setattr("pipeline.persistence.list_runs", _boom)
    monkeypatch.setattr("pipeline.persistence.load_state", _boom)
    monkeypatch.setattr("pipeline.persistence.save_state", _boom)
    monkeypatch.setattr("pipeline.agents.clarifier.ask_advisor", _boom)
    monkeypatch.setattr("pipeline.user_feedback.submit_feedback", _boom)
    monkeypatch.setattr("pipeline.sign_off.accept_design", _boom)
    return _boom


def test_chat_view_renders_without_any_action(no_pipeline_actions, monkeypatch):
    """Rendering the Chat page costs nothing — the LLM answer call happens
    only when a question is SUBMITTED (covered in test_ui_chat.py)."""
    def _llm_boom(*args, **kwargs):
        raise AssertionError("an LLM call was triggered by rendering")

    monkeypatch.setattr("pipeline.llm.llm_call", _llm_boom)

    at = _select_view(_finished_app(), "Chat")

    assert any(
        "Ask questions about the current architecture" in c.value
        for c in at.caption
    )
    assert len(at.chat_input) == 1
    assert not _main_expander_labels(at)  # no transcript, no sources yet


# ── 5. the pre-run / Clarifier flow needs no navigation ────────────────────


def test_first_visit_shows_intake_without_workspace_navigation():
    at = AppTest.from_file(_UI, default_timeout=30)
    at.run()
    assert not at.exception

    # The linear intake form is right there in the main area.
    assert any(
        "System description" in (t.label or "") for t in at.text_area
    )
    assert any("Start" == b.label for b in at.button)
    # No workspace selector exists before a run is finished.
    assert not any(b.label == "Overview" for b in at.sidebar.button)


def test_clarifier_pause_keeps_the_linear_flow():
    from pipeline.state import PendingDecision, Stage, new_run

    state = new_run("Architect a ticketing system.")
    state.stage = Stage.AWAITING_HUMAN
    state.pending_decision = PendingDecision.CLARIFICATION
    state.clarifying_questions = ["What is the expected peak user count?"]

    at = AppTest.from_file(_UI, default_timeout=30)
    at.session_state["state"] = state
    at.run()
    assert not at.exception

    # The question form renders directly; no navigation is required.
    # (AppTest has no `at.form` accessor; the form's input and submit
    # button prove the linear flow is present.)
    assert any(
        "peak user count" in (t.label or "") for t in at.text_input
    )
    assert any("Submit answers" == b.label for b in at.button)
    assert not any(b.label == "Overview" for b in at.sidebar.button)


def _optional_context_state():
    """A state exactly as the clarifier leaves it when the optional round is
    owed. The prompt carries a brownfield/scale signal, so
    `non_functional_requirements` and `existing_systems` are REQUIRED — and
    already filled/qualitative, so neither is offered again. `cloud_provider`
    and `budget` carry no signal at all, so `optional_slots` (see
    `clarifier.OPTIONAL_CONTEXT_FIELDS`) offers exactly those: relevant to no
    required question, still empty, genuinely optional."""
    from pipeline.state import ContextRecord, PendingDecision, Stage, new_run

    state = new_run(
        "We're modernizing our legacy monolithic e-commerce platform to "
        "handle a much higher scale of peak sale-day traffic."
    )
    state.require_context_approval = True
    state.stage = Stage.AWAITING_HUMAN
    state.pending_decision = PendingDecision.OPTIONAL_CONTEXT
    state.context_record = ContextRecord(
        project_name="Sneaker Hub",
        business_goal="Modernize the platform",
        problem_statement="The legacy monolith cannot handle sale-day traffic",
        functional_requirements=["Browse and buy sneakers online"],
        non_functional_requirements=[
            "Substantially higher peak traffic than today; exact target unknown"
        ],
        existing_systems=["Legacy monolithic e-commerce platform"],
        cloud_provider="",
        budget="",
    )
    return state


def test_optional_context_pause_shows_a_dedicated_skippable_step():
    """The optional round is a distinct screen from both the required
    questions and the review screen — regression for the Clarifier UX
    cleanup (a fresh E2E run found optional questions bolted onto the bottom
    of the review form, looking like an afterthought)."""
    at = AppTest.from_file(_UI, default_timeout=30)
    at.session_state["state"] = _optional_context_state()
    at.run()
    assert not at.exception

    # cloud_provider and budget are both str fields -> st.text_input.
    # non_functional_requirements is REQUIRED here (already answered), so it
    # must not appear on this screen at all.
    assert any("cloud" in (t.label or "").lower() for t in at.text_input)
    assert any("budget" in (t.label or "").lower() for t in at.text_input)
    assert not any(
        "scale, availability, or performance" in (t.label or "")
        for t in at.text_area
    )
    assert any(b.label == "Skip optional questions" for b in at.button)
    assert any(b.label == "Continue" for b in at.button)
    assert not any(b.label == "Submit answers" for b in at.button)
    assert not any("Approve and continue" in (b.label or "") for b in at.button)


def test_skipping_the_optional_round_reaches_the_review_screen(tmp_path, monkeypatch):
    """Skip is non-blocking: fields stay blank, and the next screen is the
    Context Record review — never a second copy of the optional round."""
    monkeypatch.setenv("AI_ARCHITECT_RUNS_DIR", str(tmp_path / "runs"))

    at = AppTest.from_file(_UI, default_timeout=30)
    at.session_state["state"] = _optional_context_state()
    at.run()
    assert not at.exception

    next(b for b in at.button if b.label == "Skip optional questions").click()
    at.run()
    assert not at.exception

    resumed = at.session_state["state"]
    from pipeline.state import PendingDecision

    assert resumed.pending_decision is PendingDecision.CONTEXT_LOCK
    assert resumed.context_record.cloud_provider == ""
    assert resumed.context_record.budget == ""
    # The required, already-answered field was never touched by this round.
    assert resumed.context_record.non_functional_requirements == [
        "Substantially higher peak traffic than today; exact target unknown"
    ]
    assert any("Approve and continue" in (b.label or "") for b in at.button)
    assert not any(b.label == "Skip optional questions" for b in at.button)


# ── 6. batched feedback across views — one bundle, one round ───────────────
#
# The workspace split the two feedback boxes into different views; these
# tests pin the batching that must NOT be lost: both kinds stage into ONE
# pending bundle, the bundle survives navigation, and a single sidebar
# action submits everything as ONE user_feedback.submit_feedback call.

REQ_TEXT = "peak load is 500k users, not 50k"
DESIGN_TEXT = "use SQS instead of Kafka"

# The box widget keys rotate on every staging (by design — see
# `ui_sections._feedback_key`), so tests locate a box by its stable label
# rather than a key that changes mid-flow.
_BOX_LABELS = {"requirements": "Correct the requirements",
               "design": "Direct the architect"}


def _stage(at: AppTest, kind: str, text: str) -> AppTest:
    """Type into one view's feedback box and click its add button."""

    box = next(
        t for t in at.text_area
        if (t.label or "") == _BOX_LABELS[kind] and not t.disabled
    )
    box.set_value(text)
    at.run()
    add = next(
        b for b in at.button if b.label == "Add to pending feedback"
    )
    add.click()
    at.run()
    assert not at.exception
    return at


def _stage_both() -> AppTest:
    """Context feedback, then Design feedback, staged into one bundle."""

    at = _select_view(_finished_app(), "Context")
    _stage(at, "requirements", REQ_TEXT)
    _select_view(at, "Architecture")
    _stage(at, "design", DESIGN_TEXT)
    return at


def _sidebar_texts(at: AppTest) -> str:
    """Everything the sidebar is currently saying, previews included."""

    parts = [m.value for m in at.sidebar.markdown]
    parts += [c.value for c in at.sidebar.caption]
    return "\n".join(parts)


def _pending_button(at: AppTest, label: str):
    return next(
        (b for b in at.sidebar.button if b.label == label), None
    )


def test_context_feedback_stages_without_triggering_refinement(
    no_pipeline_actions,
):
    at = _select_view(_finished_app(), "Context")
    _stage(at, "requirements", REQ_TEXT)

    # Staged — the sidebar panel appeared with the Context preview...
    assert "Pending feedback round" in _sidebar_texts(at)
    assert "Context" in _sidebar_texts(at)
    # ...and no backend action fired (the fixture booby-traps them all).
    assert REQ_TEXT in _sidebar_texts(at)


def test_design_feedback_stages_into_the_same_bundle():
    at = _stage_both()

    texts = _sidebar_texts(at)
    assert "Context" in texts and "Architecture" in texts
    assert REQ_TEXT in texts and DESIGN_TEXT in texts


def test_navigation_between_views_preserves_pending_feedback():
    at = _stage_both()
    for view in ("Knowledge", "Overview", "Review", "Architecture", "Context"):
        _select_view(at, view)
        texts = _sidebar_texts(at)
        assert REQ_TEXT in texts, view
        assert DESIGN_TEXT in texts, view


def test_one_submit_sends_both_kinds_in_a_single_backend_call(monkeypatch):
    calls: list[dict] = []

    real = __import__(
        "pipeline.user_feedback", fromlist=["submit_feedback"]
    ).submit_feedback

    def counting(state, **kwargs):
        calls.append(kwargs)
        return real(state, **kwargs)  # real semantics: charges one round

    monkeypatch.setattr(
        "pipeline.user_feedback.submit_feedback", counting
    )

    # The pipeline itself is stubbed: it "finishes" the round offline by
    # handing back the state at DONE, exactly where it started.
    from pipeline.state import Stage

    def fake_stream(state, max_steps=20):
        state.stage = Stage.DONE
        yield state

    monkeypatch.setattr(
        "pipeline.orchestrator.run_pipeline_streaming", fake_stream
    )

    at = _stage_both()
    _pending_button(at, "Submit feedback round").click()
    at.run()
    assert not at.exception

    # ONE call, carrying both kinds — one refinement round for the bundle.
    assert len(calls) == 1
    assert calls[0]["requirements"] == REQ_TEXT
    assert calls[0]["design"] == DESIGN_TEXT
    # The round really was charged exactly once.
    assert at.session_state["state"].user_rounds == 1


def test_pending_clears_after_successful_submit(monkeypatch):
    monkeypatch.setattr(
        "pipeline.user_feedback.submit_feedback",
        lambda state, **_k: (True, ""),  # accepted, nothing else recorded
    )

    from pipeline.state import Stage

    def fake_stream(state, max_steps=20):
        state.stage = Stage.DONE
        yield state

    monkeypatch.setattr(
        "pipeline.orchestrator.run_pipeline_streaming", fake_stream
    )

    at = _stage_both()
    assert _pending_button(at, "Submit feedback round") is not None
    _pending_button(at, "Submit feedback round").click()
    at.run()
    # `_run` ends in st.rerun(); AppTest only re-executes the script on the
    # NEXT at.run(), so one more pass is needed to see the post-round tree.
    at.run()
    assert not at.exception

    # The bundle is gone: the panel (and its submit action) disappeared,
    # and the session's pending store is empty again.
    assert _pending_button(at, "Submit feedback round") is None
    bundle = (
        at.session_state["pending_feedback"]
        if "pending_feedback" in at.session_state
        else {}
    )
    assert not any(str(v).strip() for v in bundle.values())


def test_failed_submit_does_not_lose_pending_feedback(monkeypatch):
    monkeypatch.setattr(
        "pipeline.user_feedback.submit_feedback",
        lambda state, **_k: (False, "user rounds"),  # refused
    )

    at = _stage_both()
    _pending_button(at, "Submit feedback round").click()
    at.run()
    assert not at.exception

    # The refusal is reported, and BOTH staged texts survive it.
    assert any("Out of refinement rounds" in w.value for w in at.warning)
    texts = _sidebar_texts(at)
    assert REQ_TEXT in texts and DESIGN_TEXT in texts
    assert _pending_button(at, "Submit feedback round") is not None


def test_design_question_stays_out_of_the_feedback_bundle(monkeypatch):
    asked: list[tuple] = []

    def fake_ask(state, question, subject="context_record"):
        asked.append((question, subject))
        return None

    monkeypatch.setattr(
        "pipeline.agents.clarifier.ask_advisor", fake_ask
    )
    monkeypatch.setattr("pipeline.persistence.save_state", lambda *_a: None)
    # If the question were routed as feedback, this would explode the test.
    monkeypatch.setattr(
        "pipeline.user_feedback.submit_feedback",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("question must not submit feedback")
        ),
    )

    at = _stage_both()  # pending exists; the question must not touch it
    at.text_input(key="design_ask_question").set_value(
        "why did it pick this pattern?"
    )
    at.run()
    next(b for b in at.button if b.label == "Ask").click()
    at.run()
    assert not at.exception

    assert asked == [("why did it pick this pattern?", "design")]
    texts = _sidebar_texts(at)
    assert REQ_TEXT in texts and DESIGN_TEXT in texts  # bundle untouched


def test_sign_off_still_works_alongside_pending_feedback(monkeypatch):
    accepted: list = []
    monkeypatch.setattr(
        "pipeline.sign_off.accept_design",
        lambda state, **_k: accepted.append(state) or None,
    )
    monkeypatch.setattr("pipeline.persistence.save_state", lambda *_a: None)

    at = _stage_both()  # pending feedback exists
    _select_view(at, "Overview")  # the sign-off action lives here
    next(
        b for b in at.button if "Accept this design" in b.label
    ).click()
    at.run()
    assert not at.exception

    assert len(accepted) == 1  # the sign-off path itself is unchanged
    # And it does not spend or clear the bundle — that is the discard
    # button's job, not the sign-off's.
    assert REQ_TEXT in _sidebar_texts(at)


def test_chat_stays_inert_with_pending_feedback(no_pipeline_actions, monkeypatch):
    at = _stage_both()
    # History has its own test file (test_ui_history.py); the Chat view
    # stays inert here even with a pending bundle staged — rendering it
    # must not answer anything (no LLM call) and must not touch the bundle.
    monkeypatch.setattr(
        "pipeline.llm.llm_call",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("LLM call triggered by rendering")
        ),
    )
    _select_view(at, "Chat")
    # The view renders (its input is present) with no transcript yet.
    assert len(at.chat_input) == 1
    assert not _main_expander_labels(at)


# ── 7. visual polish: header, no transcript, inert rendering ───────────────


def test_workspace_header_anchors_every_view():
    at = _finished_app()
    for view in WORKSPACE_VIEWS:
        _select_view(at, view)
        header = next(
            (m for m in at.markdown if "ws-header" in m.value), None
        )
        assert header is not None, view
        assert f">{view}<" in header.value          # the view name
        assert "Seasonal Shop" in header.value      # the project name
        assert "PASS" in header.value               # the verdict chip


def test_finished_screen_is_not_a_chat_transcript():
    at = _finished_app()
    # The old screen wrapped the workspace in chat bubbles and replayed the
    # whole Q&A above every view. The workspace renders none of that.
    assert len(at.chat_message) == 0
    assert not any(
        "eu-central-1 (Frankfurt)" in m.value for m in at.markdown
    )  # the Q&A lives ONLY in the Context view's collapsed history


def test_clarification_history_is_secondary_but_complete():
    at = _select_view(_finished_app(), "Context")

    assert _has_expander(at, "Clarification history")
    # The history is DRAWN FROM state, never moved: the run's Q&A is exactly
    # what the demo state built it from, still on the state afterwards.
    answers = at.session_state["state"].clarification_answers
    assert "Which AWS region must EU order data stay in?" in answers
    assert answers["Which AWS region must EU order data stay in?"] == (
        "eu-central-1 (Frankfurt)."
    )


def test_rendering_every_view_triggers_no_pipeline_action(
    no_pipeline_actions,
):
    at = _finished_app()
    for view in WORKSPACE_VIEWS:
        _select_view(at, view)  # asserts no exception internally

