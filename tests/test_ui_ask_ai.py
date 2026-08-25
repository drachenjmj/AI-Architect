"""test_ui_ask_ai.py — UI wiring for the context-aware "Ask AI" discussion
buttons (Kush integration): required clarification questions, the optional
round, the Context Record approval screen, and the pre-run intake form.

Offline throughout: `pipeline.agents.clarifier.discuss_field` is stubbed
with a fake that records every call and returns a canned structured
response. Real pipeline/persistence paths are booby-trapped so a bug that
accidentally triggers a run, a save, or a stage transition from an Ask AI
interaction fails loudly instead of passing quietly. NO LIVE Gemini/Claude
calls occur anywhere in this file.

Never `import ui` (module-level st calls — see test_ui_workspace.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pipeline.agents import clarifier as clar
from pipeline.state import ContextRecord, PendingDecision, Stage, new_run

_UI = str(Path(__file__).resolve().parents[1] / "ui.py")


class _FakeDiscuss:
    """Records every `discuss_field` call; returns a canned structured
    response (optionally carrying a `suggested_answer`)."""

    def __init__(self, reply: str = "Here is my advice.", suggested_answer: str = ""):
        self.reply = reply
        self.suggested_answer = suggested_answer
        self.calls: list[dict] = []

    def __call__(self, state, **kwargs):
        self.calls.append({"state": state, **kwargs})
        return clar.FieldDiscussionResponse(reply=self.reply, suggested_answer=self.suggested_answer)


@pytest.fixture()
def fake_discuss(monkeypatch):
    fake = _FakeDiscuss()
    monkeypatch.setattr(clar, "discuss_field", fake)
    return fake


@pytest.fixture()
def fake_discuss_with_suggestion(monkeypatch):
    fake = _FakeDiscuss(suggested_answer="Keep this qualitative for now.")
    monkeypatch.setattr(clar, "discuss_field", fake)
    return fake


@pytest.fixture()
def booby_trap(monkeypatch):
    """Real pipeline/persistence/state-transition paths an Ask AI bug must
    never touch (requirement: no automatic Submit/Approve, no generation)."""

    def _boom(*args, **kwargs):
        raise AssertionError("a pipeline/persistence path fired from Ask AI")

    monkeypatch.setattr("pipeline.orchestrator.run_pipeline_streaming", _boom)
    monkeypatch.setattr("pipeline.persistence.save_state", _boom)
    monkeypatch.setattr("pipeline.agents.clarifier.ask_advisor", _boom)
    monkeypatch.setattr("pipeline.agents.clarifier.accept_context_lock", _boom)
    monkeypatch.setattr("pipeline.agents.clarifier.submit_context_edits", _boom)
    monkeypatch.setattr("pipeline.agents.clarifier.resolve_optional_context", _boom)
    return _boom


def _clarification_state(*, questions: list[str]):
    state = new_run(
        "A monolithic shop with peak traffic issues.", require_context_approval=True
    )
    state.stage = Stage.AWAITING_HUMAN
    state.pending_decision = PendingDecision.CLARIFICATION
    state.clarifying_questions = list(questions)
    return state


def _context_lock_state(**record_fields):
    state = new_run(
        "A monolithic shop with peak traffic issues.", require_context_approval=True
    )
    state.stage = Stage.AWAITING_HUMAN
    state.pending_decision = PendingDecision.CONTEXT_LOCK
    state.context_record = ContextRecord(project_name="Shop", **record_fields)
    return state


def _app(state) -> AppTest:
    at = AppTest.from_file(_UI, default_timeout=30)
    at.session_state["state"] = state
    at.run()
    assert not at.exception
    return at


def _ask_ai_widgets(at: AppTest, needle: str):
    """(msg_input, send_button, use_button_or_None) whose keys contain `needle`."""
    msg = next(t for t in at.text_input if t.key and "ask_ai_msg_" in t.key and needle in t.key)
    send = next(b for b in at.button if b.key and "ask_ai_send_" in b.key and needle in b.key)
    use = next(
        (b for b in at.button if b.key and "ask_ai_use_" in b.key and needle in b.key), None
    )
    return msg, send, use


def _send(at: AppTest, needle: str, message: str) -> AppTest:
    msg, _send_btn, _use = _ask_ai_widgets(at, needle)
    msg.set_value(message)
    at.run()
    _msg, send_btn, _use = _ask_ai_widgets(at, needle)
    send_btn.click()
    at.run()
    assert not at.exception
    return at


def _document_order(at: AppTest) -> list[tuple[str, str]]:
    """Every (element type, key-or-label) pair in `at.main`, in actual paint
    order — a DFS over the real render tree, not the type-grouped `at.button`
    / `at.text_input` collections. This is what lets a test tell "input
    directly follows its own label" from "every input, then every label",
    which the type-grouped collections cannot: they lose position entirely.
    """
    order: list[tuple[str, str]] = []

    def walk(node) -> None:
        kind = type(node).__name__
        # Concatenate every identifying bit (key, label, raw value) rather
        # than picking one: a plain `st.button`'s `.key` is a short internal
        # id ("clarify_submit") that tells a reader nothing, so a needle
        # search has to be able to match on the visible `.label` too.
        bits = [
            str(v)
            for v in (
                getattr(node, "key", None),
                getattr(node, "label", None),
                getattr(node, "value", None),
            )
            if v
        ]
        if kind not in ("Block", "Column", "SpecialBlock", "ChatMessage"):
            order.append((kind, " | ".join(bits)))
        children = getattr(node, "children", None)
        if children:
            for child in (children.values() if isinstance(children, dict) else children):
                walk(child)

    walk(at.main)
    return order


def _index_of(order: list[tuple[str, str]], kind: str, needle: str) -> int:
    return next(i for i, (k, ident) in enumerate(order) if k == kind and needle in ident)


# ── 1-5: layout — label + Ask AI + input stay one interleaved unit ────────


def test_required_clarification_layout_interleaves_question_ask_ai_and_input(
    booby_trap, fake_discuss
):
    """Regression for the reported layout bug: Ask AI moving the label out of
    `st.form` made every ANSWER INPUT render as one block first, followed by
    every LABEL afterward. Each question's label, its Ask AI popover trigger,
    and its own answer input must appear as one unit, in question order, with
    "Submit answers" last."""
    state = _clarification_state(
        questions=["What are the core capabilities?", "What scale targets?"]
    )
    at = _app(state)
    order = _document_order(at)

    q1_label = _index_of(order, "Markdown", "What are the core capabilities?")
    q1_ask_ai = _index_of(order, "TextInput", "ask_ai_msg_")  # first match is Q1's
    q1_input = _index_of(order, "TextInput", "answer_0")
    q2_label = _index_of(order, "Markdown", "What scale targets?")
    q2_input = _index_of(order, "TextInput", "answer_1")
    submit = _index_of(order, "Button", "Submit answers")

    assert q1_label < q1_ask_ai < q1_input < q2_label < q2_input < submit


def test_opening_and_using_ask_ai_does_not_reorder_the_questions(booby_trap, fake_discuss):
    """Doc §12 item 4: sending a message (and getting a reply back into the
    popover's transcript) must not displace the surrounding question blocks."""
    state = _clarification_state(questions=["Question A?", "Question B?"])
    at = _app(state)
    at = _send(at, "Question A", "About A")

    order = _document_order(at)
    a_label = _index_of(order, "Markdown", "Question A?")
    a_input = _index_of(order, "TextInput", "answer_0")
    b_label = _index_of(order, "Markdown", "Question B?")
    b_input = _index_of(order, "TextInput", "answer_1")
    submit = _index_of(order, "Button", "Submit answers")

    assert a_label < a_input < b_label < b_input < submit


def test_context_approval_layout_interleaves_label_and_input(booby_trap, fake_discuss):
    state = _context_lock_state(business_goal="Survive peak sales.")
    at = _app(state)
    order = _document_order(at)

    goal_label = _index_of(order, "Markdown", "Business goal")
    goal_input = _index_of(order, "TextInput", "cg_field_business_goal_")
    problem_label = _index_of(order, "Markdown", "Problem statement")
    problem_input = _index_of(order, "TextInput", "cg_field_problem_statement_")

    assert goal_label < goal_input < problem_label < problem_input


def test_optional_context_layout_interleaves_label_and_input(booby_trap, fake_discuss):
    state = _context_lock_state()  # no NFR/cloud/budget/compliance set yet
    state.pending_decision = PendingDecision.OPTIONAL_CONTEXT
    at = _app(state)
    order = _document_order(at)

    cloud_input = _index_of(order, "TextInput", "oc_field_cloud_provider_")
    budget_input = _index_of(order, "TextInput", "oc_field_budget_")
    skip = _index_of(order, "Button", "Skip optional questions")

    assert cloud_input < budget_input < skip


# ── 13/14: every screen exposes Ask AI, generically ───────────────────────


def test_required_clarification_questions_expose_ask_ai(booby_trap, fake_discuss):
    """Generic: works for whatever questions the Clarifier produced this
    run — not hard-coded to any specific e-commerce question text."""
    state = _clarification_state(
        questions=["What scale, availability, or performance targets?", "What cloud provider?"]
    )
    at = _app(state)

    msg_inputs = [t.key for t in at.text_input if t.key and t.key.startswith("ask_ai_msg_")]
    assert len(msg_inputs) == 2


def test_optional_context_round_exposes_ask_ai(booby_trap, fake_discuss):
    state = _context_lock_state()  # no NFR/cloud/budget/compliance set yet
    state.pending_decision = PendingDecision.OPTIONAL_CONTEXT
    at = _app(state)

    msg_inputs = [t.key for t in at.text_input if t.key and t.key.startswith("ask_ai_msg_")]
    assert msg_inputs


def test_context_approval_screen_exposes_ask_ai_for_every_editable_field(booby_trap, fake_discuss):
    state = _context_lock_state(business_goal="Survive peak sales.")
    at = _app(state)

    msg_inputs = [t.key for t in at.text_input if t.key and t.key.startswith("ask_ai_msg_")]
    assert len(msg_inputs) == len(clar.EDITABLE_RECORD_FIELDS)


def test_intake_form_exposes_ask_ai_for_system_description_only():
    """Not the repo URL — see ui.py's comment on why."""
    at = AppTest.from_file(_UI, default_timeout=30)
    at.run()
    assert not at.exception

    msg_inputs = [t.key for t in at.text_input if t.key and t.key.startswith("ask_ai_msg_")]
    assert len(msg_inputs) == 1
    # Scoped under the pre-run draft id (see field_discussion.pre_run_scope),
    # not a bare fixed sentinel — the field key is still the stable part.
    assert msg_inputs[0].startswith("ask_ai_msg___pre_run__:")
    assert msg_inputs[0].endswith("_start.system_description")


# ── 6/7: field and run isolation ──────────────────────────────────────────


def test_two_different_fields_get_isolated_discussion_histories(booby_trap, fake_discuss):
    state = _clarification_state(questions=["Question A?", "Question B?"])
    at = _app(state)

    at = _send(at, "Question A", "About A")
    assert fake_discuss.calls[-1]["field_label"] == "Question A?"
    assert fake_discuss.calls[-1]["history"] == []

    at = _send(at, "Question B", "About B")
    assert fake_discuss.calls[-1]["field_label"] == "Question B?"
    assert fake_discuss.calls[-1]["history"] == []  # B never sees A's turn

    # A's OWN second turn correctly carries A's own prior history.
    at = _send(at, "Question A", "Follow-up on A")
    assert fake_discuss.calls[-1]["field_label"] == "Question A?"
    assert fake_discuss.calls[-1]["history"] == [("About A", fake_discuss.reply)]


def test_two_different_runs_get_isolated_discussion_histories(booby_trap, fake_discuss):
    """Same session, same question TEXT, two different runs — the realistic
    leak vector is switching the active run mid-session, not two separate
    browser sessions (which would never share state at all)."""
    state_a = _clarification_state(questions=["Q?"])
    at = _app(state_a)
    at = _send(at, "Q", "message for run A")
    assert fake_discuss.calls[-1]["history"] == []

    state_b = _clarification_state(questions=["Q?"])
    at.session_state["state"] = state_b
    at.run()
    at = _send(at, "Q", "message for run B")
    assert fake_discuss.calls[-1]["history"] == []  # NOT run A's prior turn


# ── 8/9: never mutates the field on its own ────────────────────────────────


def test_rendering_ask_ai_alone_never_calls_the_model_or_touches_the_field(booby_trap, fake_discuss):
    state = _clarification_state(questions=["Q?"])
    at = _app(state)

    answer_widget = next(t for t in at.text_input if t.key == "answer_0")
    assert not answer_widget.value
    assert fake_discuss.calls == []


def test_ai_response_alone_does_not_modify_the_field_without_explicit_apply(
    booby_trap, fake_discuss
):
    state = _clarification_state(questions=["Q?"])
    at = _app(state)
    at = _send(at, "Q", "what do you recommend?")

    answer_widget = next(t for t in at.text_input if t.key == "answer_0")
    assert not answer_widget.value
    assert at.session_state["state"].clarification_answers == {}


# ── 10: explicit "Use suggestion" updates only the intended field ─────────


def test_use_suggestion_updates_only_the_intended_field(booby_trap, fake_discuss_with_suggestion):
    state = _clarification_state(questions=["Question A?", "Question B?"])
    at = _app(state)
    at = _send(at, "Question A", "recommend?")

    _msg, _send_btn, use_a = _ask_ai_widgets(at, "Question A")
    assert use_a is not None
    use_a.click()
    at.run()
    assert not at.exception

    a_widget = next(t for t in at.text_input if t.key == "answer_0")
    b_widget = next(t for t in at.text_input if t.key == "answer_1")
    assert a_widget.value == "Keep this qualitative for now."
    assert not b_widget.value


def test_use_suggestion_on_context_approval_updates_only_that_field(
    booby_trap, fake_discuss_with_suggestion
):
    state = _context_lock_state(business_goal="Survive peak sales.")
    at = _app(state)
    at = _send(at, "context.budget", "recommend?")

    _msg, _send_btn, use_btn = _ask_ai_widgets(at, "context.budget")
    use_btn.click()
    at.run()

    budget_widget = next(t for t in at.text_input if t.key and t.key.startswith("cg_field_budget_"))
    goal_widget = next(
        t for t in at.text_input if t.key and t.key.startswith("cg_field_business_goal_")
    )
    assert budget_widget.value == "Keep this qualitative for now."
    assert goal_widget.value == "Survive peak sales."  # untouched


# ── 11: no automatic Submit/Approve from a discussion ──────────────────────


def test_no_automatic_submission_from_a_clarification_discussion(booby_trap, fake_discuss_with_suggestion):
    state = _clarification_state(questions=["Q?"])
    at = _app(state)
    at = _send(at, "Q", "recommend?")
    _msg, _send_btn, use_btn = _ask_ai_widgets(at, "Q")
    use_btn.click()
    at.run()

    assert at.session_state["state"].stage is Stage.AWAITING_HUMAN
    assert at.session_state["state"].pending_decision is PendingDecision.CLARIFICATION
    assert at.session_state["state"].clarification_answers == {}


def test_no_automatic_approval_from_a_context_record_discussion(booby_trap, fake_discuss_with_suggestion):
    state = _context_lock_state(business_goal="Survive peak sales.")
    at = _app(state)
    at = _send(at, "context.budget", "recommend?")
    _msg, _send_btn, use_btn = _ask_ai_widgets(at, "context.budget")
    use_btn.click()
    at.run()

    assert at.session_state["state"].pending_decision is PendingDecision.CONTEXT_LOCK
    assert at.session_state["state"].context_record.budget == ""  # never auto-applied to the record


# ── 15: existing behavior is unaffected when Ask AI is never opened ───────


def test_existing_clarification_submit_still_works_without_ask_ai(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("discuss_field must not fire when Ask AI was never used")

    monkeypatch.setattr(clar, "discuss_field", _boom)

    ran = {}

    def fake_stream(state):
        ran["called"] = True
        yield state

    monkeypatch.setattr("pipeline.orchestrator.run_pipeline_streaming", fake_stream)

    state = _clarification_state(questions=["Q1?", "Q2?"])
    at = _app(state)
    a0 = next(t for t in at.text_input if t.key == "answer_0")
    a0.set_value("answer one")
    at.run()
    submit = next(b for b in at.button if b.label == "Submit answers")
    submit.click()
    at.run()

    assert ran.get("called") is True


# ── pre-run intake: apply suggestion, no run created ──────────────────────


def test_intake_use_suggestion_updates_the_draft_description_only(fake_discuss_with_suggestion):
    at = AppTest.from_file(_UI, default_timeout=30)
    at.run()

    desc = at.text_area(key="intake_system_description")
    desc.set_value("Monolithic shop with peak traffic issues.")
    at.run()
    at = _send(at, "start.system_description", "what do you recommend?")

    _msg, _send_btn, use_btn = _ask_ai_widgets(at, "start.system_description")
    use_btn.click()
    at.run()

    desc_widget = at.text_area(key="intake_system_description")
    assert desc_widget.value == "Keep this qualitative for now."
    assert "state" not in at.session_state or at.session_state["state"] is None


def test_intake_ask_ai_never_starts_a_run(monkeypatch, fake_discuss):
    def _boom(*args, **kwargs):
        raise AssertionError("a run started from the intake Ask AI discussion")

    monkeypatch.setattr("pipeline.orchestrator.run_pipeline_streaming", _boom)

    at = AppTest.from_file(_UI, default_timeout=30)
    at.run()
    at.text_area(key="intake_system_description").set_value("A shop with peak issues.")
    at.run()
    at = _send(at, "start.system_description", "thoughts?")

    assert "state" not in at.session_state or at.session_state["state"] is None
    assert fake_discuss.calls[0]["state"] is None  # no ArchitectState was ever created


# ── 6/7 (doc §3): the visible DRAFT reaches Ask AI, not a stale value ─────
# `render_ask_ai` is a plain (non-form) widget's sibling now — see the note
# on why `st.form` was removed from every screen that pairs it with a field
# in ui.py / ui_sections.py. These pin the exact acceptance case: a draft
# typed but never submitted must still reach `discuss_field`.


def test_ask_ai_sees_the_unsubmitted_clarification_draft(booby_trap, fake_discuss):
    """The literal acceptance case: type an answer, do NOT click Submit, ask
    Ask AI about the SAME field — the draft must be in the prompt context."""
    state = _clarification_state(
        questions=["What scale, availability, or performance targets does this need to meet?"]
    )
    at = _app(state)

    a0 = next(t for t in at.text_input if t.key == "answer_0")
    a0.set_value("Handle much higher peak traffic")
    at.run()

    at = _send(at, "clarification::", "Is this enough?")

    assert fake_discuss.calls[-1]["current_draft_answer"] == "Handle much higher peak traffic"
    # Never submitted — the veto boundary is still Submit answers alone.
    assert at.session_state["state"].clarification_answers == {}


def test_ask_ai_sees_other_unsubmitted_clarification_drafts(booby_trap, fake_discuss):
    """Doc §12 item 8: other currently-visible, still-unsubmitted answers on
    the SAME screen must be included as known context."""
    state = _clarification_state(questions=["Question A?", "Question B?"])
    at = _app(state)

    a0 = next(t for t in at.text_input if t.key == "answer_0")
    a0.set_value("Draft answer for A")
    at.run()

    at = _send(at, "Question B", "what about B?")

    assert fake_discuss.calls[-1]["known_context"]["Question A?"] == "Draft answer for A"


def test_ask_ai_sees_the_unsubmitted_context_approval_edit(booby_trap, fake_discuss):
    """Doc §9: edit one field, then ask about a DIFFERENT field before
    Approve — the edited (unsaved) value must be visible as known context."""
    state = _context_lock_state(business_goal="Old goal.")
    at = _app(state)

    goal_widget = next(
        t for t in at.text_input if t.key and t.key.startswith("cg_field_business_goal_")
    )
    goal_widget.set_value("Survive peak sales, edited but not yet saved.")
    at.run()

    at = _send(at, "context.problem_statement", "what belongs here?")

    assert (
        fake_discuss.calls[-1]["known_context"]["business_goal"]
        == "Survive peak sales, edited but not yet saved."
    )
    # Not committed to the record — only Save/Approve does that.
    assert at.session_state["state"].context_record.business_goal == "Old goal."


def test_ask_ai_sees_the_unsubmitted_optional_context_draft(booby_trap, fake_discuss):
    state = _context_lock_state()
    state.pending_decision = PendingDecision.OPTIONAL_CONTEXT
    at = _app(state)

    cloud_widget = next(
        t for t in at.text_input if t.key and t.key.startswith("oc_field_cloud_provider_")
    )
    cloud_widget.set_value("AWS, no strong preference")
    at.run()

    at = _send(at, "context.budget", "what should I put here?")

    assert (
        fake_discuss.calls[-1]["known_context"]["cloud_provider"] == "AWS, no strong preference"
    )


# ── doc §5: pre-run draft discussion isolation ─────────────────────────────


def test_pre_run_draft_scope_is_stable_across_an_ordinary_rerun(fake_discuss):
    """The SAME unsubmitted draft keeps its discussion across a plain rerun
    (e.g. typing in an unrelated field) — no reset just from redrawing."""
    at = AppTest.from_file(_UI, default_timeout=30)
    at.run()
    at.text_area(key="intake_system_description").set_value("A shop with peak issues.")
    at.run()
    at = _send(at, "start.system_description", "thoughts?")
    first_key = next(
        t.key for t in at.text_input if t.key and t.key.startswith("ask_ai_msg_")
    )

    at.text_input(key="intake_repo_url").set_value("")  # an unrelated, no-op rerun
    at.run()
    second_key = next(
        t.key for t in at.text_input if t.key and t.key.startswith("ask_ai_msg_")
    )

    assert first_key == second_key
    # And the transcript from the first turn is still there.
    texts = [cm.markdown[0].value for cm in at.chat_message if cm.markdown]
    assert "thoughts?" in texts


def test_pre_run_draft_scope_resets_after_new_run_clears_the_session(fake_discuss):
    """Doc §5's leak scenario: clicking "New run" (`st.session_state.clear()`
    — see ui.py) must hand the NEXT pre-run draft a fresh discussion scope,
    with no memory of what was discussed about the abandoned one. Reaching
    "New run" requires a run to exist first (the button is gated on
    `state is not None`), so this simulates a run existing the cheap way —
    injecting a state directly — rather than driving the real pipeline."""
    at = AppTest.from_file(_UI, default_timeout=30)
    at.run()
    at.text_area(key="intake_system_description").set_value("Draft A: a payment system.")
    at.run()
    at = _send(at, "start.system_description", "thoughts on draft A?")
    old_key = next(t.key for t in at.text_input if t.key and t.key.startswith("ask_ai_msg_"))
    assert "field_discussions" in at.session_state

    at.session_state["state"] = new_run("Draft A: a payment system.")
    at.run()
    assert not at.exception
    next(b for b in at.sidebar.button if "New run" in (b.label or "")).click()
    at.run()
    assert not at.exception
    # The clear wipes the store; the very next render immediately recreates
    # it EMPTY (`field_discussion.history_for`'s `setdefault`), so what
    # proves the clear worked is that draft A's OWN scope is gone, not that
    # the top-level key is absent.
    assert old_key.split("ask_ai_msg_", 1)[1] not in at.session_state["field_discussions"]

    at.text_area(key="intake_system_description").set_value("Draft B: a logistics platform.")
    at.run()
    at = _send(at, "start.system_description", "thoughts on draft B?")

    new_key = next(t.key for t in at.text_input if t.key and t.key.startswith("ask_ai_msg_"))
    assert new_key != old_key
    # Nothing from draft A's discussion leaked into draft B's transcript.
    assert fake_discuss.calls[-1]["history"] == []


# ── doc §6: dynamic clarification-question identity is duplicate-safe ─────


def test_critical_clarification_questions_have_no_duplicate_text():
    """`clarification::<question text>` is only a safe field identity if no
    two critical fields can ever produce the same question text — pin that
    invariant directly, since it is what the field-key scheme relies on
    (see clarifier._CRITICAL_SLOT_QUESTIONS and CRITICAL_RECORD_FIELDS)."""
    texts = [
        clar.slot_question(field).question for field in clar.CRITICAL_RECORD_FIELDS
    ]
    assert len(texts) == len(set(texts))
