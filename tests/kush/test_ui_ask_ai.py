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

_UI = str(Path(__file__).resolve().parents[2] / "ui.py")


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
    assert msg_inputs == ["ask_ai_msg___pre_run___start.system_description"]


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
