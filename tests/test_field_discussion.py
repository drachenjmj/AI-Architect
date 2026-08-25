"""test_field_discussion.py — context-aware "Ask AI" field discussions
(Kush integration): the prompt/context builder and the `discuss_field` LLM
call shape in pipeline/agents/clarifier.py.

Session-state history keying/isolation and UI wiring are covered separately
in tests/test_ui_ask_ai.py (AppTest is required for real session_state
behavior — bare-mode `st.session_state` outside a running app does not
reliably function, so it is not unit-tested directly here).

All offline: `pipeline.agents.clarifier.llm_call` is stubbed. No live
Gemini/Claude calls anywhere in this file.
"""

from __future__ import annotations

import test_clarifier as tc
from pipeline.agents import clarifier as clar
from pipeline.state import RepoMeta, RepoRepresentation, new_run


# ── the context builder: what goes into the prompt ───────────────────────


def test_prompt_includes_the_exact_field_being_discussed():
    prompt = clar._build_field_discussion_prompt(
        raw_prompt="Modernize a shop.",
        repo_representation=None,
        known_context={},
        field_label="Scale / availability / performance targets",
        field_purpose="Scale and availability targets materially change the architecture.",
        current_draft_answer="",
        history=(),
        user_message="What do you recommend?",
    )
    assert "Scale / availability / performance targets" in prompt
    assert "Scale and availability targets materially change the architecture." in prompt


def test_prompt_includes_the_current_draft_answer():
    prompt = clar._build_field_discussion_prompt(
        raw_prompt="x",
        repo_representation=None,
        known_context={},
        field_label="Budget",
        field_purpose="",
        current_draft_answer="Medium, cost-conscious",
        history=(),
        user_message="is this reasonable?",
    )
    assert "Medium, cost-conscious" in prompt


def test_prompt_includes_other_currently_known_context_values():
    prompt = clar._build_field_discussion_prompt(
        raw_prompt="x",
        repo_representation=None,
        known_context={"business_goal": "Survive peak sales", "cloud_provider": "AWS"},
        field_label="Budget",
        field_purpose="",
        current_draft_answer="",
        history=(),
        user_message="q",
    )
    assert "Survive peak sales" in prompt
    assert "AWS" in prompt


def test_prompt_includes_repo_facts_when_available():
    repo = RepoRepresentation(meta=RepoMeta(url="https://example.invalid/shop"))
    repo.behavior.overview = "A Django monolith handling orders."
    prompt = clar._build_field_discussion_prompt(
        raw_prompt="x",
        repo_representation=repo,
        known_context={},
        field_label="Existing systems",
        field_purpose="",
        current_draft_answer="",
        history=(),
        user_message="q",
    )
    assert "<repository_context>" in prompt
    assert "A Django monolith handling orders." in prompt


def test_prompt_omits_repo_block_rather_than_inventing_when_unavailable():
    prompt = clar._build_field_discussion_prompt(
        raw_prompt="x",
        repo_representation=None,
        known_context={},
        field_label="Existing systems",
        field_purpose="",
        current_draft_answer="",
        history=(),
        user_message="q",
    )
    assert "<repository_context>" not in prompt


def test_prompt_includes_prior_discussion_turns_for_this_field():
    prompt = clar._build_field_discussion_prompt(
        raw_prompt="x",
        repo_representation=None,
        known_context={},
        field_label="Budget",
        field_purpose="",
        current_draft_answer="",
        history=[("first question", "first answer")],
        user_message="follow-up",
    )
    assert "first question" in prompt
    assert "first answer" in prompt


def test_prompt_carries_the_users_current_message():
    prompt = clar._build_field_discussion_prompt(
        raw_prompt="x",
        repo_representation=None,
        known_context={},
        field_label="Budget",
        field_purpose="",
        current_draft_answer="",
        history=(),
        user_message="What do you recommend?",
    )
    assert "What do you recommend?" in prompt


# ── discuss_field: the LLM call shape ─────────────────────────────────────


class _FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def __call__(self, state, prompt, *, system="", model="", response_schema=None):
        self.calls.append(
            {
                "state": state,
                "prompt": prompt,
                "system": system,
                "model": model,
                "response_schema": response_schema,
            }
        )
        return self.response, tc.fake_usage()


def test_discuss_field_uses_the_structured_response_schema(monkeypatch):
    fake = _FakeLLM(clar.FieldDiscussionResponse(reply="Keep it qualitative.", suggested_answer=""))
    monkeypatch.setattr(clar, "llm_call", fake)
    state = new_run("Modernize a shop.")

    response = clar.discuss_field(
        state,
        raw_prompt="Modernize a shop.",
        repo_representation=None,
        known_context={},
        field_label="Scale / availability / performance targets",
        user_message="What do you recommend?",
    )

    assert response.reply == "Keep it qualitative."
    assert fake.calls[0]["response_schema"] is clar.FieldDiscussionResponse
    assert fake.calls[0]["model"] == clar.ADVISOR_MODEL  # the existing cheap default
    assert fake.calls[0]["state"] is state


def test_discuss_field_works_before_any_run_exists(monkeypatch):
    """Pre-run intake form: state is None. `discuss_field` must still call
    the model (via a throwaway probe state) and never raise for that
    reason alone."""
    fake = _FakeLLM(clar.FieldDiscussionResponse(reply="ok"))
    monkeypatch.setattr(clar, "llm_call", fake)

    response = clar.discuss_field(
        None,
        raw_prompt="A monolithic shop with peak traffic issues.",
        repo_representation=None,
        known_context={},
        field_label="System description",
        user_message="thoughts?",
    )

    assert response.reply == "ok"
    assert fake.calls[0]["state"] is not None  # a throwaway state, never None itself


def test_discuss_field_does_not_mutate_the_passed_state(monkeypatch):
    """No advisory_turns entry, no history/StepLog, no token totals — this
    is session-only (see field_discussion.py), unlike `ask_advisor`."""
    fake = _FakeLLM(clar.FieldDiscussionResponse(reply="ok"))
    monkeypatch.setattr(clar, "llm_call", fake)
    state = new_run("x")
    before = state.model_copy(deep=True)

    clar.discuss_field(
        state, raw_prompt="x", repo_representation=None, known_context={},
        field_label="Budget", user_message="q",
    )

    assert state == before


def test_discuss_field_never_retrieves_kb_evidence(monkeypatch):
    """Not routed through the architecture literature RAG (see the module
    docstring's own statement of this) — proven by the fact that nothing
    here ever imports/calls `architect.retrieve_chunks` or
    `researcher.researcher_node`; a call that DID would need network/Chroma
    access this offline test provides none of, so a silent RAG dependency
    would surface as an error rather than passing quietly."""
    fake = _FakeLLM(clar.FieldDiscussionResponse(reply="ok"))
    monkeypatch.setattr(clar, "llm_call", fake)

    response = clar.discuss_field(
        None, raw_prompt="x", repo_representation=None, known_context={},
        field_label="Scale", user_message="q",
    )

    assert response.reply == "ok"


# ── field_purpose_hint ─────────────────────────────────────────────────────


def test_field_purpose_hint_returns_the_deterministic_template():
    hint = clar.field_purpose_hint("non_functional_requirements")
    assert hint  # non-empty
    assert hint == clar.slot_question("non_functional_requirements").why_needed


def test_field_purpose_hint_empty_for_a_field_with_no_template():
    assert clar.field_purpose_hint("project_name") == ""


# ── numeric-invention prohibition lives in the system prompt ─────────────


def test_system_prompt_prohibits_inventing_numeric_targets():
    system = clar._FIELD_DISCUSSION_SYSTEM.lower()
    assert "numeric" in system
    assert "sla" in system or "latency" in system
    assert "qualitative" in system


def test_system_prompt_frames_the_assistant_as_decision_support_not_the_architect():
    system = clar._FIELD_DISCUSSION_SYSTEM.lower()
    assert "not the architect" in system or "not design the target architecture" in system


def test_system_prompt_requires_structured_suggested_answer_contract():
    system = clar._FIELD_DISCUSSION_SYSTEM
    assert "suggested_answer" in system
    assert "reply" in system
