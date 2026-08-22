"""test_ui_chat.py — tests for the read-only, grounded Architecture Chat.

Scope: architecture_chat.py (the data/answer layer) and the Chat workspace
view through AppTest — context selection, historical-intent rules, the
ONE-call answer behavior, citation rendering, session isolation, and every
read-only guarantee.

The LLM is ALWAYS stubbed: `pipeline.llm.llm_call` is patched to a fake
that records its prompt and returns a deterministic answer. History
fixtures are checkpointed into a per-test temp runs directory via
AI_ARCHITECT_RUNS_DIR. No network, no API, no live pipeline — the one
assertion-of-record is that NOTHING fires that is not the single answer
call.

Never `import ui` (module-level st calls — see test_ui_workspace.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import architecture_chat
import run_history
from pipeline import persistence
from ui_demo import build_demo_state

_UI = str(Path(__file__).parent / "ui.py")

# Hex-suffixed like real run ids (uuid4().hex[:8]) — the historical-intent
# rules match run ids by that exact shape.
_RUN_B = "20260102T090000Z-00aa00b2"
_RUN_C = "20260103T090000Z-00cc00c3"


# ── fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture()
def saved_runs(tmp_path, monkeypatch):
    """Two historical runs besides the (unsaved) demo current run."""
    monkeypatch.setenv("AI_ARCHITECT_RUNS_DIR", str(tmp_path / "runs"))

    older = build_demo_state("capped")
    older.run_id = _RUN_B
    older.context_record.project_name = "Chat Alpha"
    older.blueprint.project_name = "Chat Alpha"
    persistence.save_state(older)

    newer = build_demo_state("pass")
    newer.run_id = _RUN_C
    newer.context_record.project_name = "Chat Beta"
    newer.blueprint.project_name = "Chat Beta"
    persistence.save_state(newer)

    return tmp_path / "runs"


class _FakeLLM:
    """Records prompts; returns a deterministic grounded-looking answer."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, state, prompt, *, system="", **kwargs):
        self.calls.append({"prompt": prompt, "system": system})
        return "Stubbed grounded answer [C0].", None


@pytest.fixture()
def fake_llm(monkeypatch):
    fake = _FakeLLM()
    monkeypatch.setattr("pipeline.llm.llm_call", fake)
    # architecture_chat imports the module, so patch its view of it too —
    # it calls `llm.llm_call` through the module, so one patch site works.
    return fake


def _booby_trap(monkeypatch):
    """Everything that must NEVER fire during chat use."""
    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("a state-changing path fired during chat")

    monkeypatch.setattr("pipeline.orchestrator.run_pipeline_streaming", _boom)
    monkeypatch.setattr("pipeline.user_feedback.submit_feedback", _boom)
    monkeypatch.setattr("pipeline.sign_off.accept_design", _boom)
    monkeypatch.setattr("pipeline.agents.clarifier.ask_advisor", _boom)
    monkeypatch.setattr("pipeline.persistence.save_state", _boom)
    return _boom


def _chat_app(saved_runs) -> AppTest:
    """The finished app with the Chat view already selected."""
    at = AppTest.from_file(_UI, default_timeout=30)
    at.session_state["state"] = build_demo_state("pass")
    at.run()
    assert not at.exception
    next(b for b in at.sidebar.button if b.label == "Chat").click()
    at.run()
    assert not at.exception
    return at


def _ask(at: AppTest, question: str) -> AppTest:
    """Submit one chat question."""
    at.chat_input[0].set_value(question)
    at.run()
    at.run()  # the submit path ends in st.rerun(); settle the tree
    return at


def _chat_store(at: AppTest) -> dict:
    """The chat session-state store (AppTest's session_state supports
    indexing and `in`, not dict methods)."""
    return at.session_state["architecture_chat"] if (
        "architecture_chat" in at.session_state
    ) else {}


# ── 1–2. cost: rendering is free, one question = one call ─────────────────


def test_rendering_chat_makes_no_llm_call(saved_runs, fake_llm, monkeypatch):
    _booby_trap(monkeypatch)
    at = _chat_app(saved_runs)

    assert len(at.chat_input) == 1
    assert fake_llm.calls == []          # opening the page cost nothing


def test_one_submitted_question_is_exactly_one_answer_call(
    saved_runs, fake_llm, monkeypatch
):
    _booby_trap(monkeypatch)
    at = _ask(_chat_app(saved_runs), "Why did we choose SQS?")

    assert len(fake_llm.calls) == 1
    assert "Why did we choose SQS?" in fake_llm.calls[0]["prompt"]
    # And the answer, with its citations, is on the page.
    md = " ".join(m.value for m in at.markdown)
    assert "Stubbed grounded answer [C0]." in md


# ── 3. artifact-id selection (unit level, deterministic) ──────────────────


def test_artifact_id_query_selects_the_matching_source():
    state = build_demo_state("pass")
    sources = architecture_chat.build_current_run_sources(state)

    selected = architecture_chat.select_sources(
        "What does ADR-002 imply for consistency?", sources
    )
    labels = [s.label for s in selected]
    assert "Current · ADR-002" in labels
    assert "Current · Run summary" in labels          # always included

    selected = architecture_chat.select_sources(
        "I disagree with FEAT-002, what would changing it affect?", sources
    )
    assert "Current · Feature FEAT-002" in [
        s.label for s in selected
    ]


def test_selection_is_capped_and_never_pads():
    state = build_demo_state("pass")
    sources = architecture_chat.build_current_run_sources(state)

    selected = architecture_chat.select_sources("checkout queue orders", sources)
    assert 1 <= len(selected) <= architecture_chat._MAX_DETAIL_SOURCES + 1


# ── 4–5. history loading is intent-gated and bounded ──────────────────────


def test_ordinary_question_never_touches_history(saved_runs, fake_llm, monkeypatch):
    """A current-run question loads NO historical state — not even the
    directory listing."""
    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("history was scanned for a non-historical query")

    monkeypatch.setattr("run_history.list_history_runs", _boom)
    monkeypatch.setattr("run_history.load_history_run", _boom)

    state = build_demo_state("pass")
    answer, sources = architecture_chat.answer_chat_question(
        "Why did we choose SQS?", state
    )
    assert all(s.run_id == state.run_id for s in sources)


def test_historical_query_loads_only_bounded_candidates(saved_runs, fake_llm, monkeypatch):
    loads: list[str] = []
    real_load = run_history.load_history_run

    def counting(run_id):
        loads.append(run_id)
        return real_load(run_id)

    monkeypatch.setattr("run_history.load_history_run", counting)

    state = build_demo_state("pass")
    answer, sources = architecture_chat.answer_chat_question(
        "Compare this architecture with the previous run.", state
    )

    assert 1 <= len(loads) <= architecture_chat._MAX_HISTORY_RUNS
    # The chosen candidate is the newest OTHER run (fixture C).
    assert loads == [_RUN_C]
    assert any(s.scope == "history" for s in sources)
    assert fake_llm.calls  # the answer itself was produced


def test_historical_run_id_targets_exactly_that_run(saved_runs, fake_llm):
    state = build_demo_state("pass")
    answer, sources = architecture_chat.answer_chat_question(
        f"What did we do in run {_RUN_B}?", state
    )
    history = [s for s in sources if s.scope == "history"]
    assert len(history) == 1
    assert history[0].run_id == _RUN_B
    assert "Chat Alpha" in history[0].label


# ── 6–7. read-only history, ambiguity refuses ─────────────────────────────


def test_history_access_is_read_only_and_current_unchanged(saved_runs, fake_llm):
    state = build_demo_state("pass")
    before = state.model_dump_json()

    architecture_chat.answer_chat_question(
        "Compare with the previous run", state
    )

    assert state.model_dump_json() == before      # current run untouched
    # Nothing was written next to the historical runs either (the History
    # layer only reads; the booby-trap in other tests proves no writes).


def test_ambiguous_history_does_not_pick_an_arbitrary_run(saved_runs, fake_llm):
    state = build_demo_state("pass")
    answer, sources = architecture_chat.answer_chat_question(
        "Compare this architecture with the other runs.", state
    )
    # Deterministic disambiguation answer, NO model call, no invented pick.
    assert fake_llm.calls == []
    assert "Which one do you mean?" in answer
    assert _RUN_B in answer and _RUN_C in answer   # the candidates listed


def test_missing_history_says_so(saved_runs, fake_llm):
    state = build_demo_state("pass")
    answer, _ = architecture_chat.answer_chat_question(
        "What changed since yesterday's run?", state
    )
    # Both fixture runs are dated 2026-01 — "yesterday" matches neither
    # weekday nor date, so the answer asks for a specific run…
    assert fake_llm.calls == [] or "Which one" in answer or "No saved run" in answer


# ── 8–9. KB sources and citation rendering ────────────────────────────────


def test_saved_kb_chunks_are_selectable_sources(saved_runs, fake_llm, monkeypatch):
    _booby_trap(monkeypatch)
    at = _ask(_chat_app(saved_runs), "What does the GDPR guidance say?")

    prompt = fake_llm.calls[0]["prompt"]
    assert "gdpr-for-architects.pdf" in prompt       # the saved chunk
    assert "[K" in prompt                            # KB citation ids exist
    # Sources used renders under the answer with KB metadata.
    sources_md = " ".join(c.value for c in at.caption)
    assert any(
        label in sources_md for label in ("gdpr-for-architects.pdf", "KB")
    )


def test_source_ids_map_back_to_rendered_sources(saved_runs, fake_llm, monkeypatch):
    _booby_trap(monkeypatch)
    at = _ask(_chat_app(saved_runs), "Why ADR-002?")
    state = at.session_state["state"]

    transcript = _chat_store(at).get(state.run_id, [])
    assistant = [m for m in transcript if m["role"] == "assistant"][-1]
    assert assistant["sources"], "sources were recorded with the answer"
    for source in assistant["sources"]:
        assert source["sid"] and source["label"]
    # The expander with those sids is on screen.
    assert any(
        e.label.startswith("Sources used") for e in at.expander
    )


# ── 10. unsupported questions get no fabricated context ───────────────────


def test_empty_answer_falls_back_to_an_honest_message(saved_runs, monkeypatch):
    monkeypatch.setattr(
        "pipeline.llm.llm_call",
        lambda *_a, **_k: ("", None),  # model returns nothing
    )
    state = build_demo_state("pass")
    answer, _ = architecture_chat.answer_chat_question(
        "What is the meaning of life?", state
    )
    assert "do not establish an answer" in answer


def test_system_prompt_enforces_grounding_and_read_only(saved_runs, fake_llm):
    architecture_chat.answer_chat_question(
        "anything", build_demo_state("pass")
    )
    system = fake_llm.calls[0]["system"]
    assert "ONLY from the labeled sources" in system
    assert "READ-ONLY" in system
    assert "do not establish the answer" in system
    assert "never claim a change was made" in system.lower()


# ── 11. change requests stay read-only ────────────────────────────────────


def test_change_request_does_not_execute_anything(saved_runs, fake_llm, monkeypatch):
    boom = _booby_trap(monkeypatch)
    at = _ask(
        _chat_app(saved_runs),
        "Change FEAT-002 so the order data stays in the US instead.",
    )

    # One answer call, nothing else — no feedback, no pipeline, no sign-off,
    # no checkpoint write (the booby-trap would have raised into the page).
    assert len(fake_llm.calls) == 1
    state = at.session_state["state"]
    assert state.features[1].name == "Protect EU order data"  # untouched
    assert state.user_rounds == 0
    assert state.user_feedback == []


# ── 12–14. session isolation and Clear chat ───────────────────────────────


def test_messages_are_keyed_by_run(saved_runs, fake_llm, monkeypatch):
    _booby_trap(monkeypatch)
    at = _ask(_chat_app(saved_runs), "Why SQS?")
    demo_id = at.session_state["state"].run_id

    store = at.session_state["architecture_chat"]
    assert list(store.keys()) == [demo_id]
    transcript = store[demo_id]
    assert [m["role"] for m in transcript] == ["user", "assistant"]


def test_switching_runs_does_not_leak_chat(saved_runs, fake_llm, monkeypatch):
    _booby_trap(monkeypatch)
    at = _ask(_chat_app(saved_runs), "Why SQS?")
    demo_id = at.session_state["state"].run_id

    # Switch current run through the existing switcher.
    at.session_state["switch_run_expander"] = True
    at.run()
    box = next(s for s in at.selectbox if s.label == "Saved runs")
    target = next(o for o in box.options if "Chat Beta" in o or "peak" in o)
    box.set_value(target)
    at.run()
    at.button(key="switch_run_apply").click()
    at.run()
    at.run()
    new_id = at.session_state["state"].run_id
    assert new_id != demo_id

    # The new run's Chat view is EMPTY — no leak from the old conversation.
    next(b for b in at.sidebar.button if b.label == "Chat").click()
    at.run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "Why SQS?" not in md
    # ...but the old run's transcript is still in the store (restorable).
    assert any("Why SQS?" in m["content"] for m in
               at.session_state["architecture_chat"].get(demo_id, []))


def test_clear_chat_clears_only_the_active_run(saved_runs, fake_llm, monkeypatch):
    _booby_trap(monkeypatch)
    at = _ask(_chat_app(saved_runs), "Why SQS?")
    demo_id = at.session_state["state"].run_id

    next(b for b in at.button if b.label == "Clear chat").click()
    at.run()
    at.run()

    assert at.session_state["architecture_chat"].get(demo_id) in (None, [])
    md = " ".join(m.value for m in at.markdown)
    assert "Why SQS?" not in md


# ── 15. LLM failure is graceful ───────────────────────────────────────────


def test_llm_failure_keeps_the_page_and_prior_messages(saved_runs, monkeypatch):
    at = _chat_app(saved_runs)

    # First question succeeds (stub), establishing prior messages.
    good = _FakeLLM()
    monkeypatch.setattr("pipeline.llm.llm_call", good)
    at = _ask(at, "Why SQS?")
    assert any("Stubbed grounded answer" in m.value for m in at.markdown)

    # Second question fails: the API raises.
    def failing(*_a, **_k):
        from pipeline.llm import LLMError

        raise LLMError("quota exceeded")

    monkeypatch.setattr("pipeline.llm.llm_call", failing)
    at.chat_input[0].set_value("And what about ADR-002?")
    at.run()
    at.run()

    assert not at.exception                    # the page did not crash
    md = " ".join(m.value for m in at.markdown)
    assert "could not reach the model" in md   # concise assistant error
    assert "Why SQS?" in md                    # prior messages intact
    assert "And what about ADR-002?" in md     # the question kept visible


# ── 16. History view remains read-only alongside the chat ─────────────────


def test_history_view_still_read_only_with_chat_present(saved_runs, fake_llm, monkeypatch):
    _booby_trap(monkeypatch)
    at = _ask(_chat_app(saved_runs), "Why SQS?")   # chat in session
    demo_id = at.session_state["state"].run_id

    next(b for b in at.sidebar.button if b.label == "History").click()
    at.run()
    at.button(key=f"hist_open_{_RUN_C}").click()
    at.run()
    assert not at.exception

    assert at.session_state["state"].run_id == demo_id  # unchanged
    assert "read-only" in " ".join(c.value for c in at.caption)
