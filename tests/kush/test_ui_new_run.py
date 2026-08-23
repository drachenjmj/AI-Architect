"""test_ui_new_run.py — regression tests for the `New run` button.

The bug: under `-- --demo` (the mode every manual smoke test runs in),
clicking `New run` cleared the session and the demo seeder immediately
re-loaded the same finished demo state — the button appeared to do
nothing. The fix is the `_NEW_RUN_KEY` marker: an explicit New run
outranks the demo auto-seeder; a fresh session still seeds normally.

Everything here is offline: demo states are seeded into session state
directly or via the real `--demo` argv path (monkeypatched sys.argv during
AppTest runs), saved runs live in a per-test temp directory, and every
generation path is booby-trapped where relevant. Never `import ui` (see
test_ui_workspace.py's docstring).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pipeline import persistence
from ui_demo import build_demo_state

_UI = str(Path(__file__).resolve().parents[2] / "ui.py")  # repo root


@pytest.fixture()
def booby_trap(monkeypatch):
    """Fail the test if any pipeline/API/storage action fires from a click."""

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003 — any call is the bug
        raise AssertionError("a pipeline/API/storage action was triggered")

    monkeypatch.setattr("pipeline.orchestrator.run_pipeline_streaming", _boom)
    monkeypatch.setattr("pipeline.persistence.load_state", _boom)
    monkeypatch.setattr("pipeline.persistence.save_state", _boom)
    monkeypatch.setattr("pipeline.agents.clarifier.ask_advisor", _boom)
    monkeypatch.setattr("pipeline.user_feedback.submit_feedback", _boom)
    monkeypatch.setattr("pipeline.sign_off.accept_design", _boom)
    monkeypatch.setattr("pipeline.llm.llm_call", _boom)
    return _boom


@pytest.fixture()
def saved_runs(tmp_path, monkeypatch):
    """One saved historical run, so History remains provably untouched."""
    monkeypatch.setenv("AI_ARCHITECT_RUNS_DIR", str(tmp_path / "runs"))
    other = build_demo_state("capped")
    other.run_id = "20260102T090000Z-00aa00b2"
    persistence.save_state(other)
    return tmp_path / "runs"


def _finished_app() -> AppTest:
    at = AppTest.from_file(_UI, default_timeout=30)
    at.session_state["state"] = build_demo_state("pass")
    at.run()
    assert not at.exception
    return at


def _click_new_run(at: AppTest) -> AppTest:
    next(b for b in at.sidebar.button if "New run" in b.label).click()
    at.run()
    at.run()  # the click path ends in st.rerun(); settle the tree
    assert not at.exception
    return at


def _intake_visible(at: AppTest) -> bool:
    return any(
        "System description" in (t.label or "") for t in at.text_area
    ) and any(b.label == "Start" for b in at.button)


@pytest.fixture()
def demo_argv(monkeypatch):
    """Run the app exactly as `streamlit run ui.py -- --demo` does."""
    monkeypatch.setattr(sys, "argv", ["ui.py", "--", "--demo"])
    return "pass"


# ── 1–2. the click leaves the run and shows intake immediately ────────────


def test_new_run_shows_intake_and_drops_current_state(saved_runs):
    at = _finished_app()
    assert not _intake_visible(at)          # the workspace is up first

    at = _click_new_run(at)

    assert at.session_state["state"] is None
    assert _intake_visible(at)
    # The old run's workspace is gone.
    assert not any("Seasonal Shop" in m.value for m in at.markdown)
    assert not any(b.label == "Overview" for b in at.sidebar.button)


def test_new_run_under_demo_mode_is_not_undone_by_the_seeder(saved_runs, demo_argv):
    """THE regression: under `--demo`, the seeder used to re-load the demo
    state right after the click's clear, so the button did nothing."""
    at = AppTest.from_file(_UI, default_timeout=30)
    at.run()
    assert at.session_state["state"] is not None      # demo seeded once

    at = _click_new_run(at)

    assert at.session_state["state"] is None          # NOT re-seeded
    assert _intake_visible(at)


def test_demo_still_seeds_a_fresh_session(saved_runs, demo_argv):
    """The fix must not break the demo's own contract: a brand-new session
    under `--demo` still loads the finished run for screenshots."""
    at = AppTest.from_file(_UI, default_timeout=30)
    at.run()
    assert at.session_state["state"] is not None
    assert at.session_state["state"].stage.value == "done"


# ──3. clicking triggers nothing ────────────────────────────────────────────


def test_clicking_new_run_triggers_no_pipeline_or_api(saved_runs, booby_trap):
    at = _click_new_run(_finished_app())
    assert _intake_visible(at)             # booby-trap never fired


# ── 4–5. History on disk is untouched and still usable ────────────────────


def test_new_run_does_not_delete_saved_runs(saved_runs):
    import run_history

    before = [s.run_id for s in run_history.list_history_runs()]
    assert before == ["20260102T090000Z-00aa00b2"]

    at = _click_new_run(_finished_app())
    assert _intake_visible(at)

    after = [s.run_id for s in run_history.list_history_runs()]
    assert after == before                  # nothing removed from disk


def test_history_remains_available_after_new_run(saved_runs):
    at = _click_new_run(_finished_app())

    # The intake screen's own History access — the resume picker — still
    # renders with the saved run among its options.
    picker = next(
        (s for s in at.selectbox if "Resume" in (s.label or "")), None
    )
    assert picker is not None and picker.options
    # The authoritative check: discovery still finds the exact run.
    import run_history

    assert [s.run_id for s in run_history.list_history_runs()] == [
        "20260102T090000Z-00aa00b2"
    ]


# ── 6. old chat does not leak into the new run ────────────────────────────


def test_old_chat_does_not_leak_into_new_run(saved_runs):
    at = _finished_app()
    old_id = at.session_state["state"].run_id
    # Stage an old conversation, as a prior Chat session would have.
    at.session_state["architecture_chat"] = {
        old_id: [{"role": "user", "content": "Why SQS?"}]
    }
    at.run()

    at = _click_new_run(at)

    # The clear wiped the store: nothing under any key.
    store = (
        at.session_state["architecture_chat"]
        if "architecture_chat" in at.session_state
        else {}
    )
    assert not store


# ── 7. workspace navigation state is reset ────────────────────────────────


def test_navigation_state_does_not_keep_the_old_view(saved_runs):
    at = _finished_app()
    next(b for b in at.sidebar.button if b.label == "Knowledge").click()
    at.run()

    at = _click_new_run(at)

    assert "workspace_view" not in at.session_state
    assert not any(b.label == "Overview" for b in at.sidebar.button)  # no nav


# ── 8. pending feedback cannot leak into the new run ─────────────────────


def test_pending_feedback_is_cleared_not_carried(saved_runs):
    at = _finished_app()
    at.session_state["pending_feedback"] = {
        "design": "use SQS instead of Kafka"
    }
    at.run()

    at = _click_new_run(at)

    bundle = (
        at.session_state["pending_feedback"]
        if "pending_feedback" in at.session_state
        else {}
    )
    assert not any(str(v).strip() for v in bundle.values())
    # And the pending panel is gone with the workspace.
    assert not any(
        "Pending feedback round" in c.value for c in at.sidebar.caption
    )
