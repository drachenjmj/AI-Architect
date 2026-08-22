"""test_ui_history.py — tests for the READ-ONLY History view.

Scope: the History workspace view end-to-end through Streamlit AppTest —
the list (newest first, card metadata, empty state), the selected run's
read-only detail (saved data rendered, NO actions of any kind), and the
separation contract (browsing history never touches the current run).

Runs against the real ui.py with the offline demo state as the CURRENT run
and two fixture runs checkpointed into a per-test temp `.cache/runs/`
(via AI_ARCHITECT_RUNS_DIR) — never the developer machine's real cache.
No server, no browser, no network, no LLM, no live pipeline.

NEVER `import ui` IN THIS FILE (same reason as test_ui_workspace.py: ui.py
runs Streamlit calls at module level). Fixtures and navigation use AppTest
only; monkeypatching targets source modules.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pipeline import persistence
from ui_demo import build_demo_state
from ui_workspace import HISTORY_EMPTY_MESSAGE

_UI = str(Path(__file__).parent / "ui.py")

# Two distinguishable saved runs: the OLDER one is the capped variant
# (FAIL verdict, 2 open issues — good read-only fodder), the NEWER one the
# passing variant. Distinct project names and dates make every list-order
# and detail assertion unambiguous.
_OLDER_ID = "20260101T090000Z-histold"
_NEWER_ID = "20260102T090000Z-histnew"
_OLDER_PROJECT = "History Alpha"
_NEWER_PROJECT = "History Beta"


def _select_view(at: AppTest, view: str) -> AppTest:
    """Navigate by clicking the sidebar's grouped view buttons."""
    next(b for b in at.sidebar.button if b.label == view).click()
    at.run()
    assert not at.exception
    return at


def _finished_app() -> AppTest:
    """The app with a completed CURRENT run loaded, as a resume produces."""
    at = AppTest.from_file(_UI, default_timeout=30)
    at.session_state["state"] = build_demo_state("pass")
    at.run()
    assert not at.exception
    return at


def _saved_run(variant: str, run_id: str, project: str, last_step_at: str):
    state = build_demo_state(variant)
    state.run_id = run_id
    state.context_record.project_name = project
    state.blueprint.project_name = project
    state.history[-1].timestamp = last_step_at
    return state


@pytest.fixture()
def saved_runs(tmp_path, monkeypatch):
    """A temp runs directory holding exactly two finished runs, older first."""
    monkeypatch.setenv(
        "AI_ARCHITECT_RUNS_DIR", str(tmp_path / "runs")
    )
    persistence.save_state(
        _saved_run(
            "capped", _OLDER_ID, _OLDER_PROJECT, "2026-01-01T09:00:00+00:00"
        )
    )
    persistence.save_state(
        _saved_run(
            "pass", _NEWER_ID, _NEWER_PROJECT, "2026-01-02T09:00:00+00:00"
        )
    )
    return tmp_path / "runs"


@pytest.fixture()
def no_pipeline_actions(monkeypatch):
    """Fail the test if any pipeline/API/storage action fires while History
    is used. Same patches as test_ui_workspace's fixture — notably the
    RESUME paths (`persistence.list_runs` / `load_state`), which History
    must not use: it reads checkpoints through run_history instead."""

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


# ── 1. the list ───────────────────────────────────────────────────────────


def test_history_empty_state_when_no_runs_exist(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_ARCHITECT_RUNS_DIR", str(tmp_path / "nothing"))

    at = _select_view(_finished_app(), "History")

    assert any(HISTORY_EMPTY_MESSAGE in i.value for i in at.info)
    assert not [b for b in at.button if b.label == "Open"]


def test_history_lists_runs_newest_first_with_card_metadata(saved_runs):
    at = _select_view(_finished_app(), "History")

    # The workspace's scoped <style> block also mentions ws-card-title, so
    # match the actual card markup, not the class name anywhere.
    cards = [
        m.value for m in at.markdown if "<div class='ws-card-title'>" in m.value
    ]
    assert len(cards) == 2  # one card per saved run, current run not saved
    assert _NEWER_PROJECT in cards[0] and _OLDER_PROJECT in cards[1]

    joined = " | ".join(cards)
    assert _NEWER_ID in joined and _OLDER_ID in joined   # run ids, secondary
    assert "seasonal-shop" in joined                     # repo name
    assert "2 features" in joined and "2 ADRs" in joined # counts
    assert "PASS" in cards[0] and "FAIL" in cards[1]     # verdicts
    assert "2026-01-02 09:00" in cards[0]                # date/time
    assert "2026-01-01 09:00" in cards[1]


# ── 2. selecting one run opens only that run, read-only ───────────────────


def _open_run(at: AppTest, run_id: str) -> AppTest:
    at.button(key=f"hist_open_{run_id}").click()
    at.run()
    assert not at.exception
    return at


def test_selecting_a_run_loads_that_run_only(saved_runs):
    at = _open_run(_select_view(_finished_app(), "History"), _OLDER_ID)

    header = next(m for m in at.markdown if "ws-header" in m.value and "Saved run" in m.value)
    assert _OLDER_PROJECT in header.value
    assert _OLDER_ID in header.value
    assert "FAIL" in header.value                 # the capped run's verdict
    assert "read-only" in header.value
    # The OTHER run's data is nowhere on screen.
    assert _NEWER_PROJECT not in " | ".join(m.value for m in at.markdown)


def test_historical_tabs_render_the_saved_data(saved_runs):
    at = _open_run(_select_view(_finished_app(), "History"), _OLDER_ID)

    md = " | ".join(m.value for m in at.markdown)
    caps = " | ".join(c.value for c in at.caption)
    labels = [e.label for e in at.expander]
    # Overview: honest verdict for a capped run, no sign-off anywhere.
    assert "BEST-EFFORT — STOPPED ON BUDGET" in md
    # Context: the saved record and its clarification history.
    assert f"Context Record — {_OLDER_PROJECT}" in md
    assert any("Clarification history" in label for label in labels)
    # Architecture: the saved deliverable (labels carry emoji prefixes).
    assert any("Blueprint" in label for label in labels)
    assert any("Architecture Decision Records" in label for label in labels)
    assert any("Components" in label for label in labels)
    # Review: the saved FAIL verdict and its issues.
    assert "REVIEW — FAIL" in md
    assert "2 blocking issues" in md
    # Knowledge: the saved chunks.
    assert "Knowledge retrieved" in md and "4 chunks" in md
    # Repository: the saved analysis.
    assert "Repository analysis" in md
    # No run trace / token / cost machinery in the historical Overview.
    assert "Run trace" not in labels
    assert len(at.metric) == 0


def test_historical_detail_has_no_actions_of_any_kind(saved_runs):
    at = _open_run(_select_view(_finished_app(), "History"), _OLDER_ID)

    buttons = [b.label for b in at.button]
    boxes = [t.label or "" for t in at.text_area]
    # No sign-off, no feedback boxes, no advisory ask, no round submission.
    assert not any("Accept this design" in label for label in buttons)
    assert "Correct the requirements" not in boxes
    assert "Direct the architect" not in boxes
    assert "Ask" not in buttons
    assert "Add to pending feedback" not in buttons
    assert "Submit feedback round" not in buttons
    # The only actions on screen are navigation: back and the sidebar.
    assert "← All saved runs" in buttons


def test_history_performs_no_backend_action(saved_runs, no_pipeline_actions):
    at = _select_view(_finished_app(), "History")
    at = _open_run(at, _NEWER_ID)          # list + full load, booby-trapped
    next(b for b in at.button if b.label == "← All saved runs").click()
    at.run()
    assert not at.exception
    # Back on the list: both runs still listed.
    cards = [
        m.value for m in at.markdown if "<div class='ws-card-title'>" in m.value
    ]
    assert len(cards) == 2


# ── 3. the current run is untouched — and still fully alive ────────────────


def test_browsing_history_never_replaces_current_run_state(saved_runs):
    at = _finished_app()
    current_id = at.session_state["state"].run_id

    def _back_to_list(app: AppTest) -> AppTest:
        next(b for b in app.button if b.label == "← All saved runs").click()
        app.run()
        assert not app.exception
        return app

    at = _select_view(at, "History")
    at = _open_run(at, _OLDER_ID)
    at = _back_to_list(at)
    at = _open_run(at, _NEWER_ID)  # a second selection replaces only the view
    at = _back_to_list(at)

    assert at.session_state["state"].run_id == current_id
    assert at.session_state["state"].stage.value == "done"
    assert at.session_state["state"].user_rounds == 0
    # History's session footprint is UI-only display state.
    assert "history_selected_run_id" not in at.session_state


def test_returning_to_overview_shows_current_run_unchanged(saved_runs):
    at = _finished_app()
    current_id = at.session_state["state"].run_id

    at = _select_view(at, "History")
    at = _open_run(at, _OLDER_ID)
    at = _select_view(at, "Overview")

    # The current run's header, project and LIVE sign-off are all back.
    header = next(
        m for m in at.markdown if "ws-header" in m.value and "Seasonal Shop" in m.value
    )
    assert current_id in header.value
    assert any("Accept this design" in b.label for b in at.button)
    assert "REVIEW PASSED" in " | ".join(
        m.value for m in at.markdown
    )
