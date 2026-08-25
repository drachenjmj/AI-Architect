"""test_ui_overview_simplification.py — tests for the Overview→destination
navigation links and the `Needs your attention` block.

Scope: the two ADDITIVE pieces of the overview-simplification pass that
`test_ui_overview_review.py` (duplicate detail removed) and
`test_ui_workspace.py` (renamed sidebar labels, stable internal identifiers)
do not already cover —

  * the inline "View all decisions / View components / View validation &
    findings" links on Overview jump to the existing Architecture / Review
    workspace views (same internal identifiers, no new router, no query
    string — see `ui_workspace._nav_link`);
  * `Needs your attention` renders exactly the real pending actions
    (staged feedback, a run awaiting sign-off) and nothing when there is
    none — no empty "all clear" card;
  * neither surfaces in the read-only History detail, which the workspace
    keeps action-free by construction (`_render_historical_overview` calls
    neither helper).

Runs against the real ui.py with the offline demo states. No network, no
LLM, no live pipeline. Never `import ui` (module-level st calls — see
test_ui_workspace.py's docstring).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pipeline import persistence
from pipeline.state import Stage
from ui_demo import build_demo_state

_UI = str(Path(__file__).resolve().parents[2] / "ui.py")  # repo root


@pytest.fixture()
def no_pipeline_actions(monkeypatch):
    """Fail the test if any pipeline/API/storage action fires — clicking a
    navigation link or staging text must never reach the backend."""

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


def _finished_app(variant: str = "pass") -> AppTest:
    at = AppTest.from_file(_UI, default_timeout=30)
    at.session_state["state"] = build_demo_state(variant)
    at.run()
    assert not at.exception
    return at


def _md(at: AppTest) -> str:
    return " | ".join(m.value for m in at.markdown)


def _click(at: AppTest, key: str) -> AppTest:
    next(b for b in at.button if b.key == key).click()
    at.run()
    assert not at.exception
    return at


# ── 1. Overview → destination links ─────────────────────────────────────


def test_view_all_decisions_link_opens_architecture():
    at = _click(_finished_app(), "ov_link_decisions")

    assert at.session_state["workspace_view"] == "Architecture"
    assert ">Recommended Architecture<" in _md(at).replace("&amp;", "&")
    # The destination's own content, not just the header, is on screen.
    assert any("Architecture Decision Records" in e.label for e in at.expander)


def test_view_components_link_opens_architecture():
    at = _click(_finished_app(), "ov_link_components")

    assert at.session_state["workspace_view"] == "Architecture"
    assert any("Components" in e.label for e in at.expander)


def test_view_validation_and_findings_link_opens_review():
    at = _click(_finished_app("capped"), "ov_link_validation")

    assert at.session_state["workspace_view"] == "Review"
    md = _md(at)
    assert "REVIEW — FAIL" in md
    assert len(at.table) == 1  # the full findings table itself


def test_overview_links_trigger_no_pipeline_or_model_call(no_pipeline_actions):
    at = _finished_app()
    for key in ("ov_link_decisions", "ov_link_components", "ov_link_validation"):
        at = _click(_finished_app(), key)  # fresh app per link — clean state
        assert not at.exception


# ── 2. Needs your attention ──────────────────────────────────────────────


def test_needs_attention_appears_for_a_pending_sign_off():
    at = _finished_app()  # the demo "pass" run: DONE, not yet accepted

    assert any("Needs your attention" in m.value for m in at.markdown)
    assert any(
        "ready for sign-off" in m.value for m in at.markdown
    )


def test_needs_attention_appears_for_staged_feedback():
    at = AppTest.from_file(_UI, default_timeout=30)
    at.session_state["state"] = build_demo_state("pass")
    at.session_state["pending_feedback"] = {"design": "Please add a cache layer."}
    at.run()
    assert not at.exception

    bullets = [m.value for m in at.markdown if m.value.startswith("- ")]
    assert any("staged change" in b and "sidebar" in b for b in bullets)


def test_needs_attention_pluralizes_multiple_staged_changes():
    at = AppTest.from_file(_UI, default_timeout=30)
    state = build_demo_state("pass")
    at.session_state["state"] = state
    at.session_state["pending_feedback"] = {
        "requirements": "Raise the concurrency target.",
        "design": "Please add a cache layer.",
    }
    at.run()
    assert not at.exception

    bullets = [m.value for m in at.markdown if m.value.startswith("- ")]
    assert any("2 staged changes are waiting" in b for b in bullets)


def test_needs_attention_renders_nothing_on_a_clean_accepted_run():
    at = AppTest.from_file(_UI, default_timeout=30)
    state = build_demo_state("pass")
    state.stage = Stage.ACCEPTED
    state.accepted_at = "2026-08-20T00:00:00+00:00"
    at.session_state["state"] = state
    at.run()
    assert not at.exception

    assert not any("Needs your attention" in m.value for m in at.markdown)


def test_needs_attention_hides_staged_feedback_once_accepted():
    """Feedback staged before sign-off is still shown in the sidebar bundle
    preview after acceptance (see `render_pending_feedback_panel`), but it
    is no longer an actionable item — `feedback_is_closed` says so."""
    at = AppTest.from_file(_UI, default_timeout=30)
    state = build_demo_state("pass")
    state.stage = Stage.ACCEPTED
    state.accepted_at = "2026-08-20T00:00:00+00:00"
    at.session_state["state"] = state
    at.session_state["pending_feedback"] = {"design": "Please add a cache layer."}
    at.run()
    assert not at.exception

    assert not any("Needs your attention" in m.value for m in at.markdown)


# ── 3. historical runs stay action-free ──────────────────────────────────


_HIST_ID = "20260101T090000Z-simplify"


@pytest.fixture()
def saved_run(tmp_path, monkeypatch):
    """One saved, finished-but-unaccepted run in a temp runs directory."""
    monkeypatch.setenv("AI_ARCHITECT_RUNS_DIR", str(tmp_path / "runs"))
    state = build_demo_state("pass")
    state.run_id = _HIST_ID
    persistence.save_state(state)
    return tmp_path / "runs"


def test_historical_overview_shows_no_needs_attention_or_nav_links(saved_run):
    at = _finished_app()
    next(b for b in at.sidebar.button if b.key == "nav_History").click()
    at.run()
    at.button(key=f"hist_open_{_HIST_ID}").click()
    at.run()
    assert not at.exception

    # The saved run is DONE, not accepted — exactly the state that shows
    # "ready for sign-off" on the LIVE Overview — but History has no
    # sign-off, no feedback staging, and draws neither helper.
    assert not any("Needs your attention" in m.value for m in at.markdown)
    assert not any(
        b.key in ("ov_link_decisions", "ov_link_components", "ov_link_validation")
        for b in at.button
    )
