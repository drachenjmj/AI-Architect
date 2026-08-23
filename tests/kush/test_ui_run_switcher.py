"""test_ui_run_switcher.py — tests for the sidebar's CURRENT-RUN switcher.

Scope: the control that makes a different saved run the ACTIVE one —
deliberate switching through the EXISTING resume semantics
(`persistence.list_runs` summaries + `load_state`), the History/switcher
distinction (History stays read-only and never changes the current run),
and the pending-feedback guard (a staged bundle can never travel to a
newly-switched run).

Offline throughout: the demo state is the starting CURRENT run and the
switch targets are fixture runs checkpointed into a per-test temp runs
directory via AI_ARCHITECT_RUNS_DIR. Generation paths (pipeline, LLM
calls) are booby-trapped where relevant — switching is persistence-only.

Never `import ui` (module-level st calls — see test_ui_workspace.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pipeline import persistence
from ui_demo import build_demo_state

_UI = str(Path(__file__).resolve().parents[2] / "ui.py")  # repo root

_RUN_B = "20260102T090000Z-switchb"   # the newer fixture run
_RUN_C = "20260103T090000Z-switchc"   # a mid-flight run at AWAITING_HUMAN


@pytest.fixture()
def saved_runs(tmp_path, monkeypatch):
    """A temp runs directory with one finished run and one mid-flight run."""
    monkeypatch.setenv("AI_ARCHITECT_RUNS_DIR", str(tmp_path / "runs"))

    finished = build_demo_state("pass")
    finished.run_id = _RUN_B
    finished.context_record.project_name = "Switchable Beta"
    finished.blueprint.project_name = "Switchable Beta"
    persistence.save_state(finished)

    from pipeline.state import PendingDecision, Stage, new_run

    mid = new_run("Architect a ticketing system for concerts.")
    mid.run_id = _RUN_C
    mid.stage = Stage.AWAITING_HUMAN
    mid.pending_decision = PendingDecision.CLARIFICATION
    mid.clarifying_questions = ["What is the expected peak user count?"]
    persistence.save_state(mid)

    return tmp_path / "runs"


def _app(saved_runs) -> AppTest:
    """A finished app with the switcher EXPANDED (its key is seeded so the
    expander's body runs without an expander-toggle API in AppTest)."""
    at = AppTest.from_file(_UI, default_timeout=30)
    at.session_state["state"] = build_demo_state("pass")
    at.session_state["switch_run_expander"] = True
    at.run()
    assert not at.exception
    return at


def _switcher_selectbox(at: AppTest):
    return next(s for s in at.selectbox if s.label == "Saved runs")


def _pick_run(at: AppTest, needle: str) -> AppTest:
    """Choose the saved-run option whose picker line contains `needle`
    (the prompt excerpt is the stable part of `_run_label`)."""
    box = _switcher_selectbox(at)
    target = next(option for option in box.options if needle in option)
    box.set_value(target)
    at.run()
    assert not at.exception
    return at


_B_NEEDLE = "Make the existing shop"       # fixture B's prompt excerpt
_C_NEEDLE = "Architect a ticketing system"  # fixture C's prompt excerpt


def _apply(at: AppTest) -> AppTest:
    """Click Make-current and let the switch (which ends in st.rerun)
    settle — AppTest needs one extra pass to show the post-switch tree."""
    at.button(key="switch_run_apply").click()
    at.run()
    at.run()
    return at


# ── 1. the control identifies the current run ─────────────────────────────


def test_switcher_names_the_current_run(saved_runs):
    at = _app(saved_runs)

    texts = " ".join(c.value for c in at.sidebar.caption) + " " + " ".join(
        m.value for m in at.sidebar.markdown
    )
    assert "Current run" in texts
    assert "Seasonal Shop" in texts          # the demo run's project
    assert "seasonal-shop" in texts          # its repository


def test_switcher_closed_scans_nothing(saved_runs, monkeypatch):
    """With the expander closed the control must not touch the runs
    directory on ordinary reruns — that is what keeps the sidebar cheap."""
    def _boom():
        raise AssertionError("runs directory scanned while the expander is closed")

    monkeypatch.setattr("pipeline.persistence.list_runs", _boom)

    at = AppTest.from_file(_UI, default_timeout=30)
    at.session_state["state"] = build_demo_state("pass")
    at.run()
    assert not at.exception


# ── 2. deliberate switching through the resume semantics ─────────────────


def test_switching_makes_the_selected_run_current(saved_runs, monkeypatch):
    """The switcher uses the EXISTING loader: after Make-current, the
    session's state IS the checkpointed run, loaded through
    persistence.load_state — not a new loader."""
    loads: list[str] = []
    real_load = persistence.load_state

    def counting(run_id):
        loads.append(run_id)
        return real_load(run_id)

    monkeypatch.setattr("pipeline.persistence.load_state", counting)

    at = _app(saved_runs)
    at = _apply(_pick_run(at, _B_NEEDLE))

    assert loads == [_RUN_B]                  # existing resume path, once
    assert at.session_state["state"].run_id == _RUN_B
    # The workspace now shows the switched run, not the demo one.
    header = next(
        m for m in at.markdown if "ws-header" in m.value and "ws-project" in m.value
    )
    assert "Switchable Beta" in header.value
    assert _RUN_B in header.value


def test_switching_to_a_mid_flight_run_lands_in_the_linear_flow(saved_runs):
    """Resume semantics, not workspace semantics: an AWAITING_HUMAN run
    becomes current and shows its clarifying-question form."""
    at = _app(saved_runs)
    at = _apply(_pick_run(at, _C_NEEDLE))

    assert at.session_state["state"].run_id == _RUN_C
    assert any(
        "peak user count" in (t.label or "") for t in at.text_input
    )                                        # the question form, live


def test_no_generation_is_triggered_by_switching(saved_runs, monkeypatch):
    """Switching is persistence-only: no pipeline run, no LLM call, no
    feedback submission, no sign-off — anywhere in the flow."""
    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003 — any call is the bug
        raise AssertionError("a generation path fired during a run switch")

    monkeypatch.setattr("pipeline.orchestrator.run_pipeline_streaming", _boom)
    monkeypatch.setattr("pipeline.agents.clarifier.ask_advisor", _boom)
    monkeypatch.setattr("pipeline.user_feedback.submit_feedback", _boom)
    monkeypatch.setattr("pipeline.sign_off.accept_design", _boom)

    at = _app(saved_runs)
    at = _apply(_pick_run(at, _B_NEEDLE))
    assert not at.exception
    assert at.session_state["state"].run_id == _RUN_B


# ── 3. pending feedback can never travel to a switched run ────────────────


def test_pending_feedback_blocks_the_switch(saved_runs):
    at = _app(saved_runs)
    # Stage unsent feedback for the CURRENT (demo) run, as the boxes do.
    at.session_state["pending_feedback"] = {"design": "use SQS instead of Kafka"}
    at.run()
    assert not at.exception
    demo_id = at.session_state["state"].run_id

    at = _pick_run(at, _B_NEEDLE)
    at.button(key="switch_run_apply").click()
    at.run()  # the refusal fires (and is drawn) in THIS run
    assert not at.exception

    # Refused with a warning; nothing moved and the bundle is intact.
    assert any("unsent feedback" in w.value for w in at.warning)
    assert at.session_state["state"].run_id == demo_id
    assert at.session_state["pending_feedback"]["design"] == (
        "use SQS instead of Kafka"
    )
    # And it STAYS refused after the rerun settles.
    at.run()
    assert at.session_state["state"].run_id == demo_id


def test_switching_works_once_the_bundle_is_discarded(saved_runs):
    at = _app(saved_runs)
    at.session_state["pending_feedback"] = {"design": "use SQS instead of Kafka"}
    at.run()
    at.session_state["pending_feedback"] = {}
    at.run()
    at = _apply(_pick_run(at, _B_NEEDLE))

    assert at.session_state["state"].run_id == _RUN_B


# ── 4. History stays the read-only counterpart ────────────────────────────


def test_browsing_history_does_not_switch_the_current_run(saved_runs):
    at = _app(saved_runs)
    demo_id = at.session_state["state"].run_id

    next(b for b in at.sidebar.button if b.label == "History").click()
    at.run()
    at.button(key=f"hist_open_{_RUN_B}").click()
    at.run()
    assert not at.exception

    # The historical view SHOWS run B, but the CURRENT run is unchanged —
    # only the explicit switcher can change it.
    assert at.session_state["history_selected_run_id"] == _RUN_B
    assert at.session_state["state"].run_id == demo_id
