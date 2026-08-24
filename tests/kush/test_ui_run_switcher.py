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


def test_switch_run_expander_key_is_tracked_by_streamlit(saved_runs):
    """Regression for the exact root cause of the "Switch run" bug:
    `st.expander(key=...)` only populates `st.session_state[key]` at all
    when `on_change="rerun"` is also passed. Without it, Streamlit never
    tracks the expanded state, `st.session_state.get(key)` is permanently
    `None` regardless of how a real user clicks the expander, and the
    guarded body (`if not st.session_state.get(_SWITCH_OPEN_KEY): return`)
    always returns before drawing the "Saved runs" selector — the control
    visually opens but silently renders nothing.

    This must hold on a run where the key was NOT pre-seeded, so it proves
    Streamlit itself is tracking the key rather than merely confirming a
    test-only workaround."""
    at = AppTest.from_file(_UI, default_timeout=30)
    at.session_state["state"] = build_demo_state("pass")
    at.run()
    assert not at.exception
    assert "switch_run_expander" in at.session_state


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


# ── 5. same persisted-run source as the working resume picker ─────────────


def test_switcher_and_resume_picker_list_the_same_persisted_runs(saved_runs):
    """Both controls are driven by the SAME `persistence.list_runs()` call —
    every OTHER persisted run visible to the working "Resume a previous
    run" selector on the intake screen must also be available to "Switch
    run" (the current demo run is deliberately excluded from the switcher
    — see `test_current_run_is_excluded_from_switch_options`)."""
    resume_at = AppTest.from_file(_UI, default_timeout=30)
    resume_at.run()  # no "state" in session — the intake/resume screen draws
    assert not resume_at.exception
    resume_box = next(s for s in resume_at.selectbox if s.label == "Resume a previous run")

    resumable_ids = {summary.run_id for summary in persistence.list_runs()}
    assert resumable_ids == {_RUN_B, _RUN_C}
    assert len(resume_box.options) == len(resumable_ids) + 1  # + the "Start a new run" sentinel

    switch_at = _app(saved_runs)
    current_id = switch_at.session_state["state"].run_id
    switch_box = _switcher_selectbox(switch_at)
    assert len(switch_box.options) == len(resumable_ids - {current_id})


def test_switching_does_not_create_a_new_run_or_checkpoint(saved_runs, monkeypatch):
    """Switching is a pure read: it must not write ANY new checkpoint file
    to disk, for the current run or the target one."""
    saves: list[str] = []
    real_save = persistence.save_state

    def counting(state):
        saves.append(state.run_id)
        return real_save(state)

    at = _app(saved_runs)
    before = {p.name for p in Path(saved_runs).iterdir()}

    monkeypatch.setattr(persistence, "save_state", counting)
    at = _apply(_pick_run(at, _B_NEEDLE))

    after = {p.name for p in Path(saved_runs).iterdir()}
    assert before == after       # no new run directory appeared
    assert saves == []           # and nothing was ever (re)persisted
    assert at.session_state["state"].run_id == _RUN_B


def test_current_run_is_excluded_from_switch_options(saved_runs):
    """The demo state `_app` seeds as current is held only in
    `session_state` (never persisted), so by default it can never appear in
    `list_runs()` regardless of the exclusion filter. Persist it here too,
    so this test actually exercises the `summary.run_id != state.run_id`
    filter rather than trivially passing because there was nothing to
    filter in the first place."""
    current = build_demo_state("pass")
    persistence.save_state(current)
    assert len(persistence.list_runs()) == 3  # current + B + C, all on disk

    at = AppTest.from_file(_UI, default_timeout=30)
    at.session_state["state"] = current
    at.session_state["switch_run_expander"] = True
    at.run()
    assert not at.exception

    box = _switcher_selectbox(at)
    assert len(box.options) == 2  # every OTHER persisted run — current excluded


def test_failed_historical_run_remains_selectable(saved_runs):
    """`list_runs()` never filters by stage — a run that stopped FAILED is
    just as resumable/switchable as a finished or mid-flight one."""
    from pipeline.state import Stage, new_run

    failed = new_run("A run that blew up mid-design.")
    failed.run_id = "20260104T090000Z-switchfailed"
    failed.stage = Stage.FAILED
    persistence.save_state(failed)

    at = _app(saved_runs)
    box = _switcher_selectbox(at)
    assert any("A run that blew up" in option for option in box.options)

    target = next(o for o in box.options if "A run that blew up" in o)
    box.set_value(target)
    at.run()
    at = _apply(at)

    assert at.session_state["state"].run_id == failed.run_id


def test_empty_history_shows_a_clean_disabled_state(tmp_path, monkeypatch):
    """No other saved runs at all: a clear caption, never a selectbox with
    nothing meaningful in it (which would look like a broken dropdown)."""
    monkeypatch.setenv("AI_ARCHITECT_RUNS_DIR", str(tmp_path / "runs"))
    only_run = build_demo_state("pass")
    persistence.save_state(only_run)

    at = AppTest.from_file(_UI, default_timeout=30)
    at.session_state["state"] = only_run
    at.session_state["switch_run_expander"] = True
    at.run()

    assert not at.exception
    assert not any(s.label == "Saved runs" for s in at.selectbox)
    assert any(
        "No other saved runs" in c.value for c in at.caption
    )


# ── 6. the existing "Resume a previous run" flow stays intact ─────────────


def test_resume_picker_still_lists_and_loads_a_saved_run(saved_runs):
    """No current run yet ("Start a new run" screen): the picker lists the
    saved fixtures and loading one makes it the CURRENT run, through the
    same `persistence.load_state` the switcher uses."""
    at = AppTest.from_file(_UI, default_timeout=30)
    at.run()  # nothing in session_state["state"] — the intake screen draws
    assert not at.exception

    box = next(s for s in at.selectbox if s.label == "Resume a previous run")
    target = next(o for o in box.options if _B_NEEDLE in o)
    box.set_value(target)
    at.run()
    assert not at.exception

    assert at.session_state["state"].run_id == _RUN_B


def test_resume_picker_default_falls_through_to_the_intake_form(saved_runs):
    """The zero-click default (index 0, the sentinel) must NOT auto-load a
    run — the intake form still draws when nothing is explicitly chosen."""
    at = AppTest.from_file(_UI, default_timeout=30)
    at.run()
    assert not at.exception
    assert "state" not in at.session_state or at.session_state["state"] is None
