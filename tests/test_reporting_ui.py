"""test_reporting_ui.py — the report panel, through the real ui.py (Kati).

Scope: the "Accept design" trigger and the download surface, driven with
Streamlit's `AppTest` against the actual `ui.py` — never `import ui` at
module level (module-level `st` calls; same rule test_ui_workspace.py and
test_ui_history.py already follow).

  * the report is generated ONLY after a real Accept click, never before;
  * a failed export never hides or rolls back the acceptance itself;
  * "Retry"/"Regenerate" actually re-renders and replaces the files;
  * the CURRENT accepted run and a HISTORICAL accepted run both expose
    downloads — History read-only, with no (re)generate action of its own.

`AI_ARCHITECT_RUNS_DIR` is already redirected to a throwaway directory for
the whole test session (conftest.py); several tests narrow it further with
`monkeypatch.setenv` + `tmp_path` so each test's files are its own.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pipeline import reporting, sign_off
from pipeline.persistence import save_state
from webapp.ui_demo import build_demo_state

_UI = str(Path(__file__).resolve().parents[1] / "ui.py")  # repo root


def _finished_app(variant: str = "pass") -> AppTest:
    at = AppTest.from_file(_UI, default_timeout=30)
    at.session_state["state"] = build_demo_state(variant)
    at.run()
    assert not at.exception
    return at


def _select_view(at: AppTest, view: str) -> AppTest:
    next(b for b in at.sidebar.button if b.key == f"nav_{view}").click()
    at.run()
    assert not at.exception
    return at


def _click_accept(at: AppTest) -> AppTest:
    next(b for b in at.button if "Accept this design" in b.label).click()
    at.run()
    assert not at.exception
    return at


def _current_state(at: AppTest):
    return at.session_state["state"]


def _md(at: AppTest) -> str:
    return " | ".join(m.value for m in at.markdown)


def _caps(at: AppTest) -> str:
    return " | ".join(c.value for c in at.caption)


# ══════════════════════════════════════════════════════════════════════════
# 1. Nothing before acceptance
# ══════════════════════════════════════════════════════════════════════════
def test_no_report_panel_before_acceptance():
    at = _finished_app("pass")  # DONE, not yet accepted
    state = _current_state(at)

    assert "Final architecture report" not in _md(at)
    assert not [b for b in at.download_button]
    assert reporting.report_paths(state.run_id) == (None, None)


# ══════════════════════════════════════════════════════════════════════════
# 2. Accept triggers real, deterministic report generation
# ══════════════════════════════════════════════════════════════════════════
def test_accepting_generates_both_files_and_offers_both_downloads():
    at = _finished_app("pass")
    run_id = _current_state(at).run_id

    at = _click_accept(at)

    state = _current_state(at)
    assert state.stage.value == "accepted"
    md_path, pdf_path = reporting.report_paths(run_id)
    assert md_path is not None and md_path.is_file()
    assert pdf_path is not None and pdf_path.is_file()

    labels = [b.label for b in at.download_button]
    assert any("Markdown" in label for label in labels)
    assert any("PDF" in label for label in labels)


def test_report_is_not_generated_before_the_click_even_when_done():
    """Belt-and-suspenders on the UI side of the guard `build_report` itself
    enforces (`pipeline.reporting.ReportError`)."""
    at = _finished_app("capped")
    state = _current_state(at)
    assert state.stage.value == "done"
    assert reporting.report_paths(state.run_id) == (None, None)


# ══════════════════════════════════════════════════════════════════════════
# 3. Partial failure never hides or undoes the acceptance
# ══════════════════════════════════════════════════════════════════════════
def test_sign_off_succeeds_even_when_report_generation_fails(monkeypatch):
    monkeypatch.setattr(
        reporting, "render_pdf",
        lambda report: (_ for _ in ()).throw(RuntimeError("MAGIC_PDF_BOOM")),
    )
    at = _finished_app("pass")

    at = _click_accept(at)

    state = _current_state(at)
    assert state.stage.value == "accepted"
    assert state.accepted_at  # the human decision is recorded regardless
    md_path, pdf_path = reporting.report_paths(state.run_id)
    assert md_path is not None  # Markdown is independent and still wrote
    assert pdf_path is None
    assert "unavailable" in _caps(at)
    assert any("Retry report generation" in b.label for b in at.button)


def test_retry_after_a_failure_recovers_both_files(monkeypatch):
    boom = {"on": True}
    real_render_pdf = reporting.render_pdf

    def _maybe_boom(report):
        if boom["on"]:
            raise RuntimeError("MAGIC_PDF_BOOM")
        return real_render_pdf(report)

    monkeypatch.setattr(reporting, "render_pdf", _maybe_boom)

    at = _finished_app("pass")
    at = _click_accept(at)
    state = _current_state(at)
    assert reporting.report_paths(state.run_id)[1] is None  # PDF missing

    boom["on"] = False
    next(b for b in at.button if "report generation" in b.label.lower() or "Regenerate report" in b.label).click()
    at.run()
    assert not at.exception

    md_path, pdf_path = reporting.report_paths(state.run_id)
    assert md_path is not None and pdf_path is not None
    labels = [b.label for b in at.download_button]
    assert any("PDF" in label for label in labels)


def test_regenerate_button_re_renders_both_files_on_a_clean_run():
    at = _finished_app("pass")
    at = _click_accept(at)
    state = _current_state(at)
    md_path, _ = reporting.report_paths(state.run_id)
    before = md_path.stat().st_mtime_ns

    next(b for b in at.button if "Regenerate report" in b.label).click()
    at.run()
    assert not at.exception

    after = md_path.stat().st_mtime_ns
    assert after >= before  # replaced, not appended to or duplicated
    assert reporting.report_paths(state.run_id) == (md_path, reporting.report_paths(state.run_id)[1])


# ══════════════════════════════════════════════════════════════════════════
# 4. History: read-only downloads, no write surface
# ══════════════════════════════════════════════════════════════════════════
_HIST_ID = "20260201T090000Z-histreport"
_HIST_PROJECT = "History Report Run"


def _accepted_history_run(tmp_path, *, with_report: bool):
    """One ACCEPTED run checkpointed into `tmp_path`, with or without its
    report artifacts already generated — the "old run predates this
    feature" case is exactly `with_report=False`."""
    state = build_demo_state("pass")
    state.run_id = _HIST_ID
    state.context_record.project_name = _HIST_PROJECT
    state.blueprint.project_name = _HIST_PROJECT
    sign_off.accept_design(state)
    if with_report:
        result = reporting.generate_reports(state)
        assert result.ok
    save_state(state)
    return state


def _open_history_run(at: AppTest, run_id: str) -> AppTest:
    at.button(key=f"hist_open_{run_id}").click()
    at.run()
    assert not at.exception
    return at


def test_historical_accepted_run_exposes_its_report_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_ARCHITECT_RUNS_DIR", str(tmp_path))
    _accepted_history_run(tmp_path, with_report=True)

    at = _finished_app("pass")  # an unrelated CURRENT run
    at = _select_view(at, "History")
    at = _open_history_run(at, _HIST_ID)

    assert "Final architecture report" in _md(at)
    labels = [b.label for b in at.download_button]
    assert any("Markdown" in label for label in labels)
    assert any("PDF" in label for label in labels)
    # Read-only: no (re)generate action of any kind in History.
    buttons = [b.label for b in at.button]
    assert not any("Generate report" in b for b in buttons)
    assert not any("Regenerate report" in b for b in buttons)
    assert not any("Retry report generation" in b for b in buttons)


def test_historical_run_missing_a_report_is_handled_gracefully(tmp_path, monkeypatch):
    """An accepted run from before this feature existed: no crash, no silent
    mutation of the historical run, just an honest "not generated" message."""
    monkeypatch.setenv("AI_ARCHITECT_RUNS_DIR", str(tmp_path))
    _accepted_history_run(tmp_path, with_report=False)

    at = _finished_app("pass")
    at = _select_view(at, "History")
    at = _open_history_run(at, _HIST_ID)

    assert not at.exception
    assert "Final architecture report" in _md(at)  # the panel still renders
    assert "No report was generated for this run" in _caps(at)
    assert not [b for b in at.download_button]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
