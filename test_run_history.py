"""test_run_history.py — tests for the History view's read-only data layer.

Scope: run_history.py ONLY — discovery of saved runs under `.cache/runs/`,
newest-first ordering, tolerance of malformed run folders, the summary
metadata extracted per run, and the on-demand load of one selected run.

Every test writes its checkpoints through the REAL `persistence.save_state`
into a per-test temp directory (via AI_ARCHITECT_RUNS_DIR, which
`persistence.runs_dir()` re-reads on every call) — never the developer
machine's real cache, and never a mocked format. What lands on disk is
therefore exactly what the orchestrator writes, which is the only thing
run_history is allowed to expect.

READ-ONLY is asserted structurally: none of these tests patch a writer
because run_history imports none — the module calls nothing but the
filesystem and `ArchitectState` validation. The mtime-fallback and
walk-back behaviours are exercised by writing files by hand.
"""

from __future__ import annotations

import os

import pytest

from pipeline import persistence
from pipeline.state import ArchitectState, Stage, StepLog, new_run
from run_history import HistoryError, list_history_runs, load_history_run


@pytest.fixture()
def runs_dir(tmp_path, monkeypatch):
    """A per-test runs directory that does not exist until written to."""
    directory = tmp_path / "runs"
    monkeypatch.setenv("AI_ARCHITECT_RUNS_DIR", str(directory))
    return directory


def _run(
    run_id: str,
    prompt: str,
    *,
    stage: Stage = Stage.DONE,
    last_step_at: str = "2026-01-01T10:00:00+00:00",
) -> ArchitectState:
    """A minimal state with ONE trace step carrying an explicit timestamp,
    so ordering tests control the saved clock instead of racing `now()`."""
    state = new_run(prompt)
    state.run_id = run_id
    state.history.append(
        StepLog(
            agent="orchestrator",
            stage_in=Stage.CREATED,
            stage_out=stage,
            note="",
            timestamp=last_step_at,
        )
    )
    state.stage = stage
    return state


# ── 1. missing / empty .cache/runs/ ────────────────────────────────────────


def test_missing_runs_dir_is_an_empty_list_not_an_error(runs_dir):
    assert not runs_dir.exists()  # never created by the test
    assert list_history_runs() == []


def test_empty_runs_dir_is_an_empty_list(runs_dir):
    runs_dir.mkdir(parents=True)
    assert list_history_runs() == []


# ── 2. chronology — newest first ──────────────────────────────────────────


def test_multiple_runs_are_ordered_newest_first(runs_dir):
    persistence.save_state(
        _run("20260101T100000Z-older", "older run", last_step_at="2026-01-01T10:00:00+00:00")
    )
    persistence.save_state(
        _run("20260102T100000Z-newer", "newer run", last_step_at="2026-01-02T10:00:00+00:00")
    )

    summaries = list_history_runs()
    assert [s.run_id for s in summaries] == [
        "20260102T100000Z-newer",
        "20260101T100000Z-older",
    ]


def test_saved_timestamp_beats_filesystem_mtime(runs_dir):
    """The run's OWN last-step timestamp orders the list; a copied/touched
    file with a misleading mtime must not reorder it."""
    persistence.save_state(
        _run("20260601T000000Z-june", "june run", last_step_at="2026-06-01T00:00:00+00:00")
    )
    persistence.save_state(
        _run("20260701T000000Z-july", "july run", last_step_at="2026-07-01T00:00:00+00:00")
    )
    # Forge june's mtime a year into the future — filesystem says "newest".
    june_ckpt = sorted(runs_dir.joinpath("20260601T000000Z-june").glob("*.json"))[-1]
    future = june_ckpt.stat().st_mtime + 365 * 24 * 3600
    os.utime(june_ckpt, (future, future))

    assert [s.run_id for s in list_history_runs()] == [
        "20260701T000000Z-july",
        "20260601T000000Z-june",
    ]


def test_mtime_fallback_when_checkpoint_has_no_trace(runs_dir):
    """A run with no trace steps (a lone created-checkpoint) still lists,
    dated from the filesystem — and says so via `from_saved_timestamp`."""
    persistence.save_state(new_run("crashed before the first transition"))

    summaries = list_history_runs()
    assert len(summaries) == 1
    assert summaries[0].updated_at  # SOMETHING was dated
    assert summaries[0].from_saved_timestamp is False


# ── 3. malformed / incomplete run folders ─────────────────────────────────


def test_malformed_run_folders_are_skipped_without_crashing(runs_dir):
    runs_dir.mkdir(parents=True)
    # A directory with no checkpoints at all.
    (runs_dir / "20260101T000000Z-empty").mkdir()
    # A directory whose only .json does not match the naming convention.
    stray = runs_dir / "20260101T000000Z-stray"
    stray.mkdir()
    (stray / "not_a_checkpoint.json").write_text("{}", encoding="utf-8")
    # A directory whose every checkpoint is corrupt JSON.
    corrupt = runs_dir / "20260101T000000Z-corrupt"
    corrupt.mkdir()
    (corrupt / "001_created.json").write_text("{ not json", encoding="utf-8")
    # A plain file in the runs root is not a run.
    (runs_dir / "README.md").write_text("not a run", encoding="utf-8")

    assert list_history_runs() == []  # tolerated, not raised


def test_newest_corrupt_checkpoint_falls_back_to_the_older_one(runs_dir):
    """History walks the audit trail backwards: a damaged last write does
    not hide the good checkpoints below it."""
    state = _run("20260101T100000Z-run", "the real run")
    persistence.save_state(state)

    run_dir = runs_dir / "20260101T100000Z-run"
    (run_dir / "002_done.json").write_text("{ broken", encoding="utf-8")

    summaries = list_history_runs()
    assert [s.run_id for s in summaries] == ["20260101T100000Z-run"]
    assert summaries[0].stage == "done"  # from 001_done, not the corrupt 002

    loaded = load_history_run("20260101T100000Z-run")
    assert loaded == state  # the last GOOD state, intact


def test_run_with_only_corrupt_checkpoints_fails_to_load_loudly(runs_dir):
    run_dir = runs_dir / "20260101T000000Z-dead"
    run_dir.mkdir(parents=True)
    (run_dir / "001_created.json").write_text("][ broken", encoding="utf-8")

    with pytest.raises(HistoryError):
        load_history_run("20260101T000000Z-dead")


# ── 4. summary metadata ───────────────────────────────────────────────────


def test_summary_extracts_saved_metadata(runs_dir, monkeypatch):
    from ui_demo import build_demo_state

    state = build_demo_state("pass")  # real shapes: record, repo, review…
    state.run_id = "20260101T120000Z-full"
    persistence.save_state(state)

    (summary,) = list_history_runs()
    assert summary.run_id == "20260101T120000Z-full"
    assert summary.project_label == "Seasonal Shop"
    assert summary.repo_name == "seasonal-shop"
    assert summary.stage == "done"
    assert summary.verdict == "pass"
    assert summary.accepted is False
    assert summary.feature_count == 2
    assert summary.adr_count == 2
    assert summary.component_count == 2
    assert summary.kb_chunk_count == 4
    assert summary.from_saved_timestamp is True  # the demo trace carries one


def test_summary_reports_acceptance(runs_dir):
    from pipeline.sign_off import accept_design

    state = _run("20260101T110000Z-taken", "a run a human took")
    accept_design(state, note="")
    persistence.save_state(state)

    (summary,) = list_history_runs()
    assert summary.accepted is True
    assert summary.stage == "accepted"


def test_summary_uses_neutral_fallbacks_for_missing_metadata(runs_dir):
    persistence.save_state(
        _run("20260101T090000Z-bare", "Architect a ticketing system for concerts.")
    )

    (summary,) = list_history_runs()
    assert summary.project_label.startswith("Architect a ticketing system")
    assert summary.repo_name == ""           # greenfield — no fabrication
    assert summary.verdict == ""             # never reviewed
    assert summary.accepted is False
    assert summary.feature_count == 0
    assert summary.adr_count == 0
    assert summary.component_count == 0
    assert summary.kb_chunk_count == 0


# ── 5. loading one selected run ───────────────────────────────────────────


def test_load_returns_exactly_the_selected_run(runs_dir):
    a = _run("20260101T100000Z-a", "run a")
    b = _run("20260101T110000Z-b", "run b")
    persistence.save_state(a)
    persistence.save_state(b)

    loaded = load_history_run("20260101T100000Z-a")
    assert isinstance(loaded, ArchitectState)
    assert loaded == a
    assert loaded.run_id != b.run_id


def test_load_unknown_run_raises(runs_dir):
    with pytest.raises(HistoryError):
        load_history_run("20260101T000000Z-neverexisted")


def test_load_rejects_path_tricks_in_run_id(runs_dir):
    for bad in ("../elsewhere", "runs/other", "", "."):
        with pytest.raises(HistoryError):
            load_history_run(bad)
