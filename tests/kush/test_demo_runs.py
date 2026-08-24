"""test_demo_runs.py — bundled evaluation demo runs (Kush): seeding,
discoverability through the canonical persistence/history code, sanitization,
and the one-shot/pipeline-run separation.

Every test redirects `AI_ARCHITECT_RUNS_DIR` to a per-test temp directory
(never the real `.cache/runs`); the SOURCE bundle under `demo_runs/` is the
real, committed one — reading it is safe, since seeding only ever COPIES
from there, never writes back into it. No live LLM call anywhere in this
file, and none of it needs one: everything bundled is already a finished,
recorded run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import demo_runs
from pipeline import persistence
from pipeline.state import ArchitectState, Stage, new_run
from run_history import list_history_runs

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UI = str(_REPO_ROOT / "ui.py")
_BUNDLE_ROOT = demo_runs.BUNDLE_ROOT


@pytest.fixture()
def runs_dir(tmp_path, monkeypatch):
    """A per-test runs directory that does not exist until seeded/written to."""
    directory = tmp_path / "runs"
    monkeypatch.setenv("AI_ARCHITECT_RUNS_DIR", str(directory))
    return directory


# ── 1/2: discoverable through the CANONICAL persistence functions ────────


def test_seeded_run_is_discoverable_through_list_runs(runs_dir):
    seeded = demo_runs.seed_bundled_demo_runs()
    assert seeded == list(demo_runs.BUNDLED_RUN_IDS)

    ids = [summary.run_id for summary in persistence.list_runs()]
    for run_id in demo_runs.BUNDLED_RUN_IDS:
        assert run_id in ids


def test_seeded_run_loads_through_canonical_load_state(runs_dir):
    demo_runs.seed_bundled_demo_runs()

    for run_id in demo_runs.BUNDLED_RUN_IDS:
        state = persistence.load_state(run_id)
        assert isinstance(state, ArchitectState)
        assert state.run_id == run_id
        assert state.stage is Stage.DONE
        assert state.review is not None and state.review.overall_status == "pass"


# ── 3/4/5: History / Switch run / Resume all see it, with NO code changes ─


def test_history_view_sees_the_bundled_run(runs_dir):
    demo_runs.seed_bundled_demo_runs()
    ids = [summary.run_id for summary in list_history_runs()]
    for run_id in demo_runs.BUNDLED_RUN_IDS:
        assert run_id in ids


def test_resume_picker_lists_the_bundled_run(runs_dir):
    # ui.py's own auto-seed only fires against the TRUE default runs dir
    # (see its guard on `AI_ARCHITECT_RUNS_DIR`) — this fixture redirects
    # that on purpose, so isolated tests are never polluted by it (see
    # `test_ui_auto_seed_is_skipped_when_runs_dir_is_redirected` below).
    # Seed the isolated dir explicitly to test what the DOC actually
    # requires: once a bundled run is on disk, the existing resume picker
    # shows it with no code of its own needing to change.
    demo_runs.seed_bundled_demo_runs()

    at = AppTest.from_file(_UI, default_timeout=30)
    at.run()
    assert not at.exception

    box = next(s for s in at.selectbox if s.label == "Resume a previous run")
    joined = " ".join(box.options)
    assert "modernize an existing e-commerce monolith" in joined.lower()


def test_switch_run_lists_the_bundled_run(runs_dir):
    from ui_demo import build_demo_state

    demo_runs.seed_bundled_demo_runs()

    at = AppTest.from_file(_UI, default_timeout=30)
    at.session_state["state"] = build_demo_state("pass")
    at.session_state["switch_run_expander"] = True
    at.run()
    assert not at.exception

    box = next(s for s in at.selectbox if s.label == "Saved runs")
    joined = " ".join(box.options)
    assert "modernize an existing e-commerce monolith" in joined.lower()


def test_resuming_the_bundled_run_loads_its_exact_stored_artifacts(runs_dir):
    demo_runs.seed_bundled_demo_runs()

    at = AppTest.from_file(_UI, default_timeout=30)
    at.run()
    assert not at.exception

    box = next(s for s in at.selectbox if s.label == "Resume a previous run")
    target = next(o for o in box.options if "e-commerce monolith" in o.lower())
    box.set_value(target)
    at.run()
    assert not at.exception

    loaded = at.session_state["state"]
    on_disk = persistence.load_state(demo_runs.BUNDLED_RUN_IDS[0])
    assert loaded.context_record == on_disk.context_record
    assert loaded.blueprint == on_disk.blueprint
    assert loaded.adrs == on_disk.adrs
    assert loaded.components == on_disk.components
    assert loaded.review == on_disk.review
    assert loaded.retrieved_knowledge == on_disk.retrieved_knowledge


# ── 6/7: idempotent, never overwrites ──────────────────────────────────────


def test_seeding_twice_is_idempotent(runs_dir):
    first = demo_runs.seed_bundled_demo_runs()
    assert first == list(demo_runs.BUNDLED_RUN_IDS)

    checkpoint_dir = runs_dir / demo_runs.BUNDLED_RUN_IDS[0]
    before = {p.name: p.read_bytes() for p in checkpoint_dir.glob("*.json")}

    second = demo_runs.seed_bundled_demo_runs()
    assert second == []  # nothing copied the second time

    after = {p.name: p.read_bytes() for p in checkpoint_dir.glob("*.json")}
    assert before == after  # byte-identical — not touched at all


def test_seeding_never_overwrites_an_existing_run_with_the_same_id(runs_dir):
    """Defensive: even if a run directory already exists under a bundled
    run's id (a real run somehow landed there, or a previous seed used a
    different snapshot), seeding must leave it exactly alone."""
    run_id = demo_runs.BUNDLED_RUN_IDS[0]
    existing = new_run("A user's own, unrelated project.")
    existing.run_id = run_id
    persistence.save_state(existing)

    seeded = demo_runs.seed_bundled_demo_runs()
    assert seeded == []  # the id was already claimed — nothing copied over it

    reloaded = persistence.load_state(run_id)
    assert reloaded.initial_request.raw_prompt == "A user's own, unrelated project."


def test_seeding_coexists_with_a_users_own_unrelated_runs(runs_dir):
    mine = new_run("My own architecture request.")
    persistence.save_state(mine)

    demo_runs.seed_bundled_demo_runs()

    ids = {summary.run_id for summary in persistence.list_runs()}
    assert mine.run_id in ids
    for run_id in demo_runs.BUNDLED_RUN_IDS:
        assert run_id in ids


# ── 8: opening a bundled run never mutates it ──────────────────────────────


def test_viewing_the_bundled_run_does_not_write_a_new_checkpoint(runs_dir):
    demo_runs.seed_bundled_demo_runs()
    checkpoint_dir = runs_dir / demo_runs.BUNDLED_RUN_IDS[0]
    before = sorted(p.name for p in checkpoint_dir.glob("*.json"))

    # Load it the way every viewing path does — resume, switch, and History
    # all end in one of these two calls, never a write.
    persistence.load_state(demo_runs.BUNDLED_RUN_IDS[0])
    for run_id in demo_runs.BUNDLED_RUN_IDS:
        from run_history import load_history_run

        load_history_run(run_id)

    after = sorted(p.name for p in checkpoint_dir.glob("*.json"))
    assert before == after


# ── 9/10/11: sanitization / security scan ─────────────────────────────────

_FORBIDDEN_PATTERNS: dict[str, re.Pattern[str]] = {
    "Google API key": re.compile(r"AIza[0-9A-Za-z_\-]{10,}"),
    "Anthropic API key": re.compile(r"sk-ant-[0-9A-Za-z_\-]{10,}"),
    "generic sk- style key": re.compile(r"\bsk-[0-9A-Za-z_\-]{10,}"),
    "Kush's absolute Windows path": re.compile(r"C:\\Users\\Kusha", re.IGNORECASE),
    "any absolute Windows user path": re.compile(r"[A-Za-z]:\\\\Users\\\\[^\\\\\"]+"),
    ".env reference": re.compile(r"\.env\b"),
    "authorization header field": re.compile(r'"[Aa]uthorization"\s*:'),
    "cookie field": re.compile(r'"[Cc]ookie"\s*:'),
}


def _bundled_files() -> list[Path]:
    files = sorted(_BUNDLE_ROOT.glob("*/*.json"))
    assert files, "expected at least one bundled checkpoint on disk"
    return files


def test_bundled_artifacts_contain_no_forbidden_patterns():
    for path in _bundled_files():
        text = path.read_text(encoding="utf-8")
        for name, pattern in _FORBIDDEN_PATTERNS.items():
            assert not pattern.search(text), f"{name} found in {path}"


def test_bundled_artifacts_parse_as_real_architect_state():
    """Not just "no secrets" — still a genuine, schema-valid recorded run."""
    for path in _bundled_files():
        state = ArchitectState.model_validate_json(path.read_text(encoding="utf-8"))
        assert state.stage is Stage.DONE
        assert state.review is not None and state.review.overall_status == "pass"


def test_sanitize_checkpoint_replaces_the_absolute_clone_path():
    dirty = {
        "repo_representation": {
            "meta": {
                "clone_path": r"C:\Users\Kusha\Documents\proj\.cache\repos\shop-abc123",
            }
        }
    }
    clean = demo_runs.sanitize_checkpoint(dirty)
    assert clean["repo_representation"]["meta"]["clone_path"] == ".cache/repos/shop-abc123"
    assert "Kusha" not in json.dumps(clean)


def test_sanitize_checkpoint_does_not_mutate_its_argument():
    dirty = {"repo_representation": {"meta": {"clone_path": r"C:\Users\Kusha\x"}}}
    before = json.dumps(dirty)
    demo_runs.sanitize_checkpoint(dirty)
    assert json.dumps(dirty) == before


def test_sanitize_checkpoint_leaves_a_missing_repo_representation_alone():
    assert demo_runs.sanitize_checkpoint({}) == {}


def test_sanitize_checkpoint_does_not_touch_authentic_architecture_content():
    """The sanitizer targets ONE known machine-local field — it must not
    also mangle real recorded output that happens to mention a path-like
    string (see the module docstring on why this is narrow, not a blanket
    regex substitution)."""
    data = {
        "context_record": {"business_goal": "Migrate off the old C:\\legacy\\ export tool."},
        "adrs": [{"decision": "Use path-style routing, not query params."}],
    }
    clean = demo_runs.sanitize_checkpoint(data)
    assert clean == data


# ── 12: only the explicitly selected run ids are bundled ─────────────────


def test_only_explicitly_selected_run_ids_are_bundled_on_disk():
    on_disk = {p.name for p in _BUNDLE_ROOT.iterdir() if p.is_dir()}
    assert on_disk == set(demo_runs.BUNDLED_RUN_IDS)


def test_bundled_run_ids_do_not_include_the_known_stale_claude_run():
    # 20260824T084650Z-4447a662 predates the material-grounding hardening
    # AND used a different repository than the canonical scenario — see
    # EVAL_DEMO_RUNS.md's PENDING section. It must never be silently bundled.
    assert "20260824T084650Z-4447a662" not in demo_runs.BUNDLED_RUN_IDS


def test_no_claude_run_is_currently_bundled():
    """Pins the current, honest state: only the Gemini run is ready. This
    test is EXPECTED to start failing the day a real, fresh Claude run is
    bundled — at which point it should be updated, not worked around."""
    bundled_models = set()
    for path in _bundled_files():
        data = json.loads(path.read_text(encoding="utf-8"))
        for step in data.get("history", []):
            if step.get("model"):
                bundled_models.update(m.strip() for m in step["model"].split(","))
    assert not any("claude" in m.lower() for m in bundled_models)
    assert any("gemini" in m.lower() for m in bundled_models)


# ── 13: one-shots are not pipeline runs ────────────────────────────────────


def test_one_shots_folder_is_not_a_bundled_run_source():
    one_shots = _REPO_ROOT / "eval" / "one_shots"
    assert one_shots.is_dir()
    # Nothing under eval/one_shots/ is a `demo_runs.BUNDLED_RUN_IDS` member —
    # the two are structurally different locations on purpose.
    assert demo_runs.BUNDLE_ROOT != one_shots
    assert not any(one_shots.rglob("*.json"))  # nothing generated/bundled yet


def test_one_shots_never_appear_in_pipeline_run_listings(runs_dir):
    demo_runs.seed_bundled_demo_runs()
    ids = {summary.run_id for summary in persistence.list_runs()}
    assert "one_shots" not in ids
    assert not any("one_shot" in run_id.lower() for run_id in ids)


# ── 14: a missing final run is reported, never silently substituted ───────


def test_missing_claude_run_manifest_states_pending_not_a_substitution():
    manifest = (_REPO_ROOT / "EVAL_DEMO_RUNS.md").read_text(encoding="utf-8")
    assert "PENDING" in manifest
    assert "4447a662" in manifest  # the stale run is named, not silently dropped
    assert "must not be substituted" in manifest or "must not be" in manifest.lower()


# ── ui.py's auto-seed only ever touches the TRUE default runs dir ────────
# A test/isolated run store (any `AI_ARCHITECT_RUNS_DIR` override — see
# pipeline/persistence.py's own docstring on why that variable exists) must
# never be silently populated with bundled content it did not ask for; see
# the failures this guard fixes in test_ui_history.py / test_ui_new_run.py /
# test_ui_run_switcher.py before it existed. `demo_runs.seed_bundled_demo_
# runs` is monkeypatched with a spy so these two prove ui.py's CALL SITE
# decision without ever touching a real filesystem location.


def test_ui_auto_seed_fires_when_runs_dir_is_the_true_default(monkeypatch):
    calls = []
    monkeypatch.setattr(demo_runs, "seed_bundled_demo_runs", lambda: calls.append(1))
    monkeypatch.delenv("AI_ARCHITECT_RUNS_DIR", raising=False)

    at = AppTest.from_file(_UI, default_timeout=30)
    at.run()
    assert not at.exception
    assert calls == [1]


def test_ui_auto_seed_is_skipped_when_runs_dir_is_redirected(runs_dir, monkeypatch):
    calls = []
    monkeypatch.setattr(demo_runs, "seed_bundled_demo_runs", lambda: calls.append(1))

    at = AppTest.from_file(_UI, default_timeout=30)
    at.run()
    assert not at.exception
    assert calls == []
