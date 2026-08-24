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


def test_resume_picker_offers_all_four_bundled_runs(runs_dir):
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
    # All four bundled runs are distinct entries in the picker (their
    # timestamps differ even where the prompt excerpt repeats).
    assert len(box.options) >= len(demo_runs.BUNDLED_RUN_IDS)
    joined = " ".join(box.options)
    assert "modernize an existing e-commerce monolith" in joined.lower()
    # the holdout run's own prompt excerpt must surface too
    assert "e-commerce application" in joined.lower()


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
    assert loaded.run_id in demo_runs.BUNDLED_RUN_IDS
    on_disk = persistence.load_state(loaded.run_id)
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
    claimed_id = demo_runs.BUNDLED_RUN_IDS[0]
    existing = new_run("A user's own, unrelated project.")
    existing.run_id = claimed_id
    persistence.save_state(existing)

    seeded = demo_runs.seed_bundled_demo_runs()
    # The claimed id was NOT copied over; every OTHER bundled id still seeds.
    assert claimed_id not in seeded
    assert set(seeded) == set(demo_runs.BUNDLED_RUN_IDS) - {claimed_id}

    reloaded = persistence.load_state(claimed_id)
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


def test_bundled_run_ids_are_exactly_the_curated_four():
    """The bundle is EXACTLY the curated set: one final clean Gemini demo
    plus the three good historical Claude runs (see EVAL_DEMO_RUNS.md) —
    and explicitly NOT the older pre-hardening Gemini runs."""
    assert set(demo_runs.BUNDLED_RUN_IDS) == {
        "20260824T141045Z-e1cdb3ee",  # Gemini — final clean demo
        "20260823T171738Z-935663e4",  # Claude — historical monolith
        "20260823T192225Z-5e6ac35c",  # Claude — historical robustness
        "20260824T084650Z-4447a662",  # Claude — historical holdout
    }
    # The older Gemini runs are deliberately NOT bundled: both predate the
    # final safeguards (unsupported numeric targets / earlier contract
    # behavior) — historically useful, not final demos.
    for stale in ("20260824T130359Z-b1ba2568", "20260823T155114Z-d1bc5c19"):
        assert stale not in demo_runs.BUNDLED_RUN_IDS
        assert not (_BUNDLE_ROOT / stale).exists()


def _bundled_run_data(run_id: str) -> dict:
    files = sorted((_BUNDLE_ROOT / run_id).glob("*.json"))
    assert files, f"no bundled checkpoint for {run_id}"
    return json.loads(files[-1].read_text(encoding="utf-8"))


def _history_models(data: dict) -> set[str]:
    models = set()
    for step in data.get("history", []):
        if step.get("model"):
            models.update(m.strip() for m in step["model"].split(","))
    return models


_GEMINI = "20260824T141045Z-e1cdb3ee"
_CLAUDE_MONOLITH = (
    "20260823T171738Z-935663e4",
    "20260823T192225Z-5e6ac35c",
)
_CLAUDE_HOLDOUT = "20260824T084650Z-4447a662"
_MONOLITH_REPO = "https://github.com/harsh020/ecommerce-monolith"
_HOLDOUT_REPO = "https://github.com/ttulka/ddd-example-ecommerce"


def test_gemini_run_metadata_resolves_to_google_gemini():
    data = _bundled_run_data(_GEMINI)
    models = _history_models(data)
    assert any("gemini" in m.lower() for m in models)
    assert not any("claude" in m.lower() for m in models)


def test_each_claude_run_metadata_resolves_to_anthropic_claude():
    """The A/B routing reroutes ONLY Architect + Reviewer; the Clarifier and
    repo ingestor legitimately stay on Gemini — so a Claude run's history
    must contain claude-opus on the design/review steps while the Gemini
    demo above has none."""
    for run_id in (*_CLAUDE_MONOLITH, _CLAUDE_HOLDOUT):
        data = _bundled_run_data(run_id)
        models = _history_models(data)
        assert any("claude" in m.lower() for m in models), run_id
        assert any("claude-opus" in m.lower() for m in models), run_id
        # Architect and Reviewer steps specifically ran on Claude.
        design_review_models = {
            m.strip()
            for step in data.get("history", [])
            if step.get("agent") in ("architect", "reviewer") and step.get("model")
            for m in step["model"].split(",")
        }
        assert design_review_models == {"claude-opus-5"}, run_id


def test_holdout_run_uses_the_holdout_repository():
    data = _bundled_run_data(_CLAUDE_HOLDOUT)
    assert data["initial_request"]["repo_url"] == _HOLDOUT_REPO
    assert data["initial_request"]["repo_url"] != _MONOLITH_REPO


def test_main_claude_runs_use_the_canonical_monolith_repository():
    for run_id in _CLAUDE_MONOLITH:
        data = _bundled_run_data(run_id)
        assert data["initial_request"]["repo_url"] == _MONOLITH_REPO, run_id


def test_historical_claude_runs_are_not_upgraded_to_final_code_output():
    """The three Claude runs predate the material-grounding / DecisionTopic
    hardening, and are bundled AS historical: the stored JSON must carry no
    invented `decision_topics` or ADR `decision_evidence_topics` mappings —
    fields the current schema knows but the model never produced back then.
    Current Pydantic additive defaults may fill them at LOAD time, but the
    bundled artifact itself must stay byte-honest to its generation time."""
    for run_id in (*_CLAUDE_MONOLITH, _CLAUDE_HOLDOUT):
        data = _bundled_run_data(run_id)
        assert "decision_topics" not in data, run_id
        for adr in data.get("adrs", []):
            assert "decision_evidence_topics" not in adr, (run_id, adr.get("id"))


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


# ── 14: a missing FINAL Claude run is reported, never substituted ───────


def test_manifest_keeps_final_claude_run_pending_not_substituted():
    manifest = (_REPO_ROOT / "EVAL_DEMO_RUNS.md").read_text(encoding="utf-8")
    flattened = " ".join(manifest.split())
    # The fresh final Claude run stays PENDING...
    assert "PENDING" in manifest
    # ...the historical runs are explicitly labeled as substitutes-for-nothing...
    assert "must not be substituted" in flattened
    # ...and the qualitative-vs-quantitative caveat is stated verbatim.
    assert "curated qualitative examples" in flattened
    assert "not the final quantitative 2×2 evaluation sample" in flattened


def test_manifest_classifies_each_bundled_run():
    manifest = (_REPO_ROOT / "EVAL_DEMO_RUNS.md").read_text(encoding="utf-8")
    for run_id in demo_runs.BUNDLED_RUN_IDS:
        assert run_id in manifest, run_id
    for classification in (
        "Final clean Gemini demo",
        "Historical Claude monolith",
        "Historical Claude robustness",
        "Historical Claude holdout",
    ):
        assert classification in manifest, classification


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
