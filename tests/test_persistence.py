"""test_persistence.py — offline tests for state-on-disk and resume (Kati).

No API key or network needed: every LLM call is replaced with a deterministic
canned response (the Clarifier / Architect / Reviewer mocks are reused from
test_clarifier). Covers five things:

  1. Round-trip — save → load preserves the state, history included.
  2. Resume — a run checkpointed at AWAITING_HUMAN can be reloaded from disk,
     answered, and driven to completion in the SAME run directory.
  3. Robustness — a corrupt checkpoint is skipped by `list_runs` and reported
     clearly by `load_state`; it never takes the listing down with it.
  4. Atomicity — a write either lands whole or not at all, and never leaves a
     `.tmp` behind, including when the final rename fails.
  5. The stream/invoke equivalence that `run_pipeline` is now built on.

Isolation: every test points `AI_ARCHITECT_RUNS_DIR` at its own empty temp
directory, so tests never see each other's runs and the real `.cache/runs` is
never touched. That is done with a plain helper rather than a pytest fixture so
this file still runs standalone (`python test_persistence.py`), like its
neighbours.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import test_clarifier as tc  # canned Clarifier / Architect responses
from pipeline import orchestrator
from pipeline.agents import architect as arch
from pipeline.agents import clarifier as clar
from pipeline.agents import reviewer as rev
from pipeline.persistence import (
    CheckpointError,
    _run_timestamp,
    checkpoint,
    list_runs,
    load_state,
    runs_dir,
    save_state,
)
from pipeline.state import (
    ArchitectState,
    Blueprint,
    ContextRecord,
    KBChunk,
    ReviewResult,
    RubricScores,
    Stage,
    new_run,
)

PROMPT = "Our monolithic shop falls over on sale days; re-architect it for peak load."

_TMP_ROOT = Path(tempfile.mkdtemp(prefix="aiarch_persistence_"))
atexit.register(shutil.rmtree, _TMP_ROOT, True)


def _isolate(name: str) -> Path:
    """Point persistence at a fresh, empty runs directory and return it."""
    directory = _TMP_ROOT / name
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    os.environ["AI_ARCHITECT_RUNS_DIR"] = str(directory)
    return directory


def _install_llm_mocks(clarifier=None) -> None:
    """Replace every LLM/retrieval call the pipeline can reach.

    `clarifier` defaults to the "stateful" mock: it reports missing critical
    facts until answers exist, then reports complete — which is what makes a
    pause-and-resume test possible without a human.
    """
    import architect as legacy_architect

    legacy_architect.retrieve_chunks = lambda query, k=3: (
        [{
            "content": "Use asynchronous processing to absorb peak traffic.",
            "source": "offline-test-kb",
            "page": 1,
            "box": 1,
            "distance": 0.1,
        }],
        "offline-test",
    )

    def _stateful(state, prompt, *, system="", model="", response_schema=None):
        if state.clarification_answers:
            return tc._complete(state, prompt)
        return tc._missing(state, prompt)

    clar.llm_call = clarifier or _stateful
    arch.llm_call = tc._architect_response
    rev.llm_call = lambda state, prompt, **kwargs: (rev.LLMJudgments(), tc.fake_usage())


def _rich_state() -> ArchitectState:
    """A state with something in every layer, so a round-trip has work to do."""
    state = new_run(PROMPT, "https://github.com/example/bugged-shop")
    state.log_step("repo_ingestor", Stage.INGESTING, "ingested 4 partition(s)")
    state.log_step("clarifier", Stage.AWAITING_HUMAN, "missing 2 critical fact(s)")
    state.clarifying_questions = ["Expected peak users?", "Is GDPR in scope?"]
    state.clarification_answers = {"Expected peak users?": "about 50k"}
    state.context_record = ContextRecord(
        project_name="Peak-Resilient Shop",
        cloud_provider="AWS",
        compliance_requirements=["GDPR"],
        summary="cloud: AWS; scale: 50k peak users; compliance: GDPR",
    )
    state.retrieved_knowledge = [
        KBChunk(content="Event-driven designs decouple peak load.",
                source="kb.pdf", page=12, box=1, distance=0.23)
    ]
    state.retry_counts = {"architect": 1}
    state.input_tokens = 4321
    state.output_tokens = 765
    return state


# ── 1. round-trip ────────────────────────────────────────────────────────
def test_round_trip_preserves_state_and_history():
    _isolate("round_trip")
    state = _rich_state()

    first = save_state(state)
    assert first.name == "001_awaiting_input.json"

    loaded = load_state(state.run_id)

    # Pydantic equality compares every field, so this alone proves the payload
    # survived; the explicit checks below name what we care about most.
    assert loaded == state
    assert loaded.run_id == state.run_id
    assert loaded.stage is Stage.AWAITING_HUMAN
    assert [(s.agent, s.stage_in, s.stage_out, s.note) for s in loaded.history] == [
        (s.agent, s.stage_in, s.stage_out, s.note) for s in state.history
    ]
    assert len(loaded.history) == 2
    assert loaded.context_record.compliance_requirements == ["GDPR"]
    assert loaded.retrieved_knowledge[0].distance == 0.23
    assert loaded.retry_counts == {"architect": 1}
    assert loaded.input_tokens == 4321

    # Saving again APPENDS — the trail grows, the newest file is the resume point.
    state.log_step("researcher", Stage.RESEARCHING, "retrieved 1 chunk")
    second = save_state(state)
    assert second.name == "002_researching.json"
    assert first.exists(), "earlier checkpoints are an audit trail, never overwritten"
    assert load_state(state.run_id).stage is Stage.RESEARCHING

    # ...and the run shows up in the picker's listing.
    (summary,) = list_runs()
    assert summary.run_id == state.run_id
    assert summary.stage == "researching"
    assert summary.updated_at.startswith("20")
    assert summary.raw_prompt_excerpt.startswith("Our monolithic shop")
    assert summary.project_name == "Peak-Resilient Shop"


def test_list_runs_resolves_project_name_with_fallback():
    """`project_name` comes from the checkpoint JSON alone (no `load_state`):
    context_record wins, blueprint is the fallback, else empty string — the
    picker label then degrades to the prompt excerpt."""
    _isolate("project_name")

    blueprint = Blueprint(
        project_name="Blueprint Name",
        stakeholder_view="Faster checkout for shoppers.",
        technical_view="Event-driven microservices with queues.",
    )

    record_wins = _rich_state()
    record_wins.blueprint = blueprint
    save_state(record_wins)

    blueprint_only = _rich_state()
    blueprint_only.context_record = None
    blueprint_only.blueprint = blueprint
    save_state(blueprint_only)

    neither = new_run("A run with no project name anywhere.")
    save_state(neither)

    by_id = {summary.run_id: summary.project_name for summary in list_runs()}
    assert by_id[record_wins.run_id] == "Peak-Resilient Shop"  # record beats blueprint
    assert by_id[blueprint_only.run_id] == "Blueprint Name"    # blueprint fallback
    assert by_id[neither.run_id] == ""                          # neither → empty


# ── 2. resume from AWAITING_HUMAN ────────────────────────────────────────
def test_resume_from_awaiting_input_continues():
    _isolate("resume")
    _install_llm_mocks()

    # First leg: run until the Clarifier pauses for the human.
    paused = orchestrator.run_pipeline(new_run(PROMPT))
    assert paused.stage is Stage.AWAITING_HUMAN
    assert paused.clarifying_questions

    run_id = paused.run_id
    run_dir = runs_dir() / run_id
    before = sorted(p.name for p in run_dir.glob("*.json"))
    assert before, "the pause must be on disk"
    assert before[-1].endswith("_awaiting_input.json")

    # Simulate the process dying: drop the in-memory object entirely and
    # rebuild the run from disk, exactly as the UI's resume picker does.
    del paused
    resumed = load_state(run_id)
    assert resumed.stage is Stage.AWAITING_HUMAN
    assert resumed.run_id == run_id
    questions_before = list(resumed.clarifying_questions)
    history_before = len(resumed.history)

    # Second leg: answer and drive on, from the reloaded state alone.
    resumed.clarification_answers = {q: "about 50k, GDPR applies" for q in questions_before}
    finished = orchestrator.run_pipeline(resumed)

    assert finished.stage is Stage.DONE
    assert finished.run_id == run_id, "a resumed run keeps its identity"
    assert finished.context_record is not None
    assert finished.blueprint is not None
    assert finished.features and finished.adrs and finished.components

    # The pre-pause trace survived the disk round-trip and the run continued
    # on top of it, rather than starting a fresh history.
    assert len(finished.history) > history_before
    agents = [step.agent for step in finished.history]
    assert agents[:history_before] == [
        step.agent for step in load_state(run_id).history[:history_before]
    ]
    assert {"researcher", "architect", "reviewer"} <= set(agents)

    # Everything landed in the SAME run directory, and the newest checkpoint is
    # the finished state — so resuming twice picks up from DONE, not the pause.
    after = sorted(p.name for p in run_dir.glob("*.json"))
    assert len(after) > len(before)
    assert after[: len(before)] == before, "the audit trail is append-only"
    assert after[-1].endswith("_done.json")
    assert load_state(run_id).stage is Stage.DONE
    assert len(list_runs()) == 1, "one directory, one run — resume never forks it"


# ── 3. corrupt checkpoints ───────────────────────────────────────────────
def test_corrupt_checkpoint_is_skipped_by_list_runs():
    root = _isolate("corrupt")

    healthy = _rich_state()
    save_state(healthy)

    # A run whose NEWEST checkpoint is truncated JSON — the resume point is
    # unusable, so the run must not be offered in the picker.
    broken = new_run("This run's newest checkpoint got truncated.")
    save_state(broken)
    (root / broken.run_id / "002_clarifying.json").write_text(
        '{"initial_request": {"raw_pro', encoding="utf-8"
    )

    # A run whose OLDER checkpoint is corrupt but whose newest one is fine:
    # still perfectly resumable, so it must survive the listing.
    survivor = _rich_state()
    save_state(survivor)
    (root / survivor.run_id / "001_awaiting_input.json").write_text("}not json{", encoding="utf-8")
    survivor.log_step("researcher", Stage.RESEARCHING, "still fine")
    save_state(survivor)

    listed = {summary.run_id for summary in list_runs()}
    assert healthy.run_id in listed
    assert survivor.run_id in listed
    assert broken.run_id not in listed, "unreadable resume point must be skipped"

    # list_runs stayed useful; load_state is the one that complains, loudly.
    try:
        load_state(broken.run_id)
    except CheckpointError as exc:
        assert "002_clarifying.json" in str(exc)
    else:
        raise AssertionError("load_state must raise on a corrupt checkpoint")

    # An unknown run is an error too, not a silent empty state.
    try:
        load_state("no-such-run")
    except CheckpointError as exc:
        assert "No checkpoints" in str(exc)
    else:
        raise AssertionError("load_state must raise for an unknown run")


# ── 4. atomic writes ─────────────────────────────────────────────────────
def test_atomic_write_leaves_no_tmp_behind():
    root = _isolate("atomic")
    state = _rich_state()

    for _ in range(3):
        save_state(state)
        state.log_step("researcher", Stage.RESEARCHING, "tick")
    run_dir = root / state.run_id
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "001_awaiting_input.json",
        "002_researching.json",
        "003_researching.json",
    ]
    assert not list(run_dir.glob("*.tmp"))

    # The real guarantee: a write that dies at the rename must clean up after
    # itself, leaving neither a stray .tmp nor a half-written checkpoint.
    surviving = sorted(p.name for p in run_dir.iterdir())
    with mock.patch("pipeline.persistence.os.replace", side_effect=OSError("disk full")):
        try:
            save_state(state)
        except OSError:
            pass
        else:
            raise AssertionError("save_state must propagate a failed write")
    assert not list(run_dir.glob("*.tmp")), "a failed write must not leave a .tmp"
    assert not list(run_dir.glob(".*")), "nor a hidden temp file"
    assert sorted(p.name for p in run_dir.iterdir()) == surviving

    # `checkpoint` is the pipeline-facing wrapper: same cleanup, but it swallows
    # the failure so a full disk can never take a run down.
    with mock.patch("pipeline.persistence.os.replace", side_effect=OSError("disk full")):
        assert checkpoint(state) is None
    assert not list(run_dir.glob("*.tmp"))
    assert sorted(p.name for p in run_dir.iterdir()) == surviving

    # A healthy write still works afterwards, at the next sequence number.
    assert save_state(state).name == "004_researching.json"


# ── 4b. the step-cap path (the one that escapes as an exception) ─────────
def test_step_cap_failure_is_checkpointed():
    """A step-cap blowout must still leave a resume point.

    Every other transition reaches the hook as a stream emission. This one does
    not: it surfaces as `GraphRecursionError`, so the FAILED state is written by
    the `finally` block in `run_pipeline_streaming`. What gets written there is
    a COPY of the caller's state carrying the FAILED marker — the caller's own
    object must come back untouched, exactly as on every other path.
    """
    _isolate("step_cap")
    _install_llm_mocks(clarifier=tc._complete)

    seed = new_run(PROMPT)
    result = orchestrator.run_pipeline(seed, max_steps=3)  # deliberately too low

    assert result.stage is Stage.FAILED
    assert result.errors == ["max_steps (3) reached before DONE"]

    # run_pipeline does not mutate its input — on THIS path too.
    assert result is not seed
    assert seed.errors == [], "the caller's state must survive a cap failure clean"
    assert seed.stage is Stage.CREATED
    assert seed.history == []

    files = sorted(p.name for p in (runs_dir() / seed.run_id).glob("*.json"))
    assert files[-1].endswith("_failed.json"), files
    assert len(files) > 1, "the transitions before the cap were checkpointed too"

    recovered = load_state(seed.run_id)
    assert recovered.stage is Stage.FAILED
    assert recovered.errors == ["max_steps (3) reached before DONE"]
    assert recovered.history[-1].agent == "orchestrator"

    # The happy path must NOT pay for that safety net with a duplicate final
    # checkpoint — the identity guard in the `finally` block keeps it a no-op.
    _isolate("step_cap_no_dupe")
    done = orchestrator.run_pipeline(new_run(PROMPT))
    assert done.stage is Stage.DONE
    names = sorted(p.name for p in (runs_dir() / done.run_id).glob("*.json"))
    assert names[-1].endswith("_done.json")
    assert len([n for n in names if n.endswith("_done.json")]) == 1, names


# ── 5. the equivalence run_pipeline is built on ──────────────────────────
_VOLATILE_KEYS = {"timestamp", "ingested_at"}


def _strip_volatile(value):
    """Blank out wall-clock fields so two runs can be compared field-for-field.

    `StepLog.timestamp` (and `RepoMeta.ingested_at`) default to "now", so two
    executions of the same deterministic pipeline differ there and nowhere
    else. Blanking them keeps the comparison exhaustive over everything that
    carries meaning, instead of weakening it to a handful of spot checks.
    """
    if isinstance(value, dict):
        return {
            k: ("" if k in _VOLATILE_KEYS else _strip_volatile(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_strip_volatile(v) for v in value]
    return value


def test_stream_final_emission_matches_invoke():
    """`run_pipeline` returns the stream's last value instead of calling invoke.

    That is only safe while the two agree, so pin it: same seed, same mocks,
    both paths, compared field-for-field. If a LangGraph upgrade ever changes
    what `stream(stream_mode="values")` emits last, this fails loudly here
    rather than silently changing what every caller of `run_pipeline` gets.
    """
    _isolate("equivalence")
    _install_llm_mocks(clarifier=tc._complete)  # no pause: run straight through

    seed = new_run(PROMPT)
    config = {"recursion_limit": 20}

    invoked = ArchitectState.model_validate(
        orchestrator.GRAPH.invoke(seed.model_copy(deep=True), config=config)
    )
    emissions = list(
        orchestrator.GRAPH.stream(
            seed.model_copy(deep=True), config=config, stream_mode="values"
        )
    )
    streamed = ArchitectState.model_validate(emissions[-1])

    assert streamed.stage is invoked.stage
    assert streamed.stage is Stage.DONE, "the seed must actually exercise the graph"

    left = _strip_volatile(invoked.model_dump(mode="json"))
    right = _strip_volatile(streamed.model_dump(mode="json"))

    # Guard against a vacuous comparison: `_strip_volatile` must have blanked
    # timestamps and NOTHING else, so the payload being compared still carries
    # the artifacts the pipeline produced.
    assert left["blueprint"] and left["adrs"] and left["components"]
    assert left["context_record"] and left["retrieved_knowledge"]
    assert len(left["history"]) >= 4
    assert all(step["timestamp"] == "" for step in left["history"])
    assert left["history"][0]["agent"], "only timestamps may be blanked"

    assert set(left) == set(right), "invoke and stream must expose the same fields"
    assert left == right, "the last emission must equal the invoke result"

    # And the property `run_pipeline` actually relies on: keeping the last
    # streamed value reproduces the old return value.
    kept = None
    for snapshot in orchestrator.run_pipeline_streaming(seed.model_copy(deep=True)):
        kept = snapshot
    assert _strip_volatile(kept.model_dump(mode="json")) == left


if __name__ == "__main__":
    test_round_trip_preserves_state_and_history()
    print("PASS  save/load round-trip preserves state and history")

    test_resume_from_awaiting_input_continues()
    print("PASS  resume from AWAITING_HUMAN continues in the same run")

    test_corrupt_checkpoint_is_skipped_by_list_runs()
    print("PASS  corrupt checkpoint skipped by list_runs, raised by load_state")

    test_atomic_write_leaves_no_tmp_behind()
    print("PASS  atomic write leaves no .tmp behind")

    test_step_cap_failure_is_checkpointed()
    print("PASS  step-cap failure still leaves a resume point")

    test_stream_final_emission_matches_invoke()
    print("PASS  stream's last emission matches invoke field-for-field")

    print("\nALL PERSISTENCE TESTS PASSED")


# ---------------------------------------------------------------------------
# 6. `ReviewResult.not_applicable` — forward and backward compatible.
#
# The field records criteria that were asked but carry no verdict weight. It was
# added after runs had already been recorded, so BOTH directions matter: a new
# report must survive the round trip, and a checkpoint written before the field
# existed must still load.
# ---------------------------------------------------------------------------


def test_not_applicable_survives_the_round_trip():
    _isolate("not_applicable_round_trip")
    state = new_run(PROMPT)
    state.review = ReviewResult(
        overall_status="pass",
        rubric_scores=RubricScores(best_practice_grounding=True),
        issues=[],
        requires_refinement=False,
        not_applicable=["best_practice_grounding"],
    )

    save_state(state)
    loaded = load_state(state.run_id)

    assert loaded.review.not_applicable == ["best_practice_grounding"]
    assert loaded.review == state.review


def test_checkpoint_written_before_the_field_existed_still_loads():
    """An older checkpoint has no `not_applicable` key at all."""

    _isolate("not_applicable_legacy")
    state = new_run(PROMPT)
    state.review = ReviewResult(overall_status="fail", requires_refinement=True)
    path = save_state(state)

    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["review"]["not_applicable"]
    assert "not_applicable" not in payload["review"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_state(state.run_id)

    # Defaults to empty, which is exactly "nothing was excluded" - the rule that
    # applied when the checkpoint was written.
    assert loaded.review.not_applicable == []
    assert loaded.review.overall_status == "fail"


def test_revision_note_survives_the_round_trip():
    _isolate("revision_note_round_trip")
    state = new_run(PROMPT)
    state.blueprint = Blueprint(
        stakeholder_view="Customers keep shopping during peak sales.",
        technical_view="Web tier offloads order writes to a queue.",
        revision_note="Added FEAT-004 to the Data Governance Service.",
    )

    save_state(state)
    loaded = load_state(state.run_id)

    assert loaded.blueprint.revision_note == (
        "Added FEAT-004 to the Data Governance Service."
    )
    assert loaded.blueprint == state.blueprint


def test_checkpoint_without_revision_note_still_loads():
    """Every run recorded before this field existed."""

    _isolate("revision_note_legacy")
    state = new_run(PROMPT)
    state.blueprint = Blueprint(
        stakeholder_view="A view.", technical_view="Another view."
    )
    path = save_state(state)

    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["blueprint"]["revision_note"]
    assert "revision_note" not in payload["blueprint"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_state(state.run_id)

    assert loaded.blueprint.revision_note == ""
    assert loaded.blueprint.stakeholder_view == "A view."


# ---------------------------------------------------------------------------
# 7. Stable run timestamps — the picker's date/time must come from the run_id,
# not the checkpoint file's filesystem mtime, so a fresh `git clone` (which
# writes every tracked file at once) does not collapse distinct historical
# runs onto one visible timestamp.
# ---------------------------------------------------------------------------


def test_run_timestamp_helper_parses_stable_stamp_from_run_id(tmp_path):
    """The run_id's own UTC stamp wins, regardless of the file's mtime."""
    checkpoint_path = tmp_path / "001_created.json"
    checkpoint_path.write_text("{}", encoding="utf-8")
    os.utime(checkpoint_path, (0, 0))  # an mtime that must be ignored here

    assert (
        _run_timestamp("20260828T123928Z-9445e65e", checkpoint_path)
        == "2026-08-28T12:39:28+00:00"
    )
    assert (
        _run_timestamp("20260827T180525Z-b60e271d", checkpoint_path)
        == "2026-08-27T18:05:25+00:00"
    )


def test_run_timestamp_helper_falls_back_for_malformed_run_id(tmp_path):
    """A run_id with no stamp, or a shape match that isn't a real date, must
    fall back to the checkpoint's mtime rather than raising or dropping the run."""
    checkpoint_path = tmp_path / "001_created.json"
    checkpoint_path.write_text("{}", encoding="utf-8")
    fixed_mtime = 1_700_000_000
    os.utime(checkpoint_path, (fixed_mtime, fixed_mtime))
    expected = datetime.fromtimestamp(fixed_mtime, tz=timezone.utc).isoformat(
        timespec="seconds"
    )

    assert _run_timestamp("not-a-timestamped-run-id", checkpoint_path) == expected
    # Matches the `\d{8}T\d{6}Z-` shape but month 13 makes it an invalid date.
    assert _run_timestamp("20261301T999999Z-deadbeef", checkpoint_path) == expected


def test_list_runs_timestamp_survives_checkpoint_mtime_changes():
    """Changing the checkpoint file's mtime must not move the displayed
    timestamp for a standard run_id — it is no longer the source of truth."""
    _isolate("mtime_independent")
    state = _rich_state()
    state.run_id = "20260827T180525Z-b60e271d"
    path = save_state(state)

    before = list_runs()[0].updated_at
    os.utime(path, (1_700_000_000, 1_700_000_000))
    after = list_runs()[0].updated_at

    assert before == after == "2026-08-27T18:05:25+00:00"


def test_list_runs_falls_back_to_mtime_for_legacy_run_id():
    """An old/nonstandard run directory name must still be listed — just with
    the previous (mtime-derived) timestamp — never dropped from the picker."""
    _isolate("legacy_run_id")
    state = _rich_state()
    state.run_id = "legacy-run-without-a-timestamp-prefix"
    path = save_state(state)

    fixed_mtime = 1_700_000_000
    os.utime(path, (fixed_mtime, fixed_mtime))
    expected = datetime.fromtimestamp(fixed_mtime, tz=timezone.utc).isoformat(
        timespec="seconds"
    )

    (summary,) = list_runs()
    assert summary.run_id == "legacy-run-without-a-timestamp-prefix"
    assert summary.updated_at == expected


def test_list_runs_orders_by_run_id_timestamp_when_mtimes_collide():
    """The exact bug scenario: a fresh clone writes every tracked checkpoint
    file with (near-)identical mtimes, but distinct historical runs must still
    sort newest-first by their own stable creation time."""
    _isolate("order_by_stable_timestamp")
    ids_oldest_to_newest = [
        "20260101T000000Z-aaaaaaaa",
        "20260615T120000Z-bbbbbbbb",
        "20260827T180525Z-cccccccc",
        "20260828T123928Z-dddddddd",
    ]
    clone_mtime = 1_800_000_000  # identical for every file, as a real clone would leave them
    for run_id in ids_oldest_to_newest:
        state = _rich_state()
        state.run_id = run_id
        path = save_state(state)
        os.utime(path, (clone_mtime, clone_mtime))

    ordered = [summary.run_id for summary in list_runs()]
    assert ordered == list(reversed(ids_oldest_to_newest))
