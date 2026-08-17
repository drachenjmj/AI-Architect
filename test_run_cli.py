"""test_run_cli.py — offline tests for the `python -m pipeline.run` answer loop (Kati).

No API key, no network, no keyboard. The pipeline itself is replaced by a
scripted generator, because what is under test here is the CALLER — the loop
that answers a pause and re-invokes — not the graph it drives.

WHAT IS MONKEYPATCHED, AND WHY THAT ONE
---------------------------------------
`pipeline.run.run_pipeline_streaming`. The loop consumes the STREAMING entry
point (so each step's note can be printed live, exactly as `ui.py:_run` does),
and the pause/resume contract is identical either way: the LAST yielded state is
what `run_pipeline` would have returned, and it is what the loop inspects for
`AWAITING_INPUT`. Patching the name inside `pipeline.run` leaves the real
orchestrator untouched for every other test in the suite.

Each fake yields ONE state per invocation — enough for the loop, and it keeps
each scripted round readable.
"""
from __future__ import annotations

import json

import pytest

from pipeline import run as cli
from pipeline.state import ArchitectState, Stage, new_run


PROMPT = "Build me a system to sell sneakers online."

Q_SCALE = "What is the expected peak user count?"
Q_CLOUD = "Which cloud provider must this run on?"


# ══════════════════════════════════════════════════════════════════════════
# Scripted pipeline
# ══════════════════════════════════════════════════════════════════════════
def _paused(state: ArchitectState, questions: list[str]) -> ArchitectState:
    """A copy of `state` parked at AWAITING_INPUT, like a real pause.

    A COPY, because the real `run_pipeline*` never mutates the state it is
    handed — a fake that mutated would let a loop bug (reading the input state
    instead of the returned one) pass.
    """
    out = state.model_copy(deep=True)
    out.clarifying_questions = list(questions)
    out.log_step("clarifier", Stage.AWAITING_INPUT, f"asked {len(questions)}")
    return out


def _done(state: ArchitectState) -> ArchitectState:
    """A copy of `state` at DONE."""
    out = state.model_copy(deep=True)
    out.clarifying_questions = []
    out.log_step("clarifier", Stage.DONE, "context locked")
    return out


class FakePipeline:
    """A `run_pipeline_streaming` stand-in that plays a scripted list of rounds.

    Records the state it was handed on each call, so a test can assert what the
    loop actually passed back in — which is where the merge-not-replace rule
    either holds or breaks.
    """

    def __init__(self, script):
        self._script = list(script)
        self.calls: list[ArchitectState] = []

    def __call__(self, state, max_steps=20):
        self.calls.append(state.model_copy(deep=True))
        if not self._script:
            raise AssertionError(
                "the loop invoked the pipeline more times than the script "
                "allows — it is not terminating"
            )
        yield self._script.pop(0)(state)


def _install(monkeypatch, script) -> FakePipeline:
    fake = FakePipeline(script)
    monkeypatch.setattr(cli, "run_pipeline_streaming", fake)
    return fake


def _answers_file(tmp_path, payload) -> str:
    path = tmp_path / "answers.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _flat(text: str) -> str:
    """Collapse the output's line wrapping.

    The report wraps prose to a page width, so a phrase can land across two
    lines. Assertions are about the SENTENCE, not the layout; flattening keeps
    them from breaking every time a wrap point moves.
    """
    return " ".join(text.split())


# ══════════════════════════════════════════════════════════════════════════
# 1. The loop resumes and terminates
# ══════════════════════════════════════════════════════════════════════════
def test_loop_answers_a_pause_and_finishes(monkeypatch, tmp_path, capsys):
    """AWAITING_INPUT on the first call, DONE on the second: --answers drives it."""
    fake = _install(
        monkeypatch,
        [lambda s: _paused(s, [Q_SCALE]), _done],
    )

    code = cli.main(
        [
            PROMPT,
            "--answers",
            _answers_file(tmp_path, {Q_SCALE: "50k concurrent users"}),
            "--no-input",
        ]
    )

    assert code == cli.EXIT_OK
    # Two invocations = the run was RESUMED, which one run_pipeline call never does.
    assert len(fake.calls) == 2
    # The second invocation carried the answer, so the clarifier could re-judge.
    assert fake.calls[1].clarification_answers == {Q_SCALE: "50k concurrent users"}
    assert fake.calls[1].stage is Stage.AWAITING_INPUT  # entry router resumes on this

    out = _flat(capsys.readouterr().out)
    assert "Final stage: done" in out
    assert Q_SCALE in out  # the question was shown, not silently consumed


def test_positional_answers_are_consumed_in_order(monkeypatch, tmp_path):
    """The array shape feeds answers by position, in the order asked."""
    fake = _install(
        monkeypatch,
        [lambda s: _paused(s, [Q_SCALE, Q_CLOUD]), _done],
    )

    code = cli.main(
        [
            PROMPT,
            "--answers",
            _answers_file(tmp_path, ["50k concurrent users", "AWS"]),
            "--no-input",
        ]
    )

    assert code == cli.EXIT_OK
    assert fake.calls[1].clarification_answers == {
        Q_SCALE: "50k concurrent users",
        Q_CLOUD: "AWS",
    }


# ══════════════════════════════════════════════════════════════════════════
# 2. Answers MERGE across rounds, never replace
# ══════════════════════════════════════════════════════════════════════════
def test_answers_merge_across_two_rounds(monkeypatch, tmp_path):
    """Round two must not wipe round one — the clarifier re-judges on BOTH.

    This is the rule `ui.py` keeps with `clarification_answers.update(...)`; a
    loop that assigned instead of merging would leave the final call carrying
    only the second answer, and the clarifier would re-ask what it already knew.
    """
    fake = _install(
        monkeypatch,
        [
            lambda s: _paused(s, [Q_SCALE]),  # round 1
            lambda s: _paused(s, [Q_CLOUD]),  # round 2 — a second pause
            _done,
        ],
    )

    code = cli.main(
        [
            PROMPT,
            "--answers",
            _answers_file(tmp_path, {Q_SCALE: "50k concurrent users", Q_CLOUD: "AWS"}),
            "--no-input",
        ]
    )

    assert code == cli.EXIT_OK
    assert len(fake.calls) == 3
    # After round 1: only the first answer exists.
    assert fake.calls[1].clarification_answers == {Q_SCALE: "50k concurrent users"}
    # After round 2: BOTH, because round 2 merged onto round 1.
    assert fake.calls[2].clarification_answers == {
        Q_SCALE: "50k concurrent users",
        Q_CLOUD: "AWS",
    }


# ══════════════════════════════════════════════════════════════════════════
# 3. The loop is bounded — it exits instead of spinning
# ══════════════════════════════════════════════════════════════════════════
def test_pause_with_no_questions_exits_instead_of_spinning(monkeypatch, capsys):
    """The known clarifier bug: AWAITING_INPUT with an EMPTY question list.

    There is nothing to ask, so a naive loop would re-invoke forever. It must
    stop, say what went wrong, and exit non-zero. (The clarifier itself is NOT
    fixed here — that is a separate task.)
    """
    fake = _install(monkeypatch, [lambda s: _paused(s, [])])

    code = cli.main([PROMPT, "--no-input"])

    assert code == cli.EXIT_CANNOT_CONTINUE
    assert len(fake.calls) == 1  # it did not re-invoke on an unanswerable pause
    err = _flat(capsys.readouterr().err)
    assert "asked NO questions" in err
    assert "known clarifier bug" in err


def test_round_cap_stops_an_endlessly_asking_clarifier(monkeypatch, tmp_path, capsys):
    """A clarifier that never converges is cut off at MAX_CLARIFICATION_ROUNDS."""
    # More scripted pauses than the cap allows: only the cap can end this.
    fake = _install(
        monkeypatch,
        [lambda s, i=i: _paused(s, [f"Question {i}?"]) for i in range(10)],
    )

    code = cli.main(
        [
            PROMPT,
            "--answers",
            _answers_file(tmp_path, ["a"] * 10),
            "--no-input",
        ]
    )

    assert code == cli.EXIT_CANNOT_CONTINUE
    # One initial run + one per answered round, then it stops.
    assert len(fake.calls) == cli.MAX_CLARIFICATION_ROUNDS + 1
    assert "not converging" in _flat(capsys.readouterr().err)


def test_non_terminal_non_pause_stage_does_not_loop(monkeypatch, capsys):
    """A stop at an unwired stage would re-stop instantly; report, do not loop."""

    def _stuck(state):
        out = state.model_copy(deep=True)
        out.log_step("reviewer", Stage.REVIEWING, "no route from here")
        return out

    fake = _install(monkeypatch, [_stuck])

    code = cli.main([PROMPT, "--no-input"])

    assert code == cli.EXIT_CANNOT_CONTINUE
    assert len(fake.calls) == 1
    assert "neither a pause nor a terminal stage" in _flat(capsys.readouterr().err)


# ══════════════════════════════════════════════════════════════════════════
# 4. Unanswered questions are never silently skipped
# ══════════════════════════════════════════════════════════════════════════
def test_no_input_fails_loudly_on_an_unanswered_question(monkeypatch, tmp_path, capsys):
    """--no-input + a question the file does not cover = print it and exit non-zero."""
    _install(monkeypatch, [lambda s: _paused(s, [Q_SCALE, Q_CLOUD])])

    code = cli.main(
        [
            PROMPT,
            "--answers",
            _answers_file(tmp_path, {Q_SCALE: "50k concurrent users"}),
            "--no-input",
        ]
    )

    assert code == cli.EXIT_CANNOT_CONTINUE
    err = _flat(capsys.readouterr().err)
    assert Q_CLOUD in err  # the UNANSWERED question is named
    assert "--answers" in err


def test_interactive_fallback_asks_for_a_missing_answer(monkeypatch, tmp_path):
    """Without --no-input, a question the file does not cover goes to stdin."""
    fake = _install(
        monkeypatch,
        [lambda s: _paused(s, [Q_SCALE, Q_CLOUD]), _done],
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": "AWS")

    code = cli.main(
        [
            PROMPT,
            "--answers",
            _answers_file(tmp_path, {Q_SCALE: "50k concurrent users"}),
        ]
    )

    assert code == cli.EXIT_OK
    assert fake.calls[1].clarification_answers == {
        Q_SCALE: "50k concurrent users",
        Q_CLOUD: "AWS",
    }


def test_closed_stdin_fails_instead_of_raising(monkeypatch, capsys):
    """EOF on stdin (piped/unattended, no --no-input) fails the same clean way."""

    def _eof(prompt=""):
        raise EOFError

    _install(monkeypatch, [lambda s: _paused(s, [Q_SCALE])])
    monkeypatch.setattr("builtins.input", _eof)

    assert cli.main([PROMPT]) == cli.EXIT_CANNOT_CONTINUE
    assert Q_SCALE in _flat(capsys.readouterr().err)


# ══════════════════════════════════════════════════════════════════════════
# 5. Terminal stages and the answers file itself
# ══════════════════════════════════════════════════════════════════════════
def test_failed_run_exits_one_and_still_prints_the_report(monkeypatch, capsys):
    def _failed(state):
        out = state.model_copy(deep=True)
        out.errors.append("clarifier: boom")
        out.log_step("clarifier", Stage.FAILED, "error: boom")
        return out

    _install(monkeypatch, [_failed])

    assert cli.main([PROMPT, "--no-input"]) == cli.EXIT_RUN_FAILED
    out = _flat(capsys.readouterr().out)
    assert "Final stage: failed" in out
    assert "clarifier: boom" in out


def test_bad_answers_file_is_rejected(monkeypatch, tmp_path, capsys):
    _install(monkeypatch, [])  # never reached

    bad = tmp_path / "answers.json"
    bad.write_text('"just a string"', encoding="utf-8")
    assert cli.main([PROMPT, "--answers", str(bad)]) == cli.EXIT_CANNOT_CONTINUE
    assert "must hold a JSON object" in _flat(capsys.readouterr().err)

    missing = tmp_path / "nope.json"
    assert cli.main([PROMPT, "--answers", str(missing)]) == cli.EXIT_CANNOT_CONTINUE
    assert "cannot read" in _flat(capsys.readouterr().err)


def test_repo_url_reaches_the_state(monkeypatch):
    fake = _install(monkeypatch, [_done])

    assert cli.main([PROMPT, "--repo-url", "https://github.com/org/repo"]) == cli.EXIT_OK
    assert fake.calls[0].initial_request.repo_url == "https://github.com/org/repo"
    assert fake.calls[0].initial_request.raw_prompt == PROMPT


def test_default_prompt_is_the_example_constant(monkeypatch):
    fake = _install(monkeypatch, [_done])

    assert cli.main(["--no-input"]) == cli.EXIT_OK
    assert fake.calls[0].initial_request.raw_prompt == cli.EXAMPLE_PROMPT


def test_max_steps_is_passed_through(monkeypatch):
    seen: list[int] = []

    def _fake(state, max_steps=20):
        seen.append(max_steps)
        yield _done(state)

    monkeypatch.setattr(cli, "run_pipeline_streaming", _fake)

    assert cli.main([PROMPT, "--max-steps", "7", "--no-input"]) == cli.EXIT_OK
    assert seen == [7]


# ══════════════════════════════════════════════════════════════════════════
# 6. The report prints every artifact, including the empty ones
# ══════════════════════════════════════════════════════════════════════════
def test_report_covers_every_section_and_says_when_empty(capsys):
    """A thin state must render honestly rather than leave headings behind.

    The empty knowledge base in particular has to be VISIBLE — it is a real
    finding about our KB and the one thing this output exists to stop hiding.
    """
    state = new_run(PROMPT)
    state.log_step("clarifier", Stage.DONE, "context locked")

    cli.print_report(state)

    out = _flat(capsys.readouterr().out)
    for heading in (
        "RUN TRACE",
        "CLARIFICATION Q&A",
        "CONTEXT RECORD",
        "FEATURES",
        "BLUEPRINT",
        "ARCHITECTURE DECISION RECORDS",
        "COMPONENTS",
        "REVIEW REPORT",
        "RETRIEVED KNOWLEDGE",
        "TOKENS AND COST",
        "WHERE TO FIND THIS RUN",
    ):
        assert heading in out
    assert "NO KNOWLEDGE-BASE RESULTS FOR THIS RUN" in out
    assert "list-price-equivalent USD" in out
    assert "NOT money spent" in out
    assert state.run_id in out


def test_report_prints_the_populated_artifacts(capsys):
    """Every field the task asked for actually reaches the page."""
    demo = pytest.importorskip("ui_demo")
    state = demo.build_demo_state("capped")

    cli.print_report(state)

    raw = capsys.readouterr().out
    out = _flat(raw)
    # Artifacts, beyond the summary line the old command never got to.
    assert state.context_record.business_goal in out
    assert state.features[0].scenario in out
    assert state.blueprint.selected_pattern in out
    assert state.blueprint.stakeholder_view[:40] in out
    assert state.blueprint.technical_view[:40] in out
    assert state.adrs[0].decision in out
    assert state.adrs[0].alternatives_considered[0] in out
    assert state.components[0].security_considerations[0] in out
    # Review: both halves of the rubric, plus the loop bookkeeping.
    assert "Code-owned rubric checks" in out
    assert "LLM-owned judgments" in out
    # Against the RAW output: these two are aligned columns, and _flat would
    # squash the padding that makes them a readable block.
    assert f"refine_iterations : {state.refine_iterations}" in raw
    assert f"stopped_on_cap    : {state.stopped_on_cap}" in raw
    # Cost: totals and the per-agent breakdown (both are METHODS on the state).
    assert f"{state.input_tokens:,} in" in out
    for agent in state.usage_by_agent():
        assert agent in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
