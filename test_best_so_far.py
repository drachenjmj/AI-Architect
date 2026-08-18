"""test_best_so_far.py — the refine gate returns its BEST round, not its last (Kati).

No API key, no network. Two halves:

  1. `score_round` on its own — a pure function over a `ReviewResult`, so every
     ordering claim in its docstring is checked without building a graph.
  2. `refine_gate_node` on its own — also callable directly, because the gate is
     pure code. Driving the whole pipeline to reach it would only add stubs
     between the assertion and the thing asserted.

WHAT THIS FEATURE MUST NOT DO
------------------------------
It must not make a failing run pass. `test_selection_changes_nothing_but_the_artifacts`
is the guard: `overall_status`, `stopped_on_cap` and `refine_iterations` come out
of a capped run exactly as they did before selection existed. The feature
changes WHICH design is handed back and nothing else — if it ever starts
improving the verdict, it has stopped being selection and started being scoring
its own homework.
"""
from __future__ import annotations

import pytest

from pipeline.persistence import load_state, save_state
from pipeline.refine_gate import (
    MAX_REFINE_ITERATIONS,
    refine_gate_node,
    score_round,
)
from pipeline.state import (
    ADR,
    ArchitectState,
    Blueprint,
    ComponentDescription,
    DesignSnapshot,
    Feature,
    ReviewIssue,
    ReviewResult,
    RubricScores,
    Stage,
    new_run,
)

PROMPT = "Our shop crashes on peak sales days."


# ══════════════════════════════════════════════════════════════════════════
# Builders
# ══════════════════════════════════════════════════════════════════════════
def _review(
    *,
    status="fail",
    code=(2, 2, 2, 2),
    judged=(True, True, True, True, True),
    high=0,
    low=0,
) -> ReviewResult:
    """One reviewer verdict, dialled to whatever the ordering claim needs."""
    issues = [
        ReviewIssue(id=f"H-{n}", severity="high", finding=f"high issue {n}")
        for n in range(high)
    ] + [
        ReviewIssue(id=f"L-{n}", severity="low", finding=f"low issue {n}")
        for n in range(low)
    ]
    return ReviewResult(
        overall_status=status,
        rubric_scores=RubricScores(
            all_artifacts_present=code[0],
            constraint_coverage=code[1],
            traceability=code[2],
            adr_presence=code[3],
            repo_grounding=judged[0],
            flaw_detection=judged[1],
            adr_soundness=judged[2],
            best_practice_grounding=judged[3],
            refinement_readiness=judged[4],
        ),
        issues=issues,
        requires_refinement=status != "pass",
    )


def _artifacts(tag: str):
    """A distinguishable artifact set, so a swap is visible in every field."""
    return (
        [Feature(id=f"FEAT-{tag}", name=f"feature {tag}", scenario=f"scenario {tag}")],
        Blueprint(
            project_name=f"project {tag}",
            stakeholder_view=f"stakeholder view {tag}",
            technical_view=f"technical view {tag}",
        ),
        [ADR(id=f"ADR-{tag}", title=f"ADR-{tag}: decision", decision=f"decision {tag}")],
        [ComponentDescription(id=f"COMP-{tag}", name=f"component {tag}",
                              description=f"description {tag}")],
    )


def _at_gate(tag: str, review: ReviewResult, iterations: int, **overrides) -> ArchitectState:
    """A state as the reviewer leaves it: REFINING, artifacts and verdict present."""
    features, blueprint, adrs, components = _artifacts(tag)
    state = new_run(PROMPT)
    state.stage = Stage.REFINING
    state.features = features
    state.blueprint = blueprint
    state.adrs = adrs
    state.components = components
    state.review = review
    state.refine_iterations = iterations
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def _snapshot(tag: str, round_number: int, review: ReviewResult) -> DesignSnapshot:
    features, blueprint, adrs, components = _artifacts(tag)
    return DesignSnapshot(
        round=round_number,
        features=features,
        blueprint=blueprint,
        adrs=adrs,
        components=components,
        review=review,
    )


# ══════════════════════════════════════════════════════════════════════════
# 1. score_round — every claim in its docstring, checked
# ══════════════════════════════════════════════════════════════════════════
def test_a_pass_beats_any_fail():
    """Criterion 1 dominates: a bare pass outranks a fail that is better on every
    other axis. If it did not, the ordering could prefer a design the reviewer
    rejected over one it accepted."""
    passing = _review(status="pass", code=(0, 0, 0, 0), judged=(False,) * 5, low=9)
    failing = _review(status="fail", code=(2, 2, 2, 2), judged=(True,) * 5)

    assert score_round(passing) > score_round(failing)


def test_fewer_high_severity_issues_wins():
    clean = _review(high=0)
    one_high = _review(high=1)
    two_high = _review(high=2)

    assert score_round(clean) > score_round(one_high) > score_round(two_high)


def test_severity_outranks_the_code_scores():
    """The one genuinely arguable line in the ordering, pinned so it is a decision
    rather than an accident.

    Tidy-but-flawed (a full 8/8 code score, one high-severity finding) must lose
    to substantive-but-untidy (5/8, nothing high). Ranking the scoreboard above
    the substance would let the loop optimise the scoreboard.
    """
    tidy_but_flawed = _review(code=(2, 2, 2, 2), high=1)
    flawed_but_sound = _review(code=(2, 1, 1, 1), high=0)

    assert score_round(flawed_but_sound) > score_round(tidy_but_flawed)


def test_higher_code_score_sum_wins():
    better = _review(code=(2, 2, 2, 2))
    worse = _review(code=(2, 2, 1, 2))

    assert score_round(better) > score_round(worse)
    # The sum is what is compared, not any single field: a point lost on one
    # check and gained on another is a draw.
    assert score_round(_review(code=(2, 1, 2, 1))) == score_round(_review(code=(1, 2, 1, 2)))


def test_more_passing_judgments_wins_then_fewer_issues():
    more_judgments = _review(judged=(True, True, True, True, True))
    fewer_judgments = _review(judged=(True, True, True, True, False))
    assert score_round(more_judgments) > score_round(fewer_judgments)

    # Same everything except issue count — the last tiebreaker.
    assert score_round(_review(low=1)) > score_round(_review(low=4))


def test_score_round_is_pure():
    """No side effects: scoring the same review twice cannot change its answer."""
    review = _review(code=(2, 1, 2, 2), high=1, low=2)
    before = review.model_dump_json()

    assert score_round(review) == score_round(review)
    assert review.model_dump_json() == before


# ══════════════════════════════════════════════════════════════════════════
# 2. The gate keeps an incumbent, and ties go to the earlier round
# ══════════════════════════════════════════════════════════════════════════
def test_first_scored_round_becomes_the_incumbent():
    out = refine_gate_node(_at_gate("r1", _review(code=(2, 2, 1, 2)), iterations=0))

    best = out["best_design"]
    assert best.round == 1
    assert [f.id for f in best.features] == ["FEAT-r1"]
    assert best.review.rubric_scores.traceability == 1
    assert "round 1 is the new best" in out["history"][0].note


def test_a_strictly_better_round_takes_the_lead():
    incumbent = _snapshot("r1", 1, _review(code=(2, 2, 1, 2)))
    state = _at_gate("r2", _review(code=(2, 2, 2, 2)), iterations=1, best_design=incumbent)

    out = refine_gate_node(state)

    best = out["best_design"]
    assert best.round == 2
    assert [f.id for f in best.features] == ["FEAT-r2"]
    assert "round 2 is the new best" in out["history"][0].note


def test_a_worse_round_does_not_displace_the_incumbent():
    incumbent = _snapshot("r1", 1, _review(code=(2, 2, 2, 2)))
    state = _at_gate("r2", _review(code=(2, 2, 1, 2)), iterations=1, best_design=incumbent)

    out = refine_gate_node(state)

    assert out["best_design"] is incumbent
    assert "best remains round 1" in out["history"][0].note


def test_a_tie_resolves_to_the_earlier_round():
    """Equal is not better. Preferring the incumbent makes selection stable —
    re-running the comparison can never change its own answer."""
    identical = _review(code=(2, 2, 1, 2), judged=(True, True, False, True, False), low=2)
    incumbent = _snapshot("r1", 1, identical)
    state = _at_gate("r2", identical.model_copy(deep=True), iterations=1,
                     best_design=incumbent)

    assert score_round(state.review) == score_round(incumbent.review)  # a real tie

    out = refine_gate_node(state)

    assert out["best_design"].round == 1
    assert [f.id for f in out["best_design"].features] == ["FEAT-r1"]


# ══════════════════════════════════════════════════════════════════════════
# 3. The stop branch — the design and its review travel together
# ══════════════════════════════════════════════════════════════════════════
def test_stop_restores_a_better_earlier_round_with_its_review():
    """THE feature. Round 2 scored better than round 3, so round 2 ships.

    The review is asserted alongside the artifacts because restoring one without
    the other would produce a report describing a design the run does not
    contain — worse than shipping the later design untouched.
    """
    round2_review = _review(code=(2, 2, 2, 2), judged=(True,) * 5)
    incumbent = _snapshot("r2", 2, round2_review)
    state = _at_gate(
        "r3",
        _review(code=(2, 2, 1, 2), judged=(True, False, True, True, False)),
        iterations=MAX_REFINE_ITERATIONS,
        best_design=incumbent,
    )

    out = refine_gate_node(state)

    assert out["stage"] is Stage.DONE
    assert out["selected_round"] == 2
    # Every artifact field comes from round 2 ...
    assert [f.id for f in out["features"]] == ["FEAT-r2"]
    assert out["blueprint"].project_name == "project r2"
    assert [a.id for a in out["adrs"]] == ["ADR-r2"]
    assert [c.id for c in out["components"]] == ["COMP-r2"]
    # ... and so does the verdict that describes them.
    assert out["review"] is round2_review
    assert out["review"].rubric_scores.traceability == 2

    note = out["history"][0].note
    assert "selected round 2 of 3" in note
    assert "discarded 1 later round" in note


def test_stop_restores_nothing_when_the_last_round_is_best():
    incumbent = _snapshot("r2", 2, _review(code=(2, 2, 1, 2)))
    state = _at_gate(
        "r3",
        _review(code=(2, 2, 2, 2)),
        iterations=MAX_REFINE_ITERATIONS,
        best_design=incumbent,
    )

    out = refine_gate_node(state)

    # No artifact keys at all: absent, not rewritten with the same values.
    for key in ("features", "blueprint", "adrs", "components", "review"):
        assert key not in out, f"{key} was restored when nothing needed restoring"
    assert out["selected_round"] == 3
    assert "scored best, nothing restored" in out["history"][0].note


def test_the_note_names_both_rounds_when_more_than_one_is_discarded():
    """The audit trail has to be readable without re-deriving it from scores."""
    incumbent = _snapshot("r1", 1, _review(code=(2, 2, 2, 2)))
    state = _at_gate("r3", _review(code=(1, 1, 1, 1)), iterations=MAX_REFINE_ITERATIONS,
                     best_design=incumbent)

    note = refine_gate_node(state)["history"][0].note

    assert "selected round 1 of 3" in note
    assert "discarded 2 later rounds" in note


def test_selection_changes_nothing_but_the_artifacts():
    """The guard: this feature must never turn a failing run into a passing one."""
    incumbent = _snapshot("r2", 2, _review(status="fail", code=(2, 2, 2, 2)))
    state = _at_gate("r3", _review(status="fail", code=(1, 1, 1, 1)),
                     iterations=MAX_REFINE_ITERATIONS, best_design=incumbent)

    out = refine_gate_node(state)

    assert out["stopped_on_cap"] is True
    assert out["review"].overall_status == "fail"     # still a fail
    assert out["review"].requires_refinement is True
    assert "refine_iterations" not in out             # the counter is not bumped on stop
    assert out["stage"] is Stage.DONE


def test_gate_survives_a_missing_review():
    """Defensive: end gracefully rather than turning a graceful stop into FAILED."""
    state = _at_gate("r1", _review(), iterations=MAX_REFINE_ITERATIONS)
    state.review = None

    out = refine_gate_node(state)

    assert out["stage"] is Stage.DONE
    assert out["stopped_on_cap"] is True
    assert out["best_design"] is None


def test_the_gate_does_not_mutate_the_state_it_is_given():
    """Nodes return updates; they never write into the state they are handed."""
    incumbent = _snapshot("r2", 2, _review(code=(2, 2, 2, 2)))
    state = _at_gate("r3", _review(code=(1, 1, 1, 1)),
                     iterations=MAX_REFINE_ITERATIONS, best_design=incumbent)
    before = state.model_dump_json()

    refine_gate_node(state)

    assert state.model_dump_json() == before


# ══════════════════════════════════════════════════════════════════════════
# 4. Persistence
# ══════════════════════════════════════════════════════════════════════════
def test_design_snapshot_round_trips_through_a_checkpoint():
    state = _at_gate("r3", _review(code=(1, 1, 1, 1)), iterations=2,
                     best_design=_snapshot("r2", 2, _review(code=(2, 2, 2, 2))))
    state.selected_round = 2

    save_state(state)
    loaded = load_state(state.run_id)

    assert loaded == state
    assert loaded.best_design.round == 2
    assert [f.id for f in loaded.best_design.features] == ["FEAT-r2"]
    assert loaded.best_design.blueprint.project_name == "project r2"
    assert loaded.best_design.review.rubric_scores.traceability == 2
    assert loaded.selected_round == 2
    # And the score survives the round trip, which is what makes a resumed run
    # compare against the same incumbent the original run had.
    assert score_round(loaded.best_design.review) == score_round(state.best_design.review)


def test_a_checkpoint_written_before_this_change_still_loads(tmp_path, monkeypatch):
    """`best_design` and `selected_round` did not exist; their defaults are the
    pre-change behaviour — no incumbent, and "the question never arose"."""
    monkeypatch.setenv("AI_ARCHITECT_RUNS_DIR", str(tmp_path))
    run_id = "20260817T000000Z-0ldc0de5"
    legacy = (
        '{"initial_request": {"raw_prompt": "old run", "repo_url": ""}, '
        f'"run_id": "{run_id}", "stage": "done", "stopped_on_cap": true, '
        '"refine_iterations": 2}'
    )
    directory = tmp_path / run_id
    directory.mkdir()
    (directory / "024_done.json").write_text(legacy, encoding="utf-8")

    loaded = load_state(run_id)

    assert loaded.stage is Stage.DONE
    assert loaded.stopped_on_cap is True
    assert loaded.refine_iterations == 2
    assert loaded.best_design is None
    assert loaded.selected_round == 0


# ══════════════════════════════════════════════════════════════════════════
# 5. Through the REAL graph — the restored artifacts must actually land
# ══════════════════════════════════════════════════════════════════════════
def test_a_flapping_judge_ships_the_better_middle_round(monkeypatch):
    """The measured scenario, reproduced deterministically.

    The reviewer's LLM judgments flip pass -> fail across rounds (which is what
    `flaw_detection` really did in run 20260818T083516Z-e92aa7cf), while the
    architect returns identical artifacts every pass. Round 2 is therefore the
    only round that differs, and it differs upward.

    Driving the real graph, not just the node, because the node returning
    `features` and `review` in its update dict is only half the claim — the
    other half is that LangGraph's channels actually carry them onto the final
    state. A unit test on the node cannot see that.
    """
    import architect as legacy_architect
    from pipeline import orchestrator
    from pipeline.agents import architect as arch
    from pipeline.agents import clarifier as clar
    from pipeline.agents import reviewer as rev
    from test_clarifier import _architect_response, _complete, fake_usage

    legacy_architect.retrieve_chunks = lambda query, k=3: ([], "offline-test")
    monkeypatch.setattr(clar, "llm_call", _complete)
    monkeypatch.setattr(arch, "llm_call", _architect_response)

    def _judge(passed: bool) -> "rev.CriterionJudgment":
        return rev.CriterionJudgment(
            passed=passed,
            reason="canned",
            suggested_fix="" if passed else "canned fix",
        )

    # Round 1: 1 of 5 pass. Round 2: 2 of 5. Round 3: back to 1 of 5.
    #
    # CHANGED DELIBERATELY from [1, 4, 1]. The judgments are applied in the
    # `names` order below, so 4-of-5 used to mean "everything but
    # refinement_readiness", and that single NO was what held the run at the cap.
    # It no longer can: refinement_readiness is advisory and
    # best_practice_grounding is not applicable here, because
    # `retrieve_chunks` is stubbed to return nothing (see reviewer.py). Under
    # 4-of-5 this run would now simply PASS in round 2, and the scenario this
    # test exists for - a flapping judge, a better middle round, a capped run -
    # would quietly stop being exercised.
    #
    # So the flapping now happens on adr_soundness, which still counts. Round 2
    # is still strictly the best round (one high-severity issue rather than
    # two) and the run still ends on the cap.
    scripted = iter([1, 2, 1])

    def _flapping_reviewer(state, prompt, **kwargs):
        passing = next(scripted)
        names = ["repo_grounding", "flaw_detection", "adr_soundness",
                 "best_practice_grounding", "refinement_readiness"]
        return rev.LLMJudgments(
            **{name: _judge(i < passing) for i, name in enumerate(names)}
        ), fake_usage()

    monkeypatch.setattr(rev, "llm_call", _flapping_reviewer)

    done = orchestrator.run_pipeline(new_run(PROMPT))

    # The run still ends exactly as it did before this feature existed.
    assert done.stage is Stage.DONE
    assert done.stopped_on_cap is True
    assert done.refine_iterations == MAX_REFINE_ITERATIONS
    assert done.review.overall_status == "fail"

    # ... but it shipped round 2, which is the one that scored best.
    assert done.selected_round == 2
    assert done.best_design.round == 2
    scores = done.review.rubric_scores
    passing = sum(
        1 for name in ("repo_grounding", "flaw_detection", "adr_soundness",
                       "best_practice_grounding", "refinement_readiness")
        if getattr(scores, name)
    )
    # 3, not 4: repo_grounding and flaw_detection genuinely passed in round 2,
    # and best_practice_grounding reads true because it was NOT APPLICABLE -
    # see review.not_applicable, which is what stops that true reading as a pass.
    assert passing == 3, "the shipped review is not round 2's"
    assert done.review.not_applicable == ["best_practice_grounding"]

    # THE consistency claim: the shipped review is the review OF the shipped
    # artifacts. Both surfaces (ui_sections, run.py) read these two off the same
    # state object independently, so this is what keeps their pairing honest.
    assert done.review is not None
    assert done.blueprint is not None
    assert done.review == done.best_design.review
    assert done.features == done.best_design.features
    assert done.blueprint == done.best_design.blueprint
    assert done.adrs == done.best_design.adrs
    assert done.components == done.best_design.components

    gate_notes = [s.note for s in done.history if s.agent == "refine_gate"]
    assert "selected round 2 of 3" in gate_notes[-1]
    assert "discarded 1 later round" in gate_notes[-1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
