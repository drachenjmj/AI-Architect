"""Offline tests for evaluation-harness metrics; no LLM or API key required."""
from __future__ import annotations

from eval.harness import run_harness
from eval.scenarios import SCENARIOS
from pipeline.state import JudgmentReasons, ReviewResult, RubricScores, Stage


def _report(passed: bool) -> ReviewResult:
    return ReviewResult(
        overall_status="pass" if passed else "fail",
        rubric_scores=RubricScores(
            all_artifacts_present=2,
            constraint_coverage=2,
            traceability=2,
            adr_presence=2,
            repo_grounding=passed,
            flaw_detection=passed,
            adr_soundness=passed,
            best_practice_grounding=passed,
            refinement_readiness=passed,
        ),
        judgment_reasons=JudgmentReasons(
            repo_grounding="offline fixture",
            flaw_detection="offline fixture",
            adr_soundness="offline fixture",
            best_practice_grounding="offline fixture",
            refinement_readiness="offline fixture",
        ),
        requires_refinement=not passed,
    )


def test_harness_reports_perfect_agreement():
    verdicts = iter([False, True])

    def reviewer(_state):
        passed = next(verdicts)
        return {
            "review": _report(passed),
            "stage": Stage.DONE if passed else Stage.REFINING,
        }

    summary = run_harness(reviewer=reviewer)

    assert summary.agreement == 1.0
    assert summary.true_positive_rate == 1.0
    assert summary.true_negative_rate == 1.0
    assert summary.true_positives == 1
    assert summary.true_negatives == 1


def test_harness_reports_false_positive_and_false_negative():
    verdicts = iter([True, False])

    def reviewer(_state):
        passed = next(verdicts)
        return {
            "review": _report(passed),
            "stage": Stage.DONE if passed else Stage.REFINING,
        }

    summary = run_harness(SCENARIOS, reviewer=reviewer)

    assert summary.agreement == 0.0
    assert summary.false_positives == 1
    assert summary.false_negatives == 1
