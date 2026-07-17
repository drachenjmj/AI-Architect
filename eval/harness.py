"""Run labeled scenarios through the real Reviewer and report agreement."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from eval.scenarios import EvalScenario, SCENARIOS
from pipeline.state import ArchitectState, ReviewResult, Stage


ReviewerCallable = Callable[[ArchitectState], dict]


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    expected_pass: bool
    actual_pass: bool
    classification: str
    report: ReviewResult

    @property
    def agrees(self) -> bool:
        return self.expected_pass == self.actual_pass


@dataclass(frozen=True)
class HarnessSummary:
    results: tuple[ScenarioResult, ...]
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    @property
    def agreement(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.agrees for result in self.results) / len(self.results)

    @property
    def true_positive_rate(self) -> float:
        positives = self.true_positives + self.false_negatives
        return self.true_positives / positives if positives else 0.0

    @property
    def true_negative_rate(self) -> float:
        negatives = self.true_negatives + self.false_positives
        return self.true_negatives / negatives if negatives else 0.0


def _real_reviewer(state: ArchitectState) -> dict:
    # Keep metric tests importable until main contains all repo-ingestor modules.
    # A live harness run still resolves and calls the production Reviewer node.
    from pipeline.agents.reviewer import reviewer_node

    return reviewer_node(state)


def _classification(expected_pass: bool, actual_pass: bool) -> str:
    if expected_pass and actual_pass:
        return "TP"
    if not expected_pass and not actual_pass:
        return "TN"
    if not expected_pass and actual_pass:
        return "FP"
    return "FN"


def run_harness(
    scenarios: Iterable[EvalScenario] = SCENARIOS,
    *,
    reviewer: ReviewerCallable = _real_reviewer,
) -> HarnessSummary:
    """Evaluate scenarios and return a confusion-matrix summary."""

    results: list[ScenarioResult] = []
    for scenario in scenarios:
        output = reviewer(scenario.build_state())
        if output.get("stage") is Stage.FAILED or "review" not in output:
            raise RuntimeError(
                f"Reviewer failed for {scenario.name}: {output.get('errors', [])}"
            )
        report: ReviewResult = output["review"]
        actual_pass = report.overall_status == "pass"
        results.append(
            ScenarioResult(
                name=scenario.name,
                expected_pass=scenario.expected_pass,
                actual_pass=actual_pass,
                classification=_classification(
                    scenario.expected_pass,
                    actual_pass,
                ),
                report=report,
            )
        )

    result_tuple = tuple(results)
    counts = {
        label: sum(result.classification == label for result in result_tuple)
        for label in ("TP", "TN", "FP", "FN")
    }
    return HarnessSummary(
        results=result_tuple,
        true_positives=counts["TP"],
        true_negatives=counts["TN"],
        false_positives=counts["FP"],
        false_negatives=counts["FN"],
    )


def _print_summary(summary: HarnessSummary) -> None:
    print(
        f"Agreement: {summary.agreement:.1%} "
        f"({sum(result.agrees for result in summary.results)}/{len(summary.results)})"
    )
    print(f"True-positive rate: {summary.true_positive_rate:.1%}")
    print(f"True-negative rate: {summary.true_negative_rate:.1%}")
    print(
        "Confusion matrix: "
        f"TP={summary.true_positives} TN={summary.true_negatives} "
        f"FP={summary.false_positives} FN={summary.false_negatives}"
    )

    for result in summary.results:
        print(
            f"[{result.classification}] {result.name}: "
            f"expected={'pass' if result.expected_pass else 'fail'}, "
            f"actual={'pass' if result.actual_pass else 'fail'}"
        )
        if result.agrees:
            continue
        reasons = result.report.judgment_reasons
        for name, reason in reasons.model_dump().items():
            print(f"  {name}: {reason or '(no reason supplied)'}")


def main() -> int:
    summary = run_harness()
    _print_summary(summary)
    return 0 if summary.false_positives == summary.false_negatives == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
