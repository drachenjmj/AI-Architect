"""Run labeled cases through the production Reviewer and report agreement."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

from eval.scenarios import (
    CODE_SCORE_FIELDS,
    JUDGMENT_FIELDS,
    EvalScenario,
    SCENARIOS,
    load_scenario_file,
)
from pipeline.state import ArchitectState, ReviewResult, Stage


ReviewerCallable = Callable[[ArchitectState], dict]


@dataclass(frozen=True)
class FieldComparison:
    kind: str
    name: str
    expected: int | bool
    actual: int | bool

    @property
    def agrees(self) -> bool:
        return self.expected == self.actual


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    domain: str
    label_status: str
    label_rationale: str
    repeat: int
    expected_pass: bool
    actual_pass: bool
    classification: str
    duration_seconds: float
    input_tokens: int
    output_tokens: int
    comparisons: tuple[FieldComparison, ...]
    report: ReviewResult

    @property
    def verdict_agrees(self) -> bool:
        return self.expected_pass == self.actual_pass

    @property
    def rubric_agrees(self) -> bool:
        return all(comparison.agrees for comparison in self.comparisons)

    @property
    def agrees(self) -> bool:
        return self.verdict_agrees and self.rubric_agrees

    @property
    def disagreements(self) -> tuple[FieldComparison, ...]:
        return tuple(
            comparison for comparison in self.comparisons if not comparison.agrees
        )


@dataclass(frozen=True)
class HarnessSummary:
    results: tuple[ScenarioResult, ...]
    model: str

    @property
    def verdict_agreement(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.verdict_agrees for result in self.results) / len(self.results)

    @property
    def rubric_agreement(self) -> float:
        comparisons = [
            comparison
            for result in self.results
            for comparison in result.comparisons
        ]
        if not comparisons:
            return 0.0
        return sum(comparison.agrees for comparison in comparisons) / len(comparisons)

    @property
    def complete_case_agreement(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.agrees for result in self.results) / len(self.results)

    @property
    def false_passes(self) -> int:
        return sum(result.classification == "false_pass" for result in self.results)

    @property
    def false_failures(self) -> int:
        return sum(result.classification == "false_fail" for result in self.results)

    @property
    def average_duration_seconds(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.duration_seconds for result in self.results) / len(self.results)

    @property
    def total_input_tokens(self) -> int:
        return sum(result.input_tokens for result in self.results)

    @property
    def total_output_tokens(self) -> int:
        return sum(result.output_tokens for result in self.results)

    @property
    def per_field_agreement(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for name in (*CODE_SCORE_FIELDS, *JUDGMENT_FIELDS):
            comparisons = [
                comparison
                for result in self.results
                for comparison in result.comparisons
                if comparison.name == name
            ]
            values[name] = (
                sum(comparison.agrees for comparison in comparisons) / len(comparisons)
                if comparisons
                else 0.0
            )
        return values


def _real_reviewer(model: str) -> ReviewerCallable:
    from pipeline.agents.reviewer import run_reviewer

    return lambda state: run_reviewer(state, model=model)


def _classification(expected_pass: bool, actual_pass: bool) -> str:
    if expected_pass and actual_pass:
        return "correct_pass"
    if not expected_pass and not actual_pass:
        return "correct_fail"
    if not expected_pass and actual_pass:
        return "false_pass"
    return "false_fail"


def _comparisons(
    scenario: EvalScenario,
    report: ReviewResult,
) -> tuple[FieldComparison, ...]:
    rubric = report.rubric_scores.model_dump()
    values: list[FieldComparison] = []
    for name in CODE_SCORE_FIELDS:
        values.append(
            FieldComparison(
                kind="code",
                name=name,
                expected=scenario.expected_code_scores[name],
                actual=rubric[name],
            )
        )
    for name in JUDGMENT_FIELDS:
        values.append(
            FieldComparison(
                kind="judgment",
                name=name,
                expected=scenario.expected_judgments[name],
                actual=rubric[name],
            )
        )
    return tuple(values)


def run_harness(
    scenarios: Iterable[EvalScenario] = SCENARIOS,
    *,
    reviewer: ReviewerCallable | None = None,
    model: str = "flash-lite",
    repeats: int = 1,
) -> HarnessSummary:
    """Evaluate cases and compare the verdict and every rubric field."""

    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    active_reviewer = reviewer or _real_reviewer(model)

    results: list[ScenarioResult] = []
    for scenario in scenarios:
        for repeat in range(1, repeats + 1):
            state = scenario.build_state()
            input_tokens_before = state.input_tokens
            output_tokens_before = state.output_tokens
            started = perf_counter()
            output = active_reviewer(state)
            duration_seconds = perf_counter() - started
            if output.get("stage") == Stage.FAILED or "review" not in output:
                raise RuntimeError(
                    f"Reviewer failed for {scenario.name}: {output.get('errors', [])}"
                )
            report: ReviewResult = output["review"]
            actual_pass = report.overall_status == "pass"
            input_tokens = output.get("input_tokens")
            output_tokens = output.get("output_tokens")
            if not isinstance(input_tokens, int):
                input_tokens = max(0, state.input_tokens - input_tokens_before)
            if not isinstance(output_tokens, int):
                output_tokens = max(0, state.output_tokens - output_tokens_before)
            results.append(
                ScenarioResult(
                    name=scenario.name,
                    domain=scenario.domain,
                    label_status=scenario.label_status,
                    label_rationale=scenario.label_rationale,
                    repeat=repeat,
                    expected_pass=scenario.expected_pass,
                    actual_pass=actual_pass,
                    classification=_classification(
                        scenario.expected_pass,
                        actual_pass,
                    ),
                    duration_seconds=duration_seconds,
                    input_tokens=max(0, input_tokens),
                    output_tokens=max(0, output_tokens),
                    comparisons=_comparisons(scenario, report),
                    report=report,
                )
            )

    return HarnessSummary(results=tuple(results), model=model)


def _print_summary(summary: HarnessSummary) -> None:
    complete = sum(result.agrees for result in summary.results)
    print(f"Model: {summary.model}")
    print(
        f"Complete case agreement: {summary.complete_case_agreement:.1%} "
        f"({complete}/{len(summary.results)})"
    )
    print(f"Verdict agreement: {summary.verdict_agreement:.1%}")
    print(f"Rubric-field agreement: {summary.rubric_agreement:.1%}")
    print(
        f"Safety errors: false_passes={summary.false_passes} "
        f"false_failures={summary.false_failures}"
    )
    print(
        f"Reviewer usage: input_tokens={summary.total_input_tokens} "
        f"output_tokens={summary.total_output_tokens} "
        f"average_latency={summary.average_duration_seconds:.2f}s"
    )
    print("Per-field agreement:")
    for name, agreement in summary.per_field_agreement.items():
        print(f"  {name}: {agreement:.1%}")

    for result in summary.results:
        marker = "OK" if result.agrees else "DISAGREE"
        print(
            f"[{marker}] {result.name} run={result.repeat}: "
            f"expected={'pass' if result.expected_pass else 'fail'}, "
            f"actual={'pass' if result.actual_pass else 'fail'}, "
            f"classification={result.classification}, labels={result.label_status}"
        )
        print(
            f"  usage: input={result.input_tokens}, output={result.output_tokens}, "
            f"latency={result.duration_seconds:.2f}s"
        )
        for comparison in result.disagreements:
            print(
                f"  {comparison.kind}.{comparison.name}: "
                f"expected={comparison.expected}, actual={comparison.actual}"
            )
            if comparison.kind == "judgment":
                reason = getattr(result.report.judgment_reasons, comparison.name)
                print(f"    reason: {reason or '(no reason supplied)'}")


def summary_to_dict(summary: HarnessSummary) -> dict:
    """Return a JSON-serialisable audit record for one harness execution."""

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": summary.model,
        "metrics": {
            "complete_case_agreement": summary.complete_case_agreement,
            "verdict_agreement": summary.verdict_agreement,
            "rubric_agreement": summary.rubric_agreement,
            "false_passes": summary.false_passes,
            "false_failures": summary.false_failures,
            "average_duration_seconds": summary.average_duration_seconds,
            "total_input_tokens": summary.total_input_tokens,
            "total_output_tokens": summary.total_output_tokens,
            "per_field_agreement": summary.per_field_agreement,
        },
        "results": [
            {
                "name": result.name,
                "domain": result.domain,
                "label_status": result.label_status,
                "label_rationale": result.label_rationale,
                "repeat": result.repeat,
                "expected_pass": result.expected_pass,
                "actual_pass": result.actual_pass,
                "classification": result.classification,
                "complete_agreement": result.agrees,
                "duration_seconds": result.duration_seconds,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "comparisons": [
                    {
                        "kind": comparison.kind,
                        "name": comparison.name,
                        "expected": comparison.expected,
                        "actual": comparison.actual,
                        "agrees": comparison.agrees,
                    }
                    for comparison in result.comparisons
                ],
                "report": result.report.model_dump(mode="json"),
            }
            for result in summary.results
        ],
    }


def write_summary(summary: HarnessSummary, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary_to_dict(summary), indent=2),
        encoding="utf-8",
    )
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="flash-lite")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Path to a labeled JSON case containing a saved ArchitectState.",
    )
    parser.add_argument("--output", help="Optional JSON result path.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    scenarios = (
        tuple(load_scenario_file(path) for path in args.case)
        if args.case
        else SCENARIOS
    )
    summary = run_harness(
        scenarios,
        model=args.model,
        repeats=args.repeats,
    )
    output_path = write_summary(summary, args.output) if args.output else None
    _print_summary(summary)
    if output_path:
        print(f"Wrote: {output_path}")
    return 0 if all(result.agrees for result in summary.results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
