"""Offline tests for evaluation-harness behavior; no API key required."""
from __future__ import annotations

import json

import pytest

from eval.harness import run_harness, write_summary
from eval.scenarios import (
    CODE_SCORE_FIELDS,
    EvalScenario,
    JUDGMENT_FIELDS,
    SCENARIOS,
    load_scenario_file,
    sound_healthcare_state,
)
from pipeline.review_checks import run_deterministic_checks
from pipeline.state import JudgmentReasons, ReviewResult, RubricScores, Stage


def _report(
    *,
    passed: bool,
    code_scores: dict[str, int],
    judgments: dict[str, bool],
) -> ReviewResult:
    return ReviewResult(
        rubric_version="3.0",
        overall_status="pass" if passed else "fail",
        rubric_scores=RubricScores(**code_scores, **judgments),
        judgment_reasons=JudgmentReasons(
            **{name: "offline labeled fixture" for name in JUDGMENT_FIELDS}
        ),
        requires_refinement=not passed,
    )


def _oracle_reports(scenarios=SCENARIOS):
    return iter(
        _report(
            passed=scenario.expected_pass,
            code_scores=dict(scenario.expected_code_scores),
            judgments=dict(scenario.expected_judgments),
        )
        for scenario in scenarios
    )


def test_harness_requires_verdict_and_every_rubric_field_to_agree():
    reports = _oracle_reports()

    def reviewer(_state):
        report = next(reports)
        return {
            "review": report,
            "stage": Stage.DONE if report.overall_status == "pass" else Stage.REFINING,
        }

    summary = run_harness(reviewer=reviewer, model="offline-oracle")

    assert summary.complete_case_agreement == 1.0
    assert summary.verdict_agreement == 1.0
    assert summary.rubric_agreement == 1.0
    assert summary.false_passes == 0
    assert summary.false_failures == 0


def test_seed_case_code_labels_match_the_real_deterministic_checks():
    for scenario in SCENARIOS:
        checks = run_deterministic_checks(scenario.build_state())
        actual = {
            name: getattr(checks, f"score_{name}")
            for name in CODE_SCORE_FIELDS
        }
        assert actual == dict(scenario.expected_code_scores), scenario.name


def test_correct_final_verdict_for_wrong_reason_is_a_disagreement():
    scenario = next(item for item in SCENARIOS if not item.expected_pass)
    wrong_judgments = dict(scenario.expected_judgments)
    expected_false = next(name for name, value in wrong_judgments.items() if not value)
    wrong_judgments[expected_false] = True
    wrong_judgments["repo_grounding"] = False
    report = _report(
        passed=False,
        code_scores=dict(scenario.expected_code_scores),
        judgments=wrong_judgments,
    )

    summary = run_harness(
        [scenario],
        reviewer=lambda _state: {"review": report, "stage": Stage.REFINING},
    )

    assert summary.verdict_agreement == 1.0
    assert summary.complete_case_agreement == 0.0
    assert summary.rubric_agreement < 1.0
    assert summary.results[0].disagreements


def test_harness_reports_false_pass_explicitly():
    scenario = next(item for item in SCENARIOS if not item.expected_pass)
    report = _report(
        passed=True,
        code_scores=dict(scenario.expected_code_scores),
        judgments={name: True for name in JUDGMENT_FIELDS},
    )

    summary = run_harness(
        [scenario],
        reviewer=lambda _state: {"review": report, "stage": Stage.DONE},
    )

    assert summary.false_passes == 1
    assert summary.false_failures == 0
    assert summary.verdict_agreement == 0.0


def test_harness_repeats_each_case():
    scenario = SCENARIOS[0]
    calls = 0

    def reviewer(_state):
        nonlocal calls
        calls += 1
        report = _report(
            passed=scenario.expected_pass,
            code_scores=dict(scenario.expected_code_scores),
            judgments=dict(scenario.expected_judgments),
        )
        return {"review": report, "stage": Stage.DONE}

    summary = run_harness([scenario], reviewer=reviewer, repeats=3)

    assert calls == 3
    assert len(summary.results) == 3
    assert summary.complete_case_agreement == 1.0


def test_harness_writes_auditable_json(tmp_path):
    scenario = SCENARIOS[0]
    report = _report(
        passed=scenario.expected_pass,
        code_scores=dict(scenario.expected_code_scores),
        judgments=dict(scenario.expected_judgments),
    )
    summary = run_harness(
        [scenario],
        reviewer=lambda _state: {"review": report, "stage": Stage.DONE},
        model="offline-oracle",
    )

    output = write_summary(summary, tmp_path / "result.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["model"] == "offline-oracle"
    assert payload["metrics"]["complete_case_agreement"] == 1.0
    assert payload["metrics"]["total_input_tokens"] == 0
    assert payload["results"][0]["label_rationale"] == scenario.label_rationale
    assert payload["results"][0]["duration_seconds"] >= 0
    assert payload["results"][0]["report"]["rubric_version"] == "3.0"


def test_saved_pipeline_state_can_be_loaded_as_a_case(tmp_path):
    state = sound_healthcare_state()
    case_path = tmp_path / "healthcare.json"
    case_path.write_text(
        json.dumps(
            {
                "name": "human_healthcare_case",
                "domain": "healthcare",
                "description": "Saved pipeline state",
                "expected_pass": True,
                "expected_code_scores": dict(SCENARIOS[2].expected_code_scores),
                "expected_judgments": dict(SCENARIOS[2].expected_judgments),
                "label_status": "human_reviewed",
                "label_rationale": "Reviewed by two architecture students.",
                "state": state.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )

    scenario = load_scenario_file(case_path)

    assert scenario.name == "human_healthcare_case"
    assert scenario.label_status == "human_reviewed"
    assert scenario.build_state() == state
    assert scenario.build_state() is not state


def test_harness_records_reviewer_token_deltas():
    scenario = SCENARIOS[0]

    def reviewer(state):
        state.input_tokens += 120
        state.output_tokens += 30
        report = _report(
            passed=scenario.expected_pass,
            code_scores=dict(scenario.expected_code_scores),
            judgments=dict(scenario.expected_judgments),
        )
        return {"review": report, "stage": Stage.DONE}

    summary = run_harness([scenario], reviewer=reviewer)

    assert summary.total_input_tokens == 120
    assert summary.total_output_tokens == 30
    assert summary.results[0].duration_seconds >= 0


def test_scenario_rejects_a_verdict_inconsistent_with_criterion_labels():
    base = SCENARIOS[0]

    with pytest.raises(ValueError, match="Inconsistent verdict label"):
        EvalScenario(
            name="inconsistent",
            domain=base.domain,
            description=base.description,
            expected_pass=False,
            expected_code_scores=base.expected_code_scores,
            expected_judgments=base.expected_judgments,
            build_state=base.build_state,
        )
