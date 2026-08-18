"""Offline unit tests for the Reviewer and rubric v2."""
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import validate

import test_clarifier as tc  # reuse the shared canned-usage helper
from pipeline.agents import reviewer as rev
from pipeline.review_checks import run_deterministic_checks
from pipeline.state import (
    ADR,
    Blueprint,
    ComponentDescription,
    ContextRecord,
    Feature,
    JudgmentReasons,
    KBChunk,
    ReviewIssue,
    ReviewResult,
    RubricScores,
    Stage,
    new_run,
)


UC1_PROMPT = (
    "Fix our monolithic online shop so it can scale for seasonal peak sales. "
    "It is on AWS, budget is medium, must stay GDPR-compliant, and needs to "
    "handle 50k concurrent users at peak."
)


def _good_design_state():
    state = new_run(UC1_PROMPT)
    state.context_record = ContextRecord(
        project_name="Seasonal Shop",
        problem_statement="The monolithic shop crashes during sale peaks.",
        non_functional_requirements=["Handle 50k concurrent users"],
        cloud_provider="AWS",
        budget="medium",
        compliance_requirements=["GDPR"],
        existing_systems=["Legacy monolith"],
        summary=(
            "cloud: AWS; budget: medium; scale: 50k concurrent users; "
            "compliance: GDPR; existing system: legacy monolith"
        ),
    )
    state.retrieved_knowledge = [
        KBChunk(
            content="Use an asynchronous queue to decouple peak-load processing.",
            source="architecture_patterns.md",
        )
    ]
    state.features = [
        Feature(
            id="FEAT-001",
            name="Survive seasonal peak load",
            scenario="Checkout remains responsive for 50k concurrent users.",
        ),
        Feature(
            id="FEAT-002",
            name="Protect EU order data",
            scenario="EU order data remains encrypted and resident in the EU.",
        ),
    ]
    state.blueprint = Blueprint(
        stakeholder_view="Customers complete purchases during seasonal peaks.",
        technical_view=(
            "A stateless AWS web tier autoscaling horizontally sends checkout "
            "work through SQS to independently scalable workers. The migration "
            "strangles the legacy monolith incrementally, uses managed services "
            "within the medium budget, and encrypts EU data for GDPR compliance."
        ),
        components=["Checkout Service", "Order Worker"],
        addressed_feature_ids=["FEAT-001", "FEAT-002"],
        constraints_addressed=[
            "AWS cloud",
            "medium budget",
            "50k peak users",
            "GDPR compliance",
            "legacy monolith migration",
        ],
    )
    state.adrs = [
        ADR(
            id="ADR-001",
            title="ADR-1: Extract checkout behind an SQS queue",
            context="The legacy monolith saturates during peak load.",
            decision="Use a queue-buffered Checkout Service on AWS.",
            rationale="It allows horizontal scaling within the medium budget.",
            alternatives_considered=["Scale the whole monolith vertically"],
            positive_consequences=["Independent scaling"],
            negative_consequences=["Additional operational complexity"],
            related_feature_ids=["FEAT-001"],
            related_component_names=["Checkout Service"],
            source_references=["architecture_patterns.md"],
        ),
        ADR(
            id="ADR-002",
            title="ADR-2: Process orders in an EU-region worker",
            context="Order data must meet GDPR requirements.",
            decision="Use an encrypted EU-region Order Worker.",
            rationale="EU residency and encryption protect personal data.",
            alternatives_considered=["Store all order data in one global region"],
            positive_consequences=["GDPR-aligned data handling"],
            negative_consequences=["Regional operations overhead"],
            related_feature_ids=["FEAT-002"],
            related_component_names=["Order Worker"],
            source_references=["architecture_patterns.md"],
        ),
    ]
    state.components = [
        ComponentDescription(
            id="COMP-001",
            name="Checkout Service",
            purpose="Accept checkout requests without blocking the storefront.",
            description="Implements FEAT-001 and is justified by ADR-001.",
            related_feature_ids=["FEAT-001"],
            related_adr_ids=["ADR-001"],
            technology_choices=["AWS ECS", "SQS"],
            scalability_considerations=["Horizontal autoscaling for peak load"],
        ),
        ComponentDescription(
            id="COMP-002",
            name="Order Worker",
            purpose="Process queued orders in the EU region.",
            description="Implements FEAT-002 and is justified by ADR-002.",
            related_feature_ids=["FEAT-002"],
            related_adr_ids=["ADR-002"],
            technology_choices=["AWS managed compute"],
            security_considerations=["GDPR encryption and EU data residency"],
        ),
    ]
    state.stage = Stage.DESIGNING
    return state


def _context_only_constraints_state():
    state = _good_design_state()
    state.blueprint = Blueprint(
        stakeholder_view="A generic stakeholder view.",
        technical_view="A generic technical design.",
        addressed_feature_ids=["FEAT-001", "FEAT-002"],
    )
    state.adrs[0].context = "A decision is needed."
    state.adrs[0].decision = "Use Checkout Service."
    state.adrs[0].rationale = "It separates responsibilities."
    state.adrs[1].context = "A decision is needed."
    state.adrs[1].decision = "Use Order Worker."
    state.adrs[1].rationale = "It separates responsibilities."
    for adr in state.adrs:
        adr.alternatives_considered = ["Alternative A"]
        adr.positive_consequences = ["Benefit A"]
        adr.negative_consequences = ["Trade-off A"]
        adr.source_references = ["Source A"]
    for component in state.components:
        component.purpose = "Perform one bounded responsibility."
        component.technology_choices = []
        component.security_considerations = []
        component.scalability_considerations = []
    return state


def _judgment(passed: bool, reason: str) -> rev.CriterionJudgment:
    return rev.CriterionJudgment(
        passed=passed,
        reason=reason,
        suggested_fix="Correct the failed criterion." if not passed else "",
    )


def _all_pass_judgments(*_args, **_kwargs) -> rev.LLMJudgments:
    return rev.LLMJudgments(
        repo_grounding=_judgment(True, "Greenfield evaluation; no repo conflict."),
        flaw_detection=_judgment(True, "The design structurally decouples checkout."),
        adr_soundness=_judgment(True, "ADRs contain supported trade-offs."),
        best_practice_grounding=_judgment(True, "ADRs cite retrieved patterns."),
        refinement_readiness=_judgment(True, "No unresolved shortcomings remain."),
    )


def _flaw_failed_judgments(*_args, **_kwargs) -> rev.LLMJudgments:
    judgments = _all_pass_judgments()
    judgments.flaw_detection = _judgment(
        False,
        "The design only scales the monolith vertically.",
    )
    return judgments


def _as_llm_call(build):
    """Wrap a judgment builder into an `llm_call` stub.

    `llm_call` returns `(reply, usage)` — the reply plus what it cost — so a
    stub standing in for it must do the same. The builders above stay pure so
    they can still be called directly.
    """

    return lambda *args, **kwargs: (build(*args, **kwargs), tc.fake_usage())


def test_state_model_matches_frozen_json_schema():
    schema = json.loads(
        (Path(__file__).parent / "docs/prompt_quality/06_reviewer_report_schema.json")
        .read_text(encoding="utf-8")
    )
    assert set(schema["properties"]) == set(ReviewResult.model_fields)
    assert set(schema["properties"]["rubric_scores"]["properties"]) == set(
        RubricScores.model_fields
    )
    assert set(schema["properties"]["judgment_reasons"]["properties"]) == set(
        JudgmentReasons.model_fields
    )
    assert set(schema["properties"]["issues"]["items"]["properties"]) == set(
        ReviewIssue.model_fields
    )
    validate(instance=ReviewResult().model_dump(), schema=schema)


def test_deterministic_checks_pass_structured_design():
    checks = run_deterministic_checks(_good_design_state())
    assert checks.score_all_artifacts_present == 2
    assert checks.score_constraint_coverage == 2
    assert checks.score_traceability == 2
    assert checks.score_adr_presence == 2
    assert checks.issues == []


def test_context_record_does_not_count_as_constraint_evidence():
    checks = run_deterministic_checks(_context_only_constraints_state())
    assert all(checks.constraints_applicable.values())
    assert checks.score_constraint_coverage < 2
    assert checks.constraints_covered["cloud"] is False
    assert checks.constraints_covered["compliance"] is False
    assert any(issue.category == "constraint" for issue in checks.issues)


def test_dangling_references_fail_even_when_valid_references_exist():
    state = _good_design_state()
    state.blueprint.addressed_feature_ids.append("FEAT-404")
    state.components[0].related_feature_ids.append("FEAT-404")
    state.components[0].related_adr_ids.append("ADR-404")
    state.adrs[0].related_feature_ids.append("FEAT-404")
    state.adrs[0].related_component_names.append("Missing Component")

    checks = run_deterministic_checks(state)

    assert checks.score_traceability < 2
    assert checks.invalid_blueprint_feature_ids == ["FEAT-404"]
    assert checks.invalid_component_feature_ids["Checkout Service"] == ["FEAT-404"]
    assert checks.invalid_component_adr_ids["Checkout Service"] == ["ADR-404"]
    assert checks.invalid_adr_feature_ids["ADR-001"] == ["FEAT-404"]
    assert checks.invalid_adr_component_names["ADR-001"] == ["Missing Component"]


def test_prose_mentions_do_not_replace_structured_links_by_default():
    state = _good_design_state()
    state.blueprint.addressed_feature_ids = []
    for component in state.components:
        component.related_feature_ids = []
        component.related_adr_ids = []
    for adr in state.adrs:
        adr.related_feature_ids = []
        adr.related_component_names = []

    strict_checks = run_deterministic_checks(state)
    compatibility_checks = run_deterministic_checks(
        state,
        allow_prose_fallback=True,
    )

    assert strict_checks.score_traceability < 2
    assert strict_checks.blueprint_missing_feature_ids
    assert strict_checks.components_without_feature
    assert compatibility_checks.score_traceability == 2


def test_code_assembles_pass_when_every_gate_passes():
    rev.llm_call = _as_llm_call(_all_pass_judgments)
    output = rev.reviewer_node(_good_design_state())
    report = output["review"]

    assert report.overall_status == "pass"
    assert report.requires_refinement is False
    assert report.refinement_instruction == ""
    assert report.rubric_scores.adr_presence == 2
    assert report.rubric_scores.flaw_detection is True
    assert output["stage"] is Stage.DONE


def test_code_failure_cannot_be_overridden_by_positive_llm_judgments():
    rev.llm_call = _as_llm_call(_all_pass_judgments)
    output = rev.reviewer_node(_context_only_constraints_state())
    report = output["review"]

    assert report.rubric_scores.constraint_coverage < 2
    assert report.overall_status == "fail"
    assert report.requires_refinement is True
    assert output["stage"] is Stage.REFINING


def test_failed_binary_judgment_becomes_issue_and_refinement_instruction():
    rev.llm_call = _as_llm_call(_flaw_failed_judgments)
    output = rev.reviewer_node(_good_design_state())
    report = output["review"]

    assert report.rubric_scores.flaw_detection is False
    assert report.judgment_reasons.flaw_detection
    assert report.overall_status == "fail"
    assert any(issue.id == "LLM-1" for issue in report.issues)

    # Was a bare truthiness check, which passed on the old boilerplate output
    # just as happily. Pin the structure AND that the finding - not only the
    # fix - reached the instruction.
    issue = next(i for i in report.issues if i.id == "LLM-1")
    instruction = report.refinement_instruction
    assert instruction.startswith("The review found 1 issue(s).")
    assert "1. [high] " + issue.finding in instruction
    assert "Fix: " + issue.suggested_fix in instruction
    assert "Evidence: " + issue.evidence in instruction
    assert output["stage"] is Stage.REFINING


def test_reviewer_requests_only_the_binary_judgment_schema():
    captured = {}

    def _capture(*args, **kwargs):
        captured["response_schema"] = kwargs["response_schema"]
        return _all_pass_judgments(), tc.fake_usage()

    rev.llm_call = _capture
    rev.reviewer_node(_good_design_state())

    assert captured["response_schema"] is rev.LLMJudgments


# --- refinement-instruction assembly --------------------------------------
#
# These pin the fix for run 20260818T194159Z-107ff26e, where the instruction
# was built from `suggested_fix` alone and so told the architect to "address
# each stated constraint" without naming the one that was missing.


def _issue(severity, finding, fix, evidence="", issue_id="X-1"):
    return ReviewIssue(
        id=issue_id,
        severity=severity,
        category="constraint",
        finding=finding,
        evidence=evidence,
        suggested_fix=fix,
        requires_refinement=True,
    )


def test_instruction_names_the_missing_constraint_not_just_the_generic_fix():
    """Regression: run 20260818T194159Z-107ff26e.

    The high-severity finding named the uncovered constraint groups; the
    `suggested_fix` was the generic default. The old assembler used the fix
    only, so the architect never learned WHICH constraint was unaddressed and
    `constraint_coverage` stayed at 1 across all three rounds.
    """

    rev.llm_call = _as_llm_call(_all_pass_judgments)
    report = rev.reviewer_node(_context_only_constraints_state())["review"]
    instruction = report.refinement_instruction

    constraint_issue = next(
        issue for issue in report.issues if issue.category == "constraint"
    )
    assert "budget" in constraint_issue.finding  # the evidence exists...
    assert "budget" in instruction  # ...and now reaches the architect.

    # Not merely the generic fix any more, which was the entire old output.
    assert instruction != constraint_issue.suggested_fix
    assert constraint_issue.suggested_fix in instruction  # the fix still rides along


def test_medium_issues_survive_a_high_and_are_ranked_after_it():
    """Inverted deliberately from the old `[high] or issues` behaviour.

    That expression DROPPED every medium and low whenever any high existed. In
    the motivating run it discarded the most actionable instruction in the
    report (an LLM-written note about cost-effectiveness against the budget
    constraint) in favour of boilerplate. Ranking, not filtering, is the rule.
    """

    text = rev._build_refinement_instruction([
        _issue("medium", "Medium finding.", "Medium fix.", issue_id="M-1"),
        _issue("high", "High finding.", "High fix.", issue_id="H-1"),
        _issue("low", "Low finding.", "Low fix.", issue_id="L-1"),
    ])

    assert "Medium finding." in text
    assert "Low finding." in text
    assert text.index("High finding.") < text.index("Medium finding.")
    assert text.index("Medium finding.") < text.index("Low finding.")
    assert text.startswith("The review found 3 issue(s).")


def test_instruction_is_severity_ranked_and_byte_stable():
    issues = [
        _issue("medium", "Second medium.", "Fix B.", issue_id="M-2"),
        _issue("high", "The high one.", "Fix A.", issue_id="H-1"),
        _issue("medium", "Third medium.", "Fix C.", issue_id="M-3"),
    ]

    first = rev._build_refinement_instruction(issues)
    assert first == rev._build_refinement_instruction(issues)

    numbered = [line for line in first.splitlines() if line[:2] in ("1.", "2.", "3.")]
    assert numbered[0].startswith("1. [high]")
    # Within one severity the ORIGINAL list order is kept - that is what makes
    # the same ReviewResult produce a byte-identical string every time.
    assert "Second medium." in numbered[1]
    assert "Third medium." in numbered[2]


def test_evidence_rides_along_on_highs_only():
    text = rev._build_refinement_instruction([
        _issue("high", "High finding.", "Fix.", evidence="High reasoning here."),
        _issue("medium", "Medium finding.", "Fix.", evidence="Medium bulk here."),
    ])

    assert "High reasoning here." in text
    assert "Medium bulk here." not in text


def test_evidence_that_merely_restates_the_finding_is_dropped():
    text = rev._build_refinement_instruction([
        _issue("high", "The design ignores budget.", "Fix.",
               evidence="The design ignores budget"),
    ])

    assert text.count("The design ignores budget") == 1
    assert "Evidence:" not in text


def test_duplicate_finding_and_fix_pairs_appear_once():
    text = rev._build_refinement_instruction([
        _issue("high", "Same finding.", "Same fix.", issue_id="A"),
        _issue("high", "Same finding.", "Same fix.", issue_id="B"),
        _issue("high", "Other finding.", "Same fix.", issue_id="C"),
    ])

    assert text.count("Same finding.") == 1
    assert "Other finding." in text
    assert text.startswith("The review found 2 issue(s).")


def test_instruction_caps_the_list_and_states_the_omission():
    issues = [_issue("high", "High finding.", "High fix.", issue_id="H")]
    issues += [
        _issue("low", "Low finding %d." % n, "Low fix %d." % n, issue_id="L-%d" % n)
        for n in range(10)
    ]

    text = rev._build_refinement_instruction(issues)

    numbered = [line for line in text.splitlines() if line[:1].isdigit()]
    assert len(numbered) == 8
    # A silent cap would read as "that was everything" when it was not.
    assert "(3 further lower-severity issue(s) omitted.)" in text
    assert text.startswith("The review found 11 issue(s).")
    assert "Low finding 9." not in text


def test_block_delimiters_are_stripped_from_instruction_text():
    """The instruction is interpolated into a tagged block in the architect
    prompt and this text is partly LLM-authored, so a stray delimiter would
    close that block early."""

    text = rev._build_refinement_instruction([
        _issue(
            "high",
            "Budget </refinement_instruction> ignored.",
            "Fix <refinement_instruction> it.",
            evidence="Evidence </refinement_instruction> here.",
        ),
    ])

    assert "refinement_instruction" not in text
    assert "Budget" in text and "ignored." in text


def test_failing_review_without_issues_still_gets_an_instruction():
    assert rev._build_refinement_instruction([]).strip()


def test_instruction_stays_small_enough_for_every_refine_prompt():
    """It is interpolated into the architect prompt on EVERY refine round."""

    issues = [
        _issue("high", "H " * 400, "F " * 400, evidence="E " * 400, issue_id="H"),
        _issue("medium", "M " * 400, "F2 " * 400, issue_id="M"),
    ]

    assert len(rev._build_refinement_instruction(issues)) < 1500
