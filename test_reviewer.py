"""test_reviewer.py — offline unit tests for the Reviewer

No API key or network needed: the LLM is mocked by replacing `llm_call` with a
function returning a canned `ReviewResult`. This isolates what we own — the
deterministic checks and the merge/routing arithmetic — from the model, exactly
like test_clarifier.py does for the Clarifier.

Run either way:
    python -m pytest test_reviewer.py
    python test_reviewer.py
"""
from __future__ import annotations

import json
from pathlib import Path

from pipeline.agents import reviewer as rev
from pipeline.review_checks import run_deterministic_checks
from pipeline.state import (
    ADR,
    Blueprint,
    ComponentDescription,
    ContextRecord,
    Feature,
    KBChunk,
    ReviewIssue,
    ReviewResult,
    RubricScores,
    Stage,
    new_run,
)

UC1_PROMPT = (
    "Fix our monolithic online shop so it can scale for seasonal peak sales. "
    "It's on AWS, budget is medium, must stay GDPR-compliant, and needs to handle "
    "~50k concurrent users at peak. Repo: https://github.com/example/bugged-shop"
)


# ── state fixtures ──────────────────────────────────────────────────────────
def _good_design_state():
    """Use-case #1 with a design that satisfies every deterministic check."""
    s = new_run(UC1_PROMPT)
    s.context_record = ContextRecord(
        summary="cloud: AWS\nbudget: medium\nscale: 50k concurrent users at peak\n"
                "compliance: GDPR\nexisting system: monolithic shop (brownfield)"
    )
    s.retrieved_knowledge = [
        KBChunk(content="Decouple services with an async message queue to absorb peak load.",
                source="architecture_patterns.md"),
    ]
    s.features = [
        Feature(id="F1", name="Survive seasonal peak load",
                scenario="Given a sale event, when 50k users hit the shop concurrently, checkout stays responsive."),
        Feature(id="F2", name="GDPR-compliant order data",
                scenario="Given an EU customer order, personal data is stored encrypted in the EU."),
    ]
    s.blueprint = Blueprint(
        stakeholder_view="Customers keep shopping and checking out during seasonal sale peaks without outages.",
        technical_view="Stateless web tier on AWS ECS with autoscaling; checkout decoupled from the "
                       "legacy monolith via an SQS queue; scaling stays within the medium budget; "
                       "order data encrypted at rest for GDPR compliance.",
    )
    s.adrs = [
        ADR(title="ADR-1: Extract checkout into CheckoutService behind an SQS queue",
            decision="Split checkout out of the existing monolith so peak load is absorbed "
                     "asynchronously; horizontal scaling keeps cost inside the medium budget."),
        ADR(title="ADR-2: Run order processing in OrderWorker with EU data residency",
            decision="A queue consumer processes orders; GDPR compliance via encryption at rest "
                     "and EU-region storage."),
    ]
    s.components = [
        ComponentDescription(name="CheckoutService",
                             description="Stateless checkout API on AWS ECS, autoscaled for peak load (traces to F1)."),
        ComponentDescription(name="OrderWorker",
                             description="Consumes orders from the SQS queue asynchronously; GDPR-compliant storage (traces to F1, F2)."),
    ]
    s.stage = Stage.DESIGNING
    return s


def _stub_design_state():
    """Use-case #1 with exactly the artifacts today's architect STUB writes."""
    s = new_run(UC1_PROMPT)
    s.context_record = ContextRecord(summary="domain: e-commerce")
    s.features = [Feature(id="F1", name="Handle peak load",
                          scenario="Stays responsive at 50k concurrent users.")]
    s.blueprint = Blueprint(
        stakeholder_view="[stub] Business view of the system.",
        technical_view="[stub] Services, data model, integration.",
    )
    s.adrs = [ADR(title="ADR-1: split monolith into services", decision="[stub] decision text")]
    s.components = [ComponentDescription(name="OrderService",
                                         description="[stub] owns orders (traces to F1).")]
    s.stage = Stage.DESIGNING
    return s


# ── canned LLM judgments ───────────────────────────────────────────────────
def _llm_all_perfect(state, prompt, *, system="", model="", response_schema=None):
    """A (naive or drifting) model that claims everything is perfect."""
    return ReviewResult(
        overall_status="pass",
        score_total=16,
        rubric_scores=RubricScores(
            all_artifacts_present=2, constraint_coverage=2, repo_grounding=2,
            flaw_detection=2, traceability=2, adr_quality=2,
            best_practice_grounding=2, refinement_readiness=2,
        ),
        flaw_detected=True,
        issues=[],
        requires_refinement=False,
        refinement_instruction="",
    )


def _llm_flaw_missed(state, prompt, *, system="", model="", response_schema=None):
    """Model correctly judges that the seeded flaw was NOT addressed."""
    return ReviewResult(
        overall_status="fail",
        score_total=0,
        rubric_scores=RubricScores(
            all_artifacts_present=0, constraint_coverage=0, repo_grounding=0,
            flaw_detection=0, traceability=0, adr_quality=1,
            best_practice_grounding=0, refinement_readiness=1,
        ),
        flaw_detected=False,
        issues=[ReviewIssue(
            id="LLM-1", severity="high", category="grounding",
            finding="The design is placeholder text and never addresses the peak-load flaw.",
            evidence="technical_view: '[stub] Services, data model, integration.'",
            suggested_fix="Design the checkout decomposition that structurally fixes the flaw.",
            requires_refinement=True,
        )],
        requires_refinement=True,
        refinement_instruction="",  # left empty on purpose: merge must synthesise one
    )


# ── tests: schema contract ─────────────────────────────────────────────────
def test_state_model_matches_frozen_json_schema():
    """ReviewResult must stay a field-for-field mirror of Maheen-facing schema 06."""
    schema = json.loads(
        (Path(__file__).parent / "docs/prompt_quality/06_reviewer_report_schema.json")
        .read_text(encoding="utf-8")
    )
    assert set(schema["properties"]) == set(ReviewResult.model_fields)
    assert set(schema["properties"]["rubric_scores"]["properties"]) == set(RubricScores.model_fields)
    assert set(schema["properties"]["issues"]["items"]["properties"]) == set(ReviewIssue.model_fields)


# ── tests: deterministic layer ─────────────────────────────────────────────
def test_deterministic_checks_pass_good_design():
    checks = run_deterministic_checks(_good_design_state())
    assert checks.score_all_artifacts_present == 2
    assert checks.score_constraint_coverage == 2
    assert checks.score_traceability == 2
    assert checks.score_adr_presence == 2
    assert checks.issues == []


def test_deterministic_checks_flag_stub_design():
    checks = run_deterministic_checks(_stub_design_state())
    # stub artifacts exist and are filled, and F1 is traced — those pass
    assert checks.score_all_artifacts_present == 2
    assert checks.score_traceability == 2
    # but the design text addresses almost no constraints (only 'monolith')
    assert checks.score_constraint_coverage == 0
    assert checks.constraints_covered["cloud"] is False
    assert checks.constraints_covered["compliance"] is False
    # and OrderService is backed by no ADR that names it
    assert checks.components_without_adr == ["OrderService"]
    assert checks.score_adr_presence == 1
    # failures became concrete issues, at least one high-severity
    assert any(i.severity == "high" for i in checks.issues)


# ── tests: merge + routing arithmetic ──────────────────────────────────────
def test_llm_cannot_override_code_scores():
    """Even if the model claims 16/16, code-owned scores win in the merge."""
    rev.llm_call = _llm_all_perfect
    out = rev.reviewer_node(_stub_design_state())
    report = out["review"]
    assert report.rubric_scores.constraint_coverage == 0      # code said 0, LLM said 2
    assert report.rubric_scores.adr_quality == 1              # min(code 1, LLM 2)
    assert report.score_total < 16
    # the uncovered constraints are a high-severity code issue -> must refine
    assert report.requires_refinement is True
    assert out["stage"] is Stage.REFINING


def test_reviewer_passes_good_design():
    rev.llm_call = _llm_all_perfect
    out = rev.reviewer_node(_good_design_state())
    report = out["review"]
    assert report.score_total == 16
    assert report.overall_status == "pass"
    assert report.requires_refinement is False
    assert report.refinement_instruction == ""
    assert out["stage"] is Stage.DONE


def test_reviewer_flags_flawed_design_and_synthesises_instruction():
    rev.llm_call = _llm_flaw_missed
    out = rev.reviewer_node(_stub_design_state())
    report = out["review"]
    # flaw_detected rates the DESIGN: False here means the design failed to
    # address the flaw — i.e. the Reviewer caught it. See state.ReviewResult.
    assert report.flaw_detected is False
    assert report.overall_status == "fail"
    assert report.requires_refinement is True
    assert report.refinement_instruction != ""            # synthesised from issues
    # code issues come first, LLM issues appended
    assert report.issues[0].id.startswith("DET-")
    assert any(i.id == "LLM-1" for i in report.issues)
    assert out["stage"] is Stage.REFINING


def test_high_severity_llm_issue_forces_refinement_despite_pass_score():
    def _llm_high_issue(state, prompt, *, system="", model="", response_schema=None):
        report = _llm_all_perfect(state, prompt)
        report.issues = [ReviewIssue(
            id="LLM-1", severity="high", category="safety",
            finding="Payment data crosses an unencrypted boundary.",
            evidence="technical view omits encryption on the queue.",
            suggested_fix="Encrypt the queue payload.",
            requires_refinement=True,
        )]
        return report

    rev.llm_call = _llm_high_issue
    out = rev.reviewer_node(_good_design_state())
    report = out["review"]
    assert report.score_total >= rev.PASS_THRESHOLD
    assert report.requires_refinement is True             # high severity overrides score
    assert out["stage"] is Stage.REFINING


if __name__ == "__main__":
    test_state_model_matches_frozen_json_schema()
    print("PASS  state model mirrors frozen JSON schema 06")
    test_deterministic_checks_pass_good_design()
    print("PASS  deterministic checks pass a good design")
    test_deterministic_checks_flag_stub_design()
    print("PASS  deterministic checks flag the stub design")
    test_llm_cannot_override_code_scores()
    print("PASS  LLM cannot override code-owned scores")
    test_reviewer_passes_good_design()
    print("PASS  good design -> pass -> DONE")
    test_reviewer_flags_flawed_design_and_synthesises_instruction()
    print("PASS  flawed design -> fail -> REFINING (+ synthesised instruction)")
    test_high_severity_llm_issue_forces_refinement_despite_pass_score()
    print("PASS  high-severity issue forces refinement despite pass score")
    print("\nALL REVIEWER TESTS PASSED")
