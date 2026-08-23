"""Offline unit tests for the Reviewer and generalized rubric v3."""
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import validate

import test_clarifier as tc  # reuse the shared canned-usage helper
from pipeline.agents import reviewer as rev
from pipeline.review_checks import requirement_catalog, run_deterministic_checks
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
        business_goal="Customers complete purchases during seasonal peaks.",
        problem_statement="The monolithic shop crashes during sale peaks.",
        functional_requirements=["Customers complete purchases"],
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
            acceptance_criteria=["Peak checkout remains available."],
            related_requirement_ids=[
                "Customers complete purchases", "Handle 50k concurrent users",
            ],
        ),
        Feature(
            id="FEAT-002",
            name="Protect EU order data",
            scenario="EU order data remains encrypted and resident in the EU.",
            acceptance_criteria=["EU order data stays encrypted in an EU region."],
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


# --- uncovered `must` feature is refinement-blocking ------------------------
#
# Regression for the final run 20260822T222414Z-1788d214: FEAT-005
# ("Traffic-Resilient Infrastructure", priority=must) had no implementing
# component, but the check classified it MEDIUM with
# requires_refinement=False — a required capability with literally no
# implementation reached DONE only because the refinement-cost cap was hit,
# instead of being blocked. The Reviewer still does not invent a component
# mapping; it only escalates the severity of an uncovered MUST feature so the
# Architect receives it as a real blocker.


def test_uncovered_must_feature_is_refinement_blocking():
    state = _good_design_state()
    assert state.features[0].priority == "must"
    state.components[0].related_feature_ids = []
    state.components[0].description = "Justified by ADR-001; implements no feature."

    checks = run_deterministic_checks(state)

    assert "FEAT-001" in checks.features_without_component
    issue = next(
        i for i in checks.issues
        if i.category == "traceability" and "FEAT-001" in i.finding
        and "no implementing component" in i.finding
    )
    assert issue.severity == "high"
    assert issue.requires_refinement is True


def test_uncovered_lower_priority_feature_keeps_existing_non_blocking_behavior():
    state = _good_design_state()
    state.features.append(
        Feature(
            id="FEAT-003",
            name="Nice-to-have export",
            scenario="A shopper exports their order history as CSV.",
            acceptance_criteria=["CSV export succeeds."],
            priority="should",
        )
    )
    state.blueprint.addressed_feature_ids.append("FEAT-003")

    checks = run_deterministic_checks(state)

    assert "FEAT-003" in checks.features_without_component
    issue = next(
        i for i in checks.issues
        if i.category == "traceability" and "FEAT-003" in i.finding
        and "no implementing component" in i.finding
    )
    assert issue.severity == "medium"
    assert issue.requires_refinement is False


def test_covered_must_feature_produces_no_uncovered_feature_finding():
    checks = run_deterministic_checks(_good_design_state())

    assert checks.features_without_component == []
    assert not any(
        "no implementing component" in issue.finding for issue in checks.issues
    )


def test_code_assembles_pass_when_every_gate_passes():
    rev.llm_call = _as_llm_call(_all_pass_judgments)
    output = rev.reviewer_node(_good_design_state())
    report = output["review"]

    assert report.overall_status == "pass"
    assert report.requires_refinement is False
    assert report.refinement_instruction == ""
    assert report.rubric_scores.adr_presence == 2
    assert report.rubric_scores.source_integrity == 2
    assert report.rubric_scores.flaw_detection is True
    assert report.rubric_version == "3.0"
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


def test_reviewer_prompt_is_derived_from_the_current_run(monkeypatch):
    from eval.scenarios import sound_healthcare_state

    captured = {}

    def _capture(_state, prompt, **_kwargs):
        captured["prompt"] = prompt
        return _all_pass_judgments(), tc.fake_usage()

    monkeypatch.setattr(rev, "llm_call", _capture)
    rev.reviewer_node(sound_healthcare_state())

    prompt = captured["prompt"].lower()
    assert "appointment platform" in prompt
    assert "seasonal shop" not in prompt
    assert "<ground_truth_flaw>" not in prompt
    assert not hasattr(rev, "GROUND_TRUTH_FLAW")


def test_yes_without_an_evidence_backed_reason_cannot_pass(monkeypatch):
    def _blank_reason(*_args, **_kwargs):
        judgments = _all_pass_judgments()
        judgments.best_practice_grounding = rev.CriterionJudgment(
            passed=True,
            reason="",
            suggested_fix="",
        )
        return judgments, tc.fake_usage()

    monkeypatch.setattr(rev, "llm_call", _blank_reason)
    output = rev.reviewer_node(_good_design_state())

    assert output["review"].overall_status == "fail"
    assert output["review"].rubric_scores.best_practice_grounding is False
    assert output["stage"] is Stage.REFINING


def test_requested_repository_cannot_be_treated_as_greenfield(monkeypatch):
    state = _good_design_state()
    state.initial_request.repo_url = "https://example.invalid/missing-repo"
    monkeypatch.setattr(rev, "llm_call", _as_llm_call(_all_pass_judgments))

    output = rev.reviewer_node(state)

    assert output["review"].overall_status == "fail"
    assert "repo_grounding" not in output["review"].not_applicable
    assert any(
        issue.category == "repo_alignment"
        for issue in output["review"].issues
    )


def test_fabricated_source_references_fail_in_code(monkeypatch):
    state = _good_design_state()
    for adr in state.adrs:
        adr.source_references = ["fabricated-source.pdf"]
    monkeypatch.setattr(rev, "llm_call", _as_llm_call(_all_pass_judgments))

    output = rev.reviewer_node(state)

    assert output["review"].rubric_scores.source_integrity == 0
    assert output["review"].overall_status == "fail"
    assert any(issue.category == "evidence" for issue in output["review"].issues)


def test_wrong_provider_budget_and_regulation_do_not_count_as_coverage():
    state = _good_design_state()
    state.context_record.cloud_provider = "Azure"
    state.context_record.budget = "low"
    state.context_record.compliance_requirements = ["HIPAA"]

    checks = run_deterministic_checks(state)

    assert checks.constraints_covered["cloud"] is False
    assert checks.constraints_covered["budget"] is False
    assert checks.constraints_covered["compliance"] is False
    assert checks.score_constraint_coverage < 2


def test_features_are_a_required_artifact():
    state = _good_design_state()
    state.features = []

    checks = run_deterministic_checks(state)

    assert checks.artifacts_present["features"] is False
    assert checks.score_all_artifacts_present == 0


# --- cross-artifact target-service consistency ------------------------------
#
# A completed E2E run once passed review (Traceability 2/2, zero blocking
# issues) although FEAT-005 redirected the Cart module to a Cart Service
# that had no Component Description, the Shared Event Bus depended on a
# Notification Service that did not exist, and ADR-002 made Inventory part
# of the target architecture with no owner. The references lived in prose
# and dependency lists, which the structured traceability checks never read.
#
# The fixture below reproduces exactly that shape with every OTHER check
# green, so the only thing that can block the design is the new rule.


def _dangling_services_state():
    state = new_run(
        "Strangle our monolithic shop incrementally so peak seasons stop "
        "taking it down."
    )
    state.context_record = ContextRecord(
        project_name="Strangler Shop",
        business_goal="Peak-season shopping keeps working during migration.",
        summary="brownfield strangler migration of a monolithic shop",
    )
    state.features = [
        Feature(
            id="FEAT-001",
            name="Decouple order processing",
            description="Orders are accepted asynchronously at peak.",
            scenario="A customer order is queued instead of blocking checkout.",
            acceptance_criteria=["Orders are persisted asynchronously."],
        ),
        Feature(
            id="FEAT-002",
            name="Take payments asynchronously",
            description="Payments are drained from the queue.",
            scenario="A queued payment is captured without blocking checkout.",
            acceptance_criteria=["Payments complete off the request path."],
        ),
        Feature(
            id="FEAT-003",
            name="Publish domain events",
            description="Extracted capabilities publish domain events.",
            scenario="An order placement emits a domain event.",
            acceptance_criteria=["Events are published once per change."],
        ),
        Feature(
            id="FEAT-004",
            name="Migrate incrementally",
            description="The monolith shrinks step by step.",
            scenario="A module is extracted without a big-bang rewrite.",
            acceptance_criteria=["Each phase ships independently."],
        ),
        Feature(
            id="FEAT-005",
            name="Redirect Cart",
            description=(
                "The Cart module is redirected to the new Cart Service so "
                "cart traffic leaves the monolith."
            ),
            scenario=(
                "During checkout the Cart module is redirected to the new "
                "Cart Service."
            ),
            acceptance_criteria=[
                "Cart requests no longer run inside the legacy monolith."
            ],
        ),
    ]
    state.blueprint = Blueprint(
        stakeholder_view=(
            "Peak-season shopping keeps working while the shop is "
            "decomposed incrementally."
        ),
        technical_view=(
            "The Order Service and Payment Service are extracted first; the "
            "Shared Event Bus carries their events while the Legacy "
            "Monolith continues to serve the remaining shop."
        ),
        components=[
            "Order Service",
            "Payment Service",
            "Shared Event Bus",
            "Legacy Monolith",
        ],
        data_flows=[
            (
                "Order Service → Shared Event Bus → Payment Service: "
                "OrderCreated events fan out to the payment consumer"
            ),
            (
                "Shared Event Bus → Notification Service: order events for "
                "downstream consumers"
            ),
        ],
        addressed_feature_ids=[
            "FEAT-001", "FEAT-002", "FEAT-003", "FEAT-004", "FEAT-005",
        ],
    )
    state.adrs = [
        ADR(
            id="ADR-001",
            title="ADR-1: Extract Order and Payment behind an event bus",
            context="Synchronous processing saturates the monolith at peak.",
            decision=(
                "The Order Service and Payment Service publish domain "
                "events to the Shared Event Bus."
            ),
            rationale="Asynchronous integration isolates peak load.",
            alternatives_considered=["Scale the monolith vertically"],
            positive_consequences=["Extracted capabilities scale independently."],
            negative_consequences=["Eventual consistency during migration."],
            related_feature_ids=["FEAT-001", "FEAT-002", "FEAT-003"],
            related_component_names=[
                "Order Service", "Payment Service", "Shared Event Bus",
            ],
        ),
        ADR(
            id="ADR-002",
            title="ADR-2: Adopt an event-driven target architecture",
            context=(
                "The remaining shop modules must integrate without new "
                "synchronous coupling."
            ),
            decision=(
                "In the target event-driven architecture, the Order "
                "Service, Payment Service, and Inventory Service "
                "communicate only through domain events."
            ),
            rationale="Event integration keeps the migration incremental.",
            alternatives_considered=["Point-to-point REST integration"],
            positive_consequences=["No new synchronous coupling."],
            negative_consequences=["Event schema governance is required."],
            related_feature_ids=["FEAT-004"],
            related_component_names=["Legacy Monolith"],
        ),
    ]
    state.components = [
        ComponentDescription(
            id="COMP-001",
            name="Order Service",
            purpose="Accept orders asynchronously at peak.",
            description="Implements FEAT-001 and is justified by ADR-001.",
            inputs=["Checkout submissions from the storefront"],
            outputs=["OrderCreated domain events"],
            dependencies=["Shared Event Bus"],
            related_feature_ids=["FEAT-001"],
            related_adr_ids=["ADR-001"],
        ),
        ComponentDescription(
            id="COMP-002",
            name="Payment Service",
            purpose="Drain queued payments.",
            description="Implements FEAT-002 and is justified by ADR-001.",
            inputs=["OrderCreated events from the Shared Event Bus"],
            outputs=["PaymentCompleted events"],
            dependencies=["Shared Event Bus"],
            related_feature_ids=["FEAT-002"],
            related_adr_ids=["ADR-001"],
        ),
        ComponentDescription(
            id="COMP-003",
            name="Shared Event Bus",
            purpose="Backbone for publishing domain events.",
            description="Implements FEAT-003 and is justified by ADR-001.",
            inputs=["OrderCreated events from the Order Service"],
            outputs=["Domain events for subscribed consumers"],
            dependencies=[
                "Order Service",
                "Payment Service",
                "Notification Service",
            ],
            related_feature_ids=["FEAT-003"],
            related_adr_ids=["ADR-001"],
        ),
        ComponentDescription(
            id="COMP-004",
            name="Legacy Monolith",
            purpose="Serves the capabilities that are not extracted yet.",
            description=(
                "Implements FEAT-004 and FEAT-005 and is justified by "
                "ADR-002."
            ),
            related_feature_ids=["FEAT-004", "FEAT-005"],
            related_adr_ids=["ADR-002"],
        ),
    ]
    state.stage = Stage.DESIGNING
    return state


def _cart_stays_in_legacy(state):
    """A valid strangler phase: Cart is explicitly retained in legacy."""

    state.features[4].description = (
        "The Cart module is redirected to the new Cart Service in a later "
        "phase. The Cart module remains in the legacy monolith in the "
        "current phase."
    )
    return state


def _notification_is_external(state):
    """Notification is explicitly declared an external SaaS provider."""

    state.adrs[1].context = (
        "The remaining shop modules must integrate without new synchronous "
        "coupling. The Notification Service is an external SaaS provider "
        "operated by the messaging team."
    )
    # The flow endpoint carries the same external disposition explicitly —
    # an externally-provided system is an external participant in the
    # diagram, not an unresolved internal one.
    state.blueprint.data_flows[1] = (
        "Shared Event Bus → External Notification Service: order events "
        "for downstream consumers"
    )
    return state


def _inventory_is_owned(state):
    """Inventory is explicitly mapped to an existing component."""

    state.adrs[1].decision += (
        " The Inventory Service is owned by the Order Service during the "
        "migration."
    )
    return state


def _with_services_described(state):
    """Every referenced target service has a Component Description."""

    state.components.extend(
        [
            ComponentDescription(
                id="COMP-005",
                name="Cart Service",
                purpose="Serve cart traffic after extraction.",
                description=(
                    "Implements FEAT-005 and is justified by ADR-002."
                ),
                related_feature_ids=["FEAT-005"],
                related_adr_ids=["ADR-002"],
            ),
            ComponentDescription(
                id="COMP-006",
                name="Notification Service",
                purpose="Deliver order notifications.",
                description=(
                    "Implements FEAT-003 and is justified by ADR-001."
                ),
                related_feature_ids=["FEAT-003"],
                related_adr_ids=["ADR-001"],
            ),
            ComponentDescription(
                id="COMP-007",
                name="Inventory Service",
                purpose="Own inventory availability events.",
                description=(
                    "Implements FEAT-001 and is justified by ADR-002."
                ),
                related_feature_ids=["FEAT-001"],
                related_adr_ids=["ADR-002"],
            ),
        ]
    )
    state.blueprint.components.extend(
        ["Cart Service", "Notification Service", "Inventory Service"]
    )
    return state


def test_missing_cart_service_reference_is_blocking():
    checks = run_deterministic_checks(_dangling_services_state())

    assert checks.unowned_target_services["Cart Service"] == ["FEAT-005"]
    assert checks.score_traceability < 2
    assert any(
        issue.severity == "high"
        and "Cart Service" in issue.finding
        and "FEAT-005" in issue.evidence
        for issue in checks.issues
    )


def test_missing_notification_service_reference_is_blocking():
    checks = run_deterministic_checks(_dangling_services_state())

    assert checks.unowned_target_services["Notification Service"] == [
        "Blueprint.data_flows",
        "Shared Event Bus.dependencies",
    ]
    assert any(
        issue.severity == "high" and "Notification Service" in issue.finding
        for issue in checks.issues
    )


def test_target_capability_without_owner_is_blocking():
    checks = run_deterministic_checks(_dangling_services_state())

    assert checks.unowned_target_services["Inventory Service"] == ["ADR-002"]
    assert any(
        issue.severity == "high" and "Inventory Service" in issue.finding
        for issue in checks.issues
    )


def test_dangling_services_were_the_only_deterministic_failure():
    """The fixture is otherwise fully green — the shape of the false PASS.
    The dangling Notification Service is now caught TWICE: once by the
    ownership check (dangling reference) and once by the flow-participant
    check (unresolved directional endpoint) — the newer invariant sees it
    too, which is exactly the layered defence working."""

    checks = run_deterministic_checks(_dangling_services_state())

    assert checks.score_all_artifacts_present == 2
    assert checks.score_constraint_coverage == 2
    assert checks.score_adr_presence == 2
    assert checks.score_source_integrity == 2
    assert set(checks.unowned_target_services) == {
        "Cart Service", "Notification Service", "Inventory Service",
    }
    assert set(checks.unresolved_flow_participants) == {"Notification Service"}
    assert checks.unrenderable_data_flows == []
    assert len(checks.issues) == 4


def test_inconsistent_design_cannot_pass_review(monkeypatch):
    monkeypatch.setattr(rev, "llm_call", _as_llm_call(_all_pass_judgments))
    output = rev.reviewer_node(_dangling_services_state())
    report = output["review"]

    assert report.overall_status == "fail"
    assert report.requires_refinement is True
    assert output["stage"] is Stage.REFINING
    instruction = report.refinement_instruction
    for name in ("Cart Service", "Notification Service", "Inventory Service"):
        assert name in instruction
    # The finding names where each service was referenced...
    assert "FEAT-005" in instruction
    # ...and the fix states every acceptable resolution.
    cart_issue = next(
        issue for issue in report.issues if "Cart Service" in issue.finding
    )
    assert "Component Description" in cart_issue.suggested_fix
    assert "legacy monolith" in cart_issue.suggested_fix
    assert "ownership" in cart_issue.suggested_fix


def test_explicit_legacy_retention_is_not_flagged():
    state = _cart_stays_in_legacy(_dangling_services_state())
    checks = run_deterministic_checks(state)

    assert "Cart Service" not in checks.unowned_target_services


def test_explicit_external_provider_is_not_flagged():
    state = _notification_is_external(_dangling_services_state())
    checks = run_deterministic_checks(state)

    assert "Notification Service" not in checks.unowned_target_services


def test_ownership_by_existing_component_is_not_flagged():
    state = _inventory_is_owned(_dangling_services_state())
    checks = run_deterministic_checks(state)

    assert "Inventory Service" not in checks.unowned_target_services


def test_fully_disposed_strangler_design_passes_review(monkeypatch):
    state = _inventory_is_owned(
        _notification_is_external(_cart_stays_in_legacy(_dangling_services_state()))
    )
    checks = run_deterministic_checks(state)

    assert checks.unowned_target_services == {}
    assert checks.score_traceability == 2
    assert checks.issues == []

    monkeypatch.setattr(rev, "llm_call", _as_llm_call(_all_pass_judgments))
    output = rev.reviewer_node(state)

    assert output["review"].overall_status == "pass"
    assert output["stage"] is Stage.DONE


def test_services_with_component_descriptions_pass_review(monkeypatch):
    state = _with_services_described(_dangling_services_state())
    checks = run_deterministic_checks(state)

    assert checks.unowned_target_services == {}
    assert checks.issues == []

    monkeypatch.setattr(rev, "llm_call", _as_llm_call(_all_pass_judgments))
    output = rev.reviewer_node(state)

    assert output["review"].overall_status == "pass"
    assert output["stage"] is Stage.DONE


# --- generic plural "… services" must not invent named services ------------
#
# Regression for the real E2E run 20260822T103036Z-dc6e159d against
# ecommerce-monolith: ADR-002's rationale "Decouples services, allowing
# individual scaling to handle spikes." was parsed as a reference to a
# component named "Decouples Service", producing a false HIGH finding.
# Generic prose (a verb/adjective before the plural "services") is not a
# named reference; only an explicit ENUMERATION of TitleCase names is.


def test_generic_plural_services_produces_no_named_services():
    from pipeline.review_checks import _extract_service_bases

    for phrase in (
        "Decouples services, allowing individual scaling to handle spikes.",
        "Allows services to evolve independently.",
        "Services communicate asynchronously through the Event Bus.",
        "Runs on Amazon Web Services.",
    ):
        assert _extract_service_bases(phrase) == [], phrase


def test_enumerated_services_remain_detected():
    from pipeline.review_checks import _extract_service_bases

    bases = _extract_service_bases(
        "The Shared Event Bus fans out order events to the Order, "
        "Payment, and Notification services for downstream consumers."
    )
    assert set(bases) == {"Order", "Payment", "Notification"}

    # The two-name "and" form is an enumeration too.
    bases = _extract_service_bases(
        "Events flow to the Payment and Notification services."
    )
    assert set(bases) == {"Payment", "Notification"}


def test_real_adr002_rationale_yields_no_invented_service():
    """The exact real ADR-002 text, against the run's real component list:
    no 'Decouples Service' may be invented or flagged."""
    state = new_run(
        "Re-architect the ecommerce monolith to handle seasonal spikes."
    )
    state.context_record = ContextRecord(
        project_name="Ecommerce Monolith Modernization",
        business_goal="Survive seasonal peaks without outages.",
        summary="event-driven modernization of a Django monolith",
    )
    state.features = [
        Feature(
            id="FEAT-001",
            name="Decouple order processing",
            description="Orders are accepted asynchronously at peak.",
            scenario="A customer order is queued instead of blocking checkout.",
            acceptance_criteria=["Orders are persisted asynchronously."],
        ),
    ]
    state.blueprint = Blueprint(
        project_name="Ecommerce Monolith Modernization",
        selected_pattern="Event-driven services behind a central event bus",
        stakeholder_view="Peak-season shopping keeps working during migration.",
        technical_view="Services communicate through EventBridge.",
        components=["Event Bus", "Order Service"],
        addressed_feature_ids=["FEAT-001"],
    )
    state.adrs = [
        ADR(
            id="ADR-002",
            title="ADR-002: Event-Driven Architecture for Scalability",
            context="The monolith's synchronous coupling saturates at peak.",
            decision="Use Amazon EventBridge as the central Event Bus.",
            rationale=(
                "Decouples services, allowing individual scaling to handle "
                "spikes."
            ),
            alternatives_considered=["Point-to-point REST integration"],
            positive_consequences=["Independent scaling per module."],
            negative_consequences=["Eventual consistency."],
            related_component_names=["Event Bus", "Order Service"],
        )
    ]
    state.components = [
        ComponentDescription(
            id="COMP-001", name="Event Bus",
            purpose="Backbone for domain events.",
            description="Implements the event-driven pattern.",
            related_feature_ids=["FEAT-001"],
            related_adr_ids=["ADR-002"],
        ),
        ComponentDescription(
            id="COMP-002", name="Order Service",
            purpose="Accept orders asynchronously.",
            description="Handles order placement.",
            related_feature_ids=["FEAT-001"],
            related_adr_ids=["ADR-002"],
        ),
    ]

    checks = run_deterministic_checks(state)

    assert "Decouples Service" not in checks.unowned_target_services
    assert not any(
        "Decouples Service" in issue.finding for issue in checks.issues
    )


# --- component/reference canonical identity ---------------------------------
#
# Regression for the final run 20260822T222414Z-1788d214 against
# ecommerce-monolith: the run reached DONE only at the refinement cost cap,
# with four HIGH findings ("missing User/Product/Order-Payment/Inventory
# Service") that manual inspection proved were false positives. The final
# artifacts actually declared all four Component Descriptions, named "User
# Service (New)", "Product Service (New)", "Order/Payment Service (New)" and
# "Inventory Service (New)" — the trailing "(New)" display qualifier survived
# into the component's identity tokens while a bare reference's did not, so
# the two never matched. Separately, "Order/Payment Service" is ONE declared
# composite component; the un-widened extraction regex captured only its last
# slash-joined word ("Payment"), splitting it into a phantom "Payment
# Service" reference to itself.


def test_component_key_folds_new_qualifier_into_bare_identity():
    from pipeline.review_checks import _component_key

    assert _component_key("User Service (New)") == _component_key("User Service")
    assert _component_key("Product Service (New)") == _component_key(
        "Product Service"
    )
    assert _component_key("Inventory Service (New)") == _component_key(
        "Inventory Service"
    )


def test_component_key_folds_legacy_qualifier_into_bare_identity():
    from pipeline.review_checks import _component_key

    assert _component_key("Django Monolith (Legacy)") == _component_key(
        "Django Monolith"
    )


def test_composite_slash_name_extracted_as_one_base_not_split():
    """'Order/Payment Service' must yield ONE base ('Order/Payment'), never a
    phantom 'Payment'-only reference from splitting on the slash."""
    from pipeline.review_checks import _extract_service_bases

    bases = _extract_service_bases(
        "Checkout events are published to the Order/Payment Service."
    )
    assert bases == ["Order/Payment"]


def _lifecycle_qualified_state():
    """The final-run shape: four declared components carry a '(New)' display
    qualifier; every other artifact references them by their bare name."""
    state = new_run(
        "Re-architect the ecommerce monolith for seasonal peak traffic."
    )
    state.context_record = ContextRecord(
        project_name="Ecommerce Monolith Modernization",
        business_goal="Survive seasonal peaks without outages.",
        summary="brownfield strangler migration of a monolithic shop",
    )
    state.features = [
        Feature(
            id="FEAT-001",
            name="Checkout",
            description=(
                "A genuinely separate Payment Service must remain "
                "addressable as its own target service."
            ),
            scenario="A customer completes checkout during peak traffic.",
            acceptance_criteria=["Checkout succeeds under peak load."],
        ),
    ]
    state.blueprint = Blueprint(
        stakeholder_view="Peak-season shopping keeps working during migration.",
        technical_view=(
            "The User Service, Product Service, Order/Payment Service and "
            "Inventory Service replace the corresponding monolith modules."
        ),
        components=[
            "User Service (New)",
            "Product Service (New)",
            "Order/Payment Service (New)",
            "Inventory Service (New)",
        ],
        data_flows=[
            "The Order/Payment Service publishes CheckoutCompleted to the "
            "Inventory Service.",
        ],
        addressed_feature_ids=["FEAT-001"],
    )
    state.adrs = [
        ADR(
            id="ADR-001",
            title="ADR-1: Extract checkout-path services",
            context=(
                "The User Service and Product Service publish events "
                "consumed downstream."
            ),
            decision=(
                "The User Service and Product Service publish domain "
                "events to the Inventory Service."
            ),
            rationale=(
                "The Order Service and the Payment Service may eventually "
                "be split for independent scaling."
            ),
            alternatives_considered=["Scale the monolith vertically"],
            positive_consequences=["Extracted capabilities scale independently."],
            negative_consequences=["Eventual consistency during migration."],
            related_feature_ids=["FEAT-001"],
            related_component_names=[
                "User Service (New)", "Product Service (New)",
            ],
        ),
    ]
    state.components = [
        ComponentDescription(
            id="COMP-001",
            name="User Service (New)",
            purpose="Own user accounts after extraction.",
            description="Implements FEAT-001 and is justified by ADR-001.",
            related_feature_ids=["FEAT-001"],
            related_adr_ids=["ADR-001"],
        ),
        ComponentDescription(
            id="COMP-002",
            name="Product Service (New)",
            purpose="Own product catalog after extraction.",
            description="Implements FEAT-001 and is justified by ADR-001.",
            related_feature_ids=["FEAT-001"],
            related_adr_ids=["ADR-001"],
        ),
        ComponentDescription(
            id="COMP-003",
            name="Order/Payment Service (New)",
            purpose="Own order placement and payment capture together.",
            description="Implements FEAT-001 and is justified by ADR-001.",
            related_feature_ids=["FEAT-001"],
            related_adr_ids=["ADR-001"],
        ),
        ComponentDescription(
            id="COMP-004",
            name="Inventory Service (New)",
            purpose="Own inventory availability after extraction.",
            description="Implements FEAT-001 and is justified by ADR-001.",
            dependencies=["Order/Payment Service"],
            related_feature_ids=["FEAT-001"],
            related_adr_ids=["ADR-001"],
        ),
    ]
    state.stage = Stage.DESIGNING
    return state


def test_bare_reference_to_new_qualified_component_is_not_missing():
    """Items 1-3: 'User/Product/Inventory Service (New)' each satisfy the
    bare-name reference the rest of the design actually uses."""
    checks = run_deterministic_checks(_lifecycle_qualified_state())

    for name in ("User Service", "Product Service", "Inventory Service"):
        assert name not in checks.unowned_target_services, name


def test_declared_composite_satisfies_its_own_bare_reference():
    """Item 5: the composite component satisfies a bare reference to its
    own full name, arriving via a Component dependency list (COMP-004
    depends on "Order/Payment Service" — one of the cross-artifact forms
    item 8 requires)."""
    checks = run_deterministic_checks(_lifecycle_qualified_state())

    assert "Order/Payment Service" not in checks.unowned_target_services


def test_composite_component_does_not_satisfy_a_split_half_reference():
    """Item 6: declaring the composite 'Order/Payment Service (New)' must
    NOT imply that separately-referenced 'Order Service' and 'Payment
    Service' also exist — both stay reportable, exactly as if no composite
    had been declared at all."""
    checks = run_deterministic_checks(_lifecycle_qualified_state())

    assert "Order Service" in checks.unowned_target_services
    assert "Payment Service" in checks.unowned_target_services


def test_separate_payment_service_reference_still_fails_against_composite_only():
    """Item 7: with only the composite 'Order/Payment Service (New)'
    declared, an explicit standalone 'Payment Service' reference is a real
    gap and must still be reported, HIGH and blocking."""
    checks = run_deterministic_checks(_lifecycle_qualified_state())

    assert "Payment Service" in checks.unowned_target_services
    assert any(
        issue.severity == "high" and "Payment Service" in issue.finding
        for issue in checks.issues
    )


def test_final_run_shape_produces_zero_declared_component_findings():
    """The fix must make the final-run-style artifact set produce ZERO
    missing-Component-Description findings for the four defined services —
    the only remaining unowned references are the genuinely separate
    'Order Service' / 'Payment Service' (item 6/7), never one of the four
    declared components."""
    checks = run_deterministic_checks(_lifecycle_qualified_state())

    assert set(checks.unowned_target_services) == {"Order Service", "Payment Service"}


def test_cross_artifact_reference_forms_share_one_canonical_identity():
    """Item 8: Blueprint data flows (Inventory), Component dependencies
    (Order/Payment, via COMP-004), and ADR related fields (User, Product)
    all resolved through the same identity contract — none of the four
    declared components is reported missing regardless of which artifact
    referenced it."""
    checks = run_deterministic_checks(_lifecycle_qualified_state())

    for name in (
        "User Service", "Product Service", "Order/Payment Service",
        "Inventory Service",
    ):
        assert name not in checks.unowned_target_services, name


def test_dangling_services_regression_still_flags_genuinely_missing_services():
    """Item 9: the pre-existing false-negative regression fixture is
    unaffected by this fix — genuinely undeclared target services are still
    reported exactly as before."""
    checks = run_deterministic_checks(_dangling_services_state())

    assert set(checks.unowned_target_services) == {
        "Cart Service", "Notification Service", "Inventory Service",
    }


# --- canonical identity reaches ADR/data-flow structured checks -----------
#
# Manual review of the previous E2E-blocker repair found `_component_key`
# wired into target-service ownership only — `_check_traceability`'s
# `invalid_adr_component_names` and `_check_flow_participants` still compared
# RAW display strings, so a bare ADR/flow reference to a component declared
# with a lifecycle qualifier ("User Service (New)") still failed. These
# fixtures deliberately keep the REFERENCE bare and the COMPONENT qualified —
# never the reverse — so passing actually proves canonical identity, not
# coincidental exact-string equality.


def _qualified_component_state():
    state = new_run("Re-architect the ecommerce monolith.")
    state.context_record = ContextRecord(
        project_name="Ecommerce Monolith Modernization",
        business_goal="Survive seasonal peaks without outages.",
        summary="brownfield strangler migration of a monolithic shop",
    )
    state.features = [
        Feature(
            id="FEAT-001",
            name="User accounts",
            scenario="A customer manages their account during checkout.",
            acceptance_criteria=["Account changes persist."],
        ),
    ]
    state.blueprint = Blueprint(
        stakeholder_view="Peak-season shopping keeps working during migration.",
        technical_view=(
            "The User Service replaces the legacy monolith's user module."
        ),
        components=[
            "User Service (New)",
            "Django Monolith (Legacy)",
            "Inventory Service (New)",
            "Order/Payment Service (New)",
        ],
        data_flows=[
            # Bare references to qualified components — must resolve.
            "User Service -> Inventory Service: reserve stock",
            "Order/Payment Service -> Inventory Service: capture payment",
            # A genuinely unknown endpoint — must still fail.
            "Inventory Service -> Ghost Queue: audit log",
        ],
        addressed_feature_ids=["FEAT-001"],
    )
    state.adrs = [
        ADR(
            id="ADR-001",
            title="ADR-1: Extract user accounts",
            context="The monolith's user module is a scaling bottleneck.",
            decision="Extract user accounts into a standalone service.",
            rationale="Independent scaling for account traffic.",
            alternatives_considered=["Scale the monolith vertically"],
            positive_consequences=["Independent scaling"],
            negative_consequences=["Additional operational complexity"],
            related_feature_ids=["FEAT-001"],
            # Bare references to qualified/composite components, PLUS one
            # genuinely unknown name that must still fail.
            related_component_names=[
                "User Service", "Django Monolith", "Order/Payment Service",
                "Ghost Service",
            ],
        ),
    ]
    state.components = [
        ComponentDescription(
            id="COMP-001",
            name="User Service (New)",
            purpose="Own user accounts after extraction.",
            description="Implements FEAT-001 and is justified by ADR-001.",
            related_feature_ids=["FEAT-001"],
            related_adr_ids=["ADR-001"],
        ),
        ComponentDescription(
            id="COMP-002",
            name="Django Monolith (Legacy)",
            purpose="Serves capabilities not yet extracted.",
            description="Implements FEAT-001 and is justified by ADR-001.",
            related_feature_ids=["FEAT-001"],
            related_adr_ids=["ADR-001"],
        ),
        ComponentDescription(
            id="COMP-003",
            name="Inventory Service (New)",
            purpose="Own inventory availability after extraction.",
            description="Implements FEAT-001 and is justified by ADR-001.",
            related_feature_ids=["FEAT-001"],
            related_adr_ids=["ADR-001"],
        ),
        ComponentDescription(
            id="COMP-004",
            name="Order/Payment Service (New)",
            purpose="Own order placement and payment capture together.",
            description="Implements FEAT-001 and is justified by ADR-001.",
            related_feature_ids=["FEAT-001"],
            related_adr_ids=["ADR-001"],
        ),
    ]
    state.stage = Stage.DESIGNING
    return state


def test_bare_adr_reference_resolves_to_qualified_component():
    """Item 1: ADR bare 'User Service' resolves to Component 'User Service
    (New)'."""
    checks = run_deterministic_checks(_qualified_component_state())

    assert "User Service" not in checks.invalid_adr_component_names.get(
        "ADR-001", []
    )


def test_bare_adr_reference_resolves_to_qualified_legacy_component():
    """Item 2: ADR bare 'Django Monolith' resolves to 'Django Monolith
    (Legacy)'."""
    checks = run_deterministic_checks(_qualified_component_state())

    assert "Django Monolith" not in checks.invalid_adr_component_names.get(
        "ADR-001", []
    )


def test_unknown_adr_component_name_still_fails():
    """Item 3: a genuinely unknown ADR component name is still reported —
    canonical identity resolves real names, it does not stop rejecting
    fabricated ones."""
    checks = run_deterministic_checks(_qualified_component_state())

    assert checks.invalid_adr_component_names["ADR-001"] == ["Ghost Service"]


def test_bare_flow_endpoint_resolves_to_qualified_component():
    """Item 4: arrow-flow endpoint 'Inventory Service' resolves to
    'Inventory Service (New)'."""
    checks = run_deterministic_checks(_qualified_component_state())

    assert "Inventory Service" not in checks.unresolved_flow_participants


def test_unknown_flow_endpoint_still_fails():
    """Item 5: a genuinely unknown flow endpoint still produces the HIGH
    unresolved-participant finding."""
    checks = run_deterministic_checks(_qualified_component_state())

    assert "Ghost Queue" in checks.unresolved_flow_participants
    assert any(
        issue.severity == "high" and "Ghost Queue" in issue.finding
        for issue in checks.issues
    )


def test_composite_protection_reaches_adr_and_flow_resolution_too():
    """Item 6: the composite-name protection from the previous fix is not
    limited to target-service ownership — the SAME canonical identity
    resolves the composite 'Order/Payment Service' bare reference against
    the declared 'Order/Payment Service (New)' in both the ADR check and the
    flow-participant check."""
    checks = run_deterministic_checks(_qualified_component_state())

    assert "Order/Payment Service" not in checks.invalid_adr_component_names.get(
        "ADR-001", []
    )
    assert "Order/Payment Service" not in checks.unresolved_flow_participants


# --- deterministic requirement catalog --------------------------------------
#
# The Architect's actual contract for `related_requirement_ids` is IDs
# ("NFR-001"), not raw requirement prose — existing Architect tests already
# used that shape. `requirement_catalog` is the ONE deterministic ID scheme
# both the Architect prompt/validation and this Reviewer check read, so the
# two sides can never silently disagree on what counts as "covered".


def test_functional_requirements_get_deterministic_fr_ids():
    """Item 1."""
    catalog = requirement_catalog(
        ContextRecord(
            project_name="P", business_goal="g",
            functional_requirements=["Customer registration and login", "Product reviews"],
        )
    )
    assert [(e.id, e.text, e.kind) for e in catalog] == [
        ("FR-001", "Customer registration and login", "functional"),
        ("FR-002", "Product reviews", "functional"),
    ]


def test_non_functional_requirements_get_deterministic_nfr_ids():
    """Item 2."""
    catalog = requirement_catalog(
        ContextRecord(
            project_name="P", business_goal="g",
            non_functional_requirements=[
                "Maintain responsive customer-facing interactions",
                "Handle 50k concurrent users",
            ],
        )
    )
    assert [(e.id, e.text, e.kind) for e in catalog] == [
        (
            "NFR-001", "Maintain responsive customer-facing interactions",
            "non_functional",
        ),
        ("NFR-002", "Handle 50k concurrent users", "non_functional"),
    ]


def test_catalog_ordering_is_stable_and_preserves_context_record_order():
    """Item 3: FR and NFR numbering are independent per-kind sequences, and
    calling twice on an unchanged record produces the identical catalog."""
    context = ContextRecord(
        project_name="P", business_goal="g",
        functional_requirements=["B capability", "A capability"],
        non_functional_requirements=["Z quality", "Y quality"],
    )

    first = requirement_catalog(context)
    second = requirement_catalog(context)

    assert [e.id for e in first] == ["FR-001", "FR-002", "NFR-001", "NFR-002"]
    assert [e.text for e in first] == [
        "B capability", "A capability", "Z quality", "Y quality",
    ]
    assert first == second


def test_cloud_budget_compliance_existing_system_get_no_catalog_entry():
    """Item 4."""
    catalog = requirement_catalog(
        ContextRecord(
            project_name="P", business_goal="g",
            cloud_provider="AWS", budget="medium",
            compliance_requirements=["GDPR"],
            existing_systems=["Legacy monolith"],
        )
    )
    assert catalog == []


def test_duplicate_requirement_text_collapses_to_one_canonical_entry():
    """Documented duplicate-handling: an identical requirement string
    appearing twice in the record gets ONE canonical entry, never two
    ambiguous IDs for the same requirement."""
    catalog = requirement_catalog(
        ContextRecord(
            project_name="P", business_goal="g",
            functional_requirements=["Product reviews", "Product reviews"],
        )
    )
    assert [(e.id, e.text) for e in catalog] == [("FR-001", "Product reviews")]


# --- Context requirement -> Feature structured coverage --------------------
#
# `_check_constraints` proves the DESIGN addresses a requirement via loose
# token overlap; that is too loose to prove a requirement was ever MODELED —
# "Product reviews" can pass on the incidental word "Product" turning up
# elsewhere while reviews were never built as a feature. This invariant is
# structural: every explicit functional/non-functional requirement must be
# covered, by canonical catalog ID or (legacy compatibility) by its exact
# text, in some Feature's `related_requirement_ids`.


def test_all_functional_requirements_mapped_produces_no_finding():
    """Item 7."""
    state = new_run("Modernize a monolith.")
    state.context_record = ContextRecord(
        project_name="P", business_goal="g",
        functional_requirements=["Customers complete purchases", "Product reviews"],
    )
    state.features = [
        Feature(
            id="FEAT-001", name="Checkout", scenario="s",
            acceptance_criteria=["a"],
            related_requirement_ids=["Customers complete purchases"],
        ),
        Feature(
            id="FEAT-002", name="Reviews", scenario="s",
            acceptance_criteria=["a"],
            related_requirement_ids=["Product reviews"],
        ),
    ]

    checks = run_deterministic_checks(state)

    assert checks.requirements_without_feature == []
    assert not any(
        "related_requirement_ids" in issue.finding for issue in checks.issues
    )


def test_all_non_functional_requirements_mapped_produces_no_finding():
    """Item 8."""
    state = new_run("Modernize a monolith.")
    state.context_record = ContextRecord(
        project_name="P", business_goal="g",
        non_functional_requirements=[
            "Handle 50k concurrent users",
            "Maintain responsive customer-facing interactions",
        ],
    )
    state.features = [
        Feature(
            id="FEAT-001", name="Scale checkout", scenario="s",
            acceptance_criteria=["a"],
            related_requirement_ids=[
                "Handle 50k concurrent users",
                "Maintain responsive customer-facing interactions",
            ],
        ),
    ]

    checks = run_deterministic_checks(state)

    assert checks.requirements_without_feature == []
    assert not any(
        "related_requirement_ids" in issue.finding for issue in checks.issues
    )


def test_missing_product_reviews_requirement_is_blocking():
    """Item 9."""
    state = new_run("Modernize a monolith.")
    state.context_record = ContextRecord(
        project_name="P", business_goal="g",
        functional_requirements=["Customers complete purchases", "Product reviews"],
    )
    state.features = [
        Feature(
            id="FEAT-001", name="Checkout", scenario="s",
            acceptance_criteria=["a"],
            related_requirement_ids=["Customers complete purchases"],
        ),
    ]

    checks = run_deterministic_checks(state)

    assert checks.requirements_without_feature == ["FR-002 - Product reviews"]
    issue = next(
        i for i in checks.issues if "related_requirement_ids" in i.finding
    )
    assert issue.severity == "high"
    assert issue.requires_refinement is True
    assert "FR-002" in issue.finding
    assert "Product reviews" in issue.finding


def test_missing_responsive_interactions_nfr_is_blocking():
    """Item 10."""
    state = new_run("Modernize a monolith.")
    state.context_record = ContextRecord(
        project_name="P", business_goal="g",
        non_functional_requirements=[
            "Maintain responsive customer-facing interactions",
        ],
    )
    state.features = [
        Feature(id="FEAT-001", name="Checkout", scenario="s", acceptance_criteria=["a"]),
    ]

    checks = run_deterministic_checks(state)

    assert checks.requirements_without_feature == [
        "NFR-001 - Maintain responsive customer-facing interactions"
    ]
    issue = next(
        i for i in checks.issues if "related_requirement_ids" in i.finding
    )
    assert issue.severity == "high"
    assert issue.requires_refinement is True
    assert "NFR-001" in issue.finding
    assert "Maintain responsive customer-facing interactions" in issue.finding


def test_incidental_word_match_does_not_satisfy_structured_coverage():
    """Item 11: 'Product' turning up all over the design text must not
    satisfy the STRUCTURED invariant for the requirement 'Product reviews' —
    only an exact `related_requirement_ids` entry does. (The loose
    constraint-text matcher elsewhere IS satisfied by this fixture, which is
    exactly the false-coverage gap this check exists to close.)"""
    state = new_run("Modernize a monolith.")
    state.context_record = ContextRecord(
        project_name="P", business_goal="g",
        functional_requirements=["Product reviews"],
    )
    state.features = [
        Feature(
            id="FEAT-001", name="Product catalog", scenario="s",
            acceptance_criteria=["a"],
            description="Manages the Product catalog end to end.",
        ),
    ]
    state.blueprint = Blueprint(
        stakeholder_view="Product browsing stays fast.",
        technical_view="A Product Service serves the Product catalog.",
        components=["Product Service"],
        addressed_feature_ids=["FEAT-001"],
    )

    checks = run_deterministic_checks(state)

    assert checks.requirements_without_feature == ["FR-001 - Product reviews"]
    # The loose token-overlap check DOES false-pass here — the reason this
    # structural invariant is necessary, not redundant with it.
    assert checks.constraints_covered.get("functional") is True


def test_cloud_budget_compliance_fields_are_not_required_as_feature_mappings():
    """Item 12: design constraints, not features — never required in any
    Feature's `related_requirement_ids`."""
    state = new_run("Modernize a monolith.")
    state.context_record = ContextRecord(
        project_name="P", business_goal="g",
        cloud_provider="AWS", budget="medium",
        compliance_requirements=["GDPR"],
        existing_systems=["Legacy monolith"],
    )
    state.features = [
        Feature(id="FEAT-001", name="Checkout", scenario="s", acceptance_criteria=["a"]),
    ]

    checks = run_deterministic_checks(state)

    assert checks.requirements_without_feature == []


def test_reviewer_does_not_invent_or_mutate_requirement_mappings():
    """Item 13: the Reviewer reports the gap and never writes a mapping."""
    state = new_run("Modernize a monolith.")
    state.context_record = ContextRecord(
        project_name="P", business_goal="g",
        functional_requirements=["Product reviews"],
    )
    state.features = [
        Feature(id="FEAT-001", name="Checkout", scenario="s", acceptance_criteria=["a"]),
    ]
    before = [feature.model_copy(deep=True) for feature in state.features]

    checks = run_deterministic_checks(state)

    assert checks.requirements_without_feature == ["FR-001 - Product reviews"]
    assert state.features == before
    assert state.features[0].related_requirement_ids == []


# --- Reviewer aligned with the catalog contract -----------------------------
#
# Manual review of the requirement->Feature micro-fix found the Reviewer's
# raw-text-only rule was incompatible with the Architect's actual contract
# (IDs, not prose) and, worse, unrepairable: refinement reuses `state.features`
# verbatim and phase 2 cannot edit Feature objects, so a HIGH finding for a
# missing mapping could never be closed by the refine loop. These tests pin
# the Reviewer's half of the fix: catalog-ID coverage, legacy raw-text
# compatibility, genuine gaps still caught, and the score/diagnostic
# consequences of a gap that does still reach it (persisted/legacy states).


def test_reviewer_accepts_canonical_catalog_ids():
    """Item 15."""
    state = new_run("Modernize a monolith.")
    state.context_record = ContextRecord(
        project_name="P", business_goal="g",
        functional_requirements=["Customer registration and login"],
        non_functional_requirements=["Maintain responsive customer-facing interactions"],
    )
    state.features = [
        Feature(
            id="FEAT-001", name="Accounts", scenario="s", acceptance_criteria=["a"],
            related_requirement_ids=["FR-001", "NFR-001"],
        ),
    ]

    checks = run_deterministic_checks(state)

    assert checks.requirements_without_feature == []


def test_reviewer_still_accepts_exact_legacy_raw_requirement_text():
    """Item 16: persisted/legacy/directly-built states written before the
    catalog contract existed must not regress."""
    state = new_run("Modernize a monolith.")
    state.context_record = ContextRecord(
        project_name="P", business_goal="g",
        functional_requirements=["Customer registration and login"],
    )
    state.features = [
        Feature(
            id="FEAT-001", name="Accounts", scenario="s", acceptance_criteria=["a"],
            related_requirement_ids=["Customer registration and login"],
        ),
    ]

    checks = run_deterministic_checks(state)

    assert checks.requirements_without_feature == []


def test_reviewer_still_rejects_a_genuinely_missing_mapping():
    """Item 17."""
    state = new_run("Modernize a monolith.")
    state.context_record = ContextRecord(
        project_name="P", business_goal="g",
        functional_requirements=["Customer registration and login"],
    )
    state.features = [
        Feature(id="FEAT-001", name="Accounts", scenario="s", acceptance_criteria=["a"]),
    ]

    checks = run_deterministic_checks(state)

    assert checks.requirements_without_feature == [
        "FR-001 - Customer registration and login"
    ]


def test_missing_mapping_lowers_deterministic_traceability_below_full_score():
    """Item 18: closes the score/finding inconsistency — a HIGH traceability
    finding for a missing requirement mapping can no longer coexist with a
    2/2 deterministic traceability score."""
    state = new_run("Modernize a monolith.")
    state.context_record = ContextRecord(
        project_name="P", business_goal="g",
        functional_requirements=["Customer registration and login"],
    )
    state.features = [
        Feature(id="FEAT-001", name="Accounts", scenario="s", acceptance_criteria=["a"]),
    ]
    state.blueprint = Blueprint(
        stakeholder_view="sv", technical_view="tv",
        components=["Accounts Service"], addressed_feature_ids=["FEAT-001"],
    )
    state.adrs = [
        ADR(
            id="ADR-001", title="ADR-1: Use a standalone accounts service",
            context="c", decision="d", rationale="r",
            related_feature_ids=["FEAT-001"],
            related_component_names=["Accounts Service"],
        ),
    ]
    state.components = [
        ComponentDescription(
            id="COMP-001", name="Accounts Service", purpose="p", description="d",
            related_feature_ids=["FEAT-001"], related_adr_ids=["ADR-001"],
        ),
    ]

    checks = run_deterministic_checks(state)

    assert checks.requirements_without_feature
    assert any(
        issue.severity == "high" and "related_requirement_ids" in issue.finding
        for issue in checks.issues
    )
    assert checks.score_traceability < 2


def test_missing_mapping_diagnostic_shows_id_and_exact_text():
    """Item 19."""
    state = new_run("Modernize a monolith.")
    state.context_record = ContextRecord(
        project_name="P", business_goal="g",
        functional_requirements=["Product reviews"],
    )
    state.features = [
        Feature(id="FEAT-001", name="Checkout", scenario="s", acceptance_criteria=["a"]),
    ]

    checks = run_deterministic_checks(state)

    assert checks.requirements_without_feature == ["FR-001 - Product reviews"]
    issue = next(
        i for i in checks.issues if "related_requirement_ids" in i.finding
    )
    assert "FR-001" in issue.finding
    assert "Product reviews" in issue.finding


def test_reviewer_check_never_mutates_feature_mappings():
    """Item 20: pins the check function itself (not only the higher-level
    behaviour already pinned above) never writes a mapping, canonical or
    otherwise, onto any Feature it inspects."""
    state = new_run("Modernize a monolith.")
    state.context_record = ContextRecord(
        project_name="P", business_goal="g",
        functional_requirements=["Customer registration and login"],
    )
    state.features = [
        Feature(id="FEAT-001", name="Accounts", scenario="s", acceptance_criteria=["a"]),
    ]
    before = [feature.model_copy(deep=True) for feature in state.features]

    run_deterministic_checks(state)

    assert state.features == before


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
    # The fix rides along as far as the assembler's per-field clip allows
    # (the enriched fix leads with the uncovered strings, so what survives
    # clipping is the actionable head, not a truncated tail).
    from pipeline.agents.reviewer import _clip as reviewer_clip

    assert reviewer_clip(constraint_issue.suggested_fix, 240) in instruction


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


# --- the verdict rule ------------------------------------------------------


def _rubric(**overrides):
    """A rubric that passes everything, minus whatever the caller breaks."""

    base = dict(
        all_artifacts_present=2,
        constraint_coverage=2,
        traceability=2,
        adr_presence=2,
        repo_grounding=True,
        flaw_detection=True,
        adr_soundness=True,
        best_practice_grounding=True,
        refinement_readiness=True,
    )
    base.update(overrides)
    return RubricScores(**base)


def test_derive_verdict_passes_a_clean_report():
    assert rev.derive_verdict(_rubric(), []) == (True, [])


def test_refinement_readiness_alone_no_longer_blocks():
    """Run 20260818T194159Z-107ff26e: it answered NO because a deterministic
    check had failed, then raised its own issue about that same failure. As the
    fifth term of an AND it could veto a design nothing else objected to."""

    passed, blocking = rev.derive_verdict(_rubric(refinement_readiness=False), [])
    assert passed is True
    assert "refinement_readiness" not in blocking


def test_a_high_severity_issue_still_blocks():
    passed, blocking = rev.derive_verdict(
        _rubric(), [_issue("high", "Structural problem.", "Fix it.")]
    )
    assert passed is False
    assert blocking == ["high_severity_issue"]


def test_a_medium_issue_does_not_block():
    passed, _ = rev.derive_verdict(_rubric(), [_issue("medium", "Minor.", "Fix.")])
    assert passed is True


def test_every_code_score_must_still_be_full():
    for field in ("all_artifacts_present", "constraint_coverage",
                  "traceability", "adr_presence"):
        passed, blocking = rev.derive_verdict(_rubric(**{field: 1}), [])
        assert passed is False, field
        assert blocking == [field]


def test_the_three_remaining_judgments_still_block():
    for field in ("repo_grounding", "flaw_detection", "adr_soundness"):
        passed, blocking = rev.derive_verdict(_rubric(**{field: False}), [])
        assert passed is False, field
        assert blocking == [field]


def test_a_not_applicable_criterion_is_excluded_from_the_verdict():
    passed, blocking = rev.derive_verdict(
        _rubric(best_practice_grounding=False),
        [],
        not_applicable=["best_practice_grounding"],
    )
    assert passed is True
    assert "best_practice_grounding" not in blocking


def test_derive_verdict_is_pure_and_order_stable():
    rubric = _rubric(traceability=0, flaw_detection=False)
    issues = [_issue("high", "F.", "X.")]
    assert rev.derive_verdict(rubric, issues) == rev.derive_verdict(rubric, issues)
    assert rev.derive_verdict(rubric, issues)[1] == [
        "traceability", "flaw_detection", "high_severity_issue",
    ]


# --- criterion applicability ----------------------------------------------


def _judgments_failing(name: str, reason: str):
    judgments = _all_pass_judgments()
    setattr(judgments, name, _judgment(False, reason))
    return lambda *_a, **_k: judgments


def _no_knowledge_state():
    state = _good_design_state()
    state.retrieved_knowledge = []
    for adr in state.adrs:
        adr.source_references = []
    return state


def test_refinement_readiness_is_recorded_but_raises_no_issue():
    rev.llm_call = _as_llm_call(
        _judgments_failing("refinement_readiness", "Circular: another check failed.")
    )
    report = rev.reviewer_node(_good_design_state())["review"]

    # Still asked, still visible - a judgment that disagrees with its own
    # instruction is evidence about the judge.
    assert report.rubric_scores.refinement_readiness is False
    assert report.judgment_reasons.refinement_readiness
    # But it no longer generates a finding, nor vetoes the design.
    assert not [i for i in report.issues if "actionable enough" in i.finding]
    assert report.overall_status == "pass"


def test_empty_knowledge_makes_best_practice_grounding_not_applicable():
    rev.llm_call = _as_llm_call(
        _judgments_failing("best_practice_grounding", "Nothing cited.")
    )
    report = rev.reviewer_node(_no_knowledge_state())["review"]

    assert report.not_applicable == ["repo_grounding", "best_practice_grounding"]
    assert not [i for i in report.issues if i.category == "evidence"]
    assert report.overall_status == "pass"
    # True ONLY so downstream readers keep working; not_applicable is the truth,
    # and every renderer consults it first.
    assert report.rubric_scores.best_practice_grounding is True


def test_retrieved_knowledge_restores_the_original_behaviour():
    rev.llm_call = _as_llm_call(
        _judgments_failing("best_practice_grounding", "Nothing cited.")
    )
    report = rev.reviewer_node(_good_design_state())["review"]  # HAS knowledge

    assert report.not_applicable == ["repo_grounding"]
    assert report.rubric_scores.best_practice_grounding is False
    assert report.overall_status == "fail"
    assert [i for i in report.issues if i.category == "evidence"]


# --- the reviewer must not read the architect's own account of its changes ---


def test_revision_note_never_reaches_the_review_prompt():
    """The architect writes `revision_note` on a refine pass to say what it
    changed and why. Feeding that to the judge grading the result would let the
    design argue its own case - a model that states it addressed a finding tends
    to be believed, and this reviewer exists to CHECK the artifacts rather than
    take them at their word. The note is persisted and shown to a human; it is
    simply not evidence.
    """

    state = _good_design_state()
    state.blueprint.revision_note = (
        "MAGIC_SELF_REPORT: added FEAT-005 to the Order Worker as instructed."
    )

    artifacts = rev._format_artifacts(state)
    prompt = rev._build_prompt(state, run_deterministic_checks(state))

    assert "MAGIC_SELF_REPORT" not in artifacts
    assert "MAGIC_SELF_REPORT" not in prompt
    assert "revision_note" not in artifacts
    # The rest of the blueprint is still there - only that one field is dropped.
    assert state.blueprint.stakeholder_view in artifacts
    assert state.blueprint.technical_view in artifacts
    assert "blueprint_id" in artifacts
