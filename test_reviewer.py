"""Offline unit tests for the Reviewer and generalized rubric v3."""
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
                "Order Service publishes OrderCreated to the Shared Event "
                "Bus, which fans out to the Payment Service and the "
                "Notification Service."
            ),
            (
                "The Shared Event Bus fans out order events to the Order, "
                "Payment, and Notification services for downstream "
                "consumers."
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
    """The fixture is otherwise fully green — the shape of the false PASS."""

    checks = run_deterministic_checks(_dangling_services_state())

    assert checks.score_all_artifacts_present == 2
    assert checks.score_constraint_coverage == 2
    assert checks.score_adr_presence == 2
    assert checks.score_source_integrity == 2
    assert set(checks.unowned_target_services) == {
        "Cart Service", "Notification Service", "Inventory Service",
    }
    assert len(checks.issues) == 3


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
