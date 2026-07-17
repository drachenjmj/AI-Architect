"""Labeled use-case #1 scenarios for Reviewer alignment evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pipeline.state import (
    ADR,
    ArchitectState,
    Blueprint,
    ComponentDescription,
    ContextRecord,
    Feature,
    KBChunk,
    RepoBehavior,
    RepoMeta,
    RepoRepresentation,
    RepoStructure,
    Stage,
    new_run,
)


UC1_PROMPT = (
    "Fix our monolithic online shop so it can scale for seasonal peak sales. "
    "It is on AWS, budget is medium, must stay GDPR-compliant, and needs to "
    "handle 50k concurrent users at peak."
)


@dataclass(frozen=True)
class EvalScenario:
    """One labeled architecture state used to measure Reviewer agreement."""

    name: str
    expected_pass: bool
    build_state: Callable[[], ArchitectState]


def _base_state() -> ArchitectState:
    state = new_run(
        UC1_PROMPT,
        repo_url="https://github.com/example/bugged-shop",
    )
    state.context_record = ContextRecord(
        project_name="Seasonal Shop",
        problem_statement="The legacy monolith crashes during seasonal peaks.",
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
    state.repo_representation = RepoRepresentation(
        meta=RepoMeta(
            url="https://github.com/example/bugged-shop",
            commit_sha="evaluation-fixture",
        ),
        structure=RepoStructure(
            file_tree="shop/\n  app.py\n  checkout.py\n  orders.py",
            repo_map=(
                "app.py: ShopApp serves catalog, checkout, and orders in one "
                "process; checkout.py calls orders.py synchronously"
            ),
        ),
        behavior=RepoBehavior(
            overview=(
                "One deployable serves browsing, checkout, and synchronous "
                "order processing with in-process sessions."
            )
        ),
    )
    state.retrieved_knowledge = [
        KBChunk(
            content="Use a queue to buffer bursts at asynchronous boundaries.",
            source="architecture_patterns.md",
        ),
        KBChunk(
            content="Stateless web tiers can scale horizontally.",
            source="microservices-on-aws.pdf",
        ),
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
            scenario="EU order data is encrypted and remains in the EU.",
        ),
    ]
    state.stage = Stage.DESIGNING
    return state


def seeded_flaw_state() -> ArchitectState:
    """A formally complete design that patches the monolith instead of fixing it."""

    state = _base_state()
    state.blueprint = Blueprint(
        stakeholder_view="Customers use the same shop on larger servers.",
        technical_view=(
            "Keep the legacy monolith on AWS as one deployable. Vertically "
            "scale its EC2 instance for the 50k-user seasonal peak and restart "
            "it nightly. Retain the medium budget by avoiding migration. Keep "
            "order data encrypted in the EU for GDPR compliance."
        ),
        components=["Shop Monolith", "Order Module"],
        addressed_feature_ids=["FEAT-001", "FEAT-002"],
        constraints_addressed=[
            "AWS cloud",
            "medium budget",
            "50k peak load",
            "GDPR compliance",
            "existing legacy monolith",
        ],
    )
    state.adrs = [
        ADR(
            id="ADR-001",
            title="ADR-1: Vertically scale the Shop Monolith",
            context="The legacy monolith slows during peak load.",
            decision="Move the Shop Monolith to a larger AWS EC2 instance.",
            rationale="This avoids migration and fits the medium budget.",
            alternatives_considered=["Restart the existing instance more often"],
            positive_consequences=["No application changes"],
            negative_consequences=["A single deployable remains"],
            related_feature_ids=["FEAT-001"],
            related_component_names=["Shop Monolith"],
            source_references=["architecture_patterns.md"],
        ),
        ADR(
            id="ADR-002",
            title="ADR-2: Encrypt the existing Order Module database",
            context="Order data must remain GDPR compliant.",
            decision="Keep the Order Module and encrypt its EU database.",
            rationale="Encryption supports GDPR without changing the monolith.",
            alternatives_considered=["Move order data outside the EU"],
            positive_consequences=["EU data residency"],
            negative_consequences=["Order processing remains coupled"],
            related_feature_ids=["FEAT-002"],
            related_component_names=["Order Module"],
            source_references=["architecture_patterns.md"],
        ),
    ]
    state.components = [
        ComponentDescription(
            id="COMP-001",
            name="Shop Monolith",
            purpose="Serve catalog, checkout, and orders in one process.",
            description="Implements FEAT-001 and is justified by ADR-001.",
            related_feature_ids=["FEAT-001"],
            related_adr_ids=["ADR-001"],
            technology_choices=["AWS EC2"],
            scalability_considerations=["Vertical scaling for peak load"],
        ),
        ComponentDescription(
            id="COMP-002",
            name="Order Module",
            purpose="Process orders inside the monolith.",
            description="Implements FEAT-002 and is justified by ADR-002.",
            related_feature_ids=["FEAT-002"],
            related_adr_ids=["ADR-002"],
            security_considerations=["GDPR encryption and EU residency"],
        ),
    ]
    return state


def sound_design_state() -> ArchitectState:
    """A structured design that removes the peak-load coupling."""

    state = _base_state()
    state.blueprint = Blueprint(
        stakeholder_view="Customers browse and check out during peaks without outages.",
        technical_view=(
            "Incrementally migrate the legacy monolith on AWS. A stateless, "
            "horizontally autoscaling Checkout Service publishes orders to SQS; "
            "an independently scalable Order Worker consumes them in the EU. "
            "Managed services fit the medium budget, while encryption and EU "
            "residency support GDPR compliance for 50k-user peak load."
        ),
        components=["Checkout Service", "Order Worker"],
        addressed_feature_ids=["FEAT-001", "FEAT-002"],
        constraints_addressed=[
            "AWS cloud",
            "medium budget",
            "50k peak load",
            "GDPR compliance",
            "legacy monolith migration",
        ],
    )
    state.adrs = [
        ADR(
            id="ADR-001",
            title="ADR-1: Extract checkout behind SQS",
            context="Synchronous checkout saturates the legacy monolith.",
            decision="Use a stateless Checkout Service and queue orders in SQS.",
            rationale="The queue buffers peaks and enables horizontal scaling.",
            alternatives_considered=["Scale the monolith vertically"],
            positive_consequences=["Independent scaling and fault isolation"],
            negative_consequences=["Eventual consistency"],
            related_feature_ids=["FEAT-001"],
            related_component_names=["Checkout Service"],
            source_references=["architecture_patterns.md", "microservices-on-aws.pdf"],
        ),
        ADR(
            id="ADR-002",
            title="ADR-2: Process orders in an EU-region worker",
            context="Order processing must remain available and GDPR compliant.",
            decision="Consume queued orders in an encrypted EU Order Worker.",
            rationale="EU residency protects data while workers scale independently.",
            alternatives_considered=["Process orders synchronously in checkout"],
            positive_consequences=["GDPR alignment and independent scaling"],
            negative_consequences=["Additional operational complexity"],
            related_feature_ids=["FEAT-002"],
            related_component_names=["Order Worker"],
            source_references=["architecture_patterns.md"],
        ),
    ]
    state.components = [
        ComponentDescription(
            id="COMP-001",
            name="Checkout Service",
            purpose="Accept checkout without blocking on order processing.",
            description="Implements FEAT-001 and is justified by ADR-001.",
            related_feature_ids=["FEAT-001"],
            related_adr_ids=["ADR-001"],
            technology_choices=["AWS ECS", "SQS"],
            scalability_considerations=["Stateless horizontal autoscaling"],
        ),
        ComponentDescription(
            id="COMP-002",
            name="Order Worker",
            purpose="Process queued orders in the EU.",
            description="Implements FEAT-002 and is justified by ADR-002.",
            related_feature_ids=["FEAT-002"],
            related_adr_ids=["ADR-002"],
            technology_choices=["AWS managed compute"],
            security_considerations=["GDPR encryption and EU residency"],
            scalability_considerations=["Independent worker autoscaling"],
        ),
    ]
    return state


SCENARIOS = (
    EvalScenario(
        name="A - seeded flaw",
        expected_pass=False,
        build_state=seeded_flaw_state,
    ),
    EvalScenario(
        name="B - sound design",
        expected_pass=True,
        build_state=sound_design_state,
    ),
)
