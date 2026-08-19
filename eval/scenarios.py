"""Cross-domain seed cases for Reviewer evaluation.

These cases are deliberately marked provisional. They exercise the benchmark
machinery and provide an initial error-analysis set; they are not a substitute
for independently reviewed labels or real pipeline outputs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

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
    REVIEW_CODE_SCORE_FIELDS,
    REVIEW_JUDGMENT_FIELDS,
    REVIEW_VERDICT_JUDGMENT_FIELDS,
    Stage,
    new_run,
)


CODE_SCORE_FIELDS = REVIEW_CODE_SCORE_FIELDS
JUDGMENT_FIELDS = REVIEW_JUDGMENT_FIELDS


def _full_code_scores(**overrides: int) -> dict[str, int]:
    scores = {name: 2 for name in CODE_SCORE_FIELDS}
    scores.update(overrides)
    return scores


def _judgments(**overrides: bool) -> dict[str, bool]:
    values = {name: True for name in JUDGMENT_FIELDS}
    values.update(overrides)
    return values


@dataclass(frozen=True)
class EvalScenario:
    """One labeled architecture state used to measure Reviewer agreement."""

    name: str
    domain: str
    description: str
    expected_pass: bool
    expected_code_scores: Mapping[str, int]
    expected_judgments: Mapping[str, bool]
    build_state: Callable[[], ArchitectState]
    label_status: str = "provisional"
    label_rationale: str = ""

    def __post_init__(self) -> None:
        missing_code = set(CODE_SCORE_FIELDS) - set(self.expected_code_scores)
        missing_judgments = set(JUDGMENT_FIELDS) - set(self.expected_judgments)
        extra_code = set(self.expected_code_scores) - set(CODE_SCORE_FIELDS)
        extra_judgments = set(self.expected_judgments) - set(JUDGMENT_FIELDS)
        if missing_code or missing_judgments or extra_code or extra_judgments:
            raise ValueError(
                f"Incomplete labels for {self.name}: "
                f"missing_code={sorted(missing_code)}, "
                f"missing_judgments={sorted(missing_judgments)}, "
                f"extra_code={sorted(extra_code)}, "
                f"extra_judgments={sorted(extra_judgments)}"
            )

        invalid_scores = {
            name: value
            for name, value in self.expected_code_scores.items()
            if type(value) is not int or not 0 <= value <= 2
        }
        invalid_judgments = {
            name: value
            for name, value in self.expected_judgments.items()
            if type(value) is not bool
        }
        if invalid_scores or invalid_judgments:
            raise ValueError(
                f"Invalid labels for {self.name}: "
                f"code={invalid_scores}, judgments={invalid_judgments}"
            )

        derived_pass = all(
            score == 2 for score in self.expected_code_scores.values()
        ) and all(
            self.expected_judgments[name]
            for name in REVIEW_VERDICT_JUDGMENT_FIELDS
        )
        if self.expected_pass != derived_pass:
            raise ValueError(
                f"Inconsistent verdict label for {self.name}: "
                f"expected_pass={self.expected_pass}, derived={derived_pass}"
            )


def _single_feature_state(
    *,
    prompt: str,
    project_name: str,
    business_goal: str,
    problem_statement: str,
    functional_requirement: str,
    non_functional_requirements: list[str],
    cloud_provider: str,
    budget: str,
    compliance_requirements: list[str],
    existing_systems: list[str],
    feature_name: str,
    feature_scenario: str,
    acceptance_criterion: str,
    pattern: str,
    technical_view: str,
    constraints_addressed: list[str],
    adr_title: str,
    adr_context: str,
    adr_decision: str,
    adr_rationale: str,
    adr_alternative: str,
    positive_consequence: str,
    negative_consequence: str,
    component_name: str,
    component_purpose: str,
    component_description: str,
    source: str,
    source_content: str,
    repo_url: str = "",
    repo_representation: RepoRepresentation | None = None,
) -> ArchitectState:
    state = new_run(prompt, repo_url=repo_url)
    state.context_record = ContextRecord(
        project_name=project_name,
        business_goal=business_goal,
        problem_statement=problem_statement,
        functional_requirements=[functional_requirement],
        non_functional_requirements=non_functional_requirements,
        cloud_provider=cloud_provider,
        budget=budget,
        compliance_requirements=compliance_requirements,
        existing_systems=existing_systems,
        summary=(
            f"Goal: {business_goal}. Problem: {problem_statement}. "
            f"Cloud: {cloud_provider or 'unspecified'}. Budget: {budget or 'unspecified'}."
        ),
    )
    state.repo_representation = repo_representation
    state.retrieved_knowledge = [
        KBChunk(content=source_content, source=source, box=1)
    ]
    state.features = [
        Feature(
            id="FEAT-001",
            name=feature_name,
            description=functional_requirement,
            scenario=feature_scenario,
            acceptance_criteria=[acceptance_criterion],
        )
    ]
    state.blueprint = Blueprint(
        project_name=project_name,
        selected_pattern=pattern,
        rationale=adr_rationale,
        stakeholder_view=business_goal,
        technical_view=technical_view,
        components=[component_name],
        data_flows=[f"Requests flow through {component_name}."],
        addressed_feature_ids=["FEAT-001"],
        constraints_addressed=constraints_addressed,
    )
    state.adrs = [
        ADR(
            id="ADR-001",
            title=adr_title,
            context=adr_context,
            decision=adr_decision,
            rationale=adr_rationale,
            alternatives_considered=[adr_alternative],
            positive_consequences=[positive_consequence],
            negative_consequences=[negative_consequence],
            related_feature_ids=["FEAT-001"],
            related_component_names=[component_name],
            source_references=[source],
        )
    ]
    state.components = [
        ComponentDescription(
            id="COMP-001",
            name=component_name,
            purpose=component_purpose,
            description=component_description,
            related_feature_ids=["FEAT-001"],
            related_adr_ids=["ADR-001"],
            technology_choices=[pattern],
            security_considerations=compliance_requirements,
            scalability_considerations=non_functional_requirements,
        )
    ]
    state.stage = Stage.DESIGNING
    return state


def _shop_repo() -> RepoRepresentation:
    return RepoRepresentation(
        meta=RepoMeta(
            url="https://example.invalid/seasonal-shop",
            commit_sha="synthetic-shop-v1",
        ),
        structure=RepoStructure(
            file_tree="shop/app.py\nshop/checkout.py\nshop/orders.py",
            repo_map=(
                "app.py serves catalog and checkout in one process; "
                "checkout.py calls orders.py synchronously"
            ),
        ),
        behavior=RepoBehavior(
            overview=(
                "One deployable serves browsing, checkout, and synchronous "
                "order processing with in-process sessions."
            )
        ),
    )


def sound_shop_state() -> ArchitectState:
    """Brownfield repair that removes the repository's peak-load bottleneck."""

    return _single_feature_state(
        prompt="Make the existing shop survive seasonal peaks without losing orders.",
        project_name="Seasonal Shop",
        business_goal="Customers can complete purchases during seasonal peaks.",
        problem_statement="Synchronous order processing saturates the shop deployable.",
        functional_requirement="Accept and process customer orders reliably.",
        non_functional_requirements=["Handle 50k concurrent users"],
        cloud_provider="AWS",
        budget="medium",
        compliance_requirements=["GDPR"],
        existing_systems=["Legacy shop monolith"],
        feature_name="Peak-safe checkout",
        feature_scenario="Checkout remains responsive while 50k users shop.",
        acceptance_criterion="Peak orders are accepted without saturating the storefront.",
        pattern="Queue-buffered incremental extraction",
        technical_view=(
            "On AWS, extract checkout behind an SQS boundary while incrementally "
            "migrating the legacy shop monolith. Stateless checkout instances scale "
            "for 50k concurrent users; encrypted EU processing supports GDPR and "
            "managed services remain within the medium budget."
        ),
        constraints_addressed=[
            "accept and process customer orders reliably", "50k concurrent users", "AWS",
            "medium budget", "GDPR", "legacy shop monolith migration",
        ],
        adr_title="ADR-1: Buffer checkout work with SQS",
        adr_context="Synchronous order processing saturates the monolith at peak.",
        adr_decision="Extract checkout incrementally and buffer orders with SQS.",
        adr_rationale="A queue isolates peak traffic while retaining an incremental migration.",
        adr_alternative="Vertically scale the whole monolith.",
        positive_consequence="Checkout and order processing scale independently.",
        negative_consequence="Order confirmation becomes eventually consistent.",
        component_name="Checkout Service",
        component_purpose="Accept checkout requests without blocking on order processing.",
        component_description="Implements FEAT-001 under ADR-001 using an AWS queue.",
        source="queue-pattern.md",
        source_content=(
            "A durable queue buffers burst traffic and decouples producers from "
            "slower consumers; consumers must be idempotent."
        ),
        repo_url="https://example.invalid/seasonal-shop",
        repo_representation=_shop_repo(),
    )


def symptom_patch_shop_state() -> ArchitectState:
    """Formally complete brownfield design that only patches the symptom."""

    state = sound_shop_state()
    state.blueprint.selected_pattern = "Vertically scaled monolith"
    state.blueprint.rationale = "Avoid migration by buying a larger instance."
    state.blueprint.technical_view = (
        "Keep the legacy shop monolith as one AWS deployable, move it to a larger "
        "instance, and restart it nightly. This claims to handle 50k concurrent "
        "users within the medium budget while retaining GDPR encryption."
    )
    state.adrs[0].title = "ADR-1: Vertically scale the shop monolith"
    state.adrs[0].decision = "Keep synchronous processing and use a larger EC2 instance."
    state.adrs[0].rationale = "This avoids migration work."
    state.adrs[0].alternatives_considered = ["Queue-buffered checkout extraction"]
    state.adrs[0].positive_consequences = ["No immediate application rewrite"]
    state.adrs[0].negative_consequences = ["The single bottleneck remains"]
    state.components[0].name = "Shop Monolith"
    state.components[0].purpose = "Continue serving catalog, checkout, and orders together."
    state.components[0].description = "Implements FEAT-001 under ADR-001 on one large EC2 instance."
    state.blueprint.components = ["Shop Monolith"]
    state.adrs[0].related_component_names = ["Shop Monolith"]
    return state


# Backward-compatible names used by ui_demo.py. Its presentation fixture edits
# a second GDPR feature/component by index, so retain that shape without making
# the new benchmark cases depend on it.
def _add_demo_gdpr_artifacts(
    state: ArchitectState,
    *,
    component_name: str,
    component_purpose: str,
    component_description: str,
    adr_decision: str,
) -> ArchitectState:
    state.features.append(
        Feature(
            id="FEAT-002",
            name="Protect EU order data",
            scenario="EU order data is encrypted and remains in the EU.",
            acceptance_criteria=["EU order data remains encrypted in an EU region."],
        )
    )
    state.blueprint.addressed_feature_ids.append("FEAT-002")
    state.blueprint.components.append(component_name)
    state.adrs.append(
        ADR(
            id="ADR-002",
            title=f"ADR-2: Protect order data in {component_name}",
            context="Order data must remain GDPR compliant.",
            decision=adr_decision,
            rationale="Encryption and EU residency protect personal order data.",
            alternatives_considered=["Store order data outside the EU"],
            positive_consequences=["EU data residency"],
            negative_consequences=["Regional operations overhead"],
            related_feature_ids=["FEAT-002"],
            related_component_names=[component_name],
            source_references=["queue-pattern.md"],
        )
    )
    state.components.append(
        ComponentDescription(
            id="COMP-002",
            name=component_name,
            purpose=component_purpose,
            description=component_description,
            related_feature_ids=["FEAT-002"],
            related_adr_ids=["ADR-002"],
            security_considerations=["GDPR encryption and EU residency"],
        )
    )
    return state


def seeded_flaw_state() -> ArchitectState:
    return _add_demo_gdpr_artifacts(
        symptom_patch_shop_state(),
        component_name="Order Module",
        component_purpose="Process orders inside the monolith.",
        component_description="Implements FEAT-002 under ADR-002 inside one deployable.",
        adr_decision="Keep the Order Module and encrypt its EU database.",
    )


def sound_design_state() -> ArchitectState:
    return _add_demo_gdpr_artifacts(
        sound_shop_state(),
        component_name="Order Worker",
        component_purpose="Process queued orders in the EU.",
        component_description="Implements FEAT-002 under ADR-002 as an independent worker.",
        adr_decision="Process encrypted queued orders in an EU-region worker.",
    )


def sound_healthcare_state() -> ArchitectState:
    """Greenfield healthcare design with explicit privacy and availability choices."""

    return _single_feature_state(
        prompt="Design a GDPR-compliant, highly available appointment platform.",
        project_name="Appointment Platform",
        business_goal="Patients can reliably book and cancel appointments.",
        problem_statement="Manual scheduling causes missed and duplicate bookings.",
        functional_requirement="Book, reschedule, and cancel appointments.",
        non_functional_requirements=["99.9% availability"],
        cloud_provider="Azure",
        budget="moderate",
        compliance_requirements=["GDPR"],
        existing_systems=[],
        feature_name="Reliable appointment booking",
        feature_scenario="A patient books one available slot without duplication.",
        acceptance_criterion="Concurrent requests cannot double-book a slot.",
        pattern="Transactional modular service",
        technical_view=(
            "An Azure appointment service uses transactional slot reservations, "
            "zone-redundant deployment for 99.9% availability, and encrypted EU "
            "storage with deletion workflows for GDPR. Autoscaling and managed "
            "database tiers fit a moderate budget."
        ),
        constraints_addressed=[
            "book reschedule cancel appointments", "99.9% availability", "Azure",
            "moderate budget", "GDPR",
        ],
        adr_title="ADR-1: Reserve appointment slots transactionally",
        adr_context="Concurrent booking requests can select the same slot.",
        adr_decision="Use transactional slot reservations in an EU-region database.",
        adr_rationale="Atomic reservations prevent duplicates and support GDPR controls.",
        adr_alternative="Coordinate bookings with an in-memory lock on one server.",
        positive_consequence="Double-booking is prevented across service instances.",
        negative_consequence="Database contention must be monitored.",
        component_name="Appointment Service",
        component_purpose="Own appointment availability and booking transactions.",
        component_description="Implements FEAT-001 and ADR-001 with EU data controls.",
        source="healthcare-privacy.md",
        source_content=(
            "Appointment data should use access controls, encryption, retention "
            "limits, and EU-region processing when GDPR applies."
        ),
    )


def healthcare_compliance_failure_state() -> ArchitectState:
    """Complete-looking design that contradicts its GDPR constraint."""

    state = sound_healthcare_state()
    state.blueprint.technical_view = (
        "The Azure appointment service offers 99.9% availability and booking, "
        "rescheduling, and cancellation. For simpler operations it stores all GDPR "
        "patient records unencrypted in a US-region database on a moderate budget."
    )
    state.adrs[0].decision = "Store unencrypted patient records in one US region."
    state.adrs[0].rationale = "A single global database is operationally simpler."
    state.adrs[0].positive_consequences = ["Simpler database operations"]
    state.adrs[0].negative_consequences = ["GDPR residency and encryption are violated"]
    state.components[0].security_considerations = [
        "GDPR noted, but records remain unencrypted in the US"
    ]
    return state


def repo_mismatch_state() -> ArchitectState:
    """Design solves the request but contradicts the supplied repository facts."""

    repo = RepoRepresentation(
        meta=RepoMeta(
            url="https://example.invalid/warehouse-api",
            commit_sha="synthetic-warehouse-v1",
        ),
        structure=RepoStructure(
            file_tree="app/main.py\napp/audit.py\nrequirements.txt",
            repo_map="main.py exposes a FastAPI API backed by PostgreSQL; no Kafka dependency",
        ),
        behavior=RepoBehavior(overview="FastAPI warehouse API with PostgreSQL persistence."),
    )
    return _single_feature_state(
        prompt="Add tamper-evident audit logging to our warehouse API.",
        project_name="Warehouse API",
        business_goal="Operators can trace every inventory change.",
        problem_statement="Inventory mutations do not have a reliable audit trail.",
        functional_requirement="Record every inventory mutation in an audit trail.",
        non_functional_requirements=["Audit records retained for seven years"],
        cloud_provider="AWS",
        budget="low",
        compliance_requirements=[],
        existing_systems=["FastAPI warehouse API with PostgreSQL"],
        feature_name="Inventory audit trail",
        feature_scenario="Every inventory update creates an immutable audit event.",
        acceptance_criterion="An operator can trace who changed an item and when.",
        pattern="Kafka event sourcing rewrite",
        technical_view=(
            "Replace the existing FastAPI warehouse API and PostgreSQL database with "
            "a Java service using the repository's existing Kafka cluster, although "
            "the supplied repository contains no Kafka dependency. Retain audit events "
            "for seven years on AWS while keeping a low budget."
        ),
        constraints_addressed=[
            "inventory mutation audit trail", "seven years", "AWS", "low budget",
            "replace FastAPI warehouse API with PostgreSQL",
        ],
        adr_title="ADR-1: Rewrite inventory around Kafka event sourcing",
        adr_context="Inventory changes need a tamper-evident audit trail.",
        adr_decision="Replace the API with Java and publish every mutation to Kafka.",
        adr_rationale="The repository already operates Kafka, so no new platform is required.",
        adr_alternative="Add an append-only audit table to PostgreSQL.",
        positive_consequence="Every change has an immutable event.",
        negative_consequence="A full rewrite increases delivery risk.",
        component_name="Inventory Event Service",
        component_purpose="Record inventory mutations as immutable events.",
        component_description="Implements FEAT-001 under ADR-001 through Kafka.",
        source="audit-logging.md",
        source_content="Append-only audit records should capture actor, action, target, and timestamp.",
        repo_url="https://example.invalid/warehouse-api",
        repo_representation=repo,
    )


def fabricated_source_state() -> ArchitectState:
    """Sound design whose ADR cites evidence that was never supplied."""

    state = sound_healthcare_state()
    state.adrs[0].source_references = ["imaginary-clinical-standard.pdf"]
    return state


def sound_modular_monolith_state() -> ArchitectState:
    """Valid non-microservice alternative that guards against pattern bias."""

    return _single_feature_state(
        prompt="Design a low-cost inventory tool for one small warehouse.",
        project_name="Small Warehouse Inventory",
        business_goal="Five operators can track stock accurately.",
        problem_statement="Spreadsheet updates overwrite each other.",
        functional_requirement="Record stock receipts, movements, and adjustments.",
        non_functional_requirements=["Support five concurrent operators"],
        cloud_provider="",
        budget="low",
        compliance_requirements=[],
        existing_systems=[],
        feature_name="Consistent stock updates",
        feature_scenario="Five operators update different stock items concurrently.",
        acceptance_criterion="Committed stock changes are not lost or duplicated.",
        pattern="Modular monolith",
        technical_view=(
            "A modular monolith provides receipt, movement, and adjustment modules "
            "over one transactional database. Optimistic concurrency supports five "
            "concurrent operators, while one deployable keeps the low budget viable."
        ),
        constraints_addressed=[
            "stock receipts movements adjustments", "five concurrent operators", "low budget",
        ],
        adr_title="ADR-1: Use a modular monolith with optimistic concurrency",
        adr_context="A small team needs consistent stock updates at low operational cost.",
        adr_decision="Use one modular application and a transactional database.",
        adr_rationale="The expected scale does not justify distributed-system overhead.",
        adr_alternative="Deploy independent microservices for each inventory operation.",
        positive_consequence="Simple deployment and transaction boundaries.",
        negative_consequence="Modules cannot be deployed independently.",
        component_name="Inventory Application",
        component_purpose="Own transactional stock operations in separated modules.",
        component_description="Implements FEAT-001 under ADR-001 without distributed overhead.",
        source="modular-monolith.md",
        source_content=(
            "A modular monolith can preserve clear boundaries and transactions when "
            "scale and team size do not justify distributed operations."
        ),
    )


SCENARIOS = (
    EvalScenario(
        name="brownfield_shop_sound",
        domain="ecommerce",
        description="Repository-grounded structural repair.",
        expected_pass=True,
        expected_code_scores=_full_code_scores(),
        expected_judgments=_judgments(),
        build_state=sound_shop_state,
        label_rationale="The design removes synchronous peak-load coupling and addresses all stated constraints.",
    ),
    EvalScenario(
        name="brownfield_shop_symptom_patch",
        domain="ecommerce",
        description="Complete design that leaves the bottleneck intact.",
        expected_pass=False,
        expected_code_scores=_full_code_scores(),
        expected_judgments=_judgments(
            flaw_detection=False,
            adr_soundness=False,
            best_practice_grounding=False,
        ),
        build_state=symptom_patch_shop_state,
        label_rationale="A larger instance does not remove the repository-evidenced single bottleneck.",
    ),
    EvalScenario(
        name="greenfield_healthcare_sound",
        domain="healthcare",
        description="Greenfield design with repository criterion not applicable.",
        expected_pass=True,
        expected_code_scores=_full_code_scores(),
        expected_judgments=_judgments(),
        build_state=sound_healthcare_state,
        label_rationale="The design satisfies booking, availability, budget, and GDPR constraints without requiring a repository.",
    ),
    EvalScenario(
        name="greenfield_healthcare_compliance_failure",
        domain="healthcare",
        description="Design explicitly contradicting GDPR evidence.",
        expected_pass=False,
        expected_code_scores=_full_code_scores(),
        expected_judgments=_judgments(
            flaw_detection=False,
            adr_soundness=False,
            best_practice_grounding=False,
        ),
        build_state=healthcare_compliance_failure_state,
        label_rationale="The architecture mentions GDPR but violates the supplied residency and encryption guidance.",
    ),
    EvalScenario(
        name="brownfield_repository_mismatch",
        domain="warehouse",
        description="Design assumes repository technology that is absent.",
        expected_pass=False,
        expected_code_scores=_full_code_scores(),
        expected_judgments=_judgments(
            repo_grounding=False,
            adr_soundness=False,
            best_practice_grounding=False,
        ),
        build_state=repo_mismatch_state,
        label_rationale="The design bases a rewrite on a Kafka platform contradicted by the repository evidence.",
    ),
    EvalScenario(
        name="fabricated_source_reference",
        domain="healthcare",
        description="Sound architecture containing a fabricated ADR citation.",
        expected_pass=False,
        expected_code_scores=_full_code_scores(source_integrity=0),
        expected_judgments=_judgments(best_practice_grounding=False),
        build_state=fabricated_source_state,
        label_rationale="The cited source is absent from both retrieved knowledge and repository evidence.",
    ),
    EvalScenario(
        name="greenfield_modular_monolith_sound",
        domain="warehouse",
        description="Low-scale valid design that should not be forced into microservices.",
        expected_pass=True,
        expected_code_scores=_full_code_scores(),
        expected_judgments=_judgments(),
        build_state=sound_modular_monolith_state,
        label_rationale="A modular monolith is proportionate to the stated scale, budget, and team context.",
    ),
)


def load_scenario_file(path: str | Path) -> EvalScenario:
    """Load a human-labeled saved ArchitectState benchmark case from JSON."""

    case_path = Path(path)
    payload = json.loads(case_path.read_text(encoding="utf-8"))
    snapshot = ArchitectState.model_validate(payload["state"])

    def build_state() -> ArchitectState:
        return snapshot.model_copy(deep=True)

    return EvalScenario(
        name=payload["name"],
        domain=payload.get("domain", "unspecified"),
        description=payload.get("description", "Saved pipeline output"),
        expected_pass=bool(payload["expected_pass"]),
        expected_code_scores=payload["expected_code_scores"],
        expected_judgments=payload["expected_judgments"],
        build_state=build_state,
        label_status=payload.get("label_status", "provisional"),
        label_rationale=payload.get("label_rationale", ""),
    )
