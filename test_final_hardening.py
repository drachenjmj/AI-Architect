"""test_final_hardening.py — the three targeted fixes from the manual E2E
review of run 20260822T164736Z-1a15e2e4 (PASS with 0 findings, yet):

  1. data-flow endpoint drift — "Order Service (New order flow)" against a
     component named "Order Service" created phantom duplicate diagram
     nodes → deterministic canonicalization at the architect boundary;
  2. unjustified technology drift — "Go-based microservices" over a
     detected Python/Django stack passed review → a deterministic
     language-drift check (justified deviations still pass);
  3. silent migration disposition — a new "User Data Service" with no
     migration step covering it → an explicit prompt contract.

All offline; benchmark names appear only as adversarial fixture INPUT.
"""

from __future__ import annotations

import test_clarifier as tc
from pipeline.agents import architect as arch
from pipeline.review_checks import run_deterministic_checks
from pipeline.state import (
    ADR,
    Blueprint,
    ComponentDescription,
    ContextRecord,
    Feature,
    RepoRepresentation,
    RepoStructure,
    TechStack,
    new_run,
)

# ── 1. data-flow endpoint canonicalization ────────────────────────────────


_COMPONENTS = [
    "API Gateway", "Legacy Monolith (Django)", "Order Service",
    "Product Catalog Service", "Event Broker (AWS SNS/SQS)",
    "Payment Gateway Integration", "User Data Service",
    "React Frontend Application",
]


def test_exact_component_names_are_fixed_points():
    flows = ["React Frontend Application -> API Gateway"]
    canonical, resolved = arch.canonicalize_data_flow_endpoints(
        flows, _COMPONENTS
    )
    assert canonical == ["React Frontend Application → API Gateway"]
    assert resolved == 0                    # nothing to fix; only formatting


def test_parenthetical_descriptors_resolve_to_the_component():
    flows = ["API Gateway -> Order Service (New order flow): order accepted"]
    canonical, resolved = arch.canonicalize_data_flow_endpoints(
        flows, _COMPONENTS
    )
    assert canonical == ["API Gateway → Order Service: order accepted"]
    assert resolved == 1

    canonical, _ = arch.canonicalize_data_flow_endpoints(
        ["API Gateway -> Product Catalog Service (Read operations)"],
        _COMPONENTS,
    )
    assert canonical == ["API Gateway → Product Catalog Service"]


def test_prefix_matches_resolve_unambiguously_with_case_tolerance():
    # "event broker" prefixes exactly one component (the SNS/SQS one).
    canonical, resolved = arch.canonicalize_data_flow_endpoints(
        ["Order Service -> event broker"], _COMPONENTS
    )
    assert canonical == ["Order Service → Event Broker (AWS SNS/SQS)"]
    assert resolved == 1


def test_ambiguous_or_unknown_endpoints_are_left_unchanged():
    catalog = ["Order Service", "Order History Service", "Payment Service"]
    flows = ["API Gateway -> Order"]        # prefix of TWO components
    canonical, resolved = arch.canonicalize_data_flow_endpoints(flows, catalog)
    assert canonical == ["API Gateway → Order"]
    assert resolved == 0

    canonical, _ = arch.canonicalize_data_flow_endpoints(
        ["Client -> Totally Unknown System"], catalog
    )
    assert canonical == ["Client → Totally Unknown System"]  # external kept


def test_external_participants_remain_external():
    catalog = ["API Gateway"]
    canonical, _ = arch.canonicalize_data_flow_endpoints(
        ["Client -> API Gateway -> Third-party Payment Provider (PSP)"],
        catalog,
    )
    # The PSP is not in the catalog: descriptor stripped for matching, but
    # no component matches, so the ORIGINAL text is preserved.
    assert "Third-party Payment Provider (PSP)" in canonical[0]


def test_multiple_endpoints_in_one_flow_chain():
    canonical, resolved = arch.canonicalize_data_flow_endpoints(
        ["Order Service -> Event Broker (Event publishing) -> Payment Gateway Integration"],
        _COMPONENTS,
    )
    assert canonical == [
        "Order Service → Event Broker (AWS SNS/SQS) → Payment Gateway Integration"
    ]
    assert resolved == 1                    # only the decorated one changed


def test_serialization_round_trip_preserves_canonical_flows():
    from pipeline.state import Blueprint as B

    flows, _ = arch.canonicalize_data_flow_endpoints(
        ["API Gateway -> Order Service (New order flow)"], _COMPONENTS
    )
    blueprint = B(
        project_name="P", stakeholder_view="s", technical_view="t",
        components=_COMPONENTS, data_flows=flows,
    )
    restored = B.model_validate_json(blueprint.model_dump_json())
    assert restored.data_flows == ["API Gateway → Order Service"]


def test_duplicate_node_regression_shape_from_the_real_run():
    """The exact drift shapes of the real run collapse to single nodes: the
    diagram catalog afterwards contains no phantom duplicates."""
    real_flows = [
        "API Gateway -> Order Service (New order flow)",
        "API Gateway -> Product Catalog Service (Read operations)",
        "API Gateway -> User Data Service (Auth/Profile)",
        "Legacy Monolith (Django) -> Event Broker (Event publishing for data synchronization)",
    ]
    canonical, resolved = arch.canonicalize_data_flow_endpoints(
        real_flows, _COMPONENTS
    )
    assert resolved == 4
    joined = " ".join(canonical)
    assert "(New order flow)" not in joined
    assert "(Read operations)" not in joined
    assert "(Auth/Profile)" not in joined
    assert "(Event publishing for data synchronization)" not in joined
    assert "Order Service →" in joined or "→ Order Service" in joined


# ── 2. deterministic technology-drift check ───────────────────────────────


def _drift_state(languages, frameworks, tech_choices, technical_view="",
                 adrs=(), pattern=""):
    state = new_run("Modernize a monolith.")
    state.context_record = ContextRecord(
        project_name="P", business_goal="g", problem_statement="p",
        functional_requirements=["orders"],
    )
    state.repo_representation = RepoRepresentation(
        structure=RepoStructure(
            tech_stack=TechStack(languages=languages, frameworks=frameworks)
        )
    )
    state.features = [Feature(
        id="FEAT-001", name="Orders", scenario="s",
        acceptance_criteria=["a"],
    )]
    state.blueprint = Blueprint(
        project_name="P", selected_pattern=pattern,
        stakeholder_view="sv", technical_view=technical_view,
        addressed_feature_ids=["FEAT-001"],
    )
    state.adrs = list(adrs)
    state.components = [
        ComponentDescription(
            id="COMP-001", name="Order Service", purpose="p",
            description="d", technology_choices=list(tech_choices),
            related_feature_ids=["FEAT-001"],
        )
    ]
    return state


def test_python_django_to_go_without_justification_finds_drift():
    state = _drift_state(
        {"Python": 17000}, ["Django"],
        tech_choices=["Go", "AWS SQS"],
        technical_view="new Go-based microservices behind a gateway",
    )
    checks = run_deterministic_checks(state)

    assert "Go" in checks.unjustified_language_drift
    assert any(
        "Go" in issue.finding and issue.severity == "high"
        for issue in checks.issues
    )


def test_python_django_to_go_with_explicit_adr_passes():
    adr = ADR(
        id="ADR-001", title="ADR-1: Select Go for the order service",
        context="The existing Python/Django stack struggles with the stated "
                "1,000 orders/minute on a single process.",
        decision="Implement the order service in Go.",
        rationale="Go's concurrency model meets the latency requirement "
                  "that Django could not; re-skilling cost is accepted.",
        alternatives_considered=["Stay on Python/Django"],
        positive_consequences=["Meets latency target."],
        negative_consequences=["Polyglot maintenance."],
    )
    state = _drift_state(
        {"Python": 17000}, ["Django"],
        tech_choices=["Go", "AWS SQS"],
        technical_view="Go-based order service", adrs=[adr],
    )
    checks = run_deterministic_checks(state)

    assert checks.unjustified_language_drift == {}
    assert not any("Go" in i.finding for i in checks.issues)


def test_java_spring_to_node_without_justification_finds_drift():
    state = _drift_state(
        {"Java": 52000}, ["Spring Boot"],
        tech_choices=["Node.js", "Express"], technical_view="Node services",
    )
    checks = run_deterministic_checks(state)
    assert "JavaScript" in checks.unjustified_language_drift


def test_existing_stack_unchanged_has_no_drift():
    state = _drift_state(
        {"Python": 17000}, ["Django"],
        tech_choices=["Python", "Django", "PostgreSQL"],
        technical_view="Python services with Django",
    )
    assert run_deterministic_checks(state).unjustified_language_drift == {}


def test_infrastructure_only_additions_have_no_drift():
    state = _drift_state(
        {"Python": 17000}, ["Django"],
        tech_choices=["AWS SQS", "AWS SNS", "Redis", "AWS API Gateway",
                      "PostgreSQL (RDS)"],
        technical_view="Python services with AWS managed infrastructure",
    )
    assert run_deterministic_checks(state).unjustified_language_drift == {}


def test_no_detected_stack_has_no_drift():
    state = _drift_state(
        {}, [], tech_choices=["Go"], technical_view="Go services",
    )
    assert run_deterministic_checks(state).unjustified_language_drift == {}


def test_greenfield_has_no_drift():
    state = _drift_state(
        {"Python": 17000}, ["Django"], tech_choices=["Go"],
    )
    state.repo_representation = None         # greenfield: nothing detected
    assert run_deterministic_checks(state).unjustified_language_drift == {}


# ── 3. migration-disposition prompt contract ─────────────────────────────


def test_prompt_requires_migration_disposition_for_new_services():
    prompt = " ".join(arch.ARCHITECTURE_SYSTEM_PROMPT.split())
    assert "MIGRATION DISPOSITION" in prompt
    assert "explicit migration disposition" in prompt
    for disposition in (
        "named migration step", "deferred to a later phase",
        "retained in the legacy system", "external/already existing",
    ):
        assert disposition in prompt, disposition
    assert "never silently introduce a service with no stated migration path" in prompt


def test_data_flow_rule_now_demands_exact_spelling():
    prompt = " ".join(arch.ARCHITECTURE_SYSTEM_PROMPT.split())
    assert "spelled EXACTLY as in the Blueprint" in prompt
    assert "no parenthetical descriptors or suffixes" in prompt


def test_no_benchmark_names_hard_coded():
    source = arch.ARCHITECTURE_SYSTEM_PROMPT
    for banned in ("User Data Service", "Product Catalog", "ecommerce",
                   "harsh020", "Go-based"):
        assert banned not in source, banned
