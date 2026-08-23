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
    MigrationStep,
    PartitionSummary,
    RepoBehavior,
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


# ── 4. cross-artifact data-flow participant resolution (invariant A) ──────
#
# Manual E2E inspection of run 20260822T170536Z-04942cfe: `data_flows`
# referenced a service and a database that had no Component Description —
# the run still passed review. These fixtures use non-benchmark names
# ("Fulfillment Service"/"Fulfillment Database") to reproduce the same
# SHAPE without hard-coding the real run's names anywhere but as adversarial
# fixture input.

_FLOW_COMPONENTS = ["API Gateway", "Order Service", "Order Database"]


def _flow_state(data_flows, components=None):
    state = new_run("Modernize a monolith.")
    state.context_record = ContextRecord(
        project_name="P", business_goal="g", problem_statement="p",
        functional_requirements=["orders"],
    )
    state.features = [Feature(
        id="FEAT-001", name="Orders", scenario="s",
        acceptance_criteria=["a"],
    )]
    names = list(components) if components is not None else _FLOW_COMPONENTS
    state.blueprint = Blueprint(
        project_name="P", stakeholder_view="sv", technical_view="tv",
        components=names, data_flows=list(data_flows),
        addressed_feature_ids=["FEAT-001"],
    )
    state.components = [
        ComponentDescription(
            id=f"COMP-{i}", name=name, purpose="p", description="d",
            related_feature_ids=["FEAT-001"],
        )
        for i, name in enumerate(names, start=1)
    ]
    return state


def test_all_flow_participants_resolve_to_components_passes():
    state = _flow_state([
        "API Gateway → Order Service: place order",
        "Order Service → Order Database: persist order",
    ])
    checks = run_deterministic_checks(state)
    assert checks.unresolved_flow_participants == {}
    assert not any("data-flow participant" in i.finding for i in checks.issues)


def test_internal_service_referenced_only_in_flow_is_high_finding():
    state = _flow_state([
        "API Gateway → Fulfillment Service: dispatch shipment",
    ])
    checks = run_deterministic_checks(state)
    assert "Fulfillment Service" in checks.unresolved_flow_participants
    assert any(
        "Fulfillment Service" in issue.finding and issue.severity == "high"
        for issue in checks.issues
    )


def test_internal_database_referenced_only_in_flow_is_high_finding():
    state = _flow_state([
        "Fulfillment Service → Fulfillment Database: read stock levels",
    ])
    checks = run_deterministic_checks(state)
    assert "Fulfillment Database" in checks.unresolved_flow_participants


def test_canonicalized_decorated_name_resolving_to_a_component_passes():
    # Mirrors what actually reaches the Reviewer: the Architect boundary
    # canonicalizes decorated endpoints against the Blueprint's own
    # `components` list BEFORE this state is ever reviewed.
    canonical, _ = arch.canonicalize_data_flow_endpoints(
        ["API Gateway -> Order Service (New order flow)"], _FLOW_COMPONENTS
    )
    state = _flow_state(canonical)
    checks = run_deterministic_checks(state)
    assert checks.unresolved_flow_participants == {}


def test_explicit_external_actor_or_provider_passes():
    state = _flow_state([
        "Client → API Gateway: checkout request",
        "Order Service → Third-party Payment Provider: capture payment",
    ])
    checks = run_deterministic_checks(state)
    assert checks.unresolved_flow_participants == {}


def test_multiple_unresolved_participants_are_each_named_exactly():
    state = _flow_state([
        "API Gateway → Fulfillment Service: dispatch shipment",
        "Fulfillment Service → Shipping Ledger: record shipment",
    ])
    checks = run_deterministic_checks(state)
    assert set(checks.unresolved_flow_participants) == {
        "Fulfillment Service", "Shipping Ledger",
    }
    findings = " ".join(issue.finding for issue in checks.issues)
    assert "Fulfillment Service" in findings
    assert "Shipping Ledger" in findings


def test_unresolved_participant_does_not_silently_create_a_component():
    state = _flow_state([
        "API Gateway → Fulfillment Service: dispatch shipment",
    ])
    before = list(state.components)
    run_deterministic_checks(state)
    assert state.components == before
    assert not any(c.name == "Fulfillment Service" for c in state.components)


def test_real_run_shaped_regression_without_hard_coded_names():
    """Reproduces the observed shape (a service and a database that exist
    only in `data_flows`) without naming the real benchmark components."""
    state = _flow_state([
        "Order Service → Fulfillment Service: request shipment",
        "Fulfillment Service → Fulfillment Database: persist shipment",
    ])
    checks = run_deterministic_checks(state)
    assert checks.unresolved_flow_participants == {
        "Fulfillment Service": [
            "Fulfillment Service → Fulfillment Database: persist shipment",
            "Order Service → Fulfillment Service: request shipment",
        ],
        "Fulfillment Database": [
            "Fulfillment Service → Fulfillment Database: persist shipment",
        ],
    }


def test_arrowless_sentence_flows_carry_no_endpoints():
    """Pre-diagram-contract prose flows are judged elsewhere (now by the
    directionality invariant), not parsed as endpoints — a lone sentence
    must not become one giant "endpoint". The property now lives in the
    ONE shared grammar both the renderer and the Reviewer use."""
    from pipeline.flow_syntax import split_directional_flow

    assert split_directional_flow(
        "Order Service publishes OrderCreated to the Shared Event Bus, "
        "which fans out to the Payment Service."
    ) is None


# ── 5. migration disposition for new internal target services (invariant B) ─
#
# Same run: "Payment Service" was a new internal target component with no
# migration plan step covering it. Fixtures use non-benchmark names.


def _migration_state(
    *,
    services,
    databases=(),
    repo_overview="Existing shop platform handling catalog and checkout.",
    partitions=(),
    migration_steps=(),
    open_risks=(),
    repo_representation=True,
):
    state = new_run("Modernize a monolithic shop.")
    state.context_record = ContextRecord(
        project_name="P", business_goal="g", problem_statement="p",
        functional_requirements=["orders"],
    )
    if repo_representation:
        state.repo_representation = RepoRepresentation(
            behavior=RepoBehavior(overview=repo_overview, partitions=list(partitions)),
        )
    else:
        state.repo_representation = None
    state.features = [Feature(
        id="FEAT-001", name="Orders", scenario="s",
        acceptance_criteria=["a"],
    )]
    names = list(services) + list(databases)
    state.blueprint = Blueprint(
        project_name="P", stakeholder_view="sv", technical_view="tv",
        components=names, addressed_feature_ids=["FEAT-001"],
        migration_steps=list(migration_steps),
        open_risks=list(open_risks),
    )
    state.components = [
        ComponentDescription(
            id=f"COMP-{i}", name=name, purpose="p", description="d",
            related_feature_ids=["FEAT-001"],
        )
        for i, name in enumerate(services, start=1)
    ] + [
        ComponentDescription(
            id=f"COMP-DB-{i}", name=name, component_type="database",
            purpose="p", description="d", related_feature_ids=["FEAT-001"],
        )
        for i, name in enumerate(databases, start=1)
    ]
    return state


def test_new_service_extracted_in_a_migration_step_passes():
    state = _migration_state(
        services=["Fulfillment Service"],
        migration_steps=[MigrationStep(
            title="Extract Fulfillment",
            objective="Extract the Fulfillment Service from the monolith.",
        )],
    )
    checks = run_deterministic_checks(state)
    assert checks.services_without_migration_disposition == []


def test_new_service_explicitly_deferred_passes():
    state = _migration_state(
        services=["Fulfillment Service"],
        open_risks=[
            "The Fulfillment Service is deferred to a later phase of the "
            "migration."
        ],
    )
    checks = run_deterministic_checks(state)
    assert checks.services_without_migration_disposition == []


def test_new_service_explicitly_retained_in_legacy_passes():
    state = _migration_state(
        services=["Fulfillment Service"],
        open_risks=[
            "The Fulfillment Service remains in the legacy monolith for now."
        ],
    )
    checks = run_deterministic_checks(state)
    assert checks.services_without_migration_disposition == []


def test_new_service_explicit_external_or_already_existing_passes():
    external_state = _migration_state(
        services=["Fulfillment Service"],
        open_risks=["The Fulfillment Service is a third-party provider."],
    )
    assert run_deterministic_checks(external_state).services_without_migration_disposition == []

    existing_state = _migration_state(
        services=["Fulfillment Service"],
        open_risks=["The Fulfillment Service already exists and is unchanged."],
    )
    assert run_deterministic_checks(existing_state).services_without_migration_disposition == []


def test_new_internal_service_with_no_disposition_is_high_finding():
    state = _migration_state(services=["Fulfillment Service"])
    checks = run_deterministic_checks(state)
    assert checks.services_without_migration_disposition == ["Fulfillment Service"]
    assert any(
        "Fulfillment Service" in issue.finding
        and "migration disposition" in issue.finding
        and issue.severity == "high"
        for issue in checks.issues
    )


def test_multiple_services_covered_by_one_named_step_passes():
    state = _migration_state(
        services=["Fulfillment Service", "Shipping Service"],
        migration_steps=[MigrationStep(
            title="Extract fulfillment and shipping",
            objective="Extract the Fulfillment Service and Shipping Service together.",
        )],
    )
    checks = run_deterministic_checks(state)
    assert checks.services_without_migration_disposition == []


def test_database_component_does_not_require_its_own_migration_step():
    state = _migration_state(
        services=[],
        databases=["Fulfillment Database"],
    )
    checks = run_deterministic_checks(state)
    assert checks.services_without_migration_disposition == []


def test_greenfield_state_does_not_trigger_disposition_logic():
    state = _migration_state(
        services=["Fulfillment Service"], repo_representation=False,
    )
    checks = run_deterministic_checks(state)
    assert checks.services_without_migration_disposition == []


def test_migration_disposition_logic_is_name_generic():
    """The check must key off structure (component_type, migration-step
    text, disposition phrases) — never a hard-coded benchmark service name.
    Proven behaviourally: the SAME fixture shape, run under an unrelated
    name, produces the same flagged/cleared outcome as the real run's
    "Payment Service" would.
    """
    flagged = _migration_state(services=["Unrelated Widget Service"])
    assert run_deterministic_checks(flagged).services_without_migration_disposition == [
        "Unrelated Widget Service"
    ]

    cleared = _migration_state(
        services=["Unrelated Widget Service"],
        migration_steps=[MigrationStep(
            title="Extract the widget capability",
            objective="Extract the Unrelated Widget Service.",
        )],
    )
    assert cleared.components  # sanity: fixture actually built components
    assert run_deterministic_checks(cleared).services_without_migration_disposition == []


# ── 6. false negative — a capability mention is not an existing service ───
#
# Manual code review of invariant B: `key <= repo_tokens` treated ANY
# identity-token subset of the repo's prose blob (overview + partition
# name/role + languages + frameworks) as proof the standalone service
# already existed. "The legacy monolith handles payments" made
# `_component_key("Payment Service") == ("payment",)` a subset of that
# blob, so a brand-new "Payment Service" with NO migration step and NO
# disposition silently passed — the exact class of silent introduction
# this invariant exists to catch.


def test_capability_mention_does_not_excuse_a_new_service_from_disposition():
    """Reproduces the bug: repo prose mentions the CAPABILITY ('payments'),
    the target is a STANDALONE SERVICE ('Payment Service') the repo never
    names, and there is no migration step or disposition for it."""
    state = _migration_state(
        services=["Payment Service"],
        repo_overview=(
            "The legacy monolith handles orders, payments, and customer "
            "accounts."
        ),
    )
    checks = run_deterministic_checks(state)
    assert checks.services_without_migration_disposition == ["Payment Service"]
    assert any(
        "Payment Service" in issue.finding
        and "migration disposition" in issue.finding
        and issue.severity == "high"
        for issue in checks.issues
    )


def test_capability_mention_false_negative_is_not_payment_specific():
    """Same bug shape, a different capability/service pair — proves the fix
    is generic, not a special case for 'payment'."""
    state = _migration_state(
        services=["Order Service"],
        repo_overview="The legacy monolith handles orders and shipping.",
    )
    checks = run_deterministic_checks(state)
    assert checks.services_without_migration_disposition == ["Order Service"]


def test_generic_capability_mention_alone_never_counts_as_existing():
    """Sweeping the shape across several capability/service pairs: none of
    them may be excused by prose mentioning the underlying capability."""
    for capability, service in [
        ("catalog", "Catalog Service"),
        ("users", "User Service"),
        ("shipping", "Shipping Service"),
    ]:
        state = _migration_state(
            services=[service],
            repo_overview=f"The legacy monolith handles {capability} today.",
        )
        checks = run_deterministic_checks(state)
        assert checks.services_without_migration_disposition == [service], capability


def test_explicit_already_existing_still_passes_after_the_fix():
    state = _migration_state(
        services=["Payment Service"],
        repo_overview="The legacy monolith handles orders and payments.",
        open_risks=["The Payment Service already exists and is unchanged."],
    )
    checks = run_deterministic_checks(state)
    assert checks.services_without_migration_disposition == []


def test_explicit_external_still_passes_after_the_fix():
    state = _migration_state(
        services=["Payment Service"],
        repo_overview="The legacy monolith handles orders and payments.",
        open_risks=["The Payment Service is a third-party provider."],
    )
    checks = run_deterministic_checks(state)
    assert checks.services_without_migration_disposition == []


def test_explicit_retained_or_deferred_still_pass_after_the_fix():
    retained = _migration_state(
        services=["Payment Service"],
        repo_overview="The legacy monolith handles orders and payments.",
        open_risks=["The Payment Service remains in the legacy monolith for now."],
    )
    assert run_deterministic_checks(retained).services_without_migration_disposition == []

    deferred = _migration_state(
        services=["Payment Service"],
        repo_overview="The legacy monolith handles orders and payments.",
        open_risks=["The Payment Service is deferred to a later phase."],
    )
    assert run_deterministic_checks(deferred).services_without_migration_disposition == []


def test_named_migration_step_still_passes_after_the_fix():
    state = _migration_state(
        services=["Payment Service"],
        repo_overview="The legacy monolith handles orders and payments.",
        migration_steps=[MigrationStep(
            title="Extract payments",
            objective="Extract the Payment Service from the monolith.",
        )],
    )
    checks = run_deterministic_checks(state)
    assert checks.services_without_migration_disposition == []


def test_greenfield_still_does_not_trigger_disposition_logic_after_the_fix():
    state = _migration_state(services=["Payment Service"], repo_representation=False)
    checks = run_deterministic_checks(state)
    assert checks.services_without_migration_disposition == []


def test_database_exclusion_unaffected_by_the_fix():
    state = _migration_state(
        services=[],
        databases=["Payment Database"],
        repo_overview="The legacy monolith handles orders and payments.",
    )
    checks = run_deterministic_checks(state)
    assert checks.services_without_migration_disposition == []


def test_structural_partition_naming_the_service_counts_as_existing():
    """Positive case for strong service-level evidence: the repo's OWN
    structural partition is named for the service, not just prose
    describing a capability it handles."""
    state = _migration_state(
        services=["Payment Service"],
        repo_overview="A modular shop platform.",
        partitions=[PartitionSummary(
            name="Payment Service",
            paths=["services/payment"],
            role="Handles payment capture and refunds.",
        )],
    )
    checks = run_deterministic_checks(state)
    assert checks.services_without_migration_disposition == []


def test_partition_role_prose_alone_does_not_count_as_existing():
    """Negative case: the partition's ROLE text mentions the capability in
    prose, but the partition itself is not named for the service — this
    must NOT be treated as strong service-level evidence."""
    state = _migration_state(
        services=["Payment Service"],
        repo_overview="A modular shop platform.",
        partitions=[PartitionSummary(
            name="Checkout",
            paths=["services/checkout"],
            role="Handles checkout, including triggering payments.",
        )],
    )
    checks = run_deterministic_checks(state)
    assert checks.services_without_migration_disposition == ["Payment Service"]


def test_legacy_monolith_component_is_never_flagged_as_a_new_service():
    """The legacy system itself must never be mistaken for a newly
    introduced target service just because it carries no migration step."""
    state = _migration_state(
        services=["Payment Service"],
        repo_overview="The legacy monolith handles orders and payments.",
        migration_steps=[MigrationStep(
            title="Extract payments",
            objective="Extract the Payment Service from the monolith.",
        )],
    )
    state.components.append(ComponentDescription(
        id="COMP-LEGACY", name="Legacy Monolith", purpose="p", description="d",
        related_feature_ids=["FEAT-001"],
    ))
    checks = run_deterministic_checks(state)
    assert checks.services_without_migration_disposition == []


# ── 7. two remaining migration-disposition false negatives ────────────────
#
# Manual review of the previous fix found it directionally correct but
# still too permissive in two ways: (A) ANY retention verb anywhere in a
# sentence ("retains", "keeps ... unchanged") was accepted as a migration
# disposition even when the sentence never places the service in the
# legacy system; (B) a repo partition named only for the CAPABILITY it
# serves ("Payment") satisfied the identity of the STANDALONE SERVICE
# ("Payment Service") via subset-token containment.


def _undisposed(state):
    return run_deterministic_checks(state).services_without_migration_disposition


# -- A: generic retention prose must NOT dispose a service ----------------


def test_generic_retains_prose_does_not_dispose_a_service():
    state = _migration_state(
        services=["Payment Service"],
        open_risks=["Payment Service retains PCI controls during migration."],
    )
    assert _undisposed(state) == ["Payment Service"]


def test_generic_keeps_unchanged_prose_does_not_dispose_a_service():
    state = _migration_state(
        services=["Payment Service"],
        open_risks=["Payment Service keeps its API unchanged."],
    )
    assert _undisposed(state) == ["Payment Service"]


def test_remains_compatible_with_legacy_clients_does_not_dispose_a_service():
    """'Legacy' merely co-occurring with a retention verb is not enough —
    this sentence never says the SERVICE remains IN the legacy system."""
    state = _migration_state(
        services=["Payment Service"],
        open_risks=["Payment Service remains compatible with legacy clients."],
    )
    assert _undisposed(state) == ["Payment Service"]


def test_remains_in_the_legacy_monolith_still_disposes_a_service():
    state = _migration_state(
        services=["Payment Service"],
        open_risks=["Payment Service remains in the legacy monolith for now."],
    )
    assert _undisposed(state) == []


def test_deferred_to_a_later_phase_still_disposes_a_service():
    state = _migration_state(
        services=["Payment Service"],
        open_risks=["Payment Service is deferred to a later phase."],
    )
    assert _undisposed(state) == []


def test_third_party_provider_still_disposes_a_service():
    state = _migration_state(
        services=["Payment Service"],
        open_risks=["Payment Service is a third-party provider."],
    )
    assert _undisposed(state) == []


def test_already_exists_prose_still_disposes_a_service():
    state = _migration_state(
        services=["Payment Service"],
        open_risks=["Payment Service already exists and is unchanged."],
    )
    assert _undisposed(state) == []


def test_named_migration_step_still_disposes_a_service_after_fix_a():
    state = _migration_state(
        services=["Payment Service"],
        migration_steps=[MigrationStep(
            title="Extract payments",
            objective="Extract the Payment Service from the monolith.",
        )],
    )
    assert _undisposed(state) == []


# -- B: a bare capability partition name is not service-level identity ----


def test_bare_capability_partition_name_does_not_prove_the_service_exists():
    state = _migration_state(
        services=["Payment Service"],
        repo_overview="A modular shop platform.",
        partitions=[PartitionSummary(
            name="Payment", paths=["payment/"], role="Payment capability.",
        )],
    )
    assert _undisposed(state) == ["Payment Service"]


def test_bare_capability_partition_name_is_not_service_specific():
    state = _migration_state(
        services=["Order Service"],
        repo_overview="A modular shop platform.",
        partitions=[PartitionSummary(
            name="Orders", paths=["orders/"], role="Order capability.",
        )],
    )
    assert _undisposed(state) == ["Order Service"]


def test_prefix_or_superset_partition_name_does_not_satisfy_the_service():
    """A partition named 'Payment Service Adapter' is a different, longer
    identity — it must not be treated as proof 'Payment Service' itself
    already exists as a standalone unit."""
    state = _migration_state(
        services=["Payment Service"],
        repo_overview="A modular shop platform.",
        partitions=[PartitionSummary(
            name="Payment Service Adapter", paths=["adapters/payment/"],
            role="Wraps the payment gateway SDK.",
        )],
    )
    assert _undisposed(state) == ["Payment Service"]
