"""test_reviewer_e2e_consistency.py — the two final Reviewer blind spots a
fresh E2E run exposed (REVIEW PASSED / 0 findings with concrete
inconsistencies still in the artifacts):

  1. a migration step promised "Extract Order, Inventory, and Shipping
     Services" while no Shipping Service existed in the component catalog;
  2. `data_flows` were prose ("API Gateway routes incoming traffic to
     ..."), which the diagram renderer cannot draw — the client-facing
     Overview showed recorded text instead of a target architecture.

Both are now deterministic HIGH invariants in `pipeline.review_checks`,
sharing the ONE flow grammar (`pipeline.flow_syntax`) and the ONE canonical
component identity (`_component_key`) with the existing checks. All
fixtures are offline; no LLM, no network.
"""

from __future__ import annotations

from pipeline.flow_syntax import split_directional_flow
from pipeline.review_checks import run_deterministic_checks
from pipeline.state import (
    ADR,
    Blueprint,
    ComponentDescription,
    ContextRecord,
    Feature,
    MigrationStep,
    new_run,
)


def _component(name, **kwargs):
    return ComponentDescription(
        id=kwargs.pop("id", f"COMP-{abs(hash(name)) % 1000:03d}"),
        name=name,
        purpose=kwargs.pop("purpose", f"Owns the {name} responsibility."),
        description=kwargs.pop("description", f"Describes the {name}."),
        **kwargs,
    )


def _base_state(components, flows, steps=(), extra_sentences=()):
    """A compact otherwise-clean design: no stated constraints (all
    constraint groups inapplicable), traceability linked, so the ONLY
    findings can come from the invariant under test."""
    state = new_run("Re-architect the monolith for peak traffic.")
    state.context_record = ContextRecord(
        project_name="Peak Shop",
        business_goal="Survive peaks.",
        summary="brownfield modernization",
    )
    state.features = [
        Feature(
            id="FEAT-001", name="Checkout",
            description="Checkout works at peak.",
            scenario="A customer checks out under load.",
            acceptance_criteria=["Checkout completes."],
        )
    ]
    component_names = [
        c.name if isinstance(c, ComponentDescription) else c for c in components
    ]
    state.blueprint = Blueprint(
        stakeholder_view="Peak-season shopping keeps working.",
        technical_view=" ".join(
            ["Services are extracted incrementally.", *extra_sentences]
        ),
        components=component_names,
        data_flows=list(flows),
        migration_steps=list(steps),
        addressed_feature_ids=["FEAT-001"],
    )
    state.adrs = [
        ADR(
            id="ADR-001",
            title="ADR-1: Extract incrementally",
            context=" ".join(["The monolith saturates at peak.", *extra_sentences]),
            decision=" ".join(
                ["Extract services behind a routing seam.", *extra_sentences]
            ),
            rationale="Incremental extraction limits risk.",
            alternatives_considered=["Big-bang rewrite"],
            positive_consequences=["Independent scaling."],
            negative_consequences=["Operational complexity."],
            related_feature_ids=["FEAT-001"],
            related_component_names=component_names[:2],
        )
    ]
    state.components = [
        c if isinstance(c, ComponentDescription) else _component(c)
        for c in components
    ]
    for component in state.components:
        component.related_feature_ids = ["FEAT-001"]
        component.related_adr_ids = ["ADR-001"]
    return state


# ═════════════════════════════════════════════════════════════════════════
# Invariant 1 — migration targets must resolve to real components
# ═════════════════════════════════════════════════════════════════════════


def test_e2e_shape_missing_shipping_service_is_blocking():
    """THE real failure: the coordinated step wording with Order and
    Inventory described but Shipping absent — Shipping must be named, with
    the step that promised it."""
    state = _base_state(
        components=["API Gateway", "Legacy Monolith", "Order Service",
                    "Inventory Service"],
        flows=["API Gateway → Order Service: checkout traffic"],
        steps=[
            MigrationStep(
                title="Introduce the routing seam",
                objective="Route all traffic through the gateway first.",
            ),
            MigrationStep(
                title="Extract Order, Inventory, and Shipping Services",
                objective="Extract the three peak-load domains next.",
            ),
        ],
    )

    checks = run_deterministic_checks(state)

    assert checks.unresolved_migration_targets == {
        "Shipping Service": ["Extract Order, Inventory, and Shipping Services"],
    }
    assert any(
        issue.severity == "high"
        and "Shipping Service" in issue.finding
        and "Extract Order, Inventory, and Shipping Services" in issue.evidence
        for issue in checks.issues
    )
    # Order and Inventory resolve — only the absent service is flagged.
    assert "Order Service" not in checks.unresolved_migration_targets
    assert "Inventory Service" not in checks.unresolved_migration_targets


def test_same_step_with_every_target_described_is_clean():
    state = _base_state(
        components=["API Gateway", "Legacy Monolith", "Order Service",
                    "Inventory Service", "Shipping Service"],
        flows=["API Gateway → Order Service: checkout traffic"],
        steps=[
            MigrationStep(
                title="Extract Order, Inventory, and Shipping Services",
                objective="Extract the three peak-load domains next.",
            ),
        ],
    )

    checks = run_deterministic_checks(state)

    assert checks.unresolved_migration_targets == {}
    assert not any(
        "migration step" in issue.finding.lower() for issue in checks.issues
    )


def test_singular_introduction_wording_is_caught_too():
    state = _base_state(
        components=["API Gateway", "Legacy Monolith", "Order Service"],
        flows=["API Gateway → Order Service: checkout traffic"],
        steps=[
            MigrationStep(
                title="Deploy the Payment Service",
                objective="Stand payments up as its own deployable.",
            ),
        ],
    )

    checks = run_deterministic_checks(state)

    assert checks.unresolved_migration_targets == {
        "Payment Service": ["Deploy the Payment Service"],
    }


def test_deferred_target_is_not_a_false_positive():
    """A migration step may legitimately name a future service — as long as
    the design SAYS it is deferred, the existing disposition vocabulary
    excuses it."""
    state = _base_state(
        components=["API Gateway", "Legacy Monolith", "Order Service"],
        flows=["API Gateway → Order Service: checkout traffic"],
        steps=[
            MigrationStep(
                title="Extract the Reporting Service",
                objective="Reporting follows once order flow is stable.",
            ),
        ],
        extra_sentences=[
            "The Reporting Service is deferred to a later phase."
        ],
    )

    checks = run_deterministic_checks(state)

    assert checks.unresolved_migration_targets == {}


def test_legacy_retained_target_is_not_a_false_positive():
    state = _base_state(
        components=["API Gateway", "Legacy Monolith", "Order Service"],
        flows=["API Gateway → Order Service: checkout traffic"],
        steps=[
            MigrationStep(
                title="Extract the Order Service",
                objective="Order leaves the monolith; the Cart Service "
                "remains in the legacy monolith for now.",
            ),
        ],
    )

    checks = run_deterministic_checks(state)

    assert checks.unresolved_migration_targets == {}


def test_incidental_prose_mentions_are_not_introductions():
    """No introduction verb, or no '<Name> Service' grammar — never a
    candidate. 'Move shipping logic' is capability prose; 'Monitor the
    Ghost Service dashboards' names a service without committing to
    creating one."""
    state = _base_state(
        components=["API Gateway", "Legacy Monolith", "Order Service"],
        flows=["API Gateway → Order Service: checkout traffic"],
        steps=[
            MigrationStep(
                title="Move shipping logic into the extracted services",
                objective="Consolidate the shipping module's helpers.",
            ),
            MigrationStep(
                title="Monitor the Ghost Service dashboards",
                objective="Observe the existing tooling during migration.",
            ),
        ],
    )

    checks = run_deterministic_checks(state)

    assert checks.unresolved_migration_targets == {}


def test_composite_declaration_does_not_cover_a_half_reference():
    """The composite exactness rule travels with this check: declaring
    'Order/Payment Service' satisfies an introduction of the COMPOSITE, not
    of the standalone 'Payment Service' half."""
    state = _base_state(
        components=["API Gateway", "Order/Payment Service"],
        flows=["API Gateway → Order/Payment Service: checkout"],
        steps=[
            MigrationStep(
                title="Extract the Payment Service",
                objective="Payments scale independently.",
            ),
        ],
    )

    checks = run_deterministic_checks(state)

    assert list(checks.unresolved_migration_targets) == ["Payment Service"]


def test_qualified_component_name_satisfies_the_step_reference():
    """A lifecycle-qualified declared component ('Shipping Service (New)')
    satisfies the step's bare 'Shipping Services' reference — the shared
    canonical identity ignores display qualifiers."""
    state = _base_state(
        components=["API Gateway", "Shipping Service (New)"],
        flows=["API Gateway → Shipping Service: shipping updates"],
        steps=[
            MigrationStep(
                title="Extract Shipping Services",
                objective="Shipping scales independently.",
            ),
        ],
    )

    checks = run_deterministic_checks(state)

    assert checks.unresolved_migration_targets == {}


# ═════════════════════════════════════════════════════════════════════════
# Invariant 2 — every non-empty flow must be directionally renderable
# ═════════════════════════════════════════════════════════════════════════


def test_e2e_shape_prose_flow_is_unrenderable_and_blocking():
    """THE real failure: prose 'routes incoming traffic to' carries no
    arrow, so the diagram has no edge — flagged whole, verbatim."""
    state = _base_state(
        components=["API Gateway", "Order Service"],
        flows=[
            "API Gateway routes incoming traffic to the Order Service "
            "during the migration."
        ],
    )

    checks = run_deterministic_checks(state)

    assert checks.unrenderable_data_flows == [
        "API Gateway routes incoming traffic to the Order Service during "
        "the migration."
    ]
    assert any(
        issue.severity == "high"
        and "directional" in issue.finding
        and "A → B" in issue.suggested_fix
        for issue in checks.issues
    )


def test_every_renderer_syntax_stays_valid():
    """The invariant uses the renderer's ACTUAL grammar: both arrow
    spellings, chains, and ':' labels all parse — no flow the diagram can
    draw is ever flagged."""
    flows = [
        "API Gateway → Order Service",
        "API Gateway -> Order Service",
        "API Gateway → Shared Event Bus → Order Service: fan-out",
        "Order Service → PostgreSQL: order records",
    ]
    for flow in flows:
        assert split_directional_flow(flow) is not None, flow

    state = _base_state(
        components=["API Gateway", "Order Service", "Shared Event Bus"],
        flows=flows,
    )
    checks = run_deterministic_checks(state)

    assert checks.unrenderable_data_flows == []
    # PostgreSQL is an external infrastructure endpoint — also no
    # participant finding for it.
    assert checks.unresolved_flow_participants == {}


def test_parseable_flow_with_unknown_endpoint_still_caught_by_participants():
    """Directionality and participant resolution stay layered: a
    well-formed edge to a nonexistent component passes the NEW check and
    is caught by the EXISTING one."""
    state = _base_state(
        components=["API Gateway", "Order Service"],
        flows=["Order Service → Ghost Queue: audit events"],
    )

    checks = run_deterministic_checks(state)

    assert checks.unrenderable_data_flows == []
    assert "Ghost Queue" in checks.unresolved_flow_participants


def test_empty_flow_entries_are_ignored_entirely():
    state = _base_state(
        components=["API Gateway", "Order Service"],
        flows=["API Gateway → Order Service: traffic", "", "   "],
    )

    checks = run_deterministic_checks(state)

    assert checks.unrenderable_data_flows == []


def test_greenfield_without_migration_steps_is_untouched():
    state = _base_state(
        components=["API Gateway", "Order Service"],
        flows=["API Gateway → Order Service: traffic"],
    )
    state.repo_representation = None

    checks = run_deterministic_checks(state)

    assert checks.unresolved_migration_targets == {}
    assert checks.unrenderable_data_flows == []
    assert checks.services_without_migration_disposition == []


# ═════════════════════════════════════════════════════════════════════════
# The shared grammar itself — renderer and Reviewer read the same parser
# ═════════════════════════════════════════════════════════════════════════


def test_renderer_parse_data_flows_uses_the_shared_grammar():
    from ui_sections import parse_data_flows

    edges, unparsed = parse_data_flows(
        [
            "A → B → C: chain label",
            "A -> B",
            "prose without an arrow",
            "",
        ]
    )

    assert edges == [
        ("A", "B", "chain label"),
        ("B", "C", "chain label"),
        ("A", "B", ""),
    ]
    assert unparsed == ["prose without an arrow"]


# ═════════════════════════════════════════════════════════════════════════
# Regression — migration action verbs are structural syntax, not names
#
# A fresh E2E after the migration-target hardening produced a phantom
# finding: the step title "Decompose Catalog and Review Services" was
# parsed into a 'Decompose Service' candidate because 'decompose' was
# missing from the never-a-name vocabulary while the REAL targets
# (Catalog Service, Review Service) were described components.
# ═════════════════════════════════════════════════════════════════════════


def test_parser_never_yields_an_action_verb_as_a_service_base():
    from pipeline.review_checks import _extract_service_bases

    # The exact fresh-E2E wordings plus every inflected action verb.
    for text, expected in (
        ("Decompose Catalog and Review Services", ["Catalog", "Review"]),
        ("Decompose Order, Payment, and Inventory", []),
        ("Extract Order, Inventory, and Shipping Services",
         ["Inventory", "Order", "Shipping"]),
        ("Establish the Payment Service gateway", ["Payment"]),
        ("Deploying the Reporting Service", ["Reporting"]),
        ("Building and launching New Services", []),
    ):
        assert _extract_service_bases(text) == expected, text


def test_action_verb_vocabulary_is_shared_by_matcher_and_blocklist():
    """One source of truth: every introduction-matcher verb is also a
    never-a-name word, so the two can never drift apart again."""
    from pipeline.review_checks import (
        _MIGRATION_ACTION_VERBS,
        _MIGRATION_INTRODUCTION_RE,
        _NON_NAME_WORDS,
    )

    for verb in _MIGRATION_ACTION_VERBS:
        assert verb in _NON_NAME_WORDS, verb
        assert _MIGRATION_INTRODUCTION_RE.search(f"we will {verb} it"), verb
    assert "decompose" in _MIGRATION_ACTION_VERBS  # the regression itself


def test_decompose_step_with_all_targets_described_is_clean():
    """The exact E2E shape: Catalog and Review Services described in the
    catalog — no finding at all for the Decompose step."""
    state = _base_state(
        components=["API Gateway", "Catalog Service", "Review Service",
                    "Order Management Service", "Payment Service",
                    "Inventory Service"],
        flows=["API Gateway → Catalog Service: browsing",
               "API Gateway → Order Management Service: checkout"],
        steps=[
            MigrationStep(
                title="Decompose Catalog and Review Services",
                objective="The two read-heavy domains scale independently.",
            ),
            MigrationStep(
                title="Decompose Order, Payment, and Inventory",
                objective="The write-heavy domains follow in phase two.",
            ),
        ],
    )

    checks = run_deterministic_checks(state)

    assert checks.unresolved_migration_targets == {}
    assert "Decompose Service" not in str(checks.model_dump_json())


def test_decompose_step_with_a_missing_target_still_flags_only_that():
    """The verb fix must not blunt the check: 'Decompose Catalog and
    Review Services' with Review ABSENT still flags Review Service —
    and never Decompose Service."""
    state = _base_state(
        components=["API Gateway", "Catalog Service", "Order Management Service"],
        flows=["API Gateway → Catalog Service: browsing"],
        steps=[
            MigrationStep(
                title="Decompose Catalog and Review Services",
                objective="The two read-heavy domains scale independently.",
            ),
        ],
    )

    checks = run_deterministic_checks(state)

    assert checks.unresolved_migration_targets == {
        "Review Service": ["Decompose Catalog and Review Services"],
    }
