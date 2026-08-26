"""test_refinement_compliance.py — Task 20: deterministic refinement
compliance hardening.

Real E2E gap: a run using University of Cologne KI Connect / Mistral Small
4 119B 2603 received valid deterministic findings and claimed (via
`revision_note`) that each was fixed, but three of them never changed the
structured field the deterministic check actually reads:

  * an aggregated data-flow endpoint ("Monitoring -> React Frontend,
    GraphQL Gateway, User Identity Service, ...") stayed one unresolvable
    endpoint string instead of becoming one flow per component;
  * a migration disposition was written as prose inside a Component
    Description instead of into `blueprint.migration_steps`;
  * an unauthorized quantitative target ("100%") remained in the
    architecture text while `revision_note` claimed it had been removed.

The fix is NOT a new finding schema and NOT a weaker check — it is:
  1. a conservative, deterministic fan-out/fan-in endpoint expansion at the
     SAME output boundary `canonicalize_data_flow_endpoints` already owns
     (pipeline/agents/architect.py), so the common aggregate-endpoint shape
     never reaches the Reviewer as an unresolved participant at all;
  2. field-actionable deterministic finding text (pipeline/review_checks.py)
     naming the exact structured field a refinement pass must change;
  3. an explicit refinement-prompt contract stating that `revision_note`
     is never evidence a deterministic finding was fixed (it already
     wasn't read by any deterministic check or by the Reviewer's own
     prompt — see `_format_artifacts`'s `exclude={"revision_note"}` in
     agents/reviewer.py — this only makes that fact explicit to the model).

All offline; no LLM calls.
"""
from __future__ import annotations

from pipeline.agents import architect as arch
from pipeline.review_checks import run_deterministic_checks
from pipeline.state import (
    ADR,
    Blueprint,
    ComponentDescription,
    ContextRecord,
    Feature,
    MigrationStep,
    RepoBehavior,
    RepoRepresentation,
    new_run,
)


def _flat(text: str) -> str:
    """Collapse prompt line-wrapping so a phrase check does not depend on
    exactly where a triple-quoted string happens to wrap."""
    return " ".join(text.split())


# ═════════════════════════════════════════════════════════════════════════
# Data flows — aggregate endpoint expansion
# (pipeline/agents/architect.py: canonicalize_data_flow_endpoints /
# _expand_aggregate_endpoint)
# ═════════════════════════════════════════════════════════════════════════

_COMPONENTS = [
    "Monitoring", "React Frontend", "GraphQL Gateway",
    "User Identity Service", "Order Service",
]


def test_one_source_one_target_flow_is_unchanged():
    flows = ["Monitoring -> React Frontend"]
    canonical, resolved = arch.canonicalize_data_flow_endpoints(flows, _COMPONENTS)
    assert canonical == ["Monitoring → React Frontend"]
    assert resolved == 0


def test_decorated_single_endpoint_still_canonicalizes():
    flows = ["Monitoring -> React Frontend (metrics)"]
    canonical, resolved = arch.canonicalize_data_flow_endpoints(flows, _COMPONENTS)
    assert canonical == ["Monitoring → React Frontend"]
    assert resolved == 1


def test_unambiguous_aggregate_target_expands_into_separate_edges():
    """The exact real-run shape: one flow, aggregated target."""
    flows = [
        "Monitoring -> React Frontend, GraphQL Gateway, User Identity Service"
    ]
    canonical, resolved = arch.canonicalize_data_flow_endpoints(flows, _COMPONENTS)
    assert canonical == [
        "Monitoring → React Frontend",
        "Monitoring → GraphQL Gateway",
        "Monitoring → User Identity Service",
    ]
    assert resolved == 3


def test_unambiguous_aggregate_source_expands_equivalently():
    flows = [
        "React Frontend, GraphQL Gateway and User Identity Service -> Monitoring"
    ]
    canonical, resolved = arch.canonicalize_data_flow_endpoints(flows, _COMPONENTS)
    assert canonical == [
        "React Frontend → Monitoring",
        "GraphQL Gateway → Monitoring",
        "User Identity Service → Monitoring",
    ]
    assert resolved == 3


def test_ambiguous_aggregate_with_one_unresolved_candidate_is_not_guessed():
    flows = ["Monitoring -> React Frontend, Some Unknown Thing"]
    canonical, resolved = arch.canonicalize_data_flow_endpoints(flows, _COMPONENTS)
    # Endpoint text is left exactly as written — no partial guess — only
    # the arrow spelling is normalized, same as every other unresolved flow.
    assert canonical == ["Monitoring → React Frontend, Some Unknown Thing"]
    assert resolved == 0


def test_both_sides_being_lists_is_refused_as_a_cartesian_guess():
    flows = [
        "React Frontend, GraphQL Gateway -> User Identity Service, Order Service"
    ]
    canonical, resolved = arch.canonicalize_data_flow_endpoints(flows, _COMPONENTS)
    assert canonical == [
        "React Frontend, GraphQL Gateway → User Identity Service, Order Service"
    ]
    assert resolved == 0


def test_expanded_flows_preserve_the_label():
    flows = ["Monitoring -> React Frontend, GraphQL Gateway: health check"]
    canonical, resolved = arch.canonicalize_data_flow_endpoints(flows, _COMPONENTS)
    assert canonical == [
        "Monitoring → React Frontend: health check",
        "Monitoring → GraphQL Gateway: health check",
    ]
    assert resolved == 2


def test_expansion_duplicate_with_an_existing_flow_is_left_as_is():
    """No new dedup mechanism is introduced — a flow produced by expansion
    that happens to duplicate an already-present flow is left exactly as a
    human/LLM writing that literal duplicate would be: both entries
    present, deterministic, no crash."""
    flows = [
        "Monitoring → React Frontend",  # already present, pre-canonical
        "Monitoring -> React Frontend, GraphQL Gateway",
    ]
    canonical, _resolved = arch.canonicalize_data_flow_endpoints(flows, _COMPONENTS)
    assert canonical == [
        "Monitoring → React Frontend",
        "Monitoring → React Frontend",
        "Monitoring → GraphQL Gateway",
    ]
    assert canonical.count("Monitoring → React Frontend") == 2


def test_exact_component_identity_is_still_enforced_after_expansion():
    """Expansion never invents a spelling — every produced endpoint is the
    EXACT catalog string, copied, never fuzzed."""
    flows = ["Monitoring -> react frontend, GRAPHQL GATEWAY"]
    canonical, _resolved = arch.canonicalize_data_flow_endpoints(flows, _COMPONENTS)
    assert canonical == [
        "Monitoring → React Frontend",
        "Monitoring → GraphQL Gateway",
    ]


def test_three_hop_chain_is_never_expanded_even_with_comma_text():
    """A `->` chain of 3+ hops has different, existing semantics
    (`split_directional_flow`) and must never be reinterpreted as a
    fan-out/fan-in aggregate."""
    flows = ["Client -> React Frontend, GraphQL Gateway -> Monitoring"]
    canonical, resolved = arch.canonicalize_data_flow_endpoints(flows, _COMPONENTS)
    assert canonical == ["Client → React Frontend, GraphQL Gateway → Monitoring"]
    assert resolved == 0


def test_a_genuine_compound_component_name_containing_and_is_not_split():
    """A real single component whose own name contains 'and' must never be
    torn apart — only a side that does NOT already resolve as one whole
    component is ever considered for splitting."""
    catalog = ["Search and Discovery Service"]
    flows = ["API Gateway -> Search and Discovery Service"]
    canonical, resolved = arch.canonicalize_data_flow_endpoints(flows, catalog)
    assert canonical == ["API Gateway → Search and Discovery Service"]
    assert resolved == 0


def test_aggregate_endpoint_that_never_resolves_reaches_the_reviewer_finding_unchanged():
    """When expansion legitimately cannot fire (a genuinely unmatched
    candidate), the deterministic Reviewer still catches the unresolved
    endpoint — this fix narrows WHEN the finding fires, it never removes
    the check."""
    state = new_run("Design a system.")
    state.context_record = ContextRecord(business_goal="g", problem_statement="p")
    state.features = [Feature(id="FEAT-001", name="F", scenario="s", acceptance_criteria=["a"])]
    state.blueprint = Blueprint(
        stakeholder_view="sv", technical_view="tv",
        components=["Monitoring", "React Frontend"],
        addressed_feature_ids=["FEAT-001"],
        data_flows=["Monitoring -> React Frontend, Some Unknown Thing"],
    )
    state.components = [
        ComponentDescription(
            id="COMP-001", name="Monitoring", purpose="p", description="d",
            related_feature_ids=["FEAT-001"],
        ),
        ComponentDescription(
            id="COMP-002", name="React Frontend", purpose="p", description="d",
            related_feature_ids=["FEAT-001"],
        ),
    ]
    checks = run_deterministic_checks(state)

    assert "React Frontend, Some Unknown Thing" in checks.unresolved_flow_participants
    issue = next(
        i for i in checks.issues
        if "React Frontend, Some Unknown Thing" in i.finding
    )
    assert "blueprint.data_flows" in issue.suggested_fix


# ═════════════════════════════════════════════════════════════════════════
# Migration disposition — the fix must land in blueprint.migration_steps
# ═════════════════════════════════════════════════════════════════════════

def _migration_state(*, migration_steps=(), description_text="d"):
    state = new_run("Modernize a monolithic shop.")
    state.context_record = ContextRecord(
        project_name="P", business_goal="g", problem_statement="p",
        functional_requirements=["orders"],
    )
    state.repo_representation = RepoRepresentation(
        behavior=RepoBehavior(overview="Existing shop platform handling checkout."),
    )
    state.features = [Feature(id="FEAT-001", name="Orders", scenario="s", acceptance_criteria=["a"])]
    state.blueprint = Blueprint(
        project_name="P", stakeholder_view="sv", technical_view="tv",
        components=["Transactional Outbox Relay"],
        addressed_feature_ids=["FEAT-001"],
        migration_steps=list(migration_steps),
    )
    state.components = [
        ComponentDescription(
            id="COMP-001", name="Transactional Outbox Relay", purpose="p",
            description=description_text, related_feature_ids=["FEAT-001"],
        ),
    ]
    return state


def test_component_description_prose_does_not_satisfy_migration_coverage():
    """The exact real-run failure shape: the model wrote the disposition
    into the component's OWN description text instead of a migration step.
    The literal wording does not even match the check's own disposition-
    phrase vocabulary, so this genuinely stays a finding — proving the
    field, not the wording, was the defect."""
    state = _migration_state(
        migration_steps=[],
        description_text=(
            "Migration disposition: extracted in named migration step as "
            "part of the checkout modernization."
        ),
    )
    checks = run_deterministic_checks(state)
    assert checks.services_without_migration_disposition == ["Transactional Outbox Relay"]


def test_actual_migration_steps_entry_satisfies_the_coverage():
    state = _migration_state(
        migration_steps=[MigrationStep(
            title="Extract the Transactional Outbox Relay",
            objective="Extract the Transactional Outbox Relay from the monolith.",
        )],
    )
    checks = run_deterministic_checks(state)
    assert checks.services_without_migration_disposition == []


def test_migration_finding_names_the_actual_field_and_exact_component():
    state = _migration_state(migration_steps=[])
    checks = run_deterministic_checks(state)
    issue = next(
        i for i in checks.issues
        if "Transactional Outbox Relay" in i.finding
        and "migration disposition" in i.finding
    )
    assert "blueprint.migration_steps" in issue.suggested_fix
    assert "Transactional Outbox Relay" in issue.suggested_fix
    assert "component's own description" in issue.suggested_fix
    assert issue.severity == "high"
    assert issue.requires_refinement is True


# ═════════════════════════════════════════════════════════════════════════
# Unauthorized quantitative target — revision_note is not evidence
# ═════════════════════════════════════════════════════════════════════════

NO_NUMERIC_TARGETS_CONTEXT = ContextRecord(
    project_name="P", business_goal="Stay responsive under peak load.",
    problem_statement="No exact numeric target has been agreed yet.",
    functional_requirements=["orders"],
)


def _quantitative_state(*, technical_view, revision_note=""):
    state = new_run("Design a system.")
    state.context_record = NO_NUMERIC_TARGETS_CONTEXT
    state.features = [Feature(id="FEAT-001", name="F", scenario="s", acceptance_criteria=["a"])]
    state.blueprint = Blueprint(
        stakeholder_view="Peak sales keep working.",
        technical_view=technical_view,
        addressed_feature_ids=["FEAT-001"],
        components=["App Instance"],
        revision_note=revision_note,
    )
    state.adrs = [ADR(
        id="ADR-001", title="ADR-1: Scale out", context="c", decision="d",
        rationale="r", related_feature_ids=["FEAT-001"],
        related_component_names=["App Instance"],
    )]
    state.components = [ComponentDescription(
        id="COMP-001", name="App Instance", purpose="p", description="d",
        related_feature_ids=["FEAT-001"],
    )]
    return state


def test_unauthorized_literal_remains_a_finding_even_when_revision_note_claims_removal():
    """revision_note says the figure was removed; the actual field still
    carries it. The deterministic check must not be fooled — it never
    reads revision_note in the first place."""
    state = _quantitative_state(
        technical_view="The gateway guarantees 100% availability under load.",
        revision_note="Removed the invented 100% availability target as requested.",
    )
    checks = run_deterministic_checks(state)

    assert "Blueprint.technical_view" in checks.invented_quantitative_targets
    assert "100%" in checks.invented_quantitative_targets["Blueprint.technical_view"]
    assert any(
        issue.category == "grounding" and issue.severity == "high"
        for issue in checks.issues
    )


def test_removing_the_literal_from_the_actual_field_resolves_the_finding():
    state = _quantitative_state(
        technical_view="The gateway scales horizontally to stay responsive under load.",
        revision_note="Removed the invented availability target.",
    )
    checks = run_deterministic_checks(state)

    assert checks.invented_quantitative_targets == {}
    assert not any(issue.category == "grounding" for issue in checks.issues)


def test_explicitly_approved_quantitative_requirement_remains_allowed():
    approved_context = ContextRecord(
        project_name="P", business_goal="g",
        problem_statement="p",
        non_functional_requirements=["Must sustain 99.9% availability."],
    )
    state = _quantitative_state(
        technical_view="The gateway is designed to sustain 99.9% availability.",
    )
    state.context_record = approved_context
    checks = run_deterministic_checks(state)

    assert checks.invented_quantitative_targets == {}


def test_quantitative_finding_points_to_actual_field_not_revision_note():
    state = _quantitative_state(
        technical_view="The gateway guarantees 100% availability under load.",
        revision_note="Removed the invented 100% availability target as requested.",
    )
    checks = run_deterministic_checks(state)
    issue = next(i for i in checks.issues if i.category == "grounding")
    assert "revision_note" in issue.suggested_fix
    assert "field(s) named above" in issue.suggested_fix


# ═════════════════════════════════════════════════════════════════════════
# Refinement prompt — the structured-field compliance contract
# ═════════════════════════════════════════════════════════════════════════

def test_refinement_discipline_states_findings_require_structured_field_changes():
    discipline = _flat(arch._REFINEMENT_DISCIPLINE)
    assert "resolved ONLY by changing the structured field it names" in discipline


def test_refinement_discipline_states_revision_note_alone_cannot_satisfy_a_finding():
    discipline = _flat(arch._REFINEMENT_DISCIPLINE)
    assert "revision_note" in discipline
    assert "deterministic checks never read it" in discipline


def test_refinement_discipline_names_migration_steps_field():
    discipline = _flat(arch._REFINEMENT_DISCIPLINE)
    assert "blueprint.migration_steps" in discipline
    assert "Component Description's own text does not satisfy it" in discipline


def test_refinement_discipline_names_data_flows_one_source_one_target():
    discipline = _flat(arch._REFINEMENT_DISCIPLINE)
    assert "blueprint.data_flows" in discipline
    assert "exactly one source and one target" in discipline


def test_refinement_discipline_quantitative_instruction_points_at_actual_content():
    discipline = _flat(arch._REFINEMENT_DISCIPLINE)
    assert "removing or replacing the exact literal in the field the finding names" in discipline


def test_data_flow_fidelity_prompt_forbids_multi_endpoint_strings():
    prompt = _flat(arch.ARCHITECTURE_SYSTEM_PROMPT)
    assert "One flow entry is ONE edge with exactly one source and one target" in prompt
    assert "Never encode multiple targets or sources into a single endpoint string" in prompt


def test_refinement_discipline_only_appears_on_a_refinement_pass():
    """Unchanged existing behavior: the compliance contract is appended
    only when there is a current design to revise."""
    state = new_run("Design a system.")
    state.context_record = ContextRecord(business_goal="g", problem_statement="p")
    state.features = [Feature(id="FEAT-001", name="F", scenario="s", acceptance_criteria=["a"])]
    prompt = arch._build_architecture_prompt(state, state.features)
    assert "<refinement_discipline>" not in prompt
