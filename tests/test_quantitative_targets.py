"""test_quantitative_targets.py — invented quantitative architecture
targets (Kush integration, Part 1 of the final grounding-hardening pass).

Real E2E gap: the Architect prompt has always forbidden inventing numeric
SLOs/latency/availability/scale figures, but that rule was prompt-only.
Gemini generated Feature text such as "Product details load in under
200ms", "10x traffic", "99.9% uptime", and "10k concurrent users" against a
Context Record that explicitly stated no numeric target was defined, and
none of it was caught deterministically — it survived refinement into a
final PASS.

Covers the ONE shared detector (`review_checks.find_unauthorized_quantitative_targets`),
the fail-fast Feature-generation gate (`architect._validate_no_invented_quantitative_targets`,
called BEFORE phase 2 ever runs so an invented number can never become
frozen, unfixable context for later design), and the defense-in-depth
deterministic Reviewer scan of Blueprint/ADR/Component/migration-step prose
(`review_checks._check_quantitative_targets`).

All offline: LLM calls are stubbed with canned structured outputs.
"""

from __future__ import annotations

import pytest

import test_clarifier as tc
from pipeline.agents import architect as arch
from pipeline.review_checks import (
    find_unauthorized_quantitative_targets,
    run_deterministic_checks,
)
from pipeline.state import (
    ADR,
    Blueprint,
    ComponentDescription,
    ContextRecord,
    Feature,
    MigrationStep,
    Stage,
    new_run,
)

# A locked Context Record that is explicit: performance/scale matters, but
# no exact figure was ever approved.
NO_NUMERIC_TARGETS_CONTEXT = ContextRecord(
    project_name="Shop",
    business_goal="Handle peak sales events without degrading.",
    problem_statement="The system falls over during peak sales events.",
    functional_requirements=["Process customer orders"],
    non_functional_requirements=[
        "Handle substantially higher peak traffic without degradation, "
        "maintain high availability during peak periods, and keep "
        "customer-facing interactions responsive. No specific numeric SLA, "
        "latency, or throughput targets are defined.",
    ],
)

AUTHORIZED_AVAILABILITY_CONTEXT = ContextRecord(
    project_name="Shop",
    business_goal="Handle peak sales events without degrading.",
    problem_statement="The system falls over during peak sales events.",
    functional_requirements=["Process customer orders"],
    non_functional_requirements=["Maintain 99.9% availability."],
)


# ── the shared detector: unauthorized vs authorized ─────────────────────

def test_unapproved_latency_target_is_detected():
    found = find_unauthorized_quantitative_targets(
        "Product details load in under 200ms.", NO_NUMERIC_TARGETS_CONTEXT
    )
    assert found == ["200ms"]


def test_unapproved_availability_percentage_is_detected():
    found = find_unauthorized_quantitative_targets(
        "System uptime remains at 99.9% during peak sales events.",
        NO_NUMERIC_TARGETS_CONTEXT,
    )
    assert found == ["99.9%"]


def test_unapproved_multiplier_claim_is_detected():
    found = find_unauthorized_quantitative_targets(
        "During a peak sales event with 10x traffic, the system stays up.",
        NO_NUMERIC_TARGETS_CONTEXT,
    )
    assert found == ["10x"]


def test_unapproved_concurrency_target_is_detected():
    found = find_unauthorized_quantitative_targets(
        "The system supports under 10k concurrent users.",
        NO_NUMERIC_TARGETS_CONTEXT,
    )
    assert found == ["10k concurrent users"]


def test_authorized_availability_figure_is_not_flagged():
    """The Context Record itself states 99.9% availability — preserving
    that EXACT figure must not be treated as an invention."""
    found = find_unauthorized_quantitative_targets(
        "The design targets 99.9% availability.",
        AUTHORIZED_AVAILABILITY_CONTEXT,
    )
    assert found == []


def test_authorized_figure_does_not_authorize_a_different_one():
    """Approving 99.9% does not blanket-approve every number."""
    found = find_unauthorized_quantitative_targets(
        "The design targets 99.99% availability and responds within 50ms.",
        AUTHORIZED_AVAILABILITY_CONTEXT,
    )
    assert "99.99%" in found
    assert "50ms" in found


# ── non-target numeric literals must never false-positive ───────────────

@pytest.mark.parametrize(
    "text",
    [
        "See ADR-001 for the rationale.",
        "Cited from source page 12.",
        "Running version v2 of the API contract.",
        "The service listens on port 8080.",
        "The database is named orders_db_2.",
        "Retry the operation up to 3 times before failing.",
        "HTTP 5xx errors are logged and alerted on.",
    ],
)
def test_non_target_numeric_literals_are_not_flagged(text):
    assert find_unauthorized_quantitative_targets(text, NO_NUMERIC_TARGETS_CONTEXT) == []


def test_empty_text_and_no_context_are_safe():
    assert find_unauthorized_quantitative_targets("", NO_NUMERIC_TARGETS_CONTEXT) == []
    assert find_unauthorized_quantitative_targets("under 200ms", None) == ["200ms"]


# ── fail-fast at Feature generation, before phase 2 ever runs ───────────


class _InventedTargetFeatureStub:
    """Phase-1-only stub: returns a Feature carrying an invented numeric
    target. Phase 2 must never be reached — asserted via `phase2_calls`."""

    def __init__(self, scenario="Loads in under 200ms.", acceptance_criteria=None):
        self.scenario = scenario
        self.acceptance_criteria = acceptance_criteria or ["Stays available."]
        self.phase2_calls = 0

    def __call__(self, state, prompt, *, system="", model="", response_schema=None, thinking_level=None):
        if response_schema is arch.FeatureDesign:
            return arch.FeatureDesign(features=[
                Feature(
                    id="FEAT-001",
                    name="Fast product page",
                    scenario=self.scenario,
                    related_requirement_ids=["FR-001", "NFR-001"],
                    acceptance_criteria=self.acceptance_criteria,
                )
            ]), tc.fake_usage()
        self.phase2_calls += 1
        raise AssertionError("phase 2 must not run when phase 1 invented a target")


def _state():
    state = new_run("Design a system.")
    state.context_record = NO_NUMERIC_TARGETS_CONTEXT
    return state


def test_invented_scenario_target_blocks_before_phase_two(monkeypatch):
    """`@node("architect")` turns the fail-fast ValueError into a clean
    FAILED step (pipeline/agents/base.py `node`), never a crash — so the
    node returns normally with an error update, and phase 2 is never
    reached (proven by `stub.phase2_calls == 0`)."""
    stub = _InventedTargetFeatureStub(scenario="Product details load in under 200ms.")
    monkeypatch.setattr(arch, "llm_call", stub)
    state = _state()

    update = arch.architect_node(state)

    assert update["stage"] == Stage.FAILED
    assert "quantitative target" in update["errors"][0]
    assert "200ms" in update["errors"][0]
    assert stub.phase2_calls == 0


def test_invented_acceptance_criterion_target_blocks_before_phase_two(monkeypatch):
    stub = _InventedTargetFeatureStub(
        scenario="Product page is fast.",
        acceptance_criteria=["System uptime remains at 99.9%."],
    )
    monkeypatch.setattr(arch, "llm_call", stub)
    state = _state()

    update = arch.architect_node(state)

    assert update["stage"] == Stage.FAILED
    assert "quantitative target" in update["errors"][0]
    assert "99.9%" in update["errors"][0]
    assert stub.phase2_calls == 0


def test_qualitative_feature_text_passes_through(monkeypatch):
    """No invented number at all — phase 2 runs normally."""
    stub = _InventedTargetFeatureStub(
        scenario="Product page stays responsive under peak load.",
        acceptance_criteria=["Remains available during a peak sales event."],
    )

    def _phase2(state, prompt, *, system="", model="", response_schema=None, thinking_level=None):
        if response_schema is arch.FeatureDesign:
            return stub(state, prompt, system=system, model=model, response_schema=response_schema)
        return arch.ArchitectureDesign(
            blueprint=Blueprint(
                project_name="Shop",
                selected_pattern="Modular monolith",
                stakeholder_view="Peak sales keep working.",
                technical_view="One deployable unit scales horizontally.",
                addressed_feature_ids=["FEAT-001"],
            ),
            adrs=[ADR(
                id="ADR-001",
                title="ADR-1: Scale horizontally under peak load",
                context="Peak sales overwhelm a single instance.",
                decision="Run multiple stateless instances behind a load balancer.",
                rationale="Requirements demand resilience to peak traffic.",
                alternatives_considered=["Vertical scaling"],
                positive_consequences=["Peak resilience."],
                negative_consequences=["Operational complexity."],
                related_feature_ids=["FEAT-001"],
                related_component_names=["App Instance"],
            )],
            components=[ComponentDescription(
                id="COMP-001", name="App Instance",
                purpose="Serve product pages.",
                description="Implements FEAT-001.",
                related_feature_ids=["FEAT-001"],
                related_adr_ids=["ADR-001"],
            )],
        ), tc.fake_usage()

    monkeypatch.setattr(arch, "llm_call", _phase2)
    state = _state()

    update = arch.architect_node(state)
    assert update["adrs"][0].id == "ADR-001"


# ── defense-in-depth: phase-2-owned prose (Blueprint/ADR/Component/migration) ──


def _design_state_with_prose(**field_overrides):
    state = new_run("Design a system.")
    state.context_record = NO_NUMERIC_TARGETS_CONTEXT
    state.features = [Feature(
        id="FEAT-001", name="F", scenario="Stays responsive.",
        acceptance_criteria=["Stays available."],
    )]
    blueprint_kwargs = dict(
        stakeholder_view="Peak sales keep working.",
        technical_view="One deployable unit scales horizontally.",
        addressed_feature_ids=["FEAT-001"],
        components=["App Instance"],
    )
    blueprint_kwargs.update(field_overrides.pop("blueprint", {}))
    state.blueprint = Blueprint(**blueprint_kwargs)
    state.adrs = field_overrides.pop("adrs", [])
    state.components = field_overrides.pop("components", [])
    return state


def test_invented_target_in_blueprint_prose_is_caught_deterministically():
    state = _design_state_with_prose(
        blueprint={"technical_view": "Responds to all requests within 200ms."}
    )
    checks = run_deterministic_checks(state)
    assert "Blueprint.technical_view" in checks.invented_quantitative_targets
    assert "200ms" in checks.invented_quantitative_targets["Blueprint.technical_view"]
    assert any(issue.category == "grounding" and issue.severity == "high" for issue in checks.issues)


def test_invented_target_in_adr_rationale_is_caught_deterministically():
    adr = ADR(
        id="ADR-001", title="ADR-1: Scale out",
        context="Peak load.", decision="Add instances.",
        rationale="This keeps p99 latency under 300ms.",
        related_feature_ids=["FEAT-001"], related_component_names=["App Instance"],
    )
    state = _design_state_with_prose(adrs=[adr])
    checks = run_deterministic_checks(state)
    assert "ADR-001.rationale" in checks.invented_quantitative_targets
    assert "300ms" in checks.invented_quantitative_targets["ADR-001.rationale"]


def test_invented_target_in_component_scalability_is_caught_deterministically():
    component = ComponentDescription(
        id="COMP-001", name="App Instance", purpose="Serve traffic.",
        description="Stateless app tier.",
        related_feature_ids=["FEAT-001"],
        scalability_considerations=["Scales to handle 10k concurrent users."],
    )
    state = _design_state_with_prose(components=[component])
    checks = run_deterministic_checks(state)
    assert "App Instance.scalability_considerations" in checks.invented_quantitative_targets


def test_invented_target_in_migration_step_is_caught_deterministically():
    state = _design_state_with_prose(
        blueprint={
            "migration_steps": [
                MigrationStep(
                    title="Extract order handling",
                    objective="Cut over with zero downtime and sub-100ms failover.",
                ),
            ],
        }
    )
    checks = run_deterministic_checks(state)
    matched_keys = [k for k in checks.invented_quantitative_targets if "objective" in k]
    assert matched_keys
    assert any("100ms" in v for values in checks.invented_quantitative_targets.values() for v in values)


def test_authorized_target_in_prose_does_not_trigger_the_gate():
    state = _design_state_with_prose(
        blueprint={"technical_view": "Sized for the approved 99.9% availability target."}
    )
    state.context_record = AUTHORIZED_AVAILABILITY_CONTEXT
    checks = run_deterministic_checks(state)
    assert checks.invented_quantitative_targets == {}


def test_clean_design_has_no_quantitative_target_findings():
    state = _design_state_with_prose()
    checks = run_deterministic_checks(state)
    assert checks.invented_quantitative_targets == {}


# ── no domain-specific hardcoding ────────────────────────────────────────

def test_quantitative_target_detection_carries_no_domain_hardcoding():
    import inspect

    import pipeline.review_checks as rc

    source = inspect.getsource(rc.find_unauthorized_quantitative_targets)
    source += inspect.getsource(rc._quantitative_target_locations)
    forbidden = ["e-commerce", "ecommerce", "shopping cart", "checkout", "order service"]
    lowered = source.lower()
    for term in forbidden:
        assert term not in lowered, f"found domain-specific term {term!r}"
