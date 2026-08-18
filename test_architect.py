"""Offline unit tests for the Architect Agent.

No API key or network is required. The two LLM phases are replaced with
deterministic structured responses.

FEATURE STABILITY ACROSS THE REFINE LOOP
----------------------------------------
The second block of tests below is about one thing: on a refine pass the
architect must REUSE `state.features` rather than re-derive them. That is what
makes a reviewer instruction like "assign FEAT-005 to a component" still refer
to something real on the next pass — measured on run
20260818T074835Z-1925fbd7, where phase 1 produced 5 features, then 4, then 5,
and the loop spent both iterations chasing an ID that kept moving.

Those tests count PHASE 1 CALLS, not tokens or flags, because the property is
"the model is not asked to invent features again" and only a call count can say
that.
"""

from __future__ import annotations

import pytest

import test_clarifier as tc  # reuse the shared canned-usage helper
from pipeline.agents import architect as arch
from pipeline.state import (
    ADR,
    Blueprint,
    ComponentDescription,
    ContextRecord,
    Feature,
    KBChunk,
    RepoRepresentation,
    ReviewResult,
    Stage,
    new_run,
)


def _fake_architect_llm(
    state,
    prompt,
    *,
    system="",
    model="",
    response_schema=None,
):
    """Return canned outputs for both Architect phases, as `(reply, usage)`."""

    if response_schema is arch.FeatureDesign:
        return (arch.FeatureDesign(
            features=[
                Feature(
                    id="FEAT-001",
                    name="Handle peak load",
                    description="Support high traffic during sale events.",
                    scenario="The platform remains responsive for 50k peak users.",
                    related_requirement_ids=["NFR-001"],
                    priority="must",
                    acceptance_criteria=[
                        "The platform stays available under peak load.",
                    ],
                )
            ]
        ), tc.fake_usage())

    if response_schema is arch.ArchitectureDesign:
        return (arch.ArchitectureDesign(
            blueprint=Blueprint(
                project_name="E-commerce Platform",
                selected_pattern="Event-Driven Microservices Architecture",
                rationale="Independent scaling reduces monolith bottlenecks.",
                stakeholder_view="Customers receive a reliable shopping experience.",
                technical_view=(
                    "An AWS-hosted frontend and API Gateway connect to "
                    "independently scalable services. Events decouple order "
                    "processing from the legacy monolith."
                ),
                components=["Frontend", "API Gateway", "Order Service"],
                data_flows=[
                    "Frontend sends order requests through the API Gateway.",
                    "Order Service emits order events.",
                ],
                addressed_feature_ids=["FEAT-001"],
                constraints_addressed=[
                    "AWS",
                    "medium budget",
                    "50k peak users",
                    "GDPR",
                    "legacy monolith migration",
                ],
            ),
            adrs=[
                ADR(
                    id="ADR-001",
                    title="ADR-1: Separate order processing from the monolith",
                    context="The monolith fails under peak load.",
                    decision="Introduce an independently scalable Order Service.",
                    rationale="Order processing can scale separately.",
                    alternatives_considered=[
                        "Scale the whole monolith vertically",
                    ],
                    positive_consequences=[
                        "Independent scaling",
                        "Improved fault isolation",
                    ],
                    negative_consequences=[
                        "Higher operational complexity",
                    ],
                    related_feature_ids=["FEAT-001"],
                    related_component_names=["Order Service"],
                    source_references=["Architecture KB"],
                )
            ],
            components=[
                ComponentDescription(
                    id="COMP-001",
                    name="Order Service",
                    component_type="service",
                    purpose="Owns and processes customer orders.",
                    description=(
                        "Implements FEAT-001 and is justified by ADR-001."
                    ),
                    inputs=["Validated order requests"],
                    outputs=["Order confirmation events"],
                    dependencies=["API Gateway"],
                    related_feature_ids=["FEAT-001"],
                    related_adr_ids=["ADR-001"],
                    technology_choices=["AWS managed compute"],
                    security_considerations=["Encryption in transit"],
                    scalability_considerations=["Horizontal scaling"],
                )
            ],
        ), tc.fake_usage())

    raise AssertionError(f"Unexpected response schema: {response_schema}")


def _base_state():
    state = new_run("Modernize an e-commerce platform.")
    state.stage = Stage.RESEARCHING
    state.context_record = ContextRecord(
        project_name="E-commerce Platform",
        business_goal="Increase reliability during sale events.",
        problem_statement="The existing monolith fails under peak load.",
        users=["Online customers", "Operations team"],
        functional_requirements=["Process customer orders"],
        non_functional_requirements=["Support 50k peak users"],
        cloud_provider="AWS",
        budget="medium",
        compliance_requirements=["GDPR"],
        existing_systems=["Legacy monolith"],
        summary="AWS e-commerce platform with 50k peak users and GDPR constraints.",
    )
    state.repo_representation = RepoRepresentation(
        summary="Legacy monolith with tightly coupled order processing."
    )
    state.retrieved_knowledge = [
        KBChunk(
            content="Event-driven microservices support independent scaling.",
            source="Architecture KB",
        )
    ]
    return state


def test_architect_generates_all_artifacts():
    arch.llm_call = _fake_architect_llm
    state = _base_state()

    output = arch.architect_node(state)

    assert output["stage"] is Stage.DESIGNING
    assert len(output["features"]) == 1
    assert output["blueprint"] is not None
    assert len(output["adrs"]) == 1
    assert len(output["components"]) == 1
    assert output["components"][0].related_feature_ids == ["FEAT-001"]
    assert output["components"][0].related_adr_ids == ["ADR-001"]
    assert output["history"][0].agent == "architect"


def test_architect_fails_without_context_record():
    arch.llm_call = _fake_architect_llm
    state = new_run("Design a system.")
    state.stage = Stage.RESEARCHING

    output = arch.architect_node(state)

    assert output["stage"] is Stage.FAILED
    assert output["errors"]
    assert "Context Record" in output["errors"][0]


def test_architect_prompts_include_rag_and_repository_context():
    state = _base_state()

    feature_prompt = arch._build_feature_prompt(state)
    design_prompt = arch._build_architecture_prompt(
        state,
        [
            Feature(
                id="FEAT-001",
                name="Handle peak load",
                scenario="Remain responsive at 50k peak users.",
            )
        ],
    )

    assert "50k peak users" in feature_prompt
    assert "Legacy monolith" in design_prompt
    assert "Architecture KB" in design_prompt
    assert "Event-driven microservices" in design_prompt

# ══════════════════════════════════════════════════════════════════════════
# Refine passes reuse the feature set instead of re-deriving it
# ══════════════════════════════════════════════════════════════════════════
def _counting_llm(calls: list[str]):
    """`_fake_architect_llm`, but recording which phase each call was for."""

    def stub(state, prompt, *, system="", model="", response_schema=None):
        calls.append("phase1" if response_schema is arch.FeatureDesign else "phase2")
        return _fake_architect_llm(
            state, prompt, system=system, model=model, response_schema=response_schema
        )

    return stub


def _refining_state(features: list[Feature] | None) -> object:
    """A state as the refine gate hands it to the architect: a failing review."""
    state = _base_state()
    state.stage = Stage.REFINING
    state.review = ReviewResult(
        overall_status="fail",
        requires_refinement=True,
        refinement_instruction="Assign FEAT-005 to a component.",
    )
    state.features = list(features or [])
    return state


def _existing_features() -> list[Feature]:
    """Five features, so a re-derivation that produced four would be visible."""
    return [
        Feature(
            id=f"FEAT-{n:03d}",
            name=f"Capability {n}",
            scenario=f"Scenario {n} holds under load.",
        )
        for n in range(1, 6)
    ]


def test_refine_pass_reuses_features_and_skips_phase_one():
    """THE convergence fix. Feature IDs must survive the loop unchanged.

    The canned phase-1 response returns exactly ONE feature (FEAT-001), so if
    phase 1 ran, the five IDs seeded here would collapse to one — the assertion
    on the ID list would fail even if the call-count assertion somehow did not.
    """
    calls: list[str] = []
    arch.llm_call = _counting_llm(calls)
    before = _existing_features()
    state = _refining_state(before)

    output = arch.architect_node(state)

    assert calls == ["phase2"], f"phase 1 ran on a refine pass: {calls}"
    # The key is ABSENT, not rewritten: `features` is a LastValue channel, so
    # leaving it out is what preserves the list.
    assert "features" not in output
    assert [f.id for f in state.features] == [f.id for f in before]
    assert output["stage"] is Stage.DESIGNING
    assert output["blueprint"] is not None
    assert "reused 5 feature(s)" in output["history"][0].note


def test_initial_design_still_runs_both_phases():
    """No review yet -> the features do not exist -> phase 1 must run."""
    calls: list[str] = []
    arch.llm_call = _counting_llm(calls)

    output = arch.architect_node(_base_state())

    assert calls == ["phase1", "phase2"]
    assert [f.id for f in output["features"]] == ["FEAT-001"]
    assert "derived 1 feature(s)" in output["history"][0].note


def test_refine_with_no_features_falls_back_to_phase_one():
    """Defensive: a refine pass with an empty feature list is a bug elsewhere.

    Re-deriving is the recoverable response; failing the run over an
    inconsistency this node did not create is not.
    """
    calls: list[str] = []
    arch.llm_call = _counting_llm(calls)

    output = arch.architect_node(_refining_state([]))

    assert calls == ["phase1", "phase2"]
    assert [f.id for f in output["features"]] == ["FEAT-001"]


def test_passing_review_is_not_treated_as_a_refine_pass():
    """`requires_refinement` is the signal, not the mere presence of a review."""
    calls: list[str] = []
    arch.llm_call = _counting_llm(calls)
    state = _refining_state(_existing_features())
    state.review.requires_refinement = False

    arch.architect_node(state)

    assert calls == ["phase1", "phase2"]


def test_refine_pass_reports_one_call_of_usage():
    """Token accounting has to follow the call count, not a hard-coded two.

    The step and the returned totals are the same numbers — that is what keeps
    `usage_by_agent()` reconciled with the run totals (see
    test_token_accounting.py), and it has to stay true now that the number of
    calls per visit varies.
    """
    arch.llm_call = _counting_llm([])

    refined = arch.architect_node(_refining_state(_existing_features()))
    initial = arch.architect_node(_base_state())

    one_call_in = tc.FAKE_INPUT_TOKENS
    one_call_out = tc.FAKE_OUTPUT_TOKENS

    assert refined["input_tokens"] == one_call_in
    assert refined["output_tokens"] == one_call_out
    assert initial["input_tokens"] == 2 * one_call_in
    assert initial["output_tokens"] == 2 * one_call_out

    for output in (refined, initial):
        step = output["history"][0]
        assert step.input_tokens == output["input_tokens"]
        assert step.output_tokens == output["output_tokens"]


def test_refine_pass_still_reaches_the_architecture_prompt_with_the_instruction():
    """Reusing features must not drop the reason the pass is happening."""
    state = _refining_state(_existing_features())

    prompt = arch._build_architecture_prompt(state, state.features)

    assert "Assign FEAT-005 to a component." in prompt
    assert "FEAT-005" in prompt  # the ID it names is in the feature set it sees
