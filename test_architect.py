"""Offline unit tests for the Architect Agent.

No API key or network is required. The two LLM phases are replaced with
deterministic structured responses.
"""

from __future__ import annotations

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