"""test_clarifier.py — offline unit tests for the Clarifier (Kati).

No API key or network needed. The Clarifier, Architect, and Reviewer LLM calls
are replaced with deterministic canned responses.
"""

from __future__ import annotations

from pipeline.state import (
    ADR,
    Blueprint,
    ComponentDescription,
    Feature,
    ClarificationResult,
    ClarifyingQuestion,
    ContextField,
    Stage,
    new_run,
)
from pipeline.agents import architect as arch
from pipeline.agents import clarifier as clar
from pipeline.agents import reviewer as rev
from pipeline import orchestrator


PROMPT = "Build me a system to sell sneakers online."


# ── canned Clarifier judgments ──────────────────────────────────────────
def _missing(state, prompt, *, system="", model="", response_schema=None):
    """Something architecture-critical is unknown."""

    return ClarificationResult(
        captured_context=[ContextField(key="domain", value="e-commerce")],
        assumptions=["Assume REST API (low-stakes)."],
        questions=[
            ClarifyingQuestion(
                question="Expected peak users?",
                why_needed="Sets scale.",
            ),
            ClarifyingQuestion(
                question="GDPR in scope?",
                why_needed="Compliance shapes design.",
            ),
        ],
        missing_critical=["expected scale", "compliance"],
    )


def _complete(state, prompt, *, system="", model="", response_schema=None):
    """Everything critical is known."""

    return ClarificationResult(
        captured_context=[
            ContextField(key="domain", value="e-commerce"),
            ContextField(key="scale", value="50k peak"),
            ContextField(key="cloud", value="AWS"),
        ],
        assumptions=["Assume English-only UI (low-stakes)."],
        questions=[],
        missing_critical=[],
    )


# ── canned Architect judgments ──────────────────────────────────────────
def _architect_response(
    state,
    prompt,
    *,
    system="",
    model="",
    response_schema=None,
):
    """Return deterministic structured output for both Architect phases."""

    if response_schema is arch.FeatureDesign:
        return arch.FeatureDesign(
            features=[
                Feature(
                    id="FEAT-001",
                    name="Handle peak load",
                    description=(
                        "Support high customer traffic during product launches."
                    ),
                    scenario=(
                        "The webshop remains responsive for 50k peak users."
                    ),
                    related_requirement_ids=["NFR-001"],
                    priority="must",
                    acceptance_criteria=[
                        "The webshop remains available during peak traffic.",
                    ],
                )
            ]
        )

    if response_schema is arch.ArchitectureDesign:
        return arch.ArchitectureDesign(
            blueprint=Blueprint(
                project_name="Sneaker Webshop",
                selected_pattern="Event-Driven Microservices Architecture",
                rationale=(
                    "Independent scaling and asynchronous processing support "
                    "peak demand."
                ),
                stakeholder_view=(
                    "Customers purchase sneakers through a reliable online "
                    "storefront."
                ),
                technical_view=(
                    "An AWS-hosted frontend and API Gateway connect to "
                    "independently scalable services. Event-based order "
                    "processing reduces pressure on the existing monolith, "
                    "while encryption and consent controls support GDPR. "
                    "Managed services control cost within the available budget."
                ),
                components=[
                    "Frontend",
                    "API Gateway",
                    "Order Service",
                ],
                data_flows=[
                    "Customer request flows from Frontend to API Gateway.",
                    "API Gateway forwards orders to Order Service.",
                ],
                addressed_feature_ids=["FEAT-001"],
                constraints_addressed=[
                    "AWS cloud",
                    "medium budget",
                    "50k peak users",
                    "GDPR compliance",
                    "existing monolith migration",
                ],
                open_risks=[
                    "Migration from the legacy monolith requires staged rollout.",
                ],
            ),
            adrs=[
                ADR(
                    id="ADR-001",
                    title=(
                        "ADR-1: Separate Order Service from the existing monolith"
                    ),
                    context=(
                        "The current monolith cannot reliably handle peak load."
                    ),
                    decision=(
                        "Use an independently scalable Order Service on AWS."
                    ),
                    rationale=(
                        "The Order Service can scale horizontally during "
                        "high-demand launches."
                    ),
                    alternatives_considered=[
                        "Scale the complete monolith vertically",
                        "Retain synchronous in-process order handling",
                    ],
                    positive_consequences=[
                        "Independent scaling",
                        "Improved fault isolation",
                    ],
                    negative_consequences=[
                        "Additional operational complexity",
                    ],
                    related_feature_ids=["FEAT-001"],
                    related_component_names=["Order Service"],
                    source_references=["Architecture knowledge base"],
                )
            ],
            components=[
                ComponentDescription(
                    id="COMP-001",
                    name="Order Service",
                    component_type="service",
                    purpose="Processes customer orders independently.",
                    description=(
                        "Order Service implements FEAT-001 and is justified by "
                        "ADR-001. It owns order processing and supports "
                        "horizontal scaling on AWS."
                    ),
                    inputs=["Validated order requests"],
                    outputs=["Order confirmation events"],
                    dependencies=["API Gateway"],
                    related_feature_ids=["FEAT-001"],
                    related_adr_ids=["ADR-001"],
                    technology_choices=["AWS managed compute"],
                    security_considerations=[
                        "Encryption in transit",
                        "GDPR-compliant customer-data handling",
                    ],
                    scalability_considerations=[
                        "Horizontal scaling during peak traffic",
                    ],
                )
            ],
        )

    raise AssertionError(
        f"Unexpected Architect response schema: {response_schema}"
    )


# ── tests ────────────────────────────────────────────────────────────────
def test_pauses_when_critical_missing():
    clar.llm_call = _missing

    out = clar.clarifier_node(new_run(PROMPT))

    assert out["stage"] is Stage.AWAITING_INPUT
    assert out["clarifying_questions"] == [
        "Expected peak users?",
        "GDPR in scope?",
    ]
    assert "context_record" not in out


def test_advances_and_locks_context_when_complete():
    clar.llm_call = _complete

    out = clar.clarifier_node(new_run(PROMPT))

    assert out["stage"] is Stage.CLARIFYING
    assert out["context_record"] is not None
    assert "scale: 50k peak" in out["context_record"].summary
    assert "Assumptions" in out["context_record"].summary
    assert out["clarifying_questions"] == []


def test_full_pause_then_resume():
    """First pass pauses; after answers, the offline pipeline resumes."""

    import architect as legacy_architect

    legacy_architect.retrieve_chunks = lambda query, k=3: (
        [
            {
                "content": "Use asynchronous processing for peak traffic.",
                "source": "offline-test-kb",
                "page": 1,
                "box": 1,
                "distance": 0.1,
            }
        ],
        "offline-test",
    )

    def _stateful(
        state,
        prompt,
        *,
        system="",
        model="",
        response_schema=None,
    ):
        if state.clarification_answers:
            return _complete(state, prompt)
        return _missing(state, prompt)

    clar.llm_call = _stateful
    arch.llm_call = _architect_response
    rev.llm_call = lambda state, prompt, **kwargs: rev.LLMJudgments()

    state = new_run(PROMPT)
    state = orchestrator.run_pipeline(state)

    assert state.stage is Stage.AWAITING_INPUT
    assert state.clarifying_questions
    assert state.context_record is None

    state.clarification_answers = {
        question: "some answer"
        for question in state.clarifying_questions
    }

    state = orchestrator.run_pipeline(state)

    assert state.stage is Stage.REFINING
    assert state.review is not None
    assert state.review.requires_refinement
    assert state.context_record is not None
    assert state.features
    assert state.blueprint is not None
    assert state.adrs
    assert state.components

    agents_run = [step.agent for step in state.history]
    assert "researcher" in agents_run
    assert "architect" in agents_run
    assert "reviewer" in agents_run


if __name__ == "__main__":
    test_pauses_when_critical_missing()
    print("PASS  pauses when critical information is missing")

    test_advances_and_locks_context_when_complete()
    print("PASS  advances and locks ContextRecord")

    test_full_pause_then_resume()
    print("PASS  full pause and resume pipeline")

    print("\nALL CLARIFIER TESTS PASSED")
