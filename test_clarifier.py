"""test_clarifier.py — offline unit tests for the Clarifier (Kati).

No API key or network needed. The Clarifier, Architect, and Reviewer LLM calls
are replaced with deterministic canned responses.

Every stub returns `(value, LLMUsage)` because that is `llm_call`'s contract —
the reply plus what it cost. `FAKE_USAGE` gives each stubbed call a non-zero,
predictable price so the token-accounting tests have real numbers to check
against; see test_token_accounting.py.
"""

from __future__ import annotations

from pipeline.llm import LLMUsage
from pipeline.state import (
    ADR,
    Blueprint,
    CapturedContext,
    ComponentDescription,
    Feature,
    ClarificationResult,
    ClarifyingQuestion,
    ContextRecord,
    PendingDecision,
    Stage,
    new_run,
)
from pipeline.agents import architect as arch
from pipeline.agents import clarifier as clar
from pipeline.agents import reviewer as rev
from pipeline import orchestrator


PROMPT = "Build me a system to sell sneakers online."


# ── canned token usage ───────────────────────────────────────────────────
# What ONE stubbed call "consumed". Non-zero and asymmetric so a test summing
# these cannot pass by accident (e.g. by swapping input and output), and priced
# through the REAL cost table against a real model ID, so the cost assertions
# exercise production pricing code rather than a hand-written number.
FAKE_MODEL = "gemini-3.1-flash-lite"
FAKE_INPUT_TOKENS = 1000
FAKE_OUTPUT_TOKENS = 500


def fake_usage() -> LLMUsage:
    """Usage for one stubbed call — the second half of `llm_call`'s contract."""

    from pipeline.llm import estimate_cost_usd

    return LLMUsage(
        model=FAKE_MODEL,
        input_tokens=FAKE_INPUT_TOKENS,
        output_tokens=FAKE_OUTPUT_TOKENS,
        cost_usd=estimate_cost_usd(
            FAKE_MODEL, FAKE_INPUT_TOKENS, FAKE_OUTPUT_TOKENS
        ),
    )


# ── canned Clarifier judgments ──────────────────────────────────────────
def _missing(state, prompt, *, system="", model="", response_schema=None):
    """Something architecture-critical is unknown."""

    return ClarificationResult(
        captured=CapturedContext(business_goal="Sell sneakers online"),
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
    ), fake_usage()


def _complete(state, prompt, *, system="", model="", response_schema=None):
    """Everything critical is known."""

    return ClarificationResult(
        captured=CapturedContext(
            business_goal="Sell sneakers online",
            non_functional_requirements=["50k peak users"],
            cloud_provider="AWS",
            compliance_requirements=["GDPR"],
        ),
        assumptions=["Assume English-only UI (low-stakes)."],
        questions=[],
        missing_critical=[],
    ), fake_usage()


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
        return (arch.FeatureDesign(
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
        ), fake_usage())

    if response_schema is arch.ArchitectureDesign:
        return (arch.ArchitectureDesign(
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
        ), fake_usage())

    raise AssertionError(
        f"Unexpected Architect response schema: {response_schema}"
    )


# ── tests ────────────────────────────────────────────────────────────────
def _locked_state(**kwargs):
    """A state whose clarifier has already locked a record — i.e. a RE-JUDGE.

    `context_record is not None` is what puts the clarifier into assume-only
    mode, so this is the shortest honest way to reach that mode in a test.
    """
    state = new_run(PROMPT, **kwargs)
    state.context_record = ContextRecord(
        business_goal="Sell sneakers online",
        cloud_provider="AWS",
        assumptions=["Assume English-only UI (low-stakes)."],
    )
    return state


def test_pauses_when_critical_missing():
    clar.llm_call = _missing

    out = clar.clarifier_node(new_run(PROMPT))

    assert out["stage"] is Stage.AWAITING_HUMAN
    assert out["clarifying_questions"] == [
        "Expected peak users?",
        "GDPR in scope?",
    ]
    assert "context_record" not in out


def test_advances_and_locks_context_when_complete():
    clar.llm_call = _complete

    out = clar.clarifier_node(new_run(PROMPT))

    assert out["stage"] is Stage.CLARIFYING
    cr = out["context_record"]
    assert cr is not None
    # mapped 1:1 onto Maheen's real ContextRecord fields
    assert cr.cloud_provider == "AWS"
    assert "50k peak users" in cr.non_functional_requirements
    assert cr.compliance_requirements == ["GDPR"]
    assert cr.assumptions == ["Assume English-only UI (low-stakes)."]
    assert "Goal: Sell sneakers online" in cr.summary
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
    rev.llm_call = lambda state, prompt, **kwargs: (rev.LLMJudgments(), fake_usage())

    state = new_run(PROMPT)
    state = orchestrator.run_pipeline(state)

    assert state.stage is Stage.AWAITING_HUMAN
    assert state.clarifying_questions
    assert state.context_record is None

    state.clarification_answers = {
        question: "some answer"
        for question in state.clarifying_questions
    }

    state = orchestrator.run_pipeline(state)

    # The reviewer mock always fails, so with the refine loop now wired the run
    # loops until the cost cap and finishes GRACEFULLY as DONE (best-effort),
    # rather than stalling at REFINING as it did before the loop was closed.
    assert state.stage is Stage.DONE
    assert state.stopped_on_cap
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


# ── the context lock: pause at the lock, or do not ───────────────────────
def test_lock_pauses_for_approval_when_required():
    """Approval on -> the lock is a PAUSE, not an advance to research."""
    clar.llm_call = _complete

    out = clar.clarifier_node(new_run(PROMPT, require_context_approval=True))

    assert out["stage"] is Stage.AWAITING_HUMAN
    assert out["pending_decision"] is PendingDecision.CONTEXT_LOCK
    # The record is still produced — the human is approving something real,
    # not waiting for the clarifier to run again.
    assert out["context_record"] is not None
    assert out["context_record"].cloud_provider == "AWS"


def test_lock_advances_when_approval_not_required():
    """Approval off (the default) -> exactly the old behaviour, no pause.

    This is the test that would fail if the gate were made mandatory, which is
    what every headless caller depends on.
    """
    clar.llm_call = _complete

    out = clar.clarifier_node(new_run(PROMPT))

    assert out["stage"] is Stage.CLARIFYING
    assert out["pending_decision"] is None
    assert out["context_record"] is not None


def test_pause_for_questions_carries_the_clarification_discriminator():
    """The two pauses share a stage, so the discriminator has to tell them apart."""
    clar.llm_call = _missing

    out = clar.clarifier_node(new_run(PROMPT, require_context_approval=True))

    assert out["stage"] is Stage.AWAITING_HUMAN
    assert out["pending_decision"] is PendingDecision.CLARIFICATION


# ── ask once, then assume ────────────────────────────────────────────────
def test_rejudge_never_pauses_with_questions():
    """A gap on a RE-JUDGE becomes a labelled assumption + an open question.

    The model here reports two critical gaps AND two matching questions — the
    well-behaved case. It still may not ask, because a human is already looking
    at this record. "Never assume silently" is satisfied by showing them the
    assumption, not by asking them again.
    """
    clar.llm_call = _missing

    out = clar.clarifier_node(_locked_state(require_context_approval=True))

    assert out["stage"] is Stage.AWAITING_HUMAN
    assert out["pending_decision"] is PendingDecision.CONTEXT_LOCK
    assert out["clarifying_questions"] == []

    record = out["context_record"]
    for gap in ("expected scale", "compliance"):
        assert any(gap in q for q in record.open_questions), gap
        assert any(gap in a for a in record.assumptions), gap
    # Every assumption on a re-judge stands in for a question that was not
    # asked, so every one of them is attributed.
    assert all(a.startswith(clar.CLARIFIER_LABEL) for a in record.assumptions)


def test_missing_critical_without_questions_does_not_park_the_run():
    """The known clarifier bug, contained: gaps but no questions -> lock, not pause.

    Pausing here used to leave the run somewhere no answer could reach it (see
    run.py's docstring). There is nothing to ask, so the gaps are absorbed the
    same way a re-judge absorbs them and the run keeps moving.
    """

    def _gaps_but_no_questions(state, prompt, **kwargs):
        return ClarificationResult(
            captured=CapturedContext(business_goal="Sell sneakers online"),
            questions=[],
            missing_critical=["expected scale"],
        ), fake_usage()

    clar.llm_call = _gaps_but_no_questions

    out = clar.clarifier_node(new_run(PROMPT))

    assert out["stage"] is Stage.CLARIFYING
    assert out["context_record"] is not None
    assert any("expected scale" in a for a in out["context_record"].assumptions)


def test_vetoed_assumption_is_not_re_proposed():
    """A strike outlives the record it was cast against.

    The model re-proposes the struck line verbatim, which is the ordinary case
    rather than the exotic one. Code filters it; the prompt asking nicely is not
    a guarantee.
    """
    clar.llm_call = _complete

    state = _locked_state()
    state.vetoed_assumptions = ["Assume English-only UI (low-stakes)."]

    out = clar.clarifier_node(state)

    assert "Assume English-only UI (low-stakes)." not in out["context_record"].assumptions


if __name__ == "__main__":
    test_pauses_when_critical_missing()
    print("PASS  pauses when critical information is missing")

    test_advances_and_locks_context_when_complete()
    print("PASS  advances and locks ContextRecord")

    test_full_pause_then_resume()
    print("PASS  full pause and resume pipeline")

    test_lock_pauses_for_approval_when_required()
    print("PASS  lock pauses for approval when required")

    test_lock_advances_when_approval_not_required()
    print("PASS  lock advances when approval is not required")

    test_pause_for_questions_carries_the_clarification_discriminator()
    print("PASS  question pause carries pending_decision=CLARIFICATION")

    test_rejudge_never_pauses_with_questions()
    print("PASS  re-judge assumes instead of asking")

    test_missing_critical_without_questions_does_not_park_the_run()
    print("PASS  gaps without questions do not park the run")

    test_vetoed_assumption_is_not_re_proposed()
    print("PASS  a struck assumption stays struck")

    print("\nALL CLARIFIER TESTS PASSED")
