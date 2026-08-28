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


# Deliberately carries an NFR signal ("scale") and a cloud signal ("cloud
# hosting") so the two conditionally-critical fields this suite exercises
# (`non_functional_requirements`, `cloud_provider`; see
# `clarifier._slot_is_relevant`) are actually IN PLAY for this request. A
# prompt with no such wording at all asks only the three unconditionally
# critical fields — see test_clarifier_hybrid_determinism.py for that case.
PROMPT = "Build me a system to sell sneakers online, sized for real scale, with a cloud hosting preference."


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
#
# `captured` is what actually drives routing now (see
# `clarifier.missing_critical_slots`): the deterministic gate recomputes
# critical gaps FROM `captured`, ignoring whatever `questions`/
# `missing_critical` the canned response also carries. Those two fields are
# kept below only because `ClarificationResult` still has them and a couple
# of assertions read `open_questions` off the frozen record, not because
# they drive the ask/lock decision any more.
#
# `_missing` deliberately leaves exactly `non_functional_requirements` and
# `cloud_provider` empty. Both are CONDITIONALLY critical fields (see
# `clarifier._slot_is_relevant`), relevant here only because `PROMPT` above
# carries their signal wording ("scale", "cloud hosting") — so this fixture
# reproducibly maps to exactly two deterministic gaps for THIS prompt.
# `compliance_requirements` is NOT used for this fixture's gap even though the
# ORIGINAL fixture used it, because compliance is conditionally critical too
# (only relevant when the raw prompt itself signals a regulated/sensitive
# domain) and PROMPT below never does.
def _missing(state, prompt, *, system="", model="", response_schema=None):
    """Something architecture-critical is unknown."""

    return ClarificationResult(
        captured=CapturedContext(
            business_goal="Sell sneakers online",
            problem_statement="The current setup cannot handle sale-day traffic",
            functional_requirements=["Browse and buy sneakers online"],
            budget="medium",
            # non_functional_requirements and cloud_provider deliberately
            # left empty — the two gaps this fixture exists to represent.
        ),
        assumptions=["Assume REST API (low-stakes)."],
        questions=[
            ClarifyingQuestion(
                question="Expected peak users?",
                why_needed="Sets scale.",
            ),
            ClarifyingQuestion(
                question="Preferred cloud provider?",
                why_needed="Shapes hosting.",
            ),
        ],
        missing_critical=["expected scale", "cloud provider"],
    ), fake_usage()


def _complete(state, prompt, *, system="", model="", response_schema=None):
    """Everything critical is known — for EVERY caller of this fixture, not
    just the ones whose own prompt happens to mention it.

    `existing_systems` is filled even though most callers' prompts never ask
    about it: `existing_systems` is conditionally critical (see
    `clarifier._slot_is_relevant`) on brownfield/modernization/legacy/monolith
    wording, and several suites elsewhere (persistence, token accounting) run
    this exact fixture against a prompt that says "monolithic" — which makes
    `existing_systems` a real, relevant gap. Leaving it empty here would make
    "everything critical is known" a lie for those prompts, and the run would
    pause on a fact this fixture's own name says is settled. A filled field is
    never a gap regardless of relevance, so this is a safe superset for every
    other caller too.
    """

    return ClarificationResult(
        captured=CapturedContext(
            business_goal="Sell sneakers online",
            problem_statement="The monolith falls over on peak sale days",
            functional_requirements=["Browse and buy sneakers online"],
            non_functional_requirements=["50k peak users"],
            cloud_provider="AWS",
            budget="medium",
            compliance_requirements=["GDPR"],
            existing_systems=["Legacy monolith being replaced"],
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
    thinking_level=None,
):
    """Return deterministic structured output for both Architect phases.

    `thinking_level` (and any future per-call provider option) is accepted
    and ignored: this double stands in for the provider seam, which is
    exactly where such options are consumed."""

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
                    related_requirement_ids=["FR-001", "NFR-001"],
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
    # Deterministic template text, in `CRITICAL_RECORD_FIELDS` order — not the
    # model's own (unused) `questions` wording. See `missing_critical_slots`.
    assert out["clarifying_questions"] == [
        clar._CRITICAL_SLOT_QUESTIONS["non_functional_requirements"].question,
        clar._CRITICAL_SLOT_QUESTIONS["cloud_provider"].question,
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
    # Nothing was absorbed (every critical field is filled). The positive
    # assumption policy still drops model-authored prose that is not
    # positively classified as safe, but "Assume English-only UI (low-stakes)."
    # from `_complete` IS safe (display/UI-language metadata), so it survives
    # labelled. See test_clarifier_hybrid_determinism.py.
    assert cr.assumptions == [
        f"{clar.CLARIFIER_LABEL} Assume English-only UI (low-stakes)."
    ]
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
    # `non_functional_requirements`/`cloud_provider` are both REQUIRED here
    # (PROMPT signals both), so absorbing them does not make them optional —
    # but `compliance_requirements` is an `OPTIONAL_CONTEXT_FIELDS` member
    # this fixture's captured never sets, and is NOT required for PROMPT, so
    # it is what stops the run here FIRST, before the review screen (see
    # `clarifier.optional_slots`).
    assert out["pending_decision"] is PendingDecision.OPTIONAL_CONTEXT
    assert out["clarifying_questions"] == []

    record = out["context_record"]
    for gap in ("non_functional_requirements", "cloud_provider"):
        assert any(gap in q for q in record.open_questions), gap
        assert any(gap in a for a in record.assumptions), gap
    # Every assumption on a re-judge stands in for a question that was not
    # asked, so every one of them is attributed.
    assert all(a.startswith(clar.CLARIFIER_LABEL) for a in record.assumptions)


def test_deterministic_questions_survive_even_when_the_model_writes_none():
    """The known clarifier bug is now impossible by construction.

    The model could always report `missing_critical` while returning NO
    `questions` — nothing on screen for a gap that used to park the run where
    no answer could reach it (see run.py's docstring). Routing no longer reads
    the model's own `questions`/`missing_critical` at all: `critical_gaps` is
    computed from `captured`, and `_CRITICAL_SLOT_QUESTIONS` has a fixed
    question for every one of `CRITICAL_RECORD_FIELDS`, so a critical gap can
    never have nothing to ask — the model returning `questions=[]` here still
    produces a proper pause with real questions on screen.
    """

    def _gaps_but_no_questions(state, prompt, **kwargs):
        return ClarificationResult(
            captured=CapturedContext(business_goal="Sell sneakers online"),
            questions=[],
            missing_critical=[],
        ), fake_usage()

    clar.llm_call = _gaps_but_no_questions

    out = clar.clarifier_node(new_run(PROMPT))

    assert out["stage"] is Stage.AWAITING_HUMAN
    assert out["pending_decision"] is PendingDecision.CLARIFICATION
    assert out["clarifying_questions"] == [
        clar._CRITICAL_SLOT_QUESTIONS[field].question
        for field in clar.missing_critical_slots(new_run(PROMPT), CapturedContext(business_goal="Sell sneakers online"))
    ]
    assert out["clarifying_questions"]  # non-empty: something IS on screen


def test_vetoed_assumption_is_not_re_proposed():
    """A struck line never reappears.

    "Assume English-only UI (low-stakes)." positively classifies as SAFE
    (display/UI-language metadata) and would normally survive a freeze (see
    `test_advances_and_locks_context_when_complete`) — which is exactly why
    the veto ledger still has to be checked for it: a human who struck it must
    never see it come back on the next re-judge. See
    test_clarifier_hybrid_determinism.py for the positive-allowlist policy
    itself.
    """
    clar.llm_call = _complete

    state = _locked_state()
    state.vetoed_assumptions = ["Assume English-only UI (low-stakes)."]

    out = clar.clarifier_node(state)

    assert out["context_record"].assumptions == []


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

    test_deterministic_questions_survive_even_when_the_model_writes_none()
    print("PASS  deterministic questions survive when the model writes none")

    test_vetoed_assumption_is_not_re_proposed()
    print("PASS  a struck assumption stays struck")

    print("\nALL CLARIFIER TESTS PASSED")


# ── the ask-round cap ────────────────────────────────────────────────────
#
# Run 20260819T083025Z-00c6557a spent 14 pause rounds before design started,
# each answer surfacing two fresh "critical" gaps. `MAX_ASK_ROUNDS` stops that.
# The point of these tests is that stopping is not the same as giving up: the
# gaps still reach the human, as assumptions they can veto at the context gate.


def test_the_cap_converts_remaining_gaps_into_assumptions_not_silence():
    """THE guarantee. Past the cap a critical gap must still be visible.

    `_missing` reports two critical gaps AND two questions, so under the cap
    this pauses. At the cap it must lock instead — and both gaps have to survive
    into the record as labelled assumptions plus open questions. Dropping them
    would make the cap a data-loss bug rather than a budget.
    """

    state = new_run(PROMPT)
    state.ask_rounds = clar.MAX_ASK_ROUNDS
    clar.llm_call = _missing

    output = clar.clarifier_node(state)

    assert output["stage"] is not Stage.AWAITING_HUMAN
    assert output["pending_decision"] is None
    record = output["context_record"]
    assert record is not None

    for gap in ("non_functional_requirements", "cloud_provider"):
        assert any(gap.lower() in a.lower() for a in record.assumptions), gap
        assert any(gap.lower() in q.lower() for q in record.open_questions), gap
    # Labelled AND explicit that the value is unresolved, never a guessed
    # fact — see Problem C / `_freeze_context_record`.
    assert all(a.startswith(clar.CLARIFIER_LABEL) for a in record.assumptions)
    assert all("unresolved" in a for a in record.assumptions)


def test_asking_is_still_allowed_below_the_cap():
    state = new_run(PROMPT)
    state.ask_rounds = clar.MAX_ASK_ROUNDS - 1
    clar.llm_call = _missing

    output = clar.clarifier_node(state)

    assert output["stage"] is Stage.AWAITING_HUMAN
    assert output["clarifying_questions"]
    # And the round is counted, absolutely rather than as a delta.
    assert output["ask_rounds"] == clar.MAX_ASK_ROUNDS


def test_each_pause_costs_exactly_one_ask_round():
    state = new_run(PROMPT)
    clar.llm_call = _missing

    assert clar.clarifier_node(state)["ask_rounds"] == 1
    state.ask_rounds = 1
    assert clar.clarifier_node(state)["ask_rounds"] == 2


def test_locking_does_not_spend_an_ask_round():
    """Only PAUSING costs. A pass that locks is the pipeline working."""

    state = new_run(PROMPT)
    clar.llm_call = _complete

    assert "ask_rounds" not in clar.clarifier_node(state)


def test_one_predicate_governs_both_the_prompt_and_the_routing():
    """`assume_only` and `_can_ask` must not drift into opposite polarity."""

    state = new_run(PROMPT)
    assert clar._may_ask(state) is True

    state.ask_rounds = clar.MAX_ASK_ROUNDS
    assert clar._may_ask(state) is False

    # A locked record closes asking regardless of the budget.
    spent = new_run(PROMPT)
    spent.context_record = ContextRecord(business_goal="Sell sneakers online")
    assert clar._may_ask(spent) is False


def test_the_cli_backstop_stays_looser_than_the_ask_cap():
    """If these invert, the CLI errors out where the clarifier would have
    assumed and carried on — the capped path would become unreachable."""

    from pipeline.run import MAX_CLARIFICATION_ROUNDS

    assert MAX_CLARIFICATION_ROUNDS > clar.MAX_ASK_ROUNDS
