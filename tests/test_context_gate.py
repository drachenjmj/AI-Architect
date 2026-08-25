"""test_context_gate.py — offline tests for the human gate at context lock (Kati).

No API key, no network, no keyboard. What is under test here is the CALLER-SIDE
half of the gate: the pure edit functions, the deterministic decision about when
the model is allowed to run again, the round budget, and the advisory
side-channel. The clarifier NODE's own behaviour at the lock (pause vs advance,
ask-once-then-assume) is pinned in test_clarifier.py; this file starts from a
state that is already paused.

THE TEST THAT MATTERS MOST
--------------------------
`test_filling_a_field_makes_no_llm_call` is the one that would catch the
expensive regression. The gate is only usable if correcting a typo is free; the
moment an edit re-runs the clarifier "just to be safe", the veto becomes the
costliest button in the product and people stop using it. So that test does not
check a flag or a return value — it counts calls, by making `llm_call` fail the
test if it is reached at all.

Every stub returns `(value, LLMUsage)` because that is `llm_call`'s contract.
`test_clarifier.fake_usage` is reused so the token assertions here are priced
through the same real cost table as the rest of the suite.
"""
from __future__ import annotations

import pytest

from pipeline import orchestrator
from pipeline.agents import clarifier as clar
from pipeline.persistence import load_state, save_state
from pipeline.refine_gate import (
    MAX_REFINE_ITERATIONS,
    MAX_USER_ROUNDS,
    begin_user_round,
    derive_max_steps,
    evaluate_user_rounds,
)
from pipeline.state import (
    ArchitectState,
    ContextEdits,
    ContextRecord,
    PendingDecision,
    Stage,
    new_run,
)
from test_clarifier import (
    _architect_response,
    _complete,
    _missing,
    fake_usage,
)

# Carries an NFR signal ("scale") and a cloud signal ("cloud hosting") so
# `non_functional_requirements` / `cloud_provider` — both conditionally
# critical, see `clarifier._slot_is_relevant` — are genuinely in play for the
# `_missing` re-judge scenario below. Matches test_clarifier.PROMPT.
PROMPT = "Build me a system to sell sneakers online, sized for real scale, with a cloud hosting preference."


def _record(**overrides) -> ContextRecord:
    """A locked record with something in every part the gate can touch."""
    base = dict(
        business_goal="Sell sneakers online",
        problem_statement="The monolith falls over on sale days",
        functional_requirements=["Browse and buy sneakers online"],
        non_functional_requirements=["50k peak users"],
        cloud_provider="AWS",
        budget="medium",
        compliance_requirements=["GDPR"],
        existing_systems=["Django monolith"],
        assumptions=["Assume English-only UI (low-stakes)."],
        open_questions=["Is a mobile app in scope?"],
    )
    base.update(overrides)
    return ContextRecord(**base)


def _paused_at_lock(**overrides) -> ArchitectState:
    """A state exactly as the clarifier leaves it when the lock needs approval."""
    state = new_run(PROMPT, require_context_approval=True)
    state.context_record = _record(**overrides)
    state.stage = Stage.AWAITING_HUMAN
    state.pending_decision = PendingDecision.CONTEXT_LOCK
    return state


def _explode(*args, **kwargs):
    """An `llm_call` that fails the test if anything reaches it."""
    raise AssertionError(
        "llm_call was reached. An edit that fills or replaces a value cannot "
        "open a gap, so it must re-freeze the record with NO model call."
    )


# ══════════════════════════════════════════════════════════════════════════
# 1. Accept — the record is approved and the run continues to research
# ══════════════════════════════════════════════════════════════════════════
def test_accept_clears_the_pause_and_targets_the_researcher():
    """Accept sets CLARIFYING, and the EXISTING entry route does the rest.

    Asserted through `_entry_route` rather than by reading the stage, because
    the claim in the design was "no new route is needed" — reading the stage
    back would only prove the assignment happened.
    """
    state = _paused_at_lock()

    clar.accept_context_lock(state)

    assert state.pending_decision is None
    assert state.stage is Stage.CLARIFYING
    assert orchestrator._entry_route(state) == "researcher"


def test_accepting_when_nothing_is_pending_is_refused():
    """A stray accept must not silently move a run that was not paused."""
    state = new_run(PROMPT)
    with pytest.raises(ValueError, match="no context lock"):
        clar.accept_context_lock(state)


def test_graph_refuses_entry_while_the_lock_is_still_pending():
    """`_entry_route` says so loudly instead of burning a call and re-locking.

    Routing to the clarifier anyway would "work" — and the wiring bug would show
    up as a gate that is merely slow, which is the hardest kind to find.
    """
    state = _paused_at_lock()
    with pytest.raises(RuntimeError, match="CONTEXT_LOCK"):
        orchestrator._entry_route(state)


# ══════════════════════════════════════════════════════════════════════════
# 2. Edit — pure, deterministic, and free unless it opens a gap
# ══════════════════════════════════════════════════════════════════════════
def test_filling_a_field_makes_no_llm_call(monkeypatch):
    """THE cost test. Replacing a value must not cost a single token."""
    monkeypatch.setattr(clar, "llm_call", _explode)
    state = _paused_at_lock()

    reasons = clar.submit_context_edits(
        state, ContextEdits(fields={"cloud_provider": "GCP"})
    )

    assert reasons == []  # nothing to re-judge
    assert state.context_record.cloud_provider == "GCP"
    # The pause stays open: an edit is not an approval. The human still has to
    # look at what they changed and sign it off.
    assert state.pending_decision is PendingDecision.CONTEXT_LOCK
    assert state.input_tokens == 0 and state.output_tokens == 0


def test_striking_an_assumption_and_answering_a_question_are_free(monkeypatch):
    """Both are "filling in", so both re-freeze deterministically."""
    monkeypatch.setattr(clar, "llm_call", _explode)
    state = _paused_at_lock()

    reasons = clar.submit_context_edits(
        state,
        ContextEdits(
            struck_assumptions=["Assume English-only UI (low-stakes)."],
            answered_questions={"Is a mobile app in scope?": "No, web only."},
        ),
    )

    assert reasons == []
    assert state.context_record.assumptions == []
    assert state.context_record.open_questions == []
    # The strike is remembered, so a later re-judge cannot resurrect it.
    assert state.vetoed_assumptions == ["Assume English-only UI (low-stakes)."]
    # The answer joins the run's Q&A, where any later re-judge will read it.
    assert state.clarification_answers == {"Is a mobile app in scope?": "No, web only."}


def test_editing_is_refused_once_the_record_is_no_longer_up_for_approval():
    """After approval the record is what the design was built on. Do not rewrite it.

    Letting an edit through here would leave a blueprint and a set of ADRs that
    trace back to a Context Record which no longer exists in that form — the
    traceability the reviewer scores would be quietly false.
    """
    state = _paused_at_lock()
    clar.accept_context_lock(state)

    with pytest.raises(ValueError, match="only editable while"):
        clar.submit_context_edits(state, ContextEdits(fields={"budget": "large"}))


def test_apply_user_edits_is_pure():
    """It returns a new record and never touches the one it was given."""
    before = _record()
    after = clar.apply_user_edits(before, ContextEdits(fields={"budget": "large"}))

    assert after.budget == "large"
    assert before.budget == "medium"
    assert after is not before
    # The summary is recomputed, so an edited record can never disagree with
    # its own one-line digest.
    assert "Budget: large" in after.summary


def test_apply_user_edits_refuses_a_bad_field_or_shape():
    """A dropped edit is a veto the human believes they cast. Raise instead."""
    record = _record()
    with pytest.raises(ValueError, match="not an editable"):
        clar.apply_user_edits(record, ContextEdits(fields={"nonsense": "x"}))
    with pytest.raises(ValueError, match="list field"):
        clar.apply_user_edits(record, ContextEdits(fields={"compliance_requirements": "GDPR"}))
    with pytest.raises(ValueError, match="text field"):
        clar.apply_user_edits(record, ContextEdits(fields={"budget": ["medium"]}))


def test_emptied_critical_fields_only_reports_NEW_gaps():
    """A field that was already empty is not a gap this edit opened."""
    before = _record(budget="")
    after = clar.apply_user_edits(
        before, ContextEdits(fields={"budget": "", "cloud_provider": ""})
    )

    assert clar.emptied_critical_fields(before, after) == ["cloud_provider"]


def test_emptying_a_critical_field_asks_for_a_re_judge():
    state = _paused_at_lock()
    reasons = clar.submit_context_edits(
        state, ContextEdits(fields={"non_functional_requirements": []})
    )
    assert reasons == ["'non_functional_requirements' was cleared"]


def test_recommend_clears_the_field_and_asks_for_a_re_judge():
    """"I don't know — you recommend" is the same mechanism, pointed the other way."""
    state = _paused_at_lock()

    reasons = clar.submit_context_edits(state, ContextEdits(recommend=["cloud_provider"]))

    assert reasons == ["'cloud_provider': you asked for a recommendation"]
    assert state.context_record.cloud_provider == ""
    assert any(
        "[recommendation requested] cloud_provider" in q
        for q in state.context_record.open_questions
    )


# ══════════════════════════════════════════════════════════════════════════
# 3. The re-judge round trip: gap opened -> clarifier -> back at the lock
# ══════════════════════════════════════════════════════════════════════════
def test_edit_that_empties_a_critical_field_re_judges_without_asking(monkeypatch):
    """The full loop, through the real graph.

    The stubbed clarifier reports two critical gaps AND two matching questions,
    i.e. it would very much like to ask. It may not: a record has already been
    locked once. The gaps come back as labelled assumptions plus open questions
    on the record the human is looking at.
    """
    monkeypatch.setattr(clar, "llm_call", _missing)
    state = _paused_at_lock()

    reasons = clar.submit_context_edits(state, ContextEdits(fields={"cloud_provider": ""}))
    assert reasons  # a gap was opened
    allowed, _ = begin_user_round(state)
    assert allowed
    clar.open_for_rejudge(state)

    resumed = orchestrator.run_pipeline(state)

    # Back at the gate, NOT at a question form. `non_functional_requirements`/
    # `cloud_provider` are both REQUIRED for PROMPT, so absorbing them does
    # not make them optional — but the fresh `_missing` captured never
    # restates `compliance_requirements`, which is an `OPTIONAL_CONTEXT_
    # FIELDS` member NOT required for PROMPT, so it is what stops the
    # non-blocking optional round here first (see `clarifier.optional_slots`).
    assert resumed.stage is Stage.AWAITING_HUMAN
    assert resumed.pending_decision is PendingDecision.OPTIONAL_CONTEXT
    assert resumed.clarifying_questions == []

    record = resumed.context_record
    for gap in ("non_functional_requirements", "cloud_provider"):
        assert any(gap in a for a in record.assumptions), f"no assumption for {gap}"
        assert any(gap in q for q in record.open_questions), f"no open question for {gap}"
    assert all(a.startswith(clar.CLARIFIER_LABEL) for a in record.assumptions)


def test_accept_after_the_gate_reaches_the_researcher(monkeypatch):
    """Approving really does spend the research/design tokens it was gating."""
    monkeypatch.setattr(clar, "llm_call", _complete)
    import architect as legacy_architect
    from pipeline.agents import architect as arch
    from pipeline.agents import reviewer as rev

    legacy_architect.retrieve_chunks = lambda query, k=3: ([], "offline-test")
    monkeypatch.setattr(arch, "llm_call", _architect_response)
    monkeypatch.setattr(rev, "llm_call", lambda state, prompt, **kw: (rev.LLMJudgments(), fake_usage()))

    state = _paused_at_lock()
    clar.accept_context_lock(state)
    done = orchestrator.run_pipeline(state)

    agents = [step.agent for step in done.history]
    assert "researcher" in agents
    assert "architect" in agents
    # The approved record survived: nothing re-ran the clarifier on the way out.
    assert "clarifier" not in agents
    assert done.context_record.cloud_provider == "AWS"


# ══════════════════════════════════════════════════════════════════════════
# 4. The human-round budget — one counter, every stage
# ══════════════════════════════════════════════════════════════════════════
def test_user_rounds_are_capped_across_stages():
    """One global counter: mixing stages does not buy extra rounds.

    A per-touchpoint counter would let the same person spend three budgets by
    spreading the same number of round trips across three screens, which is
    exactly what this cap exists to prevent.
    """
    state = _paused_at_lock()

    for spent in range(MAX_USER_ROUNDS):
        # Move the stage each round to prove the counter does not care WHERE the
        # re-entry comes from, only that it is post-lock refinement.
        state.stage = Stage.AWAITING_HUMAN if spent % 2 else Stage.DONE
        allowed, reason = begin_user_round(state)
        assert allowed, f"round {spent + 1} of {MAX_USER_ROUNDS} was refused: {reason}"

    assert state.user_rounds == MAX_USER_ROUNDS
    exhausted, reason = evaluate_user_rounds(state)
    assert exhausted
    allowed, refusal = begin_user_round(state)
    assert not allowed
    assert str(MAX_USER_ROUNDS) in refusal
    # A refused round costs nothing — the counter does not creep past the cap.
    assert state.user_rounds == MAX_USER_ROUNDS


def test_answering_clarifying_questions_costs_no_round():
    """The measured regression: the budget was gone before design started.

    Run 20260818T074835Z-1925fbd7 reached the architect at 6 of 6 rounds spent,
    because four rounds of clarifying questions plus the approval had consumed
    all of it. Answering the clarifier is the pipeline working as designed, and
    `run.MAX_CLARIFICATION_ROUNDS` is what bounds a clarifier that will not
    converge — so this counter refuses to be spent before there is a record.
    """
    state = new_run(PROMPT, require_context_approval=True)
    state.stage = Stage.AWAITING_HUMAN
    state.pending_decision = PendingDecision.CLARIFICATION
    state.clarification_answers = {"Expected peak users?": "50k"}

    with pytest.raises(ValueError, match="before any Context Record was locked"):
        begin_user_round(state)

    assert state.user_rounds == 0


def test_a_post_lock_re_judge_does_cost_a_round():
    """The one thing that still counts: an edit that sends the record back."""
    state = _paused_at_lock()

    reasons = clar.submit_context_edits(state, ContextEdits(recommend=["budget"]))
    assert reasons
    allowed, _ = begin_user_round(state)

    assert allowed
    assert state.user_rounds == 1


def test_approving_the_lock_costs_no_round():
    """Approval RELEASES work that was already waiting; it does not redo any."""
    state = _paused_at_lock()

    clar.accept_context_lock(state)

    assert state.user_rounds == 0


def test_evaluate_user_rounds_is_pure():
    state = _paused_at_lock()
    assert evaluate_user_rounds(state) == (False, "")
    assert state.user_rounds == 0


# ══════════════════════════════════════════════════════════════════════════
# 5. The advisory side-channel — costs tokens, changes nothing else
# ══════════════════════════════════════════════════════════════════════════
def test_advisory_turn_changes_nothing_but_the_ledger(monkeypatch):
    """Read-only on artifacts, append-only on history. Both halves matter.

    The append-only half is not bookkeeping pedantry: `usage_by_agent()` is
    derived from `history` and promises to sum to the run totals. A call that
    spends tokens without landing there makes the whole per-agent table wrong,
    and a cost table nobody can trust is worse than no cost table.
    """
    monkeypatch.setattr(
        clar, "llm_call", lambda state, prompt, **kw: ("Scale drives the pattern.", fake_usage())
    )
    state = _paused_at_lock()
    before = state.model_copy(deep=True)

    answer = clar.ask_advisor(state, "Why does scale matter here?")

    assert answer == "Scale drives the pattern."
    # Nothing about the run moved.
    assert state.stage is before.stage
    assert state.pending_decision is before.pending_decision
    assert state.context_record == before.context_record
    assert state.clarifying_questions == before.clarifying_questions
    assert state.clarification_answers == before.clarification_answers
    assert state.blueprint is None and state.review is None
    # It does not spend a user round: understanding a record we already paid for
    # must not compete with the budget for changing it.
    assert state.user_rounds == before.user_rounds

    # Exactly one trace entry, under its own agent, carrying real tokens.
    assert len(state.history) == len(before.history) + 1
    step = state.history[-1]
    assert step.agent == clar.ADVISOR_AGENT
    assert step.input_tokens > 0 and step.output_tokens > 0
    assert step.stage_in is step.stage_out is state.stage

    # And the two ledgers still agree, which is the point of doing it at all.
    assert state.usage_by_agent()[clar.ADVISOR_AGENT].input_tokens == step.input_tokens
    assert state.input_tokens == step.input_tokens
    assert state.output_tokens == step.output_tokens


def test_the_pause_survives_any_number_of_questions(monkeypatch):
    monkeypatch.setattr(
        clar, "llm_call", lambda state, prompt, **kw: ("Teams usually pick AWS.", fake_usage())
    )
    state = _paused_at_lock()

    for _ in range(3):
        clar.ask_advisor(state, "What do teams usually pick?")

    assert state.pending_decision is PendingDecision.CONTEXT_LOCK
    assert state.stage is Stage.AWAITING_HUMAN
    assert len(state.advisory_turns) == 3
    assert state.user_rounds == 0


# ══════════════════════════════════════════════════════════════════════════
# 6. The step budget is derived, not guessed
# ══════════════════════════════════════════════════════════════════════════
def test_derive_max_steps_exceeds_a_full_budget_run(monkeypatch):
    """A run that spends EVERY refine iteration must fit inside the derived cap.

    This is the regression the derivation exists for: with a hard-coded 20, the
    day somebody raised `MAX_REFINE_ITERATIONS` the run would have died with a
    `GraphRecursionError` and been reported as FAILED — a budget change arriving
    as a crash.
    """
    import architect as legacy_architect
    from pipeline.agents import architect as arch
    from pipeline.agents import reviewer as rev

    legacy_architect.retrieve_chunks = lambda query, k=3: ([], "offline-test")
    monkeypatch.setattr(clar, "llm_call", _complete)
    monkeypatch.setattr(arch, "llm_call", _architect_response)
    # This reviewer always fails, so the loop runs to the iteration cap.
    monkeypatch.setattr(rev, "llm_call", lambda state, prompt, **kw: (rev.LLMJudgments(), fake_usage()))

    done = orchestrator.run_pipeline(new_run(PROMPT), max_steps=derive_max_steps())

    assert done.stage is Stage.DONE
    assert done.stopped_on_cap
    assert done.refine_iterations == MAX_REFINE_ITERATIONS
    assert len(done.history) < derive_max_steps(), (
        f"a full-budget run took {len(done.history)} steps, which does not fit "
        f"inside derive_max_steps()={derive_max_steps()}"
    )
    # And the default every caller gets is the derived number, not a constant
    # that has to be remembered separately.
    assert orchestrator.MAX_STEPS == derive_max_steps()


# ══════════════════════════════════════════════════════════════════════════
# 6b. The optional-context round — a small FIXED candidate set, offered only
#     when NOT currently required, resolved entirely by the caller exactly
#     like CONTEXT_LOCK itself
#
# A fresh E2E run exposed a UX problem: non-blocking gaps surfaced as
# free-text "Open questions" bolted onto the bottom of the review form, and
# one of them combined two unrelated concepts in a single sentence with no
# safe field for an answer to land in. A first fix introduced the dedicated
# `OPTIONAL_CONTEXT` pause, but reused the REQUIRED-question eligibility test
# (`_slot_is_relevant`) to decide what counts as "optional" too — which is
# not actually a required-vs-optional split at all, and could make the
# optional screen disappear entirely in exactly the scenario that motivated
# it (see `test_optional_slots_offer_relevant_but_not_required_gaps` below).
# `clarifier.OPTIONAL_CONTEXT_FIELDS` is the real fix: a small fixed
# candidate set, offered whenever empty and NOT a currently required
# question — deliberately independent of whether the prompt signals it.
# ══════════════════════════════════════════════════════════════════════════

# Brownfield/modernization + scale signal, so `non_functional_requirements`
# and `existing_systems` are genuinely REQUIRED — but no cloud/budget/
# compliance wording anywhere. Reproduces the exact e-commerce scenario from
# the review: a required scale question answered qualitatively, with
# cloud/budget/compliance left to the optional round rather than silently
# skipped because nothing in the prompt happened to mention them.
E2E_PROMPT = (
    "We're modernizing our legacy monolithic e-commerce platform to handle "
    "a much higher scale of peak sale-day traffic."
)


def _e2e_record(**overrides) -> ContextRecord:
    """The record as the REQUIRED round would lock it for `E2E_PROMPT`:
    `non_functional_requirements` answered qualitatively, `existing_systems`
    filled (brownfield), and no cloud/budget/compliance stated at all."""
    base = dict(
        business_goal="Sell products online",
        problem_statement="The legacy monolith cannot handle sale-day traffic",
        functional_requirements=["Browse and buy products online"],
        non_functional_requirements=[
            "Substantially higher peak traffic than today; exact target unknown"
        ],
        existing_systems=["Legacy monolithic e-commerce platform"],
        cloud_provider="",
        budget="",
        compliance_requirements=[],
        assumptions=[],
        open_questions=[],
    )
    base.update(overrides)
    return ContextRecord(**base)


def _paused_at_optional(**overrides) -> ArchitectState:
    """A state exactly as the clarifier leaves it when the optional round is
    owed: `cloud_provider`/`budget`/`compliance_requirements` are candidates
    and still empty; `non_functional_requirements`/`existing_systems` are
    required and already filled."""
    state = new_run(E2E_PROMPT, require_context_approval=True)
    state.context_record = _e2e_record(**overrides)
    state.stage = Stage.AWAITING_HUMAN
    state.pending_decision = PendingDecision.OPTIONAL_CONTEXT
    return state


def test_optional_slots_offer_relevant_but_not_required_gaps():
    """THE regression this whole fix is for. `non_functional_requirements`
    is REQUIRED (relevant) and already filled, so it is never re-offered;
    `cloud_provider`/`budget`/`compliance_requirements` are NOT required (no
    signal in `E2E_PROMPT` at all) and still empty, so all three are offered
    — exactly the scenario that could skip the optional screen entirely
    under the old (required-eligibility-reused) policy, since nothing in the
    prompt signalled any of them."""
    state = new_run(E2E_PROMPT)
    record = _e2e_record()
    assert clar.optional_slots(state, record) == [
        "cloud_provider", "budget", "compliance_requirements",
    ]


def test_optional_slots_exclude_a_currently_required_missing_field():
    """Negative control: a field that IS currently required (relevant) and
    still missing must not ALSO show up as optional — required-vs-optional
    stays a semantic split, never a second chance at the same question, even
    when the required round skipped it or the ask cap absorbed it."""
    state = new_run(PROMPT)  # module PROMPT signals scale + cloud
    record = _record(non_functional_requirements=[], cloud_provider="")
    slots = clar.optional_slots(state, record)
    assert "non_functional_requirements" not in slots
    assert "cloud_provider" not in slots


def test_optional_slots_exclude_a_filled_optional_field():
    """Negative control: a candidate field that already has a value is never
    re-offered, regardless of relevance."""
    state = new_run(E2E_PROMPT)
    assert clar.optional_slots(state, _e2e_record(cloud_provider="AWS")) == [
        "budget", "compliance_requirements",
    ]


def test_optional_slots_never_offers_always_required_fields():
    """The candidate set is small and fixed — business_goal/problem_statement/
    functional_requirements are never offered here, empty or not: a design
    cannot proceed without them, so they are always required, never optional."""
    state = new_run(E2E_PROMPT)
    record = _e2e_record(
        business_goal="", problem_statement="", functional_requirements=[]
    )
    slots = clar.optional_slots(state, record)
    assert "business_goal" not in slots
    assert "problem_statement" not in slots
    assert "functional_requirements" not in slots


def test_optional_slots_never_offer_existing_systems():
    """`existing_systems` is never a candidate, even on a greenfield record
    where it is empty and not relevant — it is either a required brownfield
    question or there is nothing greenfield-specific worth asking."""
    state = new_run("Build a brand-new internal tool from scratch.")
    record = _e2e_record(existing_systems=[])
    assert "existing_systems" not in clar.optional_slots(state, record)


def test_optional_slots_exclude_fields_already_filled():
    state = new_run(E2E_PROMPT)
    assert clar.optional_slots(state, _e2e_record(
        cloud_provider="AWS", budget="medium", compliance_requirements=["GDPR"],
    )) == []


def test_optional_slots_recomputed_fresh_reflects_only_current_gaps():
    """The 'genuinely newly relevant' guarantee, structurally: `optional_
    slots` reads the CURRENT record, never a cached list, so a field that
    already got an answer (from an earlier optional round, or any edit)
    never reappears — without any special-cased revision logic."""
    state = new_run(E2E_PROMPT)
    before = clar.optional_slots(state, _e2e_record())
    assert set(before) == {"cloud_provider", "budget", "compliance_requirements"}

    after = clar.optional_slots(state, _e2e_record(cloud_provider="AWS"))
    assert after == ["budget", "compliance_requirements"]


def test_skipping_the_optional_round_advances_to_context_lock_unfilled():
    state = _paused_at_optional()

    clar.resolve_optional_context(state)

    assert state.pending_decision is PendingDecision.CONTEXT_LOCK
    assert state.context_record.cloud_provider == ""
    assert state.context_record.budget == ""
    assert state.context_record.compliance_requirements == []
    assert state.optional_context_resolved_version == state.context_record.version


def test_answering_the_optional_round_writes_only_the_named_field():
    """Answering only compliance writes only `compliance_requirements`."""
    state = _paused_at_optional()

    clar.resolve_optional_context(
        state,
        ContextEdits(fields={"compliance_requirements": ["GDPR"]}),
    )

    assert state.pending_decision is PendingDecision.CONTEXT_LOCK
    assert state.context_record.compliance_requirements == ["GDPR"]
    # The fields nobody answered stay exactly as they were — untouched, never guessed.
    assert state.context_record.cloud_provider == ""
    assert state.context_record.budget == ""


def test_optional_answer_cannot_cross_populate_a_different_field():
    """A cloud answer must never land in budget or compliance — structurally
    guaranteed by `ContextEdits.fields` being keyed by field name, the same
    safe channel a direct review-screen edit already uses. Answering only
    cloud writes only `cloud_provider`."""
    state = _paused_at_optional()

    clar.resolve_optional_context(
        state,
        ContextEdits(fields={"cloud_provider": "GCP"}),
    )

    assert state.context_record.cloud_provider == "GCP"
    assert state.context_record.budget == ""
    assert state.context_record.compliance_requirements == []


def test_cloud_provider_answer_maps_only_to_cloud_provider():
    state = _paused_at_optional()

    clar.resolve_optional_context(state, ContextEdits(fields={"cloud_provider": "GCP"}))

    assert state.context_record.cloud_provider == "GCP"
    assert state.context_record.budget == ""  # untouched, not guessed
    assert state.context_record.compliance_requirements == []  # untouched


def test_review_screen_never_calls_an_answered_optional_field_unresolved():
    """After Continue, an answered optional field must not still carry a
    deterministic 'unresolved'/'not confirmed' marker on the record the
    human is about to review at CONTEXT_LOCK."""
    state = _paused_at_optional()

    clar.resolve_optional_context(state, ContextEdits(fields={"cloud_provider": "GCP"}))

    record = state.context_record
    assert not any("(cloud_provider)" in a for a in record.assumptions)
    assert not any("(cloud_provider)" in q for q in record.open_questions)


def test_answering_the_optional_round_clears_a_stale_recommend_marker():
    """Edge case: a human can reach the optional round for a candidate field
    that ALSO carries a stale '[recommendation requested] ... (field)'
    marker from an earlier "you recommend" edit that forced it blank and
    re-judged (see `_freeze_context_record`'s `critical_recommend`
    handling). Filling the field here must clear that marker too — an
    answered field must never still read as unresolved."""
    state = _paused_at_optional(
        assumptions=[
            f"{clar.CLARIFIER_LABEL} [recommendation requested] budget or cost "
            "constraint (budget): not safe to auto-recommend — filling this "
            "would require an architecture/domain decision only a human can "
            "make.",
        ],
        open_questions=[
            "budget or cost constraint (budget) — you asked the clarifier to "
            "recommend this, but it requires an architecture/domain decision "
            "the clarifier cannot make; still unconfirmed.",
        ],
    )

    clar.resolve_optional_context(
        state, ContextEdits(fields={"budget": "small, cost-sensitive"})
    )

    assert state.context_record.budget == "small, cost-sensitive"
    assert not any("(budget)" in a for a in state.context_record.assumptions)
    assert not any("(budget)" in q for q in state.context_record.open_questions)


def test_resolving_when_nothing_is_pending_is_refused():
    state = new_run(PROMPT)
    with pytest.raises(ValueError, match="no optional-context round"):
        clar.resolve_optional_context(state)


def test_graph_refuses_entry_while_the_optional_round_is_still_pending():
    state = _paused_at_optional()
    with pytest.raises(RuntimeError, match="OPTIONAL_CONTEXT"):
        orchestrator._entry_route(state)


def test_resolving_optional_context_makes_no_llm_call(monkeypatch):
    monkeypatch.setattr(clar, "llm_call", _explode)
    state = _paused_at_optional()

    clar.resolve_optional_context(state, ContextEdits(fields={"cloud_provider": "AWS"}))

    assert state.pending_decision is PendingDecision.CONTEXT_LOCK


def test_required_and_optional_rounds_never_ask_the_same_field_in_one_pass(
    monkeypatch,
):
    """The required-question branch returns BEFORE the record is even frozen,
    so within one `clarifier_node` pass a field is asked at most once, never
    both as a required question and as an optional one."""
    monkeypatch.setattr(clar, "llm_call", _missing)

    out = clar.clarifier_node(new_run(PROMPT, require_context_approval=True))

    assert out["pending_decision"] is PendingDecision.CLARIFICATION
    assert "context_record" not in out  # no record frozen -> no optional round either


def test_already_resolved_optional_round_is_not_shown_again(monkeypatch):
    """Defensive: once the watermark covers a record's version, the round is
    not shown again for that version even though an optional-eligible gap
    remains (`compliance_requirements` — not required for PROMPT, and the
    fresh `_missing` captured never restates it) — this is what makes
    'resolved' durable rather than a one-shot flag a reload could re-trip."""
    monkeypatch.setattr(clar, "llm_call", _missing)
    state = _paused_at_lock(non_functional_requirements=[], cloud_provider="")
    # Pre-resolved for the NEXT version this re-judge is about to produce.
    state.optional_context_resolved_version = state.context_record.version + 1

    out = clar.clarifier_node(state)

    assert out["context_record"].version == state.optional_context_resolved_version
    assert out["pending_decision"] is PendingDecision.CONTEXT_LOCK


def test_clarifier_skips_the_optional_round_when_nothing_is_left_to_ask(monkeypatch):
    """A fresh lock where the model captured everything relevant goes
    straight to CONTEXT_LOCK — no empty optional screen."""
    monkeypatch.setattr(clar, "llm_call", _complete)

    out = clar.clarifier_node(new_run(PROMPT, require_context_approval=True))

    assert out["pending_decision"] is PendingDecision.CONTEXT_LOCK


def test_optional_context_state_round_trips_through_a_checkpoint():
    state = _paused_at_optional()
    clar.resolve_optional_context(state, ContextEdits(fields={"cloud_provider": "GCP"}))

    save_state(state)
    loaded = load_state(state.run_id)

    assert loaded == state
    assert loaded.pending_decision is PendingDecision.CONTEXT_LOCK
    assert loaded.optional_context_resolved_version == loaded.context_record.version
    assert loaded.context_record.cloud_provider == "GCP"


# ══════════════════════════════════════════════════════════════════════════
# 7. Persistence — new fields round-trip, old checkpoints still load
# ══════════════════════════════════════════════════════════════════════════
def test_new_gate_fields_round_trip_through_a_checkpoint():
    state = _paused_at_lock()
    state.user_rounds = 2
    state.vetoed_assumptions = ["Assume English-only UI (low-stakes)."]
    clar.submit_context_edits(state, ContextEdits(fields={"budget": "large"}))

    save_state(state)
    loaded = load_state(state.run_id)

    assert loaded == state
    assert loaded.pending_decision is PendingDecision.CONTEXT_LOCK
    assert loaded.require_context_approval is True
    assert loaded.user_rounds == 2
    assert loaded.vetoed_assumptions == ["Assume English-only UI (low-stakes)."]
    assert loaded.context_record.budget == "large"


def test_a_checkpoint_written_before_this_change_still_loads(tmp_path, monkeypatch):
    """The reason `AWAITING_HUMAN` kept the string value `"awaiting_input"`.

    This is a checkpoint as it was written BEFORE the gate existed: the old
    stage literal, and not one of the new fields. Renaming the enum MEMBER is a
    source change; renaming its VALUE would have been a data migration, and
    every run already in `.cache/runs/` would have stopped loading.
    """
    monkeypatch.setenv("AI_ARCHITECT_RUNS_DIR", str(tmp_path))
    run_id = "20260101T000000Z-deadbeef"
    legacy = (
        '{"initial_request": {"raw_prompt": "old run", "repo_url": ""}, '
        f'"run_id": "{run_id}", '
        '"stage": "awaiting_input", '
        '"clarifying_questions": ["Expected peak users?"]}'
    )
    directory = tmp_path / run_id
    directory.mkdir()
    (directory / "003_awaiting_input.json").write_text(legacy, encoding="utf-8")

    loaded = load_state(run_id)

    assert loaded.stage is Stage.AWAITING_HUMAN
    assert loaded.clarifying_questions == ["Expected peak users?"]
    # The fields that did not exist when it was written take their defaults, and
    # those defaults are the pre-gate behaviour: nothing pending, no approval
    # required, no rounds spent. An old run resumes as the run it was.
    assert loaded.pending_decision is None
    assert loaded.require_context_approval is False
    assert loaded.user_rounds == 0
    assert loaded.vetoed_assumptions == []
    assert loaded.advisory_turns == []


# ══════════════════════════════════════════════════════════════════════════
# System-managed fields are not user-editable at the gate
#
# Run 20260821T103837Z-a7ae0a73 checkpointed `"version": "1"` — a string. The
# path was `version` sitting in EDITABLE_RECORD_FIELDS while the edit applier
# stringifies values without schema validation, so a gate pass could turn the
# system's integer version counter into text and crash the DONE screen's
# `record.version <= 1`. `version`/`revision_reason` are now excluded at the
# allowlist; the UI coercion in ui_sections._version_number stays as a
# compatibility layer for runs that already carry the stringified value.
# ══════════════════════════════════════════════════════════════════════════

def test_version_and_revision_reason_are_not_editable_fields():
    assert "version" not in clar.EDITABLE_RECORD_FIELDS
    assert "revision_reason" not in clar.EDITABLE_RECORD_FIELDS
    # Ordinary domain fields stay editable, list and string alike.
    for name in ("budget", "cloud_provider", "compliance_requirements",
                 "non_functional_requirements"):
        assert name in clar.EDITABLE_RECORD_FIELDS


def test_version_cannot_be_overwritten_through_the_gate():
    """The exact write that produced the checkpointed string "1" now raises."""
    record = _record()
    assert record.version == 1

    with pytest.raises(ValueError, match="not an editable"):
        clar.apply_user_edits(record, ContextEdits(fields={"version": "1"}))


def test_version_stays_an_integer_after_a_normal_edit_pass():
    record = _record()
    assert record.version == 1

    after = clar.apply_user_edits(
        record,
        ContextEdits(
            fields={"budget": "large", "compliance_requirements": ["GDPR", "SOC 2"]}
        ),
    )

    assert after.budget == "large"
    assert after.compliance_requirements == ["GDPR", "SOC 2"]
    assert after.version == 1
    assert isinstance(after.version, int)  # not "1" — the crash is gone at the source
    assert before_intact(record)  # purity of the untouched original still holds


def before_intact(record) -> bool:
    return record.budget == "medium" and record.version == 1


def test_revision_reason_cannot_be_user_overwritten():
    """Why a version exists is `_freeze_context_record`'s words, not the user's."""
    record = _record(version=2, revision_reason="user corrected the budget")

    with pytest.raises(ValueError, match="not an editable"):
        clar.apply_user_edits(
            record, ContextEdits(fields={"revision_reason": "no reason"})
        )


def test_recommend_channel_also_refuses_system_managed_fields():
    """"You recommend" is the same write path — it must not reach them either."""
    record = _record()

    with pytest.raises(ValueError, match="nothing to recommend"):
        clar.apply_user_edits(record, ContextEdits(recommend=["version"]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
