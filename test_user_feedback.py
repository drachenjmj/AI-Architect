"""test_user_feedback.py — iterative refinement from the human at DONE (Kati).

No API key, no network: the Clarifier, Architect and Reviewer calls are the same
canned responses the rest of the suite uses.

WHAT IS ACTUALLY BEING PINNED
------------------------------
This feature's failure mode is not a crash. It is a box that takes a paragraph
from a person and quietly does nothing with it, and every test here is aimed at
one of the ways that can happen:

  * the run re-enters the graph and the stale cost caps send it straight back to
    DONE without the architect ever seeing the text (`stopped_on_cap`);
  * a second correction overwrites the first under a shared key;
  * the gate hands back a design from before the feedback, reverting it;
  * the architect faithfully PRESERVES the very thing the user asked to change,
    because the preservation rule only ever yielded to reviewer findings;
  * the round budget is spent, the box still accepts text, and it goes nowhere.

The one thing an offline test cannot pin is whether the MODEL obeys the
directive it is handed. What it can pin — and does — is that the directive
reaches the prompt, in its own block, ranked above the reviewer's instruction,
with the current design present to revise and an explicit licence to change what
it names. Whether the model then complies is measured on a live run.
"""
from __future__ import annotations

import json

import pytest

import test_clarifier as tc  # canned Clarifier + Architect responses
from pipeline import orchestrator
from pipeline import user_feedback as fb
from pipeline.agents import architect as arch
from pipeline.agents import clarifier as clar
from pipeline.agents import reviewer as rev
from pipeline.persistence import load_state, save_state
from pipeline.refine_gate import (
    MAX_REFINE_ITERATIONS,
    MAX_USER_ROUNDS,
    evaluate_user_rounds,
    refine_gate_node,
    reopen_for_user_round,
)
from pipeline.state import (
    ADR,
    ArchitectState,
    Blueprint,
    ComponentDescription,
    ContextRecord,
    DesignSnapshot,
    Feature,
    PendingDecision,
    ReviewResult,
    RubricScores,
    Stage,
    UserFeedback,
    new_run,
)

PROMPT = "Fix our monolithic online shop so it survives seasonal peak load."

REQUIREMENTS_TEXT = "Peak load is 500k users, not 50k, and we are on Azure."
DESIGN_TEXT = "Use SQS instead of Kafka — we have nobody to run a broker."


# ══════════════════════════════════════════════════════════════════════════
# Mocks and builders
# ══════════════════════════════════════════════════════════════════════════
def _failing_review(state, prompt, **kwargs):
    """All-default judgments (every criterion False) → the reviewer fails."""
    return rev.LLMJudgments(), tc.fake_usage()


def _install_mocks() -> None:
    """Point every agent at its canned response. Called by each test."""
    import architect as legacy_architect

    clar.llm_call = tc._complete
    arch.llm_call = tc._architect_response
    rev.llm_call = _failing_review
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


def _seed_designable_state(**overrides) -> ArchitectState:
    """A run parked at RESEARCHING with a locked record — the Architect runs next."""
    state = new_run(PROMPT, require_context_approval=True)
    state.context_record = ContextRecord(
        project_name="Seasonal Shop",
        business_goal="Sell sneakers online",
        problem_statement="The monolith crashes on sale days.",
        cloud_provider="AWS",
        summary="cloud: AWS; scale: 50k peak users; compliance: GDPR",
    )
    state.stage = Stage.RESEARCHING
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def _done_capped_state() -> ArchitectState:
    """A REAL finished run that stopped on the refine budget.

    Driven through the graph rather than hand-built, because the thing under
    test is precisely what a capped run leaves behind: `stopped_on_cap` True,
    `refine_iterations` at the cap, an incumbent in `best_design`, and a review
    that still wants refinement.
    """
    _install_mocks()
    state = orchestrator.run_pipeline(_seed_designable_state())
    assert state.stage is Stage.DONE
    assert state.stopped_on_cap is True
    assert state.refine_iterations == MAX_REFINE_ITERATIONS
    return state


def _done_passed_state() -> ArchitectState:
    """A finished run whose review PASSED, built directly.

    Hand-built on purpose: driving a genuine pass through the deterministic
    checks would couple this file to the canned design's rubric score, and the
    property under test — a directive on a design nobody objected to still
    revises that design instead of starting over — is about the architect, not
    about how the pass was reached.
    """
    state = _seed_designable_state()
    state.stage = Stage.DONE
    state.features = [
        Feature(id="FEAT-001", name="Survive peak load", scenario="Stays up at peak.")
    ]
    state.blueprint = Blueprint(
        project_name="Seasonal Shop",
        selected_pattern="Event-Driven Microservices",
        stakeholder_view="Customers can always check out.",
        technical_view="Kafka carries order events between services.",
        components=["Order Service", "Kafka Event Bus"],
        addressed_feature_ids=["FEAT-001"],
    )
    state.adrs = [
        ADR(
            id="ADR-001",
            title="ADR-1: Use Kafka as the event backbone",
            decision="Use Kafka.",
            related_feature_ids=["FEAT-001"],
            related_component_names=["Kafka Event Bus"],
        )
    ]
    state.components = [
        ComponentDescription(
            id="COMP-001",
            name="Kafka Event Bus",
            description="Carries order events.",
            related_feature_ids=["FEAT-001"],
            related_adr_ids=["ADR-001"],
        )
    ]
    state.review = ReviewResult(
        overall_status="pass",
        rubric_scores=RubricScores(
            all_artifacts_present=2,
            constraint_coverage=2,
            traceability=2,
            adr_presence=2,
            repo_grounding=True,
            flaw_detection=True,
            adr_soundness=True,
            best_practice_grounding=True,
            refinement_readiness=True,
        ),
        requires_refinement=False,
        refinement_instruction="",
    )
    return state


def _steps_after_feedback(state: ArchitectState) -> list[str]:
    """Agent names of every step logged after the LAST feedback submission."""
    marks = [
        index
        for index, step in enumerate(state.history)
        if step.agent == fb.FEEDBACK_AGENT
    ]
    assert marks, "no user-feedback step was logged"
    return [step.agent for step in state.history[marks[-1] + 1 :]]


# ══════════════════════════════════════════════════════════════════════════
# 1. The regression: a capped run must still reach the architect
# ══════════════════════════════════════════════════════════════════════════
def test_design_feedback_on_a_capped_run_reaches_the_architect():
    """THE test for the silent no-op that "just set REFINING and re-run" would be.

    A run that ended on the cap re-enters at the refine gate with
    `stopped_on_cap` True and `refine_iterations` at the ceiling. Without the
    re-entry reset the gate re-trips the same cap on its first visit, routes
    straight to END, and the run returns to DONE with the human's text unread —
    no error, no trace, nothing to notice.
    """
    state = _done_capped_state()
    designs_before = sum(1 for step in state.history if step.agent == "architect")

    accepted, why = fb.submit_feedback(state, design=DESIGN_TEXT)
    assert accepted, why
    assert state.stage is Stage.REFINING
    assert state.stopped_on_cap is False
    assert state.refine_iterations == 0

    state = orchestrator.run_pipeline(state)

    ran = _steps_after_feedback(state)
    assert "architect" in ran, "the directive never reached the architect"
    designs_after = sum(1 for step in state.history if step.agent == "architect")
    assert designs_after > designs_before
    # Consumed exactly once, by the pass that acted on it.
    assert [entry.status for entry in state.user_feedback] == ["applied"]
    assert state.stage is Stage.DONE


def test_capped_run_without_the_reset_would_have_gone_straight_back_to_done():
    """The same state, reset skipped: the gate stops on the stale cap.

    Pins the defect itself rather than only its fix, so a later refactor that
    drops the reset fails HERE with a clear reason instead of silently restoring
    the no-op.
    """
    state = _done_capped_state()
    # Feedback recorded by hand, deliberately WITHOUT `reopen_for_user_round`.
    state.user_feedback.append(UserFeedback(text=DESIGN_TEXT, kind="design", round=1))
    state.stage = Stage.REFINING

    out = refine_gate_node(state)

    assert out["stage"] is Stage.DONE          # the no-op, reproduced
    assert out["stopped_on_cap"] is True
    assert state.user_feedback[0].status == "pending"  # text still unread


# ══════════════════════════════════════════════════════════════════════════
# 2. The reset itself
# ══════════════════════════════════════════════════════════════════════════
def _reset_delta(state: ArchitectState) -> set[str]:
    """Which top-level state fields `reopen_for_user_round` changed."""
    before = state.model_dump()
    allowed, why = reopen_for_user_round(state)
    assert allowed, why
    after = state.model_dump()
    return {name for name in before if before[name] != after[name]}


def test_reset_clears_the_full_key_set_on_the_design_path():
    state = _done_capped_state()
    assert _reset_delta(state) == {
        "user_rounds",
        "stopped_on_cap",
        "refine_iterations",
        "best_design",
        "selected_round",
    }
    assert state.user_rounds == 1
    assert state.stopped_on_cap is False
    assert state.refine_iterations == 0
    assert state.best_design is None
    assert state.selected_round == 0


def test_reset_clears_the_full_key_set_on_the_requirements_path():
    """Identical clearing, identical assertion — the reset is NOT path-dependent.

    `best_design` in particular is cleared on both paths for reasons that differ
    only in wording: here the incumbent was scored against a record that is about
    to be superseded, on the design path it answers a question the user has since
    changed. Neither is comparable to what comes next.
    """
    state = _done_capped_state()
    state.best_design = DesignSnapshot(round=1, review=state.review)
    state.selected_round = 1

    assert _reset_delta(state) == {
        "user_rounds",
        "stopped_on_cap",
        "refine_iterations",
        "best_design",
        "selected_round",
    }


def test_reset_goes_through_begin_user_round(monkeypatch):
    """One writer for `user_rounds`, and the cap check is not re-implemented."""
    import pipeline.refine_gate as gate

    calls: list[ArchitectState] = []
    real = gate.begin_user_round

    def _spy(state):
        calls.append(state)
        return real(state)

    monkeypatch.setattr(gate, "begin_user_round", _spy)

    state = _done_capped_state()
    accepted, _ = fb.submit_feedback(state, design=DESIGN_TEXT)

    assert accepted
    assert len(calls) == 1 and calls[0] is state


def test_refused_round_records_nothing():
    """Out of budget: the state is untouched, in every visible way.

    "Never accept text and silently drop it" has a second half — never half-apply
    it either. A refusal that had already appended the feedback, or already
    cleared the caps, would leave a run claiming a round it never ran.
    """
    state = _done_capped_state()
    state.user_rounds = MAX_USER_ROUNDS
    before = state.model_dump()

    accepted, why = fb.submit_feedback(
        state, requirements=REQUIREMENTS_TEXT, design=DESIGN_TEXT
    )

    assert accepted is False
    assert "max_user_rounds" in why and str(MAX_USER_ROUNDS) in why
    assert state.model_dump() == before
    assert state.user_feedback == []
    assert state.user_rounds == MAX_USER_ROUNDS
    assert state.stage is Stage.DONE


def test_cap_blocks_further_submissions_visibly():
    """The cap is reachable by spending it, and it says why before it is hit.

    `evaluate_user_rounds` is the same function the UI disables both boxes on
    (see `ui_sections.render_feedback_box`), so the caption a person reads and
    the refusal a caller gets cannot disagree.
    """
    state = _done_capped_state()
    for spent in range(MAX_USER_ROUNDS):
        assert evaluate_user_rounds(state) == (False, "")
        accepted, _ = fb.submit_feedback(state, design=f"change {spent}")
        assert accepted
        state.stage = Stage.DONE  # stand in for the run that would follow

    exhausted, why = evaluate_user_rounds(state)
    assert exhausted
    assert f"max_user_rounds ({MAX_USER_ROUNDS})" == why
    assert fb.submit_feedback(state, design="one more") == (False, why)
    assert len(state.user_feedback) == MAX_USER_ROUNDS


def test_empty_submission_is_refused_loudly():
    state = _done_capped_state()
    with pytest.raises(ValueError, match="both boxes empty"):
        fb.submit_feedback(state, requirements="   ", design="")
    assert state.user_rounds == 0


def test_feedback_is_only_collected_at_done():
    state = _seed_designable_state()  # mid-run
    with pytest.raises(ValueError, match="collected at DONE"):
        fb.submit_feedback(state, design=DESIGN_TEXT)


# ══════════════════════════════════════════════════════════════════════════
# 3. The requirements path
# ══════════════════════════════════════════════════════════════════════════
def test_two_requirements_corrections_both_survive():
    """A unique key per submission — a fixed one would drop the first silently.

    This is the same failure the feature exists to prevent, one level down: the
    clarifier would re-judge against the correction the person had already moved
    on from, and the one they just typed would be gone with nothing to notice.
    """
    state = _done_capped_state()

    assert fb.submit_feedback(state, requirements="Actually we are on Azure.")[0]
    state.stage = Stage.DONE  # stand in for the run between the two corrections
    assert fb.submit_feedback(state, requirements="Peak load is 500k, not 50k.")[0]

    corrections = {
        key: value
        for key, value in state.clarification_answers.items()
        if key.startswith("User correction")
    }
    assert len(corrections) == 2
    assert set(corrections.values()) == {
        "Actually we are on Azure.",
        "Peak load is 500k, not 50k.",
    }
    assert [entry.round for entry in state.user_feedback] == [1, 2]


def test_requirements_feedback_versions_the_record_and_pauses_at_the_gate():
    state = _done_capped_state()
    original = state.context_record
    assert original.version == 1

    assert fb.submit_feedback(state, requirements=REQUIREMENTS_TEXT)[0]
    assert state.stage is Stage.AWAITING_HUMAN
    assert state.pending_decision is PendingDecision.CLARIFICATION

    state = orchestrator.run_pipeline(state)

    # Re-frozen as a NEW VERSION, not edited in place.
    assert state.context_record.version == 2
    assert state.context_record.revision_reason == REQUIREMENTS_TEXT
    assert [old.version for old in state.context_history] == [1]
    assert state.context_history[0] == original
    # And parked in front of the human, before the expensive work runs.
    assert state.stage is Stage.AWAITING_HUMAN
    assert state.pending_decision is PendingDecision.CONTEXT_LOCK
    assert [entry.status for entry in state.user_feedback] == ["applied"]


def test_requirements_feedback_asks_nothing_and_labels_the_gap():
    """Ask once, then assume — a correction cannot reopen a round of questions.

    The clarifier here reports two critical gaps AND matching questions (the
    well-behaved case). A record already exists, so `_may_ask` is False: the gaps
    become LABELLED assumptions on a record the human is about to be shown,
    rather than a fresh interrogation of someone who just told us something.
    """
    state = _done_capped_state()
    clar.llm_call = tc._missing  # gaps + questions on the re-judge

    assert fb.submit_feedback(state, requirements=REQUIREMENTS_TEXT)[0]
    state = orchestrator.run_pipeline(state)

    assert state.clarifying_questions == []
    # `non_functional_requirements`/`cloud_provider` both became REQUIRED by
    # this point (an earlier answer added cloud wording to the signal text —
    # see `clarifier._signal_text`), so absorbing them does not make them
    # optional — but `compliance_requirements` is an `OPTIONAL_CONTEXT_
    # FIELDS` member the fresh `_missing` captured never restates and is
    # still not required, so it is what stops the non-blocking optional
    # round here FIRST, before the review screen (see `clarifier.optional_slots`).
    assert state.pending_decision is PendingDecision.OPTIONAL_CONTEXT
    record = state.context_record
    assert record.version == 2
    for gap in ("non_functional_requirements", "cloud_provider"):
        assert any(gap in line for line in record.assumptions), gap
        assert any(gap in line for line in record.open_questions), gap
    assert all(a.startswith(clar.CLARIFIER_LABEL) for a in record.assumptions)


def test_requirements_change_re_runs_research_and_a_design_only_one_does_not():
    """Retrieval is re-run against the NEW record, and never for nothing.

    A requirements correction changes what the design must satisfy, so the KB
    query changes with it. A design directive changes nothing the researcher
    reads, so paying for retrieval again would buy the same chunks twice.
    """
    _install_mocks()
    state = _done_capped_state()

    # ── requirements: clarifier → (approve) → researcher → architect ──
    assert fb.submit_feedback(state, requirements=REQUIREMENTS_TEXT)[0]
    state = orchestrator.run_pipeline(state)
    clar.accept_context_lock(state)
    state = orchestrator.run_pipeline(state)

    ran = _steps_after_feedback(state)
    assert ran.count("clarifier") >= 1
    assert ran.count("researcher") >= 1
    assert "architect" in ran

    # ── design only: straight to the gate and the architect ──
    state.stage = Stage.DONE
    assert fb.submit_feedback(state, design=DESIGN_TEXT)[0]
    state = orchestrator.run_pipeline(state)

    ran = _steps_after_feedback(state)
    assert "researcher" not in ran
    assert "clarifier" not in ran
    assert "architect" in ran


def test_requirements_correction_drops_the_artifacts_derived_from_the_old_record():
    """Features and the review belong to the record that was just superseded.

    Keeping them would have the architect reuse a v1 feature set, under a v1
    reviewer instruction naming v1 feature IDs, while designing against a v2
    record. Cleared only on this path: a design directive supersedes nothing, so
    there the findings still apply and travel with it.
    """
    state = _done_capped_state()
    assert state.features and state.review is not None

    assert fb.submit_feedback(state, requirements=REQUIREMENTS_TEXT)[0]
    assert state.features == []
    assert state.review is None

    design_only = _done_capped_state()
    assert fb.submit_feedback(design_only, design=DESIGN_TEXT)[0]
    assert design_only.features
    assert design_only.review is not None


# ══════════════════════════════════════════════════════════════════════════
# 4. Both boxes in one submission
# ══════════════════════════════════════════════════════════════════════════
def test_both_boxes_requirements_first_design_still_pending():
    """One round, two entries, and an order that is not a preference.

    The record is the ground truth the features and the design are derived from,
    so it has to land first; the directive waits for the architect pass that
    follows the re-freeze and is consumed there.
    """
    _install_mocks()
    state = _done_capped_state()

    accepted, why = fb.submit_feedback(
        state, requirements=REQUIREMENTS_TEXT, design=DESIGN_TEXT
    )
    assert accepted, why
    assert state.user_rounds == 1, "one submission is one round, not two"
    assert [entry.kind for entry in state.user_feedback] == ["requirements", "design"]
    # The requirements route wins the routing: the record lands first.
    assert state.stage is Stage.AWAITING_HUMAN
    assert state.pending_decision is PendingDecision.CLARIFICATION

    state = orchestrator.run_pipeline(state)  # clarifier re-freezes, pauses

    applied = {entry.kind: entry.status for entry in state.user_feedback}
    assert applied == {"requirements": "applied", "design": "pending"}
    assert state.context_record.version == 2

    clar.accept_context_lock(state)
    state = orchestrator.run_pipeline(state)  # researcher → architect → …

    applied = {entry.kind: entry.status for entry in state.user_feedback}
    assert applied == {"requirements": "applied", "design": "applied"}


# ══════════════════════════════════════════════════════════════════════════
# 5. The architect prompt: two blocks, ranked, and a licence to change
# ══════════════════════════════════════════════════════════════════════════
def test_user_directive_is_its_own_block_and_outranks_the_instruction():
    state = _done_capped_state()
    state.review.refinement_instruction = "Assign FEAT-005 to a component."
    assert fb.submit_feedback(state, design=DESIGN_TEXT)[0]

    prompt = arch._build_architecture_prompt(state, state.features)

    assert "<user_directive>" in prompt
    assert DESIGN_TEXT in prompt
    assert "<refinement_instruction>" in prompt
    assert "Assign FEAT-005 to a component." in prompt
    # Two blocks, not one merged block: the directive is not inside the
    # instruction and the instruction is not inside the directive.
    directive = prompt.split("<user_directive>")[1].split("</user_directive>")[0]
    instruction = prompt.split("<refinement_instruction>")[1]
    assert DESIGN_TEXT in directive and "FEAT-005" not in directive
    assert "FEAT-005" in instruction and DESIGN_TEXT not in instruction
    # Ranked, in the one place the model will read it.
    assert "<user_directive>" in arch.ARCHITECTURE_SYSTEM_PROMPT
    assert "OUTRANKS" in arch.ARCHITECTURE_SYSTEM_PROMPT


def test_user_text_is_never_written_into_the_review():
    """Single-writer provenance on `refinement_instruction`, end to end.

    That field answers "who asked for this change?" only because everything in
    it came from the reviewer. Overloading it with user text would have been the
    smaller change and would have cost the trail its only answer.
    """
    _install_mocks()
    state = _done_capped_state()
    assert fb.submit_feedback(state, design=DESIGN_TEXT)[0]
    state = orchestrator.run_pipeline(state)

    assert state.review is not None
    assert DESIGN_TEXT not in state.review.refinement_instruction
    for snapshot in [state.best_design] if state.best_design else []:
        assert DESIGN_TEXT not in snapshot.review.refinement_instruction


def test_directive_on_a_passed_design_revises_it_rather_than_starting_over():
    """A directive naming a component must survive the PRESERVATION rule.

    Two halves, both code-checkable offline:

    * the current design is in the prompt, so the rule is in play at all — this
      is the half that breaks without `_is_revision`, because the review PASSED
      and nothing else would mark the pass as a revision. The architect would
      rebuild from the Context Record and throw away the design the person was
      commenting on;
    * the rule carries an explicit third exit for anything the directive names,
      so "keep every decision the findings do not mention" cannot be read as
      "keep the Kafka bus the user just asked you to replace".

    Whether the model then complies is a live-run measurement, not a unit test.
    """
    state = _done_passed_state()
    assert fb.submit_feedback(state, design=DESIGN_TEXT)[0]

    assert arch._is_revision(state) is True
    assert arch._reuses_features(state) is True   # IDs stay stable across the change

    prompt = arch._build_architecture_prompt(state, state.features)
    assert "<current_design>" in prompt
    assert "Kafka Event Bus" in prompt            # the thing being redirected
    assert DESIGN_TEXT in prompt

    # Flattened, because the rule is wrapped prose: a phrase that happens to
    # straddle a line break is still the same instruction to the model.
    rule = " ".join(arch.ARCHITECTURE_SYSTEM_PROMPT.split())
    assert "is a licence to change" in rule
    assert "Everything the user did NOT mention still keeps its names" in rule


# ══════════════════════════════════════════════════════════════════════════
# 6. The gate must not revert the directive it was asked to serve
# ══════════════════════════════════════════════════════════════════════════
def test_gate_does_not_renominate_the_design_the_user_just_redirected():
    """Clearing `best_design` is not enough on its own.

    The first gate visit after design feedback does NOT follow a reviewer
    verdict: the artifacts on the state are the ones the human objected to and
    the review is the stale verdict on them. Without this guard the gate
    re-nominates that design as the incumbent immediately, and two rounds later
    hands it back — silently reverting the change that was asked for.
    """
    state = _done_capped_state()
    assert fb.submit_feedback(state, design=DESIGN_TEXT)[0]

    out = refine_gate_node(state)

    assert out["best_design"] is None, "the pre-directive design became the incumbent"
    assert out["stage"] is Stage.REFINING
    assert out["refine_iterations"] == 1

    # Once the architect has applied it, scoring resumes as normal.
    state.user_feedback = state.feedback_marked_applied(state.pending_feedback("design"))
    out = refine_gate_node(state)
    assert out["best_design"] is not None


# ══════════════════════════════════════════════════════════════════════════
# 7. Persistence
# ══════════════════════════════════════════════════════════════════════════
def test_new_fields_round_trip_through_persistence():
    state = _done_capped_state()
    assert fb.submit_feedback(
        state, requirements=REQUIREMENTS_TEXT, design=DESIGN_TEXT
    )[0]
    state.context_history = [state.context_record.model_copy(update={"version": 1})]
    state.context_record = state.context_record.model_copy(
        update={"version": 2, "revision_reason": REQUIREMENTS_TEXT}
    )

    save_state(state)
    restored = load_state(state.run_id)

    assert restored == state
    assert [entry.kind for entry in restored.user_feedback] == [
        "requirements",
        "design",
    ]
    assert restored.context_record.version == 2
    assert restored.context_record.revision_reason == REQUIREMENTS_TEXT
    assert [old.version for old in restored.context_history] == [1]


def test_pre_change_checkpoints_still_load():
    """A checkpoint written before this feature existed must still resume.

    Every field it does not know about has to default to "no feedback, first
    version of the record" — which is exactly what such a run was.
    """
    state = _done_capped_state()
    payload = json.loads(state.model_dump_json())
    del payload["user_feedback"]
    del payload["context_history"]
    del payload["context_record"]["version"]
    del payload["context_record"]["revision_reason"]

    restored = ArchitectState.model_validate(payload)

    assert restored.user_feedback == []
    assert restored.context_history == []
    assert restored.context_record.version == 1
    assert restored.context_record.revision_reason == ""
    assert restored.pending_feedback("design") == []


if __name__ == "__main__":
    # Through pytest, not a hand-rolled loop: one test takes the `monkeypatch`
    # fixture, and conftest.py is what keeps every checkpoint these tests write
    # out of the real `.cache/runs` the resume picker reads.
    raise SystemExit(pytest.main([__file__, "-q"]))
