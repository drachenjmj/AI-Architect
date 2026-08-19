"""test_sign_off.py — governance and close-out at DONE (Kati, feature A2).

No API key, no network: the Clarifier, Architect and Reviewer calls are the same
canned responses the rest of the suite uses, and the helpers that build a
finished run are reused from test_user_feedback rather than rebuilt here.

WHAT IS ACTUALLY BEING PINNED
------------------------------
A1's failure mode was a box that silently dropped what a person typed. A2's is
quieter and worse, because it corrupts the record rather than an interaction:

  * an accepted best-effort design and an accepted clean one become
    indistinguishable, because nothing wrote down what was accepted DESPITE;
  * a directive the person typed and never re-ran stays `pending` on a closed
    run, so the trail claims work is outstanding that nobody will ever do;
  * the architect quietly substitutes something else for an instruction it
    cannot build, and every screen renders as though it complied;
  * and — the one that would poison the pipeline rather than merely the record —
    A2's content leaks into a prompt. A reviewer told a finding was waived is
    grading intentions, and an architect that reads its own objections back has
    learned that objecting makes a directive go away.

The last of those is why the hygiene test asserts on ASSEMBLED PROMPTS rather
than on intentions, and why it includes the case that makes the leak live: a
`UserFeedback` entry that carries an objection AND is still pending, so its text
IS in the prompt and its objection must not be.
"""
from __future__ import annotations

import json

import pytest

import test_clarifier as tc  # canned Clarifier + Architect responses
import test_user_feedback as tuf  # the finished-run builders A1 already has
from pipeline import orchestrator, sign_off
from pipeline import user_feedback as fb
from pipeline.agents import architect as arch
from pipeline.agents import clarifier as clar
from pipeline.agents import reviewer as rev
from pipeline.orchestrator import _route
from pipeline.persistence import load_state, save_state
from pipeline.review_checks import run_deterministic_checks
from pipeline.state import (
    ArchitectState,
    AdvisoryTurn,
    ReviewIssue,
    ReviewResult,
    Stage,
    UserFeedback,
    Waiver,
)

from langgraph.graph import END

DESIGN_TEXT = tuf.DESIGN_TEXT

# Deliberately unmistakable, so an assertion that a string is absent from a
# 40 kB prompt is proving something rather than getting lucky.
MAGIC_NOTE = "MAGIC_WAIVER_NOTE tracked in JIRA-123"
MAGIC_OBJECTION = "MAGIC_OBJECTION SQS gives no global ordering, so I used FIFO queues"
MAGIC_DIRECTIVE = "MAGIC_DIRECTIVE use SQS instead of Kafka"


# ══════════════════════════════════════════════════════════════════════════
# Builders
# ══════════════════════════════════════════════════════════════════════════
def _capped() -> ArchitectState:
    """A REAL finished run that stopped on the refine budget, with open findings."""
    tuf._install_mocks()
    return tuf._done_capped_state()


def _passed() -> ArchitectState:
    """A finished run whose review passed and left no findings."""
    return tuf._done_passed_state()


def _with_findings(*findings: ReviewIssue) -> ArchitectState:
    """A finished run carrying exactly the findings given. Severity fixtures."""
    state = _passed()
    state.review = ReviewResult(
        overall_status="fail",
        issues=list(findings),
        requires_refinement=True,
        refinement_instruction="Fix them.",
    )
    return state


def _issue(issue_id: str, severity: str) -> ReviewIssue:
    return ReviewIssue(
        id=issue_id,
        severity=severity,
        finding=f"{issue_id} is unresolved.",
        requires_refinement=True,
    )


def _recording_llm(box: dict):
    """An `llm_call` stand-in that keeps the prompt and the system it was given."""

    def _call(state, prompt, *, system="", model="", response_schema=None):
        box["prompt"] = prompt
        box["system"] = system
        return "An answer.", tc.fake_usage()

    return _call


def _objecting_architect(objection: str):
    """The canned architect, but phase 2 also returns `directive_objection`."""

    def _call(state, prompt, **kwargs):
        result, usage = tc._architect_response(state, prompt, **kwargs)
        if isinstance(result, arch.ArchitectureDesign):
            result = result.model_copy(update={"directive_objection": objection})
        return result, usage

    return _call


# ══════════════════════════════════════════════════════════════════════════
# 1. ACCEPTED is terminal, and only a human can write it
# ══════════════════════════════════════════════════════════════════════════
def test_accepted_routes_to_end_exactly_like_done():
    state = _capped()
    sign_off.accept_design(state)

    assert state.stage is Stage.ACCEPTED
    assert _route(state) == END
    # Absent from the determinism map for the same reason DONE is — terminal.
    assert Stage.ACCEPTED not in orchestrator.STAGE_TO_NODE


def test_re_entering_the_graph_at_accepted_does_nothing():
    """Terminal has to mean terminal even if a caller re-enters by accident."""
    state = _capped()
    sign_off.accept_design(state)
    steps_before = len(state.history)

    out = orchestrator.run_pipeline(state)

    assert out.stage is Stage.ACCEPTED
    assert len(out.history) == steps_before, "a node ran on a closed run"


def test_no_agent_ever_writes_accepted():
    """A full pipeline run must finish at DONE. ACCEPTED is not an outcome."""
    tuf._install_mocks()
    state = orchestrator.run_pipeline(tuf._seed_designable_state())

    assert state.stage is Stage.DONE
    assert all(step.stage_out is not Stage.ACCEPTED for step in state.history)


@pytest.mark.parametrize(
    "stage",
    [Stage.CREATED, Stage.CLARIFYING, Stage.AWAITING_HUMAN, Stage.REVIEWING,
     Stage.REFINING, Stage.FAILED],
)
def test_sign_off_is_refused_anywhere_but_done(stage):
    state = _capped()
    state.stage = stage
    with pytest.raises(ValueError):
        sign_off.accept_design(state)


def test_signing_off_twice_is_refused():
    """The second call would re-stamp `accepted_at` on a decision nobody made."""
    state = _capped()
    sign_off.accept_design(state)
    with pytest.raises(ValueError, match="already ACCEPTED"):
        sign_off.accept_design(state)


def test_sign_off_records_when_and_logs_a_step():
    state = _capped()
    steps_before = len(state.history)

    sign_off.accept_design(state, note=MAGIC_NOTE)

    assert state.accepted_at, "a sign-off with no timestamp records nothing"
    step = state.history[-1]
    assert len(state.history) == steps_before + 1
    assert step.agent == sign_off.SIGN_OFF_AGENT
    assert step.stage_in is Stage.DONE and step.stage_out is Stage.ACCEPTED
    # Free: it is a human decision, not a model call.
    assert step.input_tokens == 0 and step.output_tokens == 0
    assert MAGIC_NOTE in step.note


# ══════════════════════════════════════════════════════════════════════════
# 2. Waivers — what was accepted despite
# ══════════════════════════════════════════════════════════════════════════
def test_a_capped_failing_review_can_be_accepted_and_the_waiver_names_its_findings():
    """THE normal case. Most runs end here, so a sign-off that required a pass
    would leave exactly the runs that need closing out unable to be closed.
    """
    state = _capped()
    assert state.review.overall_status == "fail"
    open_ids = [issue.id for issue in state.review.issues]
    assert open_ids, "the fixture must actually carry open findings"

    waiver = sign_off.accept_design(state, note=MAGIC_NOTE)

    assert state.stage is Stage.ACCEPTED
    assert waiver is state.waiver is not None
    assert sorted(waiver.finding_ids) == sorted(open_ids)
    assert len(waiver.severities) == len(waiver.finding_ids)
    assert waiver.review_status == "fail"
    assert waiver.accepted_at == state.accepted_at
    assert waiver.note == MAGIC_NOTE
    # The artifacts are untouched: what is on screen is what was signed.
    assert state.blueprint is not None
    assert state.review.issues == [issue for issue in state.review.issues]


def test_a_clean_sign_off_records_no_waiver():
    """Absence of a waiver IS the information — it must not be an empty object."""
    state = _passed()
    assert state.review.overall_status == "pass"
    assert sign_off.open_findings(state) == []

    waiver = sign_off.accept_design(state)

    assert waiver is None
    assert state.waiver is None
    assert state.stage is Stage.ACCEPTED
    assert state.accepted_at


def test_the_waiver_lists_findings_highest_severity_first():
    state = _with_findings(
        _issue("LOW-1", "low"), _issue("HIGH-1", "high"), _issue("MED-1", "medium")
    )

    waiver = sign_off.accept_design(state)

    assert waiver.finding_ids == ["HIGH-1", "MED-1", "LOW-1"]
    assert waiver.severities == ["high", "medium", "low"]
    assert waiver.severity_counts() == {"high": 1, "medium": 1, "low": 1}


def test_severities_stay_parallel_to_finding_ids():
    """They are two lists that must not drift; the model refuses a mismatch."""
    with pytest.raises(ValueError, match="parallel"):
        Waiver(finding_ids=["A", "B"], severities=["high"])


def test_only_a_high_severity_finding_needs_a_deliberate_confirmation():
    """A confirmation everything triggers is a confirmation nobody reads."""
    requires = sign_off.requires_deliberate_confirmation
    assert requires(_with_findings(_issue("H", "high"))) is True
    assert requires(_with_findings(_issue("M", "medium"), _issue("L", "low"))) is False
    assert requires(_passed()) is False


def test_the_note_is_optional():
    state = _passed()
    state.review.issues = [_issue("MED-1", "medium")]

    waiver = sign_off.accept_design(state)

    assert waiver is not None
    assert waiver.note == ""


def test_open_findings_reads_the_current_review_not_the_incumbent():
    """`best_design.review` can be a different round's verdict on different
    artifacts. Waiving against it would name findings that are not in the design
    being signed.
    """
    state = _capped()
    assert state.best_design is not None
    state.review = ReviewResult(overall_status="fail", issues=[_issue("NOW-1", "medium")])

    assert [issue.id for issue in sign_off.open_findings(state)] == ["NOW-1"]
    assert sign_off.accept_design(state).finding_ids == ["NOW-1"]


# ══════════════════════════════════════════════════════════════════════════
# 3. Unapplied feedback at sign-off: surfaced, never blocking
# ══════════════════════════════════════════════════════════════════════════
def test_signing_off_with_a_pending_directive_does_not_block_and_abandons_it():
    """Blocking would force a full refine round — one user round, ~75k tokens —
    purely to close out a run the person has already decided to take. So the
    directive is surfaced before the click and abandoned by it, never lost and
    never left looking outstanding.
    """
    state = _capped()
    assert fb.submit_feedback(state, design=DESIGN_TEXT)[0]
    state.stage = Stage.DONE  # they never re-ran; they signed off instead

    # The screen's input: it is shown BEFORE the button, beside the findings.
    pending = sign_off.unapplied_feedback(state)
    assert [entry.text for entry in pending] == [DESIGN_TEXT]

    waiver = sign_off.accept_design(state)

    assert state.stage is Stage.ACCEPTED, "sign-off was blocked"
    assert waiver is not None
    assert [entry.status for entry in state.user_feedback] == ["abandoned"]
    assert state.pending_feedback("design") == []
    # Seen and dropped, on the record, in the person's own words.
    assert DESIGN_TEXT in state.history[-1].note


def test_signing_off_with_nothing_pending_leaves_the_feedback_list_untouched():
    state = _capped()
    assert fb.submit_feedback(state, design=DESIGN_TEXT)[0]
    state = orchestrator.run_pipeline(state)  # the architect consumes it
    assert [entry.status for entry in state.user_feedback] == ["applied"]
    before = [entry.model_copy() for entry in state.user_feedback]

    sign_off.accept_design(state)

    assert state.user_feedback == before
    assert sign_off.unapplied_feedback(state) == []


def test_an_unconsumed_requirements_correction_is_abandoned_too():
    """Same category, same treatment. It should never survive to here — its
    route re-enters the graph at once — but if it has, it is the same fact about
    the same run.
    """
    state = _capped()
    state.user_feedback.append(
        UserFeedback(text="Peak is 500k.", kind="requirements", round=1)
    )

    sign_off.accept_design(state)

    assert state.user_feedback[0].status == "abandoned"


# ══════════════════════════════════════════════════════════════════════════
# 4. After ACCEPTED: the boxes close, reading stays open
# ══════════════════════════════════════════════════════════════════════════
def test_the_feedback_boxes_are_open_at_done_and_closed_after_acceptance():
    state = _capped()
    assert sign_off.feedback_is_closed(state) == (False, "")

    sign_off.accept_design(state)

    closed, why = sign_off.feedback_is_closed(state)
    assert closed is True
    assert "accepted" in why


def test_submitting_feedback_on_an_accepted_run_is_refused():
    """The disabled box is the UI; this is the rule underneath it."""
    state = _capped()
    sign_off.accept_design(state)

    with pytest.raises(ValueError, match="ACCEPTED"):
        fb.submit_feedback(state, design=DESIGN_TEXT)

    assert state.stage is Stage.ACCEPTED
    assert state.user_feedback == []


def test_the_advisory_turn_still_works_after_acceptance(monkeypatch):
    """Reading an accepted design is not changing it, so the cheap channel that
    exists to be asked questions must not close with the boxes.
    """
    state = _capped()
    sign_off.accept_design(state)
    box: dict = {}
    monkeypatch.setattr(clar, "llm_call", _recording_llm(box))
    before_rounds = state.user_rounds

    answer = clar.ask_advisor(state, "Why this pattern?", subject="design")

    assert answer == "An answer."
    assert state.stage is Stage.ACCEPTED     # nothing moved
    assert state.user_rounds == before_rounds
    assert state.waiver is not None and state.accepted_at
    assert len(state.advisory_turns) == 1
    assert state.advisory_turns[0].subject == "design"


# ══════════════════════════════════════════════════════════════════════════
# 5. ask_advisor(subject="design")
# ══════════════════════════════════════════════════════════════════════════
def test_design_advisory_gets_the_artifacts_and_not_the_context_record(monkeypatch):
    state = _capped()
    state.context_record.business_goal = "MAGIC_RECORD_ONLY_FIELD"
    state.review.issues = [_issue("LLM-9", "high")]
    box: dict = {}
    monkeypatch.setattr(clar, "llm_call", _recording_llm(box))

    clar.ask_advisor(state, "Why this pattern?", subject="design")

    prompt = box["prompt"]
    assert state.blueprint.selected_pattern in prompt
    assert state.adrs[0].id in prompt
    assert state.components[0].name in prompt
    assert "LLM-9" in prompt, "the review's findings travel with the design"
    assert "MAGIC_RECORD_ONLY_FIELD" not in prompt
    # And the framing matches what they are looking at.
    assert "finished architecture" in box["system"]


def test_context_record_advisory_is_unchanged(monkeypatch):
    """The default subject, and the existing call sites that pass nothing."""
    state = _capped()
    state.context_record.business_goal = "MAGIC_RECORD_ONLY_FIELD"
    box: dict = {}
    monkeypatch.setattr(clar, "llm_call", _recording_llm(box))

    clar.ask_advisor(state, "Why does scale matter?")

    assert "MAGIC_RECORD_ONLY_FIELD" in box["prompt"]
    assert state.advisory_turns[0].subject == "context_record"
    assert box["system"] == clar.ADVISOR_SYSTEM


def test_only_same_subject_turns_are_threaded_into_the_prompt(monkeypatch):
    """Otherwise a design question arrives with the lock-gate Q&A about scale
    and budget stapled above it, and the model answers the wrong conversation.
    """
    state = _capped()
    state.advisory_turns = [
        AdvisoryTurn(
            question="MAGIC_LOCK_QUESTION about budget",
            answer="MAGIC_LOCK_ANSWER",
            subject="context_record",
        ),
        AdvisoryTurn(
            question="MAGIC_DESIGN_QUESTION about the bus",
            answer="MAGIC_DESIGN_ANSWER",
            subject="design",
        ),
    ]
    box: dict = {}
    monkeypatch.setattr(clar, "llm_call", _recording_llm(box))

    clar.ask_advisor(state, "And the other one?", subject="design")

    assert "MAGIC_DESIGN_QUESTION" in box["prompt"]
    assert "MAGIC_DESIGN_ANSWER" in box["prompt"]
    assert "MAGIC_LOCK_QUESTION" not in box["prompt"]
    assert "MAGIC_LOCK_ANSWER" not in box["prompt"]


def test_the_design_advisory_turn_is_read_only_and_billed(monkeypatch):
    """Same contract as the gate's: nothing moves, and the ledgers reconcile."""
    state = _capped()
    monkeypatch.setattr(clar, "llm_call", _recording_llm({}))
    before = state.model_copy(deep=True)

    clar.ask_advisor(state, "Why Kafka?", subject="design")

    assert state.stage is before.stage
    assert state.pending_decision is before.pending_decision
    assert state.blueprint == before.blueprint
    assert state.review == before.review
    assert state.user_rounds == before.user_rounds
    assert state.refine_iterations == before.refine_iterations

    step = state.history[-1]
    assert len(state.history) == len(before.history) + 1
    assert step.agent == clar.ADVISOR_AGENT
    assert step.stage_in is step.stage_out is state.stage
    assert step.input_tokens > 0 and step.output_tokens > 0
    assert "design" in step.note
    assert state.input_tokens == before.input_tokens + step.input_tokens
    assert state.output_tokens == before.output_tokens + step.output_tokens


def test_a_design_question_needs_a_design(monkeypatch):
    state = _capped()
    state.blueprint = None
    monkeypatch.setattr(clar, "llm_call", _recording_llm({}))

    with pytest.raises(ValueError, match="no design"):
        clar.ask_advisor(state, "Why?", subject="design")


# ══════════════════════════════════════════════════════════════════════════
# 6. Architect objections — a voice, not a veto
# ══════════════════════════════════════════════════════════════════════════
def test_an_objection_lands_on_the_directive_it_is_about_and_does_not_block():
    tuf._install_mocks()
    state = tuf._done_capped_state()
    assert fb.submit_feedback(state, design=MAGIC_DIRECTIVE)[0]
    arch.llm_call = _objecting_architect(MAGIC_OBJECTION)

    state = orchestrator.run_pipeline(state)

    entry = state.user_feedback[-1]
    assert entry.text == MAGIC_DIRECTIVE
    assert entry.status == "objected"
    assert entry.objection == MAGIC_OBJECTION
    # Consumed, so it is not handed to the next pass again.
    assert state.pending_feedback("design") == []
    # And the pass produced a design and carried on — a voice, not a veto.
    assert state.blueprint is not None and state.adrs and state.components
    assert state.stage is Stage.DONE
    assert any(
        step.agent == "architect" and MAGIC_OBJECTION in step.note
        for step in state.history
    ), "the objection is not in the trace"


def test_the_schema_carries_the_objection_and_nothing_else_does():
    """Named explicitly because the only other free-text slot on this pass is
    `Blueprint.revision_note`, and that is precisely where it must not go.
    """
    assert "directive_objection" in arch.ArchitectureDesign.model_fields
    for model in ("Blueprint", "ADR", "ComponentDescription"):
        fields = getattr(__import__("pipeline.state", fromlist=[model]), model).model_fields
        assert "directive_objection" not in fields
        assert "objection" not in fields


def test_no_objection_means_the_directive_was_simply_applied():
    tuf._install_mocks()
    state = tuf._done_capped_state()
    assert fb.submit_feedback(state, design=MAGIC_DIRECTIVE)[0]

    state = orchestrator.run_pipeline(state)

    entry = state.user_feedback[-1]
    assert entry.status == "applied"
    assert entry.objection == ""


def test_an_objection_with_no_directive_to_object_to_is_dropped():
    """With no directive in the prompt there is no entry to record it against,
    and the model is volunteering a caveat it was told not to write.
    """
    tuf._install_mocks()
    state = tuf._seed_designable_state()
    arch.llm_call = _objecting_architect(MAGIC_OBJECTION)

    state = orchestrator.run_pipeline(state)

    assert state.user_feedback == []
    assert all(MAGIC_OBJECTION not in step.note for step in state.history)


def test_only_the_newest_of_several_pending_directives_is_marked_objected():
    """One objection comes back for a block that may hold several directives.
    The newest is the one the person just typed and is still looking at.
    """
    state = _capped()
    older = UserFeedback(text="older directive", kind="design", round=1)
    newer = UserFeedback(text=MAGIC_DIRECTIVE, kind="design", round=2)
    state.user_feedback = [older, newer]

    out = state.feedback_marked_applied(
        state.pending_feedback("design"), objection=MAGIC_OBJECTION
    )

    assert [entry.status for entry in out] == ["applied", "objected"]
    assert out[0].objection == ""
    assert out[1].objection == MAGIC_OBJECTION


# ══════════════════════════════════════════════════════════════════════════
# 7. PROMPT HYGIENE — the constraint that governs all of the above
# ══════════════════════════════════════════════════════════════════════════
def _state_carrying_every_a2_field() -> ArchitectState:
    """One run holding a waiver, an acceptance, and BOTH shapes of objection.

    The pending-with-an-objection entry is the case that makes the leak live
    rather than hypothetical: its `text` IS in the architect's prompt by design,
    and its `objection` sits one field away on the same object. A builder that
    reached for `model_dump_json()` — the convenient thing — would ship both.
    """
    state = _capped()
    state.user_feedback = [
        UserFeedback(
            text=MAGIC_DIRECTIVE,
            kind="design",
            status="objected",
            objection=MAGIC_OBJECTION,
            round=1,
        ),
        UserFeedback(
            text="MAGIC_PENDING_DIRECTIVE keep the cache",
            kind="design",
            status="pending",
            objection=MAGIC_OBJECTION + " (still pending)",
            round=2,
        ),
    ]
    state.waiver = Waiver(
        finding_ids=["LLM-1"],
        severities=["high"],
        review_status="fail",
        note=MAGIC_NOTE,
    )
    state.accepted_at = "2026-08-19T12:00:00+00:00"
    return state


def _forbidden_in_prompts() -> list[str]:
    """Every string A2 adds that no agent may ever see."""
    return [MAGIC_NOTE, MAGIC_OBJECTION, "2026-08-19T12:00:00", "abandoned"]


def test_no_a2_content_reaches_the_reviewer_prompt():
    """It grades artifacts, not intentions. "The user asked for it" would excuse
    any deviation, and "that finding was waived" would excuse the finding — the
    same self-advocacy leak `_format_artifacts` keeps `revision_note` out for.
    The fairness concern is handled by the waiver, not by softening the judge.
    """
    state = _state_carrying_every_a2_field()

    prompt = rev._build_prompt(state, run_deterministic_checks(state))

    for forbidden in _forbidden_in_prompts():
        assert forbidden not in prompt, forbidden
    # Not even that a change was user-directed.
    assert MAGIC_DIRECTIVE not in prompt
    assert "MAGIC_PENDING_DIRECTIVE" not in prompt


def test_no_a2_content_reaches_the_architect_prompt():
    """A waived finding is not a solved one, and an architect that could read
    its own objections back would learn that objecting makes a directive go away.
    """
    state = _state_carrying_every_a2_field()

    prompt = arch._build_architecture_prompt(state, state.features)

    # The pending directive's TEXT is in the prompt — that is the feature.
    assert "MAGIC_PENDING_DIRECTIVE" in prompt
    # Its objection, sitting one field away on the same object, is not.
    for forbidden in _forbidden_in_prompts():
        assert forbidden not in prompt, forbidden


def test_no_a2_content_reaches_the_advisor_prompt(monkeypatch):
    """The advisor is an agent too. The governance record is for humans."""
    state = _state_carrying_every_a2_field()
    box: dict = {}
    monkeypatch.setattr(clar, "llm_call", _recording_llm(box))

    clar.ask_advisor(state, "Why this pattern?", subject="design")

    for forbidden in _forbidden_in_prompts():
        assert forbidden not in box["prompt"], forbidden


def test_the_directive_block_serialises_text_and_nothing_else():
    """Pins the mechanism, not only its effect: a refactor that swaps `.text`
    for `model_dump_json()` fails HERE with a reason, rather than silently
    restoring the leak on a state that happens to have no objection on it.
    """
    state = _capped()
    state.user_feedback = [
        UserFeedback(
            text="visible",
            kind="design",
            status="pending",
            objection="INVISIBLE",
            round=1,
        )
    ]

    prompt = arch._build_architecture_prompt(state, state.features)
    block = prompt.split("<user_directive>")[1].split("</user_directive>")[0]

    assert block.strip() == "visible"


# ══════════════════════════════════════════════════════════════════════════
# 8. Persistence
# ══════════════════════════════════════════════════════════════════════════
def test_new_fields_round_trip_through_persistence():
    state = _state_carrying_every_a2_field()
    state.advisory_turns = [
        AdvisoryTurn(question="Why?", answer="Because.", subject="design")
    ]
    state.stage = Stage.ACCEPTED

    save_state(state)
    restored = load_state(state.run_id)

    assert restored == state
    assert restored.stage is Stage.ACCEPTED
    assert restored.waiver.note == MAGIC_NOTE
    assert restored.waiver.finding_ids == ["LLM-1"]
    assert restored.accepted_at == state.accepted_at
    assert restored.user_feedback[0].status == "objected"
    assert restored.user_feedback[0].objection == MAGIC_OBJECTION
    assert restored.advisory_turns[0].subject == "design"


def test_pre_a2_checkpoints_still_load():
    """A checkpoint written before this feature existed must still resume, as
    what such a run was: finished, unsigned, nothing waived, all questions asked
    about the record.
    """
    state = _capped()
    state.advisory_turns = [AdvisoryTurn(question="Why?", answer="Because.")]
    payload = json.loads(state.model_dump_json())
    del payload["accepted_at"]
    del payload["waiver"]
    del payload["advisory_turns"][0]["subject"]

    restored = ArchitectState.model_validate(payload)

    assert restored.accepted_at == ""
    assert restored.waiver is None
    assert restored.advisory_turns[0].subject == "context_record"
    assert sign_off.feedback_is_closed(restored) == (False, "")
    assert sign_off.unapplied_feedback(restored) == []


def test_a_pre_a2_feedback_entry_loads_without_an_objection():
    payload = json.loads(
        UserFeedback(text="t", kind="design", round=1).model_dump_json()
    )
    del payload["objection"]

    assert UserFeedback.model_validate(payload).objection == ""


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
