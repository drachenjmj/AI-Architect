"""user_feedback.py — iterative refinement from the human, at DONE (Kati).

WHAT THIS IS
------------
The finished-run screen shows two feedback boxes, each placed under the thing it
acts on: a REQUIREMENTS box in the Context Record section, and a DESIGN box
under the Blueprint / ADRs / Components. This module is the caller-side write
path for both — the same role `clarifier.submit_context_edits` plays for the
context lock. ui.py (and any other caller with a human attached) collects the
text; everything that happens to the state happens here, once, so two callers
cannot drift apart on what a submission means.

NO CLASSIFIER. THE BOX IS THE ROUTE.
------------------------------------
Nothing reads the text to decide what to do with it. A requirements correction
and a design directive are told apart by WHICH BOX they were typed into, which
is known before any model runs — so a human-in-the-loop feature adds exactly
zero stochastic routing to the pipeline. That is also why both boxes are on
screen at once rather than behind tabs: hiding one is what makes people put
everything in the first one, and a mis-filed correction is the one input this
design cannot recover from.

THE TWO ROUTES (no new nodes, no new stages, no new PendingDecision)
--------------------------------------------------------------------
  DESIGN       → `stage = REFINING`. The existing entry route sends that to the
                 refine gate, which sends it to the architect. The architect
                 reads the directive from `state.user_feedback` and marks it
                 applied. One iteration of the (freshly reset) refine budget is
                 spent doing so, and the step this module logs says so out loud.

  REQUIREMENTS → the text is appended to `clarification_answers` under a key
                 unique to this submission, then `stage = AWAITING_HUMAN` with
                 `pending_decision = CLARIFICATION`. The entry route sends that
                 back to the clarifier, which re-judges and re-freezes the
                 record as a new VERSION. `_may_ask` is already false (a record
                 exists), so no new round of questions: any gap the correction
                 opens becomes a labelled assumption, and the clarifier pauses at
                 the approval gate so the human sees the revised record —
                 including what was newly assumed — before the expensive work
                 runs. Then: researcher (fresh retrieval) → architect → reviewer.

The key is unique per submission (`round N`) on purpose. A fixed key would let a
second correction silently overwrite the first, which is precisely the "accept
the text and drop it" failure this feature exists to prevent.

BOTH BOXES IN ONE SUBMISSION
----------------------------
Allowed, and the requirements change is applied FIRST — the person types once,
and the record has to land before anything else anyway, because the features
are re-derived from it. The design directive is simply carried forward as
`pending` and consumed by the architect pass that follows the re-freeze. One
submission is ONE user round, however many boxes were filled.

WHAT A SUBMISSION COSTS, AND WHAT IT CLEARS
--------------------------------------------
`refine_gate.reopen_for_user_round` — one function, both paths — charges the
round through the single writer of `user_rounds` and clears the four fields that
would otherwise eat it (see its docstring; `stopped_on_cap` and
`refine_iterations` are what made the obvious version of this feature a silent
no-op). When it refuses, NOTHING is recorded: no feedback entry, no counter
change, no stage change. The caller shows `reason` and the boxes stay disabled.

The requirements path clears two more things, and only that path does:
`features` and `review`. Both are DERIVED FROM THE RECORD that the correction
just superseded — the features literally so, the review as a verdict on a design
built for it. Carrying them forward would have the architect reuse a v1 feature
set under a v1 reviewer instruction naming v1 feature IDs, while designing
against a v2 record. The design path keeps both, deliberately: there the design
under discussion has not changed, so the reviewer's findings still apply and
travel into the same prompt as the user's directive — ranked below it.
"""
from __future__ import annotations

from pipeline.agents.base import make_step
from pipeline.refine_gate import reopen_for_user_round
from pipeline.state import (
    ArchitectState,
    PendingDecision,
    Stage,
    UserFeedback,
)

# The agent name this module's trace entries are filed under. It is not an
# agent — it is the human — but `history` is also the per-agent token ledger
# (`ArchitectState.usage_by_agent`), and a step that spends nothing still has to
# be findable by whoever is reading the trail for WHY the run started again.
FEEDBACK_AGENT = "user_feedback"


def requirements_key(round_number: int) -> str:
    """The `clarification_answers` key one requirements correction is filed under.

    UNIQUE PER SUBMISSION, which is the whole job: `user_rounds` is charged once
    per submission and never repeats, so two corrections cannot collide. A fixed
    key would make the second one overwrite the first — the clarifier would
    re-judge with the correction the human had already moved on from, and the
    one they just typed would be gone with no error anywhere.

    Named so it reads as what it is in the Q&A replay: a correction the human
    made after seeing a design, not an answer to a question we asked.
    """
    return f"User correction after design (round {round_number})"


def submit_feedback(
    state: ArchitectState,
    *,
    requirements: str = "",
    design: str = "",
) -> tuple[bool, str]:
    """Record one submission and point the run back into the graph.

    Returns ``(accepted, reason)``. `False` means the human-round budget is
    spent: `reason` is the text to show them, and `state` is untouched — no
    feedback recorded, no counter moved, no stage changed. The caller must not
    re-enter the graph on a `False`.

    On `True` the state is ready for `run_pipeline`: the stage and
    `pending_decision` are set for whichever route applies, the feedback is on
    `state.user_feedback` as `pending`, and the step explaining why the run
    started again is on `state.history`.

    Raises `ValueError` when both boxes are empty (there is nothing to submit)
    or when the run is not at DONE. Loudly on both, because a caller that
    submits blank text has a bug in its form handling, and one that submits
    mid-run is about to overwrite a stage the pipeline is still using.
    """
    requirements = requirements.strip()
    design = design.strip()
    if not requirements and not design:
        raise ValueError(
            "submit_feedback was called with both boxes empty. There is nothing "
            "to record, and re-entering the graph would spend a user round on a "
            "run that would produce exactly what it produced before."
        )
    if state.stage is Stage.ACCEPTED:
        # Its own message because this is not a caller bug — it is a run that
        # was CLOSED. A human took this design; changing it now would mean the
        # thing on screen is no longer the thing that was signed off, and the
        # waiver would name findings against artifacts that no longer exist.
        # `sign_off.feedback_is_closed` is what stops a UI ever getting here.
        raise ValueError(
            "submit_feedback called on an ACCEPTED run. The design was signed "
            "off, so there is nothing left to redirect — start a new run to "
            "take it further. Reading it is still open: ask_advisor works at "
            "ACCEPTED."
        )
    if state.stage is not Stage.DONE:
        raise ValueError(
            f"submit_feedback called at stage {state.stage.value!r}; feedback is "
            f"collected at DONE, on the finished run it is about. Mid-run the "
            f"stage this writes is the one the pipeline is routing on."
        )

    # ONE round for the whole submission, charged before anything is recorded so
    # a refusal leaves no trace of feedback the run will never act on.
    allowed, reason = reopen_for_user_round(state)
    if not allowed:
        return False, reason

    round_number = state.user_rounds  # after the charge: the round this IS
    notes: list[str] = []

    # ── requirements FIRST, always ───────────────────────────────────────
    # Not merely an ordering preference: the record is the ground truth the
    # features and the design are derived from, so it has to land before either
    # can be rebuilt. A design directive submitted alongside it waits.
    if requirements:
        state.user_feedback.append(
            UserFeedback(
                text=requirements, kind="requirements", round=round_number
            )
        )
        state.clarification_answers[requirements_key(round_number)] = requirements
        # The clarifier is the ONLY writer of a ContextRecord: this hands it the
        # correction and asks it to re-judge, and never touches the record here.
        state.stage = Stage.AWAITING_HUMAN
        state.pending_decision = PendingDecision.CLARIFICATION
        # Derived from the record being superseded, therefore stale — see the
        # module docstring. Dropped only on this path.
        state.features = []
        state.review = None
        notes.append(
            "requirements correction re-opens the Context Record; the clarifier "
            "re-judges, then research and design re-run"
        )

    if design:
        state.user_feedback.append(
            UserFeedback(text=design, kind="design", round=round_number)
        )
        if not requirements:
            # Design-only: straight back into the refine loop. With a
            # requirements correction in the same submission the directive is
            # left pending instead, and the architect pass that follows the
            # re-freeze picks it up.
            state.stage = Stage.REFINING
            state.pending_decision = None
        notes.append(
            "design directive goes to the architect ranked above the reviewer's "
            "instruction; the refine budget restarts, so applying it spends "
            "iteration 1 of it"
        )

    state.history.append(
        make_step(
            FEEDBACK_AGENT,
            Stage.DONE,
            state.stage,
            f"user feedback at DONE (round {round_number}): " + "; ".join(notes),
        )
    )
    return True, ""
