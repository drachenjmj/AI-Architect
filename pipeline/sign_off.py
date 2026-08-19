"""sign_off.py — closing a run out: ACCEPTED, and what it was accepted despite (Kati).

WHAT THIS IS
------------
The caller-side write path for ONE human action: taking the design. It is the
close-out counterpart to `pipeline/user_feedback.py`, and the two are separate
files because they do opposite things. `user_feedback` sends a finished run BACK
INTO the graph — it charges a round, resets the caps, sets a stage the router
acts on. This module ENDS the run: it enters no graph, spends no tokens, resets
nothing, and the stage it writes is one nothing routes from.

WHY "DONE" WAS NOT ENOUGH
--------------------------
`DONE` means the pipeline stopped. `ACCEPTED` means a person took the design.
Those are two different facts and a decision-support tool owes its reader both —
see the note on `Stage`. Everything in this file exists to record the second one
without pretending it implies the first.

THE SIGN-OFF ACCEPTS WHAT THE REVIEWER REJECTED, ON PURPOSE
------------------------------------------------------------
There is deliberately NO passing-review precondition. Most runs end
`stopped_on_cap` with open findings, so a sign-off that required a pass would
leave exactly the runs that most need closing out unable to be closed — and the
thing being recorded is the human's decision, not the reviewer's. What the
decision was made DESPITE is recorded separately and precisely, as a `Waiver`.

NOTHING HERE IS ALLOWED TO BLOCK
---------------------------------
Not the open findings, and not an unapplied directive (see
`unapplied_feedback`). Blocking on an unapplied directive would force a full
refine round — one `user_round`, ~75k tokens — purely to close out a run the
person has already decided to take. So both are SURFACED BEFORE THE CLICK and
neither stands in front of it: the caller shows what is about to be accepted,
the human confirms, and this module records both halves of that.

PROMPT HYGIENE
--------------
Nothing this module writes — `accepted_at`, the waiver, the abandonment — may
ever reach an agent prompt. A waived finding is not a solved one, and a reviewer
that learned a design had been accepted would be grading intentions instead of
artifacts. There is a test that asserts this rather than trusting it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pipeline.agents.base import make_step
from pipeline.refine_gate import evaluate_user_rounds
from pipeline.state import (
    ArchitectState,
    ReviewIssue,
    Stage,
    UserFeedback,
    Waiver,
)

# The agent name this module's trace entry is filed under. Like
# `user_feedback.FEEDBACK_AGENT` it is not an agent — it is the human — but
# `history` is the run's single trail, and "who closed this run, and when"
# belongs in it beside every step that produced the thing being closed. It
# spends no tokens, so it reads as zero in `usage_by_agent()`.
SIGN_OFF_AGENT = "sign_off"

# Highest first. Named here because THREE things order by it and must not each
# invent their own: the pre-sign-off list the human reads, the waiver's parallel
# lists, and the rule for what needs a deliberate second confirmation.
_SEVERITY_RANK: dict[str, int] = {"high": 0, "medium": 1, "low": 2}

# The severity that is not allowed to be waived on one click. A high-severity
# finding is the one case where "I clicked the obvious button" and "I decided to
# ship this" must be told apart; a medium or low one does NOT get a second
# prompt, because a confirmation everything triggers is a confirmation nobody
# reads.
DELIBERATE_SEVERITY = "high"


def open_findings(state: ArchitectState) -> list[ReviewIssue]:
    """The current review's open findings, highest severity first. Pure.

    "Current" means the review on the state — the verdict on the artifacts being
    signed off, never `best_design.review`, which may be a different round's.

    Sorted, and STABLY: findings of equal severity keep the reviewer's own
    order, so the list a person reads before signing and the list stored in the
    waiver afterwards are the same list in the same order.
    """
    review = state.review
    if review is None:
        return []
    return sorted(
        review.issues, key=lambda issue: _SEVERITY_RANK.get(issue.severity, 99)
    )


def unapplied_feedback(state: ArchitectState) -> list[UserFeedback]:
    """Feedback the human typed that NO agent ever consumed, oldest first. Pure.

    The pre-sign-off screen's other half. A1 leaves an entry `pending` until the
    agent that owns it acts on it, so a person who types a directive and then
    signs off without re-running leaves one behind — and a `pending` entry on a
    closed run is indistinguishable from one that is queued and about to run.
    The trail would then claim work was outstanding on a run nobody will touch
    again.

    Both kinds, not just directives. A requirements correction should never
    survive to here (its route re-enters the graph immediately), but if one has,
    it is the same fact about the same run and gets the same treatment.
    """
    return [entry for entry in state.user_feedback if entry.status == "pending"]


def requires_deliberate_confirmation(state: ArchitectState) -> bool:
    """Does signing off here need a second, explicit confirmation? Pure code.

    True when any open finding is high severity. The rule lives here rather than
    in the UI so it is one testable fact rather than a property of a widget, and
    so a second caller cannot quietly disagree about what deserves a second
    click.
    """
    return any(
        issue.severity == DELIBERATE_SEVERITY for issue in open_findings(state)
    )


def feedback_is_closed(state: ArchitectState) -> tuple[bool, str]:
    """May the human still send this run back? Returns ``(closed, why)``. Pure.

    TWO reasons it can be closed, and the caller does not need to know which:

      * the run was ACCEPTED — the decision has been taken, so there is nothing
        left to iterate on. Reading an accepted design is still fine (the
        advisory turn works at ACCEPTED); CHANGING one is not, because the thing
        that was signed off would no longer be the thing on screen.
      * the human-round budget is spent — the pre-existing cap.

    One function so the two cannot drift apart at the call site, and so "the
    boxes are disabled after acceptance" is a fact a test can assert without
    driving Streamlit.
    """
    if state.stage is Stage.ACCEPTED:
        return True, "this design was accepted"
    exhausted, why = evaluate_user_rounds(state)
    if exhausted:
        return True, f"{why} reached"
    return False, ""


def accept_design(state: ArchitectState, *, note: str = "") -> Optional[Waiver]:
    """Record the human's sign-off. Terminal, and never refused.

    Returns the `Waiver` this sign-off recorded, or None when there was nothing
    to waive. Four things move, all of them together:

      1. `accepted_at` — when. Not who: there is no user identity in this
         system, and inventing a signer would be the one lie a governance record
         cannot afford.
      2. `waiver` — ONLY if the current review has open findings. A clean
         sign-off records none, and that absence is itself the information.
      3. every still-`pending` feedback entry becomes `abandoned` — seen and
         dropped, rather than silently lost or left looking outstanding.
      4. `stage` becomes ACCEPTED and one `StepLog` says so.

    THE CALLER OWES THE HUMAN THE SCREEN FIRST. This function performs a
    decision; it does not obtain one. `open_findings`,
    `requires_deliberate_confirmation` and `unapplied_feedback` are what a
    caller shows BEFORE calling this, and calling it without having shown them
    would waive findings and abandon text on somebody's behalf.

    `note` is optional and stays optional — a mandatory justification field
    produces "n/a" and then means nothing. It belongs to the waiver, so on a
    clean sign-off there is no waiver to carry it; it still lands in the trace
    entry, because text a person typed is never dropped without a record.

    Raises `ValueError` unless the run is at DONE. Loudly: a caller signing off
    mid-run is about to overwrite a stage the pipeline is still routing on, and
    one signing off twice has a bug in its form handling — the second call would
    re-stamp `accepted_at` and re-waive against whatever the review says now.
    """
    if state.stage is not Stage.DONE:
        if state.stage is Stage.ACCEPTED:
            raise ValueError(
                "accept_design called on a run that is already ACCEPTED. The "
                "sign-off happens once; re-stamping it would move `accepted_at` "
                "to a decision nobody made."
            )
        raise ValueError(
            f"accept_design called at stage {state.stage.value!r}. A design is "
            f"signed off at DONE, on the finished run it is about — mid-run "
            f"there is no finished design to take, and ACCEPTED is the stage "
            f"the pipeline would be routing on."
        )

    note = note.strip()
    findings = open_findings(state)
    abandoned = unapplied_feedback(state)
    accepted_at = datetime.now(timezone.utc).isoformat()

    waiver: Optional[Waiver] = None
    if findings:
        waiver = Waiver(
            finding_ids=[issue.id for issue in findings],
            severities=[issue.severity for issue in findings],
            review_status=state.review.overall_status if state.review else "",
            accepted_at=accepted_at,
            note=note,
        )

    state.accepted_at = accepted_at
    state.waiver = waiver
    for entry in abandoned:
        # Mutated in place, like every other write in the caller-side path (see
        # `user_feedback.submit_feedback`). No node is running, so there is no
        # returned update for this to ride on.
        entry.status = "abandoned"

    parts: list[str] = []
    if waiver is None:
        parts.append("design accepted with no open findings — no waiver recorded")
    else:
        counts = waiver.severity_counts()
        breakdown = ", ".join(
            f"{counts[name]} {name}" for name in ("high", "medium", "low") if name in counts
        )
        parts.append(
            f"design accepted despite {len(waiver.finding_ids)} open finding(s) "
            f"({breakdown}; review: {waiver.review_status or 'none'}): "
            + ", ".join(waiver.finding_ids)
        )
    if abandoned:
        parts.append(
            f"{len(abandoned)} unapplied change(s) abandoned: "
            + "; ".join(entry.text for entry in abandoned)
        )
    if note:
        parts.append(f"note: {note}")

    step = make_step(SIGN_OFF_AGENT, Stage.DONE, Stage.ACCEPTED, "; ".join(parts))
    state.history.append(step)
    state.stage = Stage.ACCEPTED
    return waiver
