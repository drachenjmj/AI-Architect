"""refine_gate.py — the cost-cap gate of the reviewer→refine loop (Kati).

WHERE THIS SITS
---------------
The Reviewer (Waqar) judges *correctness*: on a failing review it sets
`stage = REFINING` and writes `state.review.refinement_instruction`. The
Architect (Maheen) already *consumes* that instruction on a re-run. What was
missing is the bit that closes the loop safely — deciding whether to loop once
more or stop. That decision is a pure *cost policy*, so it lives here, in its
own code-only node that Kati (orchestration) owns. This keeps the two concerns
cleanly separated:

    Reviewer  -> is the design correct?      (LLM judges)
    RefineGate-> can we afford another loop?  (code decides)

This mirrors the Clarifier's "LLM judges / code routes" split, and — like the
Clarifier's directional `AWAITING_HUMAN` — the `REFINING` stage is directional:
coming OUT of the reviewer it routes to this gate; coming OUT of this gate it
routes to the architect (see orchestrator `_gate_route`).

THE STOP CONDITION (two independent caps, whichever trips first)
---------------------------------------------------------------
  * MAX_REFINE_ITERATIONS — how many times the architect may re-design.
  * MAX_TOTAL_TOKENS      — a spend ceiling over the whole run
                            (state.input_tokens + state.output_tokens, which
                            llm_call accumulates at its single chokepoint).

When either cap is reached the run finishes GRACEFULLY as DONE, keeping the
best-so-far artifacts, and sets `state.stopped_on_cap = True` so the UI/report
can say "stopped on the budget, not on a clean pass." It is NOT marked FAILED —
a design that converged as far as the budget allowed is not an error.

BEST-SO-FAR IS LITERAL (it did not used to be)
-----------------------------------------------
That paragraph described the intent for a long time before it described the
behaviour. The gate halted, and whatever the LAST architect pass happened to
produce was what the run shipped — "best-so-far" in the sense of "the most
recent attempt", which is not what the words mean.

The gap became measurable rather than theoretical in run
`20260818T083516Z-e92aa7cf`: the reviewer's `flaw_detection` judgment passed in
round 2 and failed again in round 3 on the same scenario. The judge flapped
inside a single run, so the run shipped a round-3 design over a round-2 design
that had scored strictly better. Nothing was wrong with the loop; the gate was
simply not choosing.

Choosing is pure deterministic code — `score_round` below — so it belongs here,
next to the other decision this node already owns. It costs no LLM call and no
routing change. It also does NOT make a failing run pass: `overall_status`,
`stopped_on_cap` and `refine_iterations` come out exactly as they did. The only
thing that changes is WHICH of the designs the run already paid for is the one
handed back.

WHY THE HUMAN-ROUND CAP AND THE STEP BUDGET ALSO LIVE HERE
-----------------------------------------------------------
Because they are the same KIND of decision. `MAX_USER_ROUNDS` caps how many
times a human may send the run back through the graph — at the clarifier's
questions, at the context lock, and (feature A) at DONE. It is deliberately ONE
counter for all of them: it is a cost cap, and cost is global. A per-touchpoint
counter would let the same person spend three budgets by spreading the same
number of round trips across three screens.

`derive_max_steps()` is the other half of the same argument. `max_steps` was a
hard-coded 20 passed by every caller, silently coupled to the caps above:
raising `MAX_REFINE_ITERATIONS` by two would have blown the recursion limit and
surfaced as `GraphRecursionError -> FAILED`, i.e. a BUDGET change arriving as a
CRASH. Deriving the number from the caps that determine it removes the coupling
without weakening the limit — `recursion_limit` keeps its real job, which is
catching a mis-wired route, and it still does that just as loudly.

None of this makes a routing decision or calls a model. The gate node stays
pure; the helpers marked "caller-side" below run OUTSIDE the graph — in ui.py,
run.py and user_feedback.py — and are here so those callers cannot drift apart
on policy.
"""
from __future__ import annotations

from pipeline.agents.base import make_step, node
from pipeline.state import (
    ArchitectState,
    DesignSnapshot,
    REVIEW_CODE_SCORE_FIELDS,
    REVIEW_JUDGMENT_FIELDS,
    ReviewResult,
    Stage,
)


# ── Cost-cap policy ───────────────────────────────────────────────────────
# Tunable. The iteration cap is what realistically trips; the token budget is a
# safety net for a run that burns tokens abnormally fast.
MAX_REFINE_ITERATIONS = 3
# STILL UNTUNED, but no longer unmeasured — deliberately left as-is pending a
# decision, not an oversight.
#
# The number was picked while token counting was broken (nodes never returned
# their usage, so this compared 0 against 500_000 and could not trip). Counting
# works now, and two real end-to-end runs have been measured:
#
#     greenfield, both refine iterations spent:
#     16,677 in +  8,042 out =  24,719 tokens  (~$0.016 at list prices)
#
#     BROWNFIELD (run 20260818T074835Z-1925fbd7), both iterations spent, a real
#     repo ingested and four rounds of clarifying questions:
#     95,311 in + 10,262 out = 105,573 tokens  (~$0.085 at list prices)
#
# So the honest multiple is ~4.7x a full brownfield run, not the ~20x the
# greenfield number suggested. Still a backstop — MAX_REFINE_ITERATIONS trips
# first in both runs — but a far narrower one than it looked, and the headroom
# is thinnest on exactly the runs that cost the most. Quote it as a backstop,
# never as "the budget we run to". A brownfield run with more clarification
# rounds, or a third refine iteration, could plausibly reach it.
#
# Skipping phase 1 on a refine pass (see agents/architect.py) barely moves this
# number, and the first estimate of it was wrong in an instructive way. Phase 1
# looked expensive because the architect's per-VISIT total is ~12.5k in / ~2k
# out — but that is almost entirely phase 2, which carries the whole repo
# representation. Phase 1 sees only the Context Record. Measured directly, on
# the same scenario with and without the change (runs 20260818T083414Z-60d10c09
# and 20260818T083516Z-e92aa7cf): one skipped phase-1 call is worth roughly 400
# input + 500 output tokens, so two refine rounds save ~1.6% of the run, not the
# ~27% a per-visit figure suggested. That change is for CONVERGENCE; treat its
# token saving as noise, and re-measure anything else here rather than
# subtracting it on paper.
#
# Two things to weigh before retuning it, because a naive tightening to ~30k
# would fire on nothing it is meant to catch:
#   * It is only evaluated HERE, at the refine gate, i.e. after a reviewer
#     failure. Tokens burned before the first gate visit — a runaway clarifier
#     call, for instance — are never checked against it at all.
#   * The failure worth catching is exactly that runaway: one observed clarifier
#     call spent 65,521 output tokens (~$0.098, 6x a whole successful greenfield
#     run) and still returned nothing usable. A per-call `max_output_tokens` in
#     llm.py would catch that; this whole-run cap, checked only at the gate,
#     would not.
MAX_TOTAL_TOKENS = 500_000

# How many times a human may send the run back to REDO work, over the whole run
# and across every touchpoint (see the module docstring for why it is one
# counter and not three; see `ArchitectState.user_rounds` for what counts).
#
# Still untuned as a NUMBER, but no longer untuned as a rule. It first shipped
# counting every user-initiated re-entry, and the first real session showed why
# that was wrong: four rounds of clarifying questions and one approval spent the
# whole budget of 6 before the architect had run once. Answering the clarifier
# is the pipeline working as designed, so it no longer counts — only post-lock
# refinement does, which is what this was always meant to bound.
#
# 6 is kept because it is now measured against a much smaller population of
# events (a real session used ZERO of these rounds), so it is comfortably a
# backstop rather than a live constraint. Retune from a session transcript that
# actually reaches it, not from taste.
MAX_USER_ROUNDS = 6

# Inputs to `derive_max_steps`. Kept as named constants rather than inlined
# because they are the two things a reader will want to argue with.
#   BASE_STEPS — nodes on a clean straight-through run: repo_ingestor, clarifier,
#                researcher, architect, reviewer, plus the terminal super-step.
#   STEP_MARGIN — slack for a stage that legitimately re-enters a node once more
#                (a resumed run replays the clarifier before advancing).
BASE_STEPS = 6
STEP_MARGIN = 2


def derive_max_steps(
    base_steps: int = BASE_STEPS,
    margin: int = STEP_MARGIN,
) -> int:
    """The `recursion_limit` every caller should pass, derived from the caps.

    ``base_steps + 3 * (MAX_REFINE_ITERATIONS + MAX_USER_ROUNDS) + margin``

    The 3 is the length of one loop through the graph: gate -> architect ->
    reviewer. Each refine iteration really does add that many super-steps to a
    SINGLE invocation, so that term is a tight bound.

    The `MAX_USER_ROUNDS` term is not, and pretending otherwise would be
    dishonest: `recursion_limit` is per INVOCATION, and every user round starts a
    fresh one, so human rounds cost this budget nothing. It is in the formula as
    deliberate headroom — the point of deriving the number is that raising ANY
    cap can never again surface as a `GraphRecursionError`, and that guarantee is
    worth more than a tight bound on a limit whose only real job is catching a
    mis-wired route (which it still does: a self-looping edge blows through this
    just as fast as it blew through the old hard-coded 20).
    """
    return base_steps + 3 * (MAX_REFINE_ITERATIONS + MAX_USER_ROUNDS) + margin


# ── Caller-side: the human-round budget ───────────────────────────────────
# These run OUTSIDE the graph, in ui.py, pipeline/run.py and
# pipeline/user_feedback.py, before a user-initiated re-entry that asks the
# pipeline to REDO work. They are here, not there, so every caller shares one
# policy instead of each inventing its own — and so the "post-lock only" rule is
# enforced in one place rather than remembered at four call sites.
#
# `reopen_for_user_round` is the third of them and the reason the other two are
# worth reading together: re-entering after feedback on a FINISHED run has to
# clear this module's own stop signals as well as charge the round, or the gate
# stops the run again on a cap that belongs to the work already done.
def evaluate_user_rounds(state: ArchitectState) -> tuple[bool, str]:
    """Pure decision function: has the human spent their round budget?

    Returns ``(exhausted, reason)``, mirroring `evaluate_caps` so both budgets
    read the same way at their call sites.
    """
    if state.user_rounds >= MAX_USER_ROUNDS:
        return True, f"max_user_rounds ({MAX_USER_ROUNDS})"
    return False, ""


def begin_user_round(state: ArchitectState) -> tuple[bool, str]:
    """Charge one POST-LOCK refinement round to `state`. Returns ``(allowed, reason)``.

    The ONE place `user_rounds` is incremented, so what does and does not cost a
    round is a fact about this function rather than a convention three call sites
    have to remember. Mutating here is safe and deliberate: this is caller-side
    code operating on the caller's own state object before the graph is entered —
    the same place ui.py already writes `clarification_answers`. The gate NODE
    below stays pure.

    Refuses to be called before a Context Record exists, rather than counting it.
    That is the "post-lock" half of the rule made enforceable: the original
    version charged for answering clarifying questions too, which drained the
    whole budget before the work it guards had started (see
    `ArchitectState.user_rounds`). A precondition is what stops that being
    re-introduced by a well-meaning caller who just wants to bound their own loop
    — bounding a clarifier that will not converge is `MAX_CLARIFICATION_ROUNDS`,
    not this.
    """
    if state.context_record is None:
        raise ValueError(
            "begin_user_round called before any Context Record was locked. This "
            "counter bounds user-initiated REFINEMENT of a design, not the "
            "clarification that produces the ground truth in the first place. "
            "To bound a clarifier that is not converging, use "
            "run.MAX_CLARIFICATION_ROUNDS."
        )
    exhausted, reason = evaluate_user_rounds(state)
    if exhausted:
        return False, reason
    state.user_rounds += 1
    return True, ""


def reopen_for_user_round(state: ArchitectState) -> tuple[bool, str]:
    """Charge one round AND clear the four fields that would silently eat it.

    Returns ``(allowed, reason)`` — the same shape as `begin_user_round`, which
    it delegates the charge to, so `user_rounds` keeps exactly one writer and
    the cap keeps exactly one copy. A refusal changes NOTHING: the caller must
    surface `reason` and record no feedback, or it will have accepted text it is
    about to drop.

    THE FOUR RESETS, and why every one of them is load-bearing:

      * `stopped_on_cap` and `refine_iterations` are the defect that makes "just
        set REFINING and re-run" a silent no-op. A run that ended on the cap
        re-enters at the gate, the gate re-trips the same cap, and the state
        goes straight back to DONE with the human's text unread and no error
        anywhere. The feedback is a new budget, so it starts with a fresh one.
      * `best_design` and `selected_round` are cleared because feedback
        REDEFINES WHAT BEST MEANS, so rounds scored before it are no longer
        comparable. Concretely, on the design path: the user says "use SQS
        instead of Kafka", the new round scores a point lower, the cap trips,
        and the gate hands back the incumbent — silently reverting the very
        change that was asked for. On the requirements path the incumbent was
        scored against a record that has since been superseded. Same clearing on
        both paths: the reason differs, the answer does not.

    Clearing the incumbent loses nothing recoverable. Every round's artifacts
    are already on disk in that round's `NNN_designing.json` checkpoint, which
    is why there is no `design_archive` field — one would re-serialise every
    past design into every future checkpoint to store what the run already has.

    Caller-side, like `begin_user_round`: it mutates the caller's own state
    before the graph is entered. The gate NODE stays pure.
    """
    allowed, reason = begin_user_round(state)
    if not allowed:
        return False, reason

    state.stopped_on_cap = False
    state.refine_iterations = 0
    state.best_design = None
    state.selected_round = 0
    return True, ""


def evaluate_caps(state: ArchitectState) -> tuple[bool, str]:
    """Pure decision function: should the refine loop stop now?

    Returns ``(stop, reason)``. Kept free of side effects so it can be
    unit-tested in isolation without building a graph. Checks the iteration
    cap first (cheap, deterministic), then the token budget.
    """
    if state.refine_iterations >= MAX_REFINE_ITERATIONS:
        return True, f"max_iterations ({MAX_REFINE_ITERATIONS})"
    if state.input_tokens + state.output_tokens >= MAX_TOTAL_TOKENS:
        used = state.input_tokens + state.output_tokens
        return True, f"token_budget ({used} >= {MAX_TOTAL_TOKENS})"
    return False, ""


# ── Best-so-far selection ─────────────────────────────────────────────────
# Kept in state.py beside RubricScores so the Reviewer, eval harness, and
# best-so-far selector cannot silently drift onto different field sets.
_CODE_SCORE_FIELDS = REVIEW_CODE_SCORE_FIELDS
_JUDGED_FIELDS = REVIEW_JUDGMENT_FIELDS


def score_round(review: ReviewResult) -> tuple:
    """Rank one reviewed design. HIGHER IS BETTER; pure, no side effects, no LLM.

    Returns a comparable tuple, so `max()` over rounds picks the winner and
    `>` means "strictly better". The ordering, most significant first:

      1. a PASS beats any fail.
      2. fewer HIGH-severity issues.
      3. higher sum of the five code-owned rubric scores (0-10).
      4. more of the five LLM judgments passing.
      5. fewer issues in total.

    WHY SEVERITY OUTRANKS THE CODE SCORES, which is the one genuinely arguable
    line here and is therefore stated rather than buried in a comparator: a
    design that structurally addresses the flaw but has a traceability gap is
    better than one that is tidy and misses the flaw. High-severity issues are
    where "misses the flaw" shows up; the code scores are largely about
    completeness and cross-referencing. Ranking tidiness above substance would
    let the loop optimise its own scoreboard. If that judgment is wrong, it is
    wrong HERE, in five lines, and can be argued with directly.

    Note what is NOT in the key: the round number. Ties are broken by the
    caller keeping its incumbent (`refine_gate_node` replaces only on strict
    `>`), which resolves them toward the EARLIER round. A later round that is
    merely equal is not an improvement, and preferring the incumbent makes
    selection stable — re-running the comparison never changes its own answer.
    """
    rubric = review.rubric_scores
    high_severity = sum(1 for issue in review.issues if issue.severity == "high")
    code_total = sum(getattr(rubric, name) for name in _CODE_SCORE_FIELDS)
    judged_passing = sum(1 for name in _JUDGED_FIELDS if getattr(rubric, name))
    return (
        review.overall_status == "pass",
        -high_severity,
        code_total,
        judged_passing,
        -len(review.issues),
    )


def _snapshot(state: ArchitectState, round_number: int) -> DesignSnapshot:
    """Bundle the artifacts currently on `state` with the review that judged them."""
    return DesignSnapshot(
        round=round_number,
        features=list(state.features),
        blueprint=state.blueprint,
        adrs=list(state.adrs),
        components=list(state.components),
        review=state.review,
    )


@node("refine_gate")
def refine_gate_node(state: ArchitectState) -> dict:
    """Decide loop-vs-stop after a failing review, and keep the best round. No LLM.

    * stop  -> stage=DONE, stopped_on_cap=True (do NOT bump the counter), and
               restore the best round's artifacts if it is not the current one.
    * loop  -> refine_iterations += 1, keep stage=REFINING; the orchestrator's
               `_gate_route` sends this on to the architect, which redesigns and
               sets stage=DESIGNING, so the reviewer runs again.

    Selection happens on EVERY visit, not only on the stop branch, because this
    node is the only place that sees each reviewed design before the next one
    overwrites it. It runs after every FAILING review including the last, which
    is exactly the set of rounds worth choosing between — a passing review ends
    the run at the reviewer and never arrives here, so there is nothing to
    select and no gate visit is added for it.
    """
    stop, reason = evaluate_caps(state)

    # Round 1 is the initial design; the counter has not been bumped for the
    # pass being judged right now, so the current round is one ahead of it.
    current_round = state.refine_iterations + 1
    incumbent = state.best_design

    # A gate visit always follows a reviewer verdict, so `review` is never None
    # in practice. Tolerated rather than asserted: failing the run over a
    # missing review would trade a graceful stop for a FAILED one, and the point
    # of this node is to end well.
    #
    # ONE VISIT DOES NOT follow a verdict: the first one after design feedback
    # re-enters the graph here (REFINING → refine_gate), so the artifacts on the
    # state are the ones the human just objected to and the review is the stale
    # verdict on them. Clearing `best_design` at the door is not enough on its
    # own — without this the gate would immediately re-nominate that same design
    # as the incumbent and could hand it back two rounds later, reverting the
    # directive it was meant to serve. A pending directive is the signal, not the
    # visit count: it is true exactly while a design nobody has yet redirected is
    # sitting on the state, and false again the moment the architect applies it.
    if state.review is None or state.pending_feedback("design"):
        best, took_lead = incumbent, False
    elif incumbent is None or score_round(state.review) > score_round(incumbent.review):
        # Strict `>`: an equal later round does not displace the incumbent.
        best, took_lead = _snapshot(state, current_round), True
    else:
        best, took_lead = incumbent, False

    if not stop:
        next_iteration = state.refine_iterations + 1
        lead = (
            f"round {current_round} is the new best"
            if took_lead
            else f"best remains round {best.round}" if best else "no scored round yet"
        )
        step = make_step(
            "refine_gate",
            state.stage,
            Stage.REFINING,
            f"refine {next_iteration}/{MAX_REFINE_ITERATIONS} → architect; {lead}",
        )
        return {
            "refine_iterations": next_iteration,
            "stage": Stage.REFINING,
            "best_design": best,
            "history": [step],
        }

    # ── stop: hand back the BEST round, not merely the last one ──────────
    update: dict = {
        "stage": Stage.DONE,
        "stopped_on_cap": True,
        "best_design": best,
        "selected_round": current_round,
    }
    if best is not None and best.round != current_round:
        discarded = current_round - best.round
        # The review travels WITH the artifacts. Restoring one without the other
        # would produce a report describing a design the run does not contain,
        # which is worse than shipping the later design untouched.
        update.update(
            features=list(best.features),
            blueprint=best.blueprint,
            adrs=list(best.adrs),
            components=list(best.components),
            review=best.review,
            selected_round=best.round,
        )
        note = (
            f"cap reached ({reason}); selected round {best.round} of "
            f"{current_round}, discarded {discarded} later round"
            f"{'s' if discarded != 1 else ''}"
        )
    else:
        # Say it out loud either way. An artifact swap that leaves no trace is
        # an audit-trail defect, and "nothing was swapped" is exactly as much a
        # fact about this run as "round 2 was restored".
        note = (
            f"cap reached ({reason}); round {current_round} of {current_round} "
            f"scored best, nothing restored"
        )

    update["history"] = [
        make_step("refine_gate", state.stage, Stage.DONE, note)
    ]
    return update
