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
Clarifier's directional `AWAITING_INPUT` — the `REFINING` stage is directional:
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
"""
from __future__ import annotations

from pipeline.agents.base import make_step, node
from pipeline.state import ArchitectState, Stage


# ── Cost-cap policy ───────────────────────────────────────────────────────
# Tunable. The iteration cap is what realistically trips in a demo; the token
# budget is a generous safety net — retune it after watching one real run's
# `input_tokens + output_tokens`.
MAX_REFINE_ITERATIONS = 2
MAX_TOTAL_TOKENS = 500_000


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


@node("refine_gate")
def refine_gate_node(state: ArchitectState) -> dict:
    """Decide loop-vs-stop after a failing review. Pure code, no LLM.

    * stop  -> stage=DONE, stopped_on_cap=True (do NOT bump the counter).
    * loop  -> refine_iterations += 1, keep stage=REFINING; the orchestrator's
               `_gate_route` sends this on to the architect, which redesigns and
               sets stage=DESIGNING, so the reviewer runs again.
    """
    stop, reason = evaluate_caps(state)

    if stop:
        step = make_step(
            "refine_gate",
            state.stage,
            Stage.DONE,
            f"cap reached ({reason}); accepting best-effort design",
        )
        return {
            "stage": Stage.DONE,
            "stopped_on_cap": True,
            "history": [step],
        }

    next_iteration = state.refine_iterations + 1
    step = make_step(
        "refine_gate",
        state.stage,
        Stage.REFINING,
        f"refine {next_iteration}/{MAX_REFINE_ITERATIONS} → architect",
    )
    return {
        "refine_iterations": next_iteration,
        "stage": Stage.REFINING,
        "history": [step],
    }
