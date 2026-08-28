"""test_refine_loop.py — offline tests for the reviewer→refine loop (Kati).

No API key or network needed. The Architect and Reviewer LLM calls are replaced
with deterministic canned responses (the Architect mock is reused from
test_clarifier). Covers three layers:

  1. `evaluate_caps` in isolation (pure cost-policy decision).
  2. The routing wiring (the determinism map for REFINING).
  3. The full loop through `run_pipeline`: it must stop on the iteration cap and
     on the token budget, and finish gracefully as DONE (never FAILED).
"""
from __future__ import annotations

import test_clarifier as tc  # reuse the canned Architect response
from pipeline import orchestrator
from pipeline.agents import architect as arch
from pipeline.agents import reviewer as rev
from pipeline.orchestrator import _gate_route, _route
from pipeline.refine_gate import (
    MAX_REFINE_ITERATIONS,
    MAX_TOTAL_TOKENS,
    evaluate_caps,
)
from pipeline.state import ContextRecord, Stage, new_run

from langgraph.graph import END


PROMPT = "Fix our monolithic online shop so it survives seasonal peak load."


def _failing_review(state, prompt, **kwargs):
    """All-default judgments (every criterion passed=False) → reviewer fails."""

    return rev.LLMJudgments(), tc.fake_usage()


def _seed_designable_state(**overrides):
    """A state parked at RESEARCHING with a locked context, so the Architect
    runs first (RESEARCHING → architect) without needing the clarifier/researcher.
    """

    state = new_run(PROMPT)
    state.context_record = ContextRecord(
        project_name="Seasonal Shop",
        summary="cloud: AWS; scale: 50k peak users; compliance: GDPR",
    )
    state.stage = Stage.RESEARCHING
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


# ── 1. evaluate_caps in isolation ────────────────────────────────────────
def test_evaluate_caps_pure():
    fresh = new_run(PROMPT)
    assert evaluate_caps(fresh) == (False, "")

    at_iter_cap = new_run(PROMPT)
    at_iter_cap.refine_iterations = MAX_REFINE_ITERATIONS
    stop, reason = evaluate_caps(at_iter_cap)
    assert stop and "max_iterations" in reason

    over_budget = new_run(PROMPT)
    over_budget.input_tokens = MAX_TOTAL_TOKENS
    stop, reason = evaluate_caps(over_budget)
    assert stop and "token_budget" in reason


# ── 2. routing wiring (the determinism map for REFINING) ──────────────────
def test_refining_routing_is_directional():
    # After the reviewer, REFINING routes to the gate...
    refining = _seed_designable_state(stage=Stage.REFINING)
    assert _route(refining) == "refine_gate"

    # ...and after the gate, the SAME stage routes on the cost decision:
    looping = _seed_designable_state(stage=Stage.REFINING, stopped_on_cap=False)
    assert _gate_route(looping) == "architect"
    capped = _seed_designable_state(stage=Stage.REFINING, stopped_on_cap=True)
    assert _gate_route(capped) == END

    # A passing review (→ DONE) is never sent to the gate.
    assert _route(_seed_designable_state(stage=Stage.DONE)) == END


# ── 3. full loop: stops on the iteration cap, ends DONE ───────────────────
def test_loop_stops_on_iteration_cap():
    arch.llm_call = tc._architect_response
    rev.llm_call = _failing_review

    state = orchestrator.run_pipeline(_seed_designable_state())

    assert state.stage is Stage.DONE          # graceful, not FAILED
    assert state.stopped_on_cap is True
    assert state.refine_iterations == MAX_REFINE_ITERATIONS

    architect_runs = [s for s in state.history if s.agent == "architect"]
    gate_runs = [s for s in state.history if s.agent == "refine_gate"]
    # initial design + MAX refinements; the gate's last visit is the one that stops
    assert len(architect_runs) == MAX_REFINE_ITERATIONS + 1
    assert len(gate_runs) == MAX_REFINE_ITERATIONS + 1
    assert "max_iterations" in state.history[-1].note


# ── 3b. full loop: stops on the token budget before the iteration cap ─────
def test_loop_stops_on_token_budget():
    arch.llm_call = tc._architect_response
    rev.llm_call = _failing_review

    # Seed the run already at the token ceiling → the first gate visit stops it
    # on the budget, before any refinement iteration is spent.
    state = _seed_designable_state(input_tokens=MAX_TOTAL_TOKENS)
    state = orchestrator.run_pipeline(state)

    assert state.stage is Stage.DONE
    assert state.stopped_on_cap is True
    assert state.refine_iterations == 0            # capped before looping
    assert "token_budget" in state.history[-1].note


if __name__ == "__main__":
    test_evaluate_caps_pure()
    print("PASS  evaluate_caps decides both caps correctly")

    test_refining_routing_is_directional()
    print("PASS  REFINING routing is directional (reviewer→gate, gate→architect)")

    test_loop_stops_on_iteration_cap()
    print("PASS  loop stops on the iteration cap and ends DONE")

    test_loop_stops_on_token_budget()
    print("PASS  loop stops on the token budget and ends DONE")

    print("\nALL REFINE-LOOP TESTS PASSED")
