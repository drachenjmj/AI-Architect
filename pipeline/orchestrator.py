"""orchestrator.py — the deterministic spine (Kati).

Pure code, NO LLM. The orchestrator reads `state.stage`, looks up which agent
runs at that stage, runs it, and repeats until the run reaches DONE or FAILED.
All "reasoning" lives inside each agent's `_act`; routing here is 100% rules.

This is the backbone of the determinism map: every transition below is a rule.
"""
from __future__ import annotations

from pipeline.state import ArchitectState, Stage
from pipeline.agents import ClarifierStub, ResearcherStub, ArchitectStub, ReviewerStub

# ROUTING TABLE: current stage -> which agent runs next.
# One instance per agent, reused across the run (cheap; agents are stateless).
ROUTES: dict[Stage, object] = {
    Stage.CREATED:     ClarifierStub(),
    Stage.CLARIFYING:  ResearcherStub(),
    Stage.RESEARCHING: ArchitectStub(),
    Stage.DESIGNING:   ReviewerStub(),
    # TODO(W3): Stage.REFINING -> ArchitectStub()  # reviewer-triggered refine loop
}

TERMINAL = {Stage.DONE, Stage.FAILED}


def run_pipeline(state: ArchitectState, max_steps: int = 20) -> ArchitectState:
    """Loop: look at state.stage, run the matching agent, repeat until DONE/FAILED.

    max_steps is a hard safety cap so a mis-wired route can never loop forever
    (also the natural home for the Week-3 cost/retry guardrail).
    """
    for _ in range(max_steps):
        if state.stage in TERMINAL:
            break
        agent = ROUTES.get(state.stage)
        if agent is None:  # no route defined for this stage -> fail loudly, don't hang
            state.errors.append(f"no agent for stage {state.stage.value}")
            state.log_step("orchestrator", Stage.FAILED, "unroutable stage")
            break
        agent.run(state)  # agent does its work + advances the stage via log_step
    else:
        # loop finished without hitting `break` = we ran out of steps
        state.errors.append(f"max_steps ({max_steps}) reached before DONE")
        state.log_step("orchestrator", Stage.FAILED, "step cap reached")
    return state
