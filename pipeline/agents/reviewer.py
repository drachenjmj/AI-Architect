"""reviewer.py — Reviewer agent (Waqar). STUB: always passes, no checks yet.

Real behaviour: check the design against the stated constraints, flag violations.
Mostly deterministic checks + one LLM call. This is the agent that can route the
run back to REFINING — the visible 'agentic' moment. The stub always passes, so
it routes straight to DONE.

LangGraph node form — faithful port of the old stub. When the real reviewer
lands, it returns Stage.REFINING instead of Stage.DONE when the design fails;
the orchestrator's router already turns that into the refine loop.
"""
from __future__ import annotations

from pipeline.agents.base import make_step, node
from pipeline.state import ArchitectState, ReviewResult, Stage


@node("reviewer")
def reviewer_node(state: ArchitectState) -> dict:
    review = ReviewResult(passed=True, issues=[])
    # stub always passes -> DONE. Real reviewer: Stage.REFINING when not passed.
    step = make_step("reviewer", state.stage, Stage.DONE, "review passed (stub)")
    return {
        "review": review,
        "stage": Stage.DONE,
        "history": [step],
    }
