"""reviewer.py — Reviewer agent (Waqar). STUB: always passes, no checks yet.

Real behaviour: check the design against the stated constraints, flag violations.
Mostly deterministic checks + one LLM call. This is the agent that can route the
run back to REFINING — the visible 'agentic' moment. The stub always passes, so
it routes straight to DONE.
"""
from __future__ import annotations

from pipeline.agents.base import Agent
from pipeline.state import ArchitectState, ReviewResult, Stage


class ReviewerStub(Agent):
    name = "reviewer"
    target_stage = Stage.DONE  # stub always passes; real reviewer may go to REFINING

    def _act(self, state: ArchitectState) -> str:
        state.review = ReviewResult(passed=True, issues=[])
        return "review passed (stub)"
