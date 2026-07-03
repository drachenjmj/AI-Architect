"""clarifier.py — Clarifier agent (Kati). STUB: dummy data only, no LLM yet.

Real behaviour (Week 2): generate clarifying questions from the prompt, then
write the answers into a frozen Context Record before design begins.
"""
from __future__ import annotations

from pipeline.agents.base import Agent
from pipeline.state import ArchitectState, ContextRecord, Stage


class ClarifierStub(Agent):
    name = "clarifier"
    target_stage = Stage.CLARIFYING

    def _act(self, state: ArchitectState) -> str:
        state.clarifying_questions = [
            "What is the expected peak concurrency?",
            "Which cloud provider is preferred?",
        ]
        state.clarification_answers = {q: "[stub answer]" for q in state.clarifying_questions}
        state.context_record = ContextRecord(summary="[stub] context frozen from prompt + answers")
        return f"asked {len(state.clarifying_questions)} questions, wrote stub Context Record"
