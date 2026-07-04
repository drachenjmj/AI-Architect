"""clarifier.py — Clarifier agent (Kati). STUB: dummy data only, no LLM yet.

Real behaviour (Week 2): read raw_prompt, extract a draft understanding, decide
what is architecture-critically missing, and either PAUSE for the user
(awaiting input) or lock a frozen Context Record before design begins.

For now this is a faithful port of the old stub to the LangGraph node form:
same dummy writes, same target stage — no behaviour change.
"""
from __future__ import annotations

from pipeline.agents.base import make_step, node
from pipeline.state import ArchitectState, ContextRecord, Stage


@node("clarifier")
def clarifier_node(state: ArchitectState) -> dict:
    questions = [
        "What is the expected peak concurrency?",
        "Which cloud provider is preferred?",
    ]
    answers = {q: "[stub answer]" for q in questions}
    context_record = ContextRecord(summary="[stub] context frozen from prompt + answers")
    step = make_step(
        "clarifier",
        state.stage,
        Stage.CLARIFYING,
        f"asked {len(questions)} questions, wrote stub Context Record",
    )
    return {
        "clarifying_questions": questions,
        "clarification_answers": answers,
        "context_record": context_record,
        "stage": Stage.CLARIFYING,
        "history": [step],
    }
