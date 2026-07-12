"""researcher.py — Researcher agent (Kush).

Queries the knowledge base (RAG) via ``architect.retrieve_chunks`` and maps
the returned dicts onto :class:`KBChunk` objects for the architect.

LangGraph node form — returns partial state updates.
"""
from __future__ import annotations

from pipeline.agents.base import make_step, node
from pipeline.state import ArchitectState, KBChunk, Stage


@node("researcher")
def researcher_node(state: ArchitectState) -> dict:
    query_parts: list[str] = []
    if state.context_record:
        query_parts.append(state.context_record.problem_statement)
        query_parts.extend(state.context_record.non_functional_requirements)
    query = " ".join(p for p in query_parts if p).strip() or state.initial_request.raw_prompt

    from architect import retrieve_chunks
    chunks, origin = retrieve_chunks(query, k=3)

    retrieved_knowledge = [KBChunk(**c) for c in chunks]

    if not retrieved_knowledge:
        note = "no KB results"
    else:
        note = f"retrieved {len(retrieved_knowledge)} chunk(s) via {origin}"

    step = make_step(
        "researcher",
        state.stage,
        Stage.RESEARCHING,
        note,
    )
    return {
        "retrieved_knowledge": retrieved_knowledge,
        "stage": Stage.RESEARCHING,
        "history": [step],
    }
