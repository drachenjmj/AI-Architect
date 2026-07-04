"""researcher.py — Researcher agent (Kush). STUB: dummy data only, no retrieval yet.

Real behaviour: query the knowledge base (RAG) and synthesize relevant
architecture patterns / domain facts for the architect.

LangGraph node form — faithful port of the old stub (same dummy write).
"""
from __future__ import annotations

from pipeline.agents.base import make_step, node
from pipeline.state import ArchitectState, KBChunk, Stage


@node("researcher")
def researcher_node(state: ArchitectState) -> dict:
    retrieved = [
        KBChunk(
            content="[stub] Decouple services with an async message queue to absorb peak load.",
            source="architecture_patterns.md",
        )
    ]
    step = make_step(
        "researcher",
        state.stage,
        Stage.RESEARCHING,
        f"retrieved {len(retrieved)} KB chunk(s)",
    )
    return {
        "retrieved_knowledge": retrieved,
        "stage": Stage.RESEARCHING,
        "history": [step],
    }
