"""researcher.py — Researcher agent (Kush). STUB: dummy data only, no retrieval yet.

Real behaviour: query the knowledge base (RAG) and synthesize relevant
architecture patterns / domain facts for the architect.
"""
from __future__ import annotations

from pipeline.agents.base import Agent
from pipeline.state import ArchitectState, KBChunk, Stage


class ResearcherStub(Agent):
    name = "researcher"
    target_stage = Stage.RESEARCHING

    def _act(self, state: ArchitectState) -> str:
        state.retrieved_knowledge = [
            KBChunk(
                content="[stub] Decouple services with an async message queue to absorb peak load.",
                source="architecture_patterns.md",
            )
        ]
        return f"retrieved {len(state.retrieved_knowledge)} KB chunk(s)"
