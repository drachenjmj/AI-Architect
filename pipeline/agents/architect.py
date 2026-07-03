"""architect.py — Architect/Writer agent (Maheen). STUB: dummy data only, no LLM yet.

Real behaviour (two-phase): derive features FIRST, then design the architecture
from them — filling Blueprint, ADRs and Component Descriptions, each traceable
back to a feature.
"""
from __future__ import annotations

from pipeline.agents.base import Agent
from pipeline.state import (
    ADR,
    ArchitectState,
    Blueprint,
    ComponentDescription,
    Feature,
    Stage,
)


class ArchitectStub(Agent):
    name = "architect"
    target_stage = Stage.DESIGNING

    def _act(self, state: ArchitectState) -> str:
        # Phase 1: features first
        state.features = [
            Feature(id="F1", name="Handle peak load", scenario="Stays responsive at 50k concurrent users.")
        ]
        # Phase 2: design derived from the features
        state.blueprint = Blueprint(
            stakeholder_view="[stub] Business view of the system.",
            technical_view="[stub] Services, data model, integration.",
        )
        state.adrs = [ADR(title="ADR-1: split monolith into services", decision="[stub] decision text")]
        state.components = [ComponentDescription(name="OrderService", description="[stub] owns orders (traces to F1).")]
        return f"derived {len(state.features)} feature(s), drafted blueprint + {len(state.adrs)} ADR + {len(state.components)} component"
