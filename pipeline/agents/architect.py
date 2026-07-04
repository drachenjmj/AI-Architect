"""architect.py — Architect/Writer agent (Maheen). STUB: dummy data only, no LLM yet.

Real behaviour (two-phase): derive features FIRST, then design the architecture
from them — filling Blueprint, ADRs and Component Descriptions, each traceable
back to a feature.

LangGraph node form — faithful port of the old stub (same dummy writes).
"""
from __future__ import annotations

from pipeline.agents.base import make_step, node
from pipeline.state import (
    ADR,
    ArchitectState,
    Blueprint,
    ComponentDescription,
    Feature,
    Stage,
)


@node("architect")
def architect_node(state: ArchitectState) -> dict:
    # Phase 1: features first
    features = [
        Feature(id="F1", name="Handle peak load", scenario="Stays responsive at 50k concurrent users.")
    ]
    # Phase 2: design derived from the features
    blueprint = Blueprint(
        stakeholder_view="[stub] Business view of the system.",
        technical_view="[stub] Services, data model, integration.",
    )
    adrs = [ADR(title="ADR-1: split monolith into services", decision="[stub] decision text")]
    components = [ComponentDescription(name="OrderService", description="[stub] owns orders (traces to F1).")]
    step = make_step(
        "architect",
        state.stage,
        Stage.DESIGNING,
        f"derived {len(features)} feature(s), drafted blueprint + {len(adrs)} ADR + {len(components)} component",
    )
    return {
        "features": features,
        "blueprint": blueprint,
        "adrs": adrs,
        "components": components,
        "stage": Stage.DESIGNING,
        "history": [step],
    }
