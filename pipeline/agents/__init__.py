"""agents/ — the four pipeline agent NODES (stubs for now, real behaviour later).

Under LangGraph an agent is a plain node function `(state) -> dict-of-updates`
(see base.py). `base` exposes the shared helpers every node uses.
"""
from pipeline.agents.base import make_step, node
from pipeline.agents.clarifier import clarifier_node
from pipeline.agents.researcher import researcher_node
from pipeline.agents.architect import architect_node
from pipeline.agents.reviewer import reviewer_node

__all__ = [
    "make_step",
    "node",
    "clarifier_node",
    "researcher_node",
    "architect_node",
    "reviewer_node",
]
