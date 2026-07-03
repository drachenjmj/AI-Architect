"""agents/ — the four pipeline agents (stubs for now, real behaviour later)."""
from pipeline.agents.base import Agent
from pipeline.agents.clarifier import ClarifierStub
from pipeline.agents.researcher import ResearcherStub
from pipeline.agents.architect import ArchitectStub
from pipeline.agents.reviewer import ReviewerStub

__all__ = [
    "Agent",
    "ClarifierStub",
    "ResearcherStub",
    "ArchitectStub",
    "ReviewerStub",
]
