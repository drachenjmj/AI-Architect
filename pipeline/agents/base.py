"""base.py — the shared Agent base class every agent inherits from.

Uses the *template method* pattern: the base owns the shared parts
(logging the step, catching errors) inside `run()`. A teammate adding a real
agent writes ONLY `_act` and sets two labels: `name` and `target_stage`.
The orchestrator only ever calls `.run()` and never cares which agent it is.

The missing-`_act` guard is a manual
NotImplementedError, so a forgotten `_act` errors when the agent is first
called at runtime rather than at class-creation time.
"""
from __future__ import annotations

from pipeline.state import ArchitectState, Stage


class Agent:
    """Base class every agent inherits from. Orchestrator only calls .run()."""

    name: str = "agent"          # each subclass sets a short id, e.g. "clarifier"
    target_stage: Stage          # the stage this agent moves the run INTO

    def _act(self, state: ArchitectState) -> str:
        """The agent's REAL work: read/write state fields, return a one-line note.
        This is the ONLY method a teammate has to write."""
        raise NotImplementedError("Each agent must implement _act()")

    def run(self, state: ArchitectState) -> ArchitectState:
        """Shared wrapper — logging + error handling live here, once, for all agents."""
        try:
            note = self._act(state)
            state.log_step(self.name, self.target_stage, note or "")
        except Exception as e:  # a broken agent must not crash the whole pipeline
            state.errors.append(f"{self.name}: {e}")
            state.log_step(self.name, Stage.FAILED, f"error: {e}")
        return state
