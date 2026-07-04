"""orchestrator.py — the deterministic spine (Kati), now built on LangGraph.

WHAT CHANGED (and what did NOT)
-------------------------------
Previously this was a hand-rolled `while` loop over a `ROUTES` dict. It is now a
LangGraph `StateGraph`. The CONTROL LOGIC is identical and still 100% code:

  * The `ArchitectState` object is the graph state (unchanged contract).
  * Each agent is a node function (see agents/base.py).
  * Routing is a pure Python function, `_route`, driven ONLY by `state.stage`
    via the `STAGE_TO_NODE` table below. No LLM decides routing — that is the
    determinism principle, preserved. This table IS the determinism map.
  * `max_steps` maps to LangGraph's `recursion_limit`, so a mis-wired route can
    never loop forever (fails loudly, never hangs).

The public entry point `run_pipeline(state, max_steps)` keeps its old signature,
so run.py, the UI, and tests call it exactly as before.

WHY LangGraph: it hands us — for free and as industry-standard primitives — the
two things still on Kati's roadmap: human-in-the-loop PAUSE (`interrupt`) for the
clarifier, and state-on-disk recovery (`checkpointer`). We build on the standard
container instead of extending bespoke loop code.
"""
from __future__ import annotations

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from pipeline.agents import (
    architect_node,
    clarifier_node,
    researcher_node,
    reviewer_node,
)
from pipeline.state import ArchitectState, Stage

# ── DETERMINISM MAP ───────────────────────────────────────────────────────
# current stage -> which node runs next. Pure rules; the single source of
# truth for routing. A stage absent here (DONE, FAILED, or anything unwired)
# routes to END, so the graph always terminates.
STAGE_TO_NODE: dict[Stage, str] = {
    Stage.CREATED:        "clarifier",
    Stage.CLARIFYING:     "researcher",
    Stage.AWAITING_INPUT: END,   # clarifier needs the human → pause (see _route)
    Stage.RESEARCHING:    "architect",
    Stage.DESIGNING:      "reviewer",
    # TODO(W3): Stage.REFINING -> "architect"   # reviewer-triggered refine loop
}

_NODES = {
    "clarifier": clarifier_node,
    "researcher": researcher_node,
    "architect": architect_node,
    "reviewer": reviewer_node,
}

# Path map for the conditional edges: every value _route can return.
_PATH_MAP = {name: name for name in _NODES}
_PATH_MAP[END] = END


def _route(state: ArchitectState) -> str:
    """Deterministic router for the edges AFTER each node.

    Terminal or unwired stages -> END. This is the old `ROUTES.get(...) or fail`
    logic, expressed as a LangGraph conditional-edge function. Here
    `AWAITING_INPUT -> END` — reaching it mid-run means "pause for the human."
    """
    return STAGE_TO_NODE.get(state.stage, END)


def _entry_route(state: ArchitectState) -> str:
    """Deterministic router for the START edge ONLY (graph entry / resume).

    `AWAITING_INPUT` is directional: reaching it mid-run means pause (`_route`
    sends it to END), but ENTERING the graph already in `AWAITING_INPUT` means
    the user has just supplied answers and we must RE-RUN the clarifier to
    re-judge. Everything else defers to the normal table.
    """
    if state.stage is Stage.AWAITING_INPUT:
        return "clarifier"
    return _route(state)


def _build_graph():
    """Assemble and compile the StateGraph. One graph, reused across runs."""
    g = StateGraph(ArchitectState)
    for name, fn in _NODES.items():
        g.add_node(name, fn)
    # Entry uses _entry_route (handles resume); after every node uses _route.
    g.add_conditional_edges(START, _entry_route, _PATH_MAP)
    for name in _NODES:
        g.add_conditional_edges(name, _route, _PATH_MAP)
    return g.compile()


GRAPH = _build_graph()


def run_pipeline(state: ArchitectState, max_steps: int = 20) -> ArchitectState:
    """Drive one state through the graph until DONE/FAILED. Same signature as before.

    `max_steps` -> LangGraph `recursion_limit`: a hard safety cap so a mis-wired
    route can never loop forever (also the natural home for the Week-3 guardrail).
    """
    try:
        result = GRAPH.invoke(state, config={"recursion_limit": max_steps})
        return ArchitectState.model_validate(result)
    except GraphRecursionError:
        # step cap hit — mark FAILED loudly instead of hanging (old behaviour).
        state.errors.append(f"max_steps ({max_steps}) reached before DONE")
        state.log_step("orchestrator", Stage.FAILED, "step cap reached")
        return state
