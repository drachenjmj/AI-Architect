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

STATE ON DISK (added)
---------------------
Every state this module yields is checkpointed to `.cache/runs/<run_id>/` by
`pipeline.persistence`. That is deterministic file I/O, not a judgment call, so
it belongs here and does not break the LLM-free rule. The hook is ONE line
inside `run_pipeline_streaming`; `run_pipeline` delegates to that same
generator, so there is one code path and no second call site to keep in sync.
"""
from __future__ import annotations

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from pipeline.agents import (
    architect_node,
    clarifier_node,
    repo_ingestor_node,
    researcher_node,
    reviewer_node,
)
from pipeline.persistence import checkpoint
from pipeline.refine_gate import refine_gate_node
from pipeline.state import ArchitectState, Stage

# ── DETERMINISM MAP ───────────────────────────────────────────────────────
# current stage -> which node runs next. Pure rules; the single source of
# truth for routing. A stage absent here (DONE, FAILED, or anything unwired)
# routes to END, so the graph always terminates.
STAGE_TO_NODE: dict[Stage, str] = {
    Stage.CREATED:        "repo_ingestor",  # read the repo FIRST (or skip: greenfield)
    Stage.INGESTING:      "clarifier",      # so the clarifier can ground its questions in it
    Stage.CLARIFYING:     "researcher",     # context locked → research
    Stage.AWAITING_INPUT: END,   # clarifier needs the human → pause (see _route)
    Stage.RESEARCHING:    "architect",
    Stage.DESIGNING:      "reviewer",
    Stage.REFINING:       "refine_gate",  # reviewer failed → cost-cap gate decides loop-vs-stop
}

_NODES = {
    "clarifier": clarifier_node,
    "repo_ingestor": repo_ingestor_node,
    "researcher": researcher_node,
    "architect": architect_node,
    "reviewer": reviewer_node,
    "refine_gate": refine_gate_node,
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


def _gate_route(state: ArchitectState) -> str:
    """Deterministic router for the edge AFTER the refine gate ONLY.

    The gate has already made the cost-cap decision and recorded it in
    `stopped_on_cap`: True ⇒ stop the loop (→ END, the run is DONE); False ⇒
    afford one more pass (→ architect, which redesigns from the reviewer's
    instruction). This is why `REFINING` is directional — the same stage routes
    to the gate after the reviewer but to the architect after the gate.
    """
    return END if state.stopped_on_cap else "architect"


def _build_graph():
    """Assemble and compile the StateGraph. One graph, reused across runs."""
    g = StateGraph(ArchitectState)
    for name, fn in _NODES.items():
        g.add_node(name, fn)
    # Entry uses _entry_route (handles resume); after every node uses _route,
    # EXCEPT the refine gate, whose outgoing edge uses _gate_route (loop-vs-stop
    # on the cost cap — see _gate_route). Reusing _route there would self-loop,
    # since REFINING maps back to the gate.
    g.add_conditional_edges(START, _entry_route, _PATH_MAP)
    for name in _NODES:
        if name == "refine_gate":
            g.add_conditional_edges(name, _gate_route, {"architect": "architect", END: END})
        else:
            g.add_conditional_edges(name, _route, _PATH_MAP)
    return g.compile()


GRAPH = _build_graph()


def run_pipeline(state: ArchitectState, max_steps: int = 20) -> ArchitectState:
    """Drive one state through the graph until DONE/FAILED. Same signature as before.

    `max_steps` -> LangGraph `recursion_limit`: a hard safety cap so a mis-wired
    route can never loop forever (also the natural home for the Week-3 guardrail).

    Expressed as "run the stream, keep the last value". The stream's final
    emission is exactly what `GRAPH.invoke` returned before — an equivalence
    pinned by `test_persistence.test_stream_final_emission_matches_invoke`, so a
    LangGraph upgrade that breaks it fails a test instead of silently changing
    what this returns. Delegating also means checkpointing lives on ONE code
    path shared by both entry points.
    """
    final = state
    for snapshot in run_pipeline_streaming(state, max_steps):
        final = snapshot
    return final


def run_pipeline_streaming(state: ArchitectState, max_steps: int = 20):
    """Yield the validated state after EVERY node, for live progress display.

    Same graph, routing and step cap as `run_pipeline`; the only difference is
    that it surfaces the intermediate state after each node instead of only the
    final one. The LAST yielded value is the terminal state (identical to what
    `run_pipeline` returns), so callers can simply keep the last item.

    THE CHECKPOINT SITE. `stream_mode="values"` emits the FULL state once per
    super-step, so saving each emission records every agent transition —
    including the FAILED transition the `@node` wrapper produces when an agent
    raises, which arrives here as an ordinary emission and needs no special
    case. `checkpoint` never raises: a lost checkpoint must not cost us the run.
    """
    latest = state       # newest state seen, checkpointed or not
    saved = None         # newest state actually handed to `checkpoint`
    try:
        for chunk in GRAPH.stream(
            state, config={"recursion_limit": max_steps}, stream_mode="values"
        ):
            latest = saved = ArchitectState.model_validate(chunk)
            checkpoint(latest)  # ← THE hook: one call, every transition
            yield latest
    except GraphRecursionError:
        # Step cap hit — mark FAILED loudly instead of hanging (old behaviour).
        #
        # KNOWN DEFERRAL (mutation): unlike every other path, this one MUTATES
        # the caller's `state` instead of returning a fresh object, so
        # run_pipeline's "does not mutate its input" contract holds everywhere
        # except here. Left alone deliberately: it predates state-on-disk, and
        # callers (run.py, ui.py) currently observe the cap failure through the
        # object they passed in. Changing it is a contract change that belongs
        # in its own commit, not smuggled in with persistence. Recorded in
        # DETERMINISM_MAP.md so it is a tracked deferral, not a lurking bug.
        state.errors.append(f"max_steps ({max_steps}) reached before DONE")
        state.log_step("orchestrator", Stage.FAILED, "step cap reached")
        latest = state
        yield state
    finally:
        # Safety net for the step-cap path ONLY. That FAILED state is born from
        # an exception, never from a stream emission, so the hook above never
        # sees it — and a blowout is exactly when a resume point matters most.
        # The identity guard keeps the happy path free of a duplicate final
        # checkpoint: there, `latest is saved` and this is a no-op.
        if latest is not saved:
            checkpoint(latest)
