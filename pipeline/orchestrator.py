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
    never loop forever (fails loudly, never hangs). It is now DERIVED from the
    cost caps (`MAX_STEPS`, below) rather than hand-picked, so raising a cap can
    never again surface as a `GraphRecursionError` — i.e. as a crash.

ONE PAUSE STAGE, ONE DISCRIMINATOR
-----------------------------------
There is exactly one human-in-the-loop stage, `AWAITING_HUMAN`, and
`state.pending_decision` says which decision is owed. `_route` does not read the
discriminator at all — every kind of pause ends the invocation the same way —
and `_entry_route` reads it only to REFUSE the caller-resolved cases (see
below: `CONTEXT_LOCK`, and `OPTIONAL_CONTEXT` alongside it). That is the whole
cost of adding a human touchpoint to this file: nothing, plus one refusal per
caller-resolved pause. A new `AWAITING_*` stage per interaction would instead
have made the stage enum the router's problem, one row of `STAGE_TO_NODE` at a
time.

The public entry point `run_pipeline(state, max_steps)` keeps its old signature,
so run.py, the UI, and tests call it exactly as before.

WHY LangGraph: it gave us — for free and as industry-standard primitives — the
two things this rewrite needed: human-in-the-loop PAUSE (the `AWAITING_HUMAN`
stage, below) for the clarifier, and state-on-disk recovery (`checkpointer`,
see below). We build on the standard container instead of extending bespoke
loop code.

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
from pipeline.refine_gate import derive_max_steps, refine_gate_node
from pipeline.state import ArchitectState, PendingDecision, Stage

# The step cap every entry point defaults to. DERIVED from the budget caps that
# actually determine it (see refine_gate.derive_max_steps), not the hard-coded 20
# it replaces: that constant was silently coupled to MAX_REFINE_ITERATIONS, so
# raising a cost cap surfaced as a GraphRecursionError -> FAILED, i.e. a budget
# change arriving as a crash. Computed once at import; the caps are constants.
MAX_STEPS = derive_max_steps()

# ── DETERMINISM MAP ───────────────────────────────────────────────────────
# current stage -> which node runs next. Pure rules; the single source of
# truth for routing. A stage absent here (DONE, ACCEPTED, FAILED, or anything
# unwired) routes to END, so the graph always terminates.
#
# ACCEPTED is absent for the same reason DONE is: it is terminal. It is also
# unreachable from inside the graph — no node writes it. It is written by an
# explicit human action at DONE, outside the graph (pipeline/sign_off.py), which
# is why adding a second terminal stage cost this file no row and no branch.
STAGE_TO_NODE: dict[Stage, str] = {
    Stage.CREATED:        "repo_ingestor",  # read the repo FIRST (or skip: greenfield)
    Stage.INGESTING:      "clarifier",      # so the clarifier can ground its questions in it
    Stage.CLARIFYING:     "researcher",     # context locked → research
    Stage.AWAITING_HUMAN: END,   # a human owes us a decision → pause (see _route)
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
    `AWAITING_HUMAN -> END` — reaching it mid-run means "pause for the human."
    WHICH decision is pending does not matter to this direction: every kind of
    pause ends the invocation, so the discriminator stays out of the router.
    """
    return STAGE_TO_NODE.get(state.stage, END)


def _entry_route(state: ArchitectState) -> str:
    """Deterministic router for the START edge ONLY (graph entry / resume).

    `AWAITING_HUMAN` is directional: reaching it mid-run means pause (`_route`
    sends it to END), but ENTERING the graph already in `AWAITING_HUMAN` means
    the human has resolved their side — supplied answers, or edited the locked
    record in a way that opened a gap — and we must RE-RUN the clarifier to
    re-judge. Everything else defers to the normal table.

    TWO resolutions never come here: `CONTEXT_LOCK` and `OPTIONAL_CONTEXT`. The
    whole point of both pauses is that the caller resolves them OUTSIDE the
    graph (accept, edit, ask, resolve-optional — see pipeline/agents/
    clarifier.py), so arriving with either still pending means a caller
    re-entered on a half-resolved pause. Routing to the clarifier anyway would
    "work": it would burn a call, re-lock, and pause again, and the bug would
    look like a slow gate rather than a wiring error. So both are refused
    here, loudly, at the only place that can still tell the difference. Not an
    `assert` statement — those vanish under `python -O`, and a routing invariant
    that switches itself off in optimised mode is not an invariant.
    """
    if state.stage is Stage.AWAITING_HUMAN:
        if state.pending_decision is PendingDecision.CONTEXT_LOCK:
            raise RuntimeError(
                "run_pipeline was entered while a CONTEXT_LOCK decision is still "
                "pending. That pause is resolved by the CALLER, before re-entry: "
                "clarifier.accept_context_lock (approve), or "
                "clarifier.submit_context_edits + clarifier.open_for_rejudge "
                "(edit). clarifier.ask_advisor never enters the graph at all."
            )
        if state.pending_decision is PendingDecision.OPTIONAL_CONTEXT:
            raise RuntimeError(
                "run_pipeline was entered while an OPTIONAL_CONTEXT decision is "
                "still pending. Like CONTEXT_LOCK, that pause is resolved by the "
                "CALLER, before re-entry: clarifier.resolve_optional_context "
                "(skip, or apply answers) — it advances to CONTEXT_LOCK without "
                "ever entering the graph."
            )
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


def run_pipeline(state: ArchitectState, max_steps: int = MAX_STEPS) -> ArchitectState:
    """Drive one state through the graph until DONE/FAILED. Same signature as before.

    `max_steps` -> LangGraph `recursion_limit`: a hard safety cap so a mis-wired
    route can never loop forever. It DEFAULTS to `MAX_STEPS`, derived from the
    cost caps rather than hard-coded, so raising a cap can never again present
    itself as a crash. Callers may still pass a smaller number to test the cap.

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


def run_pipeline_streaming(state: ArchitectState, max_steps: int = MAX_STEPS):
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
        # The FAILED marker goes on a COPY, never on the caller's object. Every
        # other path already returns a freshly validated state, so copying here
        # is what makes "run_pipeline does not mutate its input" true on ALL
        # paths rather than merely most of them — a caller that kept the
        # pre-run state to retry or compare against still holds it untouched.
        failed = state.model_copy(deep=True)
        failed.errors.append(f"max_steps ({max_steps}) reached before DONE")
        failed.log_step("orchestrator", Stage.FAILED, "step cap reached")
        latest = failed
        yield failed
    finally:
        # Safety net for the step-cap path ONLY. That FAILED state is born from
        # an exception, never from a stream emission, so the hook above never
        # sees it — and a blowout is exactly when a resume point matters most.
        # The identity guard keeps the happy path free of a duplicate final
        # checkpoint: there, `latest is saved` and this is a no-op.
        if latest is not saved:
            checkpoint(latest)
