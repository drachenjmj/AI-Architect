"""base.py — shared helpers every agent NODE uses.

MIGRATION NOTE (LangGraph)
--------------------------
This file used to hold an `Agent` base class whose `run()` owned the loop
(logging + error handling) via the template-method pattern. Under LangGraph
that loop is gone: **LangGraph itself** invokes each node and merges the
returned state updates. So an agent is no longer a subclass — it is a plain
**node function** `(state) -> dict-of-updates`.

What survives here are the two things every node still needs, factored out so
no node repeats them:

  * `make_step(...)`  — build one StepLog trace entry (the reducer on
                        `state.history` appends it; see state.py).
  * `@node(name)`     — wrap a node function so any exception becomes a clean
                        FAILED step instead of crashing the whole graph. This
                        preserves the old `base.run()` guarantee: one broken
                        agent must not take the pipeline down.

TEAM CONVENTION (how to write a real agent now)
-----------------------------------------------
    @node("myagent")
    def myagent_node(state: ArchitectState) -> dict:
        ...                                   # read state, do the work
        step = make_step("myagent", state.stage, Stage.NEXT, "one-line note")
        return {                              # return ONLY what you changed
            "some_artifact": value,
            "stage": Stage.NEXT,              # the stage you move the run INTO
            "history": [step],                # appended via the reducer
        }
Return partial updates — never mutate `state` in place and rely on it sticking;
LangGraph only persists what you RETURN.
"""
from __future__ import annotations

from typing import Callable

from pipeline.llm import LLMUsage, usage_from_exception
from pipeline.state import ArchitectState, Stage, StepLog


def make_step(
    agent: str,
    stage_in: Stage,
    stage_out: Stage,
    note: str = "",
    usage: LLMUsage | None = None,
) -> StepLog:
    """Build one trace entry. The `history` reducer appends it (see state.py).

    `usage` (optional) stamps what THIS agent's LLM calls consumed onto the
    step, which is what makes per-agent cost attribution a groupby over
    `history` (`ArchitectState.usage_by_agent`). Nodes that call no LLM leave it
    None and the step reads as zero. A node passing `usage` here must report the
    SAME numbers in its returned dict, so the trace and the run totals agree.
    """
    return StepLog(
        agent=agent,
        stage_in=stage_in,
        stage_out=stage_out,
        note=note,
        model=usage.model if usage else "",
        input_tokens=usage.input_tokens if usage else 0,
        output_tokens=usage.output_tokens if usage else 0,
        # None means "no verified price for this model"; on the step, which is a
        # plain float, that has to read as 0.0 — the tokens are still recorded.
        cost_usd=(usage.cost_usd or 0.0) if usage else 0.0,
    )


def node(name: str) -> Callable[[Callable[[ArchitectState], dict]], Callable[[ArchitectState], dict]]:
    """Decorator: make a node resilient.

    Wraps a node function so a raised exception is turned into a clean FAILED
    step + error entry (routed to END by the orchestrator), exactly like the
    old `Agent.run()` try/except. The node stays a pure `(state) -> dict`.

    TOKENS ON THE ERROR PATH. An agent can raise AFTER a call was already
    billed — the Architect validates its phase-1 result and may raise before
    phase 2. Dropping the FAILED update would lose those real tokens and
    understate the run. So a node attaches its usage-so-far to the exception
    (`llm.attach_usage`) and this wrapper reports it here, on the FAILED step
    and in the returned update. Exactly one update is returned per invocation —
    either the node's own or this one, never both — so nothing is double-counted.
    """

    def decorate(fn: Callable[[ArchitectState], dict]) -> Callable[[ArchitectState], dict]:
        def wrapped(state: ArchitectState) -> dict:
            try:
                return fn(state)
            except Exception as e:  # a broken agent must not crash the graph
                usage = usage_from_exception(e)
                step = make_step(name, state.stage, Stage.FAILED, f"error: {e}", usage)
                out = {"stage": Stage.FAILED, "history": [step], "errors": [f"{name}: {e}"]}
                if usage is not None:
                    out["input_tokens"] = usage.input_tokens
                    out["output_tokens"] = usage.output_tokens
                return out

        wrapped.__name__ = getattr(fn, "__name__", name)
        wrapped.__doc__ = fn.__doc__
        return wrapped

    return decorate
