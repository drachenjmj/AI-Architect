"""clarifier.py — Clarifier agent (Kati). The first REAL (non-stub) node.

WHAT IT DOES
------------
The Clarifier is the ONLY LLM in the clarification phase. It reads the raw
prompt (plus any answers gathered on a previous turn), and produces a
structured judgment — a `ClarificationResult` — of what it understood and what
is still missing. Then DETERMINISTIC CODE (the "gate") decides what happens:

  * `missing_critical` non-empty  → something architecture-critical is unknown →
    surface the questions and PAUSE the run (stage AWAITING_INPUT). The
    orchestrator routes that to END; the caller collects answers, writes them
    into state, and re-runs — we land back here and re-judge.
  * `missing_critical` empty      → we know enough → LOCK a Context Record and
    advance so the Researcher can run.

This is the "LLM judges / code routes" split: the model fills in the judgment,
but a plain `if` makes the go/no-go decision so control flow stays deterministic
and unit-testable.

THE ASSUME-VS-ASK RULE (the core invariant, enforced by the system prompt)
--------------------------------------------------------------------------
Every gap is one of two kinds:
  * architecture-critical (would change the design: brownfield/greenfield,
    scale/users, compliance, cloud, budget, availability) → NEVER assume → ask.
  * low-stakes / safely inferable → make a LABELLED assumption a human can veto.
Never SILENTLY assume, and never assume anything architecture-critical.
"""
from __future__ import annotations

from pipeline.agents.base import make_step, node
from pipeline.llm import llm_call
from pipeline.state import (
    ArchitectState,
    ClarificationResult,
    ContextRecord,
    Stage,
)

# The clarifier's judgment sets the quality of everything downstream, so it uses
# the stronger model rather than the cheap default.
CLARIFIER_MODEL = "flash"

# The system prompt carries the POLICY (assume-vs-ask). The OUTPUT SHAPE is
# enforced separately by `response_schema=ClarificationResult`, so we don't
# describe JSON here — only how to judge.
CLARIFIER_SYSTEM = """\
You are the Clarifier in an automated software-architecture assistant. Your job
is NOT to design anything. Your only job is to read a user's request for a
system to be architected and judge what is understood versus what is still
missing before design can safely begin.

Split every gap into exactly one of two kinds:

1. ARCHITECTURE-CRITICAL — a fact that would change the design if it were
   different. These include (non-exhaustive): brownfield vs greenfield, existing
   stack / repo, expected scale and number of users, performance/availability
   targets, regulatory or compliance constraints (e.g. GDPR, HIPAA), cloud or
   on-prem preference, and hard budget limits. If such a fact is not clearly
   stated or safely derivable from the request, you MUST add it to
   `missing_critical` and add a matching entry to `questions`. NEVER assume an
   architecture-critical fact.

2. LOW-STAKES — a detail that is safely inferable or would not change the design.
   For these, do NOT ask. Instead record a short, explicit entry in
   `assumptions` so a human can later veto it. NEVER assume anything silently.

Also fill `captured_context` with the facts you DID ground in the request
(key -> value), e.g. "domain" -> "e-commerce", "cloud" -> "AWS".

Be conservative: it is better to ask one more critical question than to guess a
fact that reshapes the architecture. But do not pad `missing_critical` with
low-stakes items — that wastes the user's time.
"""


def _build_prompt(state: ArchitectState) -> str:
    """Assemble the user-turn content: the raw request plus any answers so far.

    On the first pass there are no answers. On a resume pass, the previously
    asked questions and the user's answers are included so the LLM re-judges
    WITH the new information (that is what lets `missing_critical` shrink).
    """
    parts = [f"USER REQUEST:\n{state.initial_request.raw_prompt}"]
    if state.clarification_answers:
        qa = "\n".join(f"Q: {q}\nA: {a}" for q, a in state.clarification_answers.items())
        parts.append(f"CLARIFYING Q&A SO FAR:\n{qa}")
    return "\n\n".join(parts)


def _freeze_context_record(result: ClarificationResult) -> ContextRecord:
    """Distil the completed clarification into the frozen Context Record.

    The Clarifier is the WRITER of the ContextRecord (its schema is Maheen's).
    Until that schema is frozen we populate the current placeholder `summary`
    field. Swapping to the real schema later touches ONLY this function, because
    everyone imports ContextRecord from state.py.
    """
    lines = [f"{c.key}: {c.value}" for c in result.captured_context]
    if result.assumptions:
        lines.append("Assumptions (human may veto): " + "; ".join(result.assumptions))
    summary = "\n".join(lines) if lines else "(no context captured)"
    # TODO(map to Maheen's ContextRecord schema once frozen)
    return ContextRecord(summary=summary)


@node("clarifier")
def clarifier_node(state: ArchitectState) -> dict:
    # 1. LLM JUDGES — returns a validated ClarificationResult (no manual parsing).
    result: ClarificationResult = llm_call(
        state,
        _build_prompt(state),
        system=CLARIFIER_SYSTEM,
        model=CLARIFIER_MODEL,
        response_schema=ClarificationResult,
    )

    # 2. CODE ROUTES — the deterministic gate.
    if result.missing_critical:
        # Something architecture-critical is still unknown → pause and ask.
        # TODO(W3): retry cap via state.bump_retry to clamp re-ask rounds.
        questions = [q.question for q in result.questions]
        step = make_step(
            "clarifier",
            state.stage,
            Stage.AWAITING_INPUT,
            f"missing {len(result.missing_critical)} critical fact(s); asked {len(questions)}",
        )
        return {
            "clarifying_questions": questions,
            "stage": Stage.AWAITING_INPUT,
            "history": [step],
        }

    # Enough is known → lock the Context Record and advance to the Researcher.
    context_record = _freeze_context_record(result)
    step = make_step(
        "clarifier",
        state.stage,
        Stage.CLARIFYING,
        f"context locked; {len(result.assumptions)} assumption(s) recorded",
    )
    return {
        "context_record": context_record,
        "clarifying_questions": [],  # clear any stale questions from a prior pause
        "stage": Stage.CLARIFYING,
        "history": [step],
    }


# ── Live smoke test: `python -m pipeline.agents.clarifier` ────────────────
# Makes ONE real LLM call (needs GEMINI_API_KEY in .env). Not a unit test — it
# checks the real behaviour the mocks can't: does `response_schema` come back as
# a valid ClarificationResult, and does the model flag scale/compliance as
# `missing_critical`? This is where you tune CLARIFIER_SYSTEM. For deterministic,
# no-network tests of the gate/routing, run test_clarifier.py instead.
if __name__ == "__main__":
    from pipeline.state import new_run

    s = new_run("We want a webshop to sell sneakers online. Around 50k users at peak sale days.")
    out = clarifier_node(s)  # side effect: token counts land in `s`
    print(f"stage           : {out['stage'].value}")
    print(f"questions       : {out.get('clarifying_questions')}")
    if out.get("context_record") is not None:
        print(f"context_record  :\n{out['context_record'].summary}")
    print(f"tokens in/out   : {s.input_tokens}/{s.output_tokens}")
    if out["stage"] is Stage.FAILED:
        print("→ FAILED — check .env / model / prompt (see errors):", out.get("errors"))

