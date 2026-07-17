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
CLARIFIER_MODEL = "flash-lite"

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

Also fill the `captured` object with everything you DID ground in the request:
`business_goal`, `problem_statement`, `users`, `functional_requirements` (the
capabilities AS THE USER STATES THEM — do not invent formal features),
`non_functional_requirements` (scale, availability, performance), `cloud_provider`,
`budget`, `compliance_requirements`, and `existing_systems`. Leave a field empty
if the request does not state it — an empty field that is architecture-critical
belongs in `missing_critical`, not guessed.

When a REPO ANALYSIS block is present, treat it as ground truth about the
EXISTING system (this is a brownfield run). Any architecture-critical fact it
already settles — the current stack, frameworks, external services, the overall
structure — is KNOWN: record it in `captured` (e.g. `existing_systems`) and do
NOT put it in `missing_critical` or ask about it. Only ask about critical facts the analysis
does not settle (e.g. target scale, budget, compliance, cloud preference).

Be conservative: it is better to ask one more critical question than to guess a
fact that reshapes the architecture. But do not pad `missing_critical` with
low-stakes items — that wastes the user's time.
"""


def _format_repo_context(state: ArchitectState) -> str:
    """Condensed repo analysis for the Clarifier, or "" on a greenfield run.

    Malte's repo_ingestor runs BEFORE the clarifier (see orchestrator table), so
    on a brownfield run `repo_representation` is already populated. Feeding a
    compact digest of it here is what lets the clarifier ground its questions in
    the actual codebase and skip asking about facts the repo already shows.
    """
    rep = state.repo_representation
    if rep is None:
        return ""
    ts = rep.structure.tech_stack
    lines = [f"REPO ANALYSIS (existing system at {rep.meta.url or 'unknown'}):"]
    if rep.behavior.overview:
        lines.append(f"Overview: {rep.behavior.overview}")
    if ts.languages or ts.frameworks or ts.external_services:
        lines.append(
            f"Tech: languages={list(ts.languages)}, frameworks={ts.frameworks}, "
            f"external services={ts.external_services or 'none detected'}"
        )
    if rep.behavior.partitions:
        lines.append(
            "Partitions: " + "; ".join(f"{p.name} ({p.role})" for p in rep.behavior.partitions)
        )
    return "\n".join(lines)


def _build_prompt(state: ArchitectState) -> str:
    """Assemble the user-turn content: the raw request, repo analysis, answers.

    On the first pass there are no answers. On a resume pass, the previously
    asked questions and the user's answers are included so the LLM re-judges
    WITH the new information (that is what lets `missing_critical` shrink). The
    repo digest (brownfield only) is injected so questions are repo-grounded.
    """
    parts = [f"USER REQUEST:\n{state.initial_request.raw_prompt}"]
    repo = _format_repo_context(state)
    if repo:
        parts.append(repo)
    if state.clarification_answers:
        qa = "\n".join(f"Q: {q}\nA: {a}" for q, a in state.clarification_answers.items())
        parts.append(f"CLARIFYING Q&A SO FAR:\n{qa}")
    return "\n\n".join(parts)


def _freeze_context_record(result: ClarificationResult) -> ContextRecord:
    """Distil the completed clarification into Maheen's frozen ContextRecord.

    The Clarifier is the WRITER of the ContextRecord (schema owned by Maheen).
    The extracted `captured` fields map ~1:1 onto it; the control signals fold in
    too: labelled `assumptions` carry across, and any remaining (non-critical)
    `questions` become `open_questions` the Architect should keep in mind.
    `summary` is kept as a short human-readable digest for backward compatibility.
    """
    c = result.captured
    bits = [
        f"Goal: {c.business_goal}" if c.business_goal else "",
        f"Problem: {c.problem_statement}" if c.problem_statement else "",
        f"Cloud: {c.cloud_provider}" if c.cloud_provider else "",
        f"Budget: {c.budget}" if c.budget else "",
    ]
    summary = " | ".join(b for b in bits if b) or "(clarified context)"
    return ContextRecord(
        project_name=c.project_name,
        business_goal=c.business_goal,
        problem_statement=c.problem_statement,
        users=c.users,
        functional_requirements=c.functional_requirements,
        non_functional_requirements=c.non_functional_requirements,
        cloud_provider=c.cloud_provider,
        budget=c.budget,
        compliance_requirements=c.compliance_requirements,
        existing_systems=c.existing_systems,
        assumptions=result.assumptions,
        open_questions=[q.question for q in result.questions],
        summary=summary,
    )


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

