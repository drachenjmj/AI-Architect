"""clarifier.py — Clarifier agent (Kati). The first REAL (non-stub) node.

WHAT IT DOES
------------
The Clarifier is the ONLY LLM in the clarification phase. It reads the raw
prompt (plus any answers gathered on a previous turn), and produces a
structured judgment — a `ClarificationResult` — of what it understood and what
is still missing. Then DETERMINISTIC CODE (the "gate") decides what happens:

  * `missing_critical` non-empty  → something architecture-critical is unknown →
    surface the questions and PAUSE the run (stage AWAITING_HUMAN,
    pending_decision CLARIFICATION). The orchestrator routes that to END; the
    caller collects answers, writes them into state, and re-runs — we land back
    here and re-judge.
  * `missing_critical` empty      → we know enough → LOCK a Context Record.

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

THE CONTEXT-LOCK GATE (where the veto in that sentence finally happens)
-----------------------------------------------------------------------
"so a human can later veto them" was, until now, a promise with nothing behind
it: the record froze and the run spent research, design and review tokens on
ground truth the human had never seen. Locking therefore no longer advances
straight to CLARIFYING. When `state.require_context_approval` is set, the lock
pauses with `pending_decision = CONTEXT_LOCK`, and the CALLER resolves it three
ways before the graph is entered again:

    Accept  -> `accept_context_lock`  — clears the pause, entry route researches
    Edit    -> `submit_context_edits` — pure, deterministic; see below
    Ask     -> `ask_advisor`          — read-only side channel, never enters the
                                        graph and never resolves the pause

Approval is OPT-IN (default False) because a mandatory pause would hang the CLI,
the eval harness and every test. The UI, which has a human in front of it, sets
it True. See `ArchitectState.require_context_approval`.

WHO WRITES THE CONTEXT RECORD (unchanged: this module, and only this module)
----------------------------------------------------------------------------
The gate introduced a second writer — the human — so it had to introduce a
second WRITE PATH, not a second writer. Everything a UI or CLI can do to the
record goes through `apply_user_edits` here; nothing outside this file ever
assigns to a `ContextRecord` field. That is what keeps "the clarifier owns
Maheen's schema" true now that the record is editable.

WHEN THE MODEL RUNS AGAIN, AND WHEN IT MUST NOT
------------------------------------------------
An edit only needs re-judging if it OPENS A GAP. Filling or replacing a value
cannot open one — the record is strictly better informed than the one already
judged — so those edits re-freeze with NO model call at all. Emptying an
architecture-critical field can, so those re-run the clarifier. Code decides
(`emptied_critical_fields`); the model only judges when there is something left
to judge. The alternative, re-judging on every keystroke-sized edit, would have
made the veto the most expensive button in the app.

ASK ONCE, THEN ASSUME
---------------------
After the first lock the clarifier LOSES the power to pause with questions
(`_can_ask`). On a re-judge, a critical gap becomes a labelled assumption plus
an `open_questions` entry instead of a new round of questions. This is not a
weakening of the invariant: the invariant was never "always ask", it was "never
assume SILENTLY", and a gap written into a record that a human is at that
moment looking at is the opposite of silent. Enforced in code, because a rule
that lives only in a prompt is a request, not an invariant.

It also contains a known clarifier bug for free: the model can report
`missing_critical` while returning NO `questions`, which used to park the run at
a pause with nothing to answer (`run.py` still refuses to spin on it). There is
no question to ask, so the gap is absorbed the same way and the run continues.
"""
from __future__ import annotations

from typing import Sequence

from pipeline.agents.base import make_step, node
from pipeline.llm import LLMUsage, attach_usage, llm_call
from pipeline.state import (
    AdvisoryTurn,
    ArchitectState,
    ClarificationResult,
    ContextEdits,
    ContextRecord,
    PendingDecision,
    Stage,
    StepLog,
)

# The clarifier's judgment sets the quality of everything downstream, so it uses
# the stronger model rather than the cheap default.
CLARIFIER_MODEL = "flash-lite"
# How many times the clarifier may PAUSE TO ASK before it must lock a record.
#
# Run 20260819T083025Z-00c6557a is why this exists: 14 pause rounds before
# design started, each answer surfacing two fresh "critical" gaps. The model
# will always find another question; nothing in its instruction tells it when
# the marginal answer stops being worth a round trip.
#
# 3 rather than a larger number because ASSUMING IS SAFE HERE, and that is the
# whole argument. Past the cap the remaining gaps are not dropped - they are
# absorbed as LABELLED assumptions on the Context Record, which the human sees
# and can veto at the context gate before a single research or design token is
# spent (see `require_context_approval` and the gate in run.py / the UI). So the
# cost of assuming too early is one veto, while the cost of asking too long is
# unbounded rounds and a person who stops reading. Three rounds is enough to
# converge on the facts a design genuinely cannot proceed without.
#
# Retune from a transcript where three rounds demonstrably was not enough, not
# from taste - and if you raise it, raise run.MAX_CLARIFICATION_ROUNDS with it,
# which is derived from this constant precisely so it cannot silently invert.
MAX_ASK_ROUNDS = 3

# ── ContextRecord field policy (deterministic, code-owned) ───────────────
# WHICH fields are architecture-critical. This list mirrors the enumeration in
# `_CLARIFIER_BASE` below, and the two ARE capable of drifting apart — one is
# prose for a model, one is data for a router. They are kept adjacent in this
# file for exactly that reason: change one, read the other.
CRITICAL_RECORD_FIELDS: tuple[str, ...] = (
    "business_goal",
    "problem_statement",
    "non_functional_requirements",  # scale, availability, performance targets
    "cloud_provider",
    "budget",
    "compliance_requirements",
    "existing_systems",             # brownfield vs greenfield
)

# Fields a human may set directly at the gate. DERIVED from the schema rather
# than listed, so adding a field to `ContextRecord` makes it editable without a
# second edit here. The three exclusions each have their own edit channel:
# assumptions are struck, open questions are answered, and `summary` is a
# rendering of the other fields that code recomputes after every edit.
_DERIVED_FIELDS = frozenset({"assumptions", "open_questions", "summary"})
EDITABLE_RECORD_FIELDS: tuple[str, ...] = tuple(
    name for name in ContextRecord.model_fields if name not in _DERIVED_FIELDS
)

# Prefix stamped on an assumption the clarifier made INSTEAD OF ASKING — a gap
# it was not allowed to raise as a question (a re-judge, or a "you recommend").
# Applied by CODE, not requested in the prompt: attribution that the model has
# to remember to write is attribution that will eventually go missing, and this
# label is what the approval panel groups its strike list by.
CLARIFIER_LABEL = "[assumed by clarifier]"


# ── System prompts: one policy, two modes ────────────────────────────────
# The POLICY (assume-vs-ask, what to capture, how to read a repo digest) is
# shared. Only the closing rule differs, because only the closing rule changes
# between "you may ask" and "you may not". The OUTPUT SHAPE is enforced
# separately by `response_schema=ClarificationResult`, so no JSON is described
# here — only how to judge.
_CLARIFIER_BASE = """\
You are the Clarifier in an automated software-architecture assistant. Your job
is NOT to design anything. Your only job is to read a user's request for a
system to be architected and judge what is understood versus what is still
missing before design can safely begin.

Split every gap into exactly one of two kinds:

1. ARCHITECTURE-CRITICAL — a fact that would change the design if it were
   different. These include (non-exhaustive): brownfield vs greenfield, existing
   stack / repo, expected scale and number of users, performance/availability
   targets, regulatory or compliance constraints (e.g. GDPR, HIPAA), cloud or
   on-prem preference, and hard budget limits.

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
"""

_ASK_MODE = """
This is the FIRST pass: you may still ask. If an architecture-critical fact is
not clearly stated or safely derivable from the request, you MUST add it to
`missing_critical` AND add a matching entry to `questions`. A `missing_critical`
entry with no question is useless to the caller — it names a gap nobody can
close. NEVER assume an architecture-critical fact on this pass.

Be conservative: it is better to ask one more critical question than to guess a
fact that reshapes the architecture. But do not pad `missing_critical` with
low-stakes items — that wastes the user's time.
"""

_ASSUME_MODE = """
This is a RE-JUDGE, and on a re-judge you may NOT ask. A human has already seen
this Context Record and is looking at it right now; `questions` will be ignored.

So for every architecture-critical fact that is still unknown — including any
field the human explicitly asked you to recommend — do all three of these:

  * put your BEST PROFESSIONAL DEFAULT into the matching `captured` field. A
    plausible, conventional value for this kind of system, not a hedge and not
    an empty string.
  * add ONE line to `assumptions` naming the field, the value you chose, and a
    ONE-SENTENCE rationale. Example: "cloud_provider: AWS — the request names no
    provider and AWS is the default for a team with no existing cloud estate."
  * still list the fact in `missing_critical`, so the caller can show the human
    exactly which values are proposals rather than things they told us.

Treat the CONTEXT RECORD block in the user turn as ground truth: it carries the
human's own edits. Do not re-fill a field they deliberately cleared with the old
value, do not contradict a value they set, and never re-propose an assumption
listed as rejected.
"""

CLARIFIER_SYSTEM = _CLARIFIER_BASE + _ASK_MODE
CLARIFIER_ASSUME_SYSTEM = _CLARIFIER_BASE + _ASSUME_MODE


# ══════════════════════════════════════════════════════════════════════════
# Prompt assembly
# ══════════════════════════════════════════════════════════════════════════
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


def _format_record(record: ContextRecord) -> str:
    """The record as the human currently has it — the ground truth of a re-judge.

    Without this the clarifier would re-derive everything from the raw prompt and
    silently undo the human's edits on the very turn their edit triggered, which
    would make the veto a no-op with extra steps.
    """
    lines = ["CONTEXT RECORD AS THE HUMAN HAS IT (their edits; ground truth):"]
    for name in EDITABLE_RECORD_FIELDS:
        value = getattr(record, name)
        rendered = "; ".join(str(v) for v in value) if isinstance(value, list) else str(value)
        lines.append(f"  {name}: {rendered.strip() or '(cleared / not stated)'}")
    if record.assumptions:
        lines.append("Assumptions currently standing:")
        lines.extend(f"  - {a}" for a in record.assumptions)
    if record.open_questions:
        lines.append("Open questions on it (a 'recommendation requested' line is an instruction to you):")
        lines.extend(f"  - {q}" for q in record.open_questions)
    return "\n".join(lines)


def _build_prompt(state: ArchitectState) -> str:
    """Assemble the user-turn content: the raw request, repo analysis, answers.

    On the first pass there are no answers. On a resume pass, the previously
    asked questions and the user's answers are included so the LLM re-judges
    WITH the new information (that is what lets `missing_critical` shrink). The
    repo digest (brownfield only) is injected so questions are repo-grounded.

    On a RE-JUDGE the current record and the human's rejected assumptions are
    injected too, so the model extends their record instead of replacing it.
    """
    parts = [f"USER REQUEST:\n{state.initial_request.raw_prompt}"]
    repo = _format_repo_context(state)
    if repo:
        parts.append(repo)
    if state.clarification_answers:
        qa = "\n".join(f"Q: {q}\nA: {a}" for q, a in state.clarification_answers.items())
        parts.append(f"CLARIFYING Q&A SO FAR:\n{qa}")
    if state.context_record is not None:
        parts.append(_format_record(state.context_record))
    if state.vetoed_assumptions:
        rejected = "\n".join(f"  - {a}" for a in state.vetoed_assumptions)
        parts.append(
            "ASSUMPTIONS THE HUMAN HAS REJECTED (never propose these again):\n" + rejected
        )
    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════
# Freezing the record (the module's single write path from the model)
# ══════════════════════════════════════════════════════════════════════════
def _summary_line(record_or_captured) -> str:
    """The one-line human-readable digest, recomputed after every write.

    Shared by the freeze and the edit path so an edited record's summary can
    never disagree with its own fields — which it would if only the freeze
    computed it.
    """
    bits = [
        f"Goal: {record_or_captured.business_goal}" if record_or_captured.business_goal else "",
        f"Problem: {record_or_captured.problem_statement}" if record_or_captured.problem_statement else "",
        f"Cloud: {record_or_captured.cloud_provider}" if record_or_captured.cloud_provider else "",
        f"Budget: {record_or_captured.budget}" if record_or_captured.budget else "",
    ]
    return " | ".join(b for b in bits if b) or "(clarified context)"


def _freeze_context_record(
    result: ClarificationResult,
    *,
    absorbed_gaps: Sequence[str] = (),
    vetoed_assumptions: Sequence[str] = (),
) -> ContextRecord:
    """Distil the completed clarification into Maheen's frozen ContextRecord.

    The Clarifier is the WRITER of the ContextRecord (schema owned by Maheen).
    The extracted `captured` fields map ~1:1 onto it; the control signals fold in
    too: labelled `assumptions` carry across, and any remaining (non-critical)
    `questions` become `open_questions` the Architect should keep in mind.

    `absorbed_gaps` are the architecture-critical facts the clarifier was NOT
    allowed to ask about (see "ask once, then assume"). Each one is guaranteed —
    by code, not by the prompt — to leave BOTH a labelled assumption and an open
    question, so a value the human never confirmed can never reach the Architect
    looking like one they did. Every assumption is stamped with
    `CLARIFIER_LABEL` in this mode, since in this mode every one of them stands
    in for a question that was not asked.

    `vetoed_assumptions` are struck verbatim. A model re-proposing something the
    human just rejected is the ordinary case, not the exotic one, and the prompt
    asking it not to is not a guarantee.
    """
    c = result.captured
    vetoed = set(vetoed_assumptions)

    assumptions = [a for a in result.assumptions if a not in vetoed]
    open_questions = [q.question for q in result.questions]

    if absorbed_gaps:
        assumptions = [
            a if a.startswith(CLARIFIER_LABEL) else f"{CLARIFIER_LABEL} {a}"
            for a in assumptions
        ]
        for gap in absorbed_gaps:
            # Best-effort de-duplication: the prompt asks the model to name the
            # field in its own assumption line, so this usually finds it. When it
            # does not, the cost is one redundant line — never a missing
            # guarantee, which is the direction this has to fail in.
            if not any(gap.lower() in a.lower() for a in assumptions):
                filled = f"{CLARIFIER_LABEL} {gap}: filled without confirmation."
                if filled not in vetoed:
                    assumptions.append(filled)
            open_questions.append(
                f"{gap} — proposed by the clarifier, not confirmed by you."
            )

    record = ContextRecord(
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
        assumptions=assumptions,
        open_questions=open_questions,
        summary=_summary_line(c),
    )
    return record


# ══════════════════════════════════════════════════════════════════════════
# The human's write path — pure functions
# ══════════════════════════════════════════════════════════════════════════
def apply_user_edits(record: ContextRecord, edits: ContextEdits) -> ContextRecord:
    """Apply one human veto pass to `record`. Pure, deterministic, LLM-free.

    Returns a NEW record; `record` is never mutated, so a caller can diff the two
    (that is exactly what `emptied_critical_fields` does to decide whether the
    model has to run again).

    Raises `ValueError` on an unknown field name or a value of the wrong shape —
    a `str` for a list field or vice versa. Loudly, because the failure mode of
    accepting it quietly is a veto the human believes they cast and nobody
    applied. Callers turn this into a message, not a crash.

    Order matters and is fixed: fields are set, then "you recommend" clears the
    fields it names, then struck assumptions and answered questions are removed,
    then the summary is recomputed. Setting a field and asking for a
    recommendation on it in the same pass therefore resolves to the
    recommendation — the more specific request wins.
    """
    updated = record.model_copy(deep=True)

    for name, value in edits.fields.items():
        if name not in EDITABLE_RECORD_FIELDS:
            raise ValueError(
                f"'{name}' is not an editable Context Record field. "
                f"Editable: {', '.join(EDITABLE_RECORD_FIELDS)}."
            )
        wants_list = isinstance(getattr(updated, name), list)
        if wants_list and not isinstance(value, list):
            raise ValueError(f"'{name}' is a list field; pass a list of strings, not a string.")
        if not wants_list and isinstance(value, list):
            raise ValueError(f"'{name}' is a text field; pass a string, not a list.")
        if wants_list:
            setattr(updated, name, [str(v).strip() for v in value if str(v).strip()])
        else:
            setattr(updated, name, str(value).strip())

    for name in edits.recommend:
        if name not in EDITABLE_RECORD_FIELDS:
            raise ValueError(
                f"'{name}' is not a Context Record field, so there is nothing to "
                f"recommend for it. Editable: {', '.join(EDITABLE_RECORD_FIELDS)}."
            )
        # "I don't know — you recommend" IS an emptied field, expressed as an
        # instruction. Clearing it is what makes this the SAME mechanism as a
        # vetoed value rather than a second one: the gap opens, the clarifier
        # re-judges, and its proposal comes back labelled and vetoable.
        setattr(updated, name, [] if isinstance(getattr(updated, name), list) else "")
        request = f"[recommendation requested] {name}: propose a value and give a one-line rationale."
        if request not in updated.open_questions:
            updated.open_questions.append(request)

    struck = set(edits.struck_assumptions)
    updated.assumptions = [a for a in updated.assumptions if a not in struck]

    answered = set(edits.answered_questions)
    updated.open_questions = [q for q in updated.open_questions if q not in answered]

    updated.summary = _summary_line(updated)
    return updated


def emptied_critical_fields(before: ContextRecord, after: ContextRecord) -> list[str]:
    """Architecture-critical fields the edit CLEARED — the re-judge trigger.

    Pure. Truthiness IS the decision: a non-empty result means a gap was opened
    and the model has to judge it; an empty one means the edit filled or replaced
    a value, which cannot open a gap, so the record re-freezes with no model call
    at all. A field that was already empty before the edit is not listed — no NEW
    gap was opened, and the clarifier has already had its say about that one.
    """
    return [
        name
        for name in CRITICAL_RECORD_FIELDS
        if getattr(before, name) and not getattr(after, name)
    ]


# ══════════════════════════════════════════════════════════════════════════
# The human's write path — caller-side, OUTSIDE the graph
# ══════════════════════════════════════════════════════════════════════════
# ui.py and run.py call these on their own state object before deciding whether
# to re-enter the graph — the same place ui.py already writes
# `clarification_answers`. They live here, not there, because they write
# `context_record`, and this module is its only writer.
def accept_context_lock(state: ArchitectState) -> None:
    """Resolution 1 of 3: the human approves the record as it stands.

    Clears the pause and puts the run back on CLARIFYING, which the EXISTING
    entry route already sends to the researcher — no new route, no new stage.
    """
    if state.pending_decision is not PendingDecision.CONTEXT_LOCK:
        raise ValueError(
            f"accept_context_lock called with pending_decision="
            f"{state.pending_decision!r}; there is no context lock to accept."
        )
    state.pending_decision = None
    state.stage = Stage.CLARIFYING


def submit_context_edits(state: ArchitectState, edits: ContextEdits) -> list[str]:
    """Resolution 2 of 3: the human changes the record. Returns re-judge reasons.

    An EMPTY list means the record was updated and no model call is needed — the
    pause simply stays open with the edited record on screen, waiting for the
    Accept. A NON-EMPTY list names the gaps this edit opened; the caller must
    then charge a user round (`refine_gate.begin_user_round`) and, if that is
    affordable, call `open_for_rejudge` and re-enter the graph.

    Splitting it that way is deliberate: a caller that cannot afford the round
    keeps the human's edits and the open gate, and the human may still accept a
    record with a hole in it. That is their call to make — this gate exists to
    give them the call, not to take it.
    """
    if state.pending_decision is not PendingDecision.CONTEXT_LOCK:
        raise ValueError(
            f"submit_context_edits called with pending_decision="
            f"{state.pending_decision!r}. The record is only editable while it is "
            f"waiting for approval — after that it is the ground truth the design "
            f"was built on, and quietly rewriting it would leave artifacts that "
            f"trace back to a record that no longer exists."
        )
    record = state.context_record
    if record is None:
        raise ValueError("submit_context_edits called with no locked Context Record.")

    updated = apply_user_edits(record, edits)

    # The ONE assignment to `context_record` outside the node, and it is inside
    # this module — which is what keeps "the clarifier is the sole writer" true.
    state.context_record = updated
    for assumption in edits.struck_assumptions:
        if assumption not in state.vetoed_assumptions:
            state.vetoed_assumptions.append(assumption)
    # An answered open question is a resolved one, and resolved Q&A already has a
    # home: the run's Q&A, where any later re-judge will read it. Code cannot know
    # which record FIELD the answer belongs in without asking a model, and asking
    # one here would break the "filling a value costs nothing" guarantee — so it
    # does not guess. A human who wants the fact in the record edits the field.
    state.clarification_answers.update(
        {q: a.strip() for q, a in edits.answered_questions.items() if a.strip()}
    )

    # A "you recommend" on a field that HAD a value also reads as a clearing, and
    # it is one — but "you asked for a recommendation" already says that, and
    # more precisely. Report the specific reason, not both.
    asked_for = set(edits.recommend)
    reasons = [
        f"'{name}' was cleared"
        for name in emptied_critical_fields(record, updated)
        if name not in asked_for
    ]
    reasons += [f"'{name}': you asked for a recommendation" for name in edits.recommend]
    return reasons


def open_for_rejudge(state: ArchitectState) -> None:
    """Hand a gap-opening edit back to the graph.

    Clears `pending_decision` — the human HAS decided; what is outstanding now is
    the clarifier's judgment, not theirs — while leaving `stage` at
    AWAITING_HUMAN, which is what `_entry_route` sends back to this node. The
    orchestrator refuses to be entered while CONTEXT_LOCK is still pending, so
    this call is also what makes that re-entry legal.
    """
    state.pending_decision = None


# ── Resolution 3 of 3: the advisory side channel ─────────────────────────
ADVISOR_AGENT = "advisor"
ADVISOR_MODEL = "flash-lite"
ADVISOR_SYSTEM = """\
You are advising a person who is reviewing a frozen Context Record for a
software-architecture run, immediately before it is used for design. They have
paused to ask you something. Your job is to help them decide.

Answer the question in your first sentence, and give an actual recommendation
rather than a survey of considerations. The Context Record is your CONTEXT, not
your boundary: most questions worth asking are about things it does not state,
so "the record does not say" is never a reason to decline. Use general
architecture and engineering knowledge freely, and make clear which part of your
answer comes from the record and which from common practice.

Name the one condition that would change your recommendation — that is more
useful than hedging across every possibility. If a project-specific fact would
decide the answer and is unknown, say which fact and what each way would imply.
Never invent project, client, or repository facts. Where it helps, point to the
field or assumption they can edit to act on your answer.

Two or three short paragraphs at most, fewer if they ask for a short answer. No
headings, no bullet lists. Never describe your own role, scope or permissions,
and never open with a caveat about what the record does not contain.
"""


def ask_advisor(state: ArchitectState, question: str) -> str:
    """Answer one read-only question about the paused record. NEVER enters the graph.

    No routing decision, no stage change, `pending_decision` untouched, no
    artifact written: the pause stays exactly as open as it was, across any
    number of questions. Understanding the record is not a round trip through the
    pipeline, and making it one would have meant re-running the whole clarifier
    to answer "why does scale matter here?".

    It is read-only on ARTIFACTS but append-only on the ledger, and that is not
    optional. The call spends real tokens; if it did not land in `history` under
    its own agent name, `usage_by_agent()` would stop summing to
    `input_tokens`/`output_tokens` and the run's cost would become
    unreconcilable — a per-agent table nobody can trust is worse than no table.
    So exactly three things move: one `AdvisoryTurn`, one `StepLog`, and the two
    token totals.

    Raises `LLMError` if the call fails. The caller reports it and leaves the
    pause open — a failed side question must never cost the human their gate.
    """
    if state.context_record is None:
        raise ValueError("ask_advisor called with no locked Context Record to ask about.")

    parts = [
        f"ORIGINAL REQUEST:\n{state.initial_request.raw_prompt}",
        _format_record(state.context_record),
    ]
    if state.advisory_turns:
        # Follow-ups ("and what about the other one?") need the thread, and this
        # is the whole thread: the turns are on the state, not in a UI session.
        thread = "\n".join(f"Q: {t.question}\nA: {t.answer}" for t in state.advisory_turns)
        parts.append(f"EARLIER QUESTIONS IN THIS REVIEW:\n{thread}")
    parts.append(f"THEIR QUESTION:\n{question}")

    answer, usage = llm_call(
        state,
        "\n\n".join(parts),
        system=ADVISOR_SYSTEM,
        model=ADVISOR_MODEL,
    )
    answer = str(answer or "").strip()

    state.advisory_turns.append(AdvisoryTurn(question=question, answer=answer))
    state.history.append(
        StepLog(
            agent=ADVISOR_AGENT,
            # Same stage in and out: this turn moved nothing. A trace reader
            # should be able to see that at a glance.
            stage_in=state.stage,
            stage_out=state.stage,
            note=f"advisory question about the context record: {question}",
            model=usage.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=usage.cost_usd or 0.0,
        )
    )
    # Keep the run totals reconciled with the trace. `history` is the per-agent
    # ledger and these two are the run ledger; state.py's `usage_by_agent`
    # promises they sum to each other, and that promise has to survive a call
    # made outside the graph as much as one made inside it.
    state.input_tokens += usage.input_tokens
    state.output_tokens += usage.output_tokens
    return answer


# ══════════════════════════════════════════════════════════════════════════
# The node
# ══════════════════════════════════════════════════════════════════════════
def _may_ask(state: ArchitectState) -> bool:
    """Is this turn still allowed to pause with questions? Deterministic.

    THE single definition of "asking is still on the table". Both the prompt
    selection and the routing gate defer to it, so the cap cannot be written
    twice in opposite polarity and drift apart.

    Two independent reasons it may not, both of them code:

      * A record has already been locked once, so a human has already seen this
        context and is the one who sent it back. Asking again would be a new
        round of questions in front of a person already holding the answer sheet
        — hence "ask once, then assume".
      * The ask budget is spent. See MAX_ASK_ROUNDS for why running out is safe.
    """
    return state.context_record is None and state.ask_rounds < MAX_ASK_ROUNDS


def _can_ask(state: ArchitectState, result: ClarificationResult) -> bool:
    """May this turn pause, AND is there anything to pause with?

    `_may_ask` is permission; this adds the one thing that is about the reply
    rather than the run: the model reported gaps but produced no questions. That
    is the known clarifier bug, and there is literally nothing to put on screen;
    pausing would park the run somewhere no answer can reach it.
    """
    return _may_ask(state) and bool(result.questions)


@node("clarifier")
def clarifier_node(state: ArchitectState) -> dict:
    # `usage` is what this node's ONE call consumed. It is RETURNED (never
    # written into state) because LangGraph persists only what a node returns;
    # the try/except hands it to `@node` if anything below the call raises, so
    # already-billed tokens survive a failure. See pipeline/llm.py.
    usage: LLMUsage | None = None
    try:
        # A re-judge has a record; a first pass does not. That one fact selects
        # the mode, so the flag cannot drift out of sync with reality — there is
        # no second copy of it to forget to update.
        assume_only = not _may_ask(state)

        # 1. LLM JUDGES — returns a validated ClarificationResult (no manual parsing).
        result: ClarificationResult
        result, usage = llm_call(
            state,
            _build_prompt(state),
            system=CLARIFIER_ASSUME_SYSTEM if assume_only else CLARIFIER_SYSTEM,
            model=CLARIFIER_MODEL,
            response_schema=ClarificationResult,
        )

        # 2. CODE ROUTES — the deterministic gate.
        if result.missing_critical and _can_ask(state, result):
            # Something architecture-critical is still unknown → pause and ask.
            questions = [q.question for q in result.questions]
            step = make_step(
                "clarifier",
                state.stage,
                Stage.AWAITING_HUMAN,
                f"missing {len(result.missing_critical)} critical fact(s); asked {len(questions)}",
                usage,
            )
            return {
                "clarifying_questions": questions,
                # ABSOLUTE, not a delta: `ask_rounds` is a plain field, so
                # LangGraph overwrites it. Counted here and nowhere else, so
                # what costs a round is one line.
                "ask_rounds": state.ask_rounds + 1,
                "stage": Stage.AWAITING_HUMAN,
                "pending_decision": PendingDecision.CLARIFICATION,
                "history": [step],
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            }

        # Enough is known — or we are no longer allowed to ask, in which case the
        # remaining gaps become labelled assumptions the human can see and veto
        # rather than questions nobody will answer. Either way: LOCK.
        absorbed = list(result.missing_critical)
        context_record = _freeze_context_record(
            result,
            absorbed_gaps=absorbed,
            vetoed_assumptions=state.vetoed_assumptions,
        )

        # 3. THE GATE. Approval required → stop and let the human veto before a
        # single research/design/review token is spent on this ground truth.
        # Not required (CLI, eval, tests) → advance exactly as before.
        if state.require_context_approval:
            note = f"context locked for approval; {len(context_record.assumptions)} assumption(s)"
            if absorbed:
                note += f", {len(absorbed)} gap(s) assumed rather than asked"
            step = make_step("clarifier", state.stage, Stage.AWAITING_HUMAN, note, usage)
            return {
                "context_record": context_record,
                "clarifying_questions": [],  # clear any stale questions from a prior pause
                "stage": Stage.AWAITING_HUMAN,
                "pending_decision": PendingDecision.CONTEXT_LOCK,
                "history": [step],
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            }

        note = f"context locked; {len(context_record.assumptions)} assumption(s) recorded"
        if absorbed:
            note += f"; {len(absorbed)} gap(s) assumed rather than asked"
        step = make_step("clarifier", state.stage, Stage.CLARIFYING, note, usage)
        return {
            "context_record": context_record,
            "clarifying_questions": [],
            "stage": Stage.CLARIFYING,
            "pending_decision": None,
            "history": [step],
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        }
    except Exception as e:
        if usage is not None:
            attach_usage(e, usage)
        raise


# ── Live smoke test: `python -m pipeline.agents.clarifier` ────────────────
# Makes ONE real LLM call (needs GEMINI_API_KEY in .env). Not a unit test — it
# checks the real behaviour the mocks can't: does `response_schema` come back as
# a valid ClarificationResult, and does the model flag scale/compliance as
# `missing_critical`? This is where you tune CLARIFIER_SYSTEM. For deterministic,
# no-network tests of the gate/routing, run test_clarifier.py and
# test_context_gate.py instead.
if __name__ == "__main__":
    from pipeline.state import new_run

    s = new_run(
        "We want a webshop to sell sneakers online. Around 50k users at peak sale days.",
        require_context_approval=True,  # exercise the gate, not the auto-approve path
    )
    out = clarifier_node(s)  # pure: token counts come back in `out`, not in `s`
    print(f"stage           : {out['stage'].value}")
    print(f"pending         : {getattr(out.get('pending_decision'), 'value', None)}")
    print(f"questions       : {out.get('clarifying_questions')}")
    if out.get("context_record") is not None:
        print(f"context_record  :\n{out['context_record'].summary}")
        for a in out["context_record"].assumptions:
            print(f"  assumption    : {a}")
    # Read the RETURNED update, not `s` — the node never mutates the state.
    print(f"tokens in/out   : {out.get('input_tokens', 0)}/{out.get('output_tokens', 0)}")
    print(f"cost_usd        : {out['history'][0].cost_usd:.6f} (list-price equiv.; free-tier key)")
    if out["stage"] is Stage.FAILED:
        print("→ FAILED — check .env / model / prompt (see errors):", out.get("errors"))
