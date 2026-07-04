"""state.py — THE state object contract (Kati / Orchestration).

WHAT THIS IS
------------
The state object is the single structure that travels through the whole pipeline.
Every agent *reads* some fields and *writes* others; nobody talks to anyone
directly — they only read from and write to this object. Freezing its *shape*
is the "contract" that lets all five of us build in parallel without breaking
each other.

HOW TO READ THIS FILE
---------------------
The model is split into four layers, top to bottom:
  1. INPUT           — what the user gives us (owned by Kati).
  2. ARTIFACTS       — the things the agents produce. These are PLACEHOLDERS for
                       contracts other people own (Maheen's schemas, Kush's KB
                       chunk, Malte's repo representation). Marked  # TODO(owner).
  3. WORKING FIELDS  — intermediate results agents pass to each other.
  4. CONTROL / META  — orchestration bookkeeping (owned entirely by Kati):
                       status, routing, trace history, retries, token usage.

This is v0.1 — a *strawman* for the team's contract-freezing session. The
placeholder classes exist so the skeleton runs today; each owner replaces the
body of their class with the real schema once frozen. Because everyone imports
from THIS file, tightening a placeholder later does not change the pipeline wiring.
"""
from __future__ import annotations

import operator
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════
# 1. INPUT LAYER  (owned by Kati)
# ══════════════════════════════════════════════════════════════════════════
class InitialRequest(BaseModel):
    """The raw request, in natural language, exactly as accepted from the user.

    ONLY user-supplied input lives here. Anything interpreted from the prompt
    (use case, constraints, repo reference) is DERIVED downstream by the
    Clarifier and written into the ContextRecord — never back into here. This
    keeps `raw_prompt` an immutable ground truth and keeps all interpretation
    (LLM work) out of the input layer.
    """

    raw_prompt: str = Field(..., description="Verbatim user message, exactly as entered. Ground truth for the run — never edited or overwritten.")


# ══════════════════════════════════════════════════════════════════════════
# 2. ARTIFACT PLACEHOLDERS  (owned by others — TODO markers, not final schemas)
# ══════════════════════════════════════════════════════════════════════════
# Each class below is a stand-in. The owner replaces the body with the real, validated schema.

class RepoRepresentation(BaseModel):
    """What the agent actually receives when it 'reads' the repo."""
    # TODO(Malte): real repo representation (README, file tree, selected files).
    summary: str = ""


class KBChunk(BaseModel):
    """One knowledge-base entry returned by a retrieval call."""
    # TODO(Kush): real KB chunk format (content + source + metadata).
    content: str = ""
    source: str = ""


class ContextRecord(BaseModel):
    """Frozen snapshot of all constraints + clarification answers before design."""
    # TODO(Maheen): real Context Record schema.
    summary: str = ""


class Feature(BaseModel):
    """One functional requirement, with a concrete scenario (p.9 feature-first)."""
    # TODO(Maheen): confirm final schema.
    id: str = ""          # stable handle, e.g. "F1" — components trace back to this
    name: str = ""        # short label of what the system must do
    scenario: str = ""    # concrete expected-behavior scenario (testable)


class Blueprint(BaseModel):
    """Architecture Blueprint — stakeholder view + technical view."""
    # TODO(Maheen): real Blueprint schema (two views).
    stakeholder_view: str = ""
    technical_view: str = ""


class ADR(BaseModel):
    """One Architecture Decision Record."""
    # TODO(Maheen): real ADR schema (title, context, options, decision, trade-offs).
    title: str = ""
    decision: str = ""


class ComponentDescription(BaseModel):
    """One component's justified description."""
    # TODO(Maheen): real Component Description schema.
    name: str = ""
    description: str = ""


class ReviewResult(BaseModel):
    """Reviewer's verdict on the current design."""
    # TODO(Waqar): align with docs/prompt_quality/06_reviewer_report_schema.json
    passed: bool = False
    issues: list[str] = Field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════
# 3. CONTROL / META  (owned by Kati)
# ══════════════════════════════════════════════════════════════════════════
class Stage(str, Enum):
    """Where the run currently is. The orchestrator routes on this value."""

    CREATED = "created"
    CLARIFYING = "clarifying"
    AWAITING_INPUT = "awaiting_input"  # clarifier needs the human; graph pauses here
    RESEARCHING = "researching"
    DESIGNING = "designing"
    REVIEWING = "reviewing"
    REFINING = "refining"
    DONE = "done"
    FAILED = "failed"


class StepLog(BaseModel):
    """One entry in the run trace: which agent ran, when, and what it did."""

    agent: str
    stage_in: Stage
    stage_out: Stage
    note: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ── Clarifier output (Kati owns) ─────────────────────────────────────────
# The lean, machine-readable object the Clarifier LLM produces and the gate
# reads. It is NOT the final ContextRecord (that is Maheen's frozen schema,
# written only once clarification is complete). Keeping them separate is the
# "LLM judges / code routes" split: the LLM fills this in, deterministic code
# decides what to do with `missing_critical`.
class ClarifyingQuestion(BaseModel):
    """One question to put to the user when something critical is missing."""

    question: str = Field(..., description="The question shown to the user.")
    why_needed: str = Field("", description="Why this changes the design — for the user and for traceability.")


class ContextField(BaseModel):
    """One grounded fact, as an explicit key/value pair.

    We use a LIST of these instead of a dict[str, str] because Gemini's
    Developer-API structured-output mode rejects open-ended maps (a dict becomes
    JSON-Schema `additionalProperties`, unsupported there). A list of fixed-key
    objects is a closed schema and is also more self-documenting.
    """

    key: str = Field(..., description="Name of the fact, e.g. 'cloud'.")
    value: str = Field(..., description="Its value, e.g. 'AWS'.")


class ClarificationResult(BaseModel):
    """The Clarifier's structured judgment of the raw prompt (+ any answers).

    `missing_critical` is THE signal the gate routes on: non-empty means an
    architecture-critical fact is still unknown, so the run must pause and ask.
    Empty means we can lock the ContextRecord and advance.
    """

    captured_context: list[ContextField] = Field(default_factory=list, description="Facts grounded in the prompt.")
    assumptions: list[str] = Field(default_factory=list, description="Low-stakes fills, labelled so a human can veto them.")
    questions: list[ClarifyingQuestion] = Field(default_factory=list, description="Questions to ask when critical info is missing.")
    missing_critical: list[str] = Field(default_factory=list, description="Architecture-critical gaps. Non-empty ⇒ pause and ask.")


# ══════════════════════════════════════════════════════════════════════════
# THE STATE OBJECT
# ══════════════════════════════════════════════════════════════════════════
class ArchitectState(BaseModel):
    """The single object passed between every agent. This IS the contract."""

    # --- 1. Input ---------------------------------------------------------
    initial_request: InitialRequest

    # --- 2. Artifacts produced along the way (start empty) ----------------
    repo_representation: Optional[RepoRepresentation] = None
    context_record: Optional[ContextRecord] = None
    blueprint: Optional[Blueprint] = None
    adrs: list[ADR] = Field(default_factory=list)
    components: list[ComponentDescription] = Field(default_factory=list)
    review: Optional[ReviewResult] = None

    # --- 3. Working fields (intermediate results) -------------------------
    clarifying_questions: list[str] = Field(default_factory=list)
    clarification_answers: dict[str, str] = Field(default_factory=dict)
    retrieved_knowledge: list[KBChunk] = Field(default_factory=list)
    features: list[Feature] = Field(default_factory=list)

    # --- 4. Control / orchestration meta (Kati owns fully) ----------------
    stage: Stage = Stage.CREATED
    # Annotated with operator.add so LangGraph MERGES (appends) each node's
    # returned step onto the running trace instead of overwriting it. This is
    # the "reducer" — nodes return {"history": [one_step]} and it accumulates.
    history: Annotated[list[StepLog], operator.add] = Field(default_factory=list)
    retry_counts: dict[str, int] = Field(default_factory=dict)
    refine_iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # Reducer (append) so a failure in any node adds to — never clobbers —
    # errors already recorded upstream. See `history` above.
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)

    # --- helpers ----------------------------------------------------------
    def log_step(self, agent: str, stage_out: Stage, note: str = "") -> None:
        """Record a transition and advance the stage. Called by every agent."""
        self.history.append(
            StepLog(agent=agent, stage_in=self.stage, stage_out=stage_out, note=note)
        )
        self.stage = stage_out

    def bump_retry(self, agent: str) -> int:
        """Increment and return this agent's retry counter (for retry caps later)."""
        self.retry_counts[agent] = self.retry_counts.get(agent, 0) + 1
        return self.retry_counts[agent]

def new_run(raw_prompt: str) -> ArchitectState:
    """Factory: build a fresh state at the start of a run.

    `raw_prompt` (the verbatim user message = ground truth) is the only input.
    Everything else is derived downstream by the agents.
    """
    return ArchitectState(initial_request=InitialRequest(raw_prompt=raw_prompt))

# ── quick self-test: `python -m pipeline.state` ──────────────────────────
if __name__ == "__main__":
    s = new_run(
        raw_prompt="We have a monolithic e-commerce app that keeps crashing on sale "
                   "days. It's on AWS, budget is medium, must stay GDPR-compliant, "
                   "and needs to handle about 50k users at peak. Repo: "
                   "https://github.com/example/bugged-shop",
    )
    s.log_step("clarifier", Stage.CLARIFYING, "asked 2 clarifying questions")
    s.log_step("researcher", Stage.RESEARCHING, "retrieved 3 KB chunks")
    s.log_step("architect", Stage.DESIGNING, "derived features, drafted blueprint")
    s.log_step("reviewer", Stage.DONE, "design passed review")

    print(s.model_dump_json(indent=2))
    print("\nTrace:")
    for step in s.history:
        print(f"  {step.agent:11s} {step.stage_in.value:11s} -> {step.stage_out.value}")

    # round-trip check: state survives a save→load cycle (proves state-on-disk works)
    assert ArchitectState.model_validate_json(s.model_dump_json()) == s
    print("\n✓ JSON round-trip OK")