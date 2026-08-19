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
                       chunk, Malte's repo representation). Marked  # TODO(owner
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
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ══════════════════════════════════════════════════════════════════════════
# 1. INPUT LAYER  (owned by Kati)
# ══════════════════════════════════════════════════════════════════════════
class InitialRequest(BaseModel):
    """The raw request, in natural language, exactly as accepted from the user.

    ONLY user-supplied input lives here — both fields come straight from the
    user, nothing is interpreted. Everything DERIVED from them (use case,
    constraints, the repo analysis) is produced downstream and written
    elsewhere, never back into here, so this stays immutable ground truth.

    `repo_url` is an EXPLICIT field. Empty = a greenfield project with no
    existing code. The repo_ingestor still falls back to scanning the
    prompt/answers when this is empty, so pasting a URL into the prompt keeps
    working.
    """

    raw_prompt: str = Field(..., description="Verbatim user message, exactly as entered. Ground truth for the run — never edited or overwritten.")
    repo_url: str = Field("", description="Existing codebase to re-architect, from the UI's URL field. Empty = greenfield.")


# ══════════════════════════════════════════════════════════════════════════
# 2. ARTIFACT PLACEHOLDERS  (owned by others — TODO markers, not final schemas)
# ══════════════════════════════════════════════════════════════════════════
# Each class below is a stand-in. The owner replaces the body with the real, validated schema.

# ── Repo representation (Malte owns) ──────────────────────────────────────
# Repository Representation is intended to systematically provide information on the
# codebase in a "as token-cheap as *possible* and as extensive as *necessary*
# Two-layer contract:
#   structure — built DETERMINISTICALLY from a "git clone --depth 1" (shallow clone)
#               operation;
#               plain code in pipeline/repo_analysis.py produces it.
#   behavior  — the ONLY LLM-written layer (overview + partition summaries).
# LLM-facing render artifacts (file tree, repo map, mermaid) are stored as
# ready-to-inject TEXT; things code must post-process (dependency edges,
# tech stack) are stored STRUCTURED. Written ONCE by the repo_ingestor node;
# lazy (meaning only done if the user asks for in-depth information about certain
# artifacts) drill-downs go to `ArchitectState.repo_deep_dives` instead (reducer),
# so this artifact never needs rewriting.

class RepoMeta(BaseModel):
    """Provenance: what was cloned, at which commit, and where it lives locally."""

    url: str = ""
    commit_sha: str = Field("", description="Exact commit analysed — makes the analysis reproducible.")
    clone_path: str = Field("", description="Local shallow-clone dir; the Architect reads files from here on drill-downs.")
    ingested_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TechStack(BaseModel):
    """What the system is built with — parsed from manifests, not guessed from code."""

    languages: dict[str, int] = Field(default_factory=dict, description="Language -> lines of code.")
    frameworks: list[str] = Field(default_factory=list, description="Recognised frameworks, e.g. ['FastAPI', 'React'].")
    dependencies: list[str] = Field(default_factory=list, description="Direct dependencies from manifest files.")
    external_services: list[str] = Field(default_factory=list, description="DBs/queues/etc. found in docker-compose/Dockerfile.")


class DependencyEdge(BaseModel):
    """One intra-repo import: file `source` imports from file `target`."""

    source: str
    target: str


class RepoStructure(BaseModel):
    """The static view. Everything here is derived by plain code — never by an LLM."""

    file_tree: str = Field("", description="Rendered directory tree with LOC annotations (LLM-facing text).")
    repo_map: str = Field("", description="aider-style map: most-imported files first, signatures only.")
    dependency_edges: list[DependencyEdge] = Field(default_factory=list)
    architecture_diagram: str = Field("", description="Mermaid source, rendered FROM dependency_edges (coarse, top-level view).")
    tech_stack: TechStack = Field(default_factory=TechStack)
    integration_interface: str = Field("", description="Condensed OpenAPI/Swagger surface. Empty = none found (drill down on demand).")


class PartitionSummary(BaseModel):
    """One partition of the repo, along whatever structure the repo itself has."""

    name: str = Field("", description="Partition label, e.g. 'shop/api' or 'Frontend'.")
    paths: list[str] = Field(default_factory=list, description="Repo-relative dirs/files covered — the drill-down anchor.")
    role: str = Field("", description="Role within the whole system (1-2 sentences).")
    functionality: str = Field("", description="What it functionally does, in plain language.")


class RepoBehavior(BaseModel):
    """The functional view — the only LLM-written layer (used as response_schema)."""

    overview: str = Field("", description="Short summary of what the whole repo does.")
    partitions: list[PartitionSummary] = Field(default_factory=list, description="At most ~5, along the repo's own structure.")


class RepoRepresentation(BaseModel):
    """What the agent actually receives when it 'reads' the repo. Write-once (repo_ingestor)."""

    meta: RepoMeta = Field(default_factory=RepoMeta)
    structure: RepoStructure = Field(default_factory=RepoStructure)
    behavior: RepoBehavior = Field(default_factory=RepoBehavior)


class DeepDive(BaseModel):
    """One cached drill-down: a deeper look at a file/dir, written by the Architect.

    Lives OUTSIDE RepoRepresentation (as `ArchitectState.repo_deep_dives`, an
    append-only reducer field) so drilling never rewrites the write-once artifact.
    """

    target: str = Field(..., description="Repo-relative file or directory path — the clear reference.")
    question: str = Field("", description="What triggered the drill-down.")
    insight: str = Field("", description="The derived insight, cached so it is never re-derived.")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class KBChunk(BaseModel):
    """One knowledge-base entry returned by a retrieval call."""
    content: str = ""
    source: str = ""
    page: int = 0
    box: int = 1          # 1=patterns, 2=domain, 3=web fallback
    distance: float | None = None   # Chroma distance, lower = better; None for web results


class ContextRecord(BaseModel):
    """Frozen snapshot of all constraints + clarification answers before design."""
    # TODO(Maheen): real Context Record schema.
    
    project_name: str = Field(
        default="",
        description="Short name identifying the architecture project.",
    )
    business_goal: str = Field(
        default="",
        description="Business outcome the proposed architecture must support.",
    )
    problem_statement: str = Field(
        default="",
        description="Current problem, limitation, or architectural flaw to address.",
    )
    users: list[str] = Field(
        default_factory=list,
        description="Users, stakeholders, or user groups affected by the system.",
    )
    functional_requirements: list[str] = Field(
        default_factory=list,
        description="Capabilities and behaviours the system must provide.",
    )
    non_functional_requirements: list[str] = Field(
        default_factory=list,
        description="Quality requirements such as scalability, availability, and performance.",
    )
    cloud_provider: str = Field(
        default="",
        description="Required or preferred cloud provider, if specified.",
    )
    budget: str = Field(
        default="",
        description="Available budget level or relevant cost constraint.",
    )
    compliance_requirements: list[str] = Field(
        default_factory=list,
        description="Legal, regulatory, security, or policy requirements.",
    )
    existing_systems: list[str] = Field(
        default_factory=list,
        description="Existing applications, services, data stores, or integrations.",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Explicit assumptions accepted during clarification.",
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Unresolved questions that may affect later architecture decisions.",
    )
    summary: str = Field(
        default="",
        description="Human-readable summary retained for backward compatibility.",
    )
    # A record is FROZEN PER VERSION, and that is the point: it is what lets the
    # audit trail say "the v1 design was correct given what we knew at v1". A
    # user correction after design does not edit v1 — it supersedes it with a v2,
    # and v1 is pushed onto `ArchitectState.context_history` intact.
    #
    # Defaults to 1 (not 0) so a checkpoint written before this field existed
    # loads as the first version of its record, which is exactly what it is.
    version: int = Field(
        default=1,
        description="1-based version of this record; a correction supersedes it with the next.",
    )
    revision_reason: str = Field(
        default="",
        description="Why this version exists — the user's own words. Empty on v1.",
    )


class Feature(BaseModel):
    """One functional requirement, with a concrete scenario (p.9 feature-first)."""
    # TODO(Maheen): confirm final schema.
    id: str = Field(
        ...,
        description="Stable feature identifier, for example FEAT-001.",
    )
    name: str = Field(
        ...,
        description="Short name describing the required system capability.",
    )
    description: str = Field(
        default="",
        description="Detailed explanation of what the feature must achieve.",
    )
    scenario: str = Field(
        ...,
        description="Concrete and testable expected-behaviour scenario.",
    )
    related_requirement_ids: list[str] = Field(
        default_factory=list,
        description="Context Record requirement identifiers that justify this feature.",
    )
    priority: Literal["must", "should", "could"] = Field(
        default="must",
        description="Business priority using a simplified MoSCoW classification.",
    )
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description="Observable conditions that indicate successful implementation.",
    )


class Blueprint(BaseModel):
    """Architecture Blueprint — stakeholder view + technical view."""
    # TODO(Maheen): real Blueprint schema (two views).

    blueprint_id: str = Field(
        default="BP-001",
        description="Stable identifier for the architecture blueprint.",
    )
    project_name: str = Field(
        default="",
        description="Name of the project described by this blueprint.",
    )
    selected_pattern: str = Field(
        default="",
        description="Primary architecture pattern selected for the solution.",
    )
    rationale: str = Field(
        default="",
        description="Why the selected architecture pattern fits the project context.",
    )
    stakeholder_view: str = Field(
        ...,
        description="Business-facing explanation of actors, value flow, and expected outcomes.",
    )
    technical_view: str = Field(
        ...,
        description="Technical architecture view describing services, data flow, and integrations.",
    )
    components: list[str] = Field(
        default_factory=list,
        description="Names of the main components included in the architecture.",
    )
    data_flows: list[str] = Field(
        default_factory=list,
        description="Important data or event flows between architecture components.",
    )
    addressed_feature_ids: list[str] = Field(
        default_factory=list,
        description="Feature identifiers supported by the overall blueprint.",
    )
    constraints_addressed: list[str] = Field(
        default_factory=list,
        description="Cloud, budget, scalability, compliance, and migration constraints addressed.",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Architecture assumptions used while producing the blueprint.",
    )
    open_risks: list[str] = Field(
        default_factory=list,
        description="Known architectural risks that still require attention.",
    )
    version: str = Field(
        default="1.0",
        description="Version of the blueprint.",
    )
    # Written by the architect on a REFINE pass only, and deliberately kept OUT
    # of the reviewer's view - see `_format_artifacts` in agents/reviewer.py.
    # A design that narrates its own corrections to the judge grading it is
    # asking to be believed rather than checked.
    revision_note: str = Field(
        default="",
        description=(
            "On a revision, a brief statement of what changed and why. "
            "Empty on the initial design."
        ),
    )


class ADR(BaseModel):
    """One Architecture Decision Record."""
    # TODO(Maheen): real ADR schema (title, context, options, decision, trade-offs).

    id: str = Field(
        default="ADR-001",
        description="Stable ADR identifier.",
    )
    title: str = Field(
        ...,
        description="ADR title in the format 'ADR-<number>: <decision>'.",
    )
    status: Literal["proposed", "accepted", "rejected", "superseded"] = Field(
        default="accepted",
        description="Current status of the architecture decision.",
    )
    context: str = Field(
        default="",
        description="Problem, constraint, or architectural situation requiring a decision.",
    )
    decision: str = Field(
        ...,
        description="The selected architecture decision.",
    )
    rationale: str = Field(
        default="",
        description="Why this decision was selected.",
    )
    alternatives_considered: list[str] = Field(
        default_factory=list,
        description="Other options evaluated before making the decision.",
    )
    positive_consequences: list[str] = Field(
        default_factory=list,
        description="Expected benefits of the decision.",
    )
    negative_consequences: list[str] = Field(
        default_factory=list,
        description="Expected disadvantages, costs, or risks.",
    )
    related_feature_ids: list[str] = Field(
        default_factory=list,
        description="Features supported or affected by this decision.",
    )
    related_component_names: list[str] = Field(
        default_factory=list,
        description="Components governed or justified by this decision.",
    )
    source_references: list[str] = Field(
        default_factory=list,
        description="Knowledge-base or repository sources supporting the decision.",
    )


class ComponentDescription(BaseModel):
    """One component's justified description."""
    # TODO(Maheen): real Component Description schema.

    id: str = Field(
        default="COMP-001",
        description="Stable component identifier.",
    )
    name: str = Field(
        ...,
        description="Unique name of the architecture component.",
    )
    component_type: str = Field(
        default="service",
        description="Type of component, for example service, database, queue, API, or UI.",
    )
    purpose: str = Field(
        default="",
        description="Why the component exists and what responsibility it owns.",
    )
    description: str = Field(
        ...,
        description="Detailed explanation of the component and its role.",
    )
    inputs: list[str] = Field(
        default_factory=list,
        description="Data, events, or requests consumed by the component.",
    )
    outputs: list[str] = Field(
        default_factory=list,
        description="Data, events, or responses produced by the component.",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Other components or external systems this component depends on.",
    )
    related_feature_ids: list[str] = Field(
        default_factory=list,
        description="Features implemented or supported by this component.",
    )
    related_adr_ids: list[str] = Field(
        default_factory=list,
        description="Architecture decisions that justify this component.",
    )
    technology_choices: list[str] = Field(
        default_factory=list,
        description="Suggested technologies, platforms, or implementation options.",
    )
    security_considerations: list[str] = Field(
        default_factory=list,
        description="Security, privacy, and compliance considerations.",
    )
    scalability_considerations: list[str] = Field(
        default_factory=list,
        description="Scaling, availability, and performance considerations.",
    )


class RubricScores(BaseModel):
    """Rubric v2 results, with field types reflecting decision ownership.

    Code-owned checks retain 0-2 diagnostic scores. The final verdict requires
    each to equal 2. LLM-owned checks are binary pass/fail judgments; code owns
    the verdict and never sums these values.
    """

    all_artifacts_present: int = Field(0, ge=0, le=2)
    constraint_coverage: int = Field(0, ge=0, le=2)
    traceability: int = Field(0, ge=0, le=2)
    adr_presence: int = Field(0, ge=0, le=2)
    repo_grounding: bool = False
    flaw_detection: bool = False
    adr_soundness: bool = False
    best_practice_grounding: bool = False
    refinement_readiness: bool = False


class JudgmentReasons(BaseModel):
    """Evidence-backed reasons for the five binary LLM judgments."""

    repo_grounding: str = ""
    flaw_detection: str = ""
    adr_soundness: str = ""
    best_practice_grounding: str = ""
    refinement_readiness: str = ""


class ReviewIssue(BaseModel):
    """One concrete problem found in the design (by code or by the LLM)."""

    id: str = ""
    severity: Literal["low", "medium", "high"] = "low"
    category: Literal[
        "completeness", "constraint", "grounding", "traceability",
        "adr", "repo_alignment", "safety",
    ] = "completeness"
    finding: str = ""
    evidence: str = ""
    suggested_fix: str = ""
    requires_refinement: bool = False


# Why a criterion can be marked NOT APPLICABLE, keyed by the names that appear in
# `ReviewResult.not_applicable`. Rendered by both the CLI report and the UI, which
# is why it lives here beside the model rather than inside the reviewer.
NOT_APPLICABLE_REASONS: dict[str, str] = {
    "best_practice_grounding": "no knowledge retrieved",
}


class ReviewResult(BaseModel):
    """Code-assembled Reviewer verdict and audit evidence."""

    overall_status: Literal["pass", "fail"] = "fail"
    rubric_scores: RubricScores = Field(default_factory=RubricScores)
    judgment_reasons: JudgmentReasons = Field(default_factory=JudgmentReasons)
    issues: list[ReviewIssue] = Field(default_factory=list)
    requires_refinement: bool = True
    refinement_instruction: str = ""
    # Criteria that were ASKED and recorded but carry no verdict weight, because
    # there was no evidence for them to be a judgment ABOUT. Deliberately a list
    # of names rather than a tri-state on RubricScores: the booleans stay
    # booleans, so persistence, the UI and every existing test keep working, and
    # this field is what stops a "true" here from reading as a pass. Absent from
    # older checkpoints, hence the default.
    not_applicable: list[str] = Field(default_factory=list)


class DesignSnapshot(BaseModel):
    """One refine round's artifacts and the review that judged them, kept together.

    The unit of BEST-SO-FAR SELECTION (see `refine_gate.score_round`). It exists
    as one object rather than five loose fields for a single reason: a design and
    the review of that design must travel together. Handing back round 2's
    blueprint next to round 3's review would be worse than shipping round 3
    outright — the report would confidently describe artifacts that are not in
    the run.

    `round` is 1-based and counts REVIEWED designs, not refine iterations: round
    1 is the initial design, round 2 the first redesign, and so on. It is
    `refine_iterations + 1` at the moment the gate sees it.
    """

    round: int = Field(1, description="1-based index of the reviewed design (initial design = 1).")
    features: list[Feature] = Field(default_factory=list)
    blueprint: Optional[Blueprint] = None
    adrs: list[ADR] = Field(default_factory=list)
    components: list[ComponentDescription] = Field(default_factory=list)
    review: Optional[ReviewResult] = Field(
        None, description="The verdict on THESE artifacts. Never a different round's."
    )


# ══════════════════════════════════════════════════════════════════════════
# 3. CONTROL / META  (owned by Kati)
# ══════════════════════════════════════════════════════════════════════════
class Stage(str, Enum):
    """Where the run currently is. The orchestrator routes on this value.

    ONE pause stage, not many. `AWAITING_HUMAN` means "the graph has stopped and
    a human owns the next move"; WHICH move is carried by
    `ArchitectState.pending_decision` (see `PendingDecision`). The alternative —
    an `AWAITING_*` stage per kind of interaction — would push every new
    human-in-the-loop touchpoint into the router's static table, and the stage
    enum would slowly become the routing problem it exists to keep simple.

    DONE IS NOT ACCEPTED. Two terminal stages, because they record two different
    facts and a decision-support tool owes its reader both. `DONE` means THE
    PIPELINE STOPPED — the reviewer passed the design, or the refine budget ran
    out and the gate handed back the best round it saw. That is a statement
    about this system. `ACCEPTED` means A HUMAN TOOK THE DESIGN, which is a
    statement about a person, and it is the only one of the two that anybody
    downstream can act on. Collapsing them would make an unread run and a signed
    -off one indistinguishable in the trail.

    `ACCEPTED` is reached ONLY by an explicit user action at DONE (see
    pipeline/sign_off.py). No agent writes it and no route leads to it, which is
    why it needs no row in `STAGE_TO_NODE`: like DONE it is absent from that
    table and therefore routes to END.

    A design the reviewer did NOT pass may be accepted, and that is the normal
    case rather than an escape hatch — most runs end `stopped_on_cap` with open
    findings, so a sign-off that required a pass would leave exactly the runs
    that need closing out unable to be closed. What was accepted DESPITE is
    recorded separately, as a `Waiver`.

    NAMING NOTE. This member was `AWAITING_INPUT` until the context gate landed;
    it is now `AWAITING_HUMAN`, because a lock approval is not "input". The
    STRING VALUE is deliberately still `"awaiting_input"`: checkpoints already
    written to `.cache/runs/` and rows already in `architect.db` carry that
    literal, and they must keep deserialising. Renaming the member is a
    source-level change; renaming the value would be a data migration.
    """

    CREATED = "created"
    CLARIFYING = "clarifying"
    AWAITING_HUMAN = "awaiting_input"  # graph paused; see `pending_decision`
    INGESTING = "ingesting"            # repo_ingestor ran (or skipped: greenfield)
    RESEARCHING = "researching"
    DESIGNING = "designing"
    REVIEWING = "reviewing"
    REFINING = "refining"
    DONE = "done"
    ACCEPTED = "accepted"              # a human took the design (terminal; see below)
    FAILED = "failed"


class PendingDecision(str, Enum):
    """WHICH decision a human owes us while `stage is Stage.AWAITING_HUMAN`.

    The discriminator that keeps `Stage` small (see the note on
    `Stage.AWAITING_HUMAN`). `None` means nothing is pending — the run is either
    moving or terminal.

      * CLARIFICATION — the clarifier asked questions it cannot safely assume
        past. Resolved by writing answers into `clarification_answers` and
        re-entering the graph, which sends the state back to the clarifier.
      * CONTEXT_LOCK  — a ContextRecord has been frozen and is waiting to be
        approved, edited, or questioned BEFORE any expensive work is spent on
        it. Resolved ENTIRELY by the caller (see `pipeline.agents.clarifier`):
        the graph is never entered with this pending, and `_entry_route` says so
        out loud rather than routing on a half-resolved pause.

    Feature A (user feedback at DONE) was expected to add a member here. It
    added neither a member nor a stage in the end, which is the stronger
    outcome: a requirements correction re-opens the run as an ordinary
    CLARIFICATION (the clarifier re-judges and re-freezes, exactly as it does
    after an edit at the lock), and a design directive is not a pause at all —
    it sets `stage = REFINING` and re-enters the loop that already exists. The
    route is decided by WHICH box the person typed into, so it is known before
    any model runs. See pipeline/user_feedback.py.
    """

    CLARIFICATION = "clarification"
    CONTEXT_LOCK = "context_lock"


def _new_run_id() -> str:
    """Generate a run identifier that sorts chronologically and never collides.

    `<UTC timestamp>-<short uuid>`: the timestamp makes a directory listing
    readable and time-ordered, the suffix makes two runs started in the same
    second distinct. Filesystem-safe on every platform (no colons).
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


class StepLog(BaseModel):
    """One entry in the run trace: which agent ran, when, what it did, what it cost.

    The token/cost fields are what make PER-AGENT attribution free: this entry
    already carries the agent name, and `history` already accumulates through a
    reducer, so "tokens per agent" is a plain groupby over the trace (see
    `ArchitectState.usage_by_agent`) and needs no custom dict reducer.

    A step with no LLM call (researcher, refine_gate, a greenfield repo_ingestor)
    simply leaves these at zero.
    """

    agent: str
    stage_in: Stage
    stage_out: Stage
    note: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model: str = Field("", description="Real model ID(s) this step called; empty when it called none.")
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = Field(0.0, description="List-price-equivalent cost of THIS step, in USD (free-tier key: not money spent).")


class AgentUsage(BaseModel):
    """One agent's share of the run, aggregated from `history`.

    A derived view, never stored on the state — see
    `ArchitectState.usage_by_agent`.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class AdvisoryTurn(BaseModel):
    """One read-only question a human asked while the run was paused, and its answer.

    The advisory side-channel (see `clarifier.ask_advisor`) runs OUTSIDE the
    graph: it changes no artifact, no stage and no `pending_decision`. It is
    still recorded here rather than in the UI's session, because the whole front
    end is DERIVED from this object — a transcript kept in `st.session_state`
    would vanish on the next widget interaction and would not survive a
    checkpoint reload, which is exactly the property the rest of the UI relies on.

    The token COST of the same turn is recorded separately, as an ordinary
    `StepLog` in `history`, so per-agent accounting stays a plain groupby.

    `subject` says WHAT was being read when the question was asked, and it is
    what keeps two conversations from being spliced into one. The side channel
    is now offered at both ends of a run — over the frozen record at the context
    lock, and over the finished artifacts at DONE — and those threads have
    nothing to do with each other. Without the tag, a question about an ADR
    would arrive at the model with the lock-gate Q&A about scale and budget
    stapled above it, and the answer would be about the wrong thing.

    Defaults to `"context_record"` so checkpoints written before the design
    thread existed load as what they were: questions about the record.
    """

    question: str
    answer: str = ""
    subject: Literal["context_record", "design"] = Field(
        "context_record",
        description="Which artifact set the question was asked about — the thread it belongs to.",
    )
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class UserFeedback(BaseModel):
    """One thing a human asked for at DONE, and what became of it.

    THE ROUTE IS THE BOX, NOT A CLASSIFIER. `kind` is not inferred from the
    text: it records WHICH of the two boxes on the finished-run screen the
    person typed into. That is what keeps the graph 100% deterministically
    routed with a human in the loop — the requirements box re-opens the Context
    Record, the design box re-enters the refine loop, and no model is asked to
    guess which one was meant.

    `status` is the consumption receipt, and every value it can take is a
    DIFFERENT ANSWER to "what became of what the person typed":

      * `pending`   — recorded, and nothing has acted on it yet.
      * `applied`   — the owning agent consumed it and did what it said. Written
                      by the clarifier for a requirements correction and by the
                      architect for a design directive. Two agents write this
                      field, never the same entry, because `kind` decides which
                      one owns it.
      * `objected`  — the architect consumed it, could not build it AS STATED,
                      built the closest feasible variant, and said so in
                      `objection`. Still consumed: this is a voice, not a veto,
                      and the pass proceeds.
      * `abandoned` — the human signed the run off without ever re-running, so
                      no agent will now consume it. Written by the sign-off (see
                      pipeline/sign_off.py) against their explicit confirmation.

    `abandoned` exists because the alternative is worse than losing the text: an
    entry left `pending` on a closed run is INDISTINGUISHABLE from one queued
    and about to run, so the trail would claim work was outstanding on a run
    nobody is going to touch again.

    `objection` is the architect's reason, and it is written HERE — one field
    away from `text`, which the architect's own prompt reads. That adjacency is
    deliberate (the objection belongs to the directive it objects to, not to the
    Blueprint) and it is exactly why the `<user_directive>` builder serialises
    `text` and nothing else: an architect that could read back its own
    objections would learn that objecting makes a directive go away.

    `round` is the `user_rounds` value this submission was charged (see
    `refine_gate.begin_user_round`), so the trail can say how much of the budget
    each piece of feedback cost.
    """

    text: str = Field(..., description="What the human typed, verbatim.")
    kind: Literal["requirements", "design"] = Field(
        ...,
        description="Which box it came from — the route, not a judgment about the text.",
    )
    status: Literal["pending", "applied", "objected", "abandoned"] = Field(
        "pending",
        description="What became of it — see the class docstring; anything but 'pending' is consumed.",
    )
    objection: str = Field(
        "",
        description=(
            "The architect's account of why this could not be built as stated and "
            "what it built instead. Set with status 'objected'; NEVER fed back to "
            "any agent."
        ),
    )
    submitted_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    round: int = Field(0, description="The `user_rounds` value this submission was charged.")


class Waiver(BaseModel):
    """The recorded decision to ship a design WITH a known problem in it.

    One per sign-off, naming every open finding it covers — not one per finding.
    A person signs off once, on the whole picture, and splitting that single
    decision into N records would invent a deliberation that did not happen.

    WHY IT EXISTS AT ALL. Without it, an accepted best-effort design and an
    accepted clean one are indistinguishable the moment the run is closed: both
    are `ACCEPTED`, both carry artifacts, and the review that objected is just
    one more expander nobody opens. The waiver is what makes "we knew, and we
    took it anyway" a fact on the record rather than something a reader has to
    reconstruct. A CLEAN sign-off writes no waiver at all — its absence is
    itself the information, and a waiver with an empty finding list would
    destroy that.

    `finding_ids` and `severities` are PARALLEL: `severities[i]` is the severity
    of `finding_ids[i]`, both ordered highest severity first, and the validator
    below refuses a pair that has drifted. Two lists rather than one list of
    objects because these are a snapshot, not a view onto `review.issues` — the
    review can be re-run, and what was waived cannot change afterwards.

    `note` is optional and stays optional. A mandatory justification field
    produces "n/a" on every second run and then means nothing; an empty note
    honestly says the person had nothing to add.

    IT IS FOR HUMANS ONLY. A waiver never enters an agent prompt — least of all
    the architect's, where a waived finding would read as a solved one.
    """

    finding_ids: list[str] = Field(
        default_factory=list,
        description="IDs of every open finding this sign-off accepted, highest severity first.",
    )
    severities: list[str] = Field(
        default_factory=list,
        description="Severity of each entry in `finding_ids`, same order, same length.",
    )
    review_status: str = Field(
        "",
        description="The review's `overall_status` at sign-off — what the gate said, kept beside what the human did.",
    )
    accepted_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="When the sign-off happened. There is no user identity in this system, so there is no signer.",
    )
    note: str = Field("", description="Optional free text from the signer, e.g. 'tracked in JIRA-123'.")

    @model_validator(mode="after")
    def _severities_match_findings(self) -> "Waiver":
        if len(self.severities) != len(self.finding_ids):
            raise ValueError(
                f"Waiver.severities has {len(self.severities)} entries for "
                f"{len(self.finding_ids)} finding(s). They are parallel lists: "
                f"severities[i] is the severity of finding_ids[i]."
            )
        return self

    def severity_counts(self) -> dict[str, int]:
        """How many findings of each severity this waiver covers. Derived, for display."""
        counts: dict[str, int] = {}
        for severity in self.severities:
            counts[severity] = counts.get(severity, 0) + 1
        return counts


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


class CapturedContext(BaseModel):
    """The structured understanding the Clarifier extracts from the prompt.

    Its fields deliberately MIRROR the extractable subset of Maheen's
    `ContextRecord`, so freezing the record is a near 1:1 copy (see
    `clarifier._freeze_context_record`). All fields are `str` / `list[str]`
    only — no `dict` — because Gemini's Developer-API structured-output mode
    rejects the `additionalProperties` a dict would generate. Empty values are
    fine: whatever is missing-and-critical is reported via `missing_critical`.
    """

    project_name: str = Field("", description="Short name identifying the project, if stated.")
    business_goal: str = Field("", description="Business outcome the architecture must support.")
    problem_statement: str = Field("", description="Current problem, limitation, or flaw to address.")
    users: list[str] = Field(default_factory=list, description="Users / stakeholders / user groups.")
    functional_requirements: list[str] = Field(default_factory=list, description="Capabilities the system must provide, AS STATED by the user (the Architect derives formal Features later).")
    non_functional_requirements: list[str] = Field(default_factory=list, description="Quality needs: scale, availability, performance, latency.")
    cloud_provider: str = Field("", description="Required/preferred cloud provider, if specified.")
    budget: str = Field("", description="Budget level or cost constraint, if specified.")
    compliance_requirements: list[str] = Field(default_factory=list, description="Legal / regulatory / security requirements, e.g. GDPR, PCI-DSS.")
    existing_systems: list[str] = Field(default_factory=list, description="Existing apps, services, data stores, or integrations (brownfield).")


class ClarificationResult(BaseModel):
    """The Clarifier's structured judgment of the raw prompt (+ any answers).

    `missing_critical` is THE signal the gate routes on: non-empty means an
    architecture-critical fact is still unknown, so the run must pause and ask.
    Empty means we can lock the ContextRecord and advance.
    """

    captured: CapturedContext = Field(default_factory=CapturedContext, description="Structured facts grounded in the prompt.")
    assumptions: list[str] = Field(default_factory=list, description="Low-stakes fills, labelled so a human can veto them.")
    questions: list[ClarifyingQuestion] = Field(default_factory=list, description="Questions to ask when critical info is missing.")
    missing_critical: list[str] = Field(default_factory=list, description="Architecture-critical gaps. Non-empty ⇒ pause and ask.")


# ── Clarifier INPUT from the human (Kati owns) ───────────────────────────
# The other direction of the same contract: `ClarificationResult` is what the
# LLM says about the record, `ContextEdits` is what the HUMAN says about it at
# the context-lock gate. It never travels on the state and never reaches a
# model — it is consumed by `clarifier.apply_user_edits`, which is pure code —
# so unlike `CapturedContext` it is free to use a dict and a union type.
class ContextEdits(BaseModel):
    """One human's veto pass over a locked ContextRecord.

    Every field is optional; an empty instance is a no-op, which is what makes
    "Accept" and "Edit" the same code path with different content.

    `fields` is keyed by ContextRecord attribute name. A `str` field takes a
    `str`, a `list[str]` field takes a `list[str]` — `apply_user_edits`
    validates both the NAME and the TYPE and raises on either being wrong,
    because a silently-dropped edit is a veto the human thinks they cast.
    Turning typed text into list items is the CALLER's convention, deliberately
    not baked in here: ui.py has a textarea and splits on newlines, run.py has
    one line to work with and splits on commas. The schema takes real lists and
    stays out of it.
    """

    fields: dict[str, str | list[str]] = Field(
        default_factory=dict,
        description="ContextRecord field name -> replacement value.",
    )
    struck_assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions the human rejects, verbatim. Removed, and remembered so a later re-judge cannot re-propose them.",
    )
    answered_questions: dict[str, str] = Field(
        default_factory=dict,
        description="Open question -> the human's answer. The question is resolved; the answer joins the run's Q&A.",
    )
    recommend: list[str] = Field(
        default_factory=list,
        description="ContextRecord field names the human wants the clarifier to propose a value for ('you recommend').",
    )

    def is_empty(self) -> bool:
        """True when this edit set would change nothing — i.e. a plain Accept."""
        return not (
            self.fields or self.struck_assumptions
            or self.answered_questions or self.recommend
        )


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
    # Every SUPERSEDED Context Record, oldest first — the record's own history.
    # `context_record` is always the current version; a correction after design
    # pushes the outgoing one here rather than overwriting it, so an artifact set
    # can still be read against the ground truth it was actually built on.
    #
    # A plain list, not a reducer: the clarifier returns the whole list it wants
    # (it is the only writer of a ContextRecord and therefore the only thing that
    # can supersede one), and an `operator.add` channel would append a second
    # copy of every past version on each re-freeze.
    context_history: list[ContextRecord] = Field(default_factory=list)
    blueprint: Optional[Blueprint] = None
    adrs: list[ADR] = Field(default_factory=list)
    components: list[ComponentDescription] = Field(default_factory=list)
    review: Optional[ReviewResult] = None

    # --- 3. Working fields (intermediate results) -------------------------
    clarifying_questions: list[str] = Field(default_factory=list)
    clarification_answers: dict[str, str] = Field(default_factory=dict)
    # Assumptions a human STRUCK at the context-lock gate. Kept because a strike
    # has to outlive the record it was cast against: an edit that empties a
    # critical field re-runs the clarifier, which would otherwise cheerfully
    # re-propose the very assumption that was just vetoed. `_freeze_context_record`
    # filters against this, so the veto is enforced by code, not by the prompt.
    vetoed_assumptions: list[str] = Field(default_factory=list)
    # Read-only Q&A the human asked while paused (see `AdvisoryTurn`). Written
    # only by `clarifier.ask_advisor`, outside the graph. Not a reducer: no node
    # returns it, so LangGraph carries the input value through untouched.
    advisory_turns: list[AdvisoryTurn] = Field(default_factory=list)
    # Every piece of feedback a human submitted at DONE, oldest first (feature
    # A). APPEND-ONLY BY CONVENTION, never replaced: the run's value as an audit
    # trail is that it says what was asked and when, so a second correction must
    # not overwrite the first — the same failure the unique key on
    # `clarification_answers` exists to prevent.
    #
    # A plain list rather than an `operator.add` reducer, unlike `history`. The
    # append happens CALLER-SIDE, outside the graph (see
    # pipeline/user_feedback.py), and the agent that consumes an entry returns
    # the whole list with that entry's `status` flipped to "applied". Under a
    # reducer that return would append a duplicate of every entry instead of
    # updating one.
    user_feedback: list[UserFeedback] = Field(default_factory=list)
    # THE SIGN-OFF, and what it was made against. Both written by
    # pipeline/sign_off.py, together, once, and never by an agent.
    #
    # `accepted_at` is a timestamp and nothing else: there is no user identity
    # anywhere in this system, so there is no signer to name and one must not be
    # invented. Empty means nobody has taken this design — the honest default
    # for every run, including the ones that finished perfectly.
    #
    # `waiver` is present ONLY when the sign-off accepted open findings. Its
    # ABSENCE is information (a clean design was accepted clean), which is why
    # this is Optional rather than an always-present object with an empty list.
    # See `Waiver`. Neither field ever enters an agent prompt.
    accepted_at: str = ""
    waiver: Optional[Waiver] = None
    retrieved_knowledge: list[KBChunk] = Field(default_factory=list)
    features: list[Feature] = Field(default_factory=list)
    # Drill-down cache (Malte): the Architect appends one DeepDive whenever it
    # takes a deeper look at part of the repo, so an insight is derived at most
    # once per run. Reducer (append) — same pattern as `history` below.
    repo_deep_dives: Annotated[list[DeepDive], operator.add] = Field(default_factory=list)

    # --- 4. Control / orchestration meta (Kati owns fully) ----------------
    # Stable identity of ONE run, assigned at birth and never rewritten by any
    # agent. It is what state-on-disk keys on: every checkpoint of this run
    # lands in `.cache/runs/<run_id>/` (see pipeline/persistence.py). Purely
    # additive — nothing upstream of persistence reads it.
    run_id: str = Field(default_factory=_new_run_id)
    stage: Stage = Stage.CREATED
    # WHICH decision the human owes us while `stage is AWAITING_HUMAN`; None the
    # rest of the time. The discriminator that keeps one pause stage rather than
    # one stage per interaction — see `PendingDecision`.
    pending_decision: Optional[PendingDecision] = None
    # ONE GLOBAL counter of POST-LOCK user-initiated refinement, across every
    # stage. The cap itself (`MAX_USER_ROUNDS`) lives in refine_gate.py, which
    # owns budget policy; the rule for what counts is:
    #
    #   COUNTS    — a graph re-entry the human caused by CHANGING something
    #               after the Context Record was first locked: an edit at the
    #               approval gate that reopened a gap, and (feature A) feedback
    #               at DONE. These ask the pipeline to REDO work.
    #   DOES NOT  — answering clarifying questions, approving the lock, and
    #               advisory turns.
    #
    # It was originally "any user-initiated re-entry, any stage", and that was
    # wrong in a way only a real run showed: run 20260818T074835Z-1925fbd7
    # reached the architect with user_rounds ALREADY 6 of 6, because four rounds
    # of clarifying questions plus the approval had consumed the entire budget
    # before any design existed. But answering the clarifier is the pipeline
    # working exactly as designed, and approving releases work rather than
    # redoing it — neither is the runaway this cap exists to stop. Charging for
    # them meant the budget was spent before the thing it was guarding began.
    # (`run.MAX_CLARIFICATION_ROUNDS` is what bounds a clarifier that will not
    # converge; that is a different failure and it already has its own cap.)
    #
    # Global rather than per-touchpoint because it is a COST cap and cost is
    # global: a per-stage counter would let the same person spend three budgets
    # by spreading the same number of round trips across three screens.
    user_rounds: int = 0
    # Does this run pause at the context lock for human approval? OFF by default
    # ON PURPOSE: `pipeline/run.py`, the eval harness and the whole test suite
    # drive the pipeline with no human attached, and a mandatory pause would turn
    # every one of them into a hang. The UI — the only caller with a human in
    # front of it — sets this True; the CLI exposes it as `--approve-context`.
    # Default-off means "auto-approve", which is exactly today's behaviour.
    require_context_approval: bool = False
    # Annotated with operator.add so LangGraph MERGES (appends) each node's
    # returned step onto the running trace instead of overwriting it. This is
    # the "reducer" — nodes return {"history": [one_step]} and it accumulates.
    history: Annotated[list[StepLog], operator.add] = Field(default_factory=list)
    retry_counts: dict[str, int] = Field(default_factory=dict)
    refine_iterations: int = 0
    # How many times the clarifier has PAUSED TO ASK, before any context lock.
    # Bounded by `clarifier.MAX_ASK_ROUNDS`. Distinct from `user_rounds`, and the
    # boundary is the lock: this counts the pipeline gathering the ground truth,
    # `user_rounds` counts a human sending finished work back to be redone. They
    # were one counter once, and four clarifying rounds spent the whole redo
    # budget before the architect had run - see refine_gate.MAX_USER_ROUNDS.
    #
    # A PLAIN int, not an operator.add reducer: the clarifier returns the
    # absolute value it wants (`state.ask_rounds + 1`), so LangGraph overwrites
    # rather than accumulates. A reducer here would double-count every replay of
    # a resumed run.
    ask_rounds: int = 0
    # Set True by the refine gate when the reviewer→refine loop stops on a cap
    # (max iterations or token budget) rather than on a clean reviewer pass.
    # An honest "finished best-effort, not perfect" signal for the UI/report;
    # the run still ends as DONE (not FAILED). Single writer: the refine gate.
    stopped_on_cap: bool = False
    # The best round SEEN SO FAR, carried forward by the refine gate so that a
    # capped run can hand back its best design instead of its last one. Single
    # writer: the refine gate (see `refine_gate.score_round`).
    #
    # Exactly ONE snapshot — the incumbent — never a list of every round. The
    # cost is what makes that the right call: a snapshot is a full artifact set,
    # and `pipeline/persistence.py` writes one file PER TRANSITION. Measured on
    # a real brownfield run (20260818T095745Z-79dceb73), the incumbent adds
    # ~12.8 kB to a ~59 kB checkpoint — 22% — on every checkpoint from the first
    # gate visit onward. Keeping one incumbent pays that once; keeping every
    # round would multiply it by the iteration cap, to store information the
    # trace in `history` already records in a few hundred bytes. That 22% is the
    # deliberate price of the feature, not an oversight.
    best_design: Optional[DesignSnapshot] = None
    # Which round's design the run actually SHIPPED. Written by the refine gate
    # on its stop branch, so it is the answer to "is what I am looking at the
    # last thing the architect produced, or an earlier one?". 0 means the
    # question never arose: the run passed at the reviewer, failed, or is still
    # going, and never reached the gate's stop branch.
    selected_round: int = 0
    # Reducers (add), same pattern as `history` and `errors`. Each node returns
    # ONLY the tokens ITS OWN calls consumed and LangGraph adds them onto the
    # running total. They must be reducers: `llm_call` is pure with respect to
    # state, so a node that merely mutated these would have its count dropped —
    # which is exactly the bug that left the refine gate's budget comparing 0
    # against 500k forever.
    input_tokens: Annotated[int, operator.add] = 0
    output_tokens: Annotated[int, operator.add] = 0
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

    def pending_feedback(
        self, kind: Literal["requirements", "design"]
    ) -> list[UserFeedback]:
        """Feedback of one kind that no agent has consumed yet, oldest first.

        READ-ONLY and pure — it returns the entries themselves, not copies, but
        writing `status` is the consuming agent's job and happens through its
        returned state update, never through this list.

        It lives here rather than in the agent that reads it because THREE
        readers need the same answer and must not each invent it: the architect
        (does a directive belong in this prompt?), the clarifier (is this
        re-freeze a user correction, and what is its reason?), and the refine
        gate (is the design on the state one the user has already redirected?).
        """
        return [
            entry
            for entry in self.user_feedback
            if entry.kind == kind and entry.status == "pending"
        ]

    def feedback_marked_applied(
        self, consumed: list[UserFeedback], *, objection: str = ""
    ) -> list[UserFeedback]:
        """The WHOLE feedback list with `consumed` no longer pending. Pure.

        Returns a new list of new objects and mutates nothing, because a node
        must not write into the state it was handed — LangGraph persists only
        what a node RETURNS, so a consuming agent returns this list under
        `user_feedback` and the flip lands with the rest of its update or not at
        all. An entry that never reaches its agent therefore stays `pending`,
        which is the honest outcome: nothing acted on it.

        Matching is by identity, on the objects `pending_feedback` handed back
        from this same state. Two submissions can carry identical text and both
        must be tracked separately, so equality would mark the wrong one.

        `objection`, when the architect could not build a directive as stated,
        lands on the LAST entry in `consumed` and marks it `objected` instead of
        `applied`. Last, because the response schema carries ONE objection while
        `consumed` can hold several directives, and the last is the newest — the
        one the person typed most recently and the one they are still looking
        at. Several pending directives at once means an earlier architect pass
        raised, which is rare; guessing the newest is better than spraying one
        reason across entries it may not be about. Everything else in `consumed`
        is `applied` as usual, because the pass DID carry them out.
        """
        consumed_ids = {id(entry) for entry in consumed}
        objected_id = id(consumed[-1]) if (objection and consumed) else None
        out: list[UserFeedback] = []
        for entry in self.user_feedback:
            if id(entry) == objected_id:
                out.append(
                    entry.model_copy(
                        update={"status": "objected", "objection": objection}
                    )
                )
            elif id(entry) in consumed_ids:
                out.append(entry.model_copy(update={"status": "applied"}))
            else:
                out.append(entry.model_copy())
        return out

    def bump_retry(self, agent: str) -> int:
        """Increment and return this agent's retry counter (for retry caps later)."""
        self.retry_counts[agent] = self.retry_counts.get(agent, 0) + 1
        return self.retry_counts[agent]

    def usage_by_agent(self) -> dict[str, AgentUsage]:
        """Per-agent token/cost totals. READ-ONLY — derived, never stored.

        A plain groupby over `history`: every StepLog carries the agent name and
        that step's own usage, so no extra reducer or bookkeeping field is
        needed. An agent that ran several times (the Architect in a refine loop)
        accumulates across all of its steps.

        Sums to `input_tokens` / `output_tokens` by construction, because each
        node writes the SAME numbers to its StepLog and to its returned update.
        """
        totals: dict[str, AgentUsage] = {}
        for step in self.history:
            bucket = totals.setdefault(step.agent, AgentUsage())
            bucket.input_tokens += step.input_tokens
            bucket.output_tokens += step.output_tokens
            bucket.cost_usd += step.cost_usd
        return totals

    def total_cost_usd(self) -> float:
        """List-price-equivalent cost of the whole run in USD, summed from `history`.

        NOT money spent: we run on free-tier keys, so this is what the run would
        have cost at Google's list prices (see `pipeline/llm.py`).
        """
        return sum(step.cost_usd for step in self.history)

def new_run(
    raw_prompt: str,
    repo_url: str = "",
    require_context_approval: bool = False,
) -> ArchitectState:
    """Factory: build a fresh state at the start of a run.

    `raw_prompt` (the verbatim user message = ground truth) and the optional
    `repo_url` (the existing codebase, empty for greenfield) are the only
    inputs. Everything else is derived downstream by the agents.

    `require_context_approval` defaults to False so that every EXISTING caller —
    the CLI, the eval harness, the tests — keeps running unattended exactly as
    before. Pass True only when there is a human to pause for.
    """
    return ArchitectState(
        initial_request=InitialRequest(raw_prompt=raw_prompt, repo_url=repo_url),
        require_context_approval=require_context_approval,
    )

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
