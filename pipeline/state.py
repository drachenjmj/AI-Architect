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

from pydantic import BaseModel, Field


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


REVIEW_CODE_SCORE_FIELDS = (
    "all_artifacts_present",
    "constraint_coverage",
    "traceability",
    "adr_presence",
    "source_integrity",
)
REVIEW_JUDGMENT_FIELDS = (
    "repo_grounding",
    "flaw_detection",
    "adr_soundness",
    "best_practice_grounding",
    "refinement_readiness",
)
REVIEW_ADVISORY_FIELDS = ("refinement_readiness",)
REVIEW_VERDICT_JUDGMENT_FIELDS = tuple(
    name for name in REVIEW_JUDGMENT_FIELDS if name not in REVIEW_ADVISORY_FIELDS
)


class RubricScores(BaseModel):
    """Reviewer results, with field types reflecting decision ownership.

    Code-owned checks retain 0-2 diagnostic scores. The final verdict requires
    each to equal 2. LLM-owned checks are binary pass/fail judgments; code owns
    the verdict and never sums these values.
    """

    all_artifacts_present: int = Field(0, ge=0, le=2)
    constraint_coverage: int = Field(0, ge=0, le=2)
    traceability: int = Field(0, ge=0, le=2)
    adr_presence: int = Field(0, ge=0, le=2)
    # Defaults to full only for checkpoints written before this check existed.
    # The production Reviewer always overwrites it with the current run's check.
    source_integrity: int = Field(2, ge=0, le=2)
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
        "adr", "repo_alignment", "evidence", "safety",
    ] = "completeness"
    finding: str = ""
    evidence: str = ""
    suggested_fix: str = ""
    requires_refinement: bool = False


# Why a criterion can be marked NOT APPLICABLE, keyed by the names that appear in
# `ReviewResult.not_applicable`. Rendered by both the CLI report and the UI, which
# is why it lives here beside the model rather than inside the reviewer.
NOT_APPLICABLE_REASONS: dict[str, str] = {
    "repo_grounding": "greenfield request with no repository",
    "best_practice_grounding": "no repository or knowledge evidence supplied",
}


class ReviewResult(BaseModel):
    """Code-assembled Reviewer verdict and audit evidence."""

    # Older checkpoints have no version field and therefore load as v2.0.
    # Fresh Reviewer reports set this explicitly to v3.0.
    rubric_version: Literal["2.0", "3.0"] = "2.0"
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

    Feature A (user feedback at DONE) adds its own member here, not its own
    stage.
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
    """

    question: str
    answer: str = ""
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
