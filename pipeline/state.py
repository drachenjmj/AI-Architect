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
    # TODO(Kush): real KB chunk format (content + source + metadata).
    content: str = ""
    source: str = ""


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
    """The eight rubric items from docs/prompt_quality/05_eval_rubric_v1.md, 0-2 each.

    Ownership per item (who actually decides the number in the final report):
      [code]   all_artifacts_present, constraint_coverage, traceability — computed
               deterministically in pipeline/review_checks.py; the reviewer's merge
               step overwrites whatever the LLM returned for them.
      [LLM]    repo_grounding, flaw_detection, best_practice_grounding,
               refinement_readiness — qualitative judgment only.
      [hybrid] adr_quality — min(code presence score, LLM soundness score).
    """

    all_artifacts_present: int = Field(0, ge=0, le=2)
    constraint_coverage: int = Field(0, ge=0, le=2)
    repo_grounding: int = Field(0, ge=0, le=2)
    flaw_detection: int = Field(0, ge=0, le=2)
    traceability: int = Field(0, ge=0, le=2)
    adr_quality: int = Field(0, ge=0, le=2)
    best_practice_grounding: int = Field(0, ge=0, le=2)
    refinement_readiness: int = Field(0, ge=0, le=2)


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


class ReviewResult(BaseModel):
    """Reviewer's verdict — frozen mirror of docs/prompt_quality/06_reviewer_report_schema.json.

    Field-for-field identical to the JSON schema (test_reviewer.py asserts this),
    so the same object serves as Gemini's response_schema AND as the state field.
    """

    overall_status: Literal["pass", "pass_with_minor_issues", "fail"] = "fail"
    score_total: int = Field(0, ge=0, le=16)
    max_score: int = 16
    rubric_scores: RubricScores = Field(default_factory=RubricScores)
    # NB: rates the DESIGN, not the Reviewer — true when the submitted design
    # itself identifies the ground-truth flaw and fixes it structurally; false
    # when the design misses or merely patches it. So on a flawed design a
    # correctly-working Reviewer reports flaw_detected=False.
    flaw_detected: bool = False
    issues: list[ReviewIssue] = Field(default_factory=list)
    requires_refinement: bool = True
    refinement_instruction: str = ""


# ══════════════════════════════════════════════════════════════════════════
# 3. CONTROL / META  (owned by Kati)
# ══════════════════════════════════════════════════════════════════════════
class Stage(str, Enum):
    """Where the run currently is. The orchestrator routes on this value."""

    CREATED = "created"
    CLARIFYING = "clarifying"
    AWAITING_INPUT = "awaiting_input"  # clarifier needs the human; graph pauses here
    INGESTING = "ingesting"            # repo_ingestor ran (or skipped: greenfield)
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
    # Drill-down cache (Malte): the Architect appends one DeepDive whenever it
    # takes a deeper look at part of the repo, so an insight is derived at most
    # once per run. Reducer (append) — same pattern as `history` below.
    repo_deep_dives: Annotated[list[DeepDive], operator.add] = Field(default_factory=list)

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

def new_run(raw_prompt: str, repo_url: str = "") -> ArchitectState:
    """Factory: build a fresh state at the start of a run.

    `raw_prompt` (the verbatim user message = ground truth) and the optional
    `repo_url` (the existing codebase, empty for greenfield) are the only
    inputs. Everything else is derived downstream by the agents.
    """
    return ArchitectState(
        initial_request=InitialRequest(raw_prompt=raw_prompt, repo_url=repo_url)
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
