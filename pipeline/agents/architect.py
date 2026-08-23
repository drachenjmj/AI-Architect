"""architect.py — Architect/Writer agent (Maheen).

Two-phase architecture design:

1. Derive testable features from the frozen Context Record.
2. Design the architecture from those features, retrieved knowledge,
   repository context, and any Reviewer refinement instruction.

The node returns structured Pydantic artifacts through ArchitectState.

PHASE 1 RUNS ONCE PER RUN, NOT ONCE PER PASS
--------------------------------------------
On a REFINE pass the features are REUSED, not re-derived, and only phase 2
runs. This is not an optimisation that happens to save a call — it is what makes
the refine loop able to close a finding at all.

Measured, on run `20260818T074835Z-1925fbd7`: every design-quality judgment
passed on the first attempt, and the single substantive finding was the same one
in all three rounds — "Feature(s) have no implementing component: FEAT-005",
with the instruction "Assign FEAT-005 to a component". But phase 1 re-derived
the feature set from the Context Record on every pass, producing 5 features,
then 4, then 5. The IDs are positional, so `FEAT-005` on round 2 was a different
feature from `FEAT-005` on round 1, and on the round that produced only four it
did not exist at all. The loop was being asked to fix a reference to a thing it
re-randomised each iteration; ~55k tokens went into that.

Reusing `state.features` makes the IDs stable by construction, so the reviewer's
instruction still refers to something real on the next pass. Removing one LLM
call per refine round is the side benefit, not the reason.

Only FEATURES are stabilised. ADR and component IDs are still regenerated every
pass and have the same drift, which is a real (unfixed) problem — features come
first because the traceability rubric check keys on feature IDs, so they are
what the loop is currently unable to converge on.

TWO INSTRUCTION BLOCKS, RANKED (feature A)
-------------------------------------------
A refine pass can now carry `<user_directive>` as well as
`<refinement_instruction>`. They are kept apart in the prompt and apart in the
state: the reviewer is the only writer of `review.refinement_instruction`, so
everything in that field is a finding, and everything in the directive block is
a person. Overloading one field with both would have been the smaller change and
would have cost the trail its ability to answer "who asked for this?", which is
the question the trail exists for.

The ranking needed a matching change to the PRESERVATION rule, or it would have
been ranking with no effect. That rule tells the architect to keep component
names, ADR IDs and every decision the findings do not mention — and a user
directive is not a finding, so as written it would have faithfully preserved the
exact thing the human asked to change. The rule now has three exits: the
findings, structural necessity, and the user's direction.

OBJECTIONS: A VOICE, NOT A VETO (feature A2)
---------------------------------------------
The directive wins — that did not change. What changed is what happens when it
cannot be built AS STATED. Previously the only available behaviours were to
comply impossibly or to substitute something else silently, and the second is
what a model actually does. So `ArchitectureDesign` gains `directive_objection`:
the architect builds the closest feasible variant AND records what was asked,
why it does not work as stated, and what it built instead.

The objection is TRANSPORTED on the phase-2 response schema and STORED on the
matching `UserFeedback` entry (status `objected`) — named explicitly here
because the only other object with a spare free-text slot is `Blueprint`, and
that is precisely where it must not go: `revision_note` is already excluded from
the reviewer's prompt as self-advocacy, and an objection is the same thing about
a different audience. It belongs to the directive it objects to.

One objection per pass, then the run proceeds. There is no argument loop, no
re-prompt and no blocked design — a design that deviates from what the user
asked, with the reason attached, is the whole deliverable.

And the architect NEVER READS ITS OWN OBJECTIONS BACK. See the note on the
`<user_directive>` builder: storing the objection beside the text the prompt
does read makes that a live leak rather than a hypothetical one.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from pipeline.agents.base import make_step, node
from pipeline.llm import (
    LLMUsage,
    attach_usage,
    llm_call,
    role_model_override,
    sum_usage,
)
from pipeline.state import (
    ADR,
    ArchitectState,
    Blueprint,
    ComponentDescription,
    Feature,
    Stage,
)


# The frozen Gemini default. Per-call routing goes through
# `role_model_override("architect", ...)` at each llm_call site, so
# ARCHITECT_LLM_PROVIDER/ARCHITECT_LLM_MODEL redirect the Architect to
# Claude for the A/B experiment and NOTHING else changes.
ARCHITECT_MODEL = "flash-lite"

# The arrow spellings a data flow may use; canonicalization rebuilds flows
# with the canonical " → " form (what the diagram parser accepts regardless).
_FLOW_SPLIT_RE = re.compile(r"\s*(?:→|->)\s*")


class FeatureDesign(BaseModel):
    """Structured output from phase 1 of the Architect Agent."""

    features: list[Feature] = Field(
        default_factory=list,
        description="Testable features derived from the Context Record.",
    )


class ArchitectureDesign(BaseModel):
    """Structured output from phase 2 of the Architect Agent."""

    blueprint: Blueprint
    adrs: list[ADR] = Field(default_factory=list)
    components: list[ComponentDescription] = Field(default_factory=list)
    directive_objection: str = Field(
        "",
        description=(
            "Empty unless a <user_directive> could not be implemented AS STATED. "
            "Then: what was asked, why it does not work as stated, and what was "
            "built instead."
        ),
    )


FEATURE_SYSTEM_PROMPT = """
You are the feature-design phase of an AI solution architect.

Derive concrete, testable system features from the supplied Context Record.

Rules:
- Do not design technical components yet.
- Every feature must have a stable ID in the format FEAT-001, FEAT-002, etc.
- The supplied <requirement_catalog> is the ONLY valid source of requirement
  IDs. `related_requirement_ids` MUST contain IDs from that catalog exactly
  as shown (e.g. "FR-001", "NFR-002") — never an invented ID, and never a
  copied label such as "REQ-1" or the requirement's own prose text.
- Every requirement in <requirement_catalog> must be covered by at least one
  feature's related_requirement_ids before you return. Check the full
  catalog before returning — do not silently omit an entry (for example
  "Product reviews") just because it is smaller than the others.
- One feature may cover multiple requirement IDs, and one requirement may be
  covered by multiple features.
- Use priority values only from: must, should, could.
- Each feature needs a concrete scenario and observable acceptance criteria.
- Do not invent requirements that are unsupported by the context.
- Return only the structured output requested by the response schema.
""".strip()


ARCHITECTURE_SYSTEM_PROMPT = """
You are the architecture-design phase of an AI solution architect.

Design a justified architecture from:
- the frozen Context Record,
- the features derived in phase 1,
- the repository representation,
- retrieved architecture knowledge,
- and any Reviewer refinement instruction.

Rules:
- Detect and address the architectural flaw described by the context or repository.
- The Blueprint must contain both stakeholder and technical views.
- Every component must reference at least one related feature ID.
- Every feature must be implemented: each FEAT-nnn must appear in at least one
  component's related feature IDs, and in the Blueprint's addressed feature IDs.
  Check the full feature list before returning — a feature with no
  implementing component is an incomplete design.
- Every component must reference at least one related ADR ID where a major
  technical choice justifies it.
- ADR titles must follow the exact format: ADR-<number>: <decision>.
- Every ADR must reference the affected feature IDs and component names.
- Use retrieved knowledge as supporting evidence, but do not copy it blindly.
- Address cloud, budget, scalability, compliance, and migration constraints
  whenever they are present.
- Clearly label assumptions and open risks.

SOURCE PROVENANCE (strict):
- `source_references` on an ADR is PROVENANCE, not prose: it names the
  evidence this design was actually given. Use ONLY source identifiers that
  appear verbatim among the supplied <retrieved_knowledge> chunk sources or
  the <repository_representation> content — copy them exactly.
- NEVER invent descriptive source labels ("internal best practices",
  "industry best practices", "architecture best practices" and the like are
  fabrications unless that string is literally a supplied source).
- If no supplied evidence supports a source reference, leave
  `source_references` EMPTY rather than fabricating one. Reasoning belongs
  in the ADR's `rationale` — a decision justified by user requirements,
  Context Record facts, or general architectural reasoning needs no source
  and must not dress that reasoning up as a cited one.
- Invalid source references are removed deterministically after you return;
  a removed reference never becomes evidence.

QUANTITATIVE TARGETS (strict):
- Do not invent numeric SLOs, throughput figures, latency budgets (e.g.
  "500ms", "sub-second"), availability percentages, or absolute guarantees
  such as "zero request loss" or "zero downtime" unless the approved
  Context Record or supplied evidence states that number explicitly.
- When the Context Record gives only a qualitative signal (for example
  "substantially higher peak traffic", "high availability", "responsive
  interactions", with the exact magnitude stated as unknown), write
  qualitative acceptance criteria or state explicitly that the target must
  be measured or agreed later. A precise-sounding fabricated figure is
  worse than an honest qualitative one.

PATTERN-NAME CONSISTENCY (strict):
- The Blueprint's `selected_pattern` label and the technical view,
  components, and migration steps must all describe the SAME architectural
  direction. Do not label the pattern "Modular Monolith" while the
  technical view or components describe extracting independently
  deployable services, or label it a services/microservices pattern while
  keeping everything inside one deployable monolith. If the design is a
  staged migration (e.g. Strangler Fig) from one to the other, the pattern
  label must name the transition, not just its starting or ending state.

SERVICE-BOUNDARY DISCIPLINE (brownfield especially):
- Existing application modules, packages and apps are evidence about
  CAPABILITIES, not automatic service boundaries. Never map one
  module/package/app to one microservice.
- Recommend a standalone service only when separation has concrete support
  from at least one driver: independent scaling characteristics; a distinct
  transactional/data-consistency boundary; independent deployment or change
  cadence; clear team/domain ownership; security or compliance isolation;
  fault isolation; an independently evolving business capability; clearly
  separate data ownership; materially different availability/performance
  needs.
- When those drivers are absent, group tightly related capabilities into a
  cohesive bounded context instead. Prefer the SMALLEST decomposition that
  satisfies the stated requirements — operational complexity is a real
  cost, and every standalone service must carry a concise boundary
  rationale in its description or the relevant ADR.
- Microservices are not the assumed target. Retaining a modular monolith,
  partial extraction, or fewer services is fully valid when the constraints
  fit it better.

TECHNOLOGY CONSERVATISM (brownfield especially):
- Existing technologies remain the DEFAULT. When a
  <detected_existing_stack> block is supplied, apply this rule to exactly
  those named technologies. Change a technology from the
  detected stack only when a concrete requirement, repository-evidenced
  problem, or architecture trade-off justifies it: a scaling need the
  current stack cannot meet cleanly; a security/compliance requirement; a
  managed-service benefit that materially reduces risk; a data-access
  pattern incompatible with the current store; an availability requirement;
  an organisational/platform constraint.
- "Microservices are usually written in X", novelty, and generic best
  practice are NOT valid reasons to swap a language, framework, or data
  store. For every substantial technology change, state the justification
  tied to a requirement, repository evidence, or trade-off in the relevant
  ADR. Polyglot architecture is allowed when justified — never merely
  fashionable.

MIGRATION SEQUENCE (brownfield modernization):
- When this is a brownfield modernization, populate the Blueprint's
  `migration_steps` as an ordered, architecture-level sequence grounded in
  the repository analysis, the target architecture, the stated constraints
  (including downtime tolerance and data-consistency needs) and the
  identified dependencies: current-state preparation/seams; the first
  extraction or change and why it is first; coexistence/routing strategy;
  data transition and ownership; later steps; target-state completion and
  validation. Each step needs a title and objective at minimum.
- Do not fabricate operational detail the repo and requirements do not
  support. For greenfield or designs with no migration objective, leave
  `migration_steps` empty.

MIGRATION DISPOSITION (brownfield):
- Every new internal target service/component that does not already exist
  in the legacy system must carry an explicit migration disposition in the
  design: extracted in a named migration step, explicitly deferred to a
  later phase, explicitly retained in the legacy system for now, or
  external/already existing. A target architecture must never silently
  introduce a service with no stated migration path.

DATA-FLOW FIDELITY (Blueprint `data_flows`):
- `data_flows` is the target architecture's communication map and feeds a
  deterministic diagram: a reader of the flows must see the SAME
  architecture a reader of the full artifacts sees.
- Use concrete Blueprint component names as flow participants whenever the
  participant is an architecture component — spelled EXACTLY as in the
  Blueprint's `components`, with no parenthetical descriptors or suffixes
  appended. Do NOT collapse multiple
  components into placeholder or grouping nodes (e.g. "Backend Services",
  "Downstream Systems", "Legacy Monolith | New Microservices"). If one
  interaction targets several real components, write a separate flow per
  component (or otherwise name each owning component explicitly).
- Every non-external Blueprint component must appear in at least one
  meaningful flow; if a component genuinely has no runtime interaction,
  state that exception explicitly rather than silently omitting it.
- Event producers and consumers must have concrete named owners — never
  unnamed aggregate labels for who receives or emits an event.
- Keep flows architecture-level (services, gateways, stores, actors), not
  method-level implementation noise. Preserve external actors and external
  systems as explicit external participants where appropriate.

- When <current_design> is present you are REVISING that design, not creating a
  new one. Keep component names, ADR IDs, the selected pattern, and every
  decision the findings do not mention EXACTLY as they are. Change only what the
  refinement instruction requires. Renaming or replacing parts the review did
  not object to discards work that was already correct.
- That rule yields to the findings, it does not override them. If a finding
  requires a structural change to the architecture itself, make it and say so.
  Never preserve a design the review says is wrong.
- It also yields to <user_directive>. Anything the user names there is a licence
  to change: revise it as instructed even though no finding mentions it.
  Everything the user did NOT mention still keeps its names, IDs and decisions.
- <user_directive> is the human who owns this system speaking, and it OUTRANKS
  <refinement_instruction>. Follow both wherever they are compatible; where they
  conflict, do what the user asked and say so in the revision_note.
- If a <user_directive> cannot be implemented AS STATED, do BOTH of these and
  neither one alone: build the closest feasible variant of what they asked for,
  AND fill `directive_objection` with what was asked, why it does not work as
  stated, and what you built instead. Substituting something else silently is
  the one response that is not allowed — they must be able to see that the
  design deviates from their instruction, and why.
  Leave `directive_objection` EMPTY whenever you did implement the directive as
  stated, and whenever there is no directive. It is not the place for caveats,
  trade-offs or things you would have preferred; it is only for an instruction
  you could not carry out as written. One objection, then proceed — you are not
  being asked to negotiate, and the directive still stands.
- On a revision, set the Blueprint's revision_note to a brief statement of what
  you changed and why. Leave it empty on the initial design.
- Return only the structured output requested by the response schema.
""".strip()


def _requirement_catalog_block(state: ArchitectState) -> str:
    """Render `<requirement_catalog>` — the ONE deterministic ID scheme the
    Architect prompt shows and the writer boundary validates output against
    (see `_normalize_and_validate_requirement_coverage`), reusing
    `review_checks.requirement_catalog` rather than re-deriving IDs here."""

    from pipeline.review_checks import requirement_catalog

    catalog = requirement_catalog(state.context_record)
    lines = "\n".join(f"{entry.id}: {entry.text}" for entry in catalog)
    return f"<requirement_catalog>\n{lines}\n</requirement_catalog>"


def _build_feature_prompt(state: ArchitectState) -> str:
    if state.context_record is None:
        raise ValueError("Architect requires a Context Record before feature design.")

    return f"""
<context_record>
{state.context_record.model_dump_json(indent=2)}
</context_record>

{_requirement_catalog_block(state)}

Derive the complete feature set before any architecture is designed. Map
every <requirement_catalog> entry to at least one feature's
related_requirement_ids, using ONLY the IDs shown there.
""".strip()


def _normalize_and_validate_requirement_coverage(
    state: ArchitectState, features: list[Feature]
) -> list[Feature]:
    """Normalize phase-1 `related_requirement_ids` against the deterministic
    requirement catalog and validate full coverage — BEFORE phase 2 ever
    runs. Pure; `features` is copied, never mutated.

    WHY THIS IS FAIL-FAST, NOT A REVIEWER REPAIR TASK: a refine pass reuses
    `state.features` verbatim and never re-runs phase 1, and phase 2 cannot
    create or edit Feature objects at all. So a bad requirement mapping that
    reached the Reviewer as a finding would be a PERMANENT, unrepairable
    blocker — the refine loop has no move that closes it. Catching it here,
    before phase 2 is even called, means the run fails once with an
    actionable error instead of looping forever on a gap it can never fix.

    Compatibility normalization: a value already equal to a catalog ID is
    kept as-is; a value equal (whitespace-normalised) to a catalog entry's
    exact TEXT is converted to that entry's ID — robustness for prompt
    behaviour before this catalog contract existed. No fuzzy matching, no
    inference from feature prose: anything else is left untouched and
    counted as unknown below.

    Raises ValueError listing every unknown ID referenced and every catalog
    requirement no feature covers, so the error is actionable on its own.
    """

    from pipeline.review_checks import requirement_catalog

    catalog = requirement_catalog(state.context_record)
    if not catalog:
        return features

    id_by_text = {entry.text: entry.id for entry in catalog}
    catalog_ids = {entry.id for entry in catalog}

    normalized: list[Feature] = []
    unknown: set[str] = set()
    for feature in features:
        new_ids: list[str] = []
        for value in feature.related_requirement_ids:
            stripped = value.strip()
            if not stripped:
                continue
            if stripped in catalog_ids:
                new_ids.append(stripped)
            elif stripped in id_by_text:
                new_ids.append(id_by_text[stripped])
            else:
                new_ids.append(stripped)
                unknown.add(stripped)
        if new_ids != feature.related_requirement_ids:
            feature = feature.model_copy(update={"related_requirement_ids": new_ids})
        normalized.append(feature)

    covered_ids = {
        value
        for feature in normalized
        for value in feature.related_requirement_ids
        if value in catalog_ids
    }
    missing = [entry for entry in catalog if entry.id not in covered_ids]

    if unknown or missing:
        problems: list[str] = []
        if missing:
            problems.append(
                "uncovered requirement(s): "
                + ", ".join(f"{entry.id} - {entry.text}" for entry in missing)
            )
        if unknown:
            problems.append(
                "unknown requirement ID(s) referenced: " + ", ".join(sorted(unknown))
            )
        raise ValueError(
            "Architect phase-1 requirement-catalog coverage is invalid: "
            + "; ".join(problems)
        )

    return normalized


def _build_architecture_prompt(
    state: ArchitectState,
    features: list[Feature],
) -> str:
    context_json = (
        state.context_record.model_dump_json(indent=2)
        if state.context_record is not None
        else "null"
    )

    repository_json = (
        state.repo_representation.model_dump_json(indent=2)
        if state.repo_representation is not None
        else "null"
    )

    knowledge_json = (
        "[\n"
        + ",\n".join(
            chunk.model_dump_json(indent=2)
            for chunk in state.retrieved_knowledge
        )
        + "\n]"
        if state.retrieved_knowledge
        else "[]"
    )

    features_json = (
        "[\n"
        + ",\n".join(feature.model_dump_json(indent=2) for feature in features)
        + "\n]"
    )

    refinement_instruction = ""
    if state.review is not None and state.review.requires_refinement:
        refinement_instruction = state.review.refinement_instruction

    # The two instruction blocks are SEPARATE and stay that way. Folding the
    # user's words into `review.refinement_instruction` would be the cheaper
    # edit and would destroy the one thing the trail exists to answer: who asked
    # for this change? That field has a single writer — the reviewer — so
    # everything in it is a finding, and everything here is a person.
    #
    # `.text` AND NOTHING ELSE. Never `model_dump_json()`, however convenient.
    # A `UserFeedback` entry also carries `objection` — this agent's own account
    # of why it could not build a previous directive — one field away from the
    # text being serialised here. Reading that back would teach the architect
    # that objecting makes a directive go away, which is the same self-advocacy
    # leak `_format_artifacts` keeps `revision_note` out of the reviewer for.
    # There is a test that asserts it, including for an entry that carries an
    # objection and is still pending.
    directives = state.pending_feedback("design")
    user_directive = "\n\n".join(entry.text for entry in directives)

    current_design = _format_current_design(state)
    no_instruction = (
        "None — revise the design above only as <user_directive> asks."
        if current_design
        else "None — create the initial architecture design."
    )

    detected_stack = _format_detected_stack(state)
    refinement_discipline = _REFINEMENT_DISCIPLINE if current_design else ""

    return f"""
<context_record>
{context_json}
</context_record>

<repository_representation>
{repository_json}
</repository_representation>
{detected_stack}
<retrieved_knowledge>
{knowledge_json}
</retrieved_knowledge>

<derived_features>
{features_json}
</derived_features>

{current_design}<user_directive>
{user_directive or "None — no direction from the user this pass."}
</user_directive>

<refinement_instruction>
{refinement_instruction or no_instruction}
</refinement_instruction>
{refinement_discipline}
Create the structured Architecture Blueprint, ADRs, and Component Descriptions.
""".strip()


def _format_detected_stack(state: ArchitectState) -> str:
    """The run's ACTUAL detected technology stack as an explicit anchor.

    The TECHNOLOGY CONSERVATISM rules in the system prompt are generic ("the
    detected existing stack"); this block names it. Derived entirely from
    `repo_representation.structure.tech_stack` — languages by LOC plus
    frameworks — with nothing invented and nothing hard-coded (run
    20260822T160643Z-8e49d732 chose Go over a detected Django/React stack
    while the guidance stayed abstract). Empty for greenfield runs, where
    there is no existing stack to preserve and the block is omitted rather
    than fabricated. Pure.
    """
    repo = state.repo_representation
    if repo is None:
        return ""
    stack = repo.structure.tech_stack
    languages = sorted(
        stack.languages.items(), key=lambda item: -item[1]
    ) if stack.languages else []
    lines = []
    if languages:
        lines.append(
            "Languages: "
            + ", ".join(f"{name} ({loc:,} LOC)" for name, loc in languages)
        )
    if stack.frameworks:
        lines.append("Frameworks: " + ", ".join(stack.frameworks))
    if not lines:
        # A repo was analysed but recognised no language/framework — say
        # only that, and invent nothing.
        lines.append(
            "No specific language or framework was identified in the "
            "repository analysis."
        )
    return (
        "\n<detected_existing_stack>\n"
        "The repository analysis detected this existing technology stack:\n"
        + "\n".join(lines)
        + "\nPreserve these technologies by default. If you propose a "
        "substantial change to any of them, the relevant ADR must explain "
        "why the existing technology is insufficient for a stated "
        "requirement — a generic claim such as 'better scalability' is not "
        "enough without requirement- or repository-linked evidence.\n"
        "</detected_existing_stack>\n"
    )


# REFINEMENT-TURN DISCIPLINE. Appended to the built prompt ONLY when a
# <current_design> block exists (the initial-design prompt never sees it).
# Forensic background (run 20260822T160643Z-8e49d732, refine 2): the model
# attempted every fix but degraded structurally — 19 required ADR/component
# fields came back empty and all technology choices were dropped, the round
# scored WORSE, and the best-round restore discarded the attempted fixes.
# Pydantic defaults absorb omitted fields silently, so this checklist makes
# full re-emission an explicit instruction rather than an assumption.
_REFINEMENT_DISCIPLINE = """
<refinement_discipline>
This is a REFINEMENT pass. Work blocker by blocker:
1. Address EVERY supplied HIGH/blocking finding in the <refinement_instruction>, one by one.
2. Preserve everything already correct in the current design — make the SMALLEST changes that resolve the blockers.
3. Before returning, verify each supplied blocker against your revised artifacts; if one cannot be resolved, say so in the relevant rationale rather than silently dropping it.
4. Re-emit ALL required fields for EVERY ADR and Component Description in full — context, rationale, alternatives, consequences, inputs/outputs/dependencies, technology choices — even for parts you did not change. Never leave a required field blank merely because it was unchanged.
5. Preserve existing technology choices, migration steps, and component details unless you are intentionally changing them to resolve a finding. Do not drop them due to response compression.
</refinement_discipline>
""".strip() + "\n\n"


def _format_current_design(state: ArchitectState) -> str:
    """The previous round's artifacts, for a refine pass to revise. Pure.

    Returns "" when the block does not belong in the prompt, so the caller can
    interpolate it unconditionally. Three reasons it can be absent:

    * this is the INITIAL design, so there is nothing to revise;
    * a user correction superseded the Context Record, so the design that exists
      answers a question that has changed and is not a thing to revise. The
      cleared `state.features` is what says so - see pipeline/user_feedback.py; or
    * a refine pass somehow has no artifacts to show. That is a bug upstream -
      the reviewer only reaches REFINING after judging a design - but the
      response is to omit the block and let the architect build from the
      context, exactly as `_reuses_features` re-derives features rather than
      failing the run over an inconsistency it did not create.

    Emitting EMPTY tags instead would be worse than omitting them: it reads as
    "the previous design was blank" rather than "there was not one".
    """

    if not _reuses_features(state):
        return ""
    if state.blueprint is None or not state.adrs or not state.components:
        return ""

    blueprint_json = state.blueprint.model_dump_json(indent=2)
    adr_json = "[\n" + ",\n".join(
        adr.model_dump_json(indent=2) for adr in state.adrs
    ) + "\n]"
    component_json = "[\n" + ",\n".join(
        component.model_dump_json(indent=2) for component in state.components
    ) + "\n]"
    return (
        "<current_design>\n"
        f"BLUEPRINT:\n{blueprint_json}\n\n"
        f"ADRS:\n{adr_json}\n\n"
        f"COMPONENTS:\n{component_json}\n"
        "</current_design>\n\n"
    )


def sanitize_adr_sources(
    adrs: list[ADR], state: ArchitectState
) -> tuple[list[ADR], list[str]]:
    """Drop ADR `source_references` that do not resolve to supplied evidence.

    THE OUTPUT BOUNDARY OF PROVENANCE: the prompt forbids fabricated source
    labels, and the Reviewer would eventually flag them — but between the
    two sits the stored artifact, and provenance that is wrong should never
    be written to it. This runs on EVERY phase-2 result (initial design and
    every refine pass), so an invented name cannot enter the run even on a
    revision.

    Resolution uses `review_checks.resolve_source_reference` — the SAME
    rule the Reviewer judges by, imported rather than reimplemented, so
    writer and judge cannot drift apart.

    Behavior for an unresolvable reference (chosen small and safe): REMOVE
    it. No guessed substitute, no repair model call, no crash — the ADR
    simply stands on its own reasoning with no claimed source, which is the
    honest state for a requirement-derived decision. Removed names are
    returned so the caller can record the fact in the run trace.

    Pure with respect to `state`; ADRs are copied, never mutated.
    """
    from pipeline.review_checks import resolve_source_reference

    kb_sources = {
        chunk.source.strip().replace("\\", "/").lower()
        for chunk in state.retrieved_knowledge
        if chunk.source.strip()
    }
    repository_text = (
        state.repo_representation.model_dump_json().lower()
        if state.repo_representation is not None
        else ""
    )

    sanitized: list[ADR] = []
    removed: list[str] = []
    for adr in adrs:
        kept = [
            reference
            for reference in adr.source_references
            if reference.strip()
            and resolve_source_reference(reference, kb_sources, repository_text)
        ]
        # Blank entries are dropped as noise but not reported as removals.
        dropped = [
            reference
            for reference in adr.source_references
            if reference not in kept and reference.strip()
        ]
        removed.extend(dropped)
        sanitized.append(
            adr if not dropped and len(kept) == len(adr.source_references)
            else adr.model_copy(update={"source_references": kept})
        )
    return sanitized, removed


def canonicalize_data_flow_endpoints(
    flows: list[str], component_names: list[str]
) -> tuple[list[str], int]:
    """Resolve decorated flow endpoints back to exact Blueprint component
    names. Pure.

    Root cause this fixes (run 20260822T164736Z-1a15e2e4): the model writes
    "Order Service (New order flow)" against a component named
    "Order Service". The string parses as an endpoint, so the deterministic
    diagram draws a PHANTOM duplicate node beside the real one.

    Rule: strip ONE trailing parenthetical descriptor, and map the remaining
    text to a component ONLY when it matches one component exactly, or is a
    proper PREFIX of exactly one component's name (case-insensitive,
    whitespace-collapsed). Anything ambiguous — a prefix of two components,
    an empty remainder, a non-matching name — is left UNCHANGED: an
    uncanonicalized endpoint is honest; a guessed one is corrupt. External
    actors ("Client") and genuinely external systems never match the
    component catalog and therefore pass through untouched. Exact component
    names are fixed points of the rule.

    Returns (canonical_flows, number_of_endpoints_resolved).
    """
    catalog: list[str] = [name.strip() for name in component_names if name.strip()]

    def resolve(endpoint: str) -> tuple[str, bool]:
        text = endpoint.strip()
        # One trailing parenthetical descriptor at most; nested/inner ones
        # belong to the name itself and stay.
        if text.endswith(")") and "(" in text:
            head, _, _tail = text.rpartition("(")
            candidate = head.strip()
            if candidate:
                text = candidate
        folded = " ".join(text.lower().split())
        if not folded:
            return endpoint, False
        exact = [name for name in catalog if " ".join(name.lower().split()) == folded]
        if len(exact) == 1:
            return exact[0], exact[0] != endpoint.strip()
        prefixed = [
            name for name in catalog
            if " ".join(name.lower().split()).startswith(folded)
            and folded
        ]
        if len(prefixed) == 1:
            return prefixed[0], prefixed[0] != endpoint.strip()
        return endpoint, False

    canonical: list[str] = []
    resolved_count = 0
    for flow in flows:
        text = str(flow or "").strip()
        if not text:
            canonical.append(flow)
            continue
        label = ""
        if ":" in text:
            text, _, label = text.partition(":")
            label = label.strip()
        parts = [part.strip() for part in _FLOW_SPLIT_RE.split(text)]
        out_parts: list[str] = []
        for part in parts:
            if not part:
                out_parts.append(part)
                continue
            resolved, changed = resolve(part)
            resolved_count += 1 if changed else 0
            out_parts.append(resolved)
        rebuilt = " → ".join(p for p in out_parts if p)
        if label:
            rebuilt = f"{rebuilt}: {label}"
        canonical.append(rebuilt)
    return canonical, resolved_count


def _is_revision(state: ArchitectState) -> bool:
    """Is this pass REVISING an existing design, rather than creating one?

    Two ways to be, and neither is a judgment about the text:

      * the reviewer failed the design and asked for changes; or
      * the human sent a design directive from the finished-run screen. This
        one matters most when the review PASSED: the run reached DONE cleanly,
        the person read the design and asked for one thing to change, and
        without this the architect would treat their directive as a blank-page
        brief and rebuild from the Context Record — throwing away the very
        design they were commenting on. A directive is a request to change one
        thing, which presupposes keeping the rest.
    """
    return (
        (state.review is not None and state.review.requires_refinement)
        or bool(state.pending_feedback("design"))
    )


def _reuses_features(state: ArchitectState) -> bool:
    """Is this a revision that already has a feature set to keep? Pure code.

    Two conditions, and the second one is the defensive half. A revision pass
    with an EMPTY `state.features` is normally a bug upstream — the reviewer only
    reaches `REFINING` after a design that had features — but the response to
    that is to re-derive them and carry on, not to fail the run over an
    inconsistency this node did not create.

    It is also the deliberate signal on one path: a requirements correction
    clears `state.features` precisely BECAUSE the record they were derived from
    has been superseded (see pipeline/user_feedback.py). Re-deriving them is then
    the correct answer, not a fallback.
    """
    return _is_revision(state) and bool(state.features)


@node("architect")
def architect_node(state: ArchitectState) -> dict:
    """Run feature derivation first, then architecture design.

    TWO LLM calls on the INITIAL design, ONE on a refine pass — see the module
    docstring for why phase 1 is skipped when refining. Either way this node
    reports the SUM of the calls it made, so `usages` is what the step and the
    returned totals are built from rather than a hard-coded count of two.

    `usages` collects each call as it happens rather than at the end, because
    every validation below can raise between the two calls — the except clause
    then hands the already-billed phase-1 tokens to `@node` instead of losing
    them. See pipeline/llm.py for the mechanism.
    """
    usages: list[LLMUsage] = []
    try:
        reuse_features = _reuses_features(state)
        # Read BEFORE the calls, marked applied only after they all succeed. A
        # pass that raises leaves the directive `pending`, so the next architect
        # pass still carries it — text the human typed is never consumed by a
        # design that does not exist.
        directives = state.pending_feedback("design")

        if reuse_features:
            # Phase 1 SKIPPED. The IDs the reviewer's instruction names have to
            # still exist when the architect acts on it, and the only way to
            # guarantee that is not to regenerate them.
            features = list(state.features)
        else:
            # Phase 1 — feature-first design
            feature_result: FeatureDesign
            feature_result, phase1_usage = llm_call(
                state,
                _build_feature_prompt(state),
                system=FEATURE_SYSTEM_PROMPT,
                model=role_model_override("architect", ARCHITECT_MODEL),
                response_schema=FeatureDesign,
            )
            usages.append(phase1_usage)

            # Only meaningful when phase 1 actually ran: on a refine pass the
            # features come from a phase 1 that already passed this check.
            if not feature_result.features:
                raise ValueError("Architect produced no features.")
            features = feature_result.features

            # FAIL FAST, BEFORE PHASE 2: see
            # `_normalize_and_validate_requirement_coverage` for why a bad
            # requirement mapping must never reach the Reviewer at all.
            features = _normalize_and_validate_requirement_coverage(state, features)

        # Phase 2 — architecture derived from those features
        design_result: ArchitectureDesign
        design_result, phase2_usage = llm_call(
            state,
            _build_architecture_prompt(state, features),
            system=ARCHITECTURE_SYSTEM_PROMPT,
            model=role_model_override("architect", ARCHITECT_MODEL),
            response_schema=ArchitectureDesign,
        )
        usages.append(phase2_usage)

        if not design_result.adrs:
            raise ValueError("Architect produced no ADRs.")

        if not design_result.components:
            raise ValueError("Architect produced no Component Descriptions.")

        # PROVENANCE AT THE BOUNDARY: drop any ADR source reference that does
        # not resolve to evidence this run actually supplied (see
        # `sanitize_adr_sources`). Runs on the initial design AND on every
        # refine pass, so an invented name cannot (re-)enter the artifacts.
        design_result.adrs, dropped_sources = sanitize_adr_sources(
            design_result.adrs, state
        )

        # DIAGRAM FIDELITY AT THE BOUNDARY: resolve decorated flow endpoints
        # ("Order Service (New order flow)") back to exact Blueprint
        # component names, so the deterministic diagram draws one node per
        # component instead of phantom duplicates. Ambiguous endpoints are
        # left unchanged by design (see `canonicalize_data_flow_endpoints`).
        blueprint = design_result.blueprint
        canonical_flows, resolved_endpoints = canonicalize_data_flow_endpoints(
            blueprint.data_flows, blueprint.components
        )
        if canonical_flows != blueprint.data_flows:
            blueprint = blueprint.model_copy(
                update={"data_flows": canonical_flows}
            )

        # An objection is only meaningful about a directive that was actually in
        # this prompt. With no directive it is the model volunteering a caveat
        # it was told not to write, and there is no entry to record it against —
        # so it is dropped rather than parked somewhere a reader would mistake
        # for a response to a human.
        objection = design_result.directive_objection.strip() if directives else ""

        usage = sum_usage(usages)  # every phase THIS pass ran, as one node total
        verb = "reused" if reuse_features else "derived"
        note = (
            f"{verb} {len(features)} feature(s); "
            f"generated blueprint, {len(design_result.adrs)} ADR(s), "
            f"and {len(design_result.components)} component(s)"
        )
        if directives:
            note += f"; applied {len(directives)} user directive(s)"
        if dropped_sources:
            # The deterministic diagnostic: what was removed, by name, in the
            # run's own trail — never silently, and never with source bodies.
            note += (
                f"; removed {len(dropped_sources)} unresolvable ADR "
                f"source reference(s): " + "; ".join(dropped_sources[:4])
                + ("; …" if len(dropped_sources) > 4 else "")
            )
        if resolved_endpoints:
            note += f"; canonicalized {resolved_endpoints} data-flow endpoint(s)"
        if objection:
            # In the trace as well as on the entry: "the design deviates from
            # what was asked" is a fact about the RUN, and the run's trail is
            # where a reader looks for what happened when.
            note += f"; objected to a directive: {objection}"
        step = make_step(
            "architect",
            state.stage,
            Stage.DESIGNING,
            note,
            usage,
        )

        update = {
            "blueprint": blueprint,
            "adrs": design_result.adrs,
            "components": design_result.components,
            "stage": Stage.DESIGNING,
            "history": [step],
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        }
        if not reuse_features:
            # Deliberately ABSENT on a refine pass. `features` is a plain
            # LastValue channel, so leaving the key out is what keeps the
            # existing list — writing it back would be a no-op today and a trap
            # the day anything in this function starts copying or re-ordering it.
            update["features"] = features
        if directives:
            # The receipt for the directive, written on the same update as the
            # design that carries it out. Also a LastValue channel, so this is
            # the whole list with those entries flipped — see
            # `ArchitectState.feedback_marked_applied`.
            #
            # An objection rides on the SAME update and marks its entry
            # `objected` rather than `applied`. It is a voice, not a veto: the
            # design still ships, the pass still succeeds, and the run carries
            # on. The stage below is DESIGNING either way.
            update["user_feedback"] = state.feedback_marked_applied(
                directives, objection=objection
            )
        return update
    except Exception as e:
        if usages:
            attach_usage(e, sum_usage(usages))
        raise