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
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from pipeline.agents.base import make_step, node
from pipeline.llm import LLMUsage, attach_usage, llm_call, sum_usage
from pipeline.state import (
    ADR,
    ArchitectState,
    Blueprint,
    ComponentDescription,
    Feature,
    Stage,
)


ARCHITECT_MODEL = "flash-lite"


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


FEATURE_SYSTEM_PROMPT = """
You are the feature-design phase of an AI solution architect.

Derive concrete, testable system features from the supplied Context Record.

Rules:
- Do not design technical components yet.
- Every feature must have a stable ID in the format FEAT-001, FEAT-002, etc.
- Every feature must trace to one or more requirement IDs.
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
- Return only the structured output requested by the response schema.
""".strip()


def _build_feature_prompt(state: ArchitectState) -> str:
    if state.context_record is None:
        raise ValueError("Architect requires a Context Record before feature design.")

    return f"""
<context_record>
{state.context_record.model_dump_json(indent=2)}
</context_record>

Derive the complete feature set before any architecture is designed.
""".strip()


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

    return f"""
<context_record>
{context_json}
</context_record>

<repository_representation>
{repository_json}
</repository_representation>

<retrieved_knowledge>
{knowledge_json}
</retrieved_knowledge>

<derived_features>
{features_json}
</derived_features>

<refinement_instruction>
{refinement_instruction or "None — create the initial architecture design."}
</refinement_instruction>

Create the structured Architecture Blueprint, ADRs, and Component Descriptions.
""".strip()


def _reuses_features(state: ArchitectState) -> bool:
    """Is this a refine pass that already has a feature set to keep? Pure code.

    Two conditions, and the second one is the defensive half. A refine pass with
    an EMPTY `state.features` is a bug somewhere upstream — the reviewer only
    reaches `REFINING` after a design that had features — but the response to
    that is to re-derive them and carry on, not to fail the run over an
    inconsistency this node did not create.
    """
    refining = state.review is not None and state.review.requires_refinement
    return refining and bool(state.features)


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
                model=ARCHITECT_MODEL,
                response_schema=FeatureDesign,
            )
            usages.append(phase1_usage)

            # Only meaningful when phase 1 actually ran: on a refine pass the
            # features come from a phase 1 that already passed this check.
            if not feature_result.features:
                raise ValueError("Architect produced no features.")
            features = feature_result.features

        # Phase 2 — architecture derived from those features
        design_result: ArchitectureDesign
        design_result, phase2_usage = llm_call(
            state,
            _build_architecture_prompt(state, features),
            system=ARCHITECTURE_SYSTEM_PROMPT,
            model=ARCHITECT_MODEL,
            response_schema=ArchitectureDesign,
        )
        usages.append(phase2_usage)

        if not design_result.adrs:
            raise ValueError("Architect produced no ADRs.")

        if not design_result.components:
            raise ValueError("Architect produced no Component Descriptions.")

        usage = sum_usage(usages)  # every phase THIS pass ran, as one node total
        verb = "reused" if reuse_features else "derived"
        step = make_step(
            "architect",
            state.stage,
            Stage.DESIGNING,
            (
                f"{verb} {len(features)} feature(s); "
                f"generated blueprint, {len(design_result.adrs)} ADR(s), "
                f"and {len(design_result.components)} component(s)"
            ),
            usage,
        )

        update = {
            "blueprint": design_result.blueprint,
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
        return update
    except Exception as e:
        if usages:
            attach_usage(e, sum_usage(usages))
        raise