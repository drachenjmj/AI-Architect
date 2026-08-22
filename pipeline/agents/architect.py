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
- Existing technologies remain the DEFAULT. Change a technology from the
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

{current_design}<user_directive>
{user_directive or "None — no direction from the user this pass."}
</user_directive>

<refinement_instruction>
{refinement_instruction or no_instruction}
</refinement_instruction>

Create the structured Architecture Blueprint, ADRs, and Component Descriptions.
""".strip()


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