"""Offline unit tests for the Architect Agent.

No API key or network is required. The two LLM phases are replaced with
deterministic structured responses.

FEATURE STABILITY ACROSS THE REFINE LOOP
----------------------------------------
The second block of tests below is about one thing: on a refine pass the
architect must REUSE `state.features` rather than re-derive them. That is what
makes a reviewer instruction like "assign FEAT-005 to a component" still refer
to something real on the next pass — measured on run
20260818T074835Z-1925fbd7, where phase 1 produced 5 features, then 4, then 5,
and the loop spent both iterations chasing an ID that kept moving.

Those tests count PHASE 1 CALLS, not tokens or flags, because the property is
"the model is not asked to invent features again" and only a call count can say
that.
"""

from __future__ import annotations

import pytest

import test_clarifier as tc  # reuse the shared canned-usage helper
from pipeline.agents import architect as arch
from pipeline.state import (
    ADR,
    Blueprint,
    ComponentDescription,
    ContextRecord,
    Feature,
    KBChunk,
    MigrationStep,
    RepoRepresentation,
    ReviewResult,
    Stage,
    new_run,
)


def _fake_architect_llm(
    state,
    prompt,
    *,
    system="",
    model="",
    response_schema=None,
    thinking_level=None,
):
    """Return canned outputs for both Architect phases, as `(reply, usage)`."""

    if response_schema is arch.FeatureDesign:
        return (arch.FeatureDesign(
            features=[
                Feature(
                    id="FEAT-001",
                    name="Handle peak load",
                    description="Support high traffic during sale events.",
                    scenario="The platform remains responsive for 50k peak users.",
                    related_requirement_ids=["FR-001", "NFR-001"],
                    priority="must",
                    acceptance_criteria=[
                        "The platform stays available under peak load.",
                    ],
                )
            ]
        ), tc.fake_usage())

    if response_schema is arch.ArchitectureDesign:
        return (arch.ArchitectureDesign(
            blueprint=Blueprint(
                project_name="E-commerce Platform",
                selected_pattern="Event-Driven Microservices Architecture",
                rationale="Independent scaling reduces monolith bottlenecks.",
                stakeholder_view="Customers receive a reliable shopping experience.",
                technical_view=(
                    "An AWS-hosted frontend and API Gateway connect to "
                    "independently scalable services. Events decouple order "
                    "processing from the legacy monolith."
                ),
                components=["Frontend", "API Gateway", "Order Service"],
                data_flows=[
                    "Frontend sends order requests through the API Gateway.",
                    "Order Service emits order events.",
                ],
                addressed_feature_ids=["FEAT-001"],
                constraints_addressed=[
                    "AWS",
                    "medium budget",
                    "50k peak users",
                    "GDPR",
                    "legacy monolith migration",
                ],
            ),
            adrs=[
                ADR(
                    id="ADR-001",
                    title="ADR-1: Separate order processing from the monolith",
                    context="The monolith fails under peak load.",
                    decision="Introduce an independently scalable Order Service.",
                    rationale="Order processing can scale separately.",
                    alternatives_considered=[
                        "Scale the whole monolith vertically",
                    ],
                    positive_consequences=[
                        "Independent scaling",
                        "Improved fault isolation",
                    ],
                    negative_consequences=[
                        "Higher operational complexity",
                    ],
                    related_feature_ids=["FEAT-001"],
                    related_component_names=["Order Service"],
                    source_references=["Architecture KB"],
                )
            ],
            components=[
                ComponentDescription(
                    id="COMP-001",
                    name="Order Service",
                    component_type="service",
                    purpose="Owns and processes customer orders.",
                    description=(
                        "Implements FEAT-001 and is justified by ADR-001."
                    ),
                    inputs=["Validated order requests"],
                    outputs=["Order confirmation events"],
                    dependencies=["API Gateway"],
                    related_feature_ids=["FEAT-001"],
                    related_adr_ids=["ADR-001"],
                    technology_choices=["AWS managed compute"],
                    security_considerations=["Encryption in transit"],
                    scalability_considerations=["Horizontal scaling"],
                )
            ],
        ), tc.fake_usage())

    raise AssertionError(f"Unexpected response schema: {response_schema}")


def _base_state():
    state = new_run("Modernize an e-commerce platform.")
    state.stage = Stage.RESEARCHING
    state.context_record = ContextRecord(
        project_name="E-commerce Platform",
        business_goal="Increase reliability during sale events.",
        problem_statement="The existing monolith fails under peak load.",
        users=["Online customers", "Operations team"],
        functional_requirements=["Process customer orders"],
        non_functional_requirements=["Support 50k peak users"],
        cloud_provider="AWS",
        budget="medium",
        compliance_requirements=["GDPR"],
        existing_systems=["Legacy monolith"],
        summary="AWS e-commerce platform with 50k peak users and GDPR constraints.",
    )
    state.repo_representation = RepoRepresentation(
        summary="Legacy monolith with tightly coupled order processing."
    )
    state.retrieved_knowledge = [
        KBChunk(
            content="Event-driven microservices support independent scaling.",
            source="Architecture KB",
        )
    ]
    return state


def test_architect_generates_all_artifacts():
    arch.llm_call = _fake_architect_llm
    state = _base_state()

    output = arch.architect_node(state)

    assert output["stage"] is Stage.DESIGNING
    assert len(output["features"]) == 1
    assert output["blueprint"] is not None
    assert len(output["adrs"]) == 1
    assert len(output["components"]) == 1
    assert output["components"][0].related_feature_ids == ["FEAT-001"]
    assert output["components"][0].related_adr_ids == ["ADR-001"]
    assert output["history"][0].agent == "architect"


def test_architect_fails_without_context_record():
    arch.llm_call = _fake_architect_llm
    state = new_run("Design a system.")
    state.stage = Stage.RESEARCHING

    output = arch.architect_node(state)

    assert output["stage"] is Stage.FAILED
    assert output["errors"]
    assert "Context Record" in output["errors"][0]


def test_architect_prompts_include_rag_and_repository_context():
    state = _base_state()

    feature_prompt = arch._build_feature_prompt(state)
    design_prompt = arch._build_architecture_prompt(
        state,
        [
            Feature(
                id="FEAT-001",
                name="Handle peak load",
                scenario="Remain responsive at 50k peak users.",
            )
        ],
    )

    assert "50k peak users" in feature_prompt
    assert "Legacy monolith" in design_prompt
    assert "Architecture KB" in design_prompt
    assert "Event-driven microservices" in design_prompt


# ══════════════════════════════════════════════════════════════════════════
# Requirement-catalog contract: no repairable-but-unrepairable Reviewer
# deadlock (final Reviewer micro-fix)
#
# The old Reviewer rule required a Feature's `related_requirement_ids` to
# name a Context Record requirement VERBATIM, but the Architect's own
# contract (and existing Architect tests) already used IDs like "NFR-001" —
# a correct Architect output could never satisfy that rule. Worse: refine
# reuses `state.features` and phase 2 cannot edit Feature objects, so a
# Reviewer HIGH for a missing mapping would be a PERMANENT blocker no refine
# pass could close. The fix is fail-fast: the Architect validates its own
# phase-1 output against a deterministic requirement catalog BEFORE phase 2
# ever runs, so a bad mapping fails the run once, with an actionable error,
# instead of reaching the Reviewer as an unrepairable finding.
# ══════════════════════════════════════════════════════════════════════════


def test_feature_prompt_includes_the_requirement_catalog():
    """Item 5. `_base_state()` states functional=["Process customer
    orders"], non_functional=["Support 50k peak users"]."""
    prompt = arch._build_feature_prompt(_base_state())

    assert "<requirement_catalog>" in prompt
    assert "FR-001: Process customer orders" in prompt
    assert "NFR-001: Support 50k peak users" in prompt


def test_normalize_accepts_canonical_catalog_ids_unchanged():
    """Item 6."""
    features = [
        Feature(
            id="FEAT-001", name="Orders", scenario="s",
            related_requirement_ids=["FR-001", "NFR-001"],
        ),
    ]

    normalized = arch._normalize_and_validate_requirement_coverage(
        _base_state(), features
    )

    assert normalized[0].related_requirement_ids == ["FR-001", "NFR-001"]


def test_normalize_converts_exact_raw_text_to_canonical_id():
    """Item 7: robustness for prompt behaviour predating the catalog
    contract — never fuzzy, only an exact (whitespace-normalised) match."""
    features = [
        Feature(
            id="FEAT-001", name="Orders", scenario="s",
            related_requirement_ids=[
                "Process customer orders", "Support 50k peak users",
            ],
        ),
    ]

    normalized = arch._normalize_and_validate_requirement_coverage(
        _base_state(), features
    )

    assert normalized[0].related_requirement_ids == ["FR-001", "NFR-001"]
    # The original Feature is untouched — normalization copies, not mutates.
    assert features[0].related_requirement_ids == [
        "Process customer orders", "Support 50k peak users",
    ]


def test_normalize_rejects_unknown_synthetic_requirement_id():
    """Item 8."""
    features = [
        Feature(
            id="FEAT-001", name="Orders", scenario="s",
            related_requirement_ids=["REQ-1", "FR-001", "NFR-001"],
        ),
    ]

    with pytest.raises(ValueError, match="REQ-1"):
        arch._normalize_and_validate_requirement_coverage(_base_state(), features)


def _incomplete_coverage_llm(calls: list[str], missing: str):
    """A phase-1 stub covering only ONE of the two catalog entries `_base_state`
    declares. Phase 2 must never be reached — it raises if it is, so a passing
    test proves the fail-fast path, not merely a call-count assertion after
    the fact."""

    def stub(state, prompt, *, system="", model="", response_schema=None, thinking_level=None):
        calls.append("phase1" if response_schema is arch.FeatureDesign else "phase2")
        if response_schema is arch.FeatureDesign:
            covered = ["NFR-001"] if missing == "functional" else ["FR-001"]
            return (arch.FeatureDesign(features=[
                Feature(
                    id="FEAT-001",
                    name="Handle peak load",
                    scenario="The platform remains responsive for 50k peak users.",
                    related_requirement_ids=covered,
                    acceptance_criteria=["The platform stays available."],
                )
            ]), tc.fake_usage())
        raise AssertionError(
            "phase 2 must never be called when phase-1 coverage is invalid"
        )

    return stub


def test_omitted_functional_requirement_fails_before_phase_two():
    """Items 9, 11."""
    calls: list[str] = []
    arch.llm_call = _incomplete_coverage_llm(calls, missing="functional")

    output = arch.architect_node(_base_state())

    assert output["stage"] is Stage.FAILED
    assert calls == ["phase1"]
    assert output["errors"]
    assert "FR-001" in output["errors"][0]
    assert "Process customer orders" in output["errors"][0]


def test_omitted_non_functional_requirement_fails_before_phase_two():
    """Item 10."""
    calls: list[str] = []
    arch.llm_call = _incomplete_coverage_llm(calls, missing="non_functional")

    output = arch.architect_node(_base_state())

    assert output["stage"] is Stage.FAILED
    assert calls == ["phase1"]
    assert "NFR-001" in output["errors"][0]
    assert "Support 50k peak users" in output["errors"][0]


def test_fail_fast_preserves_phase_one_token_accounting():
    """Item 12: the already-billed phase-1 tokens survive the validation
    failure — the same `attach_usage`/`@node` guarantee every other Architect
    exception already gets, not a new mechanism."""
    calls: list[str] = []
    arch.llm_call = _incomplete_coverage_llm(calls, missing="functional")

    output = arch.architect_node(_base_state())

    assert output["input_tokens"] == tc.FAKE_INPUT_TOKENS
    assert output["output_tokens"] == tc.FAKE_OUTPUT_TOKENS
    step = output["history"][0]
    assert step.input_tokens == output["input_tokens"]
    assert step.output_tokens == output["output_tokens"]


# Items 13 and 14 (a valid initial design still makes exactly two calls; a
# normal refine pass still makes exactly one phase-2 call and skips phase 1)
# are already covered by the pre-existing convergence regression tests below
# — `test_initial_design_still_runs_both_phases` and
# `test_refine_pass_reuses_features_and_skips_phase_one` — which still pass
# unchanged after this fix (the requirement-catalog validation runs only on
# the phase-1-executed path and does not add or remove any LLM call).


# ══════════════════════════════════════════════════════════════════════════
# Refine passes reuse the feature set instead of re-deriving it
# ══════════════════════════════════════════════════════════════════════════
def _counting_llm(calls: list[str]):
    """`_fake_architect_llm`, but recording which phase each call was for."""

    def stub(state, prompt, *, system="", model="", response_schema=None, thinking_level=None):
        calls.append("phase1" if response_schema is arch.FeatureDesign else "phase2")
        return _fake_architect_llm(
            state, prompt, system=system, model=model, response_schema=response_schema
        )

    return stub


def _refining_state(features: list[Feature] | None) -> object:
    """A state as the refine gate hands it to the architect: a failing review."""
    state = _base_state()
    state.stage = Stage.REFINING
    state.review = ReviewResult(
        overall_status="fail",
        requires_refinement=True,
        refinement_instruction="Assign FEAT-005 to a component.",
    )
    state.features = list(features or [])
    return state


def _existing_features() -> list[Feature]:
    """Five features, so a re-derivation that produced four would be visible."""
    return [
        Feature(
            id=f"FEAT-{n:03d}",
            name=f"Capability {n}",
            scenario=f"Scenario {n} holds under load.",
        )
        for n in range(1, 6)
    ]


def test_refine_pass_reuses_features_and_skips_phase_one():
    """THE convergence fix. Feature IDs must survive the loop unchanged.

    The canned phase-1 response returns exactly ONE feature (FEAT-001), so if
    phase 1 ran, the five IDs seeded here would collapse to one — the assertion
    on the ID list would fail even if the call-count assertion somehow did not.
    """
    calls: list[str] = []
    arch.llm_call = _counting_llm(calls)
    before = _existing_features()
    state = _refining_state(before)

    output = arch.architect_node(state)

    assert calls == ["phase2"], f"phase 1 ran on a refine pass: {calls}"
    # The key is ABSENT, not rewritten: `features` is a LastValue channel, so
    # leaving it out is what preserves the list.
    assert "features" not in output
    assert [f.id for f in state.features] == [f.id for f in before]
    assert output["stage"] is Stage.DESIGNING
    assert output["blueprint"] is not None
    assert "reused 5 feature(s)" in output["history"][0].note


def test_initial_design_still_runs_both_phases():
    """No review yet -> the features do not exist -> phase 1 must run."""
    calls: list[str] = []
    arch.llm_call = _counting_llm(calls)

    output = arch.architect_node(_base_state())

    assert calls == ["phase1", "phase2"]
    assert [f.id for f in output["features"]] == ["FEAT-001"]
    assert "derived 1 feature(s)" in output["history"][0].note


def test_refine_with_no_features_falls_back_to_phase_one():
    """Defensive: a refine pass with an empty feature list is a bug elsewhere.

    Re-deriving is the recoverable response; failing the run over an
    inconsistency this node did not create is not.
    """
    calls: list[str] = []
    arch.llm_call = _counting_llm(calls)

    output = arch.architect_node(_refining_state([]))

    assert calls == ["phase1", "phase2"]
    assert [f.id for f in output["features"]] == ["FEAT-001"]


def test_passing_review_is_not_treated_as_a_refine_pass():
    """`requires_refinement` is the signal, not the mere presence of a review."""
    calls: list[str] = []
    arch.llm_call = _counting_llm(calls)
    state = _refining_state(_existing_features())
    state.review.requires_refinement = False

    arch.architect_node(state)

    assert calls == ["phase1", "phase2"]


def test_refine_pass_reports_one_call_of_usage():
    """Token accounting has to follow the call count, not a hard-coded two.

    The step and the returned totals are the same numbers — that is what keeps
    `usage_by_agent()` reconciled with the run totals (see
    test_token_accounting.py), and it has to stay true now that the number of
    calls per visit varies.
    """
    arch.llm_call = _counting_llm([])

    refined = arch.architect_node(_refining_state(_existing_features()))
    initial = arch.architect_node(_base_state())

    one_call_in = tc.FAKE_INPUT_TOKENS
    one_call_out = tc.FAKE_OUTPUT_TOKENS

    assert refined["input_tokens"] == one_call_in
    assert refined["output_tokens"] == one_call_out
    assert initial["input_tokens"] == 2 * one_call_in
    assert initial["output_tokens"] == 2 * one_call_out

    for output in (refined, initial):
        step = output["history"][0]
        assert step.input_tokens == output["input_tokens"]
        assert step.output_tokens == output["output_tokens"]


def test_refine_pass_still_reaches_the_architecture_prompt_with_the_instruction():
    """Reusing features must not drop the reason the pass is happening."""
    state = _refining_state(_existing_features())

    prompt = arch._build_architecture_prompt(state, state.features)

    assert "Assign FEAT-005 to a component." in prompt
    assert "FEAT-005" in prompt  # the ID it names is in the feature set it sees


# --- revising the previous design instead of regenerating it ---------------
#
# Run 20260819T072344Z-2a29f3bd: the datastore went Aurora -> SQS -> RDS MySQL
# across three rounds and component names were replaced wholesale each pass, so
# a one-line finding triggered a full redesign and broke what the previous round
# had right. `<current_design>` is what gives the architect something to revise.


def _designed_state(features: list[Feature] | None = None):
    """A refine-pass state that already carries a previous design."""

    state = _refining_state(features or _existing_features())
    state.blueprint = Blueprint(
        blueprint_id="BP-KEEP-001",
        project_name="E-commerce Platform",
        selected_pattern="Modular Monolith with Asynchronous Task Queue",
        stakeholder_view="Customers keep shopping during peak sales.",
        technical_view="Web tier offloads order writes to a queue.",
        addressed_feature_ids=["FEAT-001"],
    )
    state.adrs = [
        ADR(
            adr_id="ADR-001",
            title="ADR-001: Use Amazon SQS",
            context="Order writes saturate the database.",
            decision="Buffer order writes through Amazon SQS.",
            rationale="It decouples the write path from the request path.",
        )
    ]
    state.components = [
        ComponentDescription(
            name="Transactional Order Worker",
            description="Drains the queue and persists orders.",
            related_feature_ids=["FEAT-001"],
        ),
        ComponentDescription(
            name="Redis Caching Layer",
            description="Serves hot catalogue reads.",
            related_feature_ids=["FEAT-002"],
        ),
    ]
    return state


def test_refine_pass_shows_the_architect_its_previous_design():
    prompt = arch._build_architecture_prompt(_designed_state(), _existing_features())

    assert "<current_design>" in prompt
    assert "</current_design>" in prompt
    # The names that kept churning are exactly what must survive the round.
    assert "Transactional Order Worker" in prompt
    assert "Redis Caching Layer" in prompt
    assert "ADR-001" in prompt
    assert "Modular Monolith with Asynchronous Task Queue" in prompt


def test_initial_design_has_no_current_design_block():
    """There is nothing to revise on the first pass, and saying so with empty
    tags would read as 'the previous design was blank'."""

    state = _base_state()  # no review at all
    prompt = arch._build_architecture_prompt(state, _existing_features())

    assert "current_design" not in prompt


def test_refine_pass_without_artifacts_omits_the_block_entirely():
    state = _designed_state()
    state.blueprint = None
    state.adrs = []
    state.components = []

    prompt = arch._build_architecture_prompt(state, _existing_features())

    assert "current_design" not in prompt
    # ... and the pass still carries the reason it is happening.
    assert "Assign FEAT-005 to a component." in prompt


def test_a_partial_previous_design_omits_the_block():
    """Components present but no ADRs: still nothing coherent to revise."""

    state = _designed_state()
    state.adrs = []

    assert arch._format_current_design(state) == ""


def test_the_revision_rules_reach_the_model():
    assert "<current_design>" in arch.ARCHITECTURE_SYSTEM_PROMPT
    assert "REVISING" in arch.ARCHITECTURE_SYSTEM_PROMPT
    # The escape hatch is not optional: anchoring to the previous design
    # unconditionally would stop the loop ever fixing a wrong architecture.
    assert "yields to the findings" in arch.ARCHITECTURE_SYSTEM_PROMPT
    assert "revision_note" in arch.ARCHITECTURE_SYSTEM_PROMPT


# ─────────────────────────────────────────────────────────────────────────
# Component identity consistency during refinement (run
# 20260823T145925Z-60fb9310): the target components were named
# 'User Identity Service', 'Order Processing Service', and 'Review Service',
# but a migration step's title read "Extract User Identity, Order, and
# Review Services" — a shared-suffix shorthand that collapses three distinct
# canonical names into wording that names none of them exactly. The SAME
# step's `changes` field got the full names right, so the model already knew
# them; the shorthand only appeared in prose. The Reviewer's deterministic
# migration-target check correctly flagged the resulting 'Order Service' and
# 'User Service' as unresolved — those are genuinely not declared components
# under exact-identity matching, and that check is not touched here.
# ─────────────────────────────────────────────────────────────────────────


def _state_with_components(names: list[str], migration_steps=()):
    """A refine-pass state whose current design has exactly these component
    names, for testing the canonical-name manifest in isolation."""

    state = _designed_state()
    state.components = [
        ComponentDescription(
            name=name,
            description=f"Owns the {name} responsibility.",
            related_feature_ids=["FEAT-001"],
        )
        for name in names
    ]
    state.blueprint.migration_steps = list(migration_steps)
    return state


def test_refinement_prompt_still_carries_exact_component_names():
    """The slimmed contract: no separate manifest block, but the previous
    design's exact component names still reach the model through the
    <current_design> COMPONENTS json, alongside the identity rule."""

    state = _state_with_components(
        ["Customer Identity Service", "Order Processing Service", "Review Service"]
    )

    prompt = arch._build_architecture_prompt(state, _existing_features())

    assert "<canonical_component_names>" not in prompt
    for name in (
        "Customer Identity Service", "Order Processing Service", "Review Service",
    ):
        assert name in prompt


def test_refinement_discipline_instructs_verbatim_canonical_names():
    assert "VERBATIM" in arch.ARCHITECTURE_SYSTEM_PROMPT


def test_system_prompt_forbids_shared_suffix_shorthand():
    assert "shorthand" in arch.ARCHITECTURE_SYSTEM_PROMPT.lower()
    assert "VERBATIM" in arch.ARCHITECTURE_SYSTEM_PROMPT
    # The exact failure shape from the real run, spelled out as the bad
    # example so the instruction is concrete rather than abstract.
    assert "User Identity, Order, and Review Services" in arch.ARCHITECTURE_SYSTEM_PROMPT


def test_initial_design_has_no_canonical_component_names_block():
    """No previous component-identity catalog exists yet — nothing to freeze."""

    state = _base_state()  # no review at all, no current design
    prompt = arch._build_architecture_prompt(state, _existing_features())

    assert "canonical_component_names" not in prompt


def test_refine_pass_without_artifacts_also_omits_the_canonical_names_block():
    """Same 'nothing coherent to revise' case as the current-design block."""

    state = _designed_state()
    state.blueprint = None
    state.adrs = []
    state.components = []

    prompt = arch._build_architecture_prompt(state, _existing_features())

    assert "canonical_component_names" not in prompt


def test_real_failure_shape_migration_title_would_now_be_forbidden_by_the_prompt():
    """Pure regression fixture from run 20260823T145925Z-60fb9310 — documents
    WHY the demonstrated shorthand is unsafe, without touching the Reviewer.

    This does not call an LLM and does not assert anything about generation
    quality; it pins the two facts the fix rests on: the canonical names are
    exposed verbatim, and the exact bad title from the real run is named as
    the forbidden example so a future prompt edit cannot silently drop it.
    """
    canonical = ["User Identity Service", "Order Processing Service", "Review Service"]
    bad_title = "Extract User Identity, Order, and Review Services"
    bad_shorthand = "User Identity, Order, and Review Services"  # the shared phrase

    state = _state_with_components(
        canonical,
        migration_steps=[MigrationStep(title=bad_title, objective="Extract core domains.")],
    )
    prompt = arch._build_architecture_prompt(state, _existing_features())

    # The exact stored names still reach the model via <current_design>'s
    # COMPONENTS json — copying them verbatim cannot reproduce the
    # shared-suffix shorthand that made "Order" and "User" resolve to no
    # declared component.
    for name in canonical:
        assert name in prompt
    # The exact shorthand shape from the real run is cited as the forbidden
    # example, so a future prompt edit cannot silently drop it.
    assert bad_shorthand in arch.ARCHITECTURE_SYSTEM_PROMPT
    assert bad_shorthand in bad_title  # sanity: the fixture reuses the same phrase


# --- Architect output discipline (final run 20260822T222414Z-1788d214) -----
#
# The final run generated unsupported numeric SLOs ("500ms", "sub-second",
# "Zero request loss") the approved Context Record never stated — it only
# said "substantially higher peak traffic", "high availability", "responsive
# interactions", with the exact peak magnitude explicitly unknown — and
# labelled its pattern "Modular Monolith" while describing extraction of
# independently deployable services. These are prompt-contract regressions:
# no new LLM call, just narrow rule additions pinned by asserting they reach
# the model.


def test_requirement_coverage_rule_reaches_the_feature_model():
    assert "<requirement_catalog>" in arch.FEATURE_SYSTEM_PROMPT
    assert "must contain ids from that catalog" in arch.FEATURE_SYSTEM_PROMPT.lower()
    assert "never an invented id" in arch.FEATURE_SYSTEM_PROMPT.lower()
    assert '"req-1"' in arch.FEATURE_SYSTEM_PROMPT.lower()
    assert "do not silently omit" in arch.FEATURE_SYSTEM_PROMPT.lower()


def test_feature_coverage_rule_reaches_the_architecture_model():
    assert (
        "a feature with no\n  implementing component is an incomplete design"
        in arch.ARCHITECTURE_SYSTEM_PROMPT.lower()
    )


def test_no_invented_quantitative_targets_rule_reaches_the_model():
    assert "QUANTITATIVE TARGETS" in arch.ARCHITECTURE_SYSTEM_PROMPT
    assert "500ms" in arch.ARCHITECTURE_SYSTEM_PROMPT
    assert "zero request loss" in arch.ARCHITECTURE_SYSTEM_PROMPT.lower()
    assert "measured or agreed later" in arch.ARCHITECTURE_SYSTEM_PROMPT


def test_pattern_name_consistency_rule_reaches_the_model():
    assert "PATTERN-NAME CONSISTENCY" in arch.ARCHITECTURE_SYSTEM_PROMPT
    assert '"Modular Monolith"' in arch.ARCHITECTURE_SYSTEM_PROMPT
    assert "SAME architectural\n  direction" in arch.ARCHITECTURE_SYSTEM_PROMPT
