"""review_checks.py — deterministic validation for architecture artifacts.

Everything that can be checked reliably in code is checked here before the
Reviewer LLM receives the design.

Covered rubric items:

1. Artifact completeness
2. Constraint coverage
3. Feature → Component → ADR traceability
4. ADR presence and structure
5. Source-reference integrity
6. Cross-artifact target-service ownership
7. Data-flow participant resolution (every flow endpoint is a Component or
   is clearly external)
8. Migration disposition for new internal target services (brownfield)

The checks use the frozen structured schemas owned by Maheen. A limited legacy
fallback exists only when callers explicitly enable it for old fixtures.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from pipeline.flow_syntax import split_directional_flow
from pipeline.state import ADR, ArchitectState, ContextRecord, ReviewIssue


# ─────────────────────────────────────────────────────────────────────────
# Constraint detection
# ─────────────────────────────────────────────────────────────────────────

CONSTRAINT_GROUPS = (
    "functional",
    "cloud",
    "budget",
    "non_functional",
    "compliance",
    "existing_system",
)

CONSTRAINT_ALIASES: dict[str, tuple[str, ...]] = {
    "aws": ("amazon web services",),
    "amazon web services": ("aws",),
    "gcp": ("google cloud",),
    "google cloud": ("gcp",),
    "microsoft azure": ("azure",),
    "on premises": ("on premise", "on-prem", "on-premises"),
    "on premise": ("on premises", "on-prem", "on-premises"),
}


# ADR titles must follow:
# ADR-1: Decision
# ADR-001: Decision
ADR_TITLE_RE = re.compile(r"^ADR-(\d+)\s*:\s*\S", re.IGNORECASE)


class DeterministicChecks(BaseModel):
    """Machine-checked facts and deterministic rubric scores."""

    artifacts_present: dict[str, bool] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)

    features_without_component: list[str] = Field(default_factory=list)
    components_without_feature: list[str] = Field(default_factory=list)
    components_without_adr: list[str] = Field(default_factory=list)
    blueprint_missing_feature_ids: list[str] = Field(default_factory=list)
    invalid_blueprint_feature_ids: list[str] = Field(default_factory=list)
    invalid_component_feature_ids: dict[str, list[str]] = Field(default_factory=dict)
    invalid_component_adr_ids: dict[str, list[str]] = Field(default_factory=dict)
    invalid_adr_feature_ids: dict[str, list[str]] = Field(default_factory=dict)
    invalid_adr_component_names: dict[str, list[str]] = Field(default_factory=dict)
    adrs_without_feature: list[str] = Field(default_factory=list)
    adrs_without_component: list[str] = Field(default_factory=list)

    malformed_adr_titles: list[str] = Field(default_factory=list)
    incomplete_adr_ids: list[str] = Field(default_factory=list)
    duplicate_adr_numbers: list[int] = Field(default_factory=list)

    repository_expected: bool = False
    repository_available: bool = False
    invalid_source_references: dict[str, list[str]] = Field(default_factory=dict)

    # Decision-level literature grounding (see
    # `_check_kb_evidence_grounding`). ADR ids with no `evidence_ids` at all,
    # despite qualifying evidence existing this run to cite.
    adrs_without_kb_evidence: list[str] = Field(default_factory=list)
    # ADR id -> evidence_ids that do not resolve to any curated-KB chunk
    # actually retrieved this run (fabricated, or a stripped/expired ID).
    invalid_evidence_ids: dict[str, list[str]] = Field(default_factory=dict)
    # True when NO curated-KB evidence was retrieved for any decision topic
    # this run — an honest evidence gap, not a design defect; see
    # `_check_kb_evidence_grounding` for why this does not block the verdict.
    kb_evidence_gap: bool = False

    # Material-decision coverage (see
    # `_check_material_decision_coverage`). ADR id -> related_decision_topic_ids
    # values that do not resolve to any DecisionTopic retrieved this run.
    invalid_adr_decision_topic_ids: dict[str, list[str]] = Field(default_factory=dict)
    # Material decision topic strings (see MATERIAL_DECISION_TOPICS /
    # MIGRATION_MATERIAL_TOPIC) planned this run with NO ADR mapped to them.
    material_decisions_without_adr: list[str] = Field(default_factory=list)
    # ADR id -> evidence_ids that ARE genuinely qualifying curated-KB
    # evidence this run, but were retrieved for a decision topic OTHER than
    # any this ADR maps to (see `_check_adr_evidence_topic_provenance`).
    adr_evidence_outside_mapped_topics: dict[str, list[str]] = Field(default_factory=dict)
    # ADR ids/titles with NO related_decision_topic_ids at all, when decision
    # topics were planned this run (see `_check_adr_topic_mapping_presence`).
    adrs_without_decision_topic_mapping: list[str] = Field(default_factory=list)
    # Required material decision topics (MATERIAL_DECISION_TOPICS /
    # MIGRATION_MATERIAL_TOPIC) planned this run with ZERO qualifying
    # curated-KB evidence retrieved for that topic specifically — an
    # honest, non_refinable per-topic gap (see
    # `_check_material_topic_evidence_gaps`), distinct from `kb_evidence_gap`
    # (which only fires when nothing was retrieved for ANY topic at all).
    material_topics_without_evidence: list[str] = Field(default_factory=list)

    # Invented performance/scale targets found in architecture-owned prose
    # (Blueprint, ADR, Component, migration-step text) that the locked
    # Context Record never authorized — defense-in-depth for the same
    # invariant enforced fail-fast on generated Features (see
    # `architect._validate_no_invented_quantitative_targets` and
    # `find_unauthorized_quantitative_targets`). Keyed by artifact location.
    invented_quantitative_targets: dict[str, list[str]] = Field(default_factory=dict)

    # Target-architecture service references (name -> referencing locations)
    # that resolve to no Component Description and carry no explicit legacy,
    # external, or ownership disposition anywhere in the design.
    unowned_target_services: dict[str, list[str]] = Field(default_factory=dict)

    # New implementation languages (canonical name) with no explicit
    # deviation justification against the detected stack. Empty when the
    # design stays in the stack, justifies the change, or none was detected.
    unjustified_language_drift: dict[str, list[str]] = Field(default_factory=dict)

    # Arrow-flow endpoints with no Component Description and no external
    # marker, mapped to the flows that reference them (invariant A).
    unresolved_flow_participants: dict[str, list[str]] = Field(default_factory=dict)

    # Non-empty data flows the shared renderer grammar cannot parse into a
    # directional edge (invariant A2), verbatim.
    unrenderable_data_flows: list[str] = Field(default_factory=list)

    # New internal application services with no explicit migration
    # disposition, by name (invariant B).
    services_without_migration_disposition: list[str] = Field(default_factory=list)

    # Services a migration step promises to introduce that resolve to no
    # Component Description, mapped to the step titles naming them (the
    # reverse of invariant B).
    unresolved_migration_targets: dict[str, list[str]] = Field(default_factory=dict)

    constraints_applicable: dict[str, bool] = Field(default_factory=dict)
    constraints_covered: dict[str, bool] = Field(default_factory=dict)
    # The exact requirement strings that failed coverage, per group — the
    # actionable form of `constraints_covered` (see `_check_constraints`).
    constraints_uncovered: dict[str, list[str]] = Field(default_factory=dict)

    # Catalog requirements ("FR-xxx"/"NFR-xxx", see `requirement_catalog`)
    # with no Feature covering them, rendered "<ID> - <text>" (see
    # `_check_requirement_feature_coverage`). Orthogonal to
    # `constraints_uncovered`: this proves the requirement entered the
    # modeled feature set, not that the final design addresses it. Feeds
    # `score_traceability` (merged into `traceability` in
    # `run_deterministic_checks`) so a HIGH finding here cannot coexist with
    # a 2/2 traceability score.
    requirements_without_feature: list[str] = Field(default_factory=list)

    score_all_artifacts_present: int = Field(0, ge=0, le=2)
    score_constraint_coverage: int = Field(0, ge=0, le=2)
    score_traceability: int = Field(0, ge=0, le=2)
    score_adr_presence: int = Field(0, ge=0, le=2)
    score_source_integrity: int = Field(0, ge=0, le=2)
    score_kb_evidence_grounding: int = Field(0, ge=0, le=2)

    issues: list[ReviewIssue] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────────

def _non_empty(values: list[str]) -> list[str]:
    """Return stripped, non-empty strings only."""

    return [
        value.strip()
        for value in values
        if isinstance(value, str) and value.strip()
    ]


def _design_text(state: ArchitectState) -> str:
    """Render generated design evidence for constraint coverage.

    The Context Record states the requirements. It is deliberately excluded
    here because repeating a constraint in the input does not prove that the
    generated design addresses it.
    """

    parts: list[str] = []

    if state.blueprint is not None:
        blueprint = state.blueprint
        parts.extend(
            [
                blueprint.project_name,
                blueprint.selected_pattern,
                blueprint.rationale,
                blueprint.stakeholder_view,
                blueprint.technical_view,
            ]
        )
        parts.extend(blueprint.components)
        parts.extend(blueprint.data_flows)
        parts.extend(blueprint.constraints_addressed)
        parts.extend(blueprint.assumptions)
        parts.extend(blueprint.open_risks)

    for adr in state.adrs:
        parts.extend(
            [
                adr.id,
                adr.title,
                adr.status,
                adr.context,
                adr.decision,
                adr.rationale,
            ]
        )
        parts.extend(adr.alternatives_considered)
        parts.extend(adr.positive_consequences)
        parts.extend(adr.negative_consequences)
        parts.extend(adr.related_feature_ids)
        parts.extend(adr.related_component_names)
        parts.extend(adr.source_references)

    for component in state.components:
        parts.extend(
            [
                component.id,
                component.name,
                component.component_type,
                component.purpose,
                component.description,
            ]
        )
        parts.extend(component.inputs)
        parts.extend(component.outputs)
        parts.extend(component.dependencies)
        parts.extend(component.related_feature_ids)
        parts.extend(component.related_adr_ids)
        parts.extend(component.technology_choices)
        parts.extend(component.security_considerations)
        parts.extend(component.scalability_considerations)

    return "\n".join(_non_empty(parts)).lower()


# ─────────────────────────────────────────────────────────────────────────
# Invented quantitative targets (Integration, Kush)
#
# Real E2E gap: the Architect prompt has always FORBIDDEN inventing numeric
# SLOs/latency/availability/scale figures ("QUANTITATIVE TARGETS (strict)"
# in agents/architect.py), but that was prompt-only — nothing deterministic
# ever checked it, so a generated Feature could carry "under 200ms" or "10x
# traffic" straight through refinement and a PASS even though the locked
# Context Record explicitly stated no numeric target was defined.
#
# Deliberately NARROW pattern set: only the categories the project actually
# needs to catch (latency, percentage/availability, multiplier claims,
# concurrency/user-load, throughput) — not a blanket "flag every number",
# which would also catch requirement ordinals, IDs, page numbers, version
# strings, and port numbers that legitimately belong in architecture text.
# ─────────────────────────────────────────────────────────────────────────

_QUANTITATIVE_TARGET_PATTERNS: tuple[re.Pattern, ...] = (
    # Percentages: availability/uptime/error-rate targets ("99.9%").
    re.compile(r"\b\d+(?:\.\d+)?\s?%"),
    # Latency budgets: "200ms", "1.5 seconds".
    re.compile(r"\b\d+(?:\.\d+)?\s?(?:milliseconds?|ms)\b", re.IGNORECASE),
    re.compile(r"\b\d+(?:\.\d+)?\s?(?:seconds?|secs?|sec)\b", re.IGNORECASE),
    # Multiplier claims: "10x traffic".
    re.compile(r"\b\d+(?:\.\d+)?\s?x\b", re.IGNORECASE),
    # Concurrency / user-load targets: "10k concurrent users", "50,000 users".
    re.compile(
        r"\b\d{1,3}(?:,\d{3})*(?:k|K)?\+?\s?(?:concurrent\s+)?users?\b"
    ),
    # Throughput targets: "1000 req/s", "500 rps", "200 transactions/sec".
    re.compile(
        r"\b\d+(?:,\d{3})*\s?"
        r"(?:req(?:uests)?/s(?:ec)?|rps|qps|tps|transactions?/(?:sec|second|s))\b",
        re.IGNORECASE,
    ),
)


def _normalize_quantitative_target(value: str) -> str:
    """Whitespace/case-insensitive identity for comparing a matched target
    string against the locked Context Record's own wording. Pure."""

    return re.sub(r"\s+", "", value).lower()


def _quantitative_target_matches(text: str) -> dict[str, str]:
    """normalized -> first-seen original spelling, for every quantitative
    target-shaped substring in `text`. Pure."""

    matches: dict[str, str] = {}
    for pattern in _QUANTITATIVE_TARGET_PATTERNS:
        for match in pattern.finditer(text or ""):
            original = match.group(0)
            matches.setdefault(_normalize_quantitative_target(original), original)
    return matches


def _authorized_quantitative_targets(context_record: ContextRecord | None) -> set[str]:
    """Quantitative targets the LOCKED Context Record itself authorizes —
    the only values a generated target may legitimately preserve. Pure."""

    if context_record is None:
        return set()
    parts = [
        context_record.business_goal,
        context_record.problem_statement,
        context_record.summary,
        context_record.budget,
        *context_record.functional_requirements,
        *context_record.non_functional_requirements,
        *context_record.compliance_requirements,
        *context_record.assumptions,
        *context_record.open_questions,
        *context_record.existing_systems,
        *context_record.users,
    ]
    text = "\n".join(_non_empty(parts))
    return set(_quantitative_target_matches(text))


def find_unauthorized_quantitative_targets(
    text: str, context_record: ContextRecord | None
) -> list[str]:
    """Quantitative performance/scale targets in generated TEXT that the
    locked Context Record never authorized. Pure; the ONE detector reused by
    both the architect's fail-fast Feature validation (before phase 2 ever
    runs) and this module's defense-in-depth deterministic Reviewer check,
    so there is exactly one place "what counts as an invented target" is
    decided rather than two that could drift.

    Returns the exact matched substrings (original spelling, not the
    normalized comparison form), sorted for determinism. Empty when nothing
    unauthorized is found, including when `text` is empty.
    """

    if not text:
        return []
    authorized = _authorized_quantitative_targets(context_record)
    found = _quantitative_target_matches(text)
    return sorted(
        original
        for normalized, original in found.items()
        if normalized not in authorized
    )


def _quantitative_target_locations(state: ArchitectState) -> list[tuple[str, str]]:
    """(location label, text) pairs across architecture-OWNED prose — the
    defense-in-depth scan surface for `find_unauthorized_quantitative_targets`.
    Pure. Deliberately excludes Feature text: that channel is validated
    fail-fast, before phase 2 ever runs (see
    `architect._validate_no_invented_quantitative_targets`), so by the time
    this deterministic Reviewer check runs, Features are already clean —
    this covers the phase-2-owned fields that fail-fast cannot reach."""

    locations: list[tuple[str, str]] = []
    if state.blueprint is not None:
        b = state.blueprint
        for label, value in (
            ("Blueprint.rationale", b.rationale),
            ("Blueprint.stakeholder_view", b.stakeholder_view),
            ("Blueprint.technical_view", b.technical_view),
        ):
            if value:
                locations.append((label, value))
        for step in b.migration_steps:
            for label, value in (
                (f"MigrationStep[{step.title}].objective", step.objective),
                (
                    f"MigrationStep[{step.title}].coexistence_or_data_strategy",
                    step.coexistence_or_data_strategy,
                ),
            ):
                if value:
                    locations.append((label, value))
            for change in step.changes:
                if change:
                    locations.append((f"MigrationStep[{step.title}].changes", change))

    for adr in state.adrs:
        label_prefix = adr.id or adr.title
        for label, value in (
            (f"{label_prefix}.context", adr.context),
            (f"{label_prefix}.decision", adr.decision),
            (f"{label_prefix}.rationale", adr.rationale),
        ):
            if value:
                locations.append((label, value))
        for value in adr.positive_consequences:
            if value:
                locations.append((f"{label_prefix}.positive_consequences", value))
        for value in adr.negative_consequences:
            if value:
                locations.append((f"{label_prefix}.negative_consequences", value))

    for component in state.components:
        for label, value in (
            (f"{component.name}.purpose", component.purpose),
            (f"{component.name}.description", component.description),
        ):
            if value:
                locations.append((label, value))
        for value in component.scalability_considerations:
            if value:
                locations.append((f"{component.name}.scalability_considerations", value))

    return locations


def _check_quantitative_targets(state: ArchitectState) -> dict[str, list[str]]:
    """Defense-in-depth deterministic scan: architecture-owned prose must not
    carry an invented quantitative target (see
    `find_unauthorized_quantitative_targets`). Primary enforcement is
    fail-fast at Feature generation; this catches the same invention if it
    enters through Blueprint/ADR/Component/migration-step prose instead.
    Pure."""

    findings: dict[str, list[str]] = {}
    for label, text in _quantitative_target_locations(state):
        targets = find_unauthorized_quantitative_targets(text, state.context_record)
        if not targets:
            continue
        existing = findings.setdefault(label, [])
        for target in targets:
            if target not in existing:
                existing.append(target)
    return findings


# ─────────────────────────────────────────────────────────────────────────
# Rubric item 1: artifact completeness
# ─────────────────────────────────────────────────────────────────────────

def _check_artifacts(
    state: ArchitectState,
) -> tuple[dict[str, bool], list[str]]:
    """Check that required artifacts exist and core fields are populated.

    Core compatibility fields remain the minimum requirement because the
    Clarifier and some older fixtures still populate only `summary`.
    Richer structured fields are validated by the Architect tests and the
    Reviewer LLM.
    """

    present = {
        "context_record": state.context_record is not None,
        "features": bool(state.features),
        "blueprint": state.blueprint is not None,
        "adrs": bool(state.adrs),
        "components": bool(state.components),
    }

    missing: list[str] = []

    if state.context_record is not None:
        context = state.context_record
        if not any(
            value.strip()
            for value in (
                context.summary,
                context.business_goal,
                context.problem_statement,
            )
        ):
            missing.append("context_record.problem_definition")

    if state.blueprint is not None:
        if not state.blueprint.stakeholder_view.strip():
            missing.append("blueprint.stakeholder_view")

        if not state.blueprint.technical_view.strip():
            missing.append("blueprint.technical_view")

    for index, adr in enumerate(state.adrs):
        if not adr.title.strip():
            missing.append(f"adrs[{index}].title")

        if not adr.decision.strip():
            missing.append(f"adrs[{index}].decision")

        if not adr.context.strip():
            missing.append(f"adrs[{index}].context")

        if not adr.rationale.strip():
            missing.append(f"adrs[{index}].rationale")

        if not _non_empty(adr.alternatives_considered):
            missing.append(f"adrs[{index}].alternatives_considered")

        if not _non_empty(adr.positive_consequences):
            missing.append(f"adrs[{index}].positive_consequences")

        if not _non_empty(adr.negative_consequences):
            missing.append(f"adrs[{index}].negative_consequences")

    for index, component in enumerate(state.components):
        if not component.name.strip():
            missing.append(f"components[{index}].name")

        if not component.description.strip():
            missing.append(f"components[{index}].description")

        if not component.purpose.strip():
            missing.append(f"components[{index}].purpose")

    for index, feature in enumerate(state.features):
        if not feature.id.strip():
            missing.append(f"features[{index}].id")

        if not feature.name.strip():
            missing.append(f"features[{index}].name")

        if not feature.scenario.strip():
            missing.append(f"features[{index}].scenario")

        if not _non_empty(feature.acceptance_criteria):
            missing.append(f"features[{index}].acceptance_criteria")

    return present, missing


# ─────────────────────────────────────────────────────────────────────────
# Rubric item 5: structured traceability
# ─────────────────────────────────────────────────────────────────────────

def _component_text(component) -> str:
    """Legacy traceability text for older fixtures."""

    return " ".join(
        _non_empty(
            [
                component.name,
                component.purpose,
                component.description,
            ]
        )
    ).lower()


def _adr_text(adr) -> str:
    """Legacy ADR text for older fixtures."""

    return " ".join(
        _non_empty(
            [
                adr.id,
                adr.title,
                adr.context,
                adr.decision,
                adr.rationale,
                *adr.related_component_names,
            ]
        )
    ).lower()


def _component_supports_feature(
    component,
    feature_id: str,
    *,
    allow_prose_fallback: bool,
) -> bool:
    """Return whether a component traces to the feature.

    Structured references are preferred. Prose fallback is retained only for
    older fixtures whose `related_feature_ids` field is empty.
    """

    explicit_ids = {
        value.strip()
        for value in component.related_feature_ids
        if value.strip()
    }

    if explicit_ids:
        return feature_id in explicit_ids

    return allow_prose_fallback and feature_id.lower() in _component_text(component)


def _component_has_valid_feature(
    component,
    valid_feature_ids: set[str],
    *,
    allow_prose_fallback: bool,
) -> bool:
    """Check that a component references at least one existing feature."""

    explicit_ids = {
        value.strip()
        for value in component.related_feature_ids
        if value.strip()
    }

    if explicit_ids:
        return bool(explicit_ids & valid_feature_ids)

    if not allow_prose_fallback:
        return False

    legacy_text = _component_text(component)
    return any(
        feature_id.lower() in legacy_text
        for feature_id in valid_feature_ids
    )


def _component_has_valid_adr(
    component,
    state: ArchitectState,
    valid_adr_ids: set[str],
    *,
    allow_prose_fallback: bool,
) -> bool:
    """Check that a component is justified by an ADR.

    Preferred link:
        component.related_adr_ids → ADR.id

    Legacy fallback:
        component name appears in an ADR's title or decision text.
    """

    explicit_ids = {
        value.strip()
        for value in component.related_adr_ids
        if value.strip()
    }

    if explicit_ids:
        return bool(explicit_ids & valid_adr_ids)

    if not allow_prose_fallback:
        return False

    component_name = component.name.strip().lower()
    if not component_name:
        return False

    return any(
        component_name in _adr_text(adr)
        for adr in state.adrs
    )


def _check_traceability(
    state: ArchitectState,
    *,
    allow_prose_fallback: bool,
) -> dict[str, object]:
    """Check Feature → Component → ADR traceability.

    New designs are checked through explicit schema fields. Legacy prose-based
    matching remains as a compatibility bridge for existing tests and old
    artifacts.
    """

    feature_ids = {
        feature.id.strip()
        for feature in state.features
        if feature.id.strip()
    }

    adr_ids = {
        adr.id.strip()
        for adr in state.adrs
        if adr.id.strip()
    }
    # Canonical identity (lifecycle qualifier and trailing "Service" both
    # ignored) — the SAME contract `_component_key` gives the target-service
    # ownership check, so an ADR that names a component bare ("User
    # Service") resolves against a Component Description declared with a
    # display qualifier ("User Service (New)") instead of failing on a raw
    # string mismatch that was never a real gap.
    component_keys = {
        _component_key(component.name)
        for component in state.components
        if _component_key(component.name)
    }

    blueprint_feature_ids = {
        value.strip()
        for value in (
            state.blueprint.addressed_feature_ids
            if state.blueprint is not None
            else []
        )
        if value.strip()
    }

    features_without_component = sorted(
        feature_id
        for feature_id in feature_ids
        if not any(
            _component_supports_feature(
                component,
                feature_id,
                allow_prose_fallback=allow_prose_fallback,
            )
            for component in state.components
        )
    )

    components_without_feature = sorted(
        component.name
        for component in state.components
        if not _component_has_valid_feature(
            component,
            feature_ids,
            allow_prose_fallback=allow_prose_fallback,
        )
    )

    components_without_adr = sorted(
        component.name
        for component in state.components
        if not _component_has_valid_adr(
            component,
            state,
            adr_ids,
            allow_prose_fallback=allow_prose_fallback,
        )
    )

    invalid_component_feature_ids = {
        component.name: sorted(
            {
                value.strip()
                for value in component.related_feature_ids
                if value.strip()
            }
            - feature_ids
        )
        for component in state.components
    }
    invalid_component_feature_ids = {
        name: values
        for name, values in invalid_component_feature_ids.items()
        if values
    }

    invalid_component_adr_ids = {
        component.name: sorted(
            {
                value.strip()
                for value in component.related_adr_ids
                if value.strip()
            }
            - adr_ids
        )
        for component in state.components
    }
    invalid_component_adr_ids = {
        name: values
        for name, values in invalid_component_adr_ids.items()
        if values
    }

    invalid_adr_feature_ids = {
        adr.id: sorted(
            {
                value.strip()
                for value in adr.related_feature_ids
                if value.strip()
            }
            - feature_ids
        )
        for adr in state.adrs
    }
    invalid_adr_feature_ids = {
        adr_id: values
        for adr_id, values in invalid_adr_feature_ids.items()
        if values
    }

    invalid_adr_component_names = {
        adr.id: sorted(
            {
                value.strip()
                for value in adr.related_component_names
                if value.strip() and _component_key(value) not in component_keys
            }
        )
        for adr in state.adrs
    }
    invalid_adr_component_names = {
        adr_id: values
        for adr_id, values in invalid_adr_component_names.items()
        if values
    }

    blueprint_missing_feature_ids = sorted(feature_ids - blueprint_feature_ids)
    if allow_prose_fallback and not blueprint_feature_ids:
        blueprint_missing_feature_ids = []

    adrs_without_feature = sorted(
        adr.id
        for adr in state.adrs
        if not _non_empty(adr.related_feature_ids)
        and not allow_prose_fallback
    )
    adrs_without_component = sorted(
        adr.id
        for adr in state.adrs
        if not _non_empty(adr.related_component_names)
        and not allow_prose_fallback
    )

    return {
        "features_without_component": features_without_component,
        "components_without_feature": components_without_feature,
        "components_without_adr": components_without_adr,
        "blueprint_missing_feature_ids": blueprint_missing_feature_ids,
        "invalid_blueprint_feature_ids": sorted(blueprint_feature_ids - feature_ids),
        "invalid_component_feature_ids": invalid_component_feature_ids,
        "invalid_component_adr_ids": invalid_component_adr_ids,
        "invalid_adr_feature_ids": invalid_adr_feature_ids,
        "invalid_adr_component_names": invalid_adr_component_names,
        "adrs_without_feature": adrs_without_feature,
        "adrs_without_component": adrs_without_component,
    }


# ─────────────────────────────────────────────────────────────────────────
# Rubric item: cross-artifact target-service ownership
#
# A completed run once passed review while FEAT-005 redirected the Cart
# module to a "Cart Service" that had no Component Description, the Shared
# Event Bus depended on a "Notification Service" that did not exist, and an
# ADR made Inventory part of the target architecture with no owner. The
# structured traceability checks never saw those references because they
# lived in prose and dependency lists, so Traceability scored 2/2.
#
# This check is deliberately DETERMINISTIC and narrow: a reference counts
# only when the name carries the "Service" suffix ("Cart Service", "the
# Order, Payment, and Notification services") or is listed outright as a
# Blueprint component. Bare nouns ("inventory", "cart") are not treated as
# references — without the suffix they are indistinguishable from generic
# prose, and blocking on them would flag valid designs.
# ─────────────────────────────────────────────────────────────────────────

#
# COMPOSITE NAMES ("Order/Payment Service") ARE ONE UNIT (regression, final
# run 20260822T222414Z-1788d214): the bare pattern captured only the LAST
# slash-joined word ("Payment"), splitting a single declared composite
# component into a phantom reference to half its name. The slash-continuation
# group captures the whole "Order/Payment" span as one base so the reference
# resolves to the composite component, not a fragment of it.
_TARGET_SINGULAR_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9-]*(?:/[A-Z][A-Za-z0-9-]*)*)\s+Service\b"
)
# The plural/list form. Continuation accepts ", and" (Oxford comma) as well
# as "," and "and", so "Order, Payment, and Notification services" captures
# the WHOLE enumeration rather than only its last name.
_TARGET_LIST_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9-]*(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+|\s+)[A-Z][A-Za-z0-9-]*)*)"
    r"\s+[Ss]ervices\b"
)

# This verb vocabulary is the one shared knob of
# the migration checks — it feeds BOTH the introduction matcher and the
# never-a-name blocklist below, so extend the tuple only, never one side.
#
# THE ONE ACTION-VERB VOCABULARY. Deliberately tiny: these verbs say the
# named thing BECOMES part of the target architecture; every inflected
# form is spelled out here once, and BOTH consumers derive from it —
#   * `_MIGRATION_INTRODUCTION_RE` (beside `_check_migration_targets`)
#     matches them in migration-step sentences, and
#   * `_NON_NAME_WORDS` (below) unions them, so the same verbs can never
#     be parsed as candidate service BASE names.
# "Migrate" is excluded on purpose — it names data movement as often as
# service introduction, and a false negative is cheaper here than a false
# positive. One source of truth: a verb added for the introduction matcher
# is automatically also a never-a-name word, and vice versa.
_MIGRATION_ACTION_VERBS: tuple[str, ...] = (
    "extract", "extracts", "extracted", "extracting",
    "create", "creates", "created", "creating",
    "introduce", "introduces", "introduced", "introducing",
    "deploy", "deploys", "deployed", "deploying",
    "build", "builds", "built", "building",
    "establish", "establishes", "established", "establishing",
    "launch", "launches", "launched", "launching",
    "split", "splits", "splitting",
    "carve", "carves", "carved", "carving",
    "decompose", "decomposes", "decomposed", "decomposing",
)

# TitleCase words that begin sentences or qualify nouns; never service names.
# Cloud providers are included so phrases like "AWS Managed Services" cannot
# manufacture a target service.
#
# MIGRATION ACTION VERBS ARE STRUCTURAL SYNTAX, NOT NAMES — a migration
# step title like "Decompose Catalog and Review Services" must yield the
# bases 'Catalog'/'Review' and NEVER 'Decompose': the verb is the action,
# the TitleCase words after it are the enumeration (regression, fresh E2E
# after the migration-target hardening: the missing 'decompose' entry
# produced a phantom 'Decompose Service' HIGH finding).
_NON_NAME_WORDS = frozenset({
    "above", "add", "added", "all", "amazon", "an", "and", "another", "any",
    "as", "at", "aws", "azure", "below", "both", "build", "built", "by",
    "call", "called", "central", "cloud", "create", "created", "current",
    "dedicated", "defer", "deploy", "deployed", "existing", "expose",
    "extract", "extracted", "external", "for", "from", "generic", "gcp",
    "google", "host", "hosted", "if", "in", "independent", "internal",
    "introduce", "invoke", "it", "its", "local", "main", "managed",
    "microsoft", "migrate", "migrated", "move", "moved", "new", "no", "of",
    "on", "one", "or", "other", "our", "publish", "published", "remote",
    "replace", "replaced", "run", "running", "same", "separate", "send",
    "sent", "shared", "such", "target", "the", "their", "then", "these",
    "they", "this", "those", "three", "to", "two", "use", "used", "using",
    "via", "we", "when", "while", "whole", "with", "you",
}) | frozenset(_MIGRATION_ACTION_VERBS)

_LEGACY_NOUN_RE = re.compile(r"\b(?:monolith|legacy)\b")
_RETENTION_VERB_RE = re.compile(
    r"\b(?:remain(?:s|ed|ing)?|stay(?:s|ed|ing)?|keep(?:s|t|ing)?|"
    r"retain(?:s|ed|ing)?|defer(?:s|red|ring)?|unchanged|"
    r"not\s+(?:be\s+)?extracted|left\s+in|served\s+by|"
    r"continue[sd]?\s+to\s+(?:be\s+)?(?:live|run|reside))\b"
)
_OWNERSHIP_PHRASE_RE = re.compile(
    r"\b(?:owned\s+by|handled\s+by|provided\s+by|managed\s+by|"
    r"responsibility\s+of|remains?\s+with|belongs\s+to|"
    r"hosted\s+(?:by|in)|embedded\s+in)\b"
)
_EXTERNAL_DISPOSITION_RE = re.compile(
    r"\b(?:is|are|as)\s+(?:an?\s+)?(?:fully\s+)?"
    r"(?:external|third[- ]party|third\s+party|saas|off-the-shelf|managed)\b"
    r"|\b(?:external|third[- ]party|third\s+party|saas|off-the-shelf)\s+"
    r"(?:provider|vendor|service|platform|system|partner|api)\b"
)


def _singularise(token: str) -> str:
    """Fold the trivial English plural so 'Notifications' meets 'Notification'."""

    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


# The one British/American spelling split a real E2E run needed tolerated:
# a component catalog named "Shipping and Fulfilment Service" (British) while
# a migration step said "...Fulfillment Services" (American) — same word,
# different orthography, and generic case/punctuation folding cannot bridge
# a spelling difference. Narrow and explicit on purpose: this is presentation
# normalization for one real, named variant, not a spelling corrector — the
# vocabulary this check must never become (see `_check_migration_targets`).
_SPELLING_VARIANTS: dict[str, str] = {"fulfillment": "fulfilment"}


def _fold_spelling_variant(token: str) -> str:
    return _SPELLING_VARIANTS.get(token, token)


def _name_tokens(name: str) -> tuple[str, ...]:
    """Normalise a display name to comparable lowercase tokens.

    '&' folds to 'and' before tokenizing (rather than vanishing, which the
    bare alphanumeric regex would otherwise do) so "Order & Checkout" and
    "Order and Checkout" compare equal — the same presentation-only
    tolerance case already covers for spacing, case and punctuation.
    """

    normalized = name.lower().replace("&", " and ")
    return tuple(
        _fold_spelling_variant(_singularise(token))
        for token in re.findall(r"[a-z0-9]+", normalized)
    )


# Component Description names sometimes carry a trailing display/lifecycle
# qualifier ("User Service (New)", "Django Monolith (Legacy)") that a target
# reference to the SAME component never repeats ("User Service"). The
# qualifier describes STATUS, not identity, so it is stripped before identity
# tokens are computed (regression, final run 20260822T222414Z-1788d214: four
# declared components were reported missing because their name's trailing
# "(New)" token survived tokenizing and a bare reference's did not).
_LIFECYCLE_QUALIFIER_RE = re.compile(
    r"\s*\((?:new|legacy|existing|current|updated|deprecated|proposed|planned)\)\s*$",
    re.IGNORECASE,
)


def _strip_lifecycle_qualifier(name: str) -> str:
    """Drop a trailing '(New)'/'(Legacy)'-style display qualifier. Pure."""

    return _LIFECYCLE_QUALIFIER_RE.sub("", name)


def _component_key(name: str) -> tuple[str, ...]:
    """Identity tokens of a component name: lifecycle qualifier and a
    trailing 'Service' both ignored. Applied symmetrically to defined
    Component Description names and extracted target references so the
    same name in either form resolves to the same identity."""

    tokens = _name_tokens(_strip_lifecycle_qualifier(name))
    while tokens and tokens[-1] == "service":
        tokens = tokens[:-1]
    return tokens


def _extract_service_bases(text: str) -> list[str]:
    """Pull candidate target-service names out of one text field.

    Matches 'Cart Service' (single TitleCase word before 'Service') and
    enumerations like 'Order, Payment, and Notification services'.

    PLURAL PROSE IS NOT A REFERENCE (regression, run 20260822T103036Z):
    'Decouples services', 'Allows services to evolve' — a lone TitleCase
    verb/adjective before the plural 'services' is sentence-initial
    capitalization, not a name, and treating it as one invented a
    'Decouples Service' and a false HIGH finding. Structurally, the plural
    form yields names ONLY when the surrounding syntax explicitly
    ENUMERATES service entities: a comma or 'and' joining at least two
    TitleCase tokens that survive the non-name filter. A single word
    before 'services' is therefore never a name; a genuine singular
    reference ('Payment Service') is caught by the singular regex.
    """
    bases: set[str] = set()
    for match in _TARGET_SINGULAR_RE.finditer(text):
        base = match.group(1)
        if base.lower() not in _NON_NAME_WORDS and base.lower() != "service":
            bases.add(base)
    for match in _TARGET_LIST_RE.finditer(text):
        enumeration = match.group(1)
        names = [
            part
            for part in re.split(r",|\s+and\s+|\s+", enumeration)
            if part
            and part.lower() not in _NON_NAME_WORDS
            and part.lower() != "service"
        ]
        is_explicit_list = ("," in enumeration) or (
            re.search(r"\s+and\s+", enumeration) is not None
        )
        if is_explicit_list and len(names) >= 2:
            bases.update(names)
    return sorted(bases)


def _sentence_spans(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def _target_service_references(state: ArchitectState) -> dict[str, set[str]]:
    """Map each referenced target-service name to where it is referenced.

    Scanned fields describe the TARGET architecture: Feature prose, ADR
    context/decision/rationale/positive consequences (alternatives and
    negative consequences are excluded — they legitimately name services
    that were rejected or are absent), Blueprint component names and data
    flows, and Component dependency/input/output lists.
    """

    references: dict[str, set[str]] = {}

    def note(display: str, location: str) -> None:
        if _component_key(display):
            references.setdefault(display, set()).add(location)

    for feature in state.features:
        for text in (
            [feature.description, feature.scenario, *feature.acceptance_criteria]
        ):
            for base in _extract_service_bases(text or ""):
                note(f"{base} Service", feature.id)

    if state.blueprint is not None:
        for entry in state.blueprint.components:
            note(entry.strip(), "Blueprint.components")
        for flow in state.blueprint.data_flows:
            for base in _extract_service_bases(flow):
                note(f"{base} Service", "Blueprint.data_flows")

    for adr in state.adrs:
        for text in (
            [adr.context, adr.decision, adr.rationale, *adr.positive_consequences]
        ):
            for base in _extract_service_bases(text or ""):
                note(f"{base} Service", adr.id)

    for component in state.components:
        for field, label in (
            ("dependencies", "dependencies"),
            ("inputs", "inputs"),
            ("outputs", "outputs"),
        ):
            for entry in getattr(component, field):
                for base in _extract_service_bases(entry):
                    note(f"{base} Service", f"{component.name}.{label}")

    return references


def _design_sentences(state: ArchitectState) -> list[str]:
    """Every artifact sentence, searched for explicit dispositions.

    Deliberately broader than the reference scan: a statement that Cart
    remains in the legacy monolith may live anywhere in the design (an
    assumption, an open risk, an ADR alternative), and allowing it there
    biases this check against false blocks.
    """

    sentences: list[str] = []

    def extend(texts: list[str]) -> None:
        for value in texts:
            if isinstance(value, str) and value.strip():
                sentences.extend(_sentence_spans(value))

    for feature in state.features:
        extend(
            [feature.name, feature.description, feature.scenario,
             *feature.acceptance_criteria]
        )
    if state.blueprint is not None:
        extend(
            [
                state.blueprint.rationale,
                state.blueprint.stakeholder_view,
                state.blueprint.technical_view,
                *state.blueprint.components,
                *state.blueprint.data_flows,
                *state.blueprint.constraints_addressed,
                *state.blueprint.assumptions,
                *state.blueprint.open_risks,
            ]
        )
    for adr in state.adrs:
        extend(
            [
                adr.title, adr.context, adr.decision, adr.rationale,
                *adr.alternatives_considered,
                *adr.positive_consequences,
                *adr.negative_consequences,
            ]
        )
    for component in state.components:
        extend(
            [
                component.name, component.purpose, component.description,
                *component.inputs, *component.outputs,
                *component.dependencies, *component.technology_choices,
            ]
        )
    return sentences


def _has_explicit_disposition(
    key: tuple[str, ...],
    sentences: list[str],
    owner_word_sets: list[set[str]],
) -> bool:
    """Does any design sentence explicitly dispose of this service?

    Allowed dispositions, mirroring the strangler/legacy escape hatches:
    retention in the legacy monolith, an external/third-party/SaaS
    declaration, or ownership mapped to a named existing component.
    """

    token_patterns = [
        re.compile(rf"\b{re.escape(token)}\b") for token in key
    ]
    for sentence in sentences:
        if not any(pattern.search(sentence) for pattern in token_patterns):
            continue
        if (
            _LEGACY_NOUN_RE.search(sentence)
            and _RETENTION_VERB_RE.search(sentence)
        ):
            return True
        if _EXTERNAL_DISPOSITION_RE.search(sentence):
            return True
        if _OWNERSHIP_PHRASE_RE.search(sentence):
            words = set(re.findall(r"[a-z0-9]+", sentence))
            if any(owner_words & words for owner_words in owner_word_sets):
                return True
    return False


def _check_target_service_ownership(
    state: ArchitectState,
) -> dict[str, list[str]]:
    """Referenced target services with no Component Description and no
    explicit legacy/external/ownership disposition, mapped to the artifacts
    that reference them."""

    references = _target_service_references(state)
    if not references:
        return {}

    # (key, is_composite) per declared component. A composite name
    # ("Order/Payment Service") is ONE declared component covering two
    # capabilities; only EXACT identity may satisfy a reference to it — the
    # ordinary suffix match ("Payment" satisfies "Shopping Cart Service"-style
    # compounds) would otherwise let a bare "Payment Service" reference
    # silently ride on a composite it was never declared to cover, hiding a
    # genuinely undeclared "Payment Service" (regression, final run
    # 20260822T222414Z-1788d214).
    component_entries = [
        (_component_key(component.name), "/" in component.name)
        for component in state.components
        if _component_key(component.name)
    ]
    owner_word_sets = [set(key) for key, _composite in component_entries]
    sentences = [
        sentence.lower() for sentence in _design_sentences(state)
    ]

    unowned: dict[str, list[str]] = {}
    for display, locations in references.items():
        key = _component_key(display)
        if any(
            existing == key or (not composite and existing[-len(key):] == key)
            for existing, composite in component_entries
        ):
            continue
        if _has_explicit_disposition(key, sentences, owner_word_sets):
            continue
        unowned[display] = sorted(locations)
    return unowned


# ─────────────────────────────────────────────────────────────────────────
# Rubric item 6: ADR structure
# ─────────────────────────────────────────────────────────────────────────

def _check_adrs(
    state: ArchitectState,
) -> tuple[list[str], list[str], list[int]]:
    """Check ADR title format, decisions, and unique title numbers."""

    malformed: list[str] = []
    incomplete: list[str] = []
    numbers: list[int] = []

    for adr in state.adrs:
        title = adr.title.strip()
        decision = adr.decision.strip()

        match = ADR_TITLE_RE.match(title)

        if match is None or not decision:
            malformed.append(title or "(empty title)")
            continue

        numbers.append(int(match.group(1)))

        if not all(
            (
                adr.context.strip(),
                adr.rationale.strip(),
                _non_empty(adr.alternatives_considered),
                _non_empty(adr.positive_consequences),
                _non_empty(adr.negative_consequences),
            )
        ):
            incomplete.append(adr.id or title)

    duplicates = sorted(
        {
            number
            for number in numbers
            if numbers.count(number) > 1
        }
    )

    return malformed, sorted(incomplete), duplicates


def _normalise_source(value: str) -> str:
    """Normalise a source label while retaining useful path information."""

    return value.strip().replace("\\", "/").lower()


def resolve_source_reference(
    reference: str,
    kb_sources: set[str],
    repository_text: str,
) -> bool:
    """Does one ADR source reference resolve to evidence supplied this run?

    THE ONE RESOLUTION RULE, shared deliberately: the Reviewer's integrity
    check and the Architect's output sanitizer both call this function, so
    "what counts as a real source" cannot drift between the writer and the
    judge. A reference resolves when its normalised form (case-insensitive,
    forward slashes) is a supplied KB chunk source, shares a basename with
    one, or appears verbatim in the repository representation. Pure.
    """
    normalised = _normalise_source(reference)
    basename = normalised.rsplit("/", 1)[-1]
    kb_basenames = {source.rsplit("/", 1)[-1] for source in kb_sources}
    return (
        normalised in kb_sources
        or basename in kb_basenames
        or bool(repository_text and normalised in repository_text)
    )


def _check_source_integrity(
    state: ArchitectState,
) -> tuple[dict[str, list[str]], int]:
    """Reject ADR citations that do not resolve to evidence supplied this run."""

    kb_sources = {
        _normalise_source(chunk.source)
        for chunk in state.retrieved_knowledge
        if chunk.source.strip()
    }
    repository_text = ""
    if state.repo_representation is not None:
        repository_text = state.repo_representation.model_dump_json().lower()

    invalid: dict[str, list[str]] = {}
    total_references = 0
    valid_references = 0
    for adr in state.adrs:
        for reference in _non_empty(adr.source_references):
            total_references += 1
            if resolve_source_reference(reference, kb_sources, repository_text):
                valid_references += 1
            else:
                invalid.setdefault(adr.id or adr.title, []).append(reference)

    if not total_references or not invalid:
        score = 2
    elif valid_references:
        score = 1
    else:
        score = 0
    return invalid, score


# ─────────────────────────────────────────────────────────────────────────
# Rubric item: decision-level KB literature grounding (Integration, Kush)
#
# The project claim is that every ADR is traceable to literature actually
# retrieved from OUR curated knowledge base for THIS run — not to a model's
# own training knowledge, and not to the web-search fallback. `ADR.evidence_ids`
# is the exact, non-fuzzy channel for that claim (see its docstring in
# state.py); this check is the deterministic gate that validates it.
#
# `qualifying_kb_evidence_ids` is the ONE resolution rule, shared exactly
# like `resolve_source_reference` above: the Architect's output-boundary
# sanitizer (`sanitize_adr_evidence_ids` in agents/architect.py) and this
# Reviewer check both call it, so "what counts as real evidence" cannot
# drift between the writer and the judge.
# ─────────────────────────────────────────────────────────────────────────


def qualifying_kb_evidence_ids(state: ArchitectState) -> set[str]:
    """Evidence IDs an ADR may legitimately cite this run. Pure.

    `KBChunk.evidence_id` truthiness IS the provenance signal, full stop.
    The researcher (pipeline/agents/researcher.py `researcher_node`) assigns
    it exclusively when a topic's retrieval call reported
    `origin == "kb"` — the actual outcome of that call, not the chunk's own
    `box` metadata (which only says WHICH curated partition a KB hit came
    from and is never consulted here as a second, potentially-drifting
    provenance authority). A web-fallback chunk is therefore never assigned
    an ID in the first place, regardless of what `box` value it happens to
    carry.
    """
    return {chunk.evidence_id for chunk in state.retrieved_knowledge if chunk.evidence_id}


def _check_kb_evidence_grounding(
    state: ArchitectState,
) -> tuple[list[str], dict[str, list[str]], bool, int, bool]:
    """Decision-level literature-grounding check. Pure.

    Returns (adrs_without_evidence, invalid_evidence_ids, kb_evidence_gap,
    score, gap_is_non_refinable).

    SCORING — the project claim is "every final ADR is traceable to
    literature actually retrieved this run", so NOTHING here may score 2
    while that claim would be false:

    * No qualifying KB evidence was retrieved for ANY topic this run
      (`kb_evidence_gap=True`), and no ADR fabricated a citation anyway →
      score 0. The design cannot be reported as literature-grounded merely
      because the gap was not its fault — an honest "not grounded" is still
      "not grounded". `gap_is_non_refinable=True` in this one case: research
      runs exactly once, before the refine loop exists at all, so no
      Architect redesign can manufacture evidence that was never retrieved.
      `run_deterministic_checks` turns this into a HIGH, `non_refinable`
      issue so the loop can stop honestly instead of burning iterations on
      an unfixable finding (see `refine_gate.evaluate_caps`).
    * Any ADR cites an unresolvable/fabricated ID → score 0, always
      refinable (the Architect can remove or correct the citation), whether
      or not a KB gap also exists.
    * Qualifying evidence exists, no fabrication, and EVERY ADR cites at
      least one valid ID → score 2.
    * Qualifying evidence exists, no fabrication, and SOME (not all) ADRs
      cite it → score 1 — a partial, still-blocking gap: it may reflect a
      genuine per-ADR KB coverage hole rather than an Architect oversight
      (see the enriched issue text in `run_deterministic_checks`, which
      names which decision topics DO have evidence so a reader — human or
      the LLM Reviewer's semantic judgment — can tell the two apart).
    * Qualifying evidence exists, no fabrication, and NO ADR cites any of
      it → score 0 — complete disengagement with evidence that is known to
      exist, which is categorically worse than a partial gap.
    """
    qualifying = qualifying_kb_evidence_ids(state)
    kb_evidence_gap = not qualifying

    if not state.adrs:
        # Handled as a completeness failure elsewhere; no claim is being
        # made by a nonexistent ADR, so there is nothing to fault here.
        return [], {}, kb_evidence_gap, 2, False

    invalid: dict[str, list[str]] = {}
    without_evidence: list[str] = []
    for adr in state.adrs:
        cited = {value.strip() for value in adr.evidence_ids if value.strip()}
        bad = sorted(cited - qualifying)
        if bad:
            # A cited ID that does not resolve is fabrication — reported and
            # scored as a failure regardless of whether qualifying evidence
            # exists at all; an empty gap is never an excuse to invent an ID.
            invalid[adr.id or adr.title] = bad
        if not cited:
            without_evidence.append(adr.id or adr.title)

    if invalid:
        # Fabrication is always refinable, so it is reported as such even
        # when a KB gap also exists — the fabricated ID is the Architect's
        # to fix regardless of what caused it to reach for one.
        return without_evidence, invalid, kb_evidence_gap, 0, False

    if kb_evidence_gap:
        # No fabrication, and nothing genuine existed to cite either.
        return [], {}, True, 0, True

    if len(without_evidence) == len(state.adrs):
        score = 0
    elif without_evidence:
        score = 1
    else:
        score = 2

    return without_evidence, invalid, False, score, False


# ─────────────────────────────────────────────────────────────────────────
# Rubric item: material architecture decision coverage (Integration, Kush)
#
# The project could previously only claim "every final ADR is traceable to
# literature actually retrieved this run" — true, but narrower than what a
# real E2E run exposed: the final design also carried MATERIAL
# recommendations (a migration strategy, a service-decomposition choice)
# that never became an ADR at all, so they were invisible to every existing
# grounding check.
#
# The extension is additive, not a new model: `ADR.related_decision_topic_ids`
# (state.py) is an EXACT, non-fuzzy mapping from an ADR to the
# `DecisionTopic`(s) it decided — reusing the SAME bounded, case-derived
# topic catalog the researcher already plans literature retrieval against
# (pipeline/agents/researcher.py `plan_decision_topics`), so "what counts as
# a material decision category" cannot drift from "what the researcher
# already treats as a decision worth grounding".
# ─────────────────────────────────────────────────────────────────────────

# Every architecture necessarily makes a decision in each of these FOUR
# baseline categories — decomposition, data ownership, integration style,
# and scaling/availability — regardless of case specifics; this is exactly
# researcher._BASELINE_TOPICS, the researcher's own universal (always
# planned) topic set. Unlike the CONDITIONAL topics (compliance,
# observability, technology conservation), which only apply when the case
# actually raises the question, these four can never be "no material
# decision was made" — a design always decomposes something, owns its data
# somehow, integrates its parts somehow, and has SOME scaling/availability
# posture. A regression test pins these strings against
# researcher._BASELINE_TOPICS so the two catalogs cannot silently drift
# apart. One ADR may legitimately cover several of these topics at once —
# this is a coverage requirement, not a forced minimum ADR count.
MATERIAL_DECISION_TOPICS: tuple[str, ...] = (
    "service decomposition and boundaries",
    "data ownership and persistence strategy",
    "integration style: synchronous vs asynchronous communication",
    "scaling and availability strategy",
)

# The fourth material category — brownfield migration/evolution strategy —
# is CONDITIONALLY material: merely planning the topic for retrieval does
# not mean a migration decision was actually made (a greenfield case never
# plans it at all; a brownfield case might still end up not modernizing).
# It becomes material exactly when the Architect actually produced a
# migration sequence (`Blueprint.migration_steps` non-empty) — see
# `_check_material_decision_coverage`.
MIGRATION_MATERIAL_TOPIC = "brownfield migration and evolution strategy"


def _adr_topic_ids(adr: ADR) -> set[str]:
    return {value.strip() for value in adr.related_decision_topic_ids if value.strip()}


def _check_material_decision_coverage(
    state: ArchitectState,
) -> tuple[dict[str, list[str]], list[str]]:
    """Deterministic material-decision coverage gate. Pure.

    Returns (invalid_topic_ids, material_decisions_without_adr):

    * `invalid_topic_ids`: ADR id -> `related_decision_topic_ids` values
      that do not resolve to any `state.decision_topics` ID this run
      (fabricated or stale) — the same non-fuzzy-reference discipline as
      `invalid_evidence_ids` and `invalid_source_references`.
    * `material_decisions_without_adr`: topic strings from
      `MATERIAL_DECISION_TOPICS` (always required once planned) plus
      `MIGRATION_MATERIAL_TOPIC` (required only when a migration sequence
      was actually produced) that were planned this run but have NO ADR
      whose `related_decision_topic_ids` names them. Deliberately NOT every
      planned `DecisionTopic` — a topic outside this bounded set (e.g.
      observability, technology conservation) never forces an ADR merely
      because it was planned for retrieval; see the module comment above
      for why. A topic never planned this run at all is not required
      either — there is nothing to be material ABOUT.
    """
    topic_id_by_name = {topic.topic: topic.id for topic in state.decision_topics}
    valid_topic_ids = {topic.id for topic in state.decision_topics}

    invalid_topic_ids: dict[str, list[str]] = {}
    covered_topic_ids: set[str] = set()
    for adr in state.adrs:
        mapped = _adr_topic_ids(adr)
        bad = sorted(mapped - valid_topic_ids)
        if bad:
            invalid_topic_ids[adr.id or adr.title] = bad
        covered_topic_ids |= (mapped & valid_topic_ids)

    required_topic_names = list(MATERIAL_DECISION_TOPICS)
    if state.blueprint is not None and state.blueprint.migration_steps:
        required_topic_names.append(MIGRATION_MATERIAL_TOPIC)

    missing: list[str] = []
    for topic_name in required_topic_names:
        topic_id = topic_id_by_name.get(topic_name)
        if topic_id is None:
            continue  # never planned this run — nothing to require coverage of
        if topic_id not in covered_topic_ids:
            missing.append(topic_name)

    return invalid_topic_ids, missing


def _check_adr_topic_mapping_presence(state: ArchitectState) -> list[str]:
    """ADR ids/titles with NO `related_decision_topic_ids` at all. Pure.

    Only asserted when `state.decision_topics` is non-empty this run — with
    topics actually planned, the Architect contract is that EVERY ADR maps
    to the decision topic(s) it addresses (see the DECISION-TOPIC MAPPING
    prompt rule in agents/architect.py), not just the ones this module
    treats as unconditionally material. Without this, an ADR outside
    `MATERIAL_DECISION_TOPICS` could carry no topic mapping at all and cite
    any globally qualifying evidence, silently escaping
    `_check_adr_evidence_topic_provenance`'s tightened check (which only
    applies to ADRs that DO declare a mapping).

    Backward compatible: a state with no decision_topics at all (old
    checkpoints, a caller/test that never planned topics) makes no claim
    here — there is nothing for an ADR to map TO, so an absent mapping is
    not a defect.
    """
    if not state.decision_topics:
        return []
    return [
        adr.id or adr.title
        for adr in state.adrs
        if not _adr_topic_ids(adr)
    ]


def _check_adr_evidence_topic_provenance(
    state: ArchitectState,
) -> dict[str, list[str]]:
    """Tightened evidence provenance for ADRs that declare a topic mapping.

    Pure. An ADR's cited `evidence_id` must belong to the qualifying
    evidence of at least one of ITS OWN mapped decision topics — not merely
    be SOME qualifying evidence retrieved this run for an unrelated topic.
    Without this, an ADR for a migration strategy could satisfy the plain
    fabrication check in `_check_kb_evidence_grounding` by citing a chunk
    that was only ever retrieved for, say, the data-ownership topic.

    Only applies to ADRs that declared `related_decision_topic_ids` — an
    ADR with no topic mapping at all is `_check_adr_topic_mapping_presence`'s
    finding to report, not this one's; this check only tightens the ADRs
    that DID map. This is additive, not a second competing evidence gate.
    """
    if not state.decision_topics:
        return {}
    evidence_by_topic_id = {
        topic.id: set(topic.evidence_ids) for topic in state.decision_topics
    }
    qualifying = qualifying_kb_evidence_ids(state)

    outside: dict[str, list[str]] = {}
    for adr in state.adrs:
        mapped_topic_ids = _adr_topic_ids(adr) & evidence_by_topic_id.keys()
        if not mapped_topic_ids:
            continue
        allowed: set[str] = set()
        for topic_id in mapped_topic_ids:
            allowed |= evidence_by_topic_id[topic_id]
        cited = {value.strip() for value in adr.evidence_ids if value.strip()}
        # Restricted to QUALIFYING citations: an already-fabricated ID is
        # `_check_kb_evidence_grounding`'s finding to report, not this one's.
        bad = sorted((cited & qualifying) - allowed)
        if bad:
            outside[adr.id or adr.title] = bad
    return outside


def _check_material_topic_evidence_gaps(state: ArchitectState) -> list[str]:
    """Required material decision topics that retrieved ZERO qualifying
    curated-KB evidence for THAT topic specifically. Pure.

    Distinct from the whole-run `kb_evidence_gap` in
    `_check_kb_evidence_grounding` (which fires only when NOTHING
    qualifying was retrieved for ANY topic all run): this fires PER TOPIC,
    so a run where every OTHER required topic has evidence but one
    material category came back completely empty is still surfaced
    honestly, rather than folded into the generic (refinable)
    `adrs_without_kb_evidence` finding. Research runs exactly once, before
    the refine loop exists, so no amount of re-designing can manufacture
    evidence for a topic that was never retrieved — `run_deterministic_checks`
    marks the resulting issue `non_refinable` for exactly the same reason
    the whole-run gap already is.

    Purely topic-centric by design: it does not inspect which ADR maps to
    the topic or what ELSE that ADR maps to. An ADR spanning both this gap
    topic and another topic that DOES have evidence is unaffected — it can
    still legitimately ground itself in the other topic's evidence, which
    `_check_adr_evidence_topic_provenance` and `_check_material_decision_coverage`
    (not this function) continue to judge on their own terms.
    """
    required_topic_names = list(MATERIAL_DECISION_TOPICS)
    if state.blueprint is not None and state.blueprint.migration_steps:
        required_topic_names.append(MIGRATION_MATERIAL_TOPIC)
    if not required_topic_names:
        return []

    return [
        topic.topic
        for topic in state.decision_topics
        if topic.topic in required_topic_names and not topic.evidence_ids
    ]


def _actionable_adrs_without_kb_evidence(
    state: ArchitectState,
    adrs_without_kb_evidence: list[str],
    material_topics_without_evidence: list[str],
) -> list[str]:
    """`adrs_without_kb_evidence`, minus any ADR whose empty citation is
    FULLY explained by a required material topic that itself retrieved
    ZERO evidence this run. Pure.

    WHY: without this, an ADR mapped only to a gapped material topic was
    double-reported — once by the generic (refinable) finding below, which
    tells the Architect to "cite a retrieved evidence_id ... if one
    genuinely supports" it, and once by `_check_material_topic_evidence_gaps`'s
    honest `non_refinable` finding, which says no such evidence exists to
    cite. The refinable copy invited a refine round that could never
    succeed — exactly the wasted-iteration bug this exists to close.

    An ADR mapped to the gap topic ALONGSIDE another topic that DOES have
    evidence is NOT excluded — it genuinely could have cited that other
    topic's evidence, so the generic finding still legitimately applies.
    An ADR with NO topic mapping at all is also not excluded here — its
    absence is `_check_adr_topic_mapping_presence`'s finding to explain,
    not this filter's to silently drop.
    """
    if not material_topics_without_evidence or not adrs_without_kb_evidence:
        return adrs_without_kb_evidence
    gap_topics = set(material_topics_without_evidence)
    topic_by_id = {topic.id: topic for topic in state.decision_topics}
    adr_by_label = {(adr.id or adr.title): adr for adr in state.adrs}

    actionable: list[str] = []
    for label in adrs_without_kb_evidence:
        adr = adr_by_label.get(label)
        mapped_ids = _adr_topic_ids(adr) if adr is not None else set()
        mapped_names = {
            topic_by_id[topic_id].topic
            for topic_id in mapped_ids
            if topic_id in topic_by_id
        }
        if mapped_names and mapped_names <= gap_topics:
            continue  # fully explained by an honest per-topic evidence gap
        actionable.append(label)
    return actionable


# ─────────────────────────────────────────────────────────────────────────
# Rubric item 2: constraints
# ─────────────────────────────────────────────────────────────────────────

def _check_constraints(
    state: ArchitectState,
) -> tuple[dict[str, bool], dict[str, bool], dict[str, list[str]]]:
    """Check explicit context constraints against generated design artifacts.

    Returns (applicable, covered, uncovered). `uncovered` maps each group to
    the exact constraint strings that failed the evidence match.

    Functional and non-functional requirements are deliberately not judged by
    prose overlap here. Their deterministic contract is structural:
    requirement_catalog -> Feature.related_requirement_ids -> Components/ADRs.
    Existing-system alignment is likewise owned by repository grounding,
    technology-drift, component-identity, and migration checks. Treating these
    fields as bags of words caused semantically valid designs to fail until
    they copied phrases such as "SQL" or an entire NFR verbatim.

    This check therefore owns only the explicit context constraints whose
    presence can be verified honestly in text: cloud, budget, and compliance.
    The Reviewer's qualitative flaw-detection judgment still decides whether
    the resulting architecture substantively satisfies the requirements.
    """

    context = state.context_record
    design_text = _design_text(state)

    requirements: dict[str, list[str]] = {
        group: [] for group in CONSTRAINT_GROUPS
    }
    if context is not None:
        requirements["cloud"] = _non_empty([context.cloud_provider])
        requirements["budget"] = _non_empty([context.budget])
        requirements["compliance"] = _non_empty(
            context.compliance_requirements
        )

    stopwords = {
        "and", "the", "for", "with", "must", "should", "system", "support",
        "provide", "users", "user", "existing", "application", "service",
        "can", "requirement", "requirements", "constraint", "constraints",
        "cloud", "budget", "compliance", "compliant", "solution", "handle",
    }

    def contains_term(term: str) -> bool:
        """Match a requirement term without accepting incidental substrings."""

        return re.search(
            rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])",
            design_text,
        ) is not None

    def evidenced(requirement: str) -> bool:
        normalised = requirement.strip().lower()
        if not normalised:
            return False
        if contains_term(normalised):
            return True
        if any(
            contains_term(alias)
            for alias in CONSTRAINT_ALIASES.get(normalised, ())
        ):
            return True
        tokens = [
            token
            for token in re.findall(r"[a-z0-9][a-z0-9+.#]*", normalised)
            if token not in stopwords
            and (len(token) >= 3 or any(character.isdigit() for character in token))
        ]
        if tokens:
            matches = sum(contains_term(token) for token in tokens)
            if matches / len(tokens) >= 0.5:
                return True
        return False

    applicable = {
        group: bool(values)
        for group, values in requirements.items()
    }
    uncovered = {
        group: [
            requirement for requirement in values
            if not evidenced(requirement)
        ]
        for group, values in requirements.items()
    }
    covered = {
        group: bool(values) and not uncovered[group]
        for group, values in requirements.items()
    }
    return applicable, covered, uncovered


# ─────────────────────────────────────────────────────────────────────────
# Rubric item: Context requirement -> Feature coverage
#
# `_check_constraints` proves the final DESIGN addresses a requirement via
# loose token overlap (>=50% of a requirement's significant tokens appearing
# anywhere in the design text) — evidence enough to catch a design that never
# mentions "GDPR", but too loose to prove a requirement was actually MODELED:
# a requirement like "Product reviews" passes that check on the incidental
# word "Product" appearing elsewhere, even when reviews were never modeled as
# a feature at all. This check is orthogonal and STRUCTURAL: every explicit
# functional/non-functional requirement must be named, in some Feature's
# `related_requirement_ids` — the field that already exists for exactly this
# purpose. Cloud/budget/compliance/existing-system are design constraints,
# not features, and stay out of scope here.
#
# CATALOG, NOT RAW TEXT (contract fix). The Architect's own contract for
# `related_requirement_ids` is IDs ("NFR-001"), and existing Architect tests
# already used that shape — a raw-text-only Reviewer rule was never
# satisfiable by a correct Architect output. `requirement_catalog` is the ONE
# deterministic ID scheme both sides read: the Architect prompt shows it, the
# Architect writer boundary validates phase-1 output against it BEFORE phase
# 2 ever runs (see architect.py), and this check reads the SAME catalog. A
# genuinely missing mapping can therefore no longer reach the Reviewer from a
# normal run — it is caught, fail-fast, at the source. This check still
# exists for persisted/legacy/directly-built states the Architect boundary
# never validated (and for the unit tests that build them directly).
# ─────────────────────────────────────────────────────────────────────────


class RequirementCatalogEntry(BaseModel):
    """One deterministic entry in a Context Record's requirement catalog.

    The single identity both the Architect and the Reviewer read: the
    Architect prompt lists it, the Architect writer boundary validates
    phase-1 `related_requirement_ids` against it, and this module's
    coverage check reads the same catalog — one contract, never three.
    """

    id: str
    text: str
    kind: Literal["functional", "non_functional"]


def requirement_catalog(
    context: ContextRecord | None,
) -> list[RequirementCatalogEntry]:
    """Deterministic FR-xxx/NFR-xxx catalog from the frozen Context Record. Pure.

    IDs are assigned by LIST ORDER, independently per kind:
    `functional_requirements` numbers FR-001, FR-002, ...;
    `non_functional_requirements` numbers NFR-001, NFR-002, ... Cloud
    provider, budget, compliance, and existing-system fields are DESIGN
    CONSTRAINTS, not modeled capabilities, and never get a catalog entry.

    DUPLICATE TEXT, anywhere in the record (either field): the first
    occurrence gets the entry; a later occurrence with byte-identical
    (whitespace-stripped) text is folded into it rather than minted a
    second, ambiguous ID — two IDs naming the same requirement would let a
    Feature "cover" only one of them and still look complete.
    """

    if context is None:
        return []

    seen: set[str] = set()
    entries: list[RequirementCatalogEntry] = []

    counter = 0
    for requirement in context.functional_requirements:
        text = requirement.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        counter += 1
        entries.append(
            RequirementCatalogEntry(id=f"FR-{counter:03d}", text=text, kind="functional")
        )

    counter = 0
    for requirement in context.non_functional_requirements:
        text = requirement.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        counter += 1
        entries.append(
            RequirementCatalogEntry(
                id=f"NFR-{counter:03d}", text=text, kind="non_functional"
            )
        )

    return entries


def _check_requirement_feature_coverage(state: ArchitectState) -> list[str]:
    """Catalog requirements with no Feature covering them, rendered
    "<ID> - <text>" for a reader/refinement instruction. Pure.

    Coverage is by the SAME deterministic catalog `requirement_catalog`
    gives the Architect: a Feature's `related_requirement_ids` entry counts
    when it is the entry's canonical ID ("FR-001") OR — compatibility for
    persisted/legacy/directly-built states written before the catalog
    contract existed — the entry's exact (whitespace-normalised) text.
    Never fuzzy/token matching, and nothing is inferred onto the "closest"
    feature.
    """

    catalog = requirement_catalog(state.context_record)
    if not catalog:
        return []

    mapped = {
        value.strip()
        for feature in state.features
        for value in feature.related_requirement_ids
        if value.strip()
    }
    return [
        f"{entry.id} - {entry.text}"
        for entry in catalog
        if entry.id not in mapped and entry.text not in mapped
    ]


# ─────────────────────────────────────────────────────────────────────────
# Rubric item: brownfield implementation-language/framework drift
#
# Run 20260822T164736Z-1a15e2e4: the detected stack was Python/Django, the
# technical view said "new Go-based microservices", and the review PASSED —
# the conservatism rule was advisory prompt text with no code-owned
# counterpart. This check is deliberately NARROW: only substantial
# implementation-LANGUAGE drift for new internal services counts. Managed
# infrastructure (SQS/SNS/Redis/API Gateway/…), databases, and frameworks
# of the SAME primary language are not drift; a justified deviation — an
# explicit ADR/rationale that names the new language AND the existing one
# together — passes. Nothing is banned; only unjustified drift blocks.
# ─────────────────────────────────────────────────────────────────────────

# Well-known implementation-language tokens → canonical name. Detection is
# token-shaped, not tied to any one ecosystem; the map exists because a
# language needs a canonical display name for the finding text.
_LANGUAGE_TOKENS: dict[str, str] = {
    "python": "Python", "django": "Python", "flask": "Python",
    "fastapi": "Python",  # frameworks indicate their implementation language
    "java": "Java", "spring": "Java", "kotlin": "Kotlin", "scala": "Scala",
    "go": "Go", "golang": "Go",
    "javascript": "JavaScript", "typescript": "TypeScript",
    "node": "JavaScript", "nodejs": "JavaScript", "express": "JavaScript",
    "react": "JavaScript", "vue": "JavaScript", "angular": "TypeScript",
    "c#": "C#", ".net": "C#", "dotnet": "C#", "rust": "Rust", "ruby": "Ruby",
    "rails": "Ruby", "php": "PHP", "laravel": "PHP",
}
# Never treated as implementation-language drift: infrastructure/managed
# services, data stores, and generic descriptors — regardless of how they
# are phrased in the design. A database or a queue is not a language.
_NON_LANGUAGE_TOKENS = frozenset({
    "aws", "sns", "sqs", "eventbridge", "kinesis", "lambda", "gateway",
    "redis", "elasticsearch", "aurora", "rds", "postgres", "postgresql",
    "mysql", "mongodb", "dynamodb", "kafka", "rabbitmq", "docker",
    "kubernetes", "s3", "cloudfront", "cognito", "iam", "service", "api",
})


def _design_technology_text(state: ArchitectState) -> str:
    """All technology-bearing design text: component technology choices,
    technical view, pattern, and every ADR title/decision/rationale."""
    parts: list[str] = []
    for component in state.components:
        parts.extend(component.technology_choices)
        parts.append(component.component_type)
    if state.blueprint is not None:
        parts.append(state.blueprint.selected_pattern)
        parts.append(state.blueprint.technical_view)
    for adr in state.adrs:
        parts.extend((adr.title, adr.decision, adr.rationale))
    return " ".join(_non_empty(parts))


def _languages_in(text: str) -> set[str]:
    """Canonical implementation-language names present in `text`."""
    tokens = re.findall(r"[a-z#][a-z0-9#./+]*", text.lower())
    languages: set[str] = set()
    for token in tokens:
        if token in _LANGUAGE_TOKENS and token not in _NON_LANGUAGE_TOKENS:
            languages.add(_LANGUAGE_TOKENS[token])
    return languages


def _check_technology_drift(
    state: ArchitectState,
) -> dict[str, list[str]]:
    """New implementation languages with no explicit deviation rationale.

    Returns {new_language: [where_it_appears]}. Empty when the design stays
    in the detected stack, justifies its deviation, or no stack was
    detected. Pure.
    """
    repo = state.repo_representation
    if repo is None:
        return {}
    existing = _languages_in(
        " ".join(
            [*repo.structure.tech_stack.languages.keys(),
             *repo.structure.tech_stack.frameworks]
        )
    )
    if not existing:
        return {}  # no clear detected stack → nothing to drift from

    design_text = _design_technology_text(state)
    proposed = _languages_in(design_text)
    new_languages = proposed - existing
    if not new_languages:
        return {}

    # JUSTIFICATION: an ADR/rationale that names the new language AND the
    # existing language together, in the same artifact, is an explicit
    # comparison of the two — the conservatism rule's requirement. A
    # pattern title naming a language alone is not a rationale.
    justifications: dict[str, list[str]] = {}
    for adr in state.adrs:
        adr_text = " ".join(
            _non_empty([adr.title, adr.context, adr.decision, adr.rationale])
        ).lower()
        for language in new_languages:
            if language.lower() in adr_text and any(
                existing_name.lower() in adr_text
                for existing_name in existing
            ):
                justifications.setdefault(language, []).append(adr.id)

    unjustified = {
        language: where
        for language, where in (
            (language, sorted(
                [component.name for component in state.components
                 if language.lower() in " ".join(
                     component.technology_choices
                 ).lower()]
                or ["design text"]
            ))
            for language in new_languages
        )
        if language not in justifications
    }
    return unjustified


# ─────────────────────────────────────────────────────────────────────────
# Rubric item: cross-artifact data-flow participant resolution (invariant A)
# + directional renderability (invariant A2)
#
# Run 20260822T170536Z-04942cfe passed review while `data_flows` referenced
# "Product Service" and "Product Database" that had no Component
# Description — the ownership check let them through on a text-disposition
# sentence that happened to contain the right tokens. This invariant is
# independent of prose dispositions: an arrow-flow ENDPOINT either matches
# a Component Description (the catalog a reader and the diagram see), is a
# clearly-external participant, or produces a finding. Nothing is inferred,
# guessed, or silently added.
#
# A later final run passed with prose-only flows ("API Gateway routes
# incoming traffic to ..."), which the diagram cannot render at all — the
# client-facing Overview showed recorded text instead of a target
# architecture. Every non-empty flow must therefore be parseable as a
# directional edge under the ONE shared grammar
# (`pipeline.flow_syntax.split_directional_flow` — the renderer's own
# parser, extracted so judge and renderer cannot drift), and only then do
# its endpoints reach the participant-resolution check below.
# ─────────────────────────────────────────────────────────────────────────

# Whole-endpoint external ACTORS and generic external INFRASTRUCTURE /
# third-party providers. Technology-generic only — no benchmark names.
_EXTERNAL_ACTOR_ENDPOINTS = frozenset({
    "client", "customer", "customers", "user", "users", "browser",
    "admin", "administrator", "operator", "partner", "public internet",
    "internet", "mobile app", "web app",
})
_EXTERNAL_INFRA_ENDPOINTS = frozenset({
    "postgres", "postgresql", "mysql", "mongodb", "mongo", "redis",
    "memcached", "elasticsearch", "kafka", "rabbitmq", "sqs", "sns",
    "eventbridge", "kinesis", "s3", "dynamodb", "aurora", "rds",
    "stripe", "paypal", "ses", "sns+", "external payment provider",
})
# Markers that make a longer endpoint CLEARLY external regardless of the
# rest of its wording ("Third-party Payment Provider", "External CRM").
_CLEARLY_EXTERNAL_MARKERS = ("third-party", "third party", "external")


def _fold(text: str) -> str:
    """Case-insensitive, whitespace-collapsed form for identity matching."""

    return " ".join(text.lower().split())


def _is_external_participant(endpoint: str) -> bool:
    """Is this flow endpoint a clearly-external actor/system/provider?"""
    folded = _fold(endpoint)
    if folded in _EXTERNAL_ACTOR_ENDPOINTS or folded in _EXTERNAL_INFRA_ENDPOINTS:
        return True
    return any(marker in folded for marker in _CLEARLY_EXTERNAL_MARKERS)


def _check_flow_directionality(state: ArchitectState) -> list[str]:
    """Non-empty data flows the shared renderer grammar cannot parse. Pure.

    A prose flow ("API Gateway routes incoming traffic to the Order
    Service") carries no arrow, so the diagram has no edge to draw and the
    client sees recorded text instead of an architecture. Such a flow is
    structurally unrenderable — reported verbatim so refinement can rewrite
    it into `A → B` form with exact component names. Parsing is the
    renderer's own (`pipeline.flow_syntax`), never a second grammar.
    """
    if state.blueprint is None:
        return []
    return [
        flow.strip()
        for flow in state.blueprint.data_flows
        if str(flow or "").strip()
        and split_directional_flow(flow) is None
    ]


def _check_flow_participants(
    state: ArchitectState,
) -> dict[str, list[str]]:
    """Arrow-flow endpoints with no Component Description and no external
    marker, mapped to the flows that reference them. Pure.

    Resolution is EXACT against Component Description names under the SAME
    canonical identity `_component_key` gives the target-service ownership
    and ADR checks (a trailing lifecycle qualifier like "(New)" ignored) —
    never fuzzy substring matching. A Blueprint.components-only name (listed
    but never described) does NOT resolve: describing it or marking it
    external is the fix. Only DIRECTIONAL flows contribute endpoints (the
    unrenderable ones are flagged whole by `_check_flow_directionality`).
    """
    if state.blueprint is None:
        return {}
    described = {
        _component_key(c.name) for c in state.components if _component_key(c.name)
    }
    unresolved: dict[str, set[str]] = {}
    for flow in _non_empty(state.blueprint.data_flows):
        parsed = split_directional_flow(flow)
        if parsed is None:
            continue
        endpoints, _label = parsed
        for endpoint in endpoints:
            if _component_key(endpoint) in described or _is_external_participant(
                endpoint
            ):
                continue
            unresolved.setdefault(endpoint, set()).add(flow)
    return {
        endpoint: sorted(flows) for endpoint, flows in unresolved.items()
    }


# ─────────────────────────────────────────────────────────────────────────
# Rubric item: migration disposition for new internal target services
# (invariant B)
#
# Same run: "Payment Service" existed as a new internal target component
# while the migration plan named only the gateway and two extractions — a
# service with no stated migration path. The Architect prompt now demands a
# disposition; this check makes the absence of one a code-owned finding.
# Only application SERVICES are in scope (databases, brokers, gateways and
# other infrastructure follow their owning service's step); the check is
# brownfield-only — no repository evidence, no legacy to migrate from.
# ─────────────────────────────────────────────────────────────────────────

# "Deferred to a later phase" / "already existing" — the two migration
# dispositions with no existing phrase pattern. "External" reuses
# `_EXTERNAL_DISPOSITION_RE` below rather than redefining that vocabulary a
# second time (see module header).
_MIGRATION_PHASE_RE = re.compile(
    r"\b(?:later|future|next)\s+phase\b|\bphase\s+\d+\b|"
    r"\balready\s+exist(?:s|ing)?\b",
    re.IGNORECASE,
)

# "Retained in legacy" — same retention-verb vocabulary as
# `_RETENTION_VERB_RE`, but restructured so a bare retention verb can never
# qualify alone. A manual review found that "Payment Service retains PCI
# controls during migration" and "keeps its API unchanged" were passing as
# dispositions merely because SOME retention verb appeared somewhere in the
# sentence. A retention verb is only a migration disposition when it is
# followed by a LOCATIVE preposition ("in"/"within"/"inside"/"part of") that
# places the service INSIDE the legacy system — "remains compatible WITH
# legacy clients" carries the word "legacy" too, but nothing there places
# the service in it, so it must not qualify either.
_LEGACY_PLACEMENT_RE = re.compile(
    r"\b(?:remain(?:s|ed|ing)?|stay(?:s|ed|ing)?|keep(?:s|t|ing)?|"
    r"retain(?:s|ed|ing)?)\b\s*(?:is\s+|are\s+)?"
    r"(?:in|within|inside|part\s+of)\s+(?:the\s+)?(?:current\s+)?"
    r"(?:legacy|monolith)\w*\b"
    r"|\bleft\s+in\s+(?:the\s+)?(?:legacy|monolith)\w*\b"
    r"|\bserved\s+by\s+(?:the\s+)?(?:legacy|monolith)\w*\b"
    r"|\bnot\s+(?:yet\s+)?(?:be\s+)?extracted\b(?:\s+\w+){0,3}?\s+"
    r"(?:the\s+)?(?:legacy|monolith)\w*\b"
    r"|\bcontinue[sd]?\s+to\s+(?:be\s+)?(?:live|run|reside)\b"
    r"(?:\s+\w+){0,3}?\s+(?:the\s+)?(?:legacy|monolith)\w*\b",
    re.IGNORECASE,
)


def _has_migration_disposition_phrase(sentence: str) -> bool:
    """Does this sentence carry an explicit migration-disposition phrase?

    `_LEGACY_PLACEMENT_RE` deliberately does NOT reduce to "legacy/monolith
    noun co-occurs with a retention verb" — see its docstring above.
    "Deferred to Phase 2" has nothing to do with the legacy system and is
    matched independently by `_MIGRATION_PHASE_RE`.
    """
    return bool(
        _LEGACY_PLACEMENT_RE.search(sentence)
        or _EXTERNAL_DISPOSITION_RE.search(sentence)
        or _MIGRATION_PHASE_RE.search(sentence)
    )


def _repo_evidence_text(state: ArchitectState) -> str:
    """What the repository analysis actually says about the legacy system.
    Empty when no real repo evidence exists (greenfield or stub repo)."""
    repo = state.repo_representation
    if repo is None:
        return ""
    parts = [
        repo.behavior.overview,
        " ".join(p.name + " " + p.role for p in repo.behavior.partitions),
        " ".join(repo.structure.tech_stack.languages.keys()),
        " ".join(repo.structure.tech_stack.frameworks),
    ]
    return " ".join(_non_empty(parts))


def _repo_partition_service_names(state: ArchitectState) -> set[str]:
    """Folded partition NAMES from the repo analysis. Pure.

    A repo partition is documented as a partition along the repo's OWN
    structure — a module, app, package, or capability, never automatically
    a service boundary (the same distinction the brownfield design prompt
    already draws for the Architect). A partition literally named for the
    standalone service ("Payment Service") is strong evidence that service
    exists; a partition named only for the capability it happens to serve
    ("Payment") is NOT — a manual review found that subset-token matching
    let "Payment" satisfy "Payment Service", which is exactly the
    capability-vs-service conflation this check exists to reject. Matching
    is therefore EXACT (folded name equality), never subset containment,
    and partition `role` prose is never consulted.
    """
    repo = state.repo_representation
    if repo is None:
        return set()
    return {
        _fold(partition.name)
        for partition in repo.behavior.partitions
        if partition.name.strip()
    }


def _check_migration_disposition(
    state: ArchitectState,
) -> list[str]:
    """New internal application services with no explicit migration
    disposition, by name. Pure.

    Identity reuses `_component_key`/`_name_tokens` — the same tokens the
    target-service-ownership check (invariant, above) resolves names with,
    so the writer's canonical spelling and this judge agree on what a
    service "is" without a second matcher.

    A service is EXISTING only on STRONG evidence: a repo partition
    EXACTLY named for it (`_repo_partition_service_names`), or an explicit
    design disposition ("already existing"). Generic capability prose ("the
    legacy monolith handles payments") is NOT that evidence — it proves the
    CAPABILITY exists, not that a standalone "Payment Service" does, and
    treating it as proof was a manual review's first false-negative finding
    (a monolith mentioning a capability let a same-named new service
    through with no migration step at all); a bare capability-named
    partition ("Payment") is the same error one field over and is rejected
    the same way. A new service is otherwise DISPOSED when some migration
    step names all its identity tokens, or a design sentence names them
    together with an explicit disposition phrase (deferred / later phase /
    retained in legacy / external / already existing). Anything else is
    silently introduced. The legacy system's own component (named
    "...Monolith" or "...Legacy...") is never a candidate: it is what is
    being migrated FROM, not a newly introduced target service.
    """
    repo_text = _repo_evidence_text(state)
    if not repo_text:
        return []  # no legacy evidence → not a brownfield disposition case

    partition_service_names = _repo_partition_service_names(state)
    step_token_sets = [
        set(_name_tokens(" ".join(_non_empty([
            step.title, step.objective, step.coexistence_or_data_strategy,
            step.exit_condition, *step.changes,
        ]))))
        for step in (state.blueprint.migration_steps if state.blueprint else [])
    ]
    design_sentences = _design_sentences(state)

    undisposed: list[str] = []
    for component in state.components:
        if "service" not in component.component_type.lower():
            continue  # infrastructure/databases ride their owning step
        if _LEGACY_NOUN_RE.search(component.name.lower()):
            continue  # the legacy system itself, not a new target service
        key = set(_component_key(component.name))
        if not key:
            continue  # name carries no identity beyond "Service" itself
        if _fold(component.name) in partition_service_names:
            continue  # a structural repo partition is EXACTLY this service
        if any(key <= step_tokens for step_tokens in step_token_sets):
            continue  # named in a migration step
        disposed = any(
            key <= set(_name_tokens(sentence))
            and _has_migration_disposition_phrase(sentence)
            for sentence in design_sentences
        )
        if not disposed:
            undisposed.append(component.name)
    return sorted(undisposed)


# ─────────────────────────────────────────────────────────────────────────
# Rubric item: migration targets must resolve to real components
# (reverse of invariant B)
#
# A fresh final E2E run passed review with a migration step reading
# "Extract Order, Inventory, and Shipping Services" while the component
# catalog contained no Shipping Service — the disposition check covers
# declared components with no migration path, but nothing covered migration
# steps promising services the target architecture never declares. This is
# the reverse direction: an INTRODUCTION sentence inside a migration step
# (extract/create/introduce/deploy/build/...) that names a concrete service
# commits the design to that service existing.
#
# The action-verb vocabulary itself is the ONE shared tuple
# `_MIGRATION_ACTION_VERBS`, defined once beside `_NON_NAME_WORDS` (which
# unions it) and compiled here into `_MIGRATION_INTRODUCTION_RE` — the
# matcher and the never-a-name blocklist cannot drift apart.
# ─────────────────────────────────────────────────────────────────────────

_MIGRATION_INTRODUCTION_RE = re.compile(
    r"\b(?:" + "|".join(_MIGRATION_ACTION_VERBS) + r")\b",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────
# Compound-name enumeration references (regression, real E2E run)
#
# `_extract_service_bases`'s list grammar splits an enumeration on ANY
# whitespace, which is right for a run of genuinely separate one-word names
# ("Order, Payment, and Notification services") but wrong for a multi-word
# compound name enumerated alongside another: "Extract Order Management and
# Fulfillment Services" — against a catalog holding "Order Management
# Service" and "Shipping and Fulfilment Service" — flattened to THREE
# one-word candidates ("Order", "Management", "Fulfillment"), and "Order
# Service" / "Fulfillment Service" were then reported missing even though
# the migration text plainly names the two existing components.
#
# The fix is structural, not a name lookup: split the enumeration ONLY on
# its explicit ',' / 'and' separators, so a bare space between two
# TitleCase words ("Order Management") stays part of ONE candidate phrase
# instead of being torn into two independent ones. This is the same
# "compare the extraction against the existing catalog" principle
# `canonicalize_data_flow_endpoints` already uses for flow endpoints —
# applied here at the extraction step, not as a second matching pass.
# `_extract_service_bases` itself is untouched (its other caller,
# `_target_service_references`, keeps its existing per-word behavior).
# ─────────────────────────────────────────────────────────────────────────
_MIGRATION_LIST_DELIM_RE = re.compile(r"\s*,\s*(?:and\s+)?|\s+and\s+")


def _extract_migration_service_phrases(text: str) -> list[str]:
    """Like `_extract_service_bases`, but an enumeration's comma/'and'
    segments are kept as whole (possibly multi-word) candidate phrases
    instead of being flattened word by word. Migration-check only.
    """
    phrases: set[str] = set()
    for match in _TARGET_SINGULAR_RE.finditer(text):
        base = match.group(1)
        if base.lower() not in _NON_NAME_WORDS and base.lower() != "service":
            phrases.add(base)
    for match in _TARGET_LIST_RE.finditer(text):
        enumeration = match.group(1)
        is_explicit_list = ("," in enumeration) or (
            re.search(r"\s+and\s+", enumeration) is not None
        )
        if not is_explicit_list:
            continue
        segments: list[str] = []
        for chunk in _MIGRATION_LIST_DELIM_RE.split(enumeration):
            words = [
                word for word in chunk.split()
                if word.lower() not in _NON_NAME_WORDS
            ]
            if words:
                segments.append(" ".join(words))
        if len(segments) >= 2:
            phrases.update(segments)
    return sorted(phrases)


def _resolve_unambiguous_component(
    key: tuple[str, ...],
    component_entries: list[tuple[tuple[str, ...], bool]],
) -> bool:
    """True when `key` resolves to EXACTLY ONE existing component — by
    exact identity, or as the trailing tokens of a non-composite
    component's identity (the same rule the ownership check uses).

    Counting matters as much as matching: two components that both end in
    'Fulfilment' (e.g. 'Shipping and Fulfilment Service' and 'Warehouse
    Fulfilment Service') must NOT let a bare 'Fulfillment Service' mention
    silently pick one — zero matches (nothing declares it) and multiple
    matches (which one is meant is genuinely unclear) are both unresolved.
    """
    matches = sum(
        1
        for existing, composite in component_entries
        if existing == key or (not composite and existing[-len(key):] == key)
    )
    return matches == 1


def _check_migration_targets(
    state: ArchitectState,
) -> dict[str, list[str]]:
    """Services a migration step promises to introduce that resolve to no
    Component Description, mapped to the step titles naming them. Pure.

    Extraction is NARROW by construction: only sentences carrying an
    introduction verb are scanned, and only names
    `_extract_migration_service_phrases` accepts — an explicit
    "<Name> Service" or an enumeration like "Order, Inventory, and Shipping
    Services" (a multi-word segment inside an enumeration, e.g. "Order
    Management", stays one candidate phrase — see the note above). Incidental
    prose ("the shipping module") carries no Service suffix and is never a
    candidate. A candidate is excused when it resolves to EXACTLY ONE
    described component under the SAME canonical identity as the ownership
    check (including the composite exactness rule; an ambiguous match
    between two or more components is NOT excused — see
    `_resolve_unambiguous_component`), or when the design gives it an
    explicit disposition — legacy retention, external, ownership, or
    deferred to a later phase — under the SAME disposition vocabulary the
    other migration checks use.
    """
    if state.blueprint is None:
        return {}
    steps = state.blueprint.migration_steps
    if not steps:
        return {}

    component_entries = [
        (_component_key(component.name), "/" in component.name)
        for component in state.components
        if _component_key(component.name)
    ]
    owner_word_sets = [set(key) for key, _composite in component_entries]
    # Disposition sentences include the migration steps' own text: a step
    # may introduce one service and retain another IN THE SAME SENTENCE
    # ("extract Order; the Cart Service remains in the legacy monolith"),
    # and `_design_sentences` does not read migration_steps.
    sentences = [
        sentence.lower() for sentence in _design_sentences(state)
    ]
    for step in steps:
        sentences.extend(
            sentence.lower()
            for sentence in _sentence_spans(" ".join(_non_empty([
                step.title, step.objective, step.coexistence_or_data_strategy,
                step.exit_condition, *step.changes,
            ])))
        )

    unresolved: dict[str, set[str]] = {}
    for step in steps:
        step_fields = _non_empty([
            step.title, step.objective, step.coexistence_or_data_strategy,
            step.exit_condition, *step.changes,
        ])
        for sentence in _sentence_spans(" ".join(step_fields)):
            if not _MIGRATION_INTRODUCTION_RE.search(sentence):
                continue
            for base in _extract_migration_service_phrases(sentence):
                display = f"{base} Service"
                key = _component_key(display)
                if not key:
                    continue
                if _resolve_unambiguous_component(key, component_entries):
                    continue  # the target exists in the catalog, unambiguously
                if _has_explicit_disposition(key, sentences, owner_word_sets):
                    continue  # legacy/external/ownership, same rule as ownership
                if any(
                    set(key) <= set(_name_tokens(s))
                    and _has_migration_disposition_phrase(s)
                    for s in sentences
                ):
                    continue  # explicitly deferred / later phase / already existing
                unresolved.setdefault(display, set()).add(step.title.strip())
    return {
        display: sorted(titles) for display, titles in unresolved.items()
    }


# ─────────────────────────────────────────────────────────────────────────
# Main deterministic validation
# ─────────────────────────────────────────────────────────────────────────

def run_deterministic_checks(
    state: ArchitectState,
    *,
    allow_prose_fallback: bool = False,
) -> DeterministicChecks:
    """Run all code-checkable rubric items and create ReviewIssues."""

    present, missing = _check_artifacts(state)

    traceability = _check_traceability(
        state,
        allow_prose_fallback=allow_prose_fallback,
    )
    features_without_component = traceability["features_without_component"]
    components_without_feature = traceability["components_without_feature"]
    components_without_adr = traceability["components_without_adr"]
    unowned_target_services = _check_target_service_ownership(state)
    # An unowned target service is a traceability failure: the artifact set
    # claims a service that no Component Description traces to.
    traceability["unowned_target_services"] = unowned_target_services

    unjustified_language_drift = _check_technology_drift(state)

    unresolved_flow_participants = _check_flow_participants(state)
    unrenderable_data_flows = _check_flow_directionality(state)
    services_without_migration_disposition = _check_migration_disposition(state)
    unresolved_migration_targets = _check_migration_targets(state)

    malformed_adrs, incomplete_adrs, duplicate_numbers = _check_adrs(state)
    constraints_applicable, constraints_covered, constraints_uncovered = (
        _check_constraints(state)
    )
    requirements_without_feature = _check_requirement_feature_coverage(state)
    # A missing requirement -> Feature mapping is a traceability failure too:
    # folding it into `traceability` (the same dict `unowned_target_services`
    # is merged into above) keeps `score_traceability` internally consistent
    # with the HIGH issue below — a design can no longer score 2/2 while
    # carrying a blocking traceability finding.
    traceability["requirements_without_feature"] = requirements_without_feature
    invalid_sources, score_source_integrity = _check_source_integrity(state)
    (
        adrs_without_kb_evidence,
        invalid_evidence_ids,
        kb_evidence_gap,
        score_kb_evidence_grounding,
        kb_evidence_gap_non_refinable,
    ) = _check_kb_evidence_grounding(state)

    invalid_adr_decision_topic_ids, material_decisions_without_adr = (
        _check_material_decision_coverage(state)
    )
    # A material decision with no governing ADR is a traceability failure,
    # exactly like `unowned_target_services`/`requirements_without_feature`
    # above — a design cannot score 2/2 traceability while a recommendation
    # it actually made is untraceable to any ADR.
    traceability["material_decisions_without_adr"] = material_decisions_without_adr

    adr_evidence_outside_mapped_topics = _check_adr_evidence_topic_provenance(state)
    adrs_without_decision_topic_mapping = _check_adr_topic_mapping_presence(state)
    material_topics_without_evidence = _check_material_topic_evidence_gaps(state)
    if (
        invalid_adr_decision_topic_ids
        or adr_evidence_outside_mapped_topics
        or adrs_without_decision_topic_mapping
        or material_topics_without_evidence
    ):
        # Fabricated topic references, off-topic citations, an ADR with no
        # topic mapping at all, and a required topic with zero retrieved
        # evidence are all the same family of failure as `invalid_evidence_ids`
        # above: the grounding claim ("every material decision is traceable
        # to literature retrieved for ITS OWN topic") is false, so the score
        # cannot read 2 regardless of what `_check_kb_evidence_grounding`
        # alone computed.
        score_kb_evidence_grounding = 0

    invented_quantitative_targets = _check_quantitative_targets(state)

    repository_expected = bool(state.initial_request.repo_url.strip())
    repository_available = state.repo_representation is not None

    # ── Artifact completeness score ──────────────────────────────────────
    if not all(present.values()):
        score_artifacts = 0
    elif missing:
        score_artifacts = 1
    else:
        score_artifacts = 2

    # ── Constraint coverage score ────────────────────────────────────────
    applicable_groups = [
        group
        for group, is_applicable in constraints_applicable.items()
        if is_applicable
    ]
    number_covered = sum(
        constraints_covered[group]
        for group in applicable_groups
    )

    if not applicable_groups or number_covered == len(applicable_groups):
        score_constraints = 2
    elif number_covered:
        score_constraints = 1
    else:
        score_constraints = 0

    # ── Traceability score ───────────────────────────────────────────────
    traceability_failures = [
        value
        for value in traceability.values()
        if value
    ]
    if not state.features or not state.components or not state.adrs:
        score_traceability = 0
    elif not traceability_failures:
        score_traceability = 2
    elif (
        len(features_without_component) < len(state.features)
        or len(components_without_feature) < len(state.components)
    ):
        score_traceability = 1
    else:
        score_traceability = 0

    # ── ADR presence score ───────────────────────────────────────────────
    if (
        not state.adrs
        or len(malformed_adrs) == len(state.adrs)
    ):
        score_adr = 0
    elif (
        malformed_adrs
        or incomplete_adrs
        or duplicate_numbers
        or components_without_adr
    ):
        score_adr = 1
    else:
        score_adr = 2

    # ── Convert deterministic failures into ReviewIssues ────────────────
    issues: list[ReviewIssue] = []

    def add_issue(
        severity: str,
        category: str,
        finding: str,
        evidence: str,
        suggested_fix: str,
        *,
        non_refinable: bool = False,
    ) -> None:
        issues.append(
            ReviewIssue(
                id=f"DET-{len(issues) + 1}",
                severity=severity,
                category=category,
                finding=finding,
                evidence=evidence,
                suggested_fix=suggested_fix,
                requires_refinement=severity == "high",
                non_refinable=non_refinable,
            )
        )

    absent = [
        artifact_name
        for artifact_name, exists in present.items()
        if not exists
    ]

    if absent:
        add_issue(
            "high",
            "completeness",
            (
                "Required artifact(s) missing: "
                f"{', '.join(absent)}."
            ),
            f"artifacts_present={present}",
            (
                "Produce every required artifact: Context Record, "
                "Features, Blueprint, ADRs, and Component Descriptions."
            ),
        )

    if missing:
        add_issue(
            "medium",
            "completeness",
            f"{len(missing)} required field(s) are empty.",
            "; ".join(missing),
            "Fill every required field on each architecture artifact.",
        )

    uncovered = [
        group
        for group, is_applicable in constraints_applicable.items()
        if is_applicable and not constraints_covered[group]
    ]

    if uncovered:
        # The exact uncovered requirement strings, per group — the
        # actionable form of this finding. A group label alone ("functional")
        # invites the architect to paraphrase the group again; the string
        # tells it precisely what the design never said (run
        # 20260822T160643Z-8e49d732: 'executing checkouts' failed two refine
        # rounds while the group label was all the instruction carried).
        # The strings LEAD both fields: the refinement instruction clips
        # long fields from the tail, so the payload must survive it.
        uncovered_detail = "; ".join(
            f"{group}: "
            + ", ".join(f"'{requirement}'"
                        for requirement in constraints_uncovered[group])
            for group in uncovered
            if constraints_uncovered[group]
        )
        add_issue(
            "high",
            "constraint",
            (
                "Context constraint group(s) not addressed in the design: "
                f"{', '.join(uncovered)}."
            ),
            (
                f"Unaddressed context constraint(s): {uncovered_detail}"
                if uncovered_detail
                else "No matching structured value or design text found for: "
                f"{', '.join(uncovered)}"
            ),
            (
                "Explain how the design satisfies each context constraint in "
                "a relevant Blueprint, ADR, or Component field; use these "
                "constraints as the repair targets, but do not merely copy "
                "them verbatim: "
                + (uncovered_detail if uncovered_detail else
                   "each applicable context constraint.")
            ),
        )

    if requirements_without_feature:
        # HIGH/blocking: an omitted requirement never entered the modeled
        # feature set at all, which is a stricter failure than the loose
        # constraint-text check above catches (regression: "Product reviews"
        # passed constraint coverage on the word "Product" alone while never
        # becoming a feature). The Reviewer does not guess which Feature
        # should own it — that decision belongs to the Architect.
        add_issue(
            "high",
            "traceability",
            (
                "Context Record requirement(s) are not represented by any "
                "Feature's related_requirement_ids: "
                f"{', '.join(requirements_without_feature)}"
            ),
            (
                "requirements_without_feature="
                f"{requirements_without_feature}"
            ),
            (
                "Add or adjust a Feature to cover each missing requirement "
                "and add its canonical ID (e.g. 'FR-002') to that Feature's "
                "related_requirement_ids."
            ),
        )

    if repository_expected and not repository_available:
        add_issue(
            "high",
            "repo_alignment",
            "A repository was requested but no repository representation is available.",
            f"repo_url={state.initial_request.repo_url}",
            "Ingest the requested repository successfully before reviewing the design.",
        )

    if features_without_component:
        # An uncovered `must` feature is a required capability with no
        # implementation at all — refinement-blocking, like the run that
        # reached DONE at the refinement-cost cap with FEAT-005 (priority
        # must) left uncovered and MEDIUM/non-blocking. Lower-priority
        # features keep the existing non-blocking behavior. The Reviewer
        # only reports the gap; it never invents a component mapping.
        feature_priority = {
            feature.id.strip(): feature.priority
            for feature in state.features
            if feature.id.strip()
        }
        must_uncovered = [
            feature_id
            for feature_id in features_without_component
            if feature_priority.get(feature_id) == "must"
        ]
        other_uncovered = [
            feature_id
            for feature_id in features_without_component
            if feature_id not in must_uncovered
        ]
        if must_uncovered:
            add_issue(
                "high",
                "traceability",
                (
                    "Required (`must`) feature(s) have no implementing "
                    f"component: {', '.join(must_uncovered)}."
                ),
                (
                    "features_without_component="
                    f"{must_uncovered}"
                ),
                (
                    "Add each missing feature ID to at least one "
                    "component's `related_feature_ids`, or introduce the "
                    "component that implements it."
                ),
            )
        if other_uncovered:
            add_issue(
                "medium",
                "traceability",
                (
                    "Feature(s) have no implementing component: "
                    f"{', '.join(other_uncovered)}."
                ),
                (
                    "features_without_component="
                    f"{other_uncovered}"
                ),
                (
                    "Add each missing feature ID to at least one component's "
                    "`related_feature_ids`."
                ),
            )

    if components_without_feature:
        add_issue(
            "medium",
            "traceability",
            (
                "Component(s) do not trace to a valid feature: "
                f"{', '.join(components_without_feature)}."
            ),
            (
                "components_without_feature="
                f"{components_without_feature}"
            ),
            (
                "Populate each component's `related_feature_ids` with an "
                "existing Feature ID."
            ),
        )

    if components_without_adr:
        add_issue(
            "medium",
            "adr",
            (
                "Component(s) are not justified by a valid ADR: "
                f"{', '.join(components_without_adr)}."
            ),
            (
                "components_without_adr="
                f"{components_without_adr}"
            ),
            (
                "Populate each component's `related_adr_ids` with an "
                "existing ADR ID and document the component in that ADR."
            ),
        )

    if unowned_target_services:
        for name, locations in sorted(unowned_target_services.items()):
            add_issue(
                "high",
                "traceability",
                (
                    f"The target architecture references '{name}' but no "
                    "Component Description exists for it and no artifact "
                    "states it remains in the legacy monolith, is "
                    "external, or is owned by an existing component."
                ),
                f"referenced in: {', '.join(locations)}",
                (
                    f"Add a Component Description for {name}, remove it "
                    "from the target architecture, explicitly keep it in "
                    "the legacy monolith, or map its ownership to an "
                    "existing component."
                ),
            )

    if unjustified_language_drift:
        # HIGH, like the other brownfield design-quality blockers: it routes
        # the run to REFINING until the deviation is justified or removed.
        for language, where in sorted(unjustified_language_drift.items()):
            add_issue(
                "high",
                "constraint",
                (
                    f"The design introduces {language} for new services "
                    "without justifying the change from the detected "
                    "existing implementation stack."
                ),
                f"{language} appears in: {', '.join(where)}",
                (
                    f"Either keep the existing stack or add an explicit "
                    f"ADR comparing {language} with the detected stack and "
                    "linking the change to a concrete requirement or "
                    "trade-off."
                ),
            )

    if unresolved_flow_participants:
        # HIGH: an unresolved flow participant means the diagram a reader
        # sees and the Component catalog a reader sees do not agree.
        for endpoint, flows in sorted(unresolved_flow_participants.items()):
            add_issue(
                "high",
                "traceability",
                (
                    f"'{endpoint}' appears as a data-flow participant but "
                    "has no Component Description and is not a "
                    "clearly-external actor/system."
                ),
                f"referenced in data_flows: {'; '.join(flows)}",
                (
                    f"Add a Component Description for '{endpoint}', "
                    "change the flow to reference an existing Component by "
                    "its exact name, or mark the participant explicitly "
                    "external."
                ),
            )

    if services_without_migration_disposition:
        # HIGH, brownfield-only (see `_check_migration_disposition`): a new
        # service with no stated migration path is a silent scope change.
        for name in services_without_migration_disposition:
            add_issue(
                "high",
                "traceability",
                (
                    f"'{name}' is a new internal service with no explicit "
                    "migration disposition."
                ),
                f"component={name}",
                (
                    f"State '{name}'s migration disposition explicitly: "
                    "extracted in a named migration step, deferred to a "
                    "later phase, retained in the legacy system for now, "
                    "or external/already existing."
                ),
            )

    if unrenderable_data_flows:
        # HIGH: a prose flow cannot be rendered, so the client-facing
        # diagram silently loses that piece of the target architecture.
        for flow in unrenderable_data_flows:
            shown = flow if len(flow) <= 160 else flow[:159] + "…"
            add_issue(
                "high",
                "traceability",
                (
                    "Blueprint data flow is not in the directional "
                    "'A → B' syntax and cannot be rendered as an "
                    "architecture edge."
                ),
                f"data_flow: {shown}",
                (
                    "Rewrite every data flow as directional edges using the "
                    "exact component names (e.g. 'API Gateway → Order "
                    "Service: checkout requests'); a chain "
                    "'A → B → C' is allowed, prose descriptions belong "
                    "after the ':' label."
                ),
            )

    if unresolved_migration_targets:
        # HIGH: the migration plan promises a service the target
        # architecture never declares — refinement must either describe the
        # component or fix the step.
        for name, titles in sorted(unresolved_migration_targets.items()):
            add_issue(
                "high",
                "traceability",
                (
                    f"Migration step(s) introduce '{name}', but no "
                    "Component Description exists for it in the target "
                    "architecture."
                ),
                f"named in migration step(s): {'; '.join(titles)}",
                (
                    f"Either add a Component Description for '{name}', or "
                    "correct the migration step to name only services the "
                    "target architecture actually declares (or explicitly "
                    "disposes as legacy, external, or deferred)."
                ),
            )

    structured_traceability_errors = {
        key: value
        for key, value in traceability.items()
        if key not in {
            "features_without_component",
            "components_without_feature",
            "components_without_adr",
            "unowned_target_services",
        }
        and value
    }
    if structured_traceability_errors:
        add_issue(
            "high",
            "traceability",
            "Structured traceability links are missing or reference unknown artifacts.",
            str(structured_traceability_errors),
            (
                "Populate every Blueprint, Component, and ADR traceability field "
                "with identifiers or component names that resolve exactly."
            ),
        )

    if malformed_adrs:
        add_issue(
            "medium",
            "adr",
            (
                "Malformed ADR(s): "
                f"{', '.join(malformed_adrs)}."
            ),
            (
                "Expected title format `ADR-<number>: <decision>` "
                "and a non-empty decision."
            ),
            (
                "Correct the ADR title format and provide a complete "
                "decision statement."
            ),
        )

    if incomplete_adrs:
        add_issue(
            "medium",
            "adr",
            f"ADR(s) omit required decision evidence: {', '.join(incomplete_adrs)}.",
            "Each ADR needs context, rationale, alternatives, and positive and negative consequences.",
            "Complete every ADR's context, rationale, alternatives, and trade-offs.",
        )

    if duplicate_numbers:
        add_issue(
            "medium",
            "adr",
            f"Duplicate ADR number(s): {duplicate_numbers}.",
            f"duplicate_adr_numbers={duplicate_numbers}",
            "Renumber ADRs so that every ADR number is unique.",
        )

    if invalid_sources:
        add_issue(
            "high",
            "evidence",
            "One or more ADR source references do not resolve to supplied evidence.",
            str(invalid_sources),
            "Use only source names present in retrieved KB chunks or repository evidence.",
        )

    if invalid_evidence_ids:
        add_issue(
            "high",
            "evidence",
            "One or more ADRs cite a KB evidence ID that was not actually "
            "retrieved from the curated knowledge base this run.",
            str(invalid_evidence_ids),
            "Cite only evidence_ids that appear in <evidence_catalog>; "
            "remove any ID that does not, or drop the claim of literature "
            "support entirely.",
            # Always refinable: a fabricated ID is the Architect's own
            # output, and the next pass can simply drop or correct it.
        )
    elif kb_evidence_gap:
        # No fabrication, and no qualifying KB evidence existed ANYWHERE
        # this run for the Architect to cite in the first place. The
        # project claim ("every final ADR is traceable to literature
        # actually retrieved this run") is therefore false for this run —
        # reported HONESTLY as a failure (score 0, see
        # `_check_kb_evidence_grounding`), not silently accepted. Marked
        # `non_refinable`: research runs exactly once, before the refine
        # loop exists, so no Architect redesign can manufacture evidence
        # that was never retrieved — `refine_gate.py` reads this flag to
        # stop the loop honestly instead of spending every iteration
        # re-describing the same architecture against an unfixable finding.
        add_issue(
            "high",
            "evidence",
            "No curated-KB evidence was retrieved for any decision topic "
            "this run, so no ADR can be literature-grounded. This design "
            "must not be reported as literature-grounded.",
            f"decision_topics={[t.topic for t in state.decision_topics]} "
            "all retrieved zero qualifying curated-KB evidence "
            "(kb_evidence_gap=true)",
            "This is a knowledge-base coverage gap, not something the "
            "Architect can fix by redesigning: expand the curated KB's "
            "coverage of this case's decision topics and re-run research, "
            "or accept the design without a literature-grounding claim.",
            non_refinable=kb_evidence_gap_non_refinable,
        )
    elif adrs_without_kb_evidence:
        # Qualifying evidence genuinely existed this run (kb_evidence_gap is
        # False here). Still not necessarily an Architect mistake for EVERY
        # named ADR — evidence relevant to one decision topic does not
        # imply relevance to another, and the deterministic layer cannot
        # judge relevance (that is the LLM Reviewer's job — see
        # REVIEWER_SYSTEM's best_practice_grounding question). So the
        # finding names exactly which ADRs are uncited AND which decision
        # topics currently hold qualifying evidence, so a reader (human or
        # the LLM Reviewer) can tell "the Architect skipped available
        # evidence" apart from "nothing retrieved was actually relevant to
        # this specific decision" — the latter being its own KB coverage
        # gap, not a fabrication risk to paper over with a decorative cite.
        #
        # ACTIONABLE ONLY: an ADR mapped exclusively to a required topic
        # `_check_material_topic_evidence_gaps` already reports as a
        # zero-evidence, non_refinable gap is excluded here — see
        # `_actionable_adrs_without_kb_evidence`. Reporting it AGAIN as a
        # refinable "go cite something" finding would invite a refine round
        # that can never succeed, exactly the wasted-iteration bug that
        # check exists to close.
        actionable_adrs_without_kb_evidence = _actionable_adrs_without_kb_evidence(
            state, adrs_without_kb_evidence, material_topics_without_evidence
        )
        if actionable_adrs_without_kb_evidence:
            topics_with_evidence = [
                topic.topic for topic in state.decision_topics if topic.evidence_ids
            ]
            add_issue(
                "high",
                "evidence",
                (
                    "ADR(s) claim no traceable curated-KB literature support: "
                    f"{', '.join(actionable_adrs_without_kb_evidence)}."
                ),
                (
                    f"adrs_without_kb_evidence={actionable_adrs_without_kb_evidence}; "
                    f"decision topics with qualifying evidence available this "
                    f"run: {topics_with_evidence or '(none)'}"
                ),
                (
                    "For each listed ADR: cite a retrieved evidence_id from "
                    "<evidence_catalog> if one genuinely supports that specific "
                    "decision (see <decision_topics> for what was retrieved per "
                    "topic); if nothing retrieved is actually relevant to it, "
                    "that is a knowledge-base coverage gap for this decision — "
                    "say so in the ADR's rationale rather than citing an "
                    "unrelated ID merely to satisfy this check."
                ),
            )

    if invalid_adr_decision_topic_ids:
        add_issue(
            "high",
            "evidence",
            "One or more ADRs reference a decision topic ID that was not "
            "planned/retrieved this run.",
            str(invalid_adr_decision_topic_ids),
            "Use only decision topic IDs (e.g. 'TOPIC-1') that appear in "
            "<decision_topics> this run; remove any that do not.",
        )

    if adr_evidence_outside_mapped_topics:
        add_issue(
            "high",
            "evidence",
            "One or more ADRs cite KB evidence that was retrieved for a "
            "different decision topic than the one(s) the ADR maps to.",
            str(adr_evidence_outside_mapped_topics),
            "Cite only evidence retrieved for a decision topic this ADR "
            "actually maps to via related_decision_topic_ids, or broaden "
            "the ADR's own topic mapping if the citation is genuinely "
            "relevant to a topic it does not yet list.",
        )

    if adrs_without_decision_topic_mapping:
        add_issue(
            "high",
            "evidence",
            "ADR(s) declare no related_decision_topic_ids even though "
            "decision topics were planned this run: "
            f"{', '.join(adrs_without_decision_topic_mapping)}.",
            f"adrs_without_decision_topic_mapping={adrs_without_decision_topic_mapping}",
            "Add related_decision_topic_ids naming the decision topic(s) "
            "this ADR actually addresses, copied exactly from "
            "<decision_topics>.",
        )

    if material_decisions_without_adr:
        add_issue(
            "high",
            "adr",
            "Material architecture decision(s) are not represented by any "
            f"ADR: {', '.join(material_decisions_without_adr)}.",
            f"material_decisions_without_adr={material_decisions_without_adr}",
            "Add or extend an ADR whose related_decision_topic_ids names "
            "each listed decision topic, recording the actual decision "
            "made and citing genuinely relevant retrieved evidence if any "
            "exists for it.",
        )

    if material_topics_without_evidence:
        add_issue(
            "high",
            "evidence",
            "No curated-KB evidence was retrieved for material decision "
            f"topic(s): {', '.join(material_topics_without_evidence)}. "
            "This design cannot claim literature grounding for those "
            "decisions.",
            f"material_topics_without_evidence={material_topics_without_evidence}",
            "This is a knowledge-base coverage gap for these specific "
            "decision topics, not something the Architect can fix by "
            "redesigning: expand the curated KB's coverage of these "
            "topics and re-run research, or accept the design without a "
            "literature-grounding claim for them.",
            non_refinable=True,
        )

    if invented_quantitative_targets:
        add_issue(
            "high",
            "grounding",
            "Architecture text states a quantitative performance/scale "
            "target the locked Context Record never authorized: "
            + "; ".join(
                f"{location}: {', '.join(values)}"
                for location, values in list(invented_quantitative_targets.items())[:4]
            )
            + ("; …" if len(invented_quantitative_targets) > 4 else ""),
            str(invented_quantitative_targets),
            "Remove the invented figure; state the target qualitatively, "
            "or note it must be measured/agreed later, unless the Context "
            "Record explicitly authorizes that exact number.",
        )

    return DeterministicChecks(
        artifacts_present=present,
        missing_fields=missing,
        features_without_component=features_without_component,
        components_without_feature=components_without_feature,
        components_without_adr=components_without_adr,
        blueprint_missing_feature_ids=traceability["blueprint_missing_feature_ids"],
        invalid_blueprint_feature_ids=traceability["invalid_blueprint_feature_ids"],
        invalid_component_feature_ids=traceability["invalid_component_feature_ids"],
        invalid_component_adr_ids=traceability["invalid_component_adr_ids"],
        invalid_adr_feature_ids=traceability["invalid_adr_feature_ids"],
        invalid_adr_component_names=traceability["invalid_adr_component_names"],
        adrs_without_feature=traceability["adrs_without_feature"],
        adrs_without_component=traceability["adrs_without_component"],
        malformed_adr_titles=malformed_adrs,
        incomplete_adr_ids=incomplete_adrs,
        duplicate_adr_numbers=duplicate_numbers,
        repository_expected=repository_expected,
        repository_available=repository_available,
        invalid_source_references=invalid_sources,
        adrs_without_kb_evidence=adrs_without_kb_evidence,
        invalid_evidence_ids=invalid_evidence_ids,
        kb_evidence_gap=kb_evidence_gap,
        invalid_adr_decision_topic_ids=invalid_adr_decision_topic_ids,
        material_decisions_without_adr=material_decisions_without_adr,
        adr_evidence_outside_mapped_topics=adr_evidence_outside_mapped_topics,
        adrs_without_decision_topic_mapping=adrs_without_decision_topic_mapping,
        material_topics_without_evidence=material_topics_without_evidence,
        invented_quantitative_targets=invented_quantitative_targets,
        unowned_target_services=unowned_target_services,
        unjustified_language_drift=unjustified_language_drift,
        unresolved_flow_participants=unresolved_flow_participants,
        unrenderable_data_flows=unrenderable_data_flows,
        services_without_migration_disposition=services_without_migration_disposition,
        unresolved_migration_targets=unresolved_migration_targets,
        constraints_applicable=constraints_applicable,
        constraints_covered=constraints_covered,
        constraints_uncovered=constraints_uncovered,
        requirements_without_feature=requirements_without_feature,
        score_all_artifacts_present=score_artifacts,
        score_constraint_coverage=score_constraints,
        score_traceability=score_traceability,
        score_adr_presence=score_adr,
        score_source_integrity=score_source_integrity,
        score_kb_evidence_grounding=score_kb_evidence_grounding,
        issues=issues,
    )
