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
from pipeline.state import ArchitectState, ContextRecord, ReviewIssue


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


def _name_tokens(name: str) -> tuple[str, ...]:
    """Normalise a display name to comparable lowercase tokens."""

    return tuple(
        _singularise(token)
        for token in re.findall(r"[a-z0-9]+", name.lower())
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
# Rubric item 2: constraints
# ─────────────────────────────────────────────────────────────────────────

def _check_constraints(
    state: ArchitectState,
) -> tuple[dict[str, bool], dict[str, bool], dict[str, list[str]]]:
    """Identify stated constraints and check only design artifacts for evidence.

    Returns (applicable, covered, uncovered). `uncovered` maps each group to
    the EXACT requirement strings that failed the evidence match — the
    actionable form of the diagnosis. The architect's refinement pass gets
    these strings verbatim (via the issue evidence/suggested fix), so it can
    close the gap in one round instead of re-paraphrasing a group label.
    Pass/fail semantics are exactly the previous two-value behaviour.
    """

    context = state.context_record
    design_text = _design_text(state)

    requirements: dict[str, list[str]] = {
        group: [] for group in CONSTRAINT_GROUPS
    }
    if context is not None:
        requirements["functional"] = _non_empty(context.functional_requirements)
        requirements["cloud"] = _non_empty([context.cloud_provider])
        requirements["budget"] = _non_empty([context.budget])
        requirements["non_functional"] = _non_empty(
            context.non_functional_requirements
        )
        requirements["compliance"] = _non_empty(
            context.compliance_requirements
        )
        requirements["existing_system"] = _non_empty(context.existing_systems)

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


def _check_migration_targets(
    state: ArchitectState,
) -> dict[str, list[str]]:
    """Services a migration step promises to introduce that resolve to no
    Component Description, mapped to the step titles naming them. Pure.

    Extraction is NARROW by construction: only sentences carrying an
    introduction verb are scanned, and only names the existing
    `_extract_service_bases` grammar accepts — an explicit
    "<Name> Service" or an enumeration like "Order, Inventory, and Shipping
    Services". Incidental prose ("the shipping module") carries no Service
    suffix and is never a candidate. A candidate is excused when it
    resolves to a described component under the SAME canonical identity as
    the ownership check (including the composite exactness rule), or when
    the design gives it an explicit disposition — legacy retention,
    external, ownership, or deferred to a later phase — under the SAME
    disposition vocabulary the other migration checks use.
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
            for base in _extract_service_bases(sentence):
                display = f"{base} Service"
                key = _component_key(display)
                if not key:
                    continue
                if any(
                    existing == key
                    or (not composite and existing[-len(key):] == key)
                    for existing, composite in component_entries
                ):
                    continue  # the target exists in the catalog
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
                "Constraint group(s) not addressed in the design: "
                f"{', '.join(uncovered)}."
            ),
            (
                f"Uncovered requirement(s): {uncovered_detail}"
                if uncovered_detail
                else "No matching structured value or design text found for: "
                f"{', '.join(uncovered)}"
            ),
            (
                "Name every uncovered requirement verbatim in the design "
                "text (Blueprint, ADRs, or Components): "
                + (uncovered_detail if uncovered_detail else
                   "each requirement the Context Record states.")
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
        issues=issues,
    )
