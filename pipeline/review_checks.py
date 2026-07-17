"""review_checks.py — deterministic validation for architecture artifacts.

Everything that can be checked reliably in code is checked here before the
Reviewer LLM receives the design.

Covered rubric items:

1. Artifact completeness
2. Constraint coverage
5. Feature → Component → ADR traceability
6. ADR presence and numbering

The checks prefer the frozen structured schemas owned by Maheen. A limited
legacy fallback remains for older fixtures and modules that still express
traceability inside prose.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from pipeline.state import ArchitectState, ReviewIssue


# ─────────────────────────────────────────────────────────────────────────
# Constraint detection
# ─────────────────────────────────────────────────────────────────────────

CONSTRAINT_KEYWORDS: dict[str, list[str]] = {
    "cloud": [
        "aws",
        "azure",
        "gcp",
        "google cloud",
        "on-prem",
        "on premise",
        "cloud",
    ],
    "budget": [
        "budget",
        "cost",
        "pricing",
        "free tier",
        "managed service",
    ],
    "scalability": [
        "scal",
        "concurrent",
        "peak",
        "load",
        "throughput",
        "autoscaling",
        "horizontal",
    ],
    "compliance": [
        "gdpr",
        "compliance",
        "pci",
        "hipaa",
        "encrypt",
        "privacy",
        "data residency",
    ],
    "existing_system": [
        "monolith",
        "brownfield",
        "greenfield",
        "legacy",
        "existing",
        "migration",
        "modernization",
    ],
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
    duplicate_adr_numbers: list[int] = Field(default_factory=list)

    constraints_applicable: dict[str, bool] = Field(default_factory=dict)
    constraints_covered: dict[str, bool] = Field(default_factory=dict)

    score_all_artifacts_present: int = Field(0, ge=0, le=2)
    score_constraint_coverage: int = Field(0, ge=0, le=2)
    score_traceability: int = Field(0, ge=0, le=2)
    score_adr_presence: int = Field(0, ge=0, le=2)

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


def _context_text(state: ArchitectState) -> str:
    """Render the locked Context Record only, to identify applicable constraints."""

    parts: list[str] = []

    if state.context_record is not None:
        context = state.context_record
        parts.extend(
            [
                context.summary,
                context.project_name,
                context.business_goal,
                context.problem_statement,
                context.cloud_provider,
                context.budget,
            ]
        )
        parts.extend(context.functional_requirements)
        parts.extend(context.non_functional_requirements)
        parts.extend(context.compliance_requirements)
        parts.extend(context.existing_systems)
        parts.extend(context.assumptions)

    return "\n".join(_non_empty(parts)).lower()


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
        "blueprint": state.blueprint is not None,
        "adrs": bool(state.adrs),
        "components": bool(state.components),
    }

    missing: list[str] = []

    if state.context_record is not None:
        if not state.context_record.summary.strip():
            missing.append("context_record.summary")

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

    for index, component in enumerate(state.components):
        if not component.name.strip():
            missing.append(f"components[{index}].name")

        if not component.description.strip():
            missing.append(f"components[{index}].description")

    for index, feature in enumerate(state.features):
        if not feature.id.strip():
            missing.append(f"features[{index}].id")

        if not feature.name.strip():
            missing.append(f"features[{index}].name")

        if not feature.scenario.strip():
            missing.append(f"features[{index}].scenario")

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
    component_names = {
        component.name.strip()
        for component in state.components
        if component.name.strip()
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
                if value.strip()
            }
            - component_names
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
# Rubric item 6: ADR structure
# ─────────────────────────────────────────────────────────────────────────

def _check_adrs(
    state: ArchitectState,
) -> tuple[list[str], list[int]]:
    """Check ADR title format, decisions, and unique title numbers."""

    malformed: list[str] = []
    numbers: list[int] = []

    for adr in state.adrs:
        title = adr.title.strip()
        decision = adr.decision.strip()

        match = ADR_TITLE_RE.match(title)

        if match is None or not decision:
            malformed.append(title or "(empty title)")
            continue

        numbers.append(int(match.group(1)))

    duplicates = sorted(
        {
            number
            for number in numbers
            if numbers.count(number) > 1
        }
    )

    return malformed, duplicates


# ─────────────────────────────────────────────────────────────────────────
# Rubric item 2: constraints
# ─────────────────────────────────────────────────────────────────────────

def _check_constraints(
    state: ArchitectState,
) -> tuple[dict[str, bool], dict[str, bool]]:
    """Identify stated constraints and check only design artifacts for evidence."""

    context_text = _context_text(state)
    design_text = _design_text(state)

    applicable = {
        group: any(
            keyword in context_text
            for keyword in keywords
        )
        for group, keywords in CONSTRAINT_KEYWORDS.items()
    }
    covered = {
        group: any(
            keyword in design_text
            for keyword in keywords
        )
        for group, keywords in CONSTRAINT_KEYWORDS.items()
    }
    return applicable, covered


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

    malformed_adrs, duplicate_numbers = _check_adrs(state)
    constraints_applicable, constraints_covered = _check_constraints(state)

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
                "Blueprint, ADRs, and Component Descriptions."
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
        add_issue(
            "high",
            "constraint",
            (
                "Constraint group(s) not addressed in the design: "
                f"{', '.join(uncovered)}."
            ),
            (
                "No matching structured value or design text found for: "
                f"{', '.join(uncovered)}"
            ),
            (
                "Address each stated constraint explicitly in the Blueprint, "
                "ADRs, or Component Descriptions."
            ),
        )

    if features_without_component:
        add_issue(
            "medium",
            "traceability",
            (
                "Feature(s) have no implementing component: "
                f"{', '.join(features_without_component)}."
            ),
            (
                "features_without_component="
                f"{features_without_component}"
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

    structured_traceability_errors = {
        key: value
        for key, value in traceability.items()
        if key not in {
            "features_without_component",
            "components_without_feature",
            "components_without_adr",
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

    if duplicate_numbers:
        add_issue(
            "medium",
            "adr",
            f"Duplicate ADR number(s): {duplicate_numbers}.",
            f"duplicate_adr_numbers={duplicate_numbers}",
            "Renumber ADRs so that every ADR number is unique.",
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
        duplicate_adr_numbers=duplicate_numbers,
        constraints_applicable=constraints_applicable,
        constraints_covered=constraints_covered,
        score_all_artifacts_present=score_artifacts,
        score_constraint_coverage=score_constraints,
        score_traceability=score_traceability,
        score_adr_presence=score_adr,
        issues=issues,
    )
