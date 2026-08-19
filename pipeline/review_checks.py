"""review_checks.py — deterministic validation for architecture artifacts.

Everything that can be checked reliably in code is checked here before the
Reviewer LLM receives the design.

Covered rubric items:

1. Artifact completeness
2. Constraint coverage
3. Feature → Component → ADR traceability
4. ADR presence and structure
5. Source-reference integrity

The checks use the frozen structured schemas owned by Maheen. A limited legacy
fallback exists only when callers explicitly enable it for old fixtures.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from pipeline.state import ArchitectState, ReviewIssue


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

    constraints_applicable: dict[str, bool] = Field(default_factory=dict)
    constraints_covered: dict[str, bool] = Field(default_factory=dict)

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


def _check_source_integrity(
    state: ArchitectState,
) -> tuple[dict[str, list[str]], int]:
    """Reject ADR citations that do not resolve to evidence supplied this run."""

    kb_sources = {
        _normalise_source(chunk.source)
        for chunk in state.retrieved_knowledge
        if chunk.source.strip()
    }
    kb_basenames = {source.rsplit("/", 1)[-1] for source in kb_sources}
    repository_text = ""
    if state.repo_representation is not None:
        repository_text = state.repo_representation.model_dump_json().lower()

    invalid: dict[str, list[str]] = {}
    total_references = 0
    valid_references = 0
    for adr in state.adrs:
        for reference in _non_empty(adr.source_references):
            total_references += 1
            normalised = _normalise_source(reference)
            basename = normalised.rsplit("/", 1)[-1]
            resolves = (
                normalised in kb_sources
                or basename in kb_basenames
                or bool(repository_text and normalised in repository_text)
            )
            if resolves:
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
) -> tuple[dict[str, bool], dict[str, bool]]:
    """Identify stated constraints and check only design artifacts for evidence."""

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
    covered = {
        group: bool(values) and all(
            evidenced(requirement)
            for requirement in values
        )
        for group, values in requirements.items()
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

    malformed_adrs, incomplete_adrs, duplicate_numbers = _check_adrs(state)
    constraints_applicable, constraints_covered = _check_constraints(state)
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

    if repository_expected and not repository_available:
        add_issue(
            "high",
            "repo_alignment",
            "A repository was requested but no repository representation is available.",
            f"repo_url={state.initial_request.repo_url}",
            "Ingest the requested repository successfully before reviewing the design.",
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
        constraints_applicable=constraints_applicable,
        constraints_covered=constraints_covered,
        score_all_artifacts_present=score_artifacts,
        score_constraint_coverage=score_constraints,
        score_traceability=score_traceability,
        score_adr_presence=score_adr,
        score_source_integrity=score_source_integrity,
        issues=issues,
    )
