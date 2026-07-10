"""review_checks.py — the Reviewer's DETERMINISTIC layer (Waqar). Plain Python, no LLM.

everything code can decide about a design is decided HERE, once, and handed to
the LLM as *input* (see docs/prompt_quality/04_reviewer_agent_prompt.md: "Trust
the deterministic results ... Do not re-litigate them"). The LLM never
recomputes these. Covered rubric items (05_eval_rubric_v1.md):

  * item 1  all_artifacts_present  — artifacts exist and required fields are filled
  * item 2  constraint_coverage    — the five constraint groups are addressed in the design
  * item 5  traceability           — feature -> component and decision -> ADR links exist
  * item 6  (presence half)        — one well-formed, uniquely numbered ADR per decision;
                                     the *soundness* half stays with the LLM.

Every failure is also synthesised into a ReviewIssue so the final report is
complete even if the LLM ignores it. Pure functions over the state object —
no I/O, no randomness — so all of this is unit-testable offline (test_reviewer.py).
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from pipeline.state import ArchitectState, ReviewIssue

# ── Rubric item 2: constraint-keyword groups ──────────────────────────────
# A constraint group counts as "addressed" when any of its keywords appears in
# the combined design text (blueprint + ADRs + components). Substring match on
# lowercased text — "scal" deliberately catches scale/scaling/scalable, same
# trick as ARCHITECTURE_KEYWORDS in the architect.py prototype.
CONSTRAINT_KEYWORDS: dict[str, list[str]] = {
    "cloud": ["aws", "azure", "gcp", "google cloud", "on-prem", "cloud"],
    "budget": ["budget", "cost", "pricing", "free tier"],
    "scalability": ["scal", "concurrent", "peak", "load", "throughput"],
    "compliance": ["gdpr", "compliance", "pci", "hipaa", "encrypt", "data residency"],
    "existing_system": ["monolith", "brownfield", "greenfield", "legacy", "existing", "migration"],
}

# Rubric item 6 (presence half): ADR titles must follow "ADR-<n>: <decision>".
ADR_TITLE_RE = re.compile(r"^ADR-(\d+)\s*:\s*\S")


class DeterministicChecks(BaseModel):
    """Machine-checked facts about the design + the [code] rubric scores.

    Serialised (model_dump_json) into the Reviewer prompt as
    <deterministic_check_results>; `issues` is merged into the final report.
    """

    # raw facts (the evidence behind the scores)
    artifacts_present: dict[str, bool] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    features_without_component: list[str] = Field(default_factory=list)
    components_without_feature: list[str] = Field(default_factory=list)
    components_without_adr: list[str] = Field(default_factory=list)
    malformed_adr_titles: list[str] = Field(default_factory=list)
    duplicate_adr_numbers: list[int] = Field(default_factory=list)
    constraints_covered: dict[str, bool] = Field(default_factory=dict)

    # [code] rubric scores derived from the facts (0-2 each)
    score_all_artifacts_present: int = Field(0, ge=0, le=2)
    score_constraint_coverage: int = Field(0, ge=0, le=2)
    score_traceability: int = Field(0, ge=0, le=2)
    score_adr_presence: int = Field(0, ge=0, le=2)

    # code-detected problems, ready to merge into the final report
    issues: list[ReviewIssue] = Field(default_factory=list)


def _design_text(state: ArchitectState) -> str:
    """All design prose in one lowercase string, for keyword scanning."""
    parts: list[str] = []
    if state.blueprint is not None:
        parts += [state.blueprint.stakeholder_view, state.blueprint.technical_view]
    for adr in state.adrs:
        parts += [adr.title, adr.decision]
    for comp in state.components:
        parts += [comp.name, comp.description]
    return "\n".join(parts).lower()


def _check_artifacts(state: ArchitectState) -> tuple[dict[str, bool], list[str]]:
    """Rubric item 1: the four artifacts exist and their required fields are filled."""
    present = {
        "context_record": state.context_record is not None,
        "blueprint": state.blueprint is not None,
        "adrs": bool(state.adrs),
        "components": bool(state.components),
    }
    missing: list[str] = []
    if state.context_record is not None and not state.context_record.summary.strip():
        missing.append("context_record.summary")
    if state.blueprint is not None:
        if not state.blueprint.stakeholder_view.strip():
            missing.append("blueprint.stakeholder_view")
        if not state.blueprint.technical_view.strip():
            missing.append("blueprint.technical_view")
    for i, adr in enumerate(state.adrs):
        if not adr.title.strip():
            missing.append(f"adrs[{i}].title")
        if not adr.decision.strip():
            missing.append(f"adrs[{i}].decision")
    for i, comp in enumerate(state.components):
        if not comp.name.strip():
            missing.append(f"components[{i}].name")
        if not comp.description.strip():
            missing.append(f"components[{i}].description")
    for i, feat in enumerate(state.features):
        if not (feat.id.strip() and feat.name.strip() and feat.scenario.strip()):
            missing.append(f"features[{i}] (id/name/scenario)")
    return present, missing


def _check_traceability(state: ArchitectState) -> tuple[list[str], list[str], list[str]]:
    """Rubric item 5: feature -> component and component -> ADR links exist.

    With the current placeholder schemas there are no dedicated link fields yet
    (Maheen's frozen schemas add `related_feature_id` etc.), so a link means the
    id/name is *mentioned* in the other artifact's text. Swapping to real link
    fields later touches only this function.
    """
    feature_ids = [f.id for f in state.features if f.id.strip()]
    comp_texts = [(c.name, f"{c.name} {c.description}".lower()) for c in state.components]
    adr_text = " ".join(f"{a.title} {a.decision}" for a in state.adrs).lower()

    features_without_component = [
        fid for fid in feature_ids
        if not any(fid.lower() in text for _, text in comp_texts)
    ]
    components_without_feature = [
        name for name, text in comp_texts
        if feature_ids and not any(fid.lower() in text for fid in feature_ids)
    ]
    # "components and major choices to ADRs": each component's choice is backed
    # by at least one ADR that names it.
    components_without_adr = [
        name for name, _ in comp_texts if name.lower() not in adr_text
    ]
    return features_without_component, components_without_feature, components_without_adr


def _check_adrs(state: ArchitectState) -> tuple[list[str], list[int]]:
    """Rubric item 6, presence half: well-formed 'ADR-<n>:' titles, unique numbers."""
    malformed: list[str] = []
    numbers: list[int] = []
    for adr in state.adrs:
        m = ADR_TITLE_RE.match(adr.title.strip())
        if m is None or not adr.decision.strip():
            malformed.append(adr.title or "(empty title)")
        else:
            numbers.append(int(m.group(1)))
    duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
    return malformed, duplicates


def _check_constraints(state: ArchitectState) -> dict[str, bool]:
    """Rubric item 2: every constraint group is addressed somewhere in the design."""
    text = _design_text(state)
    return {
        group: any(kw in text for kw in keywords)
        for group, keywords in CONSTRAINT_KEYWORDS.items()
    }


def run_deterministic_checks(state: ArchitectState) -> DeterministicChecks:
    """Run every code-checkable rubric item and synthesise issues for failures."""
    present, missing = _check_artifacts(state)
    feats_unlinked, comps_unlinked, comps_no_adr = _check_traceability(state)
    malformed_adrs, dup_numbers = _check_adrs(state)
    covered = _check_constraints(state)

    # ── scores ────────────────────────────────────────────────────────────
    if not all(present.values()):
        score_artifacts = 0
    elif missing:
        score_artifacts = 1
    else:
        score_artifacts = 2

    n_covered = sum(covered.values())
    score_constraints = 2 if n_covered == len(covered) else (1 if n_covered >= 3 else 0)

    if not state.features or not state.components:
        score_trace = 0
    elif not feats_unlinked and not comps_unlinked:
        score_trace = 2
    elif len(feats_unlinked) < len(state.features) or len(comps_unlinked) < len(state.components):
        score_trace = 1
    else:
        score_trace = 0

    if not state.adrs or len(malformed_adrs) == len(state.adrs):
        score_adr = 0
    elif malformed_adrs or dup_numbers or comps_no_adr:
        score_adr = 1
    else:
        score_adr = 2

    # ── issues (code-detected, merged verbatim into the final report) ─────
    issues: list[ReviewIssue] = []

    def add(severity: str, category: str, finding: str, evidence: str, fix: str) -> None:
        issues.append(ReviewIssue(
            id=f"DET-{len(issues) + 1}",
            severity=severity,
            category=category,
            finding=finding,
            evidence=evidence,
            suggested_fix=fix,
            requires_refinement=severity == "high",
        ))

    absent = [name for name, ok in present.items() if not ok]
    if absent:
        add("high", "completeness",
            f"Required artifact(s) missing: {', '.join(absent)}.",
            f"artifacts_present={present}",
            "Produce every artifact: Context Record, Blueprint, ADRs, Component Descriptions.")
    if missing:
        add("medium", "completeness",
            f"{len(missing)} required field(s) empty.",
            "; ".join(missing),
            "Fill every required field on each artifact.")
    uncovered = [g for g, ok in covered.items() if not ok]
    if uncovered:
        add("high", "constraint",
            f"Constraint group(s) not addressed in the design: {', '.join(uncovered)}.",
            f"No matching keywords found for: {', '.join(uncovered)}",
            "Address each stated constraint explicitly in the blueprint, ADRs, or components.")
    if feats_unlinked:
        add("medium", "traceability",
            f"Feature(s) with no component tracing to them: {', '.join(feats_unlinked)}.",
            f"features_without_component={feats_unlinked}",
            "Reference the feature id in at least one component description.")
    if comps_unlinked:
        add("medium", "traceability",
            f"Component(s) tracing to no feature: {', '.join(comps_unlinked)}.",
            f"components_without_feature={comps_unlinked}",
            "Reference a feature id in each component description.")
    if comps_no_adr:
        add("medium", "adr",
            f"Component(s) not backed by any ADR: {', '.join(comps_no_adr)}.",
            f"components_without_adr={comps_no_adr}",
            "Record the significant decision behind each component in an ADR that names it.")
    if malformed_adrs:
        add("medium", "adr",
            f"Malformed ADR(s) (bad title format or empty decision): {', '.join(malformed_adrs)}.",
            "Expected title format 'ADR-<n>: <decision>' and a non-empty decision.",
            "Fix the ADR title numbering and fill the decision.")
    if dup_numbers:
        add("medium", "adr",
            f"Duplicate ADR number(s): {dup_numbers}.",
            f"duplicate_adr_numbers={dup_numbers}",
            "Renumber ADRs so each number is unique.")

    return DeterministicChecks(
        artifacts_present=present,
        missing_fields=missing,
        features_without_component=feats_unlinked,
        components_without_feature=comps_unlinked,
        components_without_adr=comps_no_adr,
        malformed_adr_titles=malformed_adrs,
        duplicate_adr_numbers=dup_numbers,
        constraints_covered=covered,
        score_all_artifacts_present=score_artifacts,
        score_constraint_coverage=score_constraints,
        score_traceability=score_trace,
        score_adr_presence=score_adr,
        issues=issues,
    )
