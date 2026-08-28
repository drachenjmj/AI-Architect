"""reporting.py — deterministic Markdown + PDF export of an ACCEPTED run.

WHAT THIS IS
------------
The final deliverable of the "Accept design" action: two files —
`architecture_report.md` and `architecture_report.pdf` — generated straight
from the artifacts a human just signed off on, with no LLM call and no
network call anywhere in this module.

ONE CANONICAL MODEL, TWO RENDERERS
-----------------------------------
`build_report(state)` reduces an accepted `ArchitectState` to `ReportData` —
plain data, pure code, no I/O. `render_markdown` and `render_pdf` both read
ONLY `ReportData`; neither renderer touches `ArchitectState` directly and
neither decides what to include. That is what stops the two formats from
drifting apart: a field missing from `ReportData` is missing from both
outputs, and a field present in it is available to both.

Existing schemas are reused wherever they already say what is needed
(`ContextRecord`, `Feature`, `Blueprint`, `ADR`, `ComponentDescription`,
`ReviewResult`, `Waiver`) rather than re-declared here — the report is a
projection of the run, not a second copy of its data model. Only the
genuinely NEW shapes (evidence trimmed for display, findings classified by
origin, the traceability maps, the limitations digest) get their own model.

NOTHING IS INVENTED
--------------------
Every value in `ReportData` traces back to a field already on the state that
was accepted. `executive_summary` is the one derived field, and it is a
DETERMINISTIC STRING CONCATENATION of existing fields (see
`_build_executive_summary`) — never a new LLM call. A FAIL verdict, an open
finding, or a KB evidence gap is carried through unchanged: acceptance means
a human took the design, not that the record forgets what it was taken
despite (see pipeline/sign_off.py).

WHERE FILES LAND
----------------
Each run owns its own report artifacts, written beside its checkpoints —
`<runs_dir>/<run_id>/architecture_report.{md,pdf}` (see
`pipeline/persistence.py` for `runs_dir()`). Neither filename matches the
checkpoint naming convention (`NNN_stage.json`), so the report never gets
mistaken for a resume point, and `.cache` is already gitignored — nothing
here is ever staged by accident.
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from pipeline.persistence import runs_dir
from pipeline.sign_off import open_findings
from pipeline.state import (
    ADR,
    ArchitectState,
    Blueprint,
    ComponentDescription,
    ContextRecord,
    Feature,
    ReviewResult,
    Stage,
    Waiver,
)

log = logging.getLogger(__name__)

REPORT_MD_NAME = "architecture_report.md"
REPORT_PDF_NAME = "architecture_report.pdf"

# Curated-KB evidence text is trimmed to this many characters in the report —
# long enough to be useful, short enough that a report never balloons into a
# copy of the knowledge base. Never re-fetched or re-summarized: what is
# stored in `KBChunk.content` is exactly what is shown, truncated only.
_EXCERPT_CHARS = 600


class ReportError(RuntimeError):
    """A report could not be built or written. Carries the offending reason."""


# ══════════════════════════════════════════════════════════════════════════
# The canonical model
# ══════════════════════════════════════════════════════════════════════════
class ReportEvidenceItem(BaseModel):
    """One retrieved chunk, trimmed for display. See `KBChunk` for the source."""

    evidence_id: str = Field("", description="KB-Exxx id; empty for a web-fallback chunk.")
    source: str = ""
    page: int = 0
    box: int = 1
    is_curated: bool = Field(True, description="False for a grounded web-fallback chunk (box 3).")
    excerpt: str = Field("", description="Truncated `KBChunk.content` — never re-fetched.")


class ReportFinding(BaseModel):
    """One Reviewer finding, classified by where it came from.

    `origin` is read off the finding's OWN id prefix (`DET-` for a
    deterministic code check, `LLM-` for an LLM judgment — see
    `pipeline/review_checks.py` and `pipeline/agents/reviewer.py`), never
    guessed from its text.
    """

    id: str = ""
    severity: str = "low"
    category: str = ""
    finding: str = ""
    evidence: str = ""
    suggested_fix: str = ""
    origin: str = Field("unknown", description="'deterministic' | 'llm' | 'unknown'.")
    non_refinable: bool = False


class ReportDecisionTopic(BaseModel):
    """One planned decision topic and what it did or did not surface.

    `has_gap` mirrors `DecisionTopic.evidence_ids` being empty — an explicit,
    verified evidence gap for that topic, never backfilled with a web result.
    """

    id: str
    topic: str
    query: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    has_gap: bool = False


class ReportTraceability(BaseModel):
    """Deterministic ID-based mappings only — no fuzzy text matching.

    Each map is built by reading the EXISTING relationship fields the
    artifacts already carry (`related_feature_ids`, `related_decision_topic_ids`,
    `evidence_ids`, ...); an id with no entry in a map simply traces to
    nothing, which is the honest answer rather than an invented one.
    """

    feature_to_adrs: dict[str, list[str]] = Field(default_factory=dict)
    feature_to_components: dict[str, list[str]] = Field(default_factory=dict)
    decision_topic_to_adrs: dict[str, list[str]] = Field(default_factory=dict)
    adr_to_evidence: dict[str, list[str]] = Field(default_factory=dict)


class ReportLimitations(BaseModel):
    """Known open items, carried through unchanged — never invented, never hidden."""

    unresolved_high: list[str] = Field(default_factory=list)
    unresolved_medium: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    other: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.unresolved_high
            or self.unresolved_medium
            or self.evidence_gaps
            or self.other
        )


class ReportMeta(BaseModel):
    """Cover-page facts. Every field is read from the run; nothing is invented."""

    run_id: str
    project_name: str = ""
    accepted_at: str = ""
    review_status: str = Field("", description="'pass' / 'fail' / '' when never reviewed.")
    stopped_on_cap: bool = False
    selected_round: int = 0
    refine_iterations: int = 0
    user_rounds: int = 0
    models_used: list[str] = Field(
        default_factory=list,
        description="Distinct non-empty StepLog.model values, first-seen order.",
    )


class ReportData(BaseModel):
    """The one structure both `render_markdown` and `render_pdf` read.

    Reuses the artifact schemas verbatim (`context_record`, `features`,
    `blueprint` — which already carries `data_flows` and `migration_steps` —
    `adrs`, `components`, `review`, `waiver`) so the report can never say
    something about an artifact that the artifact itself does not say.
    """

    meta: ReportMeta
    executive_summary: str = ""
    context_record: Optional[ContextRecord] = None
    features: list[Feature] = Field(default_factory=list)
    blueprint: Optional[Blueprint] = None
    adrs: list[ADR] = Field(default_factory=list)
    components: list[ComponentDescription] = Field(default_factory=list)
    evidence: list[ReportEvidenceItem] = Field(default_factory=list)
    decision_topics: list[ReportDecisionTopic] = Field(default_factory=list)
    findings: list[ReportFinding] = Field(default_factory=list)
    review: Optional[ReviewResult] = None
    waiver: Optional[Waiver] = None
    traceability: ReportTraceability = Field(default_factory=ReportTraceability)
    limitations: ReportLimitations = Field(default_factory=ReportLimitations)


# ══════════════════════════════════════════════════════════════════════════
# Building the model from an accepted state — pure, no I/O, no LLM call
# ══════════════════════════════════════════════════════════════════════════
def _models_used(state: ArchitectState) -> list[str]:
    seen: list[str] = []
    for step in state.history:
        if step.model and step.model not in seen:
            seen.append(step.model)
    return seen


def _project_name(state: ArchitectState) -> str:
    record = state.context_record
    if record and record.project_name.strip():
        return record.project_name.strip()
    if state.blueprint and state.blueprint.project_name.strip():
        return state.blueprint.project_name.strip()
    return ""


def _build_executive_summary(state: ArchitectState) -> str:
    """A deterministic combination of EXISTING fields. Never an LLM call.

    Prefers `Blueprint.stakeholder_view` — already a dedicated business-facing
    narrative field — and falls back to joining the business goal, problem
    statement, Blueprint rationale and final verdict when no such narrative
    exists (e.g. a run that never reached a design).
    """
    blueprint = state.blueprint
    if blueprint and blueprint.stakeholder_view.strip():
        return blueprint.stakeholder_view.strip()

    parts: list[str] = []
    record = state.context_record
    if record:
        if record.business_goal.strip():
            parts.append(record.business_goal.strip())
        if record.problem_statement.strip():
            parts.append(record.problem_statement.strip())
    if blueprint and blueprint.rationale.strip():
        parts.append(blueprint.rationale.strip())
    if state.review is not None:
        parts.append(f"Final review verdict: {state.review.overall_status.upper()}.")
    return " ".join(parts)


def _excerpt(text: str, limit: int = _EXCERPT_CHARS) -> str:
    flat = text.strip()
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _build_evidence(state: ArchitectState) -> list[ReportEvidenceItem]:
    items = [
        ReportEvidenceItem(
            evidence_id=chunk.evidence_id,
            source=chunk.source,
            page=chunk.page,
            box=chunk.box,
            is_curated=chunk.box != 3,
            excerpt=_excerpt(chunk.content),
        )
        for chunk in state.retrieved_knowledge
    ]
    # Curated evidence first, web fallback after — "clearly distinct", never
    # interleaved arbitrarily. Stable sort preserves retrieval order within
    # each group.
    items.sort(key=lambda item: not item.is_curated)
    return items


def _origin_of(issue_id: str) -> str:
    if issue_id.startswith("DET-"):
        return "deterministic"
    if issue_id.startswith("LLM-"):
        return "llm"
    return "unknown"


def _build_findings(state: ArchitectState) -> list[ReportFinding]:
    """Every OPEN finding on the review being signed off — highest severity
    first, via `sign_off.open_findings` so the report can never show a
    different order or a different set than the sign-off screen itself did.
    """
    return [
        ReportFinding(
            id=issue.id,
            severity=issue.severity,
            category=issue.category,
            finding=issue.finding,
            evidence=issue.evidence,
            suggested_fix=issue.suggested_fix,
            origin=_origin_of(issue.id),
            non_refinable=issue.non_refinable,
        )
        for issue in open_findings(state)
    ]


def _build_decision_topics(state: ArchitectState) -> list[ReportDecisionTopic]:
    return [
        ReportDecisionTopic(
            id=topic.id,
            topic=topic.topic,
            query=topic.query,
            evidence_ids=list(topic.evidence_ids),
            has_gap=not topic.evidence_ids,
        )
        for topic in state.decision_topics
    ]


def _build_traceability(state: ArchitectState) -> ReportTraceability:
    feature_to_adrs: dict[str, list[str]] = {f.id: [] for f in state.features}
    feature_to_components: dict[str, list[str]] = {f.id: [] for f in state.features}
    for adr in state.adrs:
        for feature_id in adr.related_feature_ids:
            feature_to_adrs.setdefault(feature_id, []).append(adr.id)
    for component in state.components:
        for feature_id in component.related_feature_ids:
            feature_to_components.setdefault(feature_id, []).append(component.id)

    decision_topic_to_adrs: dict[str, list[str]] = {t.id: [] for t in state.decision_topics}
    for adr in state.adrs:
        for topic_id in adr.related_decision_topic_ids:
            decision_topic_to_adrs.setdefault(topic_id, []).append(adr.id)

    adr_to_evidence = {adr.id: list(adr.evidence_ids) for adr in state.adrs}

    return ReportTraceability(
        feature_to_adrs=feature_to_adrs,
        feature_to_components=feature_to_components,
        decision_topic_to_adrs=decision_topic_to_adrs,
        adr_to_evidence=adr_to_evidence,
    )


def _build_limitations(
    state: ArchitectState, findings: list[ReportFinding]
) -> ReportLimitations:
    other: list[str] = []
    if state.stopped_on_cap:
        other.append(
            "Refinement stopped on the iteration/token budget rather than a "
            "clean reviewer pass; the shipped design is the best round the "
            "gate saw, not a guaranteed pass."
        )
    return ReportLimitations(
        unresolved_high=[
            f"{f.id}: {f.finding}".strip(": ") for f in findings if f.severity == "high"
        ],
        unresolved_medium=[
            f"{f.id}: {f.finding}".strip(": ") for f in findings if f.severity == "medium"
        ],
        evidence_gaps=[
            f"{topic.id}: {topic.topic}".strip(": ")
            for topic in state.decision_topics
            if not topic.evidence_ids
        ],
        other=other,
    )


def build_report(state: ArchitectState) -> ReportData:
    """Reduce an ACCEPTED state to the canonical report model. Pure, no I/O.

    Raises `ReportError` unless `state.stage is Stage.ACCEPTED` — a report is
    the record of what a human took, and building one before that decision
    exists would let a draft impersonate a sign-off.
    """
    if state.stage is not Stage.ACCEPTED:
        raise ReportError(
            f"build_report requires an accepted run (stage ACCEPTED); got "
            f"{state.stage.value!r}. The report records what a human took, "
            f"and that has not happened yet."
        )

    findings = _build_findings(state)
    meta = ReportMeta(
        run_id=state.run_id,
        project_name=_project_name(state),
        accepted_at=state.accepted_at,
        review_status=state.review.overall_status if state.review else "",
        stopped_on_cap=state.stopped_on_cap,
        selected_round=state.selected_round,
        refine_iterations=state.refine_iterations,
        user_rounds=state.user_rounds,
        models_used=_models_used(state),
    )
    return ReportData(
        meta=meta,
        executive_summary=_build_executive_summary(state),
        context_record=state.context_record,
        features=list(state.features),
        blueprint=state.blueprint,
        adrs=list(state.adrs),
        components=list(state.components),
        evidence=_build_evidence(state),
        decision_topics=_build_decision_topics(state),
        findings=findings,
        review=state.review,
        waiver=state.waiver,
        traceability=_build_traceability(state),
        limitations=_build_limitations(state, findings),
    )


# ══════════════════════════════════════════════════════════════════════════
# File I/O — atomic writes, one run's own directory
# ══════════════════════════════════════════════════════════════════════════
def _run_dir(run_id: str) -> Path:
    """Directory for one run's artifacts. Same validation as
    `persistence._run_dir` / `run_history._run_dir` — restated here rather
    than imported, because a private helper of another module is not this
    module's to depend on, and a run_id reaching this function (from a UI
    selection) is untrusted input.
    """
    if not run_id or run_id in {".", ".."} or set(run_id) & set("/\\") or os.path.isabs(run_id):
        raise ReportError(f"Invalid run_id: {run_id!r}")
    return runs_dir() / run_id


def _atomic_write(path: Path, data: bytes) -> None:
    """Same publish discipline as `persistence.save_state`: full content to a
    temp file in the SAME directory, fsynced, then `os.replace`d into place.
    A reader never sees a partial file, and a failed write leaves no `.tmp`
    and no corrupt target behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def report_paths(run_id: str) -> tuple[Optional[Path], Optional[Path]]:
    """Existing `(markdown_path, pdf_path)` for a run, `None` for whichever is
    missing. Read-only — never generates, never touches an old run.
    """
    directory = _run_dir(run_id)
    md_path = directory / REPORT_MD_NAME
    pdf_path = directory / REPORT_PDF_NAME
    return (
        md_path if md_path.is_file() else None,
        pdf_path if pdf_path.is_file() else None,
    )


@dataclass
class ReportGenerationResult:
    """The outcome of one `generate_reports` call. Session-only — never
    persisted, never round-tripped; a caller reads it once, right after the
    call, to decide what to tell the human.
    """

    markdown_path: Optional[Path] = None
    markdown_error: str = ""
    pdf_path: Optional[Path] = None
    pdf_error: str = ""

    @property
    def markdown_ok(self) -> bool:
        return self.markdown_path is not None and not self.markdown_error

    @property
    def pdf_ok(self) -> bool:
        return self.pdf_path is not None and not self.pdf_error

    @property
    def ok(self) -> bool:
        return self.markdown_ok and self.pdf_ok


def generate_reports(state: ArchitectState) -> ReportGenerationResult:
    """Render and write both files for an accepted run. Idempotent: calling
    this again for the SAME run safely replaces only that run's own report
    artifacts (atomic writes, one file each).

    The two formats fail INDEPENDENTLY on purpose — see the module docstring
    on partial failure. A Markdown write that succeeds is not undone by a PDF
    render that then fails, and neither failure touches `state` or the
    acceptance it followed: this function only ever writes report files.
    """
    result = ReportGenerationResult()
    try:
        report = build_report(state)
    except ReportError as exc:
        result.markdown_error = str(exc)
        result.pdf_error = str(exc)
        return result

    directory = _run_dir(state.run_id)

    try:
        markdown = render_markdown(report)
        target = directory / REPORT_MD_NAME
        _atomic_write(target, markdown.encode("utf-8"))
        result.markdown_path = target
    except Exception as exc:  # noqa: BLE001 — reported to the human, never raised through
        log.warning("markdown report generation failed for run %s: %s", state.run_id, exc)
        result.markdown_error = str(exc)

    try:
        pdf_bytes = render_pdf(report)
        target = directory / REPORT_PDF_NAME
        _atomic_write(target, pdf_bytes)
        result.pdf_path = target
    except Exception as exc:  # noqa: BLE001 — reported to the human, never raised through
        log.warning("PDF report generation failed for run %s: %s", state.run_id, exc)
        result.pdf_error = str(exc)

    return result


# ══════════════════════════════════════════════════════════════════════════
# Markdown renderer
# ══════════════════════════════════════════════════════════════════════════
def _md_escape_cell(text: str) -> str:
    """Keep a value from breaking a Markdown table row: no raw newlines, no
    unescaped pipes."""
    return " ".join(text.split()).replace("|", "\\|")


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_md_escape_cell(cell) for cell in row) + " |")
    return lines


def _md_bullets(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items]


def _datestamp(iso: str) -> str:
    if not iso:
        return ""
    try:
        return datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return iso


def render_markdown(report: ReportData) -> str:
    """Render the canonical model to clean, deterministic Markdown.

    Section order matches `render_pdf` exactly, and both read only
    `ReportData` — see the module docstring. Sections with no data to show
    are omitted entirely rather than printed empty, so the document a run
    with no repository or no migration objective produces is honest about
    that instead of padded with "None" placeholders.
    """
    lines: list[str] = []
    meta = report.meta

    # 1. Cover / metadata
    lines.append("# AI-Architect Architecture Report")
    lines.append("")
    if meta.project_name:
        lines.append(f"**Project:** {meta.project_name}  ")
    lines.append(f"**Run ID:** `{meta.run_id}`  ")
    lines.append(
        f"**Status:** Accepted"
        + (f" — {_datestamp(meta.accepted_at)}" if meta.accepted_at else "")
        + "  "
    )
    lines.append(
        f"**Final Review Verdict:** {meta.review_status.upper() if meta.review_status else 'Not reviewed'}  "
    )
    if meta.stopped_on_cap:
        lines.append(
            f"**Refinement:** stopped on budget after round {meta.selected_round or meta.refine_iterations}  "
        )
    if meta.models_used:
        lines.append(f"**Models used:** {', '.join(meta.models_used)}  ")
    lines.append("")

    # 2. Executive Summary
    if report.executive_summary.strip():
        lines += ["## Executive Summary", "", report.executive_summary.strip(), ""]

    # 3. Requirements & Constraints
    record = report.context_record
    if record is not None:
        lines.append("## Requirements & Constraints")
        lines.append("")
        if record.business_goal.strip():
            lines += ["**Business Goal**", "", record.business_goal.strip(), ""]
        if record.problem_statement.strip():
            lines += ["**Problem Statement**", "", record.problem_statement.strip(), ""]
        if record.users:
            lines += ["**Users / Stakeholders**", "", *_md_bullets(record.users), ""]
        if record.functional_requirements:
            lines += [
                "**Functional Requirements**", "",
                *_md_bullets(record.functional_requirements), "",
            ]
        if record.non_functional_requirements:
            lines += [
                "**Non-Functional Requirements**", "",
                *_md_bullets(record.non_functional_requirements), "",
            ]
        if record.cloud_provider.strip():
            lines += [f"**Cloud Provider:** {record.cloud_provider.strip()}", ""]
        if record.budget.strip():
            lines += [f"**Budget:** {record.budget.strip()}", ""]
        if record.compliance_requirements:
            lines += [
                "**Compliance Requirements**", "",
                *_md_bullets(record.compliance_requirements), "",
            ]
        if record.existing_systems:
            lines += [
                "**Existing Systems**", "", *_md_bullets(record.existing_systems), "",
            ]
        if record.assumptions:
            lines += ["**Assumptions**", "", *_md_bullets(record.assumptions), ""]
        if record.open_questions:
            lines += ["**Open Questions**", "", *_md_bullets(record.open_questions), ""]

    # 4. Recommended Architecture
    blueprint = report.blueprint
    if blueprint is not None:
        lines.append("## Recommended Architecture")
        lines.append("")
        if blueprint.selected_pattern.strip():
            lines.append(f"**Pattern:** {blueprint.selected_pattern.strip()}  ")
        if blueprint.rationale.strip():
            lines += ["", "**Rationale**", "", blueprint.rationale.strip()]
        lines.append("")
        if report.components:
            lines.append("### Components")
            lines.append("")
            lines += _md_table(
                ["ID", "Name", "Type", "Purpose"],
                [
                    [c.id, c.name, c.component_type, c.purpose or "—"]
                    for c in report.components
                ],
            )
            lines.append("")
            for component in report.components:
                lines.append(f"#### {component.id}: {component.name}")
                lines.append("")
                lines.append(component.description.strip())
                lines.append("")
                if component.technology_choices:
                    lines += [
                        "**Technology:** " + ", ".join(component.technology_choices), "",
                    ]
                if component.inputs:
                    lines += ["**Inputs:** " + ", ".join(component.inputs), ""]
                if component.outputs:
                    lines += ["**Outputs:** " + ", ".join(component.outputs), ""]
                if component.dependencies:
                    lines += ["**Dependencies:** " + ", ".join(component.dependencies), ""]
                if component.security_considerations:
                    lines += [
                        "**Security:** " + ", ".join(component.security_considerations), "",
                    ]
                if component.scalability_considerations:
                    lines += [
                        "**Scalability:** "
                        + ", ".join(component.scalability_considerations),
                        "",
                    ]
                if component.related_feature_ids:
                    lines += [
                        "**Traces to features:** "
                        + ", ".join(component.related_feature_ids),
                        "",
                    ]
                if component.related_adr_ids:
                    lines += [
                        "**Justified by:** " + ", ".join(component.related_adr_ids), "",
                    ]

        # 5. Data Flows
        if blueprint.data_flows:
            lines.append("## Data Flows")
            lines.append("")
            lines += _md_bullets(blueprint.data_flows)
            lines.append("")

    # 6. Architecture Decision Records
    if report.adrs:
        lines.append("## Architecture Decision Records")
        lines.append("")
        for adr in report.adrs:
            lines.append(f"### {adr.id}: {adr.title}")
            lines.append("")
            lines.append(f"**Status:** {adr.status}  ")
            if adr.context.strip():
                lines += ["", "**Context**", "", adr.context.strip()]
            lines += ["", "**Decision**", "", adr.decision.strip()]
            if adr.rationale.strip():
                lines += ["", "**Rationale**", "", adr.rationale.strip()]
            if adr.alternatives_considered:
                lines += [
                    "", "**Alternatives Considered**", "",
                    *_md_bullets(adr.alternatives_considered),
                ]
            if adr.positive_consequences:
                lines += [
                    "", "**Positive Consequences**", "",
                    *_md_bullets(adr.positive_consequences),
                ]
            if adr.negative_consequences:
                lines += [
                    "", "**Negative Consequences / Trade-offs**", "",
                    *_md_bullets(adr.negative_consequences),
                ]
            if adr.related_feature_ids:
                lines += ["", "**Related Features:** " + ", ".join(adr.related_feature_ids)]
            if adr.related_component_names:
                lines += [
                    "", "**Related Components:** " + ", ".join(adr.related_component_names),
                ]
            if adr.related_decision_topic_ids:
                lines += [
                    "",
                    "**Related Decision Topics:** "
                    + ", ".join(adr.related_decision_topic_ids),
                ]
            if adr.evidence_ids:
                lines += ["", "**Evidence:** " + ", ".join(adr.evidence_ids)]
            if adr.source_references:
                lines += ["", "**Source References:** " + ", ".join(adr.source_references)]
            lines.append("")

    # 7. Migration Plan
    if blueprint is not None and blueprint.migration_steps:
        lines.append("## Migration Plan")
        lines.append("")
        for index, step in enumerate(blueprint.migration_steps, start=1):
            lines.append(f"### Step {index}: {step.title}")
            lines.append("")
            if step.objective.strip():
                lines += ["**Objective:** " + step.objective.strip(), ""]
            if step.changes:
                lines += ["**Changes**", "", *_md_bullets(step.changes), ""]
            if step.coexistence_or_data_strategy.strip():
                lines += [
                    "**Coexistence / Data Strategy:** "
                    + step.coexistence_or_data_strategy.strip(),
                    "",
                ]
            if step.exit_condition.strip():
                lines += ["**Exit Condition:** " + step.exit_condition.strip(), ""]

    # 8. Evidence / Literature
    curated = [item for item in report.evidence if item.is_curated]
    web = [item for item in report.evidence if not item.is_curated]
    if curated or web:
        lines.append("## Evidence / Literature")
        lines.append("")
        if curated:
            lines.append("### Curated Knowledge Base Evidence")
            lines.append("")
            lines += _md_table(
                ["ID", "Source", "Page", "Excerpt"],
                [
                    [item.evidence_id, item.source or "—", str(item.page), item.excerpt]
                    for item in curated
                ],
            )
            lines.append("")
        if web:
            lines.append("### Grounded Web Fallback (not curated-KB evidence)")
            lines.append("")
            lines += _md_table(
                ["Source", "Excerpt"],
                [[item.source or "—", item.excerpt] for item in web],
            )
            lines.append("")

    # 9. Validation & Reviewer Findings
    if report.review is not None:
        lines.append("## Validation & Reviewer Findings")
        lines.append("")
        lines.append(f"**Overall Verdict:** {report.review.overall_status.upper()}  ")
        if meta.stopped_on_cap:
            lines.append(
                f"**Refinement:** stopped on budget "
                f"({meta.refine_iterations} iteration(s); best round "
                f"{meta.selected_round or 'n/a'} selected)  "
            )
        else:
            lines.append(f"**Refinement rounds:** {meta.refine_iterations}  ")
        lines.append("")
        if report.findings:
            lines.append("### Open Findings")
            lines.append("")
            lines += _md_table(
                ["ID", "Severity", "Category", "Origin", "Finding"],
                [
                    [f.id, f.severity.upper(), f.category, f.origin, f.finding]
                    for f in report.findings
                ],
            )
            lines.append("")
        else:
            lines += ["_No open findings were recorded on the accepted review._", ""]

        if report.waiver is not None:
            counts = report.waiver.severity_counts()
            breakdown = ", ".join(
                f"{counts[name]} {name}" for name in ("high", "medium", "low") if name in counts
            )
            lines.append(
                f"**Waiver:** accepted despite {len(report.waiver.finding_ids)} open "
                f"finding(s) ({breakdown}) — {_datestamp(report.waiver.accepted_at)}."
            )
            if report.waiver.note:
                lines.append(f"Note: {report.waiver.note}")
            lines.append("")

    # 10. Traceability
    trace = report.traceability
    has_trace = any(
        [
            trace.feature_to_adrs, trace.feature_to_components,
            trace.decision_topic_to_adrs, trace.adr_to_evidence,
        ]
    )
    if has_trace:
        lines.append("## Traceability")
        lines.append("")
        if trace.feature_to_adrs or trace.feature_to_components:
            lines.append("### Features → Decisions / Components")
            lines.append("")
            rows = []
            for feature in report.features:
                adrs = ", ".join(trace.feature_to_adrs.get(feature.id, [])) or "—"
                comps = ", ".join(trace.feature_to_components.get(feature.id, [])) or "—"
                rows.append([feature.id, feature.name, adrs, comps])
            lines += _md_table(["Feature", "Name", "ADRs", "Components"], rows)
            lines.append("")
        if trace.decision_topic_to_adrs:
            lines.append("### Decision Topics → ADRs")
            lines.append("")
            rows = [
                [topic.id, topic.topic, ", ".join(trace.decision_topic_to_adrs.get(topic.id, [])) or "—"]
                for topic in report.decision_topics
            ]
            lines += _md_table(["Topic", "Question", "ADRs"], rows)
            lines.append("")
        if trace.adr_to_evidence:
            lines.append("### ADRs → Evidence")
            lines.append("")
            rows = [
                [adr.id, ", ".join(trace.adr_to_evidence.get(adr.id, [])) or "—"]
                for adr in report.adrs
            ]
            lines += _md_table(["ADR", "Evidence"], rows)
            lines.append("")

    # 11. Limitations / Open Items
    lines.append("## Limitations / Open Items")
    lines.append("")
    if report.limitations.is_empty():
        lines.append("_No unresolved items are recorded on this run._")
    else:
        if report.limitations.unresolved_high:
            lines += [
                "**Unresolved HIGH-severity findings**", "",
                *_md_bullets(report.limitations.unresolved_high), "",
            ]
        if report.limitations.unresolved_medium:
            lines += [
                "**Unresolved MEDIUM-severity findings**", "",
                *_md_bullets(report.limitations.unresolved_medium), "",
            ]
        if report.limitations.evidence_gaps:
            lines += [
                "**Verified knowledge-base evidence gaps**", "",
                *_md_bullets(report.limitations.evidence_gaps), "",
            ]
        if report.limitations.other:
            lines += ["**Other**", "", *_md_bullets(report.limitations.other), ""]
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ══════════════════════════════════════════════════════════════════════════
# PDF renderer
# ══════════════════════════════════════════════════════════════════════════
_FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
_FONT_REGULAR = _FONT_DIR / "DejaVuSans.ttf"
_FONT_BOLD = _FONT_DIR / "DejaVuSans-Bold.ttf"

_PAGE_MARGIN = 18
_BODY_SIZE = 10
_H1_SIZE = 18
_H2_SIZE = 14
_H3_SIZE = 11.5


def _build_pdf() -> "FPDF":
    """A fresh document with the vendored Unicode font registered and a
    page-number footer — imported lazily so `pipeline.reporting` stays
    importable even in an environment that has not installed `fpdf2` (only
    `render_pdf`/`generate_reports` need it, never `build_report` or
    `render_markdown`).
    """
    from fpdf import FPDF

    class _ReportPDF(FPDF):
        def footer(self) -> None:  # noqa: D102 — fpdf2 callback, not a public API
            self.set_y(-15)
            self.set_font("DejaVu", "", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    pdf = _ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=_PAGE_MARGIN)
    pdf.set_margins(_PAGE_MARGIN, _PAGE_MARGIN, _PAGE_MARGIN)
    pdf.add_font("DejaVu", "", str(_FONT_REGULAR))
    pdf.add_font("DejaVu", "B", str(_FONT_BOLD))
    pdf.set_text_color(20, 20, 20)
    return pdf


def _heading(pdf: "FPDF", text: str, size: float, *, gap_before: float = 4) -> None:
    pdf.ln(gap_before)
    pdf.set_font("DejaVu", "B", size)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(0, size * 0.5, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf.set_font("DejaVu", "", _BODY_SIZE)


def _paragraph(pdf: "FPDF", text: str, *, bold_label: str = "") -> None:
    """A labelled (or plain) paragraph. `bold_label` alone — text empty — is
    the "heading right before a bullet list" case (see `_bullets` callers
    below); it still prints, because the label is not conditional on there
    being body text under it.
    """
    if not text.strip() and not bold_label:
        return
    if bold_label:
        pdf.set_font("DejaVu", "B", _BODY_SIZE)
        pdf.multi_cell(0, 5.5, bold_label, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("DejaVu", "", _BODY_SIZE)
    if text.strip():
        pdf.multi_cell(0, 5.5, text.strip(), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.5)


def _bullets(pdf: "FPDF", items: list[str]) -> None:
    pdf.set_font("DejaVu", "", _BODY_SIZE)
    for item in items:
        pdf.set_x(_PAGE_MARGIN)
        pdf.multi_cell(0, 5.5, f"•  {item}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.5)


# fpdf2's `Table` cannot split one row across a page break (it raises rather
# than clip when a row's wrapped height would not fit on a single page — see
# `test_pdf_wraps_long_component_and_finding_text_without_raising`). A table
# cell is therefore capped to a length that always wraps to a handful of
# lines regardless of column width; nothing is lost by this — the SAME data
# is always available in full, unclipped, in Markdown, and in the paragraph
# sections a PDF component/ADR also gets (only genuinely tabular columns —
# findings, evidence excerpts — have no such second, unbounded home in the
# PDF, which is the deliberate trade-off this cap makes there).
_TABLE_CELL_CHARS = 320


def _table_cell(value: object) -> str:
    text = str(value)
    return text if len(text) <= _TABLE_CELL_CHARS else text[: _TABLE_CELL_CHARS - 1].rstrip() + "…"


def _table(pdf: "FPDF", headers: list[str], rows: list[list[str]], widths=None) -> None:
    if not rows:
        return
    from fpdf.fonts import FontFace

    pdf.set_font("DejaVu", "", 9)
    with pdf.table(
        col_widths=widths,
        text_align="LEFT",
        line_height=4.6,
        headings_style=FontFace(family="DejaVu", emphasis="B", size_pt=9),
    ) as table:
        table.row(headers)
        for row in rows:
            table.row([_table_cell(cell) for cell in row])
    pdf.ln(2)
    pdf.set_font("DejaVu", "", _BODY_SIZE)


def render_pdf(report: ReportData) -> bytes:
    """Render the SAME canonical model to a formal, paginated PDF report.

    Section order and content match `render_markdown`; only presentation
    differs (a cover block instead of a heading line, tables instead of pipe
    tables, page breaks instead of horizontal rules). Requires `fpdf2` —
    imported lazily inside `_build_pdf` so importing this module never
    requires it for the Markdown-only path.
    """
    pdf = _build_pdf()
    meta = report.meta
    pdf.add_page()

    # 1. Cover / metadata
    pdf.set_font("DejaVu", "B", _H1_SIZE)
    pdf.multi_cell(0, 10, "AI-Architect Architecture Report", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("DejaVu", "", _BODY_SIZE)
    cover_lines = []
    if meta.project_name:
        cover_lines.append(f"Project: {meta.project_name}")
    cover_lines.append(f"Run ID: {meta.run_id}")
    status = "Accepted"
    if meta.accepted_at:
        status += f" — {_datestamp(meta.accepted_at)}"
    cover_lines.append(f"Status: {status}")
    cover_lines.append(
        f"Final Review Verdict: {meta.review_status.upper() if meta.review_status else 'Not reviewed'}"
    )
    if meta.stopped_on_cap:
        cover_lines.append(
            f"Refinement: stopped on budget after round {meta.selected_round or meta.refine_iterations}"
        )
    if meta.models_used:
        cover_lines.append(f"Models used: {', '.join(meta.models_used)}")
    for line in cover_lines:
        pdf.multi_cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")

    # 2. Executive Summary
    if report.executive_summary.strip():
        _heading(pdf, "Executive Summary", _H2_SIZE)
        _paragraph(pdf, report.executive_summary)

    # 3. Requirements & Constraints
    record = report.context_record
    if record is not None:
        _heading(pdf, "Requirements & Constraints", _H2_SIZE)
        if record.business_goal.strip():
            _paragraph(pdf, record.business_goal, bold_label="Business Goal")
        if record.problem_statement.strip():
            _paragraph(pdf, record.problem_statement, bold_label="Problem Statement")
        if record.users:
            _heading(pdf, "Users / Stakeholders", _H3_SIZE, gap_before=1)
            _bullets(pdf, record.users)
        if record.functional_requirements:
            _heading(pdf, "Functional Requirements", _H3_SIZE, gap_before=1)
            _bullets(pdf, record.functional_requirements)
        if record.non_functional_requirements:
            _heading(pdf, "Non-Functional Requirements", _H3_SIZE, gap_before=1)
            _bullets(pdf, record.non_functional_requirements)
        if record.cloud_provider.strip():
            _paragraph(pdf, record.cloud_provider, bold_label="Cloud Provider")
        if record.budget.strip():
            _paragraph(pdf, record.budget, bold_label="Budget")
        if record.compliance_requirements:
            _heading(pdf, "Compliance Requirements", _H3_SIZE, gap_before=1)
            _bullets(pdf, record.compliance_requirements)
        if record.existing_systems:
            _heading(pdf, "Existing Systems", _H3_SIZE, gap_before=1)
            _bullets(pdf, record.existing_systems)

    # 4. Recommended Architecture
    blueprint = report.blueprint
    if blueprint is not None:
        _heading(pdf, "Recommended Architecture", _H2_SIZE)
        if blueprint.selected_pattern.strip():
            _paragraph(pdf, blueprint.selected_pattern, bold_label="Pattern")
        if blueprint.rationale.strip():
            _paragraph(pdf, blueprint.rationale, bold_label="Rationale")
        if report.components:
            _heading(pdf, "Components", _H3_SIZE, gap_before=1)
            _table(
                pdf,
                ["ID", "Name", "Type", "Purpose"],
                [[c.id, c.name, c.component_type, c.purpose or "—"] for c in report.components],
                widths=(20, 45, 30, 95),
            )
            for component in report.components:
                _heading(pdf, f"{component.id}: {component.name}", _H3_SIZE, gap_before=1)
                _paragraph(pdf, component.description)
                if component.technology_choices:
                    _paragraph(
                        pdf, ", ".join(component.technology_choices),
                        bold_label="Technology",
                    )
                if component.related_feature_ids:
                    _paragraph(
                        pdf, ", ".join(component.related_feature_ids),
                        bold_label="Traces to features",
                    )
                if component.related_adr_ids:
                    _paragraph(
                        pdf, ", ".join(component.related_adr_ids),
                        bold_label="Justified by",
                    )

        # 5. Data Flows
        if blueprint.data_flows:
            _heading(pdf, "Data Flows", _H2_SIZE)
            _bullets(pdf, blueprint.data_flows)

    # 6. Architecture Decision Records
    if report.adrs:
        _heading(pdf, "Architecture Decision Records", _H2_SIZE)
        for adr in report.adrs:
            _heading(pdf, f"{adr.id}: {adr.title}", _H3_SIZE, gap_before=1)
            _paragraph(pdf, f"Status: {adr.status}")
            if adr.context.strip():
                _paragraph(pdf, adr.context, bold_label="Context")
            _paragraph(pdf, adr.decision, bold_label="Decision")
            if adr.rationale.strip():
                _paragraph(pdf, adr.rationale, bold_label="Rationale")
            if adr.alternatives_considered:
                _paragraph(pdf, "", bold_label="Alternatives Considered")
                _bullets(pdf, adr.alternatives_considered)
            if adr.positive_consequences:
                _paragraph(pdf, "", bold_label="Positive Consequences")
                _bullets(pdf, adr.positive_consequences)
            if adr.negative_consequences:
                _paragraph(pdf, "", bold_label="Negative Consequences / Trade-offs")
                _bullets(pdf, adr.negative_consequences)
            trailer = []
            if adr.related_feature_ids:
                trailer.append(f"Related Features: {', '.join(adr.related_feature_ids)}")
            if adr.related_component_names:
                trailer.append(f"Related Components: {', '.join(adr.related_component_names)}")
            if adr.related_decision_topic_ids:
                trailer.append(
                    f"Related Decision Topics: {', '.join(adr.related_decision_topic_ids)}"
                )
            if adr.evidence_ids:
                trailer.append(f"Evidence: {', '.join(adr.evidence_ids)}")
            if adr.source_references:
                trailer.append(f"Source References: {', '.join(adr.source_references)}")
            for line in trailer:
                _paragraph(pdf, line)

    # 7. Migration Plan
    if blueprint is not None and blueprint.migration_steps:
        _heading(pdf, "Migration Plan", _H2_SIZE)
        for index, step in enumerate(blueprint.migration_steps, start=1):
            _heading(pdf, f"Step {index}: {step.title}", _H3_SIZE, gap_before=1)
            if step.objective.strip():
                _paragraph(pdf, step.objective, bold_label="Objective")
            if step.changes:
                _paragraph(pdf, "", bold_label="Changes")
                _bullets(pdf, step.changes)
            if step.coexistence_or_data_strategy.strip():
                _paragraph(
                    pdf, step.coexistence_or_data_strategy,
                    bold_label="Coexistence / Data Strategy",
                )
            if step.exit_condition.strip():
                _paragraph(pdf, step.exit_condition, bold_label="Exit Condition")

    # 8. Evidence / Literature
    curated = [item for item in report.evidence if item.is_curated]
    web = [item for item in report.evidence if not item.is_curated]
    if curated or web:
        _heading(pdf, "Evidence / Literature", _H2_SIZE)
        if curated:
            _heading(pdf, "Curated Knowledge Base Evidence", _H3_SIZE, gap_before=1)
            _table(
                pdf,
                ["ID", "Source", "Page", "Excerpt"],
                [[i.evidence_id, i.source or "—", str(i.page), i.excerpt] for i in curated],
                widths=(20, 40, 15, 115),
            )
        if web:
            _heading(
                pdf, "Grounded Web Fallback (not curated-KB evidence)", _H3_SIZE, gap_before=1
            )
            _table(
                pdf, ["Source", "Excerpt"],
                [[i.source or "—", i.excerpt] for i in web],
                widths=(50, 140),
            )

    # 9. Validation & Reviewer Findings
    if report.review is not None:
        _heading(pdf, "Validation & Reviewer Findings", _H2_SIZE)
        _paragraph(pdf, f"Overall Verdict: {report.review.overall_status.upper()}")
        if meta.stopped_on_cap:
            _paragraph(
                pdf,
                f"Refinement: stopped on budget ({meta.refine_iterations} iteration(s); "
                f"best round {meta.selected_round or 'n/a'} selected)",
            )
        else:
            _paragraph(pdf, f"Refinement rounds: {meta.refine_iterations}")
        if report.findings:
            _heading(pdf, "Open Findings", _H3_SIZE, gap_before=1)
            _table(
                pdf,
                ["ID", "Severity", "Category", "Origin", "Finding"],
                [
                    [f.id, f.severity.upper(), f.category, f.origin, f.finding]
                    for f in report.findings
                ],
                widths=(18, 20, 25, 25, 102),
            )
        else:
            _paragraph(pdf, "No open findings were recorded on the accepted review.")
        if report.waiver is not None:
            counts = report.waiver.severity_counts()
            breakdown = ", ".join(
                f"{counts[name]} {name}" for name in ("high", "medium", "low") if name in counts
            )
            _paragraph(
                pdf,
                f"Waiver: accepted despite {len(report.waiver.finding_ids)} open "
                f"finding(s) ({breakdown}) — {_datestamp(report.waiver.accepted_at)}.",
                bold_label="Waiver",
            )
            if report.waiver.note:
                _paragraph(pdf, f"Note: {report.waiver.note}")

    # 10. Traceability
    trace = report.traceability
    if any(
        [trace.feature_to_adrs, trace.feature_to_components,
         trace.decision_topic_to_adrs, trace.adr_to_evidence]
    ):
        _heading(pdf, "Traceability", _H2_SIZE)
        if trace.feature_to_adrs or trace.feature_to_components:
            _heading(pdf, "Features -> Decisions / Components", _H3_SIZE, gap_before=1)
            rows = [
                [
                    feature.id, feature.name,
                    ", ".join(trace.feature_to_adrs.get(feature.id, [])) or "—",
                    ", ".join(trace.feature_to_components.get(feature.id, [])) or "—",
                ]
                for feature in report.features
            ]
            _table(pdf, ["Feature", "Name", "ADRs", "Components"], rows, widths=(20, 50, 55, 55))
        if trace.decision_topic_to_adrs:
            _heading(pdf, "Decision Topics -> ADRs", _H3_SIZE, gap_before=1)
            rows = [
                [topic.id, topic.topic, ", ".join(trace.decision_topic_to_adrs.get(topic.id, [])) or "—"]
                for topic in report.decision_topics
            ]
            _table(pdf, ["Topic", "Question", "ADRs"], rows, widths=(20, 110, 50))
        if trace.adr_to_evidence:
            _heading(pdf, "ADRs -> Evidence", _H3_SIZE, gap_before=1)
            rows = [
                [adr.id, ", ".join(trace.adr_to_evidence.get(adr.id, [])) or "—"]
                for adr in report.adrs
            ]
            _table(pdf, ["ADR", "Evidence"], rows, widths=(30, 150))

    # 11. Limitations / Open Items
    _heading(pdf, "Limitations / Open Items", _H2_SIZE)
    if report.limitations.is_empty():
        _paragraph(pdf, "No unresolved items are recorded on this run.")
    else:
        if report.limitations.unresolved_high:
            _paragraph(pdf, "", bold_label="Unresolved HIGH-severity findings")
            _bullets(pdf, report.limitations.unresolved_high)
        if report.limitations.unresolved_medium:
            _paragraph(pdf, "", bold_label="Unresolved MEDIUM-severity findings")
            _bullets(pdf, report.limitations.unresolved_medium)
        if report.limitations.evidence_gaps:
            _paragraph(pdf, "", bold_label="Verified knowledge-base evidence gaps")
            _bullets(pdf, report.limitations.evidence_gaps)
        if report.limitations.other:
            _paragraph(pdf, "", bold_label="Other")
            _bullets(pdf, report.limitations.other)

    return bytes(pdf.output())
