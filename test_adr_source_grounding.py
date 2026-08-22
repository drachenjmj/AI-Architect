"""test_adr_source_grounding.py — ADR source references must be provenance,
not prose.

Regression for the real E2E run against a brownfield shop repo: the
Architect invented the source label "GDPR/PCI internal compliance best
practices" — not a KB chunk source, not repository text — and the Reviewer
correctly flagged it, but only AFTER the fabricated name was already on the
stored artifact.

The fix under test:
  * the Architect prompt carries a strict SOURCE PROVENANCE rule;
  * a DETERMINISTIC sanitizer at the architect output boundary
    (`architect.sanitize_adr_sources`) removes references that do not
    resolve under the Reviewer's own rule
    (`review_checks.resolve_source_reference`), with no guessed
    substitute, no repair LLM call, and no crash;
  * requirement-derived ADRs with `sources=[]` stay valid.

All offline: LLM calls are stubbed with canned structured outputs. The
benchmark strings are used here only as the adversarial INPUT a model
produced once — nothing is hard-coded in production code (pinned below).
"""

from __future__ import annotations

import pytest

import test_clarifier as tc
from pipeline.agents import architect as arch
from pipeline.review_checks import (
    resolve_source_reference,
    run_deterministic_checks,
)
from pipeline.state import (
    ADR,
    ArchitectState,
    Blueprint,
    ComponentDescription,
    ContextRecord,
    Feature,
    KBChunk,
    RepoMeta,
    RepoRepresentation,
    ReviewResult,
    Stage,
    new_run,
)

FABRICATED = "GDPR/PCI internal compliance best practices"

_KB_SOURCES = ("queue-pattern.md", "gdpr-for-architects.pdf")


def _state() -> ArchitectState:
    state = new_run("Modernize a monolithic shop for peak seasons.")
    state.stage = Stage.RESEARCHING
    state.context_record = ContextRecord(
        project_name="Shop",
        business_goal="Survive peaks.",
        problem_statement="The monolith saturates.",
        functional_requirements=["Process orders"],
        compliance_requirements=["GDPR", "PCI-DSS"],
    )
    state.retrieved_knowledge = [
        KBChunk(content="A durable queue buffers bursts.", source=source)
        for source in _KB_SOURCES
    ]
    state.repo_representation = RepoRepresentation(
        meta=RepoMeta(url="https://example.invalid/shop")
    )
    return state


def _adr(source_references: list[str]) -> ADR:
    return ADR(
        id="ADR-001",
        title="ADR-1: Buffer peak traffic behind a queue",
        context="Peak load saturates the monolith.",
        decision="Buffer orders behind a durable queue.",
        rationale="Requirements demand stable peak throughput.",
        alternatives_considered=["Scale vertically"],
        positive_consequences=["Peak isolation."],
        negative_consequences=["Eventual consistency."],
        source_references=source_references,
    )


# ── the real failure, reproduced ───────────────────────────────────────────


def test_the_real_fabricated_source_is_rejected_by_the_reviewer_rule():
    """Pre-fix behavior: the Reviewer's rule DOES reject the fabricated
    label (it was a writer bug, not a judge bug). This pins the negative
    control the sanitizer must satisfy too."""
    state = _state()
    checks = run_deterministic_checks(state)
    state.adrs = [_adr([FABRICATED])]

    invalid, score = __import__(
        "pipeline.review_checks", fromlist=["_check_source_integrity"]
    )._check_source_integrity(state)
    assert FABRICATED in invalid["ADR-001"]
    assert score == 0


def test_sanitizer_removes_the_fabricated_source_without_substitute():
    adrs, removed = arch.sanitize_adr_sources([_adr([FABRICATED])], _state())

    assert removed == [FABRICATED]
    assert adrs[0].source_references == []      # removed, not replaced
    assert isinstance(adrs[0], ADR)


def test_sanitized_output_cannot_present_the_fake_label_as_valid():
    state = _state()
    state.adrs, _ = arch.sanitize_adr_sources([_adr([FABRICATED])], state)

    checks = run_deterministic_checks(state)
    assert checks.invalid_source_references == {}
    assert checks.score_source_integrity == 2   # no refs left = clean


# ── valid sources survive ──────────────────────────────────────────────────


def test_supplied_kb_sources_remain_unchanged():
    state = _state()
    adrs, removed = arch.sanitize_adr_sources(
        [_adr(["queue-pattern.md", "gdpr-for-architects.pdf"])], state
    )

    assert removed == []
    assert adrs[0].source_references == [
        "queue-pattern.md", "gdpr-for-architects.pdf",
    ]


def test_basename_and_case_variants_resolve_like_the_reviewer():
    """The sanitizer must share the Reviewer's semantics exactly — including
    basename matching and case-insensitivity, and repository-text grounding."""
    state = _state()
    adrs, removed = arch.sanitize_adr_sources(
        [_adr(["docs/patterns/Queue-Pattern.MD", "https://example.invalid/shop"])],
        state,
    )
    # Basename of a supplied source resolves; the repo URL appears in the
    # repository representation JSON, so it resolves as repo evidence.
    assert removed == []
    assert len(adrs[0].source_references) == 2


def test_sanitizer_matches_reviewer_rule_one_to_one():
    state = _state()
    kb_sources = {
        chunk.source.strip().replace("\\", "/").lower()
        for chunk in state.retrieved_knowledge if chunk.source.strip()
    }
    repo_text = state.repo_representation.model_dump_json().lower()
    for reference in (
        FABRICATED, "queue-pattern.md", "Queue-Pattern.MD",
        "docs/queue-pattern.md", "internal best practices",
        "unknown-source.pdf", "https://example.invalid/shop",
    ):
        resolves = resolve_source_reference(reference, kb_sources, repo_text)
        adrs, removed = arch.sanitize_adr_sources([_adr([reference])], state)
        if resolves:
            assert adrs[0].source_references == [reference], reference
        else:
            assert adrs[0].source_references == [], reference


# ── empty provenance stays valid ───────────────────────────────────────────


def test_requirement_derived_adr_with_no_sources_stays_valid():
    state = _state()
    adrs, removed = arch.sanitize_adr_sources([_adr([])], state)

    assert removed == []
    assert adrs[0].source_references == []
    # And the deterministic contract allows it: empty sources score clean.
    state.adrs = adrs
    checks = run_deterministic_checks(state)
    assert checks.score_source_integrity == 2


# ── the node boundary: initial design AND refinement ───────────────────────


class _StubLLM:
    """Two-phase stub: minimal valid outputs, ADR sources as configured."""

    def __init__(self, sources):
        self.sources = sources
        self.calls = 0

    def __call__(self, state, prompt, *, system="", model="",
                 response_schema=None):
        self.calls += 1
        if response_schema is arch.FeatureDesign:
            return arch.FeatureDesign(features=[
                Feature(
                    id="FEAT-001", name="Handle peak load",
                    scenario="Responsive at peak.",
                    related_requirement_ids=["FR-001"],
                    acceptance_criteria=["Stays available."],
                )
            ]), tc.fake_usage()
        assert response_schema is arch.ArchitectureDesign
        return arch.ArchitectureDesign(
            blueprint=Blueprint(
                project_name="Shop",
                selected_pattern="Queue-buffered extraction",
                stakeholder_view="Peak sales keep working.",
                technical_view="Queue in front of order processing.",
                addressed_feature_ids=["FEAT-001"],
            ),
            adrs=[_adr(self.sources)],
            components=[ComponentDescription(
                id="COMP-001", name="Queue Front",
                purpose="Buffer peak submissions.",
                description="Implements FEAT-001.",
                related_feature_ids=["FEAT-001"],
                related_adr_ids=["ADR-001"],
            )],
        ), tc.fake_usage()


@pytest.fixture()
def run_architect(monkeypatch):
    def _run(state, sources):
        stub = _StubLLM(sources)
        monkeypatch.setattr(arch, "llm_call", stub)
        return arch.architect_node(state), stub
    return _run


def test_node_removes_fabricated_source_on_initial_design(run_architect):
    state = _state()
    update, stub = run_architect(state, ["queue-pattern.md", FABRICATED])

    assert [a.source_references for a in update["adrs"]] == [
        ["queue-pattern.md"]
    ]
    # The removal is recorded in the run trace, by name.
    note = update["history"][0].note
    assert "unresolvable ADR source" in note
    assert FABRICATED in note


def test_node_keeps_valid_sources_on_initial_design(run_architect):
    state = _state()
    update, _ = run_architect(state, ["queue-pattern.md"])

    assert update["adrs"][0].source_references == ["queue-pattern.md"]
    assert "unresolvable" not in update["history"][0].note


def test_node_allows_empty_sources(run_architect):
    state = _state()
    update, _ = run_architect(state, [])

    assert update["adrs"][0].source_references == []
    assert "unresolvable" not in update["history"][0].note


def test_refine_pass_cannot_reintroduce_fabricated_sources(run_architect):
    state = _state()
    # A refine pass: features already exist, a failing review is present.
    state.features = [Feature(
        id="FEAT-001", name="Handle peak load",
        scenario="Responsive at peak.",
        acceptance_criteria=["Stays available."],
    )]
    state.review = ReviewResult(
        overall_status="fail", requires_refinement=True,
        refinement_instruction="Tighten the queue rationale.",
    )
    state.blueprint = Blueprint(
        project_name="Shop", stakeholder_view="s", technical_view="t",
    )
    state.adrs = [_adr(["queue-pattern.md"])]
    state.components = [ComponentDescription(
        id="COMP-001", name="Queue Front", purpose="p",
        description="d",
    )]

    update, stub = run_architect(state, [FABRICATED, "gdpr-for-architects.pdf"])

    assert stub.calls == 1                          # refine = phase 2 only
    assert update["adrs"][0].source_references == ["gdpr-for-architects.pdf"]
    assert FABRICATED not in str(update["adrs"][0].model_dump_json())


def test_valid_sources_survive_refine_serialization(run_architect):
    state = _state()
    state.features = [Feature(
        id="FEAT-001", name="Handle peak load",
        scenario="Responsive at peak.",
        acceptance_criteria=["Stays available."],
    )]
    state.review = ReviewResult(
        overall_status="fail", requires_refinement=True,
        refinement_instruction="x",
    )
    state.blueprint = Blueprint(project_name="Shop",
                                stakeholder_view="s", technical_view="t")
    state.adrs = [_adr(["queue-pattern.md"])]
    state.components = [ComponentDescription(
        id="COMP-001", name="Queue Front", purpose="p", description="d",
    )]

    update, _ = run_architect(state, ["queue-pattern.md"])

    # Round-trips through serialization unchanged.
    restored = ADR.model_validate_json(update["adrs"][0].model_dump_json())
    assert restored.source_references == ["queue-pattern.md"]


# ── prompt contract + no benchmark overfit ────────────────────────────────


def test_architect_prompt_carries_the_strict_provenance_rule():
    prompt = " ".join(arch.ARCHITECTURE_SYSTEM_PROMPT.split())
    assert "SOURCE PROVENANCE" in prompt
    assert "PROVENANCE, not prose" in prompt
    assert "ONLY source identifiers" in prompt
    assert "NEVER invent descriptive source labels" in prompt
    assert "internal best practices" in prompt
    assert "leave" in prompt and "`source_references` EMPTY" in prompt
    assert "belongs in the ADR's `rationale`" in prompt


def test_no_benchmark_strings_hard_coded_in_production_code():
    source = (
        arch.ARCHITECTURE_SYSTEM_PROMPT
        + arch.sanitize_adr_sources.__doc__
        + open(arch.__file__, encoding="utf-8").read()
    )
    for banned in (
        "Payment & Data Privacy Proxy",
        FABRICATED,
        "ecommerce-microservice",
        "harsh020",
    ):
        assert banned not in source, banned
