"""test_material_decisions.py — material architecture decision coverage
(Kush integration, Part 2 of the final grounding-hardening pass).

Real E2E gap: the project could truthfully claim "every final ADR is
traceable to literature actually retrieved this run" — but a live Gemini
run's final design also carried MATERIAL recommendations (a Strangler-style
migration strategy, service-boundary extraction) that never became an ADR
at all, so no existing check could see them. The stronger claim this pass
adds: "every MATERIAL architecture decision is represented by an ADR and
traceable to literature retrieved for the relevant decision topic."

The mechanism is additive, not a new model:
  * `ADR.related_decision_topic_ids` (state.py) — an EXACT, non-fuzzy
    mapping from an ADR to the `DecisionTopic`(s) it decided, reusing the
    SAME bounded, case-derived topic catalog the researcher already plans
    literature retrieval against (`pipeline.agents.researcher`).
  * `review_checks.MATERIAL_DECISION_TOPICS` / `MIGRATION_MATERIAL_TOPIC` —
    the narrow, generic set of categories that are ALWAYS (or, for
    migration, conditionally) material, deliberately excluding the
    conditional topics (observability, technology conservation, ...) so a
    planned topic never forces ADR inflation on its own.
  * `_check_material_decision_coverage` / `_check_adr_evidence_topic_provenance`
    — the deterministic gate: every material topic must resolve to an ADR,
    every ADR's topic references must be real, and evidence cited by a
    topic-mapped ADR must belong to one of ITS OWN mapped topics.

All offline; no LLM calls.
"""

from __future__ import annotations

import inspect

from pipeline.agents import reviewer as rev
from pipeline.agents.researcher import _BASELINE_TOPICS
from pipeline.review_checks import (
    MATERIAL_DECISION_TOPICS,
    MIGRATION_MATERIAL_TOPIC,
    run_deterministic_checks,
)
from pipeline.state import (
    ADR,
    Blueprint,
    ComponentDescription,
    ContextRecord,
    DecisionTopic,
    Feature,
    KBChunk,
    MigrationStep,
    new_run,
)

DECOMPOSITION, DATA_OWNERSHIP, INTEGRATION = MATERIAL_DECISION_TOPICS


def _topic(topic_id, name, evidence_ids=()):
    return DecisionTopic(id=topic_id, topic=name, query="q", evidence_ids=list(evidence_ids))


def _chunk(evidence_id, source):
    return KBChunk(content=f"content for {evidence_id}", source=source, page=1, box=1, evidence_id=evidence_id)


def _adr(adr_id, title, topic_ids, evidence_ids=()):
    return ADR(
        id=adr_id,
        title=title,
        context="ctx",
        decision="decision",
        rationale="rationale",
        related_feature_ids=["FEAT-001"],
        related_component_names=["Svc A"],
        related_decision_topic_ids=list(topic_ids),
        evidence_ids=list(evidence_ids),
    )


def _state(*, topics, adrs, kb_chunks=(), migration_steps=()):
    state = new_run("Design a system.")
    state.context_record = ContextRecord(business_goal="g", problem_statement="p")
    state.features = [Feature(
        id="FEAT-001", name="F", scenario="s", acceptance_criteria=["a"],
    )]
    state.blueprint = Blueprint(
        stakeholder_view="sv", technical_view="tv",
        components=["Svc A"],
        addressed_feature_ids=["FEAT-001"],
        migration_steps=list(migration_steps),
    )
    state.adrs = adrs
    state.components = [ComponentDescription(
        id="COMP-001", name="Svc A", purpose="p", description="d",
        related_feature_ids=["FEAT-001"],
        related_adr_ids=[adr.id for adr in adrs],
    )]
    state.decision_topics = list(topics)
    state.retrieved_knowledge = list(kb_chunks)
    return state


def _baseline_covered(*, migration=False):
    """Fully compliant baseline: all three always-material topics covered
    by their own ADR with genuinely-mapped evidence; optionally the
    migration topic too."""
    topics = [
        _topic("TOPIC-1", DECOMPOSITION, ["KB-E001"]),
        _topic("TOPIC-2", DATA_OWNERSHIP, ["KB-E002"]),
        _topic("TOPIC-3", INTEGRATION, ["KB-E003"]),
    ]
    adrs = [
        _adr("ADR-001", "ADR-1: Decompose into services", ["TOPIC-1"], ["KB-E001"]),
        _adr("ADR-002", "ADR-2: Own data per service", ["TOPIC-2"], ["KB-E002"]),
        _adr("ADR-003", "ADR-3: Integrate asynchronously", ["TOPIC-3"], ["KB-E003"]),
    ]
    kb_chunks = [
        _chunk("KB-E001", "s1.pdf"), _chunk("KB-E002", "s2.pdf"), _chunk("KB-E003", "s3.pdf"),
    ]
    migration_steps = []
    if migration:
        topics.append(_topic("TOPIC-4", MIGRATION_MATERIAL_TOPIC, ["KB-E004"]))
        adrs.append(_adr("ADR-004", "ADR-4: Strangler-fig migration", ["TOPIC-4"], ["KB-E004"]))
        kb_chunks.append(_chunk("KB-E004", "s4.pdf"))
        migration_steps = [MigrationStep(title="Extract order handling", objective="Coexist with legacy.")]
    return topics, adrs, kb_chunks, migration_steps


# ── 1/2: decomposition/extraction — ADR present vs absent ───────────────

def test_material_decomposition_choice_linked_to_adr_passes_coverage():
    topics, adrs, kb_chunks, _ = _baseline_covered()
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks)

    checks = run_deterministic_checks(state)

    assert checks.material_decisions_without_adr == []
    assert not any(issue.category == "adr" and "Material architecture decision" in issue.finding
                   for issue in checks.issues)


def test_material_decomposition_choice_with_no_adr_is_a_blocking_failure():
    topics, adrs, kb_chunks, _ = _baseline_covered()
    # Strip the decomposition ADR's topic mapping — the decision exists
    # (the topic was planned and evidenced) but no ADR records it.
    adrs[0] = adrs[0].model_copy(update={"related_decision_topic_ids": []})
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks)

    checks = run_deterministic_checks(state)

    assert DECOMPOSITION in checks.material_decisions_without_adr
    issue = next(i for i in checks.issues if i.category == "adr" and "Material architecture decision" in i.finding)
    assert issue.severity == "high"
    assert issue.requires_refinement is True


# ── 3/4: migration strategy — material only when actually chosen ────────

def test_material_migration_strategy_with_adr_topic_and_evidence_passes():
    topics, adrs, kb_chunks, migration_steps = _baseline_covered(migration=True)
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks, migration_steps=migration_steps)

    checks = run_deterministic_checks(state)

    assert checks.material_decisions_without_adr == []
    assert checks.invalid_adr_decision_topic_ids == {}
    assert checks.adr_evidence_outside_mapped_topics == {}


def test_material_migration_strategy_outside_the_adr_trail_fails():
    topics, adrs, kb_chunks, migration_steps = _baseline_covered(migration=True)
    # The migration ADR exists textually but never declares the mapping —
    # exactly the real E2E gap (a Strangler-fig plan described in prose,
    # never recorded as a decision).
    adrs[-1] = adrs[-1].model_copy(update={"related_decision_topic_ids": []})
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks, migration_steps=migration_steps)

    checks = run_deterministic_checks(state)

    assert MIGRATION_MATERIAL_TOPIC in checks.material_decisions_without_adr


def test_migration_topic_planned_but_no_migration_steps_is_not_material():
    """The topic can be PLANNED (brownfield repo detected) without a
    migration decision ever being MADE (design stays greenfield-shaped) —
    that must not force a migration ADR."""
    topics, adrs, kb_chunks, _ = _baseline_covered()
    topics.append(_topic("TOPIC-4", MIGRATION_MATERIAL_TOPIC, ["KB-E004"]))
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks, migration_steps=[])

    checks = run_deterministic_checks(state)

    assert MIGRATION_MATERIAL_TOPIC not in checks.material_decisions_without_adr


# ── 5: fabricated / stale topic ID ───────────────────────────────────────

def test_adr_referencing_nonexistent_decision_topic_fails():
    topics, adrs, kb_chunks, _ = _baseline_covered()
    adrs[0] = adrs[0].model_copy(update={"related_decision_topic_ids": ["TOPIC-1", "TOPIC-999"]})
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks)

    checks = run_deterministic_checks(state)

    assert checks.invalid_adr_decision_topic_ids == {"ADR-001": ["TOPIC-999"]}
    assert checks.score_kb_evidence_grounding == 0
    issue = next(i for i in checks.issues if "decision topic ID" in i.finding)
    assert issue.severity == "high"


# ── 6: evidence retrieved for a DIFFERENT mapped topic ───────────────────

def test_adr_citing_evidence_outside_its_mapped_topics_fails_provenance():
    topics, adrs, kb_chunks, _ = _baseline_covered()
    # ADR-001 maps ONLY to decomposition (TOPIC-1) but cites KB-E002, which
    # is genuinely qualifying evidence — just retrieved for the DATA
    # OWNERSHIP topic, not this ADR's own mapped topic.
    adrs[0] = adrs[0].model_copy(update={"evidence_ids": ["KB-E001", "KB-E002"]})
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks)

    checks = run_deterministic_checks(state)

    assert checks.adr_evidence_outside_mapped_topics == {"ADR-001": ["KB-E002"]}
    # KB-E002 is real (retrieved this run), so this is NOT reported as plain
    # fabrication — it is the tighter, topic-scoped provenance failure.
    assert checks.invalid_evidence_ids == {}
    assert checks.score_kb_evidence_grounding == 0


# ── 7: one ADR legitimately spanning two topics ──────────────────────────

def test_one_adr_covering_two_related_topics_is_supported():
    topics = [
        _topic("TOPIC-1", DECOMPOSITION, ["KB-E001"]),
        _topic("TOPIC-2", DATA_OWNERSHIP, ["KB-E002"]),
        _topic("TOPIC-3", INTEGRATION, ["KB-E001"]),
    ]
    adrs = [
        # One decision spans decomposition AND integration style.
        _adr("ADR-001", "ADR-1: Extract services with async integration",
             ["TOPIC-1", "TOPIC-3"], ["KB-E001"]),
        _adr("ADR-002", "ADR-2: Own data per service", ["TOPIC-2"], ["KB-E002"]),
    ]
    kb_chunks = [_chunk("KB-E001", "s1.pdf"), _chunk("KB-E002", "s2.pdf")]
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks)

    checks = run_deterministic_checks(state)

    assert checks.material_decisions_without_adr == []


# ── 8/9: non-mandatory conditional topics never force an ADR ────────────

def test_observability_topic_with_no_adr_is_not_forced():
    topics, adrs, kb_chunks, _ = _baseline_covered()
    topics.append(_topic("TOPIC-5", "observability and operations", []))
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks)

    checks = run_deterministic_checks(state)

    assert checks.material_decisions_without_adr == []


def test_technology_conservation_topic_with_no_adr_is_not_forced():
    topics, adrs, kb_chunks, _ = _baseline_covered()
    topics.append(_topic("TOPIC-5", "technology conservation vs replacement", []))
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks)

    checks = run_deterministic_checks(state)

    assert checks.material_decisions_without_adr == []


# ── 10: missing literature for a real material decision is an honest gap ─

def test_missing_literature_for_material_decision_is_an_honest_gap_not_a_patch():
    topics, adrs, kb_chunks, _ = _baseline_covered()
    # The decomposition topic retrieved NOTHING this run (a real KB
    # coverage gap) — the ADR still correctly records the decision and
    # cites nothing, rather than reaching for an unrelated ID.
    topics[0] = _topic("TOPIC-1", DECOMPOSITION, [])
    adrs[0] = adrs[0].model_copy(update={"evidence_ids": []})
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks[1:])

    checks = run_deterministic_checks(state)

    # The DECISION is still represented — this is an evidence gap, not a
    # missing-ADR failure.
    assert checks.material_decisions_without_adr == []
    assert "ADR-001" in checks.adrs_without_kb_evidence
    assert checks.invalid_evidence_ids == {}
    assert checks.adr_evidence_outside_mapped_topics == {}


# ── 11: Reviewer prompt carries the coverage information ────────────────

def test_reviewer_prompt_contains_material_decision_coverage_information():
    topics, adrs, kb_chunks, migration_steps = _baseline_covered(migration=True)
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks, migration_steps=migration_steps)
    checks = run_deterministic_checks(state)

    prompt = rev._build_prompt(state, checks)

    assert "related_decision_topic_ids" in prompt
    assert "material_decisions_without_adr" in prompt
    assert "TOPIC-4" in prompt


def test_reviewer_system_prompt_asks_about_material_decision_topic_mapping():
    system = rev.REVIEWER_SYSTEM
    assert "related_decision_topic_ids" in system
    assert "hiding outside the ADR trail" in system or "outside the ADR trail" in system


# ── generic, no hard-coding, no catalog drift ────────────────────────────

def test_material_decision_topics_are_a_subset_of_researcher_baseline_topics():
    """MATERIAL_DECISION_TOPICS is an independent policy choice from
    researcher._BASELINE_TOPICS, but must never silently drift from it —
    every mandatory string here has to be one the researcher actually
    plans retrieval for."""
    for topic in MATERIAL_DECISION_TOPICS:
        assert topic in _BASELINE_TOPICS
    assert MIGRATION_MATERIAL_TOPIC not in _BASELINE_TOPICS  # it's conditional, not baseline


def test_material_decision_coverage_carries_no_domain_hardcoding():
    import pipeline.review_checks as rc

    source = "".join([
        inspect.getsource(rc._check_material_decision_coverage),
        inspect.getsource(rc._check_adr_evidence_topic_provenance),
        repr(rc.MATERIAL_DECISION_TOPICS),
        rc.MIGRATION_MATERIAL_TOPIC,
    ])
    forbidden = ["e-commerce", "ecommerce", "shopping cart", "checkout", "order service", "product catalog"]
    lowered = source.lower()
    for term in forbidden:
        assert term not in lowered, f"found domain-specific term {term!r}"
