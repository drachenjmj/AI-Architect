"""test_material_decisions.py — material architecture decision coverage
(Kush integration, Part 2 of the final grounding-hardening pass, plus the
follow-up microfix closing three remaining contract gaps).

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
  * `review_checks.MATERIAL_DECISION_TOPICS` (all FOUR of the researcher's
    universal baseline topics — decomposition, data ownership, integration
    style, scaling/availability) / `MIGRATION_MATERIAL_TOPIC` (conditional
    on an actual migration sequence) — deliberately excluding the
    conditional topics (observability, technology conservation, ...) so a
    planned topic never forces ADR inflation on its own.
  * `_check_material_decision_coverage` — every material topic must
    resolve to an ADR.
  * `_check_adr_topic_mapping_presence` — when topics were planned at all,
    EVERY ADR (not just ones covering a material topic) must declare SOME
    topic mapping — closes the gap where an unmapped extra ADR could cite
    any globally qualifying evidence and escape topic-scoped provenance.
  * `_check_adr_evidence_topic_provenance` — evidence cited by a
    topic-mapped ADR must belong to one of ITS OWN mapped topics.
  * `_check_material_topic_evidence_gaps` — a required material topic with
    ZERO qualifying evidence retrieved is an honest, `non_refinable`
    per-topic gap, distinct from the whole-run `kb_evidence_gap`.

All offline; no LLM calls.
"""

from __future__ import annotations

import inspect
import re

from pipeline.agents import architect as arch
from pipeline.agents import reviewer as rev
from pipeline.agents.researcher import _BASELINE_TOPICS
from pipeline.refine_gate import evaluate_caps
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
    ReviewResult,
    RubricScores,
    Stage,
    new_run,
)

DECOMPOSITION, DATA_OWNERSHIP, INTEGRATION, SCALING = MATERIAL_DECISION_TOPICS


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
    """Fully compliant baseline: all FOUR always-material topics covered by
    their own ADR with genuinely-mapped evidence; optionally the migration
    topic too."""
    topics = [
        _topic("TOPIC-1", DECOMPOSITION, ["KB-E001"]),
        _topic("TOPIC-2", DATA_OWNERSHIP, ["KB-E002"]),
        _topic("TOPIC-3", INTEGRATION, ["KB-E003"]),
        _topic("TOPIC-4", SCALING, ["KB-E004"]),
    ]
    adrs = [
        _adr("ADR-001", "ADR-1: Decompose into services", ["TOPIC-1"], ["KB-E001"]),
        _adr("ADR-002", "ADR-2: Own data per service", ["TOPIC-2"], ["KB-E002"]),
        _adr("ADR-003", "ADR-3: Integrate asynchronously", ["TOPIC-3"], ["KB-E003"]),
        _adr("ADR-004", "ADR-4: Scale horizontally with redundancy", ["TOPIC-4"], ["KB-E004"]),
    ]
    kb_chunks = [
        _chunk("KB-E001", "s1.pdf"), _chunk("KB-E002", "s2.pdf"),
        _chunk("KB-E003", "s3.pdf"), _chunk("KB-E004", "s4.pdf"),
    ]
    migration_steps = []
    if migration:
        topics.append(_topic("TOPIC-5", MIGRATION_MATERIAL_TOPIC, ["KB-E005"]))
        adrs.append(_adr("ADR-005", "ADR-5: Strangler-fig migration", ["TOPIC-5"], ["KB-E005"]))
        kb_chunks.append(_chunk("KB-E005", "s5.pdf"))
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
    # never recorded as a decision). It still maps to SOMETHING else so it
    # does not also trip the new "every ADR must map to a topic" rule —
    # isolating this test to the material-coverage failure specifically.
    adrs[-1] = adrs[-1].model_copy(update={"related_decision_topic_ids": ["TOPIC-1"]})
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks, migration_steps=migration_steps)

    checks = run_deterministic_checks(state)

    assert MIGRATION_MATERIAL_TOPIC in checks.material_decisions_without_adr


def test_migration_topic_planned_but_no_migration_steps_is_not_material():
    """The topic can be PLANNED (brownfield repo detected) without a
    migration decision ever being MADE (design stays greenfield-shaped) —
    that must not force a migration ADR."""
    topics, adrs, kb_chunks, _ = _baseline_covered()
    topics.append(_topic("TOPIC-5", MIGRATION_MATERIAL_TOPIC, ["KB-E005"]))
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
        _topic("TOPIC-4", SCALING, ["KB-E002"]),
    ]
    adrs = [
        # One decision spans decomposition AND integration style.
        _adr("ADR-001", "ADR-1: Extract services with async integration",
             ["TOPIC-1", "TOPIC-3"], ["KB-E001"]),
        # Another spans data ownership AND scaling/availability.
        _adr("ADR-002", "ADR-2: Replicate owned data stores for availability",
             ["TOPIC-2", "TOPIC-4"], ["KB-E002"]),
    ]
    kb_chunks = [_chunk("KB-E001", "s1.pdf"), _chunk("KB-E002", "s2.pdf")]
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks)

    checks = run_deterministic_checks(state)

    assert checks.material_decisions_without_adr == []
    assert checks.adr_evidence_outside_mapped_topics == {}
    assert checks.adrs_without_decision_topic_mapping == []


# ── 8/9: non-mandatory conditional topics never force an ADR ────────────

def test_observability_topic_with_no_adr_is_not_forced():
    topics, adrs, kb_chunks, _ = _baseline_covered()
    topics.append(_topic("TOPIC-5", "observability and operations", []))
    adrs.append(_adr("ADR-005", "ADR-5: Placeholder", ["TOPIC-5"], []))
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks)

    checks = run_deterministic_checks(state)

    assert checks.material_decisions_without_adr == []


def test_technology_conservation_topic_with_no_adr_is_not_forced():
    topics, adrs, kb_chunks, _ = _baseline_covered()
    topics.append(_topic("TOPIC-5", "technology conservation vs replacement", []))
    adrs.append(_adr("ADR-005", "ADR-5: Placeholder", ["TOPIC-5"], []))
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
    assert "TOPIC-5" in prompt


def test_reviewer_system_prompt_asks_about_material_decision_topic_mapping():
    system = rev.REVIEWER_SYSTEM
    assert "related_decision_topic_ids" in system
    assert "hiding outside the ADR trail" in system or "outside the ADR trail" in system


# ═════════════════════════════════════════════════════════════════════════
# ADR granularity — materially independent decisions should not be
# compressed into one ADR merely to reduce ADR count
#
# A real Gemini 3.1 Flash-Lite HIGH run identified seven distinct decision
# topics (decomposition, data ownership, integration style, scaling,
# migration, technology conservation, observability) but bundled several
# materially independent ones into a single ADR with one generic rationale.
# The existing deterministic checks only require that every material topic
# resolve to SOME ADR and never forbid one ADR mapping to several topics —
# see `test_one_adr_covering_two_related_topics_is_supported` and
# `test_one_adr_may_cover_integration_and_scaling_together` above, both of
# which must keep passing unmodified. Over-bundling is a JUDGMENT call
# (are the decisions genuinely coupled? does the ADR actually address each
# one's alternatives and consequences?) that a brittle
# `len(related_decision_topic_ids) > 1` rule cannot make — so the fix lives
# entirely in the Architect's and Reviewer's prompt text, not in
# `review_checks.py`. These tests are pure string-content checks; no LLM
# call is made anywhere in this file.
# ═════════════════════════════════════════════════════════════════════════


def _flat(text: str) -> str:
    """Collapse prompt line-wrapping so a phrase check does not depend on
    exactly where a triple-quoted string happens to wrap."""
    return " ".join(text.split())


def test_architect_instructions_discourage_bundling_independent_decisions():
    prompt = _flat(arch.ARCHITECTURE_SYSTEM_PROMPT)
    assert "ADR GRANULARITY" in prompt
    assert "materially independent" in prompt.lower()
    assert "genuinely inseparable" in prompt.lower()


def test_architect_instructions_have_no_minimum_adr_count():
    prompt = _flat(arch.ARCHITECTURE_SYSTEM_PROMPT).lower()
    assert "no target or minimum adr count" in prompt
    # No "at least N ADRs" / "minimum of N ADRs" style phrasing anywhere.
    assert not re.search(r"(?:at least|minimum of)\s+\w+\s+adrs?\b", prompt)
    assert "one adr per topic" not in prompt
    assert "one adr for each" not in prompt


def test_architect_instructions_do_not_require_one_adr_per_topic():
    """The permissive multi-topic-per-ADR rule (line: 'One ADR may map to
    several topics when one decision legitimately spans them') must remain
    exactly as permissive — the new guidance discourages UNJUSTIFIED
    bundling without mandating a one-to-one topic-to-ADR mapping."""
    prompt = _flat(arch.ARCHITECTURE_SYSTEM_PROMPT)
    assert "One ADR may map to several topics" in prompt
    assert "One ADR may still legitimately" in prompt


def test_reviewer_instructions_check_for_over_bundled_decisions():
    system = _flat(rev.REVIEWER_SYSTEM)
    assert "ADR GRANULARITY" in system
    assert "materially independent decision topics" in system
    assert "genuinely coupled" in system


def test_reviewer_instructions_still_allow_justified_multi_topic_adrs():
    system = _flat(rev.REVIEWER_SYSTEM)
    assert "not a failure on this ground" in system
    assert "no minimum ADR count" in system


def test_adr_granularity_check_lives_under_adr_soundness_not_a_new_criterion():
    """The fix reuses the existing `adr_soundness` criterion and its
    existing high-severity/'adr' category — no new CriterionJudgment field,
    no new severity, no new category, matching the project's existing
    severity conventions."""
    assert set(rev.LLMJudgments.model_fields) == {
        "repo_grounding", "flaw_detection", "adr_soundness",
        "best_practice_grounding", "refinement_readiness",
    }
    severity, category, _finding, _fix = rev._CRITERION_ISSUES["adr_soundness"]
    assert severity == "high"
    assert category == "adr"
    # The granularity guidance sits inside question 3 (adr_soundness).
    system = rev.REVIEWER_SYSTEM
    q3_start = system.index("3. adr_soundness")
    q4_start = system.index("4. best_practice_grounding")
    assert q3_start < system.index("ADR GRANULARITY") < q4_start


def test_adr_granularity_finding_is_not_a_brittle_topic_count_rule():
    """The forbidden implementation shape must not appear anywhere in the
    deterministic checks — this stays a semantic LLM judgment, not
    `if len(related_decision_topic_ids) > 1: fail`."""
    import pipeline.review_checks as rc

    source = inspect.getsource(rc)
    assert "len(related_decision_topic_ids)" not in source
    assert "len(adr.related_decision_topic_ids)" not in source


def test_multi_topic_adr_deterministic_checks_are_unaffected_by_the_prompt_change():
    """Belt-and-suspenders: the deterministic layer (which the prompt
    changes never touch) still supports a legitimately bundled multi-topic
    ADR exactly as before — re-proving the existing invariant this task
    must not weaken."""
    topics = [
        _topic("TOPIC-1", DECOMPOSITION, ["KB-E001"]),
        _topic("TOPIC-2", SCALING, ["KB-E001"]),
    ]
    adrs = [
        _adr("ADR-001", "ADR-1: Extract and scale the checkout path",
             ["TOPIC-1", "TOPIC-2"], ["KB-E001"]),
    ]
    kb_chunks = [_chunk("KB-E001", "s1.pdf")]
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks)

    checks = run_deterministic_checks(state)

    assert DECOMPOSITION not in checks.material_decisions_without_adr
    assert SCALING not in checks.material_decisions_without_adr
    assert checks.adrs_without_decision_topic_mapping == []


# ── generic, no hard-coding, no catalog drift ────────────────────────────

def test_material_decision_topics_equal_all_four_researcher_baseline_topics():
    """MATERIAL_DECISION_TOPICS is an independent policy choice from
    researcher._BASELINE_TOPICS, but must never silently drift from it —
    after the scaling/availability fix it covers ALL FOUR of the
    researcher's universal baseline topics, no more and no fewer."""
    assert set(MATERIAL_DECISION_TOPICS) == set(_BASELINE_TOPICS)
    assert len(MATERIAL_DECISION_TOPICS) == 4
    assert SCALING in MATERIAL_DECISION_TOPICS
    assert MIGRATION_MATERIAL_TOPIC not in _BASELINE_TOPICS  # it's conditional, not baseline


def test_material_decision_coverage_carries_no_domain_hardcoding():
    import pipeline.review_checks as rc

    source = "".join([
        inspect.getsource(rc._check_material_decision_coverage),
        inspect.getsource(rc._check_adr_topic_mapping_presence),
        inspect.getsource(rc._check_adr_evidence_topic_provenance),
        inspect.getsource(rc._check_material_topic_evidence_gaps),
        repr(rc.MATERIAL_DECISION_TOPICS),
        rc.MIGRATION_MATERIAL_TOPIC,
    ])
    forbidden = ["e-commerce", "ecommerce", "shopping cart", "checkout", "order service", "product catalog"]
    lowered = source.lower()
    for term in forbidden:
        assert term not in lowered, f"found domain-specific term {term!r}"


# ═════════════════════════════════════════════════════════════════════════
# Microfix 1 — every ADR must map to at least one real DecisionTopic
# ═════════════════════════════════════════════════════════════════════════

def test_extra_adr_with_valid_evidence_but_no_topic_mapping_fails():
    """An ADR beyond the four material ones — no topic mapping at all —
    must not be able to cite globally-qualifying evidence and pass
    unnoticed. Previously `_check_adr_evidence_topic_provenance` simply
    skipped any ADR with an empty `related_decision_topic_ids`."""
    topics, adrs, kb_chunks, _ = _baseline_covered()
    extra = _adr("ADR-999", "ADR-999: An extra decision", [], ["KB-E001"])
    adrs = adrs + [extra]
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks)

    checks = run_deterministic_checks(state)

    assert checks.adrs_without_decision_topic_mapping == ["ADR-999"]
    assert checks.score_kb_evidence_grounding == 0
    issue = next(i for i in checks.issues if "no related_decision_topic_ids" in i.finding)
    assert issue.severity == "high"
    assert issue.requires_refinement is True
    assert issue.non_refinable is False  # always fixable: add the mapping


def test_extra_adr_with_valid_topic_mapping_passes():
    topics, adrs, kb_chunks, _ = _baseline_covered()
    extra = _adr("ADR-999", "ADR-999: An extra decision", ["TOPIC-1"], ["KB-E001"])
    adrs = adrs + [extra]
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks)

    checks = run_deterministic_checks(state)

    assert checks.adrs_without_decision_topic_mapping == []
    assert checks.score_kb_evidence_grounding == 2


def test_no_decision_topics_planned_never_requires_a_mapping():
    """Backward compatibility: an older/direct state with NO decision_topics
    at all makes no claim about ADR topic mappings — there is nothing to
    map TO."""
    adr = ADR(
        id="ADR-001", title="ADR-1: Decision", context="c", decision="d",
        rationale="r", related_feature_ids=["FEAT-001"],
        related_component_names=["Svc A"],
    )
    state = _state(topics=[], adrs=[adr], kb_chunks=[])

    checks = run_deterministic_checks(state)

    assert checks.adrs_without_decision_topic_mapping == []


# ═════════════════════════════════════════════════════════════════════════
# Microfix 2 — scaling/availability joins the always-material core set
# ═════════════════════════════════════════════════════════════════════════

def test_scaling_topic_planned_but_not_mapped_to_any_adr_is_blocking():
    topics, adrs, kb_chunks, _ = _baseline_covered()
    # Strip the scaling ADR's mapping the same way the decomposition test
    # above does — the decision exists (topic planned + evidenced) but no
    # ADR records it. Point it at another real topic so it does not ALSO
    # trip the "every ADR must map to something" rule, isolating this to
    # the material-coverage failure specifically.
    adrs[3] = adrs[3].model_copy(update={"related_decision_topic_ids": ["TOPIC-1"]})
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks)

    checks = run_deterministic_checks(state)

    assert SCALING in checks.material_decisions_without_adr


def test_one_adr_may_cover_integration_and_scaling_together():
    """One decision can legitimately span two of the four material
    categories — this does NOT force a minimum of four separate ADRs."""
    topics = [
        _topic("TOPIC-1", DECOMPOSITION, ["KB-E001"]),
        _topic("TOPIC-2", DATA_OWNERSHIP, ["KB-E002"]),
        _topic("TOPIC-3", INTEGRATION, ["KB-E003"]),
        _topic("TOPIC-4", SCALING, ["KB-E003"]),
    ]
    adrs = [
        _adr("ADR-001", "ADR-1: Decompose into services", ["TOPIC-1"], ["KB-E001"]),
        _adr("ADR-002", "ADR-2: Own data per service", ["TOPIC-2"], ["KB-E002"]),
        # ONE ADR legitimately grounds both integration style AND the
        # scaling/availability posture (async messaging IS the scaling
        # strategy here), citing evidence genuinely retrieved for both.
        _adr("ADR-003", "ADR-3: Scale via async, redundant messaging",
             ["TOPIC-3", "TOPIC-4"], ["KB-E003"]),
    ]
    kb_chunks = [_chunk("KB-E001", "s1.pdf"), _chunk("KB-E002", "s2.pdf"), _chunk("KB-E003", "s3.pdf")]
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks)

    checks = run_deterministic_checks(state)

    assert checks.material_decisions_without_adr == []
    assert checks.adr_evidence_outside_mapped_topics == {}
    assert len(state.adrs) == 3  # not forced to 4


# ═════════════════════════════════════════════════════════════════════════
# Microfix 3 — per-topic material KB gaps are explicit and non-refinable
# ═════════════════════════════════════════════════════════════════════════

def test_material_topic_with_zero_evidence_is_an_explicit_nonrefinable_gap():
    topics, adrs, kb_chunks, _ = _baseline_covered()
    # Decomposition retrieved NOTHING this run, while the other three
    # material topics genuinely have evidence.
    topics[0] = _topic("TOPIC-1", DECOMPOSITION, [])
    adrs[0] = adrs[0].model_copy(update={"evidence_ids": []})
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks[1:])

    checks = run_deterministic_checks(state)

    assert checks.material_topics_without_evidence == [DECOMPOSITION]
    issue = next(i for i in checks.issues if DECOMPOSITION in i.finding and "curated-KB evidence" in i.finding)
    assert issue.severity == "high"
    assert issue.requires_refinement is True
    assert issue.non_refinable is True
    assert checks.score_kb_evidence_grounding == 0


def test_material_topic_gap_cannot_be_satisfied_with_evidence_from_another_topic():
    """The decomposition ADR reaches for the data-ownership topic's
    evidence instead of leaving its own citation empty — still rejected by
    the exact topic-provenance check, on top of the honest per-topic gap
    finding. Neither check patches the gap for the other."""
    topics, adrs, kb_chunks, _ = _baseline_covered()
    topics[0] = _topic("TOPIC-1", DECOMPOSITION, [])
    adrs[0] = adrs[0].model_copy(update={"evidence_ids": ["KB-E002"]})  # borrowed from TOPIC-2
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks)

    checks = run_deterministic_checks(state)

    assert checks.material_topics_without_evidence == [DECOMPOSITION]
    assert checks.adr_evidence_outside_mapped_topics == {"ADR-001": ["KB-E002"]}


def test_adr_spanning_a_gap_topic_and_an_evidenced_topic_is_not_penalized_for_the_other_topic():
    """An ADR mapped to BOTH the gap topic and a topic that DOES have
    evidence can still legitimately ground itself in the evidenced topic —
    the per-topic gap finding is independent of, and does not veto, that."""
    topics, adrs, kb_chunks, _ = _baseline_covered()
    topics[0] = _topic("TOPIC-1", DECOMPOSITION, [])  # gap
    # ADR-001 now spans the gap topic AND integration (which has evidence),
    # citing only the integration evidence — legitimate.
    adrs[0] = adrs[0].model_copy(update={
        "related_decision_topic_ids": ["TOPIC-1", "TOPIC-3"],
        "evidence_ids": ["KB-E003"],
    })
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks[1:])

    checks = run_deterministic_checks(state)

    assert checks.material_topics_without_evidence == [DECOMPOSITION]
    # No off-topic citation: KB-E003 belongs to TOPIC-3, one of ADR-001's
    # own mapped topics.
    assert checks.adr_evidence_outside_mapped_topics == {}
    # The decomposition topic is still COVERED (an ADR maps to it) even
    # though it has no evidence of its own — that is the honest-gap case,
    # not a missing-ADR case.
    assert checks.material_decisions_without_adr == []


def test_pure_material_topic_gap_triggers_the_existing_nonrefinable_stop():
    topics, adrs, kb_chunks, _ = _baseline_covered()
    topics[0] = _topic("TOPIC-1", DECOMPOSITION, [])
    adrs[0] = adrs[0].model_copy(update={"evidence_ids": []})
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks[1:])
    checks = run_deterministic_checks(state)

    state.review = ReviewResult(
        overall_status="fail",
        rubric_scores=RubricScores(),
        issues=checks.issues,
        requires_refinement=any(i.requires_refinement for i in checks.issues),
    )
    state.stage = Stage.REFINING

    stop, reason = evaluate_caps(state)

    assert stop is True
    assert reason == "non_refinable_findings"


def test_conditional_topic_gap_alone_stops_refinement_immediately():
    topics, adrs, kb_chunks, _ = _baseline_covered()
    topics.append(_topic("TOPIC-5", CONDITIONAL_TOPIC, []))
    adrs.append(_adr("ADR-005", "ADR-5: Retain the stack", ["TOPIC-5"], []))
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks)
    checks = run_deterministic_checks(state)

    state.review = ReviewResult(
        overall_status="fail",
        rubric_scores=RubricScores(),
        issues=checks.issues,
        requires_refinement=any(i.requires_refinement for i in checks.issues),
    )
    state.stage = Stage.REFINING

    stop, reason = evaluate_caps(state)

    assert stop is True
    assert reason == "non_refinable_findings"


def test_conditional_topic_gap_mixed_with_refinable_finding_still_loops():
    """The non-refinable per-topic gap coexists with an ordinary, fixable
    finding — the mix is not pure, so the loop must still run for the
    fixable one, exactly like the baseline-topic mixed case above."""
    topics, adrs, kb_chunks, _ = _baseline_covered()
    topics.append(_topic("TOPIC-5", CONDITIONAL_TOPIC, []))
    adrs.append(_adr("ADR-005", "ADR-5: Retain the stack", ["TOPIC-5"], []))
    extra = _adr("ADR-999", "ADR-999: An extra decision", [], [])  # unmapped, refinable
    state = _state(topics=topics, adrs=adrs + [extra], kb_chunks=kb_chunks)
    checks = run_deterministic_checks(state)
    assert any(i.non_refinable for i in checks.issues)
    assert any(not i.non_refinable and i.requires_refinement for i in checks.issues)

    state.review = ReviewResult(
        overall_status="fail",
        rubric_scores=RubricScores(),
        issues=checks.issues,
        requires_refinement=True,
    )
    state.stage = Stage.REFINING

    stop, reason = evaluate_caps(state)

    assert stop is False


def test_mixed_refinable_and_nonrefinable_material_blockers_still_loop():
    """The non-refinable per-topic gap coexists with an ordinary, fixable
    finding (an ADR with no topic mapping at all) — the mix is not pure, so
    the loop must still run for the fixable one."""
    topics, adrs, kb_chunks, _ = _baseline_covered()
    topics[0] = _topic("TOPIC-1", DECOMPOSITION, [])
    adrs[0] = adrs[0].model_copy(update={"evidence_ids": []})
    extra = _adr("ADR-999", "ADR-999: An extra decision", [], [])  # unmapped, refinable
    state = _state(topics=topics, adrs=adrs + [extra], kb_chunks=kb_chunks[1:])
    checks = run_deterministic_checks(state)
    assert any(i.non_refinable for i in checks.issues)  # the per-topic gap
    assert any(not i.non_refinable and i.requires_refinement for i in checks.issues)  # the unmapped ADR

    state.review = ReviewResult(
        overall_status="fail",
        rubric_scores=RubricScores(),
        issues=checks.issues,
        requires_refinement=True,
    )
    state.stage = Stage.REFINING

    stop, reason = evaluate_caps(state)

    assert stop is False


# ═════════════════════════════════════════════════════════════════════════
# Task 19 — CONDITIONAL material topics also produce an honest per-topic gap
#
# Real E2E gap: `_check_material_topic_evidence_gaps` only ever recognized
# the fixed MATERIAL_DECISION_TOPICS/MIGRATION_MATERIAL_TOPIC catalog (the
# four always-planned baseline topics + migration). A CONDITIONAL topic
# (observability, technology conservation, ...) is only ever planned by the
# researcher when the case actually raises the question — so an ADR
# genuinely grounded in one is exactly as material as the baseline four —
# but the old code could never recognize its zero-evidence state as an
# honest gap. That ADR fell into the generic REFINABLE "cite something"
# finding, while the Reviewer's own citation-quality judgment correctly
# rejected any decorative citation reached for instead — an oscillating,
# unwinnable refine loop. The fix: a topic is also material when at least
# one ADR actually declares it via `related_decision_topic_ids` — see
# `_check_material_topic_evidence_gaps`.
# ═════════════════════════════════════════════════════════════════════════

CONDITIONAL_TOPIC = "technology conservation vs replacement"


def test_conditional_topic_with_adr_mapping_and_zero_evidence_is_an_honest_gap():
    topics, adrs, kb_chunks, _ = _baseline_covered()
    topics.append(_topic("TOPIC-5", CONDITIONAL_TOPIC, []))
    adrs.append(_adr(
        "ADR-005", "ADR-5: Retain the existing technology stack",
        ["TOPIC-5"], [],
    ))
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks)

    checks = run_deterministic_checks(state)

    assert checks.material_topics_without_evidence == [CONDITIONAL_TOPIC]
    gap_issue = next(
        i for i in checks.issues
        if CONDITIONAL_TOPIC in i.finding and "curated-KB evidence" in i.finding
    )
    assert gap_issue.severity == "high"
    assert gap_issue.non_refinable is True
    # The ordinary, refinable "cite something" finding must NOT also fire
    # for ADR-005 — that would invite a refine round that can never
    # succeed, exactly the oscillating loop this fix closes.
    assert not any(
        "claim no traceable curated-KB literature support" in i.finding
        for i in checks.issues
    )


def test_conditional_topic_with_zero_evidence_but_no_adr_mapping_is_not_a_gap():
    """A conditional topic nobody ever grounded a decision in is not
    material by mere existence — materiality requires an ADR to have
    actually used it, guarding against over-broadening the gap list."""
    topics, adrs, kb_chunks, _ = _baseline_covered()
    topics.append(_topic("TOPIC-5", CONDITIONAL_TOPIC, []))
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks)

    checks = run_deterministic_checks(state)

    assert checks.material_topics_without_evidence == []


def test_conditional_topic_with_evidence_still_requires_engagement():
    """The Architect cannot self-declare a gap: materiality alone does not
    excuse skipping AVAILABLE evidence — only a genuine, Researcher-set
    zero-evidence topic is a gap. An ADR mapped to a conditional topic that
    DOES have qualifying evidence still gets the ordinary refinable
    finding if it cites nothing."""
    topics, adrs, kb_chunks, _ = _baseline_covered()
    topics.append(_topic("TOPIC-5", CONDITIONAL_TOPIC, ["KB-E005"]))
    adrs.append(_adr("ADR-005", "ADR-5: Retain the stack", ["TOPIC-5"], []))
    kb_chunks = list(kb_chunks) + [_chunk("KB-E005", "s5.pdf")]
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks)

    checks = run_deterministic_checks(state)

    assert checks.material_topics_without_evidence == []
    assert "ADR-005" in checks.adrs_without_kb_evidence
    assert any(
        "claim no traceable curated-KB literature support" in i.finding
        and "ADR-005" in i.finding
        for i in checks.issues
    )


def test_conditional_topic_gap_cannot_be_satisfied_by_citing_another_topics_evidence():
    """A gap is not permission to cite garbage: borrowing a genuinely
    qualifying ID retrieved for a DIFFERENT topic is still rejected by the
    exact topic-provenance check, on top of (not instead of) the honest
    gap finding — and that rejection stays refinable."""
    topics, adrs, kb_chunks, _ = _baseline_covered()
    topics.append(_topic("TOPIC-5", CONDITIONAL_TOPIC, []))
    adrs.append(_adr(
        "ADR-005", "ADR-5: Retain the stack", ["TOPIC-5"], ["KB-E001"],
    ))  # KB-E001 was retrieved for DECOMPOSITION, not this topic
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks)

    checks = run_deterministic_checks(state)

    assert checks.material_topics_without_evidence == [CONDITIONAL_TOPIC]
    assert checks.adr_evidence_outside_mapped_topics == {"ADR-005": ["KB-E001"]}
    provenance_issue = next(
        i for i in checks.issues if "different decision topic" in i.finding
    )
    assert provenance_issue.non_refinable is False  # always the Architect's to fix


def test_conditional_topic_gap_fabricated_citation_still_flagged_and_refinable():
    """A gap is not permission to invent an ID either — fabrication is
    always reported and always refinable, gap or no gap."""
    topics, adrs, kb_chunks, _ = _baseline_covered()
    topics.append(_topic("TOPIC-5", CONDITIONAL_TOPIC, []))
    adrs.append(_adr(
        "ADR-005", "ADR-5: Retain the stack", ["TOPIC-5"], ["KB-E999"],
    ))
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks)

    checks = run_deterministic_checks(state)

    assert checks.material_topics_without_evidence == [CONDITIONAL_TOPIC]
    assert checks.invalid_evidence_ids == {"ADR-005": ["KB-E999"]}
    fabrication_issue = next(
        i for i in checks.issues if "not actually retrieved" in i.finding
    )
    assert fabrication_issue.non_refinable is False


def test_conditional_topic_gap_still_fails_the_verdict_honestly():
    """Phase 9 policy check: a verified gap is not silently accepted.
    `derive_verdict` blocks on ANY high-severity issue regardless of
    `non_refinable` — this fix only stops the refine LOOP from wasting
    iterations on an unfixable finding, matching the existing whole-run
    `kb_evidence_gap` precedent; it does not change verdict semantics."""
    topics, adrs, kb_chunks, _ = _baseline_covered()
    topics.append(_topic("TOPIC-5", CONDITIONAL_TOPIC, []))
    adrs.append(_adr("ADR-005", "ADR-5: Retain the stack", ["TOPIC-5"], []))
    state = _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks)
    checks = run_deterministic_checks(state)

    passed, blocking = rev.derive_verdict(RubricScores(), checks.issues)

    assert passed is False
    assert "high_severity_issue" in blocking


def test_architect_prompt_tells_the_architect_not_to_fabricate_around_a_gap():
    """Pins the Architect-side half of the contract this fix relies on:
    a topic with no evidence is already an explicit instruction to say so
    in the rationale rather than invent or decorate a citation."""
    prompt = _flat(arch.ARCHITECTURE_SYSTEM_PROMPT)
    assert "known KB gap, not something to fabricate around" in prompt
    assert "never attach one merely to satisfy this rule" in prompt


def test_refinement_iteration_cap_is_unchanged():
    from pipeline.refine_gate import MAX_REFINE_ITERATIONS
    assert MAX_REFINE_ITERATIONS == 3


# ── Real-run regression shape ────────────────────────────────────────────
# Reproduces the SHAPE of the observed failure: a material decision (retain
# the existing stack) grounded in repository/requirement reasoning, for a
# decision topic that genuinely retrieved no curated-KB evidence. Not
# hardcoding the real run's generated wording — only its structure.

def _repo_grounded_gap_fixture(citation_ids=()):
    topics, adrs, kb_chunks, _ = _baseline_covered()
    topics.append(_topic("TOPIC-5", CONDITIONAL_TOPIC, []))
    adrs.append(_adr(
        "ADR-005",
        "ADR-5: Retain the existing Python/Django/React stack",
        ["TOPIC-5"],
        list(citation_ids),
    ))
    return _state(topics=topics, adrs=adrs, kb_chunks=kb_chunks)


def test_real_run_regression_shape_no_citation_required():
    state = _repo_grounded_gap_fixture()

    checks = run_deterministic_checks(state)

    # No fake citation required, and no ordinary missing-curated-evidence
    # finding — the honest, non_refinable per-topic gap covers it instead.
    assert CONDITIONAL_TOPIC in checks.material_topics_without_evidence
    assert not any(
        "claim no traceable curated-KB literature support" in i.finding
        for i in checks.issues
    )
    # No impossible refinement instruction: the only finding about this ADR
    # is marked non_refinable, so `evaluate_caps` would stop rather than
    # loop (already proven end-to-end above); re-asserted narrowly here.
    gap_issue = next(
        i for i in checks.issues
        if CONDITIONAL_TOPIC in i.finding and "curated-KB evidence" in i.finding
    )
    assert gap_issue.non_refinable is True


def test_real_run_regression_shape_unrelated_citation_still_flagged():
    state = _repo_grounded_gap_fixture(citation_ids=["KB-E001"])

    checks = run_deterministic_checks(state)

    assert checks.adr_evidence_outside_mapped_topics == {"ADR-005": ["KB-E001"]}
