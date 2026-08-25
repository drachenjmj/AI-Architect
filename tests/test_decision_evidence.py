"""test_decision_evidence.py — decision-level RAG grounding (Kush).

Covers the replacement of one global Top-3 retrieval with bounded,
case-derived decision-topic retrieval (pipeline/agents/researcher.py), the
stable per-run evidence-ID contract (KBChunk.evidence_id / ADR.evidence_ids),
the Architect's decision-evidence prompt blocks and output-boundary
sanitizer (pipeline/agents/architect.py), the deterministic
literature-grounding gate (pipeline/review_checks.py), the non-refinable
refine-loop short-circuit (pipeline/refine_gate.py), and the Reviewer's
semantic-evidence prompt support (pipeline/agents/reviewer.py).

CORRECTIVE PASS (post c62f390/c338096): three real bugs in the first cut are
fixed and tested here —

1. a total KB evidence gap used to score a PERFECT 2 with no issue at all,
   contradicting the project claim; it now scores 0, raises an explicit HIGH
   `non_refinable` finding, and the refine gate stops on it immediately
   instead of burning iterations.
2. the 15-chunk cap used to let the FIRST topic alone fill the whole pool;
   allocation is now a fair rank-layered round-robin across topics, with
   curated-KB topics exhausted before web-fallback ones ever contribute.
3. evidence-ID eligibility is now governed by the retrieval call's actual
   `origin` ("kb" vs "web"/"none"), not by `box` metadata, which was never a
   provenance switch.
4. the same evidence chunk supporting several topics used to have its full
   content rendered once PER TOPIC in the Architect prompt; the prompt is
   now split into a compact `<decision_topics>` ID mapping and a
   `<evidence_catalog>` where each chunk's content appears exactly once.
5. the Reviewer prompt now receives the same evidence catalog/topic mapping
   the Architect saw, and `best_practice_grounding` is explicitly asked to
   judge semantic relevance (does the cited evidence actually support the
   ADR, or was it attached merely to satisfy the deterministic gate).

Everything here is offline: `architect.retrieve_chunks` is replaced with a
recording fake — no Chroma index, no API key, no network.
"""

from __future__ import annotations

import inspect

import architect
from pipeline.agents import architect as arch
from pipeline.agents import researcher as res
from pipeline.agents import reviewer as rev
from pipeline.refine_gate import evaluate_caps
from pipeline.review_checks import (
    qualifying_kb_evidence_ids,
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
    RepoRepresentation,
    ReviewIssue,
    ReviewResult,
    RubricScores,
    Stage,
    TechStack,
    new_run,
)


# ── fakes ────────────────────────────────────────────────────────────────

def _chunk(content, source, page=1, box=1, distance=0.1):
    return {"content": content, "source": source, "page": page, "box": box, "distance": distance}


class FakeRetrieval:
    """Records every query issued and returns canned (chunks, origin) per
    call, cycling through `by_query` (keyed by exact query string) or
    falling back to `default`."""

    def __init__(self, by_query=None, default=None):
        self.by_query = by_query or {}
        self.default = default if default is not None else ([], "none")
        self.calls: list[tuple[str, int]] = []

    def __call__(self, query, k=3):
        self.calls.append((query, k))
        if query in self.by_query:
            return self.by_query[query]
        return self.default


def _state(context_record=None, repo=None):
    state = new_run("Design a system.")
    state.stage = Stage.CLARIFYING
    state.context_record = context_record
    state.repo_representation = repo
    return state


# ── decision-topic planning ─────────────────────────────────────────────

def test_greenfield_plans_only_the_baseline_topics():
    topics = res.plan_decision_topics(None, None)
    assert 4 <= len(topics) <= 8
    assert topics == list(res._BASELINE_TOPICS)


def test_brownfield_with_repo_adds_migration_and_conservatism_topics():
    context = ContextRecord(existing_systems=["Legacy monolith"])
    repo = RepoRepresentation()
    repo.structure.tech_stack = TechStack(languages={"Python": 1000})

    topics = res.plan_decision_topics(context, repo)

    assert "brownfield migration and evolution strategy" in topics
    assert "technology conservation vs replacement" in topics
    assert 4 <= len(topics) <= 8


def test_compliance_requirements_add_a_compliance_topic():
    context = ContextRecord(compliance_requirements=["GDPR"])
    topics = res.plan_decision_topics(context, None)
    assert "security and compliance architecture" in topics


def test_operations_keywords_add_an_observability_topic():
    context = ContextRecord(non_functional_requirements=["24/7 monitoring and alerting"])
    topics = res.plan_decision_topics(context, None)
    assert "observability and operations" in topics


def test_topic_catalog_is_bounded_and_deduplicated_even_with_every_signal():
    context = ContextRecord(
        existing_systems=["Legacy monolith"],
        compliance_requirements=["GDPR"],
        non_functional_requirements=["monitoring required"],
    )
    repo = RepoRepresentation()
    repo.structure.tech_stack = TechStack(languages={"Python": 1000})

    topics = res.plan_decision_topics(context, repo)

    assert len(topics) == len(set(topics))  # no duplicates
    assert len(topics) <= 8


def test_topic_catalog_carries_no_domain_specific_hardcoding():
    """Generic-topic guarantee: scan the actual module source, not just
    behaviour, for e-commerce-specific vocabulary sneaking into the topic
    catalog, its derivation logic, or the fair-allocation algorithm."""

    source = inspect.getsource(res)
    forbidden = ["e-commerce", "ecommerce", "shopping cart", "checkout", "order service", "product catalog"]
    lowered = source.lower()
    for term in forbidden:
        assert term not in lowered, f"found domain-specific term {term!r} in researcher.py"


# ── per-topic retrieval, dedup, cap, ordering ───────────────────────────

def test_multiple_topics_retrieve_independently_with_distinct_queries(monkeypatch):
    context = ContextRecord(problem_statement="orders fail under load")
    state = _state(context_record=context)
    fake = FakeRetrieval(default=([], "none"))
    monkeypatch.setattr(architect, "retrieve_chunks", fake)

    res.researcher_node(state)

    topics = res.plan_decision_topics(context, None)
    assert len(fake.calls) == len(topics)
    queries = [call[0] for call in fake.calls]
    assert len(queries) == len(set(queries))  # every topic gets its own query
    assert all(k == res.PER_TOPIC_K for _q, k in fake.calls)


def test_dedup_collapses_the_same_chunk_retrieved_by_two_topics(monkeypatch):
    state = _state(context_record=ContextRecord())
    shared = _chunk("Use event-driven boundaries.", "pattern.pdf", page=3)
    fake = FakeRetrieval(default=([shared], "kb"))
    monkeypatch.setattr(architect, "retrieve_chunks", fake)

    output = res.researcher_node(state)

    # Every topic returned the SAME chunk; it must appear exactly once.
    assert len(output["retrieved_knowledge"]) == 1
    assert output["retrieved_knowledge"][0].evidence_id == "KB-E001"
    # But every topic's DecisionTopic still records it as its evidence.
    assert all(t.evidence_ids == ["KB-E001"] for t in output["decision_topics"])


def test_evidence_ids_are_stable_for_the_same_evidence_set(monkeypatch):
    state = _state(context_record=ContextRecord())
    chunk = _chunk("stable content", "stable.pdf")
    fake = FakeRetrieval(default=([chunk], "kb"))
    monkeypatch.setattr(architect, "retrieve_chunks", fake)

    first = res.researcher_node(state)["retrieved_knowledge"][0].evidence_id
    second = res.researcher_node(state)["retrieved_knowledge"][0].evidence_id

    assert first == second == "KB-E001"


def test_evidence_gap_is_explicit_per_topic(monkeypatch):
    state = _state(context_record=ContextRecord())
    monkeypatch.setattr(architect, "retrieve_chunks", FakeRetrieval(default=([], "none")))

    output = res.researcher_node(state)

    assert all(topic.evidence_ids == [] for topic in output["decision_topics"])
    assert output["retrieved_knowledge"] == []
    assert "no qualifying evidence" in output["history"][0].note


# ── Finding 2: fair, rank-layered allocation across topics ──────────────

def test_first_topic_no_longer_starves_every_other_topic(monkeypatch):
    """Regression for the original bug: 8 topics, each with a REALISTIC
    k<=3 response (never a fake returning 15 items for one k=3 call). With
    a cap of 15 and 8 topics, every topic must receive its own rank-0 hit
    before any topic receives a second — proving a later topic's first hit
    survives even though 7 other topics exist."""
    context = ContextRecord(
        existing_systems=["Legacy monolith"],
        compliance_requirements=["GDPR"],
        non_functional_requirements=["monitoring required"],
    )
    repo = RepoRepresentation()
    repo.structure.tech_stack = TechStack(languages={"Python": 1000})
    state = _state(context_record=context, repo=repo)

    topics = res.plan_decision_topics(context, repo)
    assert len(topics) == 8  # every conditional signal fired

    by_query = {}
    for index, topic in enumerate(topics):
        query = res._topic_query(topic, context)
        # Realistic: at most PER_TOPIC_K (3) chunks per call, like the real
        # retrieve_chunks contract.
        chunks = [
            _chunk(f"topic {index} rank {rank}", f"topic-{index}-rank-{rank}.pdf")
            for rank in range(res.PER_TOPIC_K)
        ]
        by_query[query] = (chunks, "kb")
    fake = FakeRetrieval(by_query=by_query)
    monkeypatch.setattr(architect, "retrieve_chunks", fake)

    output = res.researcher_node(state)

    assert len(output["retrieved_knowledge"]) == res.MAX_TOTAL_EVIDENCE  # 8*3=24 candidates, capped at 15
    # Every one of the 8 topics has AT LEAST ONE evidence ID — the bug this
    # test exists to catch let the first topic alone consume the cap.
    topics_with_evidence = [t for t in output["decision_topics"] if t.evidence_ids]
    assert len(topics_with_evidence) == 8
    # And every topic's FIRST (rank-0) hit specifically survived.
    rank0_sources = {f"topic-{i}-rank-0.pdf" for i in range(8)}
    kept_sources = {chunk.source for chunk in output["retrieved_knowledge"]}
    assert rank0_sources <= kept_sources


def test_allocation_is_deterministic_and_reproducible(monkeypatch):
    context = ContextRecord(compliance_requirements=["GDPR"])
    state = _state(context_record=context)
    topics = res.plan_decision_topics(context, None)
    by_query = {
        res._topic_query(topic, context): (
            [_chunk(f"c{index}-{rank}", f"s{index}-{rank}.pdf") for rank in range(3)],
            "kb",
        )
        for index, topic in enumerate(topics)
    }
    monkeypatch.setattr(architect, "retrieve_chunks", FakeRetrieval(by_query=dict(by_query)))

    first = [c.evidence_id for c in res.researcher_node(state)["retrieved_knowledge"]]
    second = [c.evidence_id for c in res.researcher_node(state)["retrieved_knowledge"]]

    assert first == second


def test_allocate_evidence_fills_rank_layers_across_topics():
    """Direct unit test of the allocator: 3 topics, 3 candidates each, cap
    of 5. Rank 0 for every topic fills first (3 slots), then rank 1 for the
    first 2 topics only (cap reached) — never topic 0's rank 2 before
    topic 1/2's rank 0."""
    topic_chunks = [
        [_chunk(f"t{t}-r{r}", f"t{t}-r{r}.pdf") for r in range(3)]
        for t in range(3)
    ]
    topic_origins = ["kb", "kb", "kb"]

    pool, pool_origins, indices = res.allocate_evidence(topic_chunks, topic_origins, cap=5)

    assert len(pool) == 5
    sources = [c["source"] for c in pool]
    assert sources == ["t0-r0.pdf", "t1-r0.pdf", "t2-r0.pdf", "t0-r1.pdf", "t1-r1.pdf"]
    assert pool_origins == ["kb"] * 5
    assert indices[0] == [0, 3]  # topic 0 contributed rank-0 and rank-1
    assert indices[1] == [1, 4]
    assert indices[2] == [2]     # topic 2's rank-1 never made it in


def test_allocate_evidence_exhausts_kb_topics_before_touching_web():
    """A web-origin topic must not crowd out a KB-origin topic's evidence
    while KB candidates remain unconsidered — even when the web topic is
    processed first in the input order."""
    topic_chunks = [
        [_chunk("web content", "web.pdf")],       # topic 0: web-origin
        [_chunk(f"kb-{r}", f"kb-{r}.pdf") for r in range(3)],  # topic 1: kb-origin
    ]
    topic_origins = ["web", "kb"]

    pool, pool_origins, indices = res.allocate_evidence(topic_chunks, topic_origins, cap=2)

    # Cap is 2: both of the KB topic's candidates fit before the web topic
    # contributes anything at all.
    assert pool_origins == ["kb", "kb"]
    assert [c["source"] for c in pool] == ["kb-0.pdf", "kb-1.pdf"]
    assert indices[0] == []  # the web topic got nothing — cap already spent


def test_web_topic_still_fills_remaining_capacity_after_kb_is_exhausted():
    topic_chunks = [
        [_chunk("web content", "web.pdf")],
        [_chunk("kb content", "kb.pdf")],
    ]
    topic_origins = ["web", "kb"]

    pool, pool_origins, _indices = res.allocate_evidence(topic_chunks, topic_origins, cap=5)

    assert pool_origins == ["kb", "web"]  # kb processed first regardless of input order
    assert {c["source"] for c in pool} == {"kb.pdf", "web.pdf"}


# ── cross-origin evidence dedup provenance ───────────────────────────────

def test_allocate_evidence_web_topic_does_not_inherit_kb_evidence_via_dedup():
    """Direct unit test of the allocator for the cross-origin dedup bug: a
    KB topic and a web topic both retrieve the IDENTICAL chunk. The web
    topic must not be handed the KB pool index — it gets nothing, not a
    borrowed citation."""
    shared = _chunk("shared content", "shared.pdf")
    topic_chunks = [[shared], [shared]]
    topic_origins = ["kb", "web"]

    pool, pool_origins, indices = res.allocate_evidence(topic_chunks, topic_origins, cap=15)

    assert len(pool) == 1
    assert pool_origins == ["kb"]
    assert indices[0] == [0]  # KB topic owns the pool item
    assert indices[1] == []  # web topic's identical hit is a no-op, not a borrowed citation


def test_cross_origin_dedup_web_topic_does_not_inherit_kb_evidence_id(monkeypatch):
    """End-to-end regression: KB Topic A and web Topic B both retrieve the
    same chunk. A gets the citable KB-E001; B must NOT inherit it merely
    because A's identical chunk was already in the pool."""
    context = ContextRecord()
    state = _state(context_record=context)
    shared = _chunk("Use event-driven boundaries.", "pattern.pdf", page=3)
    topic_a, topic_b = res._BASELINE_TOPICS[0], res._BASELINE_TOPICS[1]
    by_query = {
        res._topic_query(topic_a, context): ([shared], "kb"),
        res._topic_query(topic_b, context): ([shared], "web"),
    }
    fake = FakeRetrieval(by_query=by_query, default=([], "none"))
    monkeypatch.setattr(architect, "retrieve_chunks", fake)

    output = res.researcher_node(state)

    topics_by_name = {t.topic: t for t in output["decision_topics"]}
    assert topics_by_name[topic_a].evidence_ids == ["KB-E001"]
    assert topics_by_name[topic_b].evidence_ids == []
    assert len(output["retrieved_knowledge"]) == 1
    assert output["retrieved_knowledge"][0].evidence_id == "KB-E001"


def test_two_kb_topics_sharing_identical_chunk_both_get_the_same_evidence_id(monkeypatch):
    context = ContextRecord()
    state = _state(context_record=context)
    shared = _chunk("Use event-driven boundaries.", "pattern.pdf", page=3)
    topic_a, topic_b = res._BASELINE_TOPICS[0], res._BASELINE_TOPICS[1]
    by_query = {
        res._topic_query(topic_a, context): ([shared], "kb"),
        res._topic_query(topic_b, context): ([shared], "kb"),
    }
    fake = FakeRetrieval(by_query=by_query, default=([], "none"))
    monkeypatch.setattr(architect, "retrieve_chunks", fake)

    output = res.researcher_node(state)

    topics_by_name = {t.topic: t for t in output["decision_topics"]}
    assert topics_by_name[topic_a].evidence_ids == ["KB-E001"]
    assert topics_by_name[topic_b].evidence_ids == ["KB-E001"]
    assert len(output["retrieved_knowledge"]) == 1


def test_two_web_topics_sharing_identical_chunk_neither_gets_an_evidence_id(monkeypatch):
    context = ContextRecord()
    state = _state(context_record=context)
    shared = _chunk("Use event-driven boundaries.", "pattern.pdf", page=3)
    topic_a, topic_b = res._BASELINE_TOPICS[0], res._BASELINE_TOPICS[1]
    by_query = {
        res._topic_query(topic_a, context): ([shared], "web"),
        res._topic_query(topic_b, context): ([shared], "web"),
    }
    fake = FakeRetrieval(by_query=by_query, default=([], "none"))
    monkeypatch.setattr(architect, "retrieve_chunks", fake)

    output = res.researcher_node(state)

    topics_by_name = {t.topic: t for t in output["decision_topics"]}
    assert topics_by_name[topic_a].evidence_ids == []
    assert topics_by_name[topic_b].evidence_ids == []
    assert len(output["retrieved_knowledge"]) == 1
    assert output["retrieved_knowledge"][0].evidence_id == ""


# ── Finding 3: origin, not box, is the provenance authority ─────────────

def test_web_origin_never_gets_an_evidence_id_even_with_malformed_box(monkeypatch):
    """A malformed/future caller could report a web-origin chunk with a
    non-3 box; origin must still govern, so it gets no evidence_id."""
    state = _state(context_record=ContextRecord())
    malformed_web_chunk = _chunk("looks curated but is not", "sketchy.pdf", box=1)
    fake = FakeRetrieval(default=([malformed_web_chunk], "web"))
    monkeypatch.setattr(architect, "retrieve_chunks", fake)

    output = res.researcher_node(state)

    assert all(chunk.evidence_id == "" for chunk in output["retrieved_knowledge"])
    assert all(topic.evidence_ids == [] for topic in output["decision_topics"])


def test_kb_origin_is_required_for_an_evidence_id(monkeypatch):
    state = _state(context_record=ContextRecord())
    chunk = _chunk("real curated content", "curated.pdf", box=1)
    fake = FakeRetrieval(default=([chunk], "kb"))
    monkeypatch.setattr(architect, "retrieve_chunks", fake)

    output = res.researcher_node(state)

    assert output["retrieved_knowledge"][0].evidence_id == "KB-E001"


def test_raw_chunk_dict_contract_is_unchanged(monkeypatch):
    """`retrieve_chunks` chunk dicts still carry exactly the original five
    keys — origin is tracked alongside by the researcher, never merged into
    the chunk dict itself."""
    state = _state(context_record=ContextRecord())
    chunk = _chunk("content", "source.pdf")
    fake = FakeRetrieval(default=([chunk], "kb"))
    monkeypatch.setattr(architect, "retrieve_chunks", fake)

    res.researcher_node(state)
    for call_query, k in fake.calls:
        assert k == res.PER_TOPIC_K
    assert set(chunk) == {"content", "source", "page", "box", "distance"}


# ── deterministic Reviewer literature-grounding gate ────────────────────

def _adr(**overrides):
    base = dict(
        id="ADR-001",
        title="ADR-1: Do the thing",
        context="ctx",
        decision="decision",
        rationale="rationale",
        alternatives_considered=["alt"],
        positive_consequences=["pos"],
        negative_consequences=["neg"],
        related_feature_ids=["FEAT-001"],
        related_component_names=["Order Service"],
    )
    base.update(overrides)
    return ADR(**base)


def _design_state(adrs, retrieved_knowledge, decision_topics=None):
    state = new_run("Design a system.")
    state.context_record = ContextRecord(business_goal="g", problem_statement="p")
    state.features = [Feature(id="FEAT-001", name="F", scenario="s", acceptance_criteria=["a"])]
    state.blueprint = Blueprint(
        stakeholder_view="sv", technical_view="tv", components=["Order Service"],
        addressed_feature_ids=["FEAT-001"],
    )
    state.adrs = adrs
    state.components = [
        ComponentDescription(
            id="COMP-001", name="Order Service", purpose="p", description="d",
            related_feature_ids=["FEAT-001"], related_adr_ids=["ADR-001"],
        )
    ]
    state.retrieved_knowledge = retrieved_knowledge
    state.decision_topics = decision_topics or []
    return state


def test_adr_with_valid_retrieved_kb_evidence_passes_the_gate():
    kb = KBChunk(content="c", source="s.pdf", page=1, box=1, evidence_id="KB-E001")
    state = _design_state([_adr(evidence_ids=["KB-E001"])], [kb])

    assert qualifying_kb_evidence_ids(state) == {"KB-E001"}
    checks = run_deterministic_checks(state)

    assert checks.score_kb_evidence_grounding == 2
    assert checks.adrs_without_kb_evidence == []
    assert checks.invalid_evidence_ids == {}
    assert checks.issues == []


def test_adr_with_fabricated_evidence_id_fails_the_gate():
    kb = KBChunk(content="c", source="s.pdf", page=1, box=1, evidence_id="KB-E001")
    state = _design_state([_adr(evidence_ids=["KB-E999"])], [kb])

    checks = run_deterministic_checks(state)

    assert checks.score_kb_evidence_grounding == 0
    assert checks.invalid_evidence_ids == {"ADR-001": ["KB-E999"]}
    issue = next(i for i in checks.issues if i.category == "evidence")
    assert issue.severity == "high"
    assert issue.requires_refinement is True
    # Fabrication is always the Architect's own output to fix — never
    # treated as an unfixable KB coverage gap.
    assert issue.non_refinable is False


def test_adr_with_no_evidence_fails_when_some_other_adr_engages():
    """Partial coverage: one ADR cites real evidence, another cites
    nothing, while qualifying evidence genuinely existed. Still blocks
    (score < 2, never silently passes), and the finding text names both
    the uncited ADR and which decision topics DO have evidence — so a
    reader can tell an Architect oversight apart from a per-decision KB
    coverage gap (Finding 6)."""
    kb = KBChunk(content="c", source="s.pdf", page=1, box=1, evidence_id="KB-E001")
    engaged = _adr(id="ADR-001", evidence_ids=["KB-E001"], related_decision_topic_ids=["TOPIC-1"])
    unengaged = _adr(
        id="ADR-002", title="ADR-2: Do another thing", evidence_ids=[],
        related_decision_topic_ids=["TOPIC-1"],
    )
    topics = [DecisionTopic(id="TOPIC-1", topic="data ownership", query="q", evidence_ids=["KB-E001"])]
    state = _design_state([engaged, unengaged], [kb], decision_topics=topics)

    checks = run_deterministic_checks(state)

    assert checks.score_kb_evidence_grounding == 1  # partial, still blocking
    assert checks.adrs_without_kb_evidence == ["ADR-002"]
    assert checks.kb_evidence_gap is False
    issue = next(i for i in checks.issues if i.category == "evidence")
    assert issue.severity == "high"
    assert "ADR-002" in issue.finding
    assert "data ownership" in issue.evidence  # topics_with_evidence is named


def test_complete_disengagement_with_available_evidence_fails():
    """No ADR cites anything at all despite qualifying evidence existing —
    categorically worse than a partial gap, and must not pass."""
    kb = KBChunk(content="c", source="s.pdf", page=1, box=1, evidence_id="KB-E001")
    state = _design_state([_adr(evidence_ids=[])], [kb])

    checks = run_deterministic_checks(state)

    assert checks.score_kb_evidence_grounding == 0
    assert checks.adrs_without_kb_evidence == ["ADR-001"]
    assert checks.kb_evidence_gap is False


def test_total_kb_gap_fails_honestly_and_is_non_refinable():
    """BLOCKER fix: a run with zero qualifying curated-KB evidence anywhere
    must NOT be reported as fully literature-grounded. It scores 0, raises
    an explicit HIGH finding, and that finding is marked non_refinable so
    the refine loop does not spend iterations on an unfixable gap."""
    state = _design_state([_adr(evidence_ids=[])], [])

    checks = run_deterministic_checks(state)

    assert checks.kb_evidence_gap is True
    assert checks.score_kb_evidence_grounding == 0
    issue = next(i for i in checks.issues if i.category == "evidence")
    assert issue.severity == "high"
    assert issue.requires_refinement is True
    assert issue.non_refinable is True
    assert "literature-grounded" in issue.finding


def test_total_kb_gap_fabrication_is_still_reported_as_fabrication():
    """Zero qualifying evidence exists, yet an ADR cites an ID anyway — that
    is fabrication ON TOP OF a gap, not an excused gap, and it must not be
    marked non_refinable (removing the fabricated ID is always possible)."""
    state = _design_state([_adr(evidence_ids=["KB-E001"])], [])

    checks = run_deterministic_checks(state)

    assert checks.kb_evidence_gap is True
    assert checks.invalid_evidence_ids == {"ADR-001": ["KB-E001"]}
    assert checks.score_kb_evidence_grounding == 0


def test_web_fallback_evidence_cannot_satisfy_the_gate():
    """A web chunk carries no evidence_id (see researcher.py), so it can
    never be cited — proven here by an ADR trying to cite the KB-shaped ID
    a web chunk would have needed, which never resolves."""
    web = KBChunk(content="c", source="https://example.com", box=3, evidence_id="")
    state = _design_state([_adr(evidence_ids=["KB-E001"])], [web])

    assert qualifying_kb_evidence_ids(state) == set()
    checks = run_deterministic_checks(state)

    # No qualifying evidence exists (web doesn't count) AND the ADR still
    # tried to cite something → fabrication, not a silent gap.
    assert checks.score_kb_evidence_grounding == 0
    assert checks.invalid_evidence_ids == {"ADR-001": ["KB-E001"]}


def test_duplicate_evidence_references_are_normalized():
    kb = KBChunk(content="c", source="s.pdf", page=1, box=1, evidence_id="KB-E001")
    adrs, dropped = arch.sanitize_adr_evidence_ids(
        [_adr(evidence_ids=["KB-E001", "KB-E001"])],
        _design_state([], [kb]),
    )
    assert adrs[0].evidence_ids == ["KB-E001"]
    assert dropped == []


def test_sanitizer_drops_fabricated_ids_and_reports_them():
    kb = KBChunk(content="c", source="s.pdf", page=1, box=1, evidence_id="KB-E001")
    adrs, dropped = arch.sanitize_adr_evidence_ids(
        [_adr(evidence_ids=["KB-E001", "KB-E999"])],
        _design_state([], [kb]),
    )
    assert adrs[0].evidence_ids == ["KB-E001"]
    assert dropped == ["KB-E999"]


# ── Finding 1: refine gate stops on a purely non-refinable review ───────

def _review_with_issues(issues: list[ReviewIssue]) -> ReviewResult:
    return ReviewResult(
        overall_status="fail",
        rubric_scores=RubricScores(),
        issues=issues,
        requires_refinement=any(i.requires_refinement for i in issues),
    )


def test_refine_gate_stops_immediately_on_a_pure_non_refinable_gap():
    state = _design_state([_adr(evidence_ids=[])], [])
    state.review = _review_with_issues(run_deterministic_checks(state).issues)
    state.stage = Stage.REFINING

    stop, reason = evaluate_caps(state)

    assert stop is True
    assert reason == "non_refinable_findings"


def test_refine_gate_keeps_looping_when_a_refinable_finding_is_mixed_in():
    """A run with BOTH the unfixable gap AND an ordinary fixable finding
    must still loop — the mix is not pure, so there is real work left."""
    gap_issue = ReviewIssue(
        id="DET-1", severity="high", category="evidence",
        finding="no evidence", evidence="e", suggested_fix="expand KB",
        requires_refinement=True, non_refinable=True,
    )
    fixable_issue = ReviewIssue(
        id="DET-2", severity="high", category="completeness",
        finding="missing field", evidence="e", suggested_fix="fill it in",
        requires_refinement=True, non_refinable=False,
    )
    state = _design_state([_adr(evidence_ids=[])], [])
    state.review = _review_with_issues([gap_issue, fixable_issue])
    state.stage = Stage.REFINING

    stop, reason = evaluate_caps(state)

    assert stop is False


def test_refine_gate_does_not_short_circuit_ordinary_refinable_reviews():
    fixable_issue = ReviewIssue(
        id="DET-1", severity="high", category="completeness",
        finding="missing field", evidence="e", suggested_fix="fill it in",
        requires_refinement=True, non_refinable=False,
    )
    state = _design_state([_adr(evidence_ids=[])], [])
    state.review = _review_with_issues([fixable_issue])
    state.stage = Stage.REFINING

    stop, reason = evaluate_caps(state)

    assert stop is False
    assert reason == ""


def test_refine_gate_unaffected_when_review_is_none():
    state = _design_state([_adr(evidence_ids=[])], [])
    state.review = None

    stop, reason = evaluate_caps(state)

    assert stop is False


# ── Finding 4: split decision-evidence prompt blocks, no duplication ────

def test_decision_topics_block_is_id_only_no_content_duplication():
    kb1 = KBChunk(content="content one", source="s1.pdf", page=1, box=1, evidence_id="KB-E001")
    state = new_run("Design a system.")
    state.retrieved_knowledge = [kb1]
    state.decision_topics = [
        DecisionTopic(id="TOPIC-1", topic="data ownership", query="q1", evidence_ids=["KB-E001"]),
        DecisionTopic(id="TOPIC-2", topic="scaling", query="q2", evidence_ids=["KB-E001"]),
    ]

    topics_block = arch.decision_topics_block(state)

    assert "data ownership" in topics_block
    assert "scaling" in topics_block
    assert "KB-E001" in topics_block
    assert "content one" not in topics_block  # no chunk body in the mapping


def test_evidence_catalog_renders_each_chunk_exactly_once_across_topics():
    """Regression: the same chunk supporting FOUR topics used to have its
    full content duplicated four times in the Architect prompt."""
    kb1 = KBChunk(content="shared content", source="s1.pdf", page=1, box=1, evidence_id="KB-E001")
    state = new_run("Design a system.")
    state.retrieved_knowledge = [kb1]
    state.decision_topics = [
        DecisionTopic(id=f"TOPIC-{n}", topic=f"topic {n}", query="q", evidence_ids=["KB-E001"])
        for n in range(1, 5)
    ]

    catalog = arch.evidence_catalog_block(state)
    full_block = arch._decision_evidence_block(state)

    assert catalog.count("shared content") == 1
    assert full_block.count("shared content") == 1
    assert full_block.count("KB-E001") >= 4  # cited by every topic in the mapping...
    # ...but the CONTENT-bearing catalog entry is not repeated per topic.


def test_web_context_block_is_separate_and_labelled_non_citable():
    web = KBChunk(content="web finding", source="https://example.com", box=3, evidence_id="")
    state = new_run("Design a system.")
    state.retrieved_knowledge = [web]

    catalog = arch.evidence_catalog_block(state)
    web_block = arch.web_context_block(state)

    assert "web finding" not in catalog  # never mixed into the citable catalog
    assert "web finding" in web_block
    assert "NOT curated KB literature" in web_block


def test_web_context_block_is_empty_string_when_nothing_retrieved():
    state = new_run("Design a system.")
    assert arch.web_context_block(state) == ""


def test_decision_topics_block_states_the_gap_for_an_empty_topic():
    state = new_run("Design a system.")
    state.decision_topics = [
        DecisionTopic(id="TOPIC-1", topic="security and compliance architecture", query="q", evidence_ids=[]),
    ]

    block = arch.decision_topics_block(state)

    assert "KB gap for this topic" in block


def test_decision_evidence_block_falls_back_when_no_topics_planned():
    """States built without topic planning (older checkpoints, direct-state
    tests) still show the Architect what it may cite."""
    kb1 = KBChunk(content="c1", source="s1.pdf", page=1, box=1, evidence_id="KB-E001")
    state = new_run("Design a system.")
    state.retrieved_knowledge = [kb1]

    block = arch._decision_evidence_block(state)

    assert "KB-E001" in block
    assert "c1" in block


# ── Finding 5: Reviewer prompt carries evidence content + topic mapping ─

def test_reviewer_prompt_contains_the_same_evidence_catalog_and_topic_mapping():
    kb1 = KBChunk(content="the actual cited content", source="s1.pdf", page=1, box=1, evidence_id="KB-E001")
    state = _design_state(
        [_adr(evidence_ids=["KB-E001"])],
        [kb1],
        decision_topics=[DecisionTopic(id="TOPIC-1", topic="data ownership", query="q", evidence_ids=["KB-E001"])],
    )
    checks = run_deterministic_checks(state)

    prompt = rev._build_prompt(state, checks)

    assert "<decision_topics>" in prompt
    assert "<evidence_catalog>" in prompt
    assert "data ownership" in prompt
    assert "the actual cited content" in prompt
    # The ADR's own evidence_ids are visible too (inside architecture_artifacts),
    # so the Reviewer can connect a claim to its content.
    assert "KB-E001" in prompt


def test_reviewer_system_prompt_asks_about_semantic_relevance_and_decorative_citation():
    system = rev.REVIEWER_SYSTEM
    assert "evidence_catalog" in system or "decision_topics" in system
    assert "relevan" in system.lower()  # relevant/relevance
    assert "decorative" in system.lower() or "unrelated" in system.lower()


# ── no domain-specific hard-coding anywhere in the corrective pass ──────

def test_review_checks_kb_grounding_carries_no_domain_hardcoding():
    import pipeline.review_checks as rc

    source = inspect.getsource(rc._check_kb_evidence_grounding)
    forbidden = ["e-commerce", "ecommerce", "shopping cart", "checkout", "order service"]
    lowered = source.lower()
    for term in forbidden:
        assert term not in lowered
