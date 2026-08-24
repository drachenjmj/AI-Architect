"""test_decision_evidence.py — decision-level RAG grounding (Kush).

Covers the replacement of one global Top-3 retrieval with bounded,
case-derived decision-topic retrieval (pipeline/agents/researcher.py), the
stable per-run evidence-ID contract (KBChunk.evidence_id / ADR.evidence_ids),
the Architect's decision-evidence prompt block and output-boundary sanitizer
(pipeline/agents/architect.py), and the deterministic literature-grounding
gate (pipeline/review_checks.py).

Everything here is offline: `architect.retrieve_chunks` is replaced with a
recording fake — no Chroma index, no API key, no network.
"""

from __future__ import annotations

import architect
from pipeline.agents import architect as arch
from pipeline.agents import researcher as res
from pipeline.review_checks import (
    qualifying_kb_evidence_ids,
    run_deterministic_checks,
)
from pipeline.state import (
    ADR,
    Blueprint,
    ComponentDescription,
    ContextRecord,
    Feature,
    KBChunk,
    RepoRepresentation,
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
    catalog or its derivation logic."""
    import inspect

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


def test_total_evidence_is_bounded_and_deterministically_ordered(monkeypatch):
    state = _state(context_record=ContextRecord())
    topics = res.plan_decision_topics(state.context_record, None)

    by_query = {}
    for index, topic in enumerate(topics):
        query = res._topic_query(topic, state.context_record)
        # Each topic contributes MAX_TOTAL_EVIDENCE unique chunks so the
        # pool overflows the cap well before topics run out.
        chunks = [
            _chunk(f"content {index}-{i}", f"source-{index}-{i}.pdf")
            for i in range(res.MAX_TOTAL_EVIDENCE)
        ]
        by_query[query] = (chunks, "kb")
    fake = FakeRetrieval(by_query=by_query)
    monkeypatch.setattr(architect, "retrieve_chunks", fake)

    output = res.researcher_node(state)

    assert len(output["retrieved_knowledge"]) == res.MAX_TOTAL_EVIDENCE
    ids = [chunk.evidence_id for chunk in output["retrieved_knowledge"]]
    assert ids == sorted(ids)  # KB-E001, KB-E002, ... strictly increasing
    # First topic's chunks fill the pool first (deterministic, topic-order).
    assert output["retrieved_knowledge"][0].source == "source-0-0.pdf"


def test_evidence_ids_are_stable_for_the_same_evidence_set(monkeypatch):
    state = _state(context_record=ContextRecord())
    chunk = _chunk("stable content", "stable.pdf")
    fake = FakeRetrieval(default=([chunk], "kb"))
    monkeypatch.setattr(architect, "retrieve_chunks", fake)

    first = res.researcher_node(state)["retrieved_knowledge"][0].evidence_id
    second = res.researcher_node(state)["retrieved_knowledge"][0].evidence_id

    assert first == second == "KB-E001"


def test_web_fallback_chunks_get_no_evidence_id(monkeypatch):
    state = _state(context_record=ContextRecord())
    web_chunk = _chunk("web content", "https://example.com", box=3, distance=None)
    fake = FakeRetrieval(default=([web_chunk], "web"))
    monkeypatch.setattr(architect, "retrieve_chunks", fake)

    output = res.researcher_node(state)

    assert all(chunk.evidence_id == "" for chunk in output["retrieved_knowledge"])
    assert all(topic.evidence_ids == [] for topic in output["decision_topics"])


def test_evidence_gap_is_explicit_per_topic(monkeypatch):
    state = _state(context_record=ContextRecord())
    monkeypatch.setattr(architect, "retrieve_chunks", FakeRetrieval(default=([], "none")))

    output = res.researcher_node(state)

    assert all(topic.evidence_ids == [] for topic in output["decision_topics"])
    assert output["retrieved_knowledge"] == []
    assert "no qualifying evidence" in output["history"][0].note


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
        related_component_names=["Service"],
    )
    base.update(overrides)
    return ADR(**base)


def _design_state(adrs, retrieved_knowledge):
    state = new_run("Design a system.")
    state.context_record = ContextRecord(business_goal="g", problem_statement="p")
    state.features = [Feature(id="FEAT-001", name="F", scenario="s", acceptance_criteria=["a"])]
    state.blueprint = Blueprint(
        stakeholder_view="sv", technical_view="tv", components=["Service"],
        addressed_feature_ids=["FEAT-001"],
    )
    state.adrs = adrs
    state.components = [
        ComponentDescription(
            id="COMP-001", name="Service", purpose="p", description="d",
            related_feature_ids=["FEAT-001"], related_adr_ids=["ADR-001"],
        )
    ]
    state.retrieved_knowledge = retrieved_knowledge
    return state


def test_adr_with_valid_retrieved_kb_evidence_passes_the_gate():
    kb = KBChunk(content="c", source="s.pdf", page=1, box=1, evidence_id="KB-E001")
    state = _design_state([_adr(evidence_ids=["KB-E001"])], [kb])

    assert qualifying_kb_evidence_ids(state) == {"KB-E001"}
    checks = run_deterministic_checks(state)

    assert checks.score_kb_evidence_grounding == 2
    assert checks.adrs_without_kb_evidence == []
    assert checks.invalid_evidence_ids == {}


def test_adr_with_fabricated_evidence_id_fails_the_gate():
    kb = KBChunk(content="c", source="s.pdf", page=1, box=1, evidence_id="KB-E001")
    state = _design_state([_adr(evidence_ids=["KB-E999"])], [kb])

    checks = run_deterministic_checks(state)

    assert checks.score_kb_evidence_grounding == 0
    assert checks.invalid_evidence_ids == {"ADR-001": ["KB-E999"]}
    assert any(issue.category == "evidence" and issue.severity == "high" for issue in checks.issues)


def test_adr_with_no_evidence_fails_when_qualifying_evidence_exists():
    kb = KBChunk(content="c", source="s.pdf", page=1, box=1, evidence_id="KB-E001")
    state = _design_state([_adr(evidence_ids=[])], [kb])

    checks = run_deterministic_checks(state)

    assert checks.score_kb_evidence_grounding == 0
    assert checks.adrs_without_kb_evidence == ["ADR-001"]
    assert checks.kb_evidence_gap is False


def test_total_kb_gap_does_not_block_the_verdict():
    """No curated-KB evidence was retrieved for anything this run — an
    honest gap, not a design defect. Must not fail the run forever."""
    state = _design_state([_adr(evidence_ids=[])], [])

    checks = run_deterministic_checks(state)

    assert checks.kb_evidence_gap is True
    assert checks.score_kb_evidence_grounding == 2


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


# ── Architect prompt: grouped decision evidence, no duplicate dumping ───

def test_decision_evidence_block_groups_by_topic_without_duplicating_chunks():
    from pipeline.state import DecisionTopic

    state = new_run("Design a system.")
    kb1 = KBChunk(content="c1", source="s1.pdf", page=1, box=1, evidence_id="KB-E001")
    kb2 = KBChunk(content="c2", source="s2.pdf", page=2, box=1, evidence_id="KB-E002")
    state.retrieved_knowledge = [kb1, kb2]
    state.decision_topics = [
        DecisionTopic(id="TOPIC-1", topic="data ownership", query="q1", evidence_ids=["KB-E001"]),
        DecisionTopic(id="TOPIC-2", topic="scaling", query="q2", evidence_ids=["KB-E002"]),
    ]

    block = arch._decision_evidence_block(state)

    assert "data ownership" in block
    assert "scaling" in block
    assert block.count("KB-E001") == 1  # each chunk rendered exactly once
    assert block.count("KB-E002") == 1


def test_decision_evidence_block_states_the_gap_for_an_empty_topic():
    from pipeline.state import DecisionTopic

    state = new_run("Design a system.")
    state.decision_topics = [
        DecisionTopic(id="TOPIC-1", topic="security and compliance architecture", query="q", evidence_ids=[]),
    ]

    block = arch._decision_evidence_block(state)

    assert "no qualifying KB evidence" in block
