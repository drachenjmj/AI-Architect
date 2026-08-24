"""researcher.py — Researcher agent (Kush).

DECISION-LEVEL RETRIEVAL, NOT ONE GLOBAL TOP-3
------------------------------------------------
The researcher used to issue a single Top-3 query built from the problem
statement and non-functional requirements, and every ADR the Architect wrote
afterwards was free to cite it — or nothing at all — with no way to tell
whether an ADR's claimed literature support was ever actually retrieved for
that decision. A strong model can write plausible ADRs from three chunks and
its own training knowledge; that is real reasoning, but it does not make the
design traceable to OUR curated knowledge base, which is the actual project
claim.

This module now plans a bounded set of CASE-DERIVED decision topics (4–8,
see `plan_decision_topics`), retrieves a small Top-K per topic through the
EXISTING `architect.retrieve_chunks` (same threshold, same logging, same web
fallback), deduplicates the results across topics, bounds the total, and
assigns each curated-KB chunk a stable per-run evidence ID. The Architect
prompt (pipeline/agents/architect.py) shows this evidence grouped by topic,
and the deterministic Reviewer (pipeline/review_checks.py) validates that
every ADR's cited evidence IDs actually resolve to something retrieved this
run — see `review_checks.qualifying_kb_evidence_ids`.

Deliberately NOT implemented: "retrieve everything under threshold and dump
it in the prompt". That would trade decision-traceability for prompt noise;
topic-scoped retrieval with a total cap is the bounded middle ground the
project spec asks for.

LangGraph node form — returns partial state updates.
"""
from __future__ import annotations

from pipeline.agents.base import make_step, node
from pipeline.state import ContextRecord, ArchitectState, DecisionTopic, KBChunk, RepoRepresentation, Stage

# Retrieved per topic. Kept at the project's existing Top-3 behaviour — the
# spec asks to preserve per-topic Top-K unless repository evidence strongly
# suggests otherwise, and nothing here does.
PER_TOPIC_K = 3

# Deterministic cap on the TOTAL unique evidence pool carried into the
# Architect prompt, after cross-topic dedup. Bounds token cost regardless of
# how many topics are planned; chosen in the middle of the spec's suggested
# 12–18 range. Evidence beyond the cap is simply not carried forward — the
# ordering (topic-first-seen, then per-topic distance) means the dropped
# entries are the weakest-ranked tail, not an arbitrary cut.
MAX_TOTAL_EVIDENCE = 15

# Bounded, generic decision-topic catalog. Always includes the four baseline
# topics (every architecture case has a decomposition, a data strategy, an
# integration style, and a scaling/availability posture worth grounding in
# literature); the rest are CASE-DERIVED from Context Record / repository
# signals so a greenfield toy request does not pay for a migration or
# compliance query it has no facts to ground. Bounded at 8 by construction
# (4 baseline + at most 4 conditional).
_BASELINE_TOPICS: tuple[str, ...] = (
    "service decomposition and boundaries",
    "data ownership and persistence strategy",
    "integration style: synchronous vs asynchronous communication",
    "scaling and availability strategy",
)

_OPERATIONS_KEYWORDS = (
    "monitor", "observab", "telemetry", "metric", "log", "trace",
    "alert", "on-call", "oncall", "incident", "slo", "sla",
)


def _has_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def _conditional_topics(
    context_record: ContextRecord | None,
    repo_representation: RepoRepresentation | None,
) -> list[str]:
    """Case-derived topics beyond the baseline four. Pure, no hard-coding of
    any domain — every branch reads a generic structural or textual signal
    already present on the Context Record / RepoRepresentation."""

    topics: list[str] = []

    existing_systems = bool(context_record and context_record.existing_systems)
    if existing_systems or repo_representation is not None:
        topics.append("brownfield migration and evolution strategy")

    if repo_representation is not None:
        stack = repo_representation.structure.tech_stack
        if stack.languages or stack.frameworks:
            topics.append("technology conservation vs replacement")

    if context_record and context_record.compliance_requirements:
        topics.append("security and compliance architecture")

    requirement_text = " ".join(
        (context_record.functional_requirements + context_record.non_functional_requirements)
        if context_record
        else []
    )
    if _has_keyword(requirement_text, _OPERATIONS_KEYWORDS):
        topics.append("observability and operations")

    return topics


def plan_decision_topics(
    context_record: ContextRecord | None,
    repo_representation: RepoRepresentation | None,
) -> list[str]:
    """The bounded (4–8), deduplicated, case-derived set of decision topics
    for this run. Pure.

    These are retrieval/planning aids, not final architecture decisions —
    the Architect still owns every design choice; this only decides WHAT to
    look up literature for.
    """

    topics: list[str] = list(_BASELINE_TOPICS)
    for topic in _conditional_topics(context_record, repo_representation):
        if topic not in topics:
            topics.append(topic)
    return topics


def _topic_query(topic: str, context_record: ContextRecord | None) -> str:
    """The retrieval query issued for one topic: the topic itself, grounded
    in the case's own problem/goal so retrieval is not a generic textbook
    lookup. Pure."""

    parts = [topic]
    if context_record is not None:
        if context_record.problem_statement.strip():
            parts.append(context_record.problem_statement.strip())
        if context_record.business_goal.strip():
            parts.append(context_record.business_goal.strip())
    return " ".join(parts).strip()


def _dedupe_key(chunk: dict) -> tuple:
    """Cross-topic dedup identity for one raw chunk dict. Pure.

    Exact-tuple identity (not fuzzy): the same underlying KB chunk resurfacing
    for two topics is the common case this exists to collapse, and a fuzzy
    match risks silently merging two genuinely different passages.
    """
    return (
        str(chunk.get("source", "")).strip().lower(),
        chunk.get("page", 0),
        chunk.get("box", 1),
        str(chunk.get("content", "")).strip(),
    )


@node("researcher")
def researcher_node(state: ArchitectState) -> dict:
    from architect import retrieve_chunks

    topics = plan_decision_topics(state.context_record, state.repo_representation)

    seen: dict[tuple, int] = {}  # dedupe key -> index into `pool`
    pool: list[dict] = []
    topic_chunk_indices: list[list[int]] = []
    gaps: list[str] = []

    for topic in topics:
        query = _topic_query(topic, state.context_record)
        chunks, _origin = retrieve_chunks(query, k=PER_TOPIC_K)
        indices: list[int] = []
        for chunk in chunks:
            key = _dedupe_key(chunk)
            if key in seen:
                indices.append(seen[key])
                continue
            index = len(pool)
            pool.append(chunk)
            seen[key] = index
            indices.append(index)
        topic_chunk_indices.append(indices)
        if not indices:
            gaps.append(topic)

    # Deterministic total bound: keep the first MAX_TOTAL_EVIDENCE unique
    # chunks in first-seen (topic order, then per-topic distance) order.
    bounded_pool = pool[:MAX_TOTAL_EVIDENCE]
    kept_indices = set(range(len(bounded_pool)))

    # Stable per-run evidence IDs — ONLY for curated-KB chunks (box != 3).
    # A web-fallback chunk keeps evidence_id="" so it can never be cited by
    # an ADR's evidence_ids and never satisfies the literature gate.
    retrieved_knowledge: list[KBChunk] = []
    evidence_id_by_index: dict[int, str] = {}
    counter = 0
    for index, chunk in enumerate(bounded_pool):
        evidence_id = ""
        if chunk.get("box", 1) != 3:
            counter += 1
            evidence_id = f"KB-E{counter:03d}"
            evidence_id_by_index[index] = evidence_id
        retrieved_knowledge.append(KBChunk(**chunk, evidence_id=evidence_id))

    decision_topics: list[DecisionTopic] = []
    for topic_number, (topic, indices) in enumerate(zip(topics, topic_chunk_indices), start=1):
        evidence_ids = [
            evidence_id_by_index[index]
            for index in indices
            if index in kept_indices and index in evidence_id_by_index
        ]
        decision_topics.append(
            DecisionTopic(
                id=f"TOPIC-{topic_number}",
                topic=topic,
                query=_topic_query(topic, state.context_record),
                evidence_ids=evidence_ids,
            )
        )

    kb_evidence_count = len(evidence_id_by_index)
    web_evidence_count = len(retrieved_knowledge) - kb_evidence_count
    note = (
        f"planned {len(topics)} decision topic(s); retrieved {kb_evidence_count} "
        f"curated-KB evidence item(s)"
    )
    if web_evidence_count:
        note += f" and {web_evidence_count} web-fallback item(s) (not literature-qualifying)"
    if gaps:
        note += f"; {len(gaps)} topic(s) with no qualifying evidence: {', '.join(gaps)}"

    step = make_step(
        "researcher",
        state.stage,
        Stage.RESEARCHING,
        note,
    )
    return {
        "retrieved_knowledge": retrieved_knowledge,
        "decision_topics": decision_topics,
        "stage": Stage.RESEARCHING,
        "history": [step],
    }
