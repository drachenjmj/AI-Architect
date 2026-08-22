"""architecture_chat.py — the read-only, grounded Architecture Chat (Kati).

WHAT THIS IS
------------
The data + answer layer behind the workspace's Chat view: a global
architecture assistant for the CURRENT active run that may also answer
explicitly historical questions from the local read-only History layer.

READ-ONLY IS THE CONTRACT
-------------------------
One submitted question causes AT MOST one LLM answer call — nothing else.
The chat explains, compares and traces evidence; it never modifies an
artifact, submits feedback, consumes a refinement round, signs off,
switches runs or writes a checkpoint. A request to CHANGE the architecture
is answered with its implications and a pointer to the existing
Context/Architecture feedback workflow. Where a deterministic rule can
decide something (context selection, historical intent, ambiguity), no
model decides it.

REUSE, NOT REINVENTION
----------------------
- Answers go through the project's ONE LLM door, `pipeline.llm.llm_call`
  (called through the module so tests can patch it), on the default model
  — no second Gemini client, no new model constant.
- Historical runs come from `run_history` (the same read-only History the
  History view uses) — no second history parser, no mutation.
- Knowledge evidence is limited to the KB chunks ALREADY SAVED on the
  runs. Root `architect.retrieve_chunks` was inspected and is NOT reused
  on purpose: it can trigger the Box-3 live web fallback and mutates the
  RAG query log, which is pipeline behavior this chat must not cause.
  (Reported limitation; broader ad-hoc KB retrieval can come later.)

SESSION OWNERSHIP
-----------------
Chat messages live ONLY in Streamlit session state, keyed by run id (see
`messages_for` / `submit_question`); nothing here is ever written to
`.cache/runs/`, checkpoints, Chroma or the pipeline state object.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import streamlit as st

import run_history
from pipeline import llm
from pipeline.state import ArchitectState

# ── the chat's single source of truth for one answer's evidence ───────────


@dataclass(frozen=True)
class ChatSource:
    """One selectable unit of evidence: an artifact slice or a KB chunk.

    `sid` is the compact stable id the answer cites inline ([C1], [H1],
    [K1]); `scope` distinguishes current-run facts from historical-run
    facts, which the assistant must keep apart in its answers.
    """

    sid: str            # "C1", "C2", … "H1" … "K1" …
    scope: str          # "current" | "history" | "kb"
    kind: str           # run | context_record | feature | blueprint | adr | component | review | repository | kb | history_summary
    label: str          # human-readable, e.g. "Current · ADR-002"
    text: str           # the verbatim artifact slice
    run_id: str = ""
    run_date: str = ""  # ISO date, historical sources only
    box: int = 0        # KB chunks only
    distance: float | None = None  # KB chunks only


# ── caps ──────────────────────────────────────────────────────────────────
_MAX_DETAIL_SOURCES = 8     # detailed sources per answer, current + history
_MAX_HISTORY_RUNS = 2       # historical runs fully loaded per answer
_MAX_KB_SOURCES = 3         # saved KB chunks per answer

# Words too generic to score relevance by their presence.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "what", "which", "who", "why",
    "how", "when", "where", "does", "do", "did", "this", "that", "these",
    "those", "it", "its", "we", "our", "you", "your", "i", "me", "my",
    "about", "into", "from", "at", "by", "as", "than", "then", "so", "if",
    "not", "no", "can", "could", "would", "should", "will", "shall", "may",
    "might", "must", "have", "has", "had", "there", "their", "them", "they",
    "please", "tell", "explain", "give", "show", "list", "compare", "me",
    "us", "architecture", "run", "chat",
})

# Artifact identifiers a question can name exactly: FEAT-005, ADR-2, COMP-001.
_ID_RE = re.compile(r"\b([A-Za-z]{2,10})-(\d{1,4})\b")

# Historical/comparison intent. Deterministic — no LLM router.
_HISTORY_INTENT_RE = re.compile(
    r"\b(?:previous|earlier|last|prior|recent)\s+run"
    r"|\bhistory\b|\bhistorical\b"
    r"|\bcompare[d]?\b|\bdifference[s]?\b|\bdiffer(?:s|ed|ent)?\b"
    r"|\byesterday\b|\bmonday\b|\btuesday\b|\bwednesday\b|\bthursday\b"
    r"|\bfriday\b|\bsaturday\b|\bsunday\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{8}T\d{6}Z-[0-9a-f]+\b",
    re.IGNORECASE,
)
_RUN_ID_RE = re.compile(r"\b(\d{8}T\d{6}Z-[0-9a-f]+)\b", re.IGNORECASE)
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday")


def has_historical_intent(question: str) -> bool:
    """Deterministic: does this question explicitly ask about past runs?"""
    return bool(_HISTORY_INTENT_RE.search(question))


# ── current-run source units (built from EXISTING saved state only) ───────


def _clip(text: str, limit: int) -> str:
    """Verbatim but capped — a source unit is a slice, not the whole field."""
    body = text.strip()
    return body if len(body) <= limit else body[: limit - 1].rstrip() + "…"


def _run_identity_text(state: ArchitectState) -> str:
    """The small always-included summary: what this run is and produced."""
    record = state.context_record
    project = (
        (record.project_name if record is not None else "")
        or (state.blueprint.project_name if state.blueprint is not None else "")
        or "Architecture run"
    ).strip() or "Architecture run"
    parts = [
        f"Project: {project}",
        f"Run: {state.run_id} (stage: {state.stage.value})",
    ]
    if record is not None and record.business_goal.strip():
        parts.append(f"Business goal: {record.business_goal.strip()}")
    if record is not None and record.problem_statement.strip():
        parts.append(f"Problem: {record.problem_statement.strip()}")
    if state.blueprint is not None and state.blueprint.selected_pattern.strip():
        parts.append(f"Selected pattern: {state.blueprint.selected_pattern.strip()}")
    if state.review is not None:
        parts.append(f"Review verdict: {state.review.overall_status}")
    return "\n".join(parts)


def _kb_label(source: str, page: int) -> str:
    return f"KB · {source}" + (f", p. {page}" if page else "")


def build_current_run_sources(state: ArchitectState) -> list[ChatSource]:
    """Every current-run artifact as candidate source units, in a stable
    order. No selection happens here — that is `select_sources`."""
    sources: list[ChatSource] = []
    sid = 0

    def next_id(prefix: str) -> str:
        nonlocal sid
        sid += 1
        return f"{prefix}{sid}"

    sources.append(
        ChatSource("C0", "current", "run", "Current · Run summary",
                   _run_identity_text(state), run_id=state.run_id)
    )

    record = state.context_record
    if record is not None:
        lines = [
            f"Business goal: {record.business_goal}",
            f"Problem: {record.problem_statement}",
            f"Users: {', '.join(record.users)}",
            f"Functional requirements: {'; '.join(record.functional_requirements)}",
            f"Non-functional requirements: {'; '.join(record.non_functional_requirements)}",
            f"Cloud: {record.cloud_provider}; Budget: {record.budget}",
            f"Compliance: {'; '.join(record.compliance_requirements)}",
            f"Existing systems: {'; '.join(record.existing_systems)}",
            f"Assumptions: {'; '.join(record.assumptions)}",
        ]
        sources.append(
            ChatSource(next_id("C"), "current", "context_record",
                       "Current · Context Record", "\n".join(lines),
                       run_id=state.run_id)
        )

    for feature in state.features:
        text = (
            f"{feature.id}: {feature.name} ({feature.priority})\n"
            f"{feature.description}\nScenario: {feature.scenario}"
        )
        sources.append(
            ChatSource(next_id("C"), "current", "feature",
                       f"Current · Feature {feature.id}", text,
                       run_id=state.run_id)
        )

    blueprint = state.blueprint
    if blueprint is not None:
        text = "\n".join(
            [
                f"Pattern: {blueprint.selected_pattern}",
                f"Rationale: {blueprint.rationale}",
                f"Stakeholder view: {blueprint.stakeholder_view}",
                f"Technical view: {blueprint.technical_view}",
                "Data flows:\n- " + "\n- ".join(blueprint.data_flows),
                f"Open risks: {'; '.join(blueprint.open_risks)}",
                f"Constraints addressed: {'; '.join(blueprint.constraints_addressed)}",
            ]
        )
        sources.append(
            ChatSource(next_id("C"), "current", "blueprint",
                       "Current · Blueprint", text, run_id=state.run_id)
        )

    for adr in state.adrs:
        text = "\n".join(
            [
                f"{adr.id}: {adr.title} ({adr.status})",
                f"Context: {adr.context}",
                f"Decision: {adr.decision}",
                f"Rationale: {adr.rationale}",
                f"Alternatives: {'; '.join(adr.alternatives_considered)}",
                f"Positive consequences: {'; '.join(adr.positive_consequences)}",
                f"Negative consequences: {'; '.join(adr.negative_consequences)}",
            ]
        )
        sources.append(
            ChatSource(next_id("C"), "current", "adr",
                       f"Current · {adr.id}", text, run_id=state.run_id)
        )

    for component in state.components:
        text = "\n".join(
            [
                f"{component.name} [{component.component_type}] ({component.id})",
                f"Purpose: {component.purpose}",
                f"Description: {component.description}",
                f"Dependencies: {'; '.join(component.dependencies)}",
                f"Inputs: {'; '.join(component.inputs)}",
                f"Outputs: {'; '.join(component.outputs)}",
                f"Related features: {'; '.join(component.related_feature_ids)}; "
                f"ADRs: {'; '.join(component.related_adr_ids)}",
            ]
        )
        sources.append(
            ChatSource(next_id("C"), "current", "component",
                       f"Current · Component · {component.name}", text,
                       run_id=state.run_id)
        )

    if state.review is not None:
        review = state.review
        findings = "\n".join(
            f"- [{issue.severity}] {issue.finding}" for issue in review.issues
        ) or "none"
        text = (
            f"Verdict: {review.overall_status}\n"
            f"Blocking issues:\n{findings}\n"
            f"Refinement instruction: {review.refinement_instruction}"
        )
        sources.append(
            ChatSource(next_id("C"), "current", "review",
                       "Current · Review", text, run_id=state.run_id)
        )

    repo = state.repo_representation
    if repo is not None:
        stack = repo.structure.tech_stack
        partitions = "; ".join(
            f"{part.name}: {part.role}" for part in repo.behavior.partitions
        )
        text = (
            f"Repository: {repo.meta.url}\n"
            f"Overview: {repo.behavior.overview}\n"
            f"Languages: {', '.join(f'{k} {v:,}' for k, v in sorted(stack.languages.items(), key=lambda kv: -kv[1]))}\n"
            f"Frameworks: {', '.join(stack.frameworks)}\n"
            f"Partitions: {partitions}"
        )
        sources.append(
            ChatSource(next_id("C"), "current", "repository",
                       "Current · Repository", text, run_id=state.run_id)
        )

    for chunk in state.retrieved_knowledge:
        sources.append(
            ChatSource(next_id("K"), "kb", "kb",
                       _kb_label(chunk.source, chunk.page),
                       _clip(chunk.content, 900), run_id=state.run_id,
                       box=chunk.box, distance=chunk.distance)
        )

    return sources


# ── deterministic relevance selection ─────────────────────────────────────


def _question_tokens(question: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9-]+", question.lower())
        if token not in _STOPWORDS and len(token) > 2
    ]


def _score_source(question: str, tokens: list[str], source: ChatSource) -> int:
    """Priority: exact artifact-id match >> component/artifact name match >>
    query-term overlap. Deterministic; no reranker, no model."""
    score = 0
    haystack = (source.label + "\n" + source.text).lower()

    # Exact artifact ids named in the question (ADR-002, FEAT-005, …).
    for prefix, number in _ID_RE.findall(question):
        wanted = f"{prefix.upper()}-{int(number):03d}"
        wanted_alt = f"{prefix.upper()}-{number}"
        if wanted in source.label.upper() or wanted_alt in source.label.upper():
            score += 100
            break

    # Multi-word artifact names (component names) named in the question.
    name = source.label.rsplit("· ", 1)[-1].strip().lower()
    if name and len(name) > 3 and name in question.lower():
        score += 80

    # Query-term overlap with the source text.
    if tokens:
        hits = sum(1 for token in tokens if token in haystack)
        score += int(1000 * hits / len(tokens))  # 0–1000 scaled overlap
    return score


def select_sources(
    question: str, sources: list[ChatSource]
) -> list[ChatSource]:
    """The always-included run summary plus the top relevance-scored detail
    sources, capped. Deterministic and stable: equal scores keep build
    order, and nothing is inferred."""
    tokens = _question_tokens(question)
    scored = [
        (_score_source(question, tokens, source), index, source)
        for index, source in enumerate(sources)
        if source.kind != "run"
    ]
    scored.sort(key=lambda entry: (-entry[0], entry[1]))

    selected: list[ChatSource] = [sources[0]]  # the run summary, always
    kb_used = 0
    for score, _index, source in scored:
        if len(selected) >= _MAX_DETAIL_SOURCES + 1:
            break
        if score <= 0:
            break  # nothing else is relevant; do not pad
        if source.scope == "kb":
            if kb_used >= _MAX_KB_SOURCES:
                continue
            kb_used += 1
        selected.append(source)
    return selected


# ── historical candidates (through the existing History layer) ────────────


def _summary_date(summary: run_history.HistorySummary) -> str:
    try:
        return datetime.fromisoformat(summary.updated_at).strftime("%Y-%m-%d")
    except ValueError:
        return summary.updated_at[:10]


def _resolve_history_candidates(
    question: str,
    current: ArchitectState,
    summaries: list[run_history.HistorySummary],
) -> tuple[list[run_history.HistorySummary], str]:
    """Deterministically pick which historical runs the answer may use.

    Returns (chosen, note). `note` is non-empty when the question cannot be
    answered from history as asked — no identifiable run, or several equally
    plausible ones — and becomes the answer WITHOUT any LLM call: silently
    comparing against an arbitrary run is the failure this exists to
    prevent.
    """
    others = [s for s in summaries if s.run_id != current.run_id]
    if not others:
        return [], (
            "There are no other saved runs on this machine to compare "
            "against. `.cache/runs/` is local, so only runs executed here "
            "are available."
        )

    # 1. An explicit run id always wins.
    run_id_match = _RUN_ID_RE.search(question)
    if run_id_match:
        wanted = run_id_match.group(1).lower()
        exact = [s for s in others if s.run_id.lower() == wanted]
        if exact:
            return exact[:_MAX_HISTORY_RUNS], ""
        return [], (
            f"No saved run with id '{run_id_match.group(1)}' exists on "
            f"this machine."
        )

    # 2. "previous/last/prior/earlier run" — the newest other run.
    if re.search(
        r"\b(?:previous|last|prior|earlier|recent)\s+run\b", question, re.I
    ):
        return others[:1], ""  # others are newest-first from list_history_runs

    # 3. An explicit ISO date.
    date_match = _DATE_RE.search(question)
    if date_match:
        wanted = date_match.group(1)
        matches = [s for s in others if _summary_date(s) == wanted]
        if len(matches) == 1:
            return matches[:_MAX_HISTORY_RUNS], ""
        if not matches:
            return [], f"No saved run is dated {wanted} on this machine."
        # several runs the same day — fall through to the label list below

    # 4. A weekday name — resolve against each summary's date.
    weekday_hits = [
        word for word in question.lower().split() if word.strip(",.?!") in _WEEKDAYS
    ]
    if weekday_hits:
        wanted_index = _WEEKDAYS.index(weekday_hits[0])
        matches = [
            s for s in others
            if _try_weekday(s) == wanted_index
        ]
        if len(matches) == 1:
            return matches[:_MAX_HISTORY_RUNS], ""
        if not matches:
            return [], (
                f"No saved run on this machine is from the {weekday_hits[0]} "
                f"you named."
            )

    # 5. A project/repo label named in the question.
    label_matches = [
        s for s in others
        if s.project_label and s.project_label.lower() in question.lower()
    ] or [
        s for s in others
        if s.repo_name and s.repo_name.lower() in question.lower()
    ]
    if len(label_matches) == 1:
        return label_matches[:_MAX_HISTORY_RUNS], ""

    # 6. Generic historical/compare intent with no identifiable target.
    if len(others) == 1:
        return others[:1], ""
    labels = "\n".join(
        f"- {_summary_date(s)} · {s.project_label} · {s.repo_name or 'no repo'} "
        f"· run {s.run_id}"
        for s in others[:5]
    )
    return [], (
        "Several saved runs could match that. Which one do you mean?\n"
        f"{labels}\nName the run id, its date, or its project to pick one."
    )


def _try_weekday(summary: run_history.HistorySummary) -> int | None:
    try:
        return datetime.fromisoformat(summary.updated_at).weekday()
    except ValueError:
        return None


def build_history_sources(
    chosen: list[run_history.HistorySummary],
) -> list[ChatSource]:
    """One compact source per chosen historical run, from the run's own
    saved artifacts (loaded read-only through `run_history`)."""
    sources: list[ChatSource] = []
    for index, summary in enumerate(chosen, start=1):
        try:
            state = run_history.load_history_run(summary.run_id)
        except run_history.HistoryError as exc:
            sources.append(
                ChatSource(f"H{index}", "history", "history_summary",
                           f"History · {summary.run_id} (unreadable)",
                           f"This saved run could not be read back: {exc}",
                           run_id=summary.run_id)
            )
            continue
        date = _summary_date(summary)
        record = state.context_record
        project = summary.project_label
        parts = [
            f"Run {state.run_id} · {date} · project: {project}",
            f"Stage: {state.stage.value}; accepted: {bool(state.accepted_at)}",
        ]
        if state.blueprint is not None:
            parts.append(f"Pattern: {state.blueprint.selected_pattern}")
            parts.append(
                f"Components: {', '.join(c.name for c in state.components)}"
                if state.components else "Components: none"
            )
            parts.append(
                "ADR decisions:\n- "
                + "\n- ".join(f"{a.id}: {a.decision}" for a in state.adrs)
                if state.adrs else "ADR decisions: none"
            )
            parts.append(
                f"Risks: {'; '.join(state.blueprint.open_risks)}"
                if state.blueprint.open_risks else "Risks: none recorded"
            )
        if state.review is not None:
            parts.append(f"Review verdict: {state.review.overall_status}")
            if state.review.issues:
                parts.append(
                    "Open findings: "
                    + "; ".join(i.finding for i in state.review.issues)
                )
        if record is not None:
            parts.append(
                f"Constraints: cloud {record.cloud_provider or '—'}, "
                f"budget {record.budget or '—'}, "
                f"compliance {'; '.join(record.compliance_requirements) or '—'}"
            )
        sources.append(
            ChatSource(
                f"H{index}", "history", "history_summary",
                f"History · {date} · {project}", "\n".join(parts),
                run_id=summary.run_id, run_date=date,
            )
        )
    return sources


# ── the grounded answer ───────────────────────────────────────────────────

_CHAT_SYSTEM = (
    "You are the Architecture Chat assistant for an architecture run. You "
    "answer ONLY from the labeled sources supplied in the prompt.\n"
    "Rules:\n"
    "1. Cite sources inline with their bracket ids, e.g. [C1], [H1], [K1].\n"
    "2. Keep current-run facts ([C…], [K…]) strictly apart from historical-"
    "run facts ([H…]) and say which run a fact comes from when it matters.\n"
    "3. Do not claim anything the sources do not state. If the available "
    "artifacts do not establish the answer, say exactly that.\n"
    "4. You are READ-ONLY. You never changed, and cannot change, any "
    "artifact. If the user asks to change something (a feature, component, "
    "ADR, the blueprint), explain what the change would affect based on the "
    "sources, then direct them to the existing feedback workflow: the "
    "requirements box in the Context view, or the change box in the "
    "Architecture view. Never claim a change was made.\n"
    "5. Be concise: a few short paragraphs at most."
)


def _build_prompt(
    question: str,
    current: ArchitectState,
    sources: list[ChatSource],
) -> str:
    blocks = []
    for source in sources:
        blocks.append(f"[{source.sid}] {source.label}\n{source.text}")
    return (
        f"Question: {question}\n\n"
        f"--- Labeled sources (the ONLY evidence you may use) ---\n"
        + "\n\n".join(blocks)
        + "\n--- end of sources ---\n"
        "Answer the question using only these sources, citing the bracket "
        "ids inline."
    )


def answer_chat_question(
    question: str, current: ArchitectState
) -> tuple[str, list[ChatSource]]:
    """One grounded answer for one submitted question.

    Returns (answer, sources_used). At most ONE `llm_call` is made, and
    NONE when a deterministic rule already settles the answer (no local
    history, unidentifiable/ambiguous historical run). Raises nothing for
    LLM failures — the caller decides how to surface an error.

    Historical sources are appended AFTER selection, not scored: the
    deterministic intent rules already chose them (bounded to
    _MAX_HISTORY_RUNS), so term overlap must never drop the very runs the
    user explicitly asked about.
    """
    sources = build_current_run_sources(current)
    history_sources: list[ChatSource] = []

    if has_historical_intent(question):
        summaries = run_history.list_history_runs()
        chosen, note = _resolve_history_candidates(question, current, summaries)
        if note:
            return note, []
        history_sources = build_history_sources(chosen)

    selected = select_sources(question, sources) + history_sources

    prompt = _build_prompt(question, current, selected)
    answer, _usage = llm.llm_call(current, prompt, system=_CHAT_SYSTEM)
    answer = str(answer or "").strip()
    if not answer:
        answer = (
            "The model returned an empty answer. The available artifacts do "
            "not establish an answer — try rephrasing or naming an artifact "
            "id (e.g. ADR-001)."
        )
    return answer, selected


# ── session state (Streamlit only; never persisted anywhere else) ─────────

_CHAT_SESSION_KEY = "architecture_chat"


def messages_for(run_id: str) -> list[dict[str, Any]]:
    """This run's in-session chat transcript. Kept per run id so switching
    the current run can never mix conversations."""
    store = st.session_state.setdefault(_CHAT_SESSION_KEY, {})
    return store.setdefault(run_id, [])


def clear_chat(run_id: str) -> None:
    """Drop ONLY the active run's transcript."""
    store = st.session_state.get(_CHAT_SESSION_KEY) or {}
    store.pop(run_id, None)


def submit_question(current: ArchitectState, question: str) -> None:
    """REACT half: append the user's message, produce one grounded answer
    (or a graceful error), append it with its sources. Never raises into
    the page; a failure leaves the user's message and all prior messages
    intact."""
    question = question.strip()
    if not question:
        return
    messages = messages_for(current.run_id)
    messages.append({"role": "user", "content": question})
    try:
        answer, sources = answer_chat_question(question, current)
        messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": [
                    {
                        "sid": s.sid,
                        "scope": s.scope,
                        "kind": s.kind,
                        "label": s.label,
                        "run_id": s.run_id,
                        "run_date": s.run_date,
                        "box": s.box,
                        "distance": s.distance,
                    }
                    for s in sources
                ],
            }
        )
    except llm.LLMError as exc:
        messages.append(
            {
                "role": "assistant",
                "content": (
                    f"The chat could not reach the model ({exc}). Your "
                    f"question is kept above — try asking again."
                ),
                "sources": [],
            }
        )
