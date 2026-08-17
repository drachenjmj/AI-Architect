"""ui_sections.py — the finished-run output surface, drawn from state (Kati).

WHY THIS FILE EXISTS
--------------------
`ui.py` is the event/DRAW spine: it owns the session, the forms, and the
routing between screens. This module owns only the *output surface* — the
section renderers the DONE branch calls. Splitting them keeps the spine
readable while the finished-run screen grows to show everything a run did.

THE ONE RULE
------------
Every function here is DRAW-only. It takes the `ArchitectState`, reads it, and
emits Streamlit widgets. Nothing here computes a new fact, calls an LLM,
mutates the state, or reaches for a second source of truth — so re-running any
of them against the same state produces the same screen. That is the same
property that lets a checkpoint reload replay a whole run for free.

WHAT IT MAKES VISIBLE (and why that is the point)
-------------------------------------------------
The pipeline demonstrates five agentic behaviours, and a behaviour nobody can
see scores nothing. So each section is labelled with the behaviour it evidences:

    Run trace            -> multi-step reasoning
    Knowledge retrieved  -> retrieval
    Repository analysis  -> tool use
    Review report        -> iteration + quality gate
    (the Q&A replay, which lives in ui.py -> clarification)

Everything below that is the design itself, rendered in full: the point of the
detail sections is that the artifacts are already rich, and only the UI was
thin.

COST HONESTY
------------
Every cost here is the LIST-PRICE EQUIVALENT at Google's published Gemini
prices. This project runs on free-tier keys, so it is NOT money spent, and
every place a cost appears says so.

`StepLog.cost_usd` is a plain `float`, so a model with no verified list price
lands there as 0.0 — indistinguishable from a step that genuinely cost nothing.
`_step_cost_is_known` re-derives that distinction from the model ID the step
recorded, and this module prints "unknown" rather than a confident "$0.0000".
That is a rendering-side guard only; the collapse itself happens in
`pipeline/agents/base.py`, which is pipeline code and deliberately untouched.
"""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Sequence

import streamlit as st
import streamlit.components.v1 as components

from pipeline.llm import PRICING_USD_PER_MTOK
from pipeline.persistence import runs_dir
from pipeline.refine_gate import MAX_REFINE_ITERATIONS
from pipeline.state import ADR, ArchitectState, ComponentDescription, Feature, StepLog

# ── BCG brand palette (widget theme lives in .streamlit/config.toml) ──────
# Defined here rather than in ui.py so the spine and the sections share one
# source; ui.py imports these back under its original private names.
BCG_GREEN = "#29BA74"  # signature green — the active step
BCG_DARK = "#147B58"   # dark BCG green — headings, completed steps
GREY = "#ADB5B1"       # pending steps, secondary text
RED = "#C0392B"        # failure

# `KBChunk.box` is an int on the wire; this is what each box actually is.
_BOX_LABELS: dict[int, str] = {
    1: "curated architecture patterns",
    2: "domain knowledge",
    3: "live web-search fallback",
}

# The standing disclaimer. Every cost on screen carries this, in some form.
_PRICE_NOTE = (
    "List-price equivalent at Google's published Gemini prices. We run on "
    "free-tier keys, so this is what the run WOULD have cost — not money spent."
)


# ══════════════════════════════════════════════════════════════════════════
# Cost honesty helpers
# ══════════════════════════════════════════════════════════════════════════
def _step_cost_is_known(step: StepLog) -> bool:
    """Can this step's `cost_usd` be trusted as a real number?

    `LLMUsage.cost_usd` is `float | None` (None = no verified price for that
    model) but `StepLog.cost_usd` is a plain float, so None is written as 0.0.
    A step that called an unpriced model is therefore stored exactly like a step
    that made no call at all. We recover the difference from the one piece of
    evidence the step still carries — the model ID(s) it recorded:

      * no model and no tokens  -> a code-only node (refine_gate, researcher, a
                                   greenfield repo_ingestor). 0.0 is the truth.
      * no model but tokens     -> spend we cannot attribute to a price. Unknown.
      * model(s) recorded       -> known only if every one of them is priced.

    `sum_usage` joins several model IDs with ", ", so a node that called two
    models is checked model by model.
    """
    if not step.model:
        return step.input_tokens == 0 and step.output_tokens == 0
    return all(
        PRICING_USD_PER_MTOK.get(name.strip()) is not None
        for name in step.model.split(",")
        if name.strip()
    )


def _cost_label(cost_usd: float, known: bool) -> str:
    """Format one cost, or say plainly that we cannot price it."""
    return f"${cost_usd:,.4f}" if known else "unknown"


def _run_cost_is_known(state: ArchitectState) -> bool:
    """True when every step in the run has a verifiable price."""
    return all(_step_cost_is_known(step) for step in state.history)


def _agent_cost_is_known(state: ArchitectState, agent: str) -> bool:
    """True when every step BY THIS AGENT has a verifiable price."""
    return all(
        _step_cost_is_known(step) for step in state.history if step.agent == agent
    )


def run_cost_label(state: ArchitectState) -> str:
    """Whole-run cost, or "unknown" if any step used an unpriced model."""
    return _cost_label(state.total_cost_usd(), _run_cost_is_known(state))


def live_step_caption(snapshot: ArchitectState) -> str:
    """The running-total line `_run` prints under each live step.

    `input_tokens` / `output_tokens` are reducer fields, so every streamed
    snapshot already carries the run total so far — no accumulation needed on
    the UI side. This is what makes the cost guardrail visible as a behaviour:
    the viewer watches the number climb through the refine loop.
    """
    total = snapshot.input_tokens + snapshot.output_tokens
    return (
        f"{total:,} tokens so far "
        f"({snapshot.input_tokens:,} in / {snapshot.output_tokens:,} out) "
        f"· ≈ {run_cost_label(snapshot)} at list prices (free-tier key)"
    )


# ══════════════════════════════════════════════════════════════════════════
# Small field helpers — each returns True when it actually drew something,
# so a caller can tell "artifact present but entirely empty" from "populated".
# ══════════════════════════════════════════════════════════════════════════
def _text(label: str, value: str | None) -> bool:
    """One prose field. Silent when empty."""
    body = str(value or "").strip()
    if not body:
        return False
    st.markdown(f"**{label}**")
    st.markdown(body)
    return True


def _bullets(label: str, values: Sequence[str] | None) -> bool:
    """One list field, as bullets. Silent when empty."""
    items = [str(v).strip() for v in (values or []) if str(v).strip()]
    if not items:
        return False
    st.markdown(f"**{label}**")
    st.markdown("\n".join(f"- {item}" for item in items))
    return True


def _chips(label: str, values: Sequence[str] | None) -> bool:
    """One list of short IDs, inline as code chips. Silent when empty."""
    items = [str(v).strip() for v in (values or []) if str(v).strip()]
    if not items:
        return False
    st.markdown(f"**{label}** " + " ".join(f"`{item}`" for item in items))
    return True


def _missing(label: str, why: str) -> None:
    """State that a section is absent, and why. Never an empty heading."""
    st.caption(f"**{label}** — {why}")


def _clock(timestamp: str) -> str:
    """HH:MM:SS from an ISO timestamp; the raw value if it will not parse."""
    try:
        return datetime.fromisoformat(timestamp).strftime("%H:%M:%S")
    except (TypeError, ValueError):
        return str(timestamp or "")


def _tag(text: str, color: str) -> str:
    """A small coloured inline label (HTML, so callers pass unsafe_allow_html)."""
    return (
        f"<span style='color:{color}; font-weight:600'>{html.escape(str(text))}</span>"
    )


# ══════════════════════════════════════════════════════════════════════════
# A. Honest run status
# ══════════════════════════════════════════════════════════════════════════
def render_run_status(state: ArchitectState) -> None:
    """The headline verdict — which must never overstate what happened.

    The old screen showed an unconditional green "Design complete." even when
    the review had failed and the run had stopped on the refine budget. An
    honest "stopped on budget, N open issues" is evidence the cost guardrail
    works; a false success is just wrong.
    """
    review = state.review
    open_issues = len(review.issues) if review else 0
    issue_text = f"{open_issues} open issue{'s' if open_issues != 1 else ''}"

    if review is None:
        st.info(
            "**Design produced — not reviewed.** The run finished without a "
            "review result, so nothing here has passed the quality gate."
        )
        return

    if state.stopped_on_cap:
        st.warning(
            f"**Best-effort design — stopped on the refine budget.** "
            f"The architect re-designed {state.refine_iterations} time"
            f"{'s' if state.refine_iterations != 1 else ''} "
            f"(cap: {MAX_REFINE_ITERATIONS}), and the reviewer still reports "
            f"**{issue_text}**. The run ended on the cost guardrail, not on a "
            f"clean pass — the artifacts below are the best version reached "
            f"within budget."
        )
        return

    if review.overall_status != "pass":
        st.warning(
            f"**Design produced — review did not pass.** The reviewer reports "
            f"**{issue_text}** after {state.refine_iterations} refine "
            f"iteration{'s' if state.refine_iterations != 1 else ''}. "
            f"See the review report below."
        )
        return

    st.success(
        f"**Design complete — review passed.**"
        + (
            f" Reached after {state.refine_iterations} refine "
            f"iteration{'s' if state.refine_iterations != 1 else ''}."
            if state.refine_iterations
            else " Passed on the first pass."
        )
    )


# ══════════════════════════════════════════════════════════════════════════
# B. Status strip
# ══════════════════════════════════════════════════════════════════════════
def _checkpoint_dir(state: ArchitectState) -> str:
    """Where this run's checkpoints live, written the shortest honest way.

    `run_id` is only useful if you can find what it keys, so it is shown with
    its directory rather than alone. The directory is read from
    `persistence.runs_dir()` — never hardcoded — because
    `AI_ARCHITECT_RUNS_DIR` can move it, and a path printed on screen that is
    not the real one would be worse than printing none. Shown relative to the
    working directory when it sits underneath it, which is the normal case and
    keeps the line short enough for a screen recording.
    """
    try:
        location = Path(runs_dir()) / state.run_id
        cwd = Path.cwd()
        # `cwd.parent == cwd` means the cwd IS a filesystem root, where the
        # relative form would silently drop the drive letter and read as an
        # absolute path that is not one. Only shorten against a real directory.
        if cwd.parent != cwd:
            try:
                location = location.relative_to(cwd)
            except ValueError:
                pass  # runs dir lives outside the cwd — show it in full
        return location.as_posix()
    except Exception:  # noqa: BLE001 — a caption is never worth a crash
        return ""


def render_status_strip(state: ArchitectState) -> None:
    """One metrics row: the whole run's shape at a glance."""
    review = state.review
    if review is None:
        verdict = "none"
    else:
        verdict = "PASS" if review.overall_status == "pass" else "FAIL"

    total_tokens = state.input_tokens + state.output_tokens
    chunks = len(state.retrieved_knowledge)

    cols = st.columns(6)
    with cols[0]:
        st.metric(
            "Review",
            verdict,
            border=True,
            help="The reviewer's verdict. Code owns this decision, not the LLM.",
        )
    with cols[1]:
        st.metric(
            "Refine loops",
            f"{state.refine_iterations} / {MAX_REFINE_ITERATIONS}",
            border=True,
            help="Architect re-designs triggered by a failing review, against "
                 "the cap in pipeline/refine_gate.py.",
        )
    with cols[2]:
        st.metric(
            "Tokens",
            f"{total_tokens:,}",
            border=True,
            help=f"{state.input_tokens:,} in / {state.output_tokens:,} out, "
                 "summed over every LLM call in the run.",
        )
    with cols[3]:
        st.metric(
            "Cost",
            run_cost_label(state),
            border=True,
            help=_PRICE_NOTE,
        )
    with cols[4]:
        st.metric(
            "Feat / ADR / Comp",
            f"{len(state.features)} / {len(state.adrs)} / {len(state.components)}",
            border=True,
            help="Features derived, Architecture Decision Records written, "
                 "components described.",
        )
    with cols[5]:
        st.metric(
            "KB chunks",
            chunks,
            border=True,
            help="Knowledge-base passages retrieved and given to the architect.",
        )

    st.caption(f"💡 {_PRICE_NOTE}")

    # The run's identity, and what it keys. Every transition of this run was
    # checkpointed under this id, so it is the handle for resuming it from the
    # picker and the anchor for auditing it on disk after the fact.
    location = _checkpoint_dir(state)
    st.caption(
        f"Run `{state.run_id}` · checkpointed after every step"
        + (f" to `{location}/`" if location else "")
    )


# ══════════════════════════════════════════════════════════════════════════
# D. Run trace  [multi-step reasoning]
# ══════════════════════════════════════════════════════════════════════════
def render_run_trace(state: ArchitectState) -> None:
    """The whole trace as a readable timeline, plus per-agent token totals."""
    steps = state.history
    if not steps:
        _missing("Run trace", "no steps were recorded for this run.")
        return

    with st.expander(
        f"🧭  Run trace — {len(steps)} steps  ·  multi-step reasoning",
        expanded=False,
    ):
        st.caption(
            "Every node the orchestrator ran, in order, with what each one "
            "cost. Routing is deterministic code — no LLM decides the path."
        )
        for index, step in enumerate(steps, start=1):
            _render_step(index, step)

        st.divider()
        st.markdown(f"**Tokens by agent** — {_tag('token efficiency', BCG_DARK)}"
                    " is a design input, so this is the evidence for it.",
                    unsafe_allow_html=True)
        st.dataframe(_agent_usage_rows(state), hide_index=True)
        st.caption(_PRICE_NOTE)


def _render_step(index: int, step: StepLog) -> None:
    """One trace line: who ran, the transition, the note, and what it cost."""
    st.markdown(
        f"{_tag(f'{index:02d}. {step.agent}', BCG_DARK)} &nbsp; "
        f"{_tag(f'{step.stage_in.value} → {step.stage_out.value}', GREY)}",
        unsafe_allow_html=True,
    )
    note = (step.note or "").strip()
    if note:
        st.markdown(note)

    meta = [_clock(step.timestamp)]
    if step.model:
        meta.append(step.model)
    if step.input_tokens or step.output_tokens:
        meta.append(f"{step.input_tokens:,} in / {step.output_tokens:,} out")
        meta.append(_cost_label(step.cost_usd, _step_cost_is_known(step)))
    else:
        meta.append("no LLM call")
    st.caption(" · ".join(meta))


def _agent_usage_rows(state: ArchitectState) -> list[dict]:
    """Per-agent totals as table rows — a groupby over `history`, nothing more."""
    step_counts: dict[str, int] = {}
    for step in state.history:
        step_counts[step.agent] = step_counts.get(step.agent, 0) + 1

    rows = []
    for agent, usage in state.usage_by_agent().items():
        rows.append(
            {
                "Agent": agent,
                "Steps": step_counts.get(agent, 0),
                "Input": usage.input_tokens,
                "Output": usage.output_tokens,
                "Total tokens": usage.total_tokens,
                "Cost": _cost_label(
                    usage.cost_usd, _agent_cost_is_known(state, agent)
                ),
            }
        )
    return rows


# ══════════════════════════════════════════════════════════════════════════
# E. Knowledge retrieved  [retrieval]
# ══════════════════════════════════════════════════════════════════════════
def render_knowledge(state: ArchitectState) -> None:
    """What the researcher pulled out of the knowledge base.

    Opens itself when EMPTY. A run that retrieved nothing is a real finding
    about our KB, and hiding it behind a closed expander would be the one thing
    this screen exists to stop.
    """
    chunks = state.retrieved_knowledge
    count = len(chunks)
    label = (
        f"📚  Knowledge retrieved — {count} chunk{'s' if count != 1 else ''}"
        f"  ·  retrieval"
        if count
        else "📚  Knowledge retrieved — none  ·  retrieval"
    )

    with st.expander(label, expanded=not count):
        if not count:
            st.warning(
                "**No knowledge-base results for this run.** The researcher ran "
                "and returned zero chunks, so the design below is grounded in "
                "the request and the repository only — not in the curated "
                "pattern library."
            )
            return

        st.caption(
            "Passages handed to the architect as grounding. Box 1 = curated "
            "architecture patterns, box 2 = domain knowledge, "
            "box 3 = live web-search fallback."
        )
        for index, chunk in enumerate(chunks, start=1):
            box_name = _BOX_LABELS.get(chunk.box, f"box {chunk.box}")
            source = chunk.source or "unknown source"
            head = f"{index:02d}. {source}"
            if chunk.page:
                head += f", p. {chunk.page}"
            st.markdown(
                f"{_tag(head, BCG_DARK)} &nbsp; {_tag(f'box {chunk.box} — {box_name}', GREY)}",
                unsafe_allow_html=True,
            )
            st.markdown(chunk.content or "_(empty chunk)_")
            if chunk.distance is None:
                st.caption("no distance recorded (web-search result)")
            else:
                st.caption(f"distance {chunk.distance:.4f} — lower is closer")


# ══════════════════════════════════════════════════════════════════════════
# F. Repository analysis  [tool use]
# ══════════════════════════════════════════════════════════════════════════
def render_repo_analysis(state: ArchitectState) -> None:
    """The repo representation the ingestor built, when there is one."""
    repo = state.repo_representation
    if repo is None:
        _missing(
            "Repository analysis",
            "greenfield run: no repository URL was given, so nothing was cloned "
            "or analysed.",
        )
        _render_deep_dives(state)  # never silently dropped, even without a repo
        return

    with st.expander("🗂️  Repository analysis  ·  tool use", expanded=False):
        meta, structure, behavior = repo.meta, repo.structure, repo.behavior

        _text("Repository", meta.url)
        if meta.commit_sha:
            st.caption(f"commit `{meta.commit_sha}` · ingested {meta.ingested_at}")

        stack = structure.tech_stack
        if stack.languages:
            st.markdown("**Languages** (lines of code)")
            st.markdown(
                "\n".join(
                    f"- {name} — {loc:,} LOC"
                    for name, loc in sorted(
                        stack.languages.items(), key=lambda kv: -kv[1]
                    )
                )
            )
        _bullets("Frameworks", stack.frameworks)
        _bullets("Dependencies", stack.dependencies)
        _bullets("External services", stack.external_services)

        _text("What the repository does", behavior.overview)
        if behavior.partitions:
            st.markdown("**Partitions**")
            for part in behavior.partitions:
                st.markdown(f"- **{part.name or 'unnamed'}** — {part.role}")
                if part.functionality:
                    st.markdown(f"  {part.functionality}")
                if part.paths:
                    st.caption("paths: " + ", ".join(f"`{p}`" for p in part.paths))

        if structure.architecture_diagram:
            st.markdown("**Architecture diagram** (derived from import edges)")
            diagram_tab, source_tab = st.tabs(["Diagram", "Mermaid source"])
            with diagram_tab:
                _render_mermaid(structure.architecture_diagram)
            with source_tab:
                st.code(structure.architecture_diagram, language="text")

        if structure.file_tree:
            st.markdown("**File tree**")
            st.code(structure.file_tree, language="text")
        if structure.repo_map:
            st.markdown("**Repo map** (most-imported files first)")
            st.code(structure.repo_map, language="text")
        if structure.integration_interface:
            st.markdown("**Integration interface** (condensed API surface)")
            st.code(structure.integration_interface, language="text")
        if structure.dependency_edges:
            st.markdown(f"**Import edges** — {len(structure.dependency_edges)}")
            st.dataframe(
                [
                    {"imports from": edge.target, "file": edge.source}
                    for edge in structure.dependency_edges
                ],
                hide_index=True,
            )

        _render_deep_dives(state, inline=True)


def _render_deep_dives(state: ArchitectState, inline: bool = False) -> None:
    """Cached drill-downs, if any.

    Nothing writes `repo_deep_dives` today, so this is normally a no-op — it
    renders if the field is ever populated. `inline=True` is the call from
    inside the repository expander (Streamlit forbids nesting one expander in
    another, so it draws flat there); `inline=False` is the greenfield path,
    where there is no repository expander to sit inside.
    """
    dives = state.repo_deep_dives
    if not dives:
        return

    def _body() -> None:
        for dive in dives:
            st.markdown(f"**`{dive.target}`** — {dive.question}")
            st.markdown(dive.insight)
            st.caption(_clock(dive.timestamp))

    if inline:
        st.divider()
        st.markdown(f"**Deep dives** — {len(dives)}")
        _body()
    else:
        with st.expander(f"🔍  Repository deep dives — {len(dives)}"):
            _body()


def _render_mermaid(source: str, height: int = 420) -> None:
    """Render Mermaid source as an actual picture.

    Uses the Mermaid CDN inside `st.components.v1.html` rather than a Python
    package, so this adds NO dependency to the project. The trade-off is that
    the diagram NEEDS INTERNET AT RENDER TIME — offline, the iframe stays blank
    and the "Mermaid source" tab next to it is the fallback.

    The source is HTML-escaped before being injected; Mermaid reads the node's
    text content, so the escaped entities arrive as the original characters.
    """
    escaped = html.escape(source)
    components.html(
        f"""
        <div style="background:#ffffff; padding:8px; border-radius:6px;">
          <pre class="mermaid" style="margin:0; background:transparent;">{escaped}</pre>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <script>
          mermaid.initialize({{ startOnLoad: true, theme: "neutral" }});
        </script>
        """,
        height=height,
        scrolling=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# G. Review report  [iteration + quality gate]
# ══════════════════════════════════════════════════════════════════════════
# The four code-owned rubric checks, in report order. Each is a 0-2 diagnostic
# score and the verdict requires every one of them to be 2 (see reviewer.py).
_CODE_CHECKS: list[tuple[str, str]] = [
    ("all_artifacts_present", "All artifacts present"),
    ("constraint_coverage", "Constraint coverage"),
    ("traceability", "Traceability"),
    ("adr_presence", "ADR presence"),
]
# The five LLM-owned judgments: binary, each paired with its written reason.
_LLM_CHECKS: list[tuple[str, str]] = [
    ("repo_grounding", "Repo grounding"),
    ("flaw_detection", "Flaw detection"),
    ("adr_soundness", "ADR soundness"),
    ("best_practice_grounding", "Best-practice grounding"),
    ("refinement_readiness", "Refinement readiness"),
]


def render_review_report(state: ArchitectState) -> None:
    """The quality gate, with the code-owned / LLM-owned split made obvious.

    That split is a core design claim of the project — deterministic code owns
    the verdict, the LLM only supplies judgments and reasons — and this screen
    is the only place it can actually be seen.
    """
    review = state.review
    if review is None:
        _missing(
            "Review report",
            "this run produced no review result, so the design never reached "
            "the quality gate.",
        )
        return

    verdict = "PASS" if review.overall_status == "pass" else "FAIL"
    with st.expander(
        f"⚖️  Review report — {verdict}, {len(review.issues)} issue"
        f"{'s' if len(review.issues) != 1 else ''}  ·  iteration + quality gate",
        expanded=False,
    ):
        left, right = st.columns(2)

        with left:
            st.markdown(
                f"{_tag('CODE-OWNED — rubric checks', BCG_DARK)}<br>"
                f"{_tag('deterministic Python · scored 0-2 · all must be 2 to pass', GREY)}",
                unsafe_allow_html=True,
            )
            st.write("")
            for field, label in _CODE_CHECKS:
                score = getattr(review.rubric_scores, field, 0)
                mark, color = ("✓", BCG_GREEN) if score == 2 else ("✕", RED)
                st.markdown(
                    f"{_tag(mark, color)} &nbsp; {html.escape(label)} &nbsp; "
                    f"{_tag(f'{score}/2', GREY)}",
                    unsafe_allow_html=True,
                )

        with right:
            st.markdown(
                f"{_tag('LLM-OWNED — judgments', BCG_DARK)}<br>"
                f"{_tag('model judgment · binary · never summed into the verdict', GREY)}",
                unsafe_allow_html=True,
            )
            st.write("")
            for field, label in _LLM_CHECKS:
                passed = bool(getattr(review.rubric_scores, field, False))
                mark, color = ("✓", BCG_GREEN) if passed else ("✕", RED)
                st.markdown(
                    f"{_tag(mark, color)} &nbsp; {html.escape(label)}",
                    unsafe_allow_html=True,
                )
                reason = getattr(review.judgment_reasons, field, "") or ""
                st.caption(reason.strip() or "_no reason recorded_")

        st.divider()
        if review.issues:
            st.markdown(f"**Issues** — {len(review.issues)}")
            # `st.table`, NOT `st.dataframe`. The interactive grid clips every
            # cell to one line — `column_config` has no wrap option — so the
            # findings, evidence and fixes were unreadable without scrolling
            # sideways. The static table wraps every cell, fits all six columns
            # at 1080p, and shows no index. It gives up sorting and
            # click-to-expand, which this screen is read and filmed rather than
            # operated, so it never needed.
            st.table(
                [
                    {
                        "ID": issue.id,
                        "Severity": issue.severity,
                        "Category": issue.category,
                        "Finding": issue.finding,
                        "Evidence": issue.evidence,
                        "Suggested fix": issue.suggested_fix,
                        # A bool would print "True"/"False" in a static table.
                        "Needs refine": "yes" if issue.requires_refinement else "—",
                    }
                    for issue in review.issues
                ]
            )
        else:
            st.caption("No issues were recorded.")

        st.divider()
        st.markdown(
            f"**Refine loop** — {state.refine_iterations} of "
            f"{MAX_REFINE_ITERATIONS} iteration"
            f"{'s' if MAX_REFINE_ITERATIONS != 1 else ''} used · "
            + (
                "stopped on the cost cap"
                if state.stopped_on_cap
                else "stopped on the reviewer's verdict"
            )
        )
        _text(
            "Refinement instruction fed back to the architect",
            review.refinement_instruction,
        )


# ══════════════════════════════════════════════════════════════════════════
# H. Full artifact detail
# ══════════════════════════════════════════════════════════════════════════
def render_context_record(state: ArchitectState) -> None:
    """The frozen context — every field, not just the one-line summary."""
    record = state.context_record
    if record is None:
        _missing(
            "Context Record",
            "clarification never completed, so no context was locked for this run.",
        )
        return

    title = record.project_name or "Context Record"
    with st.expander(f"📋  Context Record — {title}  ·  locked after clarification"):
        drawn = [
            _text("Business goal", record.business_goal),
            _text("Problem statement", record.problem_statement),
            _bullets("Users and stakeholders", record.users),
            _bullets("Functional requirements", record.functional_requirements),
            _bullets(
                "Non-functional requirements", record.non_functional_requirements
            ),
            _text("Cloud provider", record.cloud_provider),
            _text("Budget", record.budget),
            _bullets("Compliance requirements", record.compliance_requirements),
            _bullets("Existing systems", record.existing_systems),
            _bullets("Assumptions", record.assumptions),
            _bullets("Open questions", record.open_questions),
            _text("Summary", record.summary),
        ]
        if not any(drawn):
            st.caption("The context record exists but every field is empty.")


def render_features(state: ArchitectState) -> None:
    """The derived features — previously not rendered at all."""
    features = state.features
    if not features:
        _missing(
            "Features",
            "no features were derived, so there is nothing for the design to "
            "trace back to.",
        )
        return

    with st.expander(f"🎯  Features — {len(features)}  ·  what the system must do"):
        for feature in features:
            _render_feature(feature)


def _render_feature(feature: Feature) -> None:
    st.markdown(
        f"{_tag(feature.id, BCG_DARK)} &nbsp; **{html.escape(feature.name)}** &nbsp; "
        f"{_tag(feature.priority.upper(), GREY)}",
        unsafe_allow_html=True,
    )
    _text("Description", feature.description)
    _text("Scenario", feature.scenario)
    _bullets("Acceptance criteria", feature.acceptance_criteria)
    _chips("Traces to requirements", feature.related_requirement_ids)
    st.divider()


def render_blueprint(state: ArchitectState) -> None:
    """The blueprint — both views plus the reasoning around them."""
    blueprint = state.blueprint
    if blueprint is None:
        _missing("Blueprint", "the architect produced no blueprint for this run.")
        return

    headline = blueprint.selected_pattern or blueprint.project_name or "architecture"
    with st.expander(f"🏛️  Blueprint — {headline}", expanded=True):
        st.caption(
            f"{blueprint.blueprint_id} · v{blueprint.version}"
            + (f" · {blueprint.project_name}" if blueprint.project_name else "")
        )
        _text("Selected pattern", blueprint.selected_pattern)
        _text("Rationale", blueprint.rationale)

        drew_view = False
        if blueprint.stakeholder_view.strip():
            st.markdown("**Stakeholder view** (business-facing)")
            st.markdown(blueprint.stakeholder_view)
            drew_view = True
        if blueprint.technical_view.strip():
            st.markdown("**Technical view**")
            st.markdown(blueprint.technical_view)
            drew_view = True
        if not drew_view:
            st.caption("Both blueprint views are empty.")

        _bullets("Components", blueprint.components)
        _bullets("Data flows", blueprint.data_flows)
        _bullets("Constraints addressed", blueprint.constraints_addressed)
        _bullets("Assumptions", blueprint.assumptions)
        _bullets("Open risks", blueprint.open_risks)
        _chips("Addresses features", blueprint.addressed_feature_ids)


def render_adrs(state: ArchitectState) -> None:
    """Every ADR in full — context, rationale, alternatives, consequences."""
    adrs = state.adrs
    if not adrs:
        _missing(
            "Architecture Decision Records",
            "no ADRs were written, so the design's decisions are unrecorded.",
        )
        return

    with st.expander(f"📑  Architecture Decision Records — {len(adrs)}"):
        for adr in adrs:
            _render_adr(adr)


def _render_adr(adr: ADR) -> None:
    status_color = BCG_GREEN if adr.status == "accepted" else GREY
    st.markdown(
        f"{_tag(adr.id, BCG_DARK)} &nbsp; **{html.escape(adr.title)}** &nbsp; "
        f"{_tag(adr.status.upper(), status_color)}",
        unsafe_allow_html=True,
    )
    _text("Context", adr.context)
    _text("Decision", adr.decision)
    _text("Rationale", adr.rationale)
    _bullets("Alternatives considered", adr.alternatives_considered)
    _bullets("Positive consequences", adr.positive_consequences)
    _bullets("Negative consequences", adr.negative_consequences)
    _chips("Related features", adr.related_feature_ids)
    _chips("Related components", adr.related_component_names)
    _bullets("Sources", adr.source_references)
    st.divider()


def render_components(state: ArchitectState) -> None:
    """Every component in full — I/O, dependencies, security, scalability."""
    components_list = state.components
    if not components_list:
        _missing(
            "Components",
            "no components were described, so the blueprint has no parts to build.",
        )
        return

    with st.expander(f"🧩  Components — {len(components_list)}"):
        for component in components_list:
            _render_component(component)


def _render_component(component: ComponentDescription) -> None:
    st.markdown(
        f"{_tag(component.id, BCG_DARK)} &nbsp; **{html.escape(component.name)}** "
        f"&nbsp; {_tag(component.component_type, GREY)}",
        unsafe_allow_html=True,
    )
    _text("Purpose", component.purpose)
    _text("Description", component.description)
    _bullets("Inputs", component.inputs)
    _bullets("Outputs", component.outputs)
    _bullets("Dependencies", component.dependencies)
    _bullets("Technology choices", component.technology_choices)
    _bullets("Security considerations", component.security_considerations)
    _bullets("Scalability considerations", component.scalability_considerations)
    _chips("Implements features", component.related_feature_ids)
    _chips("Justified by ADRs", component.related_adr_ids)
    st.divider()
