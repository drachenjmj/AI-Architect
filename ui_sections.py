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

`render_context_approval` and `render_optional_context` are the two INTERACTIVE
panels in the file, and both keep the rule rather than bending it: each draws
the frozen record from the state and RETURNS what the human asked for as a
plain value. Neither writes anything. Every write — the edit, the accept, the
optional answers, the advisory call — happens in ui.py, which owns the REACT
half, and lands in the pipeline through `pipeline.agents.clarifier`, which is
the only writer of a `ContextRecord`. Neither panel is allowed to touch the
record for the same reason the UI is not: there would then be two writers of
Maheen's schema and no way to tell which one produced a given field.

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
import re
from datetime import datetime
from pathlib import Path
from typing import Sequence

import streamlit as st
import streamlit.components.v1 as components

import field_discussion
from pipeline.agents.clarifier import (
    CLARIFIER_LABEL,
    EDITABLE_RECORD_FIELDS,
    field_purpose_hint,
    optional_slots,
    slot_question,
)
from pipeline.agents.reviewer import ADVISORY_CRITERIA
from pipeline.llm import PRICING_USD_PER_MTOK
from pipeline.persistence import runs_dir
from pipeline.refine_gate import (
    MAX_REFINE_ITERATIONS,
    MAX_USER_ROUNDS,
)
from pipeline.flow_syntax import classify_flow, split_directional_flow
from pipeline.sign_off import (
    feedback_is_closed,
    open_findings,
    requires_deliberate_confirmation,
    unapplied_feedback,
)
from pipeline.state import (
    ADR,
    ArchitectState,
    ComponentDescription,
    ContextEdits,
    ContextRecord,
    Feature,
    KBChunk,
    NOT_APPLICABLE_REASONS,
    Stage,
    StepLog,
)

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


def _datestamp(timestamp: str) -> str:
    """Date AND time from an ISO timestamp; the raw value if it will not parse.

    `_clock` drops the date, which is right for a trace where every step
    happened in the same sitting. A sign-off is read weeks later by someone
    asking WHEN this was decided, and "14:32:10" does not answer that.
    """
    try:
        return datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M UTC")
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

    Rendered as ONE compact line (the workspace header carries the verdict
    chip; this is the sentence behind it) rather than a padded alert box —
    the finished screen is a workspace, not a notification feed.
    """
    review = state.review
    open_issues = len(review.issues) if review else 0
    issue_text = f"{open_issues} open issue{'s' if open_issues != 1 else ''}"

    if review is None:
        st.markdown(
            f"{_tag('NOT REVIEWED', _SEVERITY_COLORS['medium'])} &nbsp; "
            "Design produced without a review result — nothing here has "
            "passed the quality gate.",
            unsafe_allow_html=True,
        )
        return

    if state.stopped_on_cap:
        st.markdown(
            f"{_tag('BEST-EFFORT — STOPPED ON BUDGET', _SEVERITY_COLORS['high'])} &nbsp; "
            f"{state.refine_iterations} redesign"
            f"{'s' if state.refine_iterations != 1 else ''} (cap "
            f"{MAX_REFINE_ITERATIONS}); **{issue_text}** still open. The run "
            f"ended on the cost guardrail, not on a clean pass.",
            unsafe_allow_html=True,
        )
        return

    if review.overall_status != "pass":
        st.markdown(
            f"{_tag('REVIEW DID NOT PASS', _SEVERITY_COLORS['high'])} &nbsp; "
            f"**{issue_text}** after {state.refine_iterations} refine "
            f"iteration{'s' if state.refine_iterations != 1 else ''} — "
            f"see the Review view.",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"{_tag('DESIGN COMPLETE — REVIEW PASSED', BCG_GREEN)} &nbsp; "
        + (
            f"reached after {state.refine_iterations} refine "
            f"iteration{'s' if state.refine_iterations != 1 else ''}."
            if state.refine_iterations
            else "passed on the first pass."
        ),
        unsafe_allow_html=True,
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
def _source_card_html(
    index: int, source: str, box_line: str, distance_line: str, body: str
) -> str:
    """One retrieved chunk as a compact HTML source card.

    Meta (source, box, distance) sits in the card header so it is readable at
    a glance; a short passage shows in full, a long one is clipped with the
    complete text behind a native `<details>` toggle — expandable without
    costing a Streamlit rerun. Everything is escaped — chunk text is
    retrieved data, not trusted markup.
    """

    head = (
        f"<div class='ws-card-head'>"
        f"<span class='ws-card-title'>{index:02d}. {html.escape(source)}</span>"
        f"<span class='ws-card-meta'>{html.escape(box_line)}"
        + (f" &nbsp;·&nbsp; {html.escape(distance_line)}" if distance_line else "")
        + "</span></div>"
    )
    body = body or "(empty chunk)"
    if len(body) <= 420:
        passage = html.escape(body).replace("\n", "<br>")
    else:
        clipped = html.escape(_clip_sentence(body, 380)).replace("\n", "<br>")
        full = html.escape(body).replace("\n", "<br>")
        passage = (
            f"{clipped}…"
            f"<details style='margin-top:.35rem'><summary style='cursor:pointer'>"
            f"Show full passage</summary><div style='margin-top:.35rem'>{full}</div>"
            f"</details>"
        )
    return f"<div class='ws-card'>{head}<div class='ws-card-body'>{passage}</div></div>"


def render_knowledge(state: ArchitectState) -> None:
    """What the researcher pulled out of the knowledge base.

    One compact card per chunk — source, box and distance readable at a
    glance, the passage below them — instead of a single long expander. The
    empty case stays loud: a run that retrieved nothing is a real finding
    about our KB, and it must not hide behind a closed section.
    """
    chunks = state.retrieved_knowledge
    count = len(chunks)

    if not count:
        st.warning(
            "**No knowledge-base results for this run.** The researcher ran "
            "and returned zero chunks, so the design is grounded in the "
            "request and the repository only — not in the curated pattern "
            "library."
        )
        return

    st.markdown(
        f"**📚 Knowledge retrieved — {count} chunk"
        f"{'s' if count != 1 else ''}** &nbsp; "
        f"{_tag('retrieval', BCG_DARK)}",
        unsafe_allow_html=True,
    )
    st.caption(
        "Passages handed to the architect as grounding. Box 1 = curated "
        "architecture patterns, box 2 = domain knowledge, box 3 = live "
        "web-search fallback."
    )
    for index, chunk in enumerate(chunks, start=1):
        box_name = _BOX_LABELS.get(chunk.box, f"box {chunk.box}")
        source = chunk.source or "unknown source"
        head_source = f"{source}, p. {chunk.page}" if chunk.page else source
        distance_line = (
            "" if chunk.distance is None else f"distance {chunk.distance:.4f}"
        )
        st.markdown(
            _source_card_html(
                index,
                head_source,
                f"box {chunk.box} — {box_name}",
                distance_line,
                chunk.content or "(empty chunk)",
            ),
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════
# F. Repository analysis  [tool use]
# ══════════════════════════════════════════════════════════════════════════
def render_repo_analysis(state: ArchitectState) -> None:
    """The repo representation the ingestor built, when there is one.

    SUMMARY FIRST: what the repository is, what it is built with, and how it
    is partitioned render directly — the old single expander made the page
    look empty and hid the analysis behind a click. The genuinely deep
    artifacts (file tree, repo map, import edges, API surface, diagram) stay
    in expanders, where their density belongs.
    """

    repo = state.repo_representation
    if repo is None:
        _missing(
            "Repository analysis",
            "greenfield run: no repository URL was given, so nothing was cloned "
            "or analysed.",
        )
        _render_deep_dives(state)  # never silently dropped, even without a repo
        return

    st.markdown(
        f"**🗂️  Repository analysis** &nbsp; {_tag('tool use', BCG_DARK)}",
        unsafe_allow_html=True,
    )
    meta, structure, behavior = repo.meta, repo.structure, repo.behavior

    _text("Repository", meta.url)
    if meta.commit_sha:
        st.caption(f"commit `{meta.commit_sha}` · ingested {meta.ingested_at}")

    # ── Summary, directly visible ────────────────────────────────────────
    stack = structure.tech_stack
    if stack.languages:
        st.markdown(
            "**Languages** — "
            + " · ".join(
                f"{name} {loc:,} LOC"
                for name, loc in sorted(
                    stack.languages.items(), key=lambda kv: -kv[1]
                )
            )
        )
    _bullets("Frameworks", stack.frameworks)
    _bullets("External services", stack.external_services)
    _text("What the repository does", behavior.overview)

    if behavior.partitions:
        st.markdown("**Partitions**")
        for part in behavior.partitions:
            st.markdown(f"- **{part.name or 'unnamed'}** — {part.role}")

    # ── Technical detail, collapsible ───────────────────────────────────
    if structure.architecture_diagram:
        with st.expander("Architecture diagram (derived from import edges)"):
            diagram_tab, source_tab = st.tabs(["Diagram", "Mermaid source"])
            with diagram_tab:
                _render_mermaid(structure.architecture_diagram)
            with source_tab:
                st.code(structure.architecture_diagram, language="text")

    if structure.file_tree:
        with st.expander("File tree"):
            st.code(structure.file_tree, language="text")
    if structure.repo_map:
        with st.expander("Repo map (most-imported files first)"):
            st.code(structure.repo_map, language="text")
    if structure.integration_interface:
        with st.expander("Integration interface (condensed API surface)"):
            st.code(structure.integration_interface, language="text")
    if structure.dependency_edges:
        with st.expander(f"Import edges — {len(structure.dependency_edges)}"):
            st.dataframe(
                [
                    {"imports from": edge.target, "file": edge.source}
                    for edge in structure.dependency_edges
                ],
                hide_index=True,
            )

    for part in behavior.partitions:
        if part.functionality.strip() or part.paths:
            with st.expander(f"Partition detail — {part.name or 'unnamed'}"):
                _text("What it does", part.functionality)
                if part.paths:
                    st.caption("paths: " + ", ".join(f"`{p}`" for p in part.paths))

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
# The five code-owned rubric checks, in report order. Each is a 0-2 diagnostic
# score and the verdict requires every one of them to be 2 (see reviewer.py).
_CODE_CHECKS: list[tuple[str, str]] = [
    ("all_artifacts_present", "All artifacts present"),
    ("constraint_coverage", "Constraint coverage"),
    ("traceability", "Traceability"),
    ("adr_presence", "ADR presence"),
    ("source_integrity", "Source integrity"),
]
# The five LLM-owned judgments: binary, each paired with its written reason.
_LLM_CHECKS: list[tuple[str, str]] = [
    ("repo_grounding", "Repo grounding"),
    ("flaw_detection", "Flaw detection"),
    ("adr_soundness", "ADR soundness"),
    ("best_practice_grounding", "Best-practice grounding"),
    ("refinement_readiness", "Refinement readiness"),
]


def _render_rubric_columns(review) -> None:
    """The code-owned / LLM-owned split, side by side.

    That split is a core design claim of the project — deterministic code owns
    the verdict, the LLM only supplies judgments and reasons — and the Review
    view is the only place it can actually be seen.
    """
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
            # not_applicable FIRST - see the same guard in pipeline/run.py.
            # The stored boolean is true for these, so checking it first
            # would paint "no evidence existed" as a green tick.
            if field in review.not_applicable:
                why = NOT_APPLICABLE_REASONS.get(field, "not applicable")
                st.markdown(
                    f"{_tag('n/a', GREY)} &nbsp; {html.escape(label)} &nbsp; "
                    f"{_tag(html.escape(why) + ' · excluded from verdict', GREY)}",
                    unsafe_allow_html=True,
                )
            else:
                passed = bool(getattr(review.rubric_scores, field, False))
                mark, color = ("✓", BCG_GREEN) if passed else ("✕", RED)
                # See the same note in pipeline/run.py: an advisory ✕ beside
                # a PASS verdict needs to say why it did not block.
                advisory = (
                    f" &nbsp; {_tag('advisory · excluded from verdict', GREY)}"
                    if field in ADVISORY_CRITERIA and not passed else ""
                )
                st.markdown(
                    f"{_tag(mark, color)} &nbsp; {html.escape(label)}{advisory}",
                    unsafe_allow_html=True,
                )
            reason = getattr(review.judgment_reasons, field, "") or ""
            st.caption(reason.strip() or "_no reason recorded_")


def render_review_dimensions(review) -> None:
    """Every dimension the verdict rests on, DIRECTLY visible: the actual
    check names with a ✓/✕ and nothing else.

    This is the layer a reader scans to see WHAT was examined before
    deciding whether to open the evidence — the per-criterion scores and
    written reasons stay behind `Detailed reviewer evidence`. Only the
    existing rubric's names are used; no dimensions are invented.
    """
    st.markdown("#### Review dimensions")
    left, right = st.columns(2)
    with left:
        st.markdown(
            f"{_tag('Deterministic checks', BCG_DARK)}<br>"
            f"{_tag('code-owned · all must pass', GREY)}",
            unsafe_allow_html=True,
        )
        st.write("")
        for field, label in _CODE_CHECKS:
            score = getattr(review.rubric_scores, field, 0)
            mark, color = ("✓", BCG_GREEN) if score == 2 else ("✕", RED)
            st.markdown(
                f"{_tag(mark, color)} &nbsp; {html.escape(label)}",
                unsafe_allow_html=True,
            )
    with right:
        st.markdown(
            f"{_tag('LLM judgments', BCG_DARK)}<br>"
            f"{_tag('binary · never summed into the verdict', GREY)}",
            unsafe_allow_html=True,
        )
        st.write("")
        for field, label in _LLM_CHECKS:
            # not_applicable FIRST — the stored boolean is true for these,
            # so checking it would paint "no evidence existed" as a tick.
            if field in review.not_applicable:
                why = NOT_APPLICABLE_REASONS.get(field, "not applicable")
                st.markdown(
                    f"{_tag('n/a', GREY)} &nbsp; {html.escape(label)} &nbsp; "
                    f"{_tag(html.escape(why), GREY)}",
                    unsafe_allow_html=True,
                )
                continue
            passed = bool(getattr(review.rubric_scores, field, False))
            mark, color = ("✓", BCG_GREEN) if passed else ("✕", RED)
            advisory = (
                f" &nbsp; {_tag('advisory', GREY)}"
                if field in ADVISORY_CRITERIA and not passed
                else ""
            )
            st.markdown(
                f"{_tag(mark, color)} &nbsp; {html.escape(label)}{advisory}",
                unsafe_allow_html=True,
            )


def render_review_report(state: ArchitectState) -> None:
    """The quality gate: verdict, blocking findings and the dimension
    summary in the main flow; the per-criterion evidence one click away.

    The verdict line, the findings table, the loop status and the dimension
    summary are what a reader acts on, so they sit directly on the page;
    the detailed rubric — scores, judgment reasons, advisory findings, the
    instruction fed back to the architect — is the audit trail behind one
    clearly-named expander.
    """
    review = state.review
    if review is None:
        _missing(
            "Review report",
            "this run produced no review result, so the design never reached "
            "the quality gate.",
        )
        return

    # "blocking", not "issue". Advisory and not-applicable criteria are
    # deliberately kept OUT of `issues` (see agents/reviewer.py), so a run
    # can carry a real complaint and still show zero here. Saying "0 issues"
    # beside a FAIL, or beside an advisory finding, reads as a contradiction.
    verdict = "PASS" if review.overall_status == "pass" else "FAIL"
    verdict_color = BCG_GREEN if verdict == "PASS" else RED
    st.markdown(
        f"{_tag(f'REVIEW — {verdict}', verdict_color)} &nbsp; "
        f"**{len(review.issues)} blocking issue"
        f"{'s' if len(review.issues) != 1 else ''}** &nbsp; "
        f"{_tag('iteration + quality gate', GREY)}",
        unsafe_allow_html=True,
    )

    if review.issues:
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
        st.caption("No blocking issues were recorded.")

    # Refinement status stays VISIBLE — it is the state of the loop, not a
    # diagnostic — and so does the dimension summary directly under it.
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
    render_review_dimensions(review)

    with st.expander("Detailed reviewer evidence"):
        _render_rubric_columns(review)

        # ADVISORY FINDINGS. A criterion excluded from the verdict still gets
        # asked and still answers, and its answer can be substantive - in run
        # 20260819T080216Z-f981f8ef `refinement_readiness` named the exact
        # unlinked feature while the blocking-issue list was empty. It raises no
        # ReviewIssue by design, so without this block that finding is visible
        # only as a red cross in the column above, and the report looks like it
        # found nothing. Shown, and shown as NOT counting.
        advisory_findings = [
            (label, (getattr(review.judgment_reasons, field, "") or "").strip())
            for field, label in _LLM_CHECKS
            if field in ADVISORY_CRITERIA
            and not bool(getattr(review.rubric_scores, field, False))
        ]
        if advisory_findings:
            st.markdown(
                f"**Advisory findings** — {len(advisory_findings)} &nbsp; "
                f"{_tag('recorded, excluded from the verdict', GREY)}",
                unsafe_allow_html=True,
            )
            for label, reason in advisory_findings:
                st.caption(f"**{html.escape(label)}** — {reason or '_no reason recorded_'}")

        _text(
            "Refinement instruction fed back to the architect",
            review.refinement_instruction,
        )


# ══════════════════════════════════════════════════════════════════════════
# H. Full artifact detail
# ══════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════
# The context lock — the human's veto, before anything is spent on the record
# ══════════════════════════════════════════════════════════════════════════
# Field name -> the words a person would use for it. Only the ones that need
# rewording are listed; anything else falls back to its own name, title-cased,
# so adding a field to `ContextRecord` shows up here without a second edit.
_FIELD_LABELS: dict[str, str] = {
    "project_name": "Project name",
    "business_goal": "Business goal",
    "problem_statement": "Problem statement",
    "users": "Users and stakeholders",
    "functional_requirements": "Functional requirements",
    "non_functional_requirements": "Non-functional requirements (scale, availability)",
    "cloud_provider": "Cloud provider",
    "budget": "Budget",
    "compliance_requirements": "Compliance requirements",
    "existing_systems": "Existing systems",
}

# A list field is edited as a textarea, ONE ITEM PER LINE. That is this caller's
# convention, not the schema's (`ContextEdits` deliberately takes real lists) —
# the CLI, which has one line to work with, comma-separates instead.
_LIST_HELP = "One per line."


def _field_label(name: str) -> str:
    return _FIELD_LABELS.get(name, name.replace("_", " ").capitalize())


def _record_nonce(record: ContextRecord) -> str:
    """A short id that changes whenever the record does. Used only in widget keys.

    Streamlit remembers a widget by its key and ignores the `value=` you pass on
    later reruns, so a stable key would keep showing the text a field held BEFORE
    the clarifier re-judged it — the human would strike a value, watch the panel
    redraw, and see the struck value still sitting in the box. Rotating the keys
    with the record's content is what makes the panel show the record rather than
    a memory of it.
    """
    return f"{abs(hash(record.model_dump_json())):x}"


def _render_advisory_thread(
    state: ArchitectState, subject: str = "context_record"
) -> None:
    """Replay the read-only questions asked about one subject, oldest first.

    Drawn from `state.advisory_turns`, not from a UI session variable, which is
    why it survives a rerun, a page refresh and a checkpoint reload — the same
    reason the rest of the conversation is derived from the state object.

    Filtered by `subject` for the same reason the prompt is: the questions asked
    over the frozen record and the questions asked over the finished design are
    two conversations about two different objects, and showing them as one would
    make each answer look like a reply to the wrong question.
    """
    for turn in [t for t in state.advisory_turns if t.subject == subject]:
        with st.chat_message("user"):
            st.write(turn.question)
        with st.chat_message("assistant"):
            st.write(turn.answer)


def render_ask_ai(
    *, scope_id: str, field_key: str, field_label: str
) -> tuple[str, str] | None:
    """DRAW one compact "Ask AI" discussion beside a field. Pure — see the
    module's ONE RULE: reads the session-only history
    (`field_discussion.history_for`), draws it, and RETURNS what the human
    just did. It never calls the model and never writes anywhere; the
    caller (ui.py) acts on the returned intent:

        ("ask",   message)     -> field_discussion.ask(...), then rerun
        ("apply", suggestion)  -> copy `suggestion` into the field's own
                                   draft/widget value, then rerun
        None                   -> nothing submitted this render

    Draws two plain `st.button`s ("Send", "Use suggestion") inside a
    `st.popover`, so it MUST NOT be called from inside an `st.form(...)`:
    Streamlit forbids any button but `st.form_submit_button` in a form. That
    is also WHY the three screens that use this (the required clarification
    questions, the optional round, and the Context Record approval screen)
    do not wrap their fields in `st.form` at all any more — a form batches
    every widget inside it until one submit, which silently fed a stale,
    pre-edit draft to a discussion opened before the human submitted, and
    reordered the page into "every input, then every label" because a form
    is one reserved slot that paints as a single block whichever field's
    label/Ask-AI pair was drawn beside it. Each caller instead renders the
    label, this popover, and the field's own plain (non-form) input inline,
    one question at a time — see `render_context_approval` /
    `render_optional_context` / ui.py's clarification and intake fields for
    the exact interleaving, and each one's own note on why `st.form` is
    gone. A popover, not an expander: its content is cheap (a session-state
    read, no disk/network I/O), so it needs none of the lazy-render gating
    the sidebar's "Switch run" expander does.

    `scope_id`/`field_key` are the SAME stable identity `field_discussion`
    keys history by — see that module's docstring for the exact scheme
    each screen uses (run_id + "context.<field>"/"clarification::<question>",
    or `field_discussion.pre_run_scope()` + "start.system_description" before
    a run exists).
    """
    history = field_discussion.history_for(scope_id, field_key)
    intent: tuple[str, str] | None = None
    with st.popover("💬 Ask AI"):
        st.caption(f"Discuss: {field_label}")
        if not history:
            st.caption(
                "Ask before you fill this in — I can already see the "
                "project context, so you do not need to repeat it."
            )
        for turn in history:
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant"):
                st.write(turn["reply"])
        message = st.text_input(
            "Ask something",
            key=f"ask_ai_msg_{scope_id}_{field_key}",
            label_visibility="collapsed",
            placeholder="What do you recommend?",
        )
        if st.button("Send", key=f"ask_ai_send_{scope_id}_{field_key}") and message.strip():
            intent = ("ask", message.strip())
        if history and history[-1]["suggested_answer"]:
            st.info(f"Suggested answer: {history[-1]['suggested_answer']}")
            if st.button("Use suggestion", key=f"ask_ai_use_{scope_id}_{field_key}"):
                intent = ("apply", history[-1]["suggested_answer"])
    return intent


def render_optional_context(state: ArchitectState) -> tuple[str, object] | None:
    """The optional-context round. Returns the human's intent, or None.

    One of:
        ("skip",       None)          leave every optional field as it stands
        ("continue",   ContextEdits)  apply whatever was filled in (may be none)
        ("field_ask",  dict)          a per-field "Ask AI" message was sent —
                                       kwargs for `field_discussion.ask(state, ...)`
        None                          this rerun carried no submission

    Shown BETWEEN the required clarification questions and the Context Record
    review (`render_context_approval`) — never at the same time as either.
    Every field here is one entry from `clarifier.optional_slots`: a relevant
    Context Record field the required round did not (or could not) ask
    about, each rendered as exactly ONE input tied to exactly ONE field, so
    an answer can never land anywhere but the field it was asked about (see
    `optional_slots`'s docstring on the combined-question bug this
    structurally prevents). Leaving every box blank and clicking "Skip
    optional questions" is a fully valid, unstyled outcome — this panel never
    adds warning/error styling for an empty field.

    ASK AI, PER FIELD: each field also carries a compact `render_ask_ai`
    popover, drawn inline beside the field's own label (see its docstring
    for why there is no wrapping `st.form` here at all any more). An
    "apply" intent from it is handled RIGHT HERE — it is a session-state-only
    write, not an LLM call or a ContextRecord mutation, so it stays within
    the DRAW-only rule. An "ask" intent DOES need the model, so it is the one
    thing this otherwise-DRAW-only function returns for ui.py to act on,
    exactly like "skip"/"continue" already are — never called directly.
    """
    record = state.context_record
    if record is None:  # defensive: only reachable with a record locked
        _missing("Optional context", "no record was locked, so there is nothing to ask.")
        return None

    slots = optional_slots(state, record)
    if not slots:  # defensive: the caller only shows this panel when non-empty
        return "skip", None

    nonce = _record_nonce(record)
    scope_id = state.run_id
    st.markdown("#### Optional context")
    st.caption(
        "These details may improve the design, but they are not required. "
        "You can answer any of them, leave fields blank, or skip this step."
    )

    # NOT `st.form`. A form batches every widget inside it until submit —
    # which is exactly wrong here: the per-field "Ask AI" button sits OUTSIDE
    # the form (Streamlit forbids anything but `form_submit_button` inside
    # one) and needs the CURRENTLY VISIBLE draft the human is looking at when
    # they click it, not the last-submitted value. A plain, session-state-
    # backed widget commits its value to `st.session_state` on every blur/
    # change — including the blur that fires when the human clicks Ask AI —
    # so the draft `render_ask_ai` reads below is always live. It is also
    # what keeps every question block (label + Ask AI + input) ONE unit in
    # the page's actual paint order: a form is a single reserved slot whose
    # own widgets render as one block AFTER it, wherever in the script it was
    # opened, which is what silently pushed every input above every label
    # here before. "Continue"/"Skip" stay the only commit boundary into the
    # pipeline — nothing here writes `ContextEdits` until one of them fires.
    fields: dict[str, str | list[str]] = {}
    for field in slots:
        question = slot_question(field)
        help_text = question.why_needed or None
        input_key = f"oc_field_{field}_{nonce}"
        field_key = f"context.{field}"

        col1, col2 = st.columns([6, 1])
        with col1:
            st.markdown(question.question)
        with col2:
            intent = render_ask_ai(
                scope_id=scope_id, field_key=field_key, field_label=question.question
            )
        if intent is not None:
            action, value = intent
            if action == "apply":
                st.session_state[input_key] = value
                st.rerun()
            else:  # "ask" — needs the model; bubble up to ui.py's REACT half
                other_known: dict[str, object] = {
                    other: st.session_state.get(f"oc_field_{other}_{nonce}", "")
                    for other in slots
                    if other != field
                }
                other_known.update(
                    {
                        name: getattr(record, name)
                        for name in EDITABLE_RECORD_FIELDS
                        if getattr(record, name)
                    }
                )
                return "field_ask", {
                    "scope_id": scope_id,
                    "field_key": field_key,
                    "raw_prompt": state.initial_request.raw_prompt,
                    "repo_representation": state.repo_representation,
                    "known_context": other_known,
                    "field_label": question.question,
                    "field_purpose": question.why_needed,
                    "current_draft_answer": st.session_state.get(input_key, ""),
                    "message": value,
                }

        if isinstance(getattr(record, field), list):
            text = st.text_area(
                question.question,
                value="",
                help=help_text,
                height=68,
                key=input_key,
                label_visibility="collapsed",
            )
            value = [line.strip() for line in text.splitlines() if line.strip()]
        else:
            value = st.text_input(
                question.question,
                value="",
                help=help_text,
                key=input_key,
                label_visibility="collapsed",
            ).strip()
        if value:
            fields[field] = value

    left, right = st.columns(2)
    with left:
        skip = st.button(
            "Skip optional questions", key=f"oc_skip_{nonce}", use_container_width=True
        )
    with right:
        cont = st.button(
            "Continue",
            key=f"oc_continue_{nonce}",
            type="primary",
            use_container_width=True,
        )

    if skip:
        return "skip", None
    if cont:
        return "continue", ContextEdits(fields=fields)
    return None


def render_context_approval(state: ArchitectState) -> tuple[str, object] | None:
    """The approval panel. Returns the human's intent, or None if they did nothing.

    One of:
        ("accept",    None)            approve the record as it stands
        ("edit",      ContextEdits)    a veto pass — fields, strikes, "you recommend"
        ("ask",       "question")      a read-only question about the WHOLE record
        ("field_ask", dict)            a per-field "Ask AI" message was sent —
                                        kwargs for `field_discussion.ask(state, ...)`
        None                          this rerun carried no submission

    Called AFTER the dedicated optional-context round (`render_optional_
    context`, resolved before this screen is ever reached — see
    `pipeline.agents.clarifier.PendingDecision.OPTIONAL_CONTEXT`), so any
    `record.open_questions` still outstanding here are shown PASSIVELY, not
    as a second active answer form.

    Returning the intent instead of acting on it is what keeps this file
    DRAW-only (see the module docstring): ui.py performs every write, and every
    write goes through `pipeline.agents.clarifier`.

    The edits and the whole-record question are two separate, DELIBERATELY
    different mechanisms. The edit fields are plain, session-state-backed
    widgets (see `render_ask_ai`'s docstring for why: a wrapping `st.form`
    both hid a typed-but-unsubmitted edit from an Ask AI call made before
    "Save changes"/"Approve", and pushed every input above every label),
    committed only when "Save changes" or "✓ Approve and continue" is
    clicked below. The whole-record question keeps its OWN small `st.form`
    (`context_gate_ask_{nonce}`) — a single field with nothing else in it,
    so batching costs it nothing — specifically so it stays a separate
    submission: batching a QUESTION together with the edit fields would mean
    you could not ask what a field means without also submitting your
    changes to it.

    ASK AI, PER FIELD: same pattern as `render_optional_context` — an
    "apply" intent from `render_ask_ai` is a session-state-only write,
    handled right here; an "ask" intent needs the model and is returned as
    "field_ask" for ui.py to act on.
    """
    record = state.context_record
    if record is None:  # defensive: the gate is only reachable with a record
        _missing("Context Lock", "no record was locked, so there is nothing to approve.")
        return None

    nonce = _record_nonce(record)
    scope_id = state.run_id
    st.markdown("#### Approve the ground truth before any design work starts")
    st.caption(
        "Everything below was either taken from your request or assumed on your "
        "behalf. Nothing has been researched, designed or reviewed yet — so a "
        "correction here costs nothing, and the same correction after the design "
        "costs the whole run. Change what is wrong, strike what you do not "
        "accept, then approve."
    )

    # NOT `st.form` — see the identical note in `render_optional_context`.
    # Batching every field until one submit is exactly what let a typed-but-
    # unsent edit go missing from an Ask AI call made before "Save changes"/
    # "Approve", and what pushed every input above every label on this
    # screen: a form is one reserved slot, and everything inside it paints as
    # a single block wherever in the script it was opened, regardless of
    # where the label/Ask-AI pair for the same field was drawn. Plain widgets
    # commit to `st.session_state` on the same blur that opens the popover,
    # so the draft `render_ask_ai` reads is always the one on screen, and the
    # page paints in call order — label, Ask AI, input, next field, ...
    # "Save changes"/"Approve and continue" stay the only commit boundary:
    # nothing here writes back to the Context Record until one of them fires.
    st.markdown("**The record**")
    edited: dict[str, str | list[str]] = {}
    for name in EDITABLE_RECORD_FIELDS:
        current = getattr(record, name)
        label = _field_label(name)
        input_key = f"cg_field_{name}_{nonce}"
        field_key = f"context.{name}"

        col1, col2 = st.columns([6, 1])
        with col1:
            st.markdown(f"**{label}**")
        with col2:
            intent = render_ask_ai(scope_id=scope_id, field_key=field_key, field_label=label)
        if intent is not None:
            action, value = intent
            if action == "apply":
                st.session_state[input_key] = value
                st.rerun()
            else:  # "ask" — needs the model; bubble up to ui.py's REACT half
                # LIVE, not the frozen record: a field drawn earlier in this
                # same pass (business_goal, say) already has its just-typed,
                # not-yet-saved edit sitting in `st.session_state` under its
                # own widget key by the time we reach a LATER field's Ask AI
                # (problem_statement, say) — see the doc's acceptance case
                # for this screen. `getattr(record, other)` would only ever
                # see the last-Approved value, silently hiding that edit from
                # the discussion until Save/Approve was clicked.
                other_known = {}
                for other in EDITABLE_RECORD_FIELDS:
                    if other == name:
                        continue
                    live = st.session_state.get(
                        f"cg_field_{other}_{nonce}", getattr(record, other)
                    )
                    if live:
                        other_known[other] = live
                return "field_ask", {
                    "scope_id": scope_id,
                    "field_key": field_key,
                    "raw_prompt": state.initial_request.raw_prompt,
                    "repo_representation": state.repo_representation,
                    "known_context": other_known,
                    "field_label": label,
                    "field_purpose": field_purpose_hint(name),
                    "current_draft_answer": st.session_state.get(input_key, str(current)),
                    "message": value,
                }

        if isinstance(current, list):
            text = st.text_area(
                label,
                value="\n".join(str(v) for v in current),
                help=_LIST_HELP,
                height=90,
                key=input_key,
                label_visibility="collapsed",
            )
            edited[name] = [line.strip() for line in text.splitlines() if line.strip()]
        else:
            edited[name] = st.text_input(
                label,
                value=str(current),
                key=input_key,
                label_visibility="collapsed",
            ).strip()

    recommend = st.multiselect(
        "I do not know — you recommend",
        options=list(EDITABLE_RECORD_FIELDS),
        format_func=_field_label,
        help=(
            "The clarifier proposes a value with a one-line reason, recorded "
            "as its own labelled assumption. It comes back here for you to "
            "accept or override — the same veto, in the other direction."
        ),
        key=f"cg_recommend_{nonce}",
    )

    struck: list[str] = []
    if record.assumptions:
        st.markdown("**Assumptions — tick anything you do not accept**")
        st.caption(
            f"`{CLARIFIER_LABEL}` marks a value the clarifier filled in "
            f"instead of asking you. A struck assumption stays struck: it "
            f"will not be proposed again later in this run."
        )
        for assumption in record.assumptions:
            if st.checkbox(assumption, key=f"cg_strike_{assumption}_{nonce}"):
                struck.append(assumption)

    if record.open_questions:
        # PASSIVE on purpose: the dedicated optional round (see
        # `render_optional_context`, shown BEFORE this screen) is where
        # a relevant, unresolved field gets an actual answer box, tied
        # to exactly one Context Record field. Anything still listed
        # here — a field the human skipped there, or a free-text bonus
        # question the model raised that maps to no single field — is
        # shown for traceability only, never as a second active form:
        # that second form is exactly what let one combined question
        # ("expected traffic and compliance?") risk an answer landing in
        # the wrong field, since a raw question has no field of its own.
        st.markdown("**Unresolved optional context**")
        st.caption(
            "Not required to approve. Edit a field above to answer any of "
            "these directly."
        )
        for question in record.open_questions:
            st.caption(f"• {question}")

    left, right = st.columns(2)
    with left:
        save = st.button(
            "Save changes", key=f"cg_save_{nonce}", use_container_width=True
        )
    with right:
        accept = st.button(
            "✓ Approve and continue",
            key=f"cg_accept_{nonce}",
            type="primary",
            use_container_width=True,
        )

    if accept:
        return "accept", None
    if save:
        # Only fields that ACTUALLY changed are sent. An untouched box is not an
        # edit, and passing it as one would make every save look like a rewrite
        # of the whole record in the trace.
        changed = {
            name: value
            for name, value in edited.items()
            if value != getattr(record, name)
        }
        edits = ContextEdits(
            fields=changed,
            struck_assumptions=struck,
            recommend=list(recommend),
        )
        return None if edits.is_empty() else ("edit", edits)

    st.divider()
    st.markdown("**Ask about this record**")
    st.caption(
        "Read-only. It does not enter the pipeline, change the record or resolve "
        "this approval — ask as many as you like. It does spend tokens, so it is "
        "billed to the run trace under its own agent."
    )
    _render_advisory_thread(state)
    with st.form(f"context_gate_ask_{nonce}", clear_on_submit=True):
        question = st.text_input(
            "Your question",
            placeholder="e.g. why does expected scale matter here? what do teams usually pick?",
            label_visibility="collapsed",
        )
        asked = st.form_submit_button("Ask")
    if asked and question.strip():
        return "ask", question.strip()
    return None


def render_user_rounds(state: ArchitectState) -> None:
    """How much of the human-round budget is left. Silent until any is spent.

    The same honesty rule the cost figures follow: a cap the user cannot see is
    a cap that surprises them at the worst moment.

    TWO actions spend it, and both ask for work to be REDONE: an edit at the
    context gate that reopens a gap, and feedback on a finished run. Answering
    clarifying questions, approving the record and asking about it are all free,
    so a run that does neither never sees this line at all (see
    `ArchitectState.user_rounds`).
    """
    if not state.user_rounds:
        return
    left = max(MAX_USER_ROUNDS - state.user_rounds, 0)
    st.caption(
        f"Refinement rounds used: {state.user_rounds} of {MAX_USER_ROUNDS} "
        f"({left} left). An edit that reopens a gap costs one, and so does "
        f"feedback on a finished run — answering, approving and asking are free."
    )


def _version_number(record) -> int:
    """`record.version` as an int, for display branching only.

    The context gate's edit path stores text values without schema validation,
    so a checkpointed record can carry `version` as a numeric string ("1")
    instead of an int — and `record.version <= 1` would raise TypeError right
    on the DONE screen. Coerce numeric strings; anything non-numeric falls
    back to 1, the "locked after clarification" branch, so the page still
    renders. Read-only: the state on disk is never rewritten for display.
    """
    try:
        return int(record.version)
    except (TypeError, ValueError):
        return 1


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
    version = _version_number(record)
    locked = (
        "locked after clarification"
        if version <= 1
        else f"v{version} · revised after your feedback"
    )
    with st.expander(f"📋  Context Record — {title}  ·  {locked}"):
        if record.revision_reason.strip():
            # WHY this version exists, in the words that caused it. The earlier
            # versions are not shown but are not lost either — they are on the
            # state, and every one of them is in this run's checkpoints.
            superseded = ", ".join(
                f"v{old.version}" for old in state.context_history
            ) or "the previous version"
            st.info(
                f"**Revised — supersedes {superseded}.** You asked for: "
                f"{record.revision_reason.strip()}"
            )
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


# ══════════════════════════════════════════════════════════════════════════
# The Context view's compact presentation of the locked record
# ══════════════════════════════════════════════════════════════════════════
def _record_card_html(title: str, values: list[str]) -> str | None:
    """One record field-group as a compact HTML card. None when empty.

    Values are escaped: the record holds user- and clarifier-written text,
    not trusted markup. The card classes are the workspace's own (`ws-card`,
    defined in ui_workspace's scoped styles), so this stays inside the
    lightweight visual system rather than growing a second one.
    """
    items = [str(v).strip() for v in values if str(v).strip()]
    if not items:
        return None
    body = "<br>".join(html.escape(item) for item in items)
    return (
        f"<div class='ws-card'>"
        f"<div class='ws-card-title'>{html.escape(title)}</div>"
        f"<div class='ws-card-body'>{body}</div>"
        f"</div>"
    )


def render_context_cards(state: ArchitectState) -> None:
    """The locked Context Record as the PRIMARY object of the Context view.

    The old presentation (one expander, every field as a bullet list) buried
    the record's shape. This renders it as compact grouped cards — objective,
    scope, platform, compliance — with nothing invented: every card maps to
    existing `ContextRecord` fields, and a group whose fields are empty
    simply does not appear. The revision banner and version line from the
    expander version are kept, above the cards.
    """
    record = state.context_record
    if record is None:
        _missing(
            "Context Record",
            "clarification never completed, so no context was locked for this run.",
        )
        return

    title = record.project_name or "Context Record"
    version = _version_number(record)
    locked = (
        "locked after clarification"
        if version <= 1
        else f"v{version} · revised after your feedback"
    )
    st.markdown(f"### Context Record — {html.escape(title)}")
    st.caption(locked)
    if record.revision_reason.strip():
        superseded = ", ".join(
            f"v{old.version}" for old in state.context_history
        ) or "the previous version"
        st.info(
            f"**Revised — supersedes {superseded}.** You asked for: "
            f"{record.revision_reason.strip()}"
        )

    objective = _record_card_html(
        "Objective",
        [
            *([record.business_goal] if record.business_goal.strip() else []),
            *([f"Problem: {record.problem_statement}"]
              if record.problem_statement.strip() else []),
        ],
    )
    if objective:
        st.markdown(objective, unsafe_allow_html=True)

    card_rows: list[list[tuple[str, list[str]]]] = [
        [
            ("Users & stakeholders", record.users),
            ("Functional scope", record.functional_requirements),
        ],
        [
            ("Platform & budget",
             [record.cloud_provider, record.budget]),
            ("Scale & reliability", record.non_functional_requirements),
        ],
        [
            ("Compliance", record.compliance_requirements),
            ("Existing systems", record.existing_systems),
        ],
        [
            ("Assumptions", record.assumptions),
            ("Open questions", record.open_questions),
        ],
    ]
    drew_any = objective is not None
    for row in card_rows:
        left, right = st.columns(2)
        for column, (card_title, values) in zip((left, right), row):
            card = _record_card_html(card_title, values)
            if card is not None:
                with column:
                    st.markdown(card, unsafe_allow_html=True)
                drew_any = True

    if record.summary.strip():
        st.markdown(
            _record_card_html("Summary", [record.summary]),
            unsafe_allow_html=True,
        )
        drew_any = True

    if not drew_any:
        st.caption("The context record exists but every field is empty.")


def render_clarification_history(state: ArchitectState) -> None:
    """The run's Q&A — original request plus every clarification exchange.

    The linear pre-run flow shows these as chat bubbles because the
    conversation IS the interface there. The finished workspace is an
    artifact view, so the same data (drawn from `state`, nothing stored
    separately) becomes a collapsed, clearly-labelled secondary section:
    the questions the clarifier asked, the answers given, and any
    corrections filed after design — all of them, unchanged.
    """
    exchanges = list(state.clarification_answers.items())
    with st.expander(
        f"💬 Clarification history — {len(exchanges)} exchange"
        f"{'s' if len(exchanges) != 1 else ''}  ·  clarification"
    ):
        st.markdown("**Original request**")
        st.markdown(f"> {html.escape(state.initial_request.raw_prompt.strip())}")
        for question, answer in exchanges:
            st.markdown(f"**{html.escape(question)}**")
            st.markdown(f"> {html.escape(answer)}")


# ══════════════════════════════════════════════════════════════════════════
# The two feedback boxes at DONE — iterative refinement, human-driven
# ══════════════════════════════════════════════════════════════════════════
# NEVER TABS, never interchangeable. Which box the text is typed into IS the
# route (see pipeline/user_feedback.py), so blurring the two is what makes
# people put a design directive in the requirements box — and a directive
# filed as a requirements correction re-opens the record and re-runs the whole
# pipeline for nothing. Two clearly-labelled boxes, each in the view of the
# artifact it acts on (the workspace split — see ui_workspace.py), are the
# cheapest possible classifier and the only one that cannot be wrong about
# what the person meant.
#
# BATCHING. Before the workspace split both boxes shared one page, so one
# submission could carry a correction AND a directive and cost a single
# refinement round. The split would have quietly taken that away — one round
# per view — so the boxes no longer submit directly. Each STAGES its text
# into a session-state bundle (`stage_pending_feedback` below), the bundle
# survives navigation, and ONE submit action in the sidebar sends everything
# as a single `user_feedback.submit_feedback` call: one round, however many
# views contributed. The bundle is UI-only and unsent: it lives in
# `st.session_state`, never in the checkpointed state, and a page refresh
# discards it — the same deal every other unsent widget value already gets.
#
# Bundle shape mirrors the backend payload exactly — one text per kind,
# `{"requirements": str, "design": str}` — so submission is a kwarg-for-kwarg
# hand-off with no translation layer that could blur the intent.
_FEEDBACK_COPY: dict[str, dict[str, str]] = {
    "requirements": {
        "heading": "✏️  Is anything above wrong or missing?",
        "expander": "✏️  Correct the context",
        "label": "Correct the requirements",
        "placeholder": (
            "e.g. peak load is 500k users, not 50k — and we are on Azure, not AWS."
        ),
        "help": (
            "Facts about what the system must do or must respect: scale, cloud, "
            "budget, compliance, existing systems."
        ),
        # Short, but the cost warning survives: this is the expensive path —
        # it re-opens the record and re-runs research, design and review.
        "note": (
            "Submitting this **re-opens the Context Record** — research, "
            "design and review then all re-run. For a change to the "
            "architecture itself, use the change box in the "
            "**Architecture** view."
        ),
    },
    "design": {
        "heading": "✏️  Want the architecture changed?",
        "expander": "✏️  Request a design change",
        "label": "Direct the architect",
        "placeholder": (
            "e.g. use SQS instead of Kafka — we have no team to run a broker."
        ),
        "help": (
            "Changes to the design itself: a technology, a component, a pattern, "
            "a decision you disagree with."
        ),
        "note": (
            "Goes straight to the architect, ranked **above** the reviewer's "
            "instruction; only what you name is changed. Submitted from the "
            "sidebar as one refinement round."
        ),
    },
}


def _feedback_key(kind: str, state: ArchitectState) -> str:
    """Widget key for one box, rotated by round AND by every staging.

    The same trick as `_record_nonce`, for the same Streamlit reason: a widget
    is remembered by its key, so a stable key would leave the text a person
    just staged sitting in the box afterwards, looking unsent. `user_rounds`
    ticks on every accepted submission and `fb_staged_count` on every staging
    (either box — see `stage_pending_feedback`), so the next screen gets a
    fresh, empty box in both cases.
    """
    staged = st.session_state.get("fb_staged_count", 0)
    return f"fb_{kind}_{state.user_rounds}_{staged}"


# The pending bundle lives ONLY in session state — unsent UI draft, never a
# fact about the run, so it must not be checkpointed with the state object.
_PENDING_BUNDLE_KEY = "pending_feedback"


def get_pending_feedback() -> dict[str, str]:
    """The unsent feedback bundle: one text per kind. Read-only accessor."""

    return dict(st.session_state.get(_PENDING_BUNDLE_KEY) or {})


def pending_feedback_kinds() -> list[str]:
    """Which kinds currently have staged text, in submission order."""

    bundle = st.session_state.get(_PENDING_BUNDLE_KEY) or {}
    return [
        kind for kind in ("requirements", "design")
        if str(bundle.get(kind) or "").strip()
    ]


def stage_pending_feedback(kind: str, text: str) -> None:
    """Add one box's text to the pending bundle. Session state only.

    Staging twice from the same kind CONCATENATES rather than replaces: the
    second entry is a second thing the person wants changed, and silently
    dropping the first would be the "accept the text and lose it" failure the
    feedback trail everywhere else in this system exists to prevent.
    """

    bundle = st.session_state.setdefault(_PENDING_BUNDLE_KEY, {})
    existing = str(bundle.get(kind) or "").strip()
    bundle[kind] = (
        f"{existing}\n\n{text.strip()}" if existing else text.strip()
    )
    # Rotates BOTH boxes' widget keys (see `_feedback_key`): whichever box
    # staged, both come back empty on the next render.
    st.session_state["fb_staged_count"] = st.session_state.get(
        "fb_staged_count", 0
    ) + 1


def clear_pending_feedback() -> None:
    """Empty the bundle after its content was submitted (or discarded)."""

    st.session_state[_PENDING_BUNDLE_KEY] = {}


# What each `UserFeedback.status` looks like in the history line under a box.
# Every value gets its own mark, because the whole point of the status field is
# that "we did it", "we did something near it", "you dropped it" and "nothing
# has happened yet" are four different answers to the same question.
_STATUS_MARKS: dict[str, str] = {
    "applied": "✓ applied",
    "objected": "⚠ applied with an objection",
    "abandoned": "✕ abandoned at sign-off",
    "pending": "◷ pending",
}


def _render_feedback_history(state: ArchitectState, kind: str) -> None:
    """What was already asked for on this axis, and what became of it."""
    entries = [entry for entry in state.user_feedback if entry.kind == kind]
    for entry in entries:
        mark = _STATUS_MARKS.get(entry.status, entry.status)
        st.caption(f"{mark} · round {entry.round} · you asked: {entry.text}")


def render_feedback_box(state: ArchitectState, kind: str) -> None:
    """Draw ONE feedback box, visually SECONDARY. Its button stages, never
    submits.

    The artifact this box acts on is the primary content of its view, so the
    entry form lives inside a collapsed expander — one labelled line on the
    page until it is opened. What stays OUTSIDE the expander is state, not
    chrome: the feedback history, a staged-text preview, and the reason the
    box is closed. DRAW-only like everything here: the box writes nothing but
    the UI's own session state (`stage_pending_feedback`), and the actual
    submission — one `user_feedback.submit_feedback` call for the WHOLE
    bundle — is performed by ui.py when the sidebar panel's submit action
    fires.

    At the cap — and after sign-off — the box is disabled and says why. Never
    accept text and quietly drop it: a box that takes a paragraph and does
    nothing with it is worse than a box that is visibly closed. AFTER
    SIGN-OFF it is closed for a second reason — the design was taken, so
    changing it would leave the thing on screen different from the thing that
    was accepted. `feedback_is_closed` knows about both, so this call site
    does not have to.
    """
    copy = _FEEDBACK_COPY[kind]
    closed, why = feedback_is_closed(state)

    _render_feedback_history(state, kind)
    if closed:
        st.caption(
            f"Feedback is closed for this run: {why}. Start a new run to keep "
            f"going — the artifacts above are what this one produced. You can "
            f"still ask questions about the design; that changes nothing."
        )
        return

    staged = str(get_pending_feedback().get(kind) or "").strip()
    if staged:
        st.caption(f"**Queued for this round:** {' '.join(staged.split())}")

    with st.expander(copy["expander"]):
        st.caption(copy["note"])
        text = st.text_area(
            copy["label"],
            key=_feedback_key(kind, state),
            placeholder=copy["placeholder"],
            help=copy["help"],
            height=90,
            label_visibility="collapsed",
        )
        added = st.button(
            "Add to pending feedback",
            key=f"fb_add_{kind}_{state.user_rounds}",
            help="Adds this to the pending feedback round. Submit everything at "
                 "once from the sidebar — one submission is one refinement "
                 "round, however many boxes contributed.",
        )
        if added:
            if not text.strip():
                st.warning("Type what you would like changed first.")
            else:
                stage_pending_feedback(kind, text.strip())
                st.rerun()  # redraw with an empty box and the preview updated


# ══════════════════════════════════════════════════════════════════════════
# Close-out at DONE — objections, the advisory turn, the sign-off, the waiver
# ══════════════════════════════════════════════════════════════════════════
# Everything below is DRAW-only like the rest of this file: the panels RETURN
# what the human asked for and ui.py performs it through `pipeline.sign_off` /
# `pipeline.agents.clarifier`. Nothing here waives anything or accepts anything.
_SEVERITY_COLORS: dict[str, str] = {"high": RED, "medium": "#B9770E", "low": GREY}


def render_objections(state: ArchitectState) -> None:
    """Where the design DEVIATES from what the user asked for, and why.

    Silent unless the architect recorded an objection. When it did, this is the
    single most important thing on the screen: the person told the system to do
    something, the system did something adjacent instead, and every other
    section would render exactly as if it had complied. Placed against the
    artifacts rather than in the trace expander for that reason — an objection
    inside a collapsed section is an objection nobody reads.
    """
    objections = [entry for entry in state.user_feedback if entry.objection.strip()]
    if not objections:
        return

    st.error(
        f"**The design deviates from {len(objections)} thing"
        f"{'s' if len(objections) != 1 else ''} you asked for.** The architect "
        f"built the closest version it could and said why — read this before "
        f"anything below it."
    )
    for entry in objections:
        st.markdown(
            f"{_tag('YOU ASKED', BCG_DARK)} &nbsp; {html.escape(entry.text)}",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"{_tag('THE ARCHITECT', RED)} &nbsp; {html.escape(entry.objection.strip())}",
            unsafe_allow_html=True,
        )
        st.caption(
            f"round {entry.round} · recorded against your directive, never fed "
            f"back to the architect — objecting is a voice, not a way out."
        )


def render_design_advisory(state: ArchitectState) -> str | None:
    """Ask a question about the finished design. Returns the question, or None.

    THE CHEAP HALF OF THE FEEDBACK PAIR, and it sits beside the expensive one on
    purpose. Most people's first reaction to a finished design is a question,
    not an instruction — and with only a directive box in front of them, the
    question gets typed as an instruction and costs a full refine round
    (~75k tokens) to answer something one flash-lite call would have. Asking
    costs that one call, changes nothing, and the directive they write
    afterwards is a better directive.

    Works at ACCEPTED as well as DONE: reading an accepted design is not
    changing one, which is exactly why this is not disabled with the boxes.
    """
    st.markdown("##### 💬  Want to understand something first?")
    st.caption(
        "Read-only. It does not change the design, spend a refinement round, or "
        "re-run anything — ask as many as you like. It does spend tokens, so it "
        "is billed to the run trace under its own agent."
    )
    _render_advisory_thread(state, subject="design")
    with st.form("design_ask", clear_on_submit=True):
        question = st.text_input(
            "Your question",
            key="design_ask_question",
            placeholder="e.g. why did it pick this pattern? what breaks if we drop the cache?",
            label_visibility="collapsed",
        )
        asked = st.form_submit_button("Ask")
    return question.strip() if asked and question.strip() else None


def _render_open_findings(state: ArchitectState) -> list:
    """The findings a sign-off would accept, highest severity first. Returns them."""
    findings = open_findings(state)
    if not findings:
        st.success(
            "**Nothing outstanding.** The review recorded no blocking findings, "
            "so accepting this records no waiver."
        )
        return findings

    st.markdown(
        f"**You would be accepting {len(findings)} open finding"
        f"{'s' if len(findings) != 1 else ''}** — these do not go away, they get "
        f"recorded as accepted:"
    )
    for issue in findings:
        color = _SEVERITY_COLORS.get(issue.severity, GREY)
        st.markdown(
            f"{_tag(issue.severity.upper(), color)} &nbsp; "
            f"`{html.escape(issue.id or '—')}` &nbsp; {html.escape(issue.finding)}",
            unsafe_allow_html=True,
        )
        if issue.suggested_fix.strip():
            st.caption(f"suggested fix: {issue.suggested_fix.strip()}")
    return findings


def render_sign_off(state: ArchitectState) -> tuple[str, str] | None:
    """The sign-off panel. Returns ("accept", note) when confirmed, else None.

    DONE means the pipeline stopped. This button is the only thing in the system
    that can say a HUMAN TOOK THE DESIGN, and the panel exists to make sure that
    what they took is in front of them when they take it:

      * every open finding, highest severity first, ABOVE the button;
      * any change they typed and never re-ran, which acceptance will abandon;
      * an optional note, which stays optional — a mandatory justification field
        produces "n/a" and then means nothing.

    A HIGH-severity finding requires a second, deliberate confirmation; a medium
    or low one does not. That asymmetry is the point: a confirmation everything
    triggers is a confirmation nobody reads. The rule itself is
    `sign_off.requires_deliberate_confirmation`, not a property of this widget.

    Nothing here BLOCKS. Not the findings — most runs end capped, and a sign-off
    that required a pass could never close them — and not the unapplied change,
    which would force a whole refine round to close out a run the person has
    already decided to take.

    Returns None once the run is ACCEPTED: the decision has been made and this
    panel has nothing left to ask. `render_acceptance` shows what was decided.
    """
    if state.stage is Stage.ACCEPTED:
        return None

    st.markdown("##### ✍️  Take this design")
    st.caption(
        "This run is finished, which is not the same as accepted. Signing off "
        "records that a person took this design — and, if the review still has "
        "findings open, exactly which ones they took it with."
    )

    findings = _render_open_findings(state)
    unapplied = unapplied_feedback(state)
    if unapplied:
        # The same category as an open finding — a known thing being accepted
        # despite — so it is shown on the same screen rather than behind a
        # second one. Accepting abandons it; that is stated before the click,
        # never discovered after it.
        st.warning(
            f"**You have {len(unapplied)} unapplied change"
            f"{'s' if len(unapplied) != 1 else ''}.** Accepting drops "
            f"{'them' if len(unapplied) != 1 else 'it'} — the run closes with "
            f"the design as it stands. To apply "
            f"{'them' if len(unapplied) != 1 else 'it'} instead, submit "
            f"{'them' if len(unapplied) != 1 else 'it'} from the matching "
            f"feedback view first and sign off after the run."
        )
        for entry in unapplied:
            st.caption(f"· “{entry.text}” ({entry.kind} box, round {entry.round})")

    note = ""
    if findings:
        note = st.text_input(
            "Note (optional)",
            key="sign_off_note",
            placeholder="e.g. accepted for the pilot, tracked in JIRA-123",
            help="Recorded on the waiver. Optional — leave it empty if you have "
                 "nothing to add.",
        ).strip()

    deliberate = True
    if requires_deliberate_confirmation(state):
        high = sum(1 for issue in findings if issue.severity == "high")
        deliberate = st.checkbox(
            f"I am accepting {high} HIGH-severity finding"
            f"{'s' if high != 1 else ''} in this design.",
            key="sign_off_high_ack",
        )

    accepted = st.button(
        "✓ Accept this design",
        type="primary",
        disabled=not deliberate,
        help="Records the sign-off and closes the run. The artifacts stay "
             "exactly as they are; nothing re-runs.",
    )
    return ("accept", note) if accepted else None


def render_acceptance(state: ArchitectState) -> None:
    """The sign-off, once it exists: when it happened and what it was made against.

    Silent on a run nobody has taken — most of them — so it costs an unaccepted
    run nothing.

    NO SIGNER. `accepted_at` is a timestamp because there is no user identity in
    this system, and a governance record that invented one would be worse than
    no record at all.
    """
    if not state.accepted_at:
        return

    waiver = state.waiver
    if waiver is None:
        st.success(
            f"**Accepted — {_datestamp(state.accepted_at)}.** The review left no "
            f"open findings, so nothing was waived."
        )
        return

    counts = waiver.severity_counts()
    breakdown = ", ".join(
        f"{counts[name]} {name}" for name in ("high", "medium", "low") if name in counts
    )
    st.warning(
        f"**Accepted with a waiver — {_datestamp(waiver.accepted_at)}.** "
        f"{len(waiver.finding_ids)} open finding"
        f"{'s' if len(waiver.finding_ids) != 1 else ''} ({breakdown}) were "
        f"accepted, against a review that said **{waiver.review_status or 'nothing'}**."
    )
    for finding_id, severity in zip(waiver.finding_ids, waiver.severities):
        st.markdown(
            f"{_tag(severity.upper(), _SEVERITY_COLORS.get(severity, GREY))} &nbsp; "
            f"`{html.escape(finding_id or '—')}`",
            unsafe_allow_html=True,
        )
    if waiver.note:
        st.caption(f"Note: {waiver.note}")


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
        # Shown to the human, never to the reviewer - see agents/reviewer.py.
        if blueprint.revision_note.strip():
            st.info(f"**Revised this round** — {blueprint.revision_note.strip()}")
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

    # Evidence lookup for `_render_adr`'s "Literature
    # evidence" block, built once here rather than per-ADR.
    evidence_by_id = {
        chunk.evidence_id: chunk
        for chunk in state.retrieved_knowledge
        if chunk.evidence_id
    }
    with st.expander(f"📑  Architecture Decision Records — {len(adrs)}"):
        for adr in adrs:
            _render_adr(adr, evidence_by_id)


def _render_adr(adr: ADR, evidence_by_id: dict[str, KBChunk] | None = None) -> None:
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
    _chips("Decision topics", adr.related_decision_topic_ids)
    _render_literature_evidence(adr, evidence_by_id or {})
    _bullets("Sources", adr.source_references)
    st.divider()


def _render_literature_evidence(adr: ADR, evidence_by_id: dict[str, KBChunk]) -> None:
    """Compact, honest evidence trail: only curated-KB items this ADR cites
    by exact evidence_id — never raw chunk text, never a web-fallback item
    mislabelled as curated literature. Silent when the ADR cites nothing.
    """
    ids = [value.strip() for value in adr.evidence_ids if value.strip()]
    if not ids:
        return
    st.markdown("**Literature evidence**")
    for evidence_id in ids:
        chunk = evidence_by_id.get(evidence_id)
        if chunk is None:
            # Should not happen post-sanitization, but never invent a
            # location for an ID this render pass cannot resolve.
            st.markdown(f"- `{html.escape(evidence_id)}`")
            continue
        location = f"p. {chunk.page}" if chunk.page else ""
        detail = " — ".join(part for part in (chunk.source, location) if part)
        with st.expander(f"{evidence_id} — {detail}" if detail else evidence_id):
            st.markdown(html.escape(chunk.content))


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
    label = f"{component.id}  {component.name}  [{component.component_type}]"
    with st.expander(label):
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


# ══════════════════════════════════════════════════════════════════════════
# The Overview view — the client-facing result, drawn from existing artifacts
# ══════════════════════════════════════════════════════════════════════════
# Nothing below invents architecture content: every line is an existing
# Blueprint / ADR / Component / Review field, shown compactly enough that
# the four client questions are answered in one screenful. The full detail
# of each artifact stays in the Architecture view; these renderers are the
# executive cut of the same state.

def _clip_sentence(text: str, limit: int = 110) -> str:
    """One line of a longer field, cut on a word boundary."""

    one_line = " ".join((text or "").split())
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:.") + "…"


def render_architecture_summary(state: ArchitectState) -> None:
    """The recommended architecture, DIRECTLY visible — not behind an
    expander. The Blueprint's own stakeholder and technical views are the
    summary; this only frames them."""

    blueprint = state.blueprint
    if blueprint is None:
        _missing(
            "Recommended architecture",
            "the architect produced no blueprint for this run.",
        )
        return

    st.markdown("#### Recommended architecture")
    if blueprint.selected_pattern.strip():
        st.markdown(
            f"**Pattern:** {html.escape(blueprint.selected_pattern.strip())}"
        )
    if blueprint.stakeholder_view.strip():
        st.markdown(blueprint.stakeholder_view.strip())
    if blueprint.technical_view.strip():
        with st.expander("Technical view"):
            st.markdown(blueprint.technical_view.strip())


# Where a one-line decision summary may stop: a short decision shows whole,
# a longer one shows its complete FIRST sentence — never a mid-sentence cut.
_DECISION_FULL_LIMIT = 140


def _decision_summary(decision: str) -> str:
    """One readable line for a decision, cut only on a sentence boundary.

    Deterministic string handling: the full text when it is already short,
    otherwise everything up to the first sentence terminator (kept). No
    ellipsis is appended — an argument the client reads must not end
    mid-thought, and the full decision is one tab away in Architecture.
    A decision with no terminator stays whole however long it is; inventing
    a stopping point would be worse than a wrapping line.
    """
    text = decision.strip()
    if len(text) <= _DECISION_FULL_LIMIT:
        return text
    match = re.search(r"[.!?](?=\s|$)", text)
    if match:
        return text[: match.end()].rstrip()
    return text


def render_key_decisions(state: ArchitectState) -> None:
    """Every ADR as one line: its title (which states the decision) and a
    complete decision sentence. Full trade-offs stay in Architecture."""

    if not state.adrs:
        _missing(
            "Key decisions",
            "no ADRs were written, so the design's decisions are unrecorded.",
        )
        return

    st.markdown("#### Key decisions")
    for adr in state.adrs:
        st.markdown(
            f"{_tag(adr.id, BCG_DARK)} &nbsp; **{html.escape(adr.title)}**",
            unsafe_allow_html=True,
        )
        st.caption(_decision_summary(adr.decision))


def render_component_summary(state: ArchitectState) -> None:
    """The components as a compact grid: name, type, one-line purpose."""

    if not state.components:
        _missing(
            "Components",
            "no components were described, so the blueprint has no parts to build.",
        )
        return

    st.markdown("#### Key components")
    pairs = [
        (
            state.components[i],
            state.components[i + 1] if i + 1 < len(state.components) else None,
        )
        for i in range(0, len(state.components), 2)
    ]
    for left, right in pairs:
        cols = st.columns(2)
        for column, component in zip(cols, (left, right)):
            if component is None:
                continue
            body = (
                f"[{html.escape(component.component_type or 'component')}] "
                + html.escape(
                    _clip_sentence(component.purpose or component.description, 120)
                )
            )
            with column:
                st.markdown(
                    f"<div class='ws-card'>"
                    f"<div class='ws-card-title'>{html.escape(component.name)}"
                    f"</div><div class='ws-card-body'>{body}</div></div>",
                    unsafe_allow_html=True,
                )


def render_blocking_findings(state: ArchitectState) -> None:
    """What the review left open — the client-cut of `review.issues`.

    Full evidence and suggested fixes stay in the Review view's table; here
    each finding is one line, severity-first.
    """

    review = state.review
    if review is None:
        return  # the verdict line already said "not reviewed"
    if not review.issues:
        st.caption("No blocking findings were recorded.")
        return

    st.markdown("#### Blocking findings")
    for issue in review.issues:
        color = _SEVERITY_COLORS.get(issue.severity, GREY)
        st.markdown(
            f"{_tag(issue.severity.upper(), color)} &nbsp; "
            f"`{html.escape(issue.id or '—')}` &nbsp; "
            f"{html.escape(_clip_sentence(issue.finding, 130))}",
            unsafe_allow_html=True,
        )


def render_key_risks(state: ArchitectState) -> None:
    """The design's own stated open risks — one compact card."""

    if state.blueprint is None or not state.blueprint.open_risks:
        return
    st.markdown("#### Key risks")
    st.markdown(
        _record_card_html("Stated open risks", state.blueprint.open_risks),
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# The rebuilt Overview — the CLIENT-FACING ANSWER (workspace UX pass).
#
# Overview is the page a consultant shows a client first, so these sections
# answer the six client questions in order: what do we recommend, why, what
# does it consist of, what was decided, what are the risks, and did it pass.
# Every renderer below is DRAW-only and assembled DETERMINISTICALLY from
# EXISTING artifact fields — no LLM rewrite, no invented claims, no new
# schema. A section whose source is empty is omitted or says so; generic
# architecture boilerplate is never used to fill a gap.
#
# Deliberately NOT here: run trace, token/cost metrics, stage machinery,
# run identifiers in the body, full ADR bodies, full Component
# descriptions, KB chunks, file trees — operations and evidence belong to
# their own views.
# ══════════════════════════════════════════════════════════════════════════
def render_executive_recommendation(state: ArchitectState) -> None:
    """A. The one block a client reads first: WHAT is recommended, for which
    goal, solving which problem.

    The Blueprint's own pattern and stakeholder view ARE the recommendation
    — this only frames them with the Context Record's goal and problem so
    the "why are we here" is answered without leaving the page.
    """
    blueprint = state.blueprint
    if blueprint is None:
        _missing(
            "Executive recommendation",
            "the architect produced no blueprint for this run.",
        )
        return

    st.markdown("#### Executive recommendation")
    if blueprint.selected_pattern.strip():
        st.markdown(f"**{html.escape(blueprint.selected_pattern.strip())}**")

    record = state.context_record
    if record is not None:
        if record.business_goal.strip():
            st.markdown(
                f"**Goal** — {html.escape(record.business_goal.strip())}"
            )
        if record.problem_statement.strip():
            st.markdown(
                f"**Problem** — {html.escape(record.problem_statement.strip())}"
            )

    if blueprint.stakeholder_view.strip():
        st.markdown(blueprint.stakeholder_view.strip())


# The arrow spellings and the directional grammar itself now live in ONE
# shared module (`pipeline.flow_syntax`) used by BOTH this renderer and the
# deterministic Reviewer checks — what renders is exactly what the Reviewer
# accepts as a flow, by construction rather than by convention.

# The two node styles: a component the design OWNS (solid, brand-tinted box)
# and everything else — actors, queues, databases, external systems (rounded,
# greyed). The split is deterministic from `state.components` names, and it
# is what lets a client read the diagram the way the ownership check reads
# the design: SQS or a notification service beside the architecture is not
# a gap in the architecture, it is something the design does not own.
_OWNED_NODE = 'shape=box style="filled" fillcolor="#EFF6F2" color="#147B58" penwidth=1.2'
_EXTERNAL_NODE = (
    'shape=ellipse style="filled,dashed" fillcolor="#F5F7F6" color="#ADB5B1"'
)


def parse_data_flows(
    flows: Sequence[str],
) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Split `blueprint.data_flows` into (source, target, label) edges.

    Deterministic string handling only: a flow is an `A → B` pair, optionally
    followed by `: description` (the edge label); an `A → B → C` chain becomes
    its consecutive pairs. Anything with no arrow (prose, headings, half a
    sentence) is returned verbatim in the second list — never dropped, never
    guessed into a direction. The parsing itself is the shared
    `pipeline.flow_syntax.split_directional_flow`, so the Reviewer's
    renderability invariant and this renderer read the same grammar.

    Returns (edges, unparsed) in the artifact's own order.
    """
    edges: list[tuple[str, str, str]] = []
    unparsed: list[str] = []
    for flow in flows:
        parsed = split_directional_flow(flow)
        if parsed is None:
            if str(flow or "").strip():
                unparsed.append(str(flow).strip())
            continue
        endpoints, label = parsed
        for source, target in zip(endpoints, endpoints[1:]):
            edges.append((source, target, label))
    return edges, unparsed


def architecture_flow_dot(
    edges: list[tuple[str, str, str]], component_names: Sequence[str]
) -> str:
    """Render parsed data-flow edges as a Graphviz DOT digraph. Pure.

    Owned components (by exact `state.components` name) draw as solid boxes;
    every other endpoint — the customer, a managed queue, a database, an
    external service — draws as a dashed ellipse, so ownership is legible on
    the diagram itself. Node order is first-seen, edge order is the
    artifact's own; nothing is inferred beyond the arrow that is already
    on the page.

    TOP-TO-BOTTOM on purpose: the flow reads as a stack of stages (customer
    at the top, storage/notification at the bottom), which keeps long
    service names and edge labels on their own lines instead of shrinking
    the whole picture to fit a wide column. The spacing/font numbers are
    conservative readability settings, nothing decorative.
    """
    def quote(text: str) -> str:
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'

    owned = {name.strip() for name in component_names if name.strip()}
    lines = [
        "digraph architecture {",
        "  rankdir=TB;",
        '  graph [fontname="Helvetica", nodesep=0.5, ranksep=0.85];',
        '  node [fontname="Helvetica", fontsize=12, margin="0.18,0.12"];',
        '  edge [fontname="Helvetica", fontsize=10, color="#6b7a74"];',
    ]
    seen: list[str] = []

    def declare(node: str) -> None:
        if node not in seen:
            seen.append(node)
            style = _OWNED_NODE if node in owned else _EXTERNAL_NODE
            lines.append(f"  {quote(node)} [{style}];")

    for source, target, label in edges:
        declare(source)
        declare(target)
        tag = f' [label={quote(label)}]' if label else ""
        lines.append(f"  {quote(source)} -> {quote(target)}{tag};")
    lines.append("}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# Overview graph + Detailed flow filters.
#
# A real holdout run's diagram carried ~15 components and ~44 data flows —
# every edge was real information, but one undifferentiated graph that size
# is not something a reader can take in. Overview answers "what are the
# major boundaries" in seconds; Detailed keeps every flow reachable, with a
# filter so a reader can ask one question (events? data? observability?) at
# a time. Nothing is deleted: `All` is exactly `parse_data_flows`'s full
# output, unchanged — see `test_all_filter_preserves_every_detailed_flow`.
# ══════════════════════════════════════════════════════════════════════════

FLOW_FILTERS: tuple[str, ...] = ("All", "Business", "Events", "Data", "Observability")


def build_overview_edges(
    components: Sequence[ComponentDescription],
) -> list[tuple[str, str]]:
    """The Overview's structural edges: one per declared component
    dependency, deduplicated. Pure.

    Component Descriptions are the DECLARED relationships in the schema —
    `component.dependencies` names what that component depends on, which is
    a coarser, more deliberate signal than re-deriving structure from
    dozens of granular data-flow labels. A dependency naming something
    outside `components` (a queue, a database, an external system) is kept
    as an edge to an external node — `architecture_flow_dot` already draws
    any non-component endpoint as external, so nothing extra is needed here.
    Self-edges are dropped as noise, not information.
    """
    seen: set[tuple[str, str]] = set()
    edges: list[tuple[str, str]] = []
    for component in components:
        source = component.name.strip()
        if not source:
            continue
        for dependency in component.dependencies:
            target = dependency.strip()
            if not target or target == source:
                continue
            pair = (source, target)
            if pair in seen:
                continue
            seen.add(pair)
            edges.append(pair)
    return edges


def build_overview_edges_from_flows(
    edges: list[tuple[str, str, str]],
) -> list[tuple[str, str]]:
    """Fallback Overview edges when no component declares a dependency:
    the same detailed edges, collapsed to unique (source, target) pairs
    regardless of label. Pure. Still strictly a dedup of real data — never
    an invented relationship.
    """
    seen: set[tuple[str, str]] = set()
    collapsed: list[tuple[str, str]] = []
    for source, target, _label in edges:
        pair = (source, target)
        if pair in seen:
            continue
        seen.add(pair)
        collapsed.append(pair)
    return collapsed


def filter_edges_by_category(
    edges: list[tuple[str, str, str]], category: str
) -> list[tuple[str, str, str]]:
    """The Detailed-view subset for one filter. Pure.

    `category` is one of `FLOW_FILTERS`; "All" returns `edges` unchanged —
    the exact set `parse_data_flows` produced, nothing added or removed.
    """
    if category == "All":
        return edges
    wanted = category.lower()
    return [edge for edge in edges if classify_flow(*edge) == wanted]


def render_target_architecture(state: ArchitectState) -> None:
    """B. The structural result, DIRECTLY visible — as a diagram.

    Two views, chosen with a radio (Overview is the default so a reader
    lands on the calm picture first):

    * Overview — declared components as nodes, deduplicated dependency
      edges (`build_overview_edges`, falling back to collapsed data-flow
      pairs when no dependency is declared). No labels; the point is
      boundaries, not detail.
    * Detailed — the full `parse_data_flows` edge set, exactly as before,
      with an additional display-only filter (`FLOW_FILTERS`) that never
      removes a flow from the underlying data — switching back to "All"
      always shows everything again.

    Built deterministically from nothing but the saved flows, dependencies,
    and component names — no LLM, no new inference, no new schema. Flows
    that carry no arrow cannot be drawn safely and stay as visible text
    below the diagram, in EITHER view, unaffected by the filter — a filter
    narrows what is DRAWN, never what is RECORDED.
    """
    blueprint = state.blueprint
    if blueprint is None:
        _missing(
            "Target architecture",
            "the architect produced no blueprint for this run.",
        )
        return

    st.markdown("#### Target architecture")
    edges, unparsed = parse_data_flows(blueprint.data_flows)
    component_names = [c.name for c in state.components]

    view = st.radio(
        "Diagram view",
        ("Overview", "Detailed"),
        horizontal=True,
        key=f"diagram_view_{state.run_id}",
    )

    if view == "Overview":
        overview_edges = build_overview_edges(state.components)
        if not overview_edges:
            overview_edges = build_overview_edges_from_flows(edges)
        st.caption(f"Overview: {len(overview_edges)} relationship(s) · All: {len(edges)} flow(s)")
        if overview_edges:
            dot = architecture_flow_dot(
                [(source, target, "") for source, target in overview_edges],
                component_names,
            )
            with st.expander("Architecture overview", expanded=True):
                st.graphviz_chart(dot, use_container_width=True)
        else:
            st.caption("No component relationships are recorded for this design yet.")
    else:
        selected = st.radio(
            "Flow filter", FLOW_FILTERS, horizontal=True,
            key=f"flow_filter_{state.run_id}",
        )
        shown = filter_edges_by_category(edges, selected)
        st.caption(f"{selected}: {len(shown)} of {len(edges)} flow(s)")
        if shown:
            with st.expander("Architecture diagram", expanded=True):
                st.graphviz_chart(
                    architecture_flow_dot(shown, component_names),
                    use_container_width=True,
                )
        else:
            st.caption(f"No '{selected}' flows are recorded for this design.")

    if unparsed:
        _bullets("Also recorded, without a direction to draw", unparsed)
    # The verbatim data-flow list and the full technical view are NOT
    # duplicated here — both already render in full on the dedicated
    # Recommended Architecture screen (`render_blueprint`).


# C. is capped at a handful of reasons on purpose: a list of twelve is a
# list nobody reads, and the full reasoning is one tab away in Architecture.
_WHY_MAX_REASONS = 5


def render_why_architecture(state: ArchitectState) -> None:
    """C. Why THIS design: the Blueprint's rationale and each ADR's own
    rationale, in FULL, deduplicated, capped in COUNT.

    Verbatim artifact text — a client-facing result must not end an argument
    mid-sentence with an ellipsis, so a selected reason is rendered whole and
    simply wraps. The cap stays on the NUMBER of reasons (a list of twelve
    is a list nobody reads); generic architecture benefits are never
    invented, and fewer grounded reasons means fewer bullets.
    """
    candidates: list[tuple[str, str]] = []  # (attribution tag, text)
    if state.blueprint is not None and state.blueprint.rationale.strip():
        candidates.append(("", state.blueprint.rationale.strip()))
    for adr in state.adrs:
        if adr.rationale.strip():
            candidates.append((adr.id, adr.rationale.strip()))

    reasons: list[tuple[str, str]] = []
    seen: set[str] = set()
    for tag, text in candidates:
        key = text.lower()[:80]  # dedupe on the opening words
        if key in seen:
            continue
        seen.add(key)
        reasons.append((tag, text))
        if len(reasons) == _WHY_MAX_REASONS:
            break

    if not reasons:
        _missing(
            "Why this architecture",
            "neither the blueprint nor the ADRs recorded a rationale.",
        )
        return

    st.markdown("#### Why this architecture")
    for tag, text in reasons:
        prefix = f"`{html.escape(tag)}` &nbsp; " if tag else ""
        st.markdown(f"- {prefix}{html.escape(text)}")


def render_risks_and_tradeoffs(state: ArchitectState) -> None:
    """G. What the client is accepting, in two grounded groups: the
    design's own open risks, and the trade-off each decision explicitly
    accepted (ADR negative consequences, attributed). Existing text only,
    and deduplicated ACROSS groups so the same sentence never appears
    twice. Open review findings are NOT repeated here — they have their own
    compact count in `render_review_confidence` and their full table on the
    dedicated Validation & Findings screen."""
    blueprint = state.blueprint
    risks = [
        risk.strip()
        for risk in (blueprint.open_risks if blueprint else [])
        if risk.strip()
    ]
    tradeoffs = [
        (adr.id, consequence.strip())
        for adr in state.adrs
        for consequence in adr.negative_consequences
        if consequence.strip()
    ]

    if not risks and not tradeoffs:
        st.caption("No risks or trade-offs were recorded for this design.")
        return

    # One shared seen-set, filled in reading order, so a sentence the
    # blueprint states as a risk is not repeated as a trade-off below it.
    seen: set[str] = set()

    def _first_time(text: str) -> bool:
        key = " ".join(text.lower().split())[:80]
        if key in seen:
            return False
        seen.add(key)
        return True

    st.markdown("#### Risks and trade-offs")
    risks = [risk for risk in risks if _first_time(risk)]
    if risks:
        st.markdown("**Open risks**")
        st.markdown("\n".join(f"- {html.escape(risk)}" for risk in risks))
    tradeoffs = [(adr_id, text) for adr_id, text in tradeoffs if _first_time(text)]
    if tradeoffs:
        st.markdown("**Trade-offs the decisions accepted**")
        st.markdown(
            "\n".join(
                f"- `{html.escape(adr_id)}` {html.escape(text)}"
                for adr_id, text in tradeoffs
            )
        )
    # Individual findings are NOT listed here — that table already exists,
    # in full, on the dedicated Validation & Findings screen. This section
    # stays about the design's own risks and each decision's trade-offs;
    # `render_review_confidence` carries the compact open-findings count.


def render_migration_approach(state: ArchitectState) -> None:
    """The compact client cut of `blueprint.migration_steps`: number, title,
    one-line objective per step.

    Silent when the run carries no structured steps (greenfield, or any
    checkpoint written before the field existed) — an absent migration plan
    is a valid state, not an empty section. The full fields stay in the
    Architecture view; prose mentions of "incremental" migration in
    rationale text are NOT rendered as a sequence.
    """
    steps = state.blueprint.migration_steps if state.blueprint else []
    if not steps:
        return

    st.markdown("#### Migration approach")
    for index, step in enumerate(steps, start=1):
        title = html.escape(step.title.strip() or f"Step {index}")
        st.markdown(f"{index}. **{title}**")
        if step.objective.strip():
            st.caption(_clip_sentence(step.objective, 160))


def render_migration_steps(state: ArchitectState) -> None:
    """The full ordered migration sequence, every structured field — the
    Architecture view's rendering. Same silence rule as the compact cut."""
    steps = state.blueprint.migration_steps if state.blueprint else []
    if not steps:
        return

    with st.expander(
        f"🧭  Migration approach — {len(steps)} step"
        f"{'s' if len(steps) != 1 else ''}", expanded=False
    ):
        st.caption(
            "The ordered modernization sequence from the Blueprint — "
            "architecture-level steps, coexistence and data strategy, and "
            "each step's validation."
        )
        for index, step in enumerate(steps, start=1):
            title = step.title.strip() or f"Step {index}"
            st.markdown(
                f"{_tag(f'{index:02d}. {title}', BCG_DARK)}",
                unsafe_allow_html=True,
            )
            _text("Objective", step.objective)
            _bullets("Changes", step.changes)
            _text("Coexistence / data strategy",
                  step.coexistence_or_data_strategy)
            _text("Exit condition", step.exit_condition)
            st.divider()


def render_review_confidence(state: ArchitectState) -> None:
    """H. How much to trust the page above: verdict, open findings,
    refinement effort, acceptance — the four facts a client asks about.

    Compact by design; the rubric that produced the verdict stays in the
    Review view. Same honesty rule as `render_run_status`: never a bare
    "passed" without the findings that are still open."""
    st.markdown("#### Validation status")
    review = state.review
    if review is None:
        st.markdown(
            f"{_tag('NOT REVIEWED', _SEVERITY_COLORS['medium'])} &nbsp; "
            "The design never reached the quality gate.",
            unsafe_allow_html=True,
        )
        return

    verdict = "REVIEW PASSED" if review.overall_status == "pass" else "REVIEW FAILED"
    color = BCG_GREEN if review.overall_status == "pass" else RED
    open_count = len(review.issues)
    parts = [
        f"{open_count} open finding{'s' if open_count != 1 else ''}",
        (
            f"{state.refine_iterations} refinement round"
            f"{'s' if state.refine_iterations != 1 else ''}"
            if state.refine_iterations
            else "no refinement rounds"
        ),
    ]
    if state.stopped_on_cap:
        parts.append("stopped on the cost cap")
    if state.accepted_at:
        parts.append(f"accepted {_datestamp(state.accepted_at)}")
    elif state.stage is Stage.DONE:
        parts.append("not yet accepted")

    st.markdown(
        f"{_tag(verdict, color)} &nbsp; "
        + " · ".join(html.escape(part) for part in parts),
        unsafe_allow_html=True,
    )
