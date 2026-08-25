"""test_ui_overview_review.py — tests for the rebuilt client-facing Overview
and the restructured Review view.

Scope: the CONTENT promises of the workspace UX pass —

  * Overview answers the six client questions from EXISTING artifacts only
    (executive recommendation, target architecture, why, decisions,
    components, risks/trade-offs, review confidence), with no technical
    run internals anywhere on the page;
  * the Migration Approach section appears exactly when the Blueprint
    carries STRUCTURED `migration_steps` — prose mentions alone
    ("incrementally migrating…") never conjure it up;
  * Review exposes verdict, counts and the dimension summary directly,
    with only detailed evidence left collapsible.

Runs against the real ui.py with the offline demo states ("pass" for the
client view, "capped" for a failing review). No network, no LLM, no live
pipeline. Never `import ui` (module-level st calls — see
test_ui_workspace.py's docstring).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from webapp.ui_demo import build_demo_state

_UI = str(Path(__file__).resolve().parents[2] / "ui.py")  # repo root


@pytest.fixture()
def no_pipeline_actions(monkeypatch):
    """Fail the test if any pipeline/API/storage action fires while the
    Overview renders. Same patches as test_ui_workspace's fixture."""

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003 — any call is the bug
        raise AssertionError("a pipeline/API/storage action was triggered")

    monkeypatch.setattr("pipeline.orchestrator.run_pipeline_streaming", _boom)
    monkeypatch.setattr("pipeline.persistence.list_runs", _boom)
    monkeypatch.setattr("pipeline.persistence.load_state", _boom)
    monkeypatch.setattr("pipeline.persistence.save_state", _boom)
    monkeypatch.setattr("pipeline.agents.clarifier.ask_advisor", _boom)
    monkeypatch.setattr("pipeline.user_feedback.submit_feedback", _boom)
    monkeypatch.setattr("pipeline.sign_off.accept_design", _boom)
    return _boom


def _finished_app(variant: str = "pass") -> AppTest:
    at = AppTest.from_file(_UI, default_timeout=30)
    at.session_state["state"] = build_demo_state(variant)
    at.run()
    assert not at.exception
    return at


def _select_view(at: AppTest, view: str) -> AppTest:
    """`view` is the internal identifier, matched by the button's stable
    `nav_{view}` key rather than its (renamed) human-facing label."""
    next(b for b in at.sidebar.button if b.key == f"nav_{view}").click()
    at.run()
    assert not at.exception
    return at


def _md(at: AppTest) -> str:
    return " | ".join(m.value for m in at.markdown)


def _caps(at: AppTest) -> str:
    return " | ".join(c.value for c in at.caption)


# ── 1. Executive recommendation (section A) ────────────────────────────────


def test_overview_executive_recommendation_assembles_existing_fields():
    at = _finished_app()
    md = _md(at)

    assert "Executive recommendation" in md
    # The Blueprint's own pattern, as the headline.
    assert "Event-driven services behind a managed queue" in md
    # The Context Record's goal and problem, framing the why.
    assert "**Goal**" in md
    assert "Capture seasonal peak revenue instead of losing it to outages." in md
    assert "**Problem**" in md
    assert "Synchronous order processing saturates the shop deployable." in md
    # The Blueprint's stakeholder view, verbatim.
    assert "Customers can complete purchases during seasonal peaks." in md


# ── 2. Target architecture (section B) ─────────────────────────────────────


def _expander(at: AppTest, needle: str):
    return next(e for e in at.expander if needle in e.label)


def test_overview_section_order_is_the_accepted_sequence():
    """The client scans recommendation → reasoning → decisions →
    components → risks → verdict → the large diagram → sign-off. The
    diagram must NOT dominate the first screenful, and the sign-off stays
    below it (it acts on the whole picture)."""
    at = _finished_app()
    md = _md(at)

    positions = {
        "Executive recommendation": md.index("Executive recommendation"),
        "Why this architecture": md.index("Why this architecture"),
        "Key decisions": md.index("Key decisions"),
        "Key components": md.index("Key components"),
        "Risks and trade-offs": md.index("Risks and trade-offs"),
        "Validation status": md.index("Validation status"),
        "Target architecture": md.index("#### Target architecture"),
        "Take this design": md.index("Take this design"),
    }
    order = list(positions.values())
    assert order == sorted(order), positions   # strictly the accepted order
    # The explicit contracts of this pass, spelled out:
    assert positions["Target architecture"] > positions["Validation status"]
    assert positions["Target architecture"] < positions["Take this design"]


def test_overview_target_architecture_sits_at_the_bottom_expanded():
    at = _finished_app()

    assert "Target architecture" in _md(at)
    # The diagram view defaults to "Overview" (the
    # calm, deduplicated graph); its accordion exists and is EXPANDED the
    # moment the page opens — see `render_target_architecture`.
    diagram = _expander(at, "Architecture overview")
    assert diagram.proto.expanded is True
    # The verbatim data-flow list and the full technical view are NOT
    # duplicated on Overview any more — both live in full on the dedicated
    # Recommended Architecture screen (see the next test). Neither detail
    # accordion exists here.
    assert not any("All data flows" in e.label for e in at.expander)
    assert not any("Technical view" in e.label for e in at.expander)
    assert "Checkout Service → SQS: order placement message (EU region)" not in _md(at)


def test_architecture_view_still_carries_the_verbatim_data_flows():
    """The full detail Overview no longer duplicates stays reachable, in
    full, on the dedicated Recommended Architecture screen."""
    at = _select_view(_finished_app(), "Architecture")
    md = _md(at)

    assert "Checkout Service → SQS: order placement message (EU region)" in md
    assert "Technical view" in md


def test_architecture_flow_dot_builds_from_the_saved_data_flows():
    """The diagram's nodes and connections come from `blueprint.data_flows`
    and `components` — nothing else. Pure function, no state, no calls."""
    from webapp.ui_sections import architecture_flow_dot, parse_data_flows

    state = build_demo_state("pass")
    edges, unparsed = parse_data_flows(state.blueprint.data_flows)

    assert not unparsed                      # the demo flows all parse
    assert len(edges) == 5                   # one edge per saved flow
    assert ("Customer", "Checkout Service", "cart submission over HTTPS") in edges
    assert ("Order Worker", "notification service",
            "asynchronous confirmation mail") in edges

    dot = architecture_flow_dot(edges, [c.name for c in state.components])
    # Top-to-bottom layout.
    assert "rankdir=TB" in dot
    # Every flow's endpoints and its description are in the diagram…
    for node in ("Customer", "Checkout Service", "SQS", "Order Worker",
                 "PostgreSQL", "notification service"):
        assert f'"{node}"' in dot, node
    assert 'label="cart submission over HTTPS"' in dot
    assert '"Checkout Service" -> "SQS"' in dot
    assert '"Order Worker" -> "notification service"' in dot
    # …owned components draw as the owned style, everything else as external.
    owned_line = next(line for line in dot.splitlines()
                      if line.strip().startswith('"Checkout Service"'))
    external_line = next(line for line in dot.splitlines()
                         if line.strip().startswith('"notification service"'))
    assert "fillcolor=\"#EFF6F2\"" in owned_line      # owned: solid brand box
    assert "dashed" in external_line                   # external: greyed


def test_unparseable_flow_stays_visible_text_not_a_guess():
    from webapp.ui_sections import parse_data_flows

    edges, unparsed = parse_data_flows(
        [
            "Customer → Checkout Service: cart submission over HTTPS",
            "Data is replicated continuously across regions",
        ]
    )
    assert edges == [("Customer", "Checkout Service",
                      "cart submission over HTTPS")]
    assert unparsed == ["Data is replicated continuously across regions"]


def test_overview_renders_an_unparseable_flow_as_text_below_the_diagram():
    at = AppTest.from_file(_UI, default_timeout=30)
    state = build_demo_state("pass")
    state.blueprint.data_flows.append("Data is replicated continuously")
    at.session_state["state"] = state
    at.run()
    assert not at.exception

    # The flow that cannot be drawn is still on the page, verbatim.
    assert "Data is replicated continuously" in _md(at)


def test_rendering_the_diagram_calls_nothing(no_pipeline_actions):
    """Rendering the visual architecture is pure string work — no LLM, no
    API, no pipeline, no retrieval. The fixture booby-traps every path."""
    at = AppTest.from_file(_UI, default_timeout=30)
    at.session_state["state"] = build_demo_state("pass")
    at.run()
    assert not at.exception


# ── 3. Why this architecture (section C) ──────────────────────────────────


def test_overview_why_uses_existing_artifact_text_only():
    at = _finished_app()
    md = _md(at)

    assert "Why this architecture" in md
    # Verbatim rationale text from the artifacts, attributed.
    assert "A queue isolates peak traffic while retaining an incremental migration." in md
    assert "Encryption and EU residency protect personal order data." in md
    assert "ADR-001" in md  # attribution tags


def test_overview_why_never_truncates_a_selected_reason():
    """A client-facing argument must not end mid-sentence: the selected
    Blueprint rationale renders IN FULL, all the way to its last word."""
    at = _finished_app()
    md = _md(at)

    # The demo rationale's ending — the part the old clip cut off.
    assert "Managed services keep it inside the medium budget." in md
    # And no selected reason ends in an ellipsis (each bullet is its own
    # markdown element, so check elements rather than joined lines).
    why_bullets = [
        m.value for m in at.markdown
        if m.value.startswith("- ") and "…" in m.value
    ]
    assert not why_bullets, why_bullets


def test_overview_why_caps_the_reason_list():
    """A list of twelve reasons is a list nobody reads — the section is
    capped even for a run with many ADRs."""
    at = AppTest.from_file(_UI, default_timeout=30)
    state = build_demo_state("pass")
    # Pad with extra ADRs carrying distinct rationales.
    from pipeline.state import ADR

    for i in range(3, 9):
        state.adrs.append(
            ADR(id=f"ADR-{i:03d}", title=f"ADR-{i}: filler", decision="x",
                rationale=f"Filler rationale number {i} for the cap test.")
        )
    at.session_state["state"] = state
    at.run()
    assert not at.exception

    # Each Why reason is its own markdown bullet element.
    bullets = [
        m.value for m in at.markdown
        if m.value.startswith("- ")
        and any(
            needle in m.value
            for needle in (
                "Filler rationale", "queue isolates",
                "Encryption and EU residency",
                "repository analysis shows",
            )
        )
    ]
    assert 0 < len(bullets) <= 5
    assert not any("Filler rationale number 8" in b for b in bullets)  # cap bit


def test_key_decisions_cut_only_on_a_sentence_boundary():
    at = AppTest.from_file(_UI, default_timeout=30)
    state = build_demo_state("pass")
    # A long multi-sentence decision: the compact line must show the first
    # COMPLETE sentence, with no mid-sentence ellipsis.
    state.adrs[0].decision = (
        "Extract checkout incrementally and buffer orders with SQS. "
        "The queue isolates peak traffic, lets the web tier scale "
        "horizontally, and keeps the migration inside the medium budget "
        "by reusing managed services throughout."
    )
    at.session_state["state"] = state
    at.run()
    assert not at.exception

    caps = _caps(at)
    assert "Extract checkout incrementally and buffer orders with SQS." in caps
    assert "…" not in caps.split("Key decisions")[-1].split("Key components")[0]


# ── 4. Migration approach (section D) — omitted unless explicit ───────────


def test_overview_shows_no_migration_section_from_prose_mentions():
    """The demo blueprint's technical view explicitly says "incrementally
    migrating the legacy shop monolith" — prose, not a sequence. Rendering
    a Migration Approach from that would be inference, so the section must
    stay absent. The prose mention itself no longer renders verbatim on
    Overview at all (that full-detail view moved to Architecture — see
    `test_architecture_view_still_carries_the_verbatim_data_flows`'s
    neighbourhood), so it is checked directly against the artifact."""
    at = _finished_app()

    state = at.session_state["state"]
    assert "incrementally migrating" in state.blueprint.technical_view
    assert "Migration" not in _md(at)            # no section was inferred


# ── 5. Key decisions / components (sections E, F) ─────────────────────────


def test_overview_key_decisions_come_from_saved_adrs():
    at = _finished_app()
    md = _md(at)

    assert "Key decisions" in md
    assert "ADR-1: Buffer checkout work with SQS" in md
    assert "ADR-2: Protect order data in Order Worker" in md
    # Compact: no full ADR bodies on the client page.
    assert "Alternatives considered" not in md


def test_overview_key_components_come_from_saved_components():
    at = _finished_app()
    md = _md(at)

    assert "Key components" in md
    assert "Checkout Service" in md and "Order Worker" in md
    # Compact cards, not full Component Descriptions.
    assert "Security considerations" not in md


# ── 6. Risks and trade-offs (section G) ───────────────────────────────────


def test_overview_risks_and_tradeoffs_split_the_grounding_sources():
    at = _finished_app()
    md = _md(at)

    assert "Risks and trade-offs" in md
    # The Blueprint's own open risks.
    assert "Open risks" in md
    assert "queue depth needs alerting" in md.lower()
    # Each decision's explicitly accepted trade-off, attributed.
    assert "Trade-offs the decisions accepted" in md
    assert "Order confirmation becomes eventually consistent." in md
    assert "Regional operations overhead" in md


def test_overview_no_longer_lists_individual_review_findings():
    """The Risks and trade-offs section no longer repeats the browsable
    findings list (that table lives, in full, on Validation & Findings —
    see test_review_of_a_failed_run_shows_dimensions_and_findings). Section
    H keeps the compact count, with a link to that screen.

    NOTE: `render_sign_off`'s own pre-acceptance disclosure ("you would be
    accepting N open findings") still lists each finding, further down —
    that is not the browsing duplicate this section removed; it is the
    thing a person must see immediately above the button that accepts it."""
    at = _finished_app("capped")  # the failing variant: 2 open issues
    md = _md(at)

    assert "Open review findings" not in md
    # The compact count is still here…
    assert "2 open findings" in md
    # …with a link to the screen that carries the full table.
    assert any(
        "validation & findings" in b.label.lower() for b in at.button
    )


# ── 7. Validation status (section H) ───────────────────────────────────────


def test_overview_review_confidence_is_compact_and_honest():
    at = _finished_app()
    md = _md(at)

    assert "Validation status" in md
    assert "REVIEW PASSED" in md
    assert "0 open findings" in md
    assert "1 refinement round" in md
    assert "not yet accepted" in md


def test_overview_review_confidence_reports_a_failed_review():
    at = _finished_app("capped")
    md = _md(at)

    assert "REVIEW FAILED" in md
    assert "2 open findings" in md
    assert "stopped on the cost cap" in md


# ── 8. No technical run internals anywhere on Overview ────────────────────


def test_overview_carries_no_run_internals():
    at = _finished_app()
    md, caps = _md(at), _caps(at)

    assert len(at.metric) == 0
    assert "Tokens" not in md + caps
    assert "Run trace" not in [e.label for e in at.expander]
    # The client action is still here.
    assert any("Accept this design" in b.label for b in at.button)


# ── 9. Review view: dimensions visible, evidence collapsible ──────────────


def test_review_dimensions_visible_without_opening_anything():
    at = _select_view(_finished_app(), "Review")
    md = _md(at)

    assert "Review dimensions" in md
    # The ACTUAL check names of the existing rubric, both columns.
    for label in (
        "All artifacts present", "Constraint coverage", "Traceability",
        "ADR presence", "Source integrity",
        "Repo grounding", "Flaw detection", "ADR soundness",
        "Best-practice grounding", "Refinement readiness",
    ):
        assert label in md


def test_review_detailed_evidence_stays_collapsible():
    at = _select_view(_finished_app(), "Review")
    labels = [e.label for e in at.expander]

    assert any("Detailed reviewer evidence" in label for label in labels)
    assert not any("Rubric detail" in label for label in labels)


def test_review_of_a_failed_run_shows_dimensions_and_findings():
    at = _select_view(_finished_app("capped"), "Review")
    md = _md(at)

    assert "REVIEW — FAIL" in md
    assert "2 blocking issues" in md
    assert "Review dimensions" in md
    assert len(at.table) == 1  # the findings table itself


# ── 10. Migration approach — structured steps, client cut ─────────────────


def _with_steps(variant: str = "pass"):
    """A demo state carrying a structured migration sequence (GENERIC
    placeholder content — no benchmark-specific names)."""
    from pipeline.state import MigrationStep

    state = build_demo_state(variant)
    state.blueprint.migration_steps = [
        MigrationStep(
            title="Prepare the order-processing seam",
            objective="Isolate the boundary the first extraction will use.",
            changes=["Introduce an interface seam at the module edge"],
            coexistence_or_data_strategy="Seam routes both paths during transition.",
            exit_condition="Seam traffic observable.",
        ),
        MigrationStep(
            title="Extract the first capability",
            objective="Move the highest-value, lowest-risk capability first.",
        ),
    ]
    return state


def test_overview_shows_migration_approach_when_steps_exist():
    at = AppTest.from_file(_UI, default_timeout=30)
    at.session_state["state"] = _with_steps()
    at.run()
    assert not at.exception
    md = _md(at)

    assert "Migration approach" in md
    assert "Prepare the order-processing seam" in md
    assert "Extract the first capability" in md
    # Compact: number + title + one-line objective, not every field.
    assert "Coexistence" not in md.split("Key decisions")[0]
    assert "Exit condition" not in md


def test_overview_omits_migration_section_cleanly_when_no_steps():
    at = _finished_app()          # the demo carries no structured steps

    assert "Migration approach" not in _md(at)


def test_overview_migration_sits_between_why_and_key_decisions():
    at = AppTest.from_file(_UI, default_timeout=30)
    at.session_state["state"] = _with_steps()
    at.run()
    md = _md(at)

    why = md.index("Why this architecture")
    migration = md.index("Migration approach")
    decisions = md.index("Key decisions")
    assert why < migration < decisions


def test_architecture_view_shows_the_full_migration_steps():
    at = AppTest.from_file(_UI, default_timeout=30)
    at.session_state["state"] = _with_steps()
    at.run()
    at = _select_view(at, "Architecture")

    labels = [e.label for e in at.expander]
    assert any(
        "Migration approach — 2 steps" in label for label in labels
    )
    # Full detail renders inside the expander's block, which AppTest
    # traverses regardless of open state.
    assert not at.exception


def test_architecture_view_omits_migration_expander_when_no_steps():
    at = _select_view(_finished_app(), "Architecture")

    assert not any(
        "Migration approach" in e.label for e in at.expander
    )


def test_rendering_migration_sections_calls_nothing(no_pipeline_actions):
    """Rendering with steps on screen is pure drawing — no LLM, no
    pipeline, no retrieval (the fixture booby-traps every path)."""
    at = AppTest.from_file(_UI, default_timeout=30)
    at.session_state["state"] = _with_steps()
    at.run()
    assert not at.exception
    _select_view(at, "Architecture")
    assert not at.exception
