"""test_diagram_overview_filters.py — Architecture Overview + Detailed flow
filters (Kush).

Covers `pipeline.flow_syntax.classify_flow` and the pure diagram helpers in
`ui_sections.py` (`build_overview_edges`, `build_overview_edges_from_flows`,
`filter_edges_by_category`) that back the diagram's Overview/Detailed radio
and the Detailed-view Business/Events/Data/Observability filter.

A DENSE, GENERIC fixture (no product/domain vocabulary) is used throughout —
never the e-commerce demo output — so nothing here doubles as hard-coded
product logic. Component/flow names below are deliberately generic
("Gateway", "Ledger Service", "Metrics Sink", ...).
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from pipeline.flow_syntax import classify_flow, split_directional_flow
from pipeline.state import ComponentDescription
from webapp.ui_demo import build_demo_state
from webapp.ui_sections import (
    FLOW_FILTERS,
    architecture_flow_dot,
    build_overview_edges,
    build_overview_edges_from_flows,
    filter_edges_by_category,
    parse_data_flows,
)

_UI = str(Path(__file__).resolve().parents[1] / "ui.py")  # repo root


# ── classify_flow ───────────────────────────────────────────────────────

def test_events_flow_classifies_as_events():
    assert classify_flow("Order Gateway", "Notification Worker",
                          "publishes an OrderPlaced event to the message bus") == "events"


def test_data_flow_classifies_as_data():
    assert classify_flow("Ledger Service", "Ledger Database",
                          "writes and queries account records") == "data"


def test_observability_flow_classifies_as_observability():
    assert classify_flow("Ledger Service", "Metrics Sink",
                          "emits latency metrics and traces") == "observability"


def test_unclassified_flow_falls_back_to_business():
    assert classify_flow("Client", "Gateway", "submits a request") == "business"


def test_observability_takes_precedence_over_events():
    """An edge naming both an event-bus concern and a metrics concern is
    classified as observability first — the documented precedence in
    `pipeline.flow_syntax._CATEGORY_STEMS`."""
    result = classify_flow("Gateway", "Metrics Dashboard",
                            "publishes request metrics to the event bus")
    assert result == "observability"


def test_classification_is_deterministic_and_generic():
    """Same input -> same output, and nothing in the stems names a
    domain/product (scanned on the actual keyword tables, not behaviour)."""
    import pipeline.flow_syntax as fs

    a = classify_flow("A", "B", "replicates data")
    b = classify_flow("A", "B", "replicates data")
    assert a == b == "data"

    forbidden = ["e-commerce", "ecommerce", "shopping", "checkout", "cart", "order"]
    all_stems = " ".join(
        stem
        for _cat, stems in fs._CATEGORY_STEMS
        for stem in stems
    ).lower()
    for term in forbidden:
        assert term not in all_stems


# ── dense generic fixture ───────────────────────────────────────────────

def _dense_components() -> list[ComponentDescription]:
    """15 components, each with 2-4 dependencies — a generic stand-in for
    the holdout run's ~15-component design, with no product vocabulary."""
    names = [f"Service {letter}" for letter in "ABCDEFGHIJKLMNO"]
    components = []
    for index, name in enumerate(names):
        deps = [names[(index + 1) % len(names)], names[(index + 2) % len(names)]]
        if index % 3 == 0:
            deps.append("Shared Database")
        components.append(
            ComponentDescription(
                id=f"COMP-{index:03d}", name=name, purpose="p", description="d",
                dependencies=deps,
            )
        )
    return components


def _dense_flows() -> list[str]:
    """~44 flows across all four categories, several sharing endpoints under
    different labels (so Overview's dedup has real work to do)."""
    flows = []
    names = [f"Service {letter}" for letter in "ABCDEFGHIJKLMNO"]
    for index, name in enumerate(names):
        nxt = names[(index + 1) % len(names)]
        flows.append(f"{name} → {nxt}: forwards a business request")
        flows.append(f"{name} → {nxt}: publishes a completion event")
        flows.append(f"{name} → Shared Database: persists and queries state")
    flows.append("Client → Service A: submits a request")
    flows.append("Service A → Metrics Sink: emits latency metrics and traces")
    return flows


def test_dense_fixture_has_far_more_detailed_flows_than_overview_edges():
    components = _dense_components()
    flows = _dense_flows()
    edges, _unparsed = parse_data_flows(flows)

    overview_edges = build_overview_edges(components)

    assert len(edges) >= 40  # dense, matches the holdout-run order of magnitude
    assert len(overview_edges) < len(edges)  # Overview is materially smaller


# ── build_overview_edges ────────────────────────────────────────────────

def test_overview_edges_come_from_declared_dependencies_deduplicated():
    components = [
        ComponentDescription(id="C1", name="A", purpose="p", description="d",
                              dependencies=["B", "B", "C"]),
        ComponentDescription(id="C2", name="B", purpose="p", description="d",
                              dependencies=["C"]),
    ]
    edges = build_overview_edges(components)
    assert edges == [("A", "B"), ("A", "C"), ("B", "C")]


def test_overview_edges_drop_self_references():
    components = [
        ComponentDescription(id="C1", name="A", purpose="p", description="d",
                              dependencies=["A", "B"]),
    ]
    assert build_overview_edges(components) == [("A", "B")]


def test_overview_edges_preserve_external_dependencies():
    """A dependency naming something outside the component catalog (a queue,
    a database) is kept — Overview must not silently drop external seams."""
    components = [
        ComponentDescription(id="C1", name="A", purpose="p", description="d",
                              dependencies=["Message Queue"]),
    ]
    assert build_overview_edges(components) == [("A", "Message Queue")]


def test_overview_never_invents_a_relationship_not_in_the_source_data():
    components = _dense_components()
    overview_edges = set(build_overview_edges(components))
    declared = {
        (component.name, dep.strip())
        for component in components
        for dep in component.dependencies
        if dep.strip() and dep.strip() != component.name
    }
    assert overview_edges <= declared


def test_overview_falls_back_to_collapsed_data_flows_with_no_dependencies():
    """When no component declares a dependency, Overview derives a minimal
    structural edge set from data flows instead — still deduplicated,
    still never inventing an endpoint."""
    components = [
        ComponentDescription(id="C1", name="A", purpose="p", description="d"),
        ComponentDescription(id="C2", name="B", purpose="p", description="d"),
    ]
    assert build_overview_edges(components) == []  # no dependencies declared

    edges, _unparsed = parse_data_flows([
        "A → B: first message",
        "A → B: second message",  # same pair, different label
        "B → A: reply",
    ])
    fallback = build_overview_edges_from_flows(edges)
    assert fallback == [("A", "B"), ("B", "A")]  # collapsed, order preserved


# ── Detailed-view filter ────────────────────────────────────────────────

def test_all_filter_preserves_every_detailed_flow():
    edges, _unparsed = parse_data_flows(_dense_flows())
    assert filter_edges_by_category(edges, "All") == edges


def test_each_filter_is_deterministic_and_a_subset_of_all():
    edges, _unparsed = parse_data_flows(_dense_flows())
    for category in FLOW_FILTERS:
        first = filter_edges_by_category(edges, category)
        second = filter_edges_by_category(edges, category)
        assert first == second
        assert set(first) <= set(edges)


def test_filters_partition_every_edge_into_exactly_one_category():
    """Every edge in `All` is classified into exactly one of the four
    named buckets — the union of the four non-'All' filters reconstructs
    the full detailed set with no flow lost and none duplicated."""
    edges, _unparsed = parse_data_flows(_dense_flows())
    buckets = [filter_edges_by_category(edges, cat) for cat in FLOW_FILTERS if cat != "All"]
    total = sum(len(bucket) for bucket in buckets)
    assert total == len(edges)
    union: set[tuple[str, str, str]] = set()
    for bucket in buckets:
        union.update(bucket)
    assert union == set(edges)


def test_events_and_data_and_observability_filters_pick_up_the_right_edges():
    flows = [
        "A → B: publishes an OrderPlaced event",
        "A → B: writes to the database",
        "A → B: emits latency metrics",
        "A → B: submits a plain request",
    ]
    edges, _unparsed = parse_data_flows(flows)

    assert len(filter_edges_by_category(edges, "Events")) == 1
    assert len(filter_edges_by_category(edges, "Data")) == 1
    assert len(filter_edges_by_category(edges, "Observability")) == 1
    assert len(filter_edges_by_category(edges, "Business")) == 1


def test_unknown_flow_falls_back_to_business_filter():
    edges, _unparsed = parse_data_flows(["A → B: does a generic thing"])
    assert filter_edges_by_category(edges, "Business") == edges
    assert filter_edges_by_category(edges, "Events") == []


def test_empty_filter_state_is_a_plain_empty_list_not_an_error():
    edges, _unparsed = parse_data_flows(["A → B: submits a request"])
    assert filter_edges_by_category(edges, "Observability") == []


def test_unparsed_flows_are_unaffected_by_any_filter():
    """A filter narrows what is DRAWN, never what is RECORDED — prose flows
    stay in `unparsed` regardless of the Detailed-view filter selection."""
    flows = [
        "A → B: publishes an event",
        "Data is replicated continuously across regions",
    ]
    edges, unparsed = parse_data_flows(flows)
    assert unparsed == ["Data is replicated continuously across regions"]
    for category in FLOW_FILTERS:
        # The filter operates only on `edges`; `unparsed` has no filter
        # concept at all, which is the point — it never disappears.
        filter_edges_by_category(edges, category)
    assert unparsed == ["Data is replicated continuously across regions"]


def test_filtered_edges_still_render_as_valid_dot():
    edges, _unparsed = parse_data_flows(_dense_flows())
    shown = filter_edges_by_category(edges, "Events")
    dot = architecture_flow_dot(shown, [f"Service {c}" for c in "ABCDEFGHIJKLMNO"])
    assert dot.startswith("digraph architecture {")
    assert dot.strip().endswith("}")


# ── wired into the real UI (AppTest) ────────────────────────────────────

def _finished_app() -> AppTest:
    at = AppTest.from_file(_UI, default_timeout=30)
    at.session_state["state"] = build_demo_state("pass")
    at.run()
    assert not at.exception
    return at


def _diagram_radio(at: AppTest, label: str):
    return next(r for r in at.radio if r.label == label)


def test_overview_is_the_default_diagram_view():
    at = _finished_app()
    view = _diagram_radio(at, "Diagram view")
    assert view.value == "Overview"
    assert any("Architecture overview" in e.label for e in at.expander)
    assert not any(e.label == "Architecture diagram" for e in at.expander)


def test_switching_to_detailed_exposes_the_flow_filter_and_all_matches_full_set():
    from webapp.ui_sections import parse_data_flows as _parse

    at = _finished_app()
    state = at.session_state["state"]
    total_flows = len(_parse(state.blueprint.data_flows)[0])

    _diagram_radio(at, "Diagram view").set_value("Detailed").run()
    assert not at.exception

    filter_radio = _diagram_radio(at, "Flow filter")
    assert list(filter_radio.options) == list(FLOW_FILTERS)
    assert filter_radio.value == "All"
    caption_text = " | ".join(c.value for c in at.caption)
    assert f"All: {total_flows} of {total_flows} flow(s)" in caption_text
