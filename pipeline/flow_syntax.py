"""flow_syntax.py — the ONE directional data-flow syntax shared by the UI
diagram renderer and the deterministic Reviewer checks.

Extracted verbatim from the renderer's parser (ui_sections.parse_data_flows)
so the judge and the renderer can never disagree about what a renderable
flow IS: a flow is `A → B` (the ASCII `->` spelling accepted), optionally a
chain `A → B → C`, optionally followed by `: label`. Anything else — prose,
headings, half a sentence — is not directional and carries no endpoints.

Pure string handling only. No LLM, no network, no state.
"""
from __future__ import annotations

import re

# The arrow spellings a data flow may use. Kept as the single compiled
# pattern for both consumers; adding a spelling here changes the renderer
# and the Reviewer in the same breath.
FLOW_ARROW_RE = re.compile(r"\s*(?:→|->)\s*")


FlowCategory = str  # "business" | "events" | "data" | "observability"

# ─────────────────────────────────────────────────────────────────────────
# Integration note (Kush): display-only flow classification, for the
# Architecture diagram's Detailed-view filter (see ui_sections.py
# render_target_architecture). ONE central, generic, deterministic rule —
# never a second parsing grammar: classification runs on the ALREADY-PARSED
# edge (endpoints + label) that `split_directional_flow` produced, so it can
# never disagree with what the renderer or the Reviewer accept as a flow.
#
# Classification is DISPLAY-ONLY. It never touches ArchitectureDesign or the
# Reviewer's checks — every flow is still validated, stored, and rendered
# under `All` exactly as before; this only decides which of the four
# Detailed-view buckets a flow additionally falls into.
#
# Keyword stems, not whole-word lists — the same "generic substring stem"
# convention the rest of this codebase already uses (see
# architect.py:ARCHITECTURE_KEYWORDS). Nothing here names a domain/product;
# every keyword is a generic architectural/technical term.
# ─────────────────────────────────────────────────────────────────────────

_OBSERVABILITY_STEMS: tuple[str, ...] = (
    "metric", "log", "trace", "telemetry", "monitor", "alert", "dashboard",
    "observab", "audit trail",
)
_EVENTS_STEMS: tuple[str, ...] = (
    "event", "publish", "subscri", "topic", "queue", "broker", "message bus",
    " bus", "emit", "consum", "notif",
)
_DATA_STEMS: tuple[str, ...] = (
    "database", " db", "datastore", "data store", "store", "read", "write",
    "persist", "query", "replicat", "cache", "table", "record",
)

# Checked in this order — deliberately: an edge naming both a metrics
# concern and a messaging one ("publishes latency metrics to the event
# bus") is FIRST an observability signal in this project's sense (what a
# reader clicks to understand "is this system healthy", not "how do
# components talk"), so Observability is checked before Events; Events
# before Data for the same reason (a flow that both emits an event AND
# writes it to a store is, architecturally, the event). Business is
# everything left over — the default bucket, not a keyword match.
_CATEGORY_STEMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("observability", _OBSERVABILITY_STEMS),
    ("events", _EVENTS_STEMS),
    ("data", _DATA_STEMS),
)


def classify_flow(source: str, target: str, label: str) -> FlowCategory:
    """Deterministic display category for ONE parsed edge. Pure.

    Text scanned is the edge's own endpoints and label — a flow's category
    is as often carried by an endpoint name ("Metrics Dashboard",
    "Notification Service") as by its label. Falls back to "business" when
    no stem matches; that is the correct default, not a failure — most
    application-level interactions are exactly that.
    """
    text = f"{source} {target} {label}".lower()
    for category, stems in _CATEGORY_STEMS:
        if any(stem in text for stem in stems):
            return category
    return "business"


def split_directional_flow(flow: str) -> tuple[list[str], str] | None:
    """Parse one data flow into `(endpoints, label)`, or None when the flow
    is not in the directional syntax at all.

    `endpoints` are the stripped arrow-separated participants in order (at
    least two — a lone token is not an edge); `label` is the text after the
    first `:` (the edge description), empty when absent. A non-empty flow
    that returns None here cannot be rendered as a component-to-component
    edge and must not be silently guessed into one.
    """
    text = str(flow or "").strip()
    if not text:
        return None
    label = ""
    if ":" in text:
        text, _, label = text.partition(":")
        label = label.strip()
    if not FLOW_ARROW_RE.search(text):
        return None
    endpoints = [part.strip() for part in FLOW_ARROW_RE.split(text) if part.strip()]
    if len(endpoints) < 2:
        return None
    return endpoints, label
