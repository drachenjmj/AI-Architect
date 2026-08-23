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
