"""test_probe_retrieval.py - Offline pytest tests for the raw-distance probe.

Verifies the probe measures RAW Chroma distances: it calls
similarity_search_with_score() directly on the (faked) store, keeps distances
above the current threshold visible, and needs no retrieve_chunks() / web-
fallback call path. Fakes only — no Chroma store, no API key, no network.
"""

from __future__ import annotations

import architect
from tools import probe_retrieval


class FakeDoc:
    def __init__(self, source="kb_doc.pdf"):
        self.page_content = "content"
        self.metadata = {"source": source, "page": 1, "box": 1}


class FakeVectorstore:
    """Chroma stand-in recording every similarity_search_with_score call."""

    def __init__(self, results):
        self.results = results
        self.calls = []

    def similarity_search_with_score(self, query, k=3):
        self.calls.append((query, k))
        return self.results[:k]


def _forbidden(name):
    def _raise(*args, **kwargs):
        raise AssertionError(
            f"probe must not call {name} — raw measurement only"
        )
    return _raise


def test_probe_calls_similarity_search_directly(monkeypatch):
    # If the probe went through retrieve_chunks()/web fallback, these
    # sentinels would explode. Raw measurement only.
    monkeypatch.setattr(architect, "retrieve_chunks", _forbidden("retrieve_chunks"))
    monkeypatch.setattr(architect, "web_search_fallback", _forbidden("web_search_fallback"))
    vs = FakeVectorstore([(FakeDoc("a.pdf"), 0.30), (FakeDoc("b.pdf"), 0.90)])

    row = probe_retrieval.probe_query(vs, "event-driven architecture", k=5)

    assert vs.calls == [("event-driven architecture", 5)]
    assert row["n"] == 2
    assert row["min"] == 0.30
    assert row["max"] == 0.90
    assert abs(row["mean"] - 0.60) < 1e-9
    assert row["sources"] == ["a.pdf", "b.pdf"]


def test_raw_distances_above_threshold_remain_visible():
    row = {
        "query": "unrelated topic",
        "n": 2,
        "min": 0.30,
        "max": 0.90,  # well above DISTANCE_THRESHOLD = 0.65
        "mean": 0.60,
        "sources": ["far.pdf"],
    }

    line = probe_retrieval.format_row(3, row)

    assert "0.9000" in line  # NOT filtered away
    assert "unrelated topic" in line
    assert "n=2" in line


def test_empty_raw_results_are_handled_cleanly():
    vs = FakeVectorstore([])

    row = probe_retrieval.probe_query(vs, "nothing matches", k=5)
    line = probe_retrieval.format_row(1, row)

    assert row["n"] == 0
    assert row["min"] is row["max"] is row["mean"] is None
    assert row["sources"] == []
    assert "nothing matches" in line
    assert "--" in line  # stats render as placeholders, no crash


def test_footer_states_raw_and_unfiltered():
    footer = probe_retrieval.footer_text()

    assert "RAW, unfiltered" in footer
    assert "DISTANCE_THRESHOLD" in footer
    assert "lower distance = better" in footer
