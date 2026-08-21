"""test_kb_gap_report.py - Offline tests for KB-gap grouping under the new
web-fallback statuses.

Core property under test: a KB gap is any "web_fallback*" record, regardless
of the external fallback's own outcome. A failed or empty web search must not
hide the fact that the internal knowledge base had no answer. Rule-based
only — the report is run against temp JSONL fixtures, no LLM, no network.
"""

from __future__ import annotations

import json

import kb_gap_report


def _record(query, status, ts="2026-08-20T10:00:00"):
    return {
        "timestamp": ts,
        "query": query,
        "duration_ms": 1.0,
        "chunks_returned": 0,
        "best_distance": None,
        "source_files": [],
        "status": status,
    }


def _write_log(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(path)


def test_all_fallback_statuses_count_as_kb_gaps(tmp_path, capsys):
    log = _write_log(tmp_path / "log.jsonl", [
        _record("How do I scale microservices?", "web_fallback"),
        _record("How should we scale microservices?", "web_fallback"),
        _record("GDPR data residency rules", "web_fallback_empty"),
        _record("strangler fig migration", "web_fallback_error",
                ts="2026-08-20T11:00:00"),
        _record("cheap message queues", "web_fallback_disabled",
                ts="2026-08-20T12:00:00"),
    ])

    rc = kb_gap_report.main(["--log", log])

    out = capsys.readouterr().out
    assert rc == 0
    assert "5 web-fallback queries across 4 topics." in out
    # "how ... scale microservices" variants collapse into one topic.
    assert "scale microservices" in out
    assert "strangler fig migration" in out
    assert "cheap message queues" in out


def test_kb_gap_still_counted_when_the_web_fallback_failed(tmp_path, capsys):
    """THE regression: an API error must not erase the KB gap."""
    log = _write_log(tmp_path / "log.jsonl", [
        _record("event sourcing basics", "web_fallback_error"),
    ])

    rc = kb_gap_report.main(["--log", log])

    out = capsys.readouterr().out
    assert rc == 0
    assert "1 web-fallback query across 1 topic." in out
    assert "event sourcing basics" in out


def test_phrasing_variants_group_into_one_topic(tmp_path, capsys):
    log = _write_log(tmp_path / "log.jsonl", [
        _record("How do I scale microservices?", "web_fallback"),
        _record("How should we scale microservices?", "web_fallback_empty"),
        _record("What are the CQRS trade-offs?", "web_fallback"),
    ])

    rc = kb_gap_report.main(["--log", log])

    out = capsys.readouterr().out
    assert rc == 0
    assert "3 web-fallback queries across 2 topics." in out


def test_non_fallback_statuses_are_not_counted(tmp_path, capsys):
    """KB hits and the pre-fallback miss records are not gap entries.

    Every fallback query already logs no_results/below_threshold first;
    counting those would double-report each gap.
    """
    log = _write_log(tmp_path / "log.jsonl", [
        _record("plain kb hit", "success"),
        _record("kb miss before fallback", "no_results"),
        _record("weak kb match", "below_threshold"),
    ])

    rc = kb_gap_report.main(["--log", log])

    out = capsys.readouterr().out
    assert rc == 0
    assert "No web-fallback queries found" in out


def test_threshold_flag_marks_recurring_gaps(tmp_path, capsys):
    log = _write_log(tmp_path / "log.jsonl", [
        _record("How do I scale microservices?", "web_fallback"),
        _record("How should we scale microservices?", "web_fallback_empty"),
    ])

    rc = kb_gap_report.main(["--log", log, "--threshold", "2"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "<- Consider adding to KB box 1/2." in out


def test_unreadable_log_returns_error(tmp_path, capsys):
    rc = kb_gap_report.main(["--log", str(tmp_path / "missing.jsonl")])

    assert rc == 1
    assert "cannot read" in capsys.readouterr().err


def test_malformed_lines_are_skipped(tmp_path, capsys):
    path = tmp_path / "log.jsonl"
    path.write_text(
        "{not json}\n"
        + json.dumps(_record("healthy fallback query", "web_fallback")) + "\n",
        encoding="utf-8",
    )

    rc = kb_gap_report.main(["--log", str(path)])

    out = capsys.readouterr().out
    assert rc == 0
    assert "1 web-fallback query across 1 topic." in out
    assert "healthy fallback query" in out
