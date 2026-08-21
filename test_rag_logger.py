"""test_rag_logger.py - Offline regression tests for RagLogger.

Uses only tmp_path — never touches the real rag_log.jsonl. Covers the
append-only JSON-Lines contract, the seven persisted fields, round(·,1)
duration rounding, Unicode preservation (ensure_ascii=False), and the
non-crashing OSError path with its stderr diagnostic.
"""

from __future__ import annotations

import json

import pytest

from rag_logger import RagLogger

EXPECTED_FIELDS = {
    "timestamp", "query", "duration_ms", "chunks_returned",
    "best_distance", "source_files", "status",
}


def _read_records(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(ln) for ln in lines]


def test_one_call_appends_exactly_one_valid_jsonl_line(tmp_path):
    log = RagLogger(str(tmp_path / "rag_log.jsonl"))

    log.log_query(
        query="microservices vs monolith",
        duration_ms=12.34,
        chunks_returned=2,
        best_distance=0.41,
        source_files=["a.pdf", "b.pdf"],
        status="success",
    )

    raw = (tmp_path / "rag_log.jsonl").read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert raw.count("\n") == 1  # exactly one line
    rec = _read_records(tmp_path / "rag_log.jsonl")
    assert len(rec) == 1
    assert isinstance(rec[0], dict)


def test_record_contains_expected_fields(tmp_path):
    log = RagLogger(str(tmp_path / "rag_log.jsonl"))

    log.log_query(
        query="api gateway responsibilities",
        duration_ms=5.0,
        chunks_returned=0,
        best_distance=None,
        source_files=[],
        status="web_fallback_empty",
    )

    rec = _read_records(tmp_path / "rag_log.jsonl")[0]
    assert set(rec) == EXPECTED_FIELDS
    assert rec["query"] == "api gateway responsibilities"
    assert rec["chunks_returned"] == 0
    assert rec["best_distance"] is None
    assert rec["source_files"] == []
    assert rec["status"] == "web_fallback_empty"
    # timestamp is an ISO string (parseable, so ordering/debugging works)
    assert isinstance(rec["timestamp"], str) and "T" in rec["timestamp"]


def test_duration_ms_is_rounded_to_one_decimal(tmp_path):
    log = RagLogger(str(tmp_path / "rag_log.jsonl"))

    log.log_query(
        query="q", duration_ms=123.456789, chunks_returned=1,
        best_distance=0.5, source_files=["x.pdf"], status="success",
    )

    rec = _read_records(tmp_path / "rag_log.jsonl")[0]
    assert rec["duration_ms"] == 123.5  # round(123.456789, 1)
    assert isinstance(rec["duration_ms"], float)


def test_multiple_calls_append_not_overwrite(tmp_path):
    log = RagLogger(str(tmp_path / "rag_log.jsonl"))

    for i, status in enumerate(
        ["success", "no_results", "web_fallback", "web_fallback_error"]
    ):
        log.log_query(
            query=f"query-{i}", duration_ms=float(i), chunks_returned=i,
            best_distance=None, source_files=[], status=status,
        )

    recs = _read_records(tmp_path / "rag_log.jsonl")
    assert [r["query"] for r in recs] == [f"query-{i}" for i in range(4)]
    assert [r["status"] for r in recs] == [
        "success", "no_results", "web_fallback", "web_fallback_error",
    ]


def test_unicode_query_and_sources_are_preserved(tmp_path):
    log = RagLogger(str(tmp_path / "rag_log.jsonl"))
    query = "Wie funktionieren Ereignisstrom-Architekturen? — naïve café ☕"
    source = "architektur-muster_üß.pdf"

    log.log_query(
        query=query, duration_ms=1.0, chunks_returned=1,
        best_distance=0.2, source_files=[source], status="success",
    )

    raw = (tmp_path / "rag_log.jsonl").read_text(encoding="utf-8")
    # ensure_ascii=False keeps the characters literally in the file
    assert "☕" in raw and "üß" in raw
    rec = _read_records(tmp_path / "rag_log.jsonl")[0]
    assert rec["query"] == query
    assert rec["source_files"] == [source]


def test_oserror_does_not_crash_and_prints_diagnostic(
    tmp_path, monkeypatch, capsys
):
    target = tmp_path / "no_such_dir" / "rag_log.jsonl"
    log = RagLogger(str(target))

    def _boom(*args, **kwargs):
        raise OSError("disk on fire")

    monkeypatch.setattr("builtins.open", _boom)

    # Must not raise — the pipeline degrades, it does not crash.
    log.log_query(
        query="q", duration_ms=1.0, chunks_returned=0,
        best_distance=None, source_files=[], status="web_fallback_error",
    )

    err = capsys.readouterr().err
    assert "RagLogger: failed to write" in err
    assert "disk on fire" in err
    assert not target.exists()


def test_writing_to_unwritable_path_reports_specific_oserror(capsys):
    """Concrete unwritable target (Windows): path points at a directory."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        log = RagLogger(d)  # open() on a directory raises OSError/PermissionError
        log.log_query(
            query="q", duration_ms=1.0, chunks_returned=0,
            best_distance=None, source_files=[], status="success",
        )
        err = capsys.readouterr().err
        assert "RagLogger: failed to write" in err
