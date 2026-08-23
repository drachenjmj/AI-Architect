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


# ── candidate Box-3 sources — exact URL kept, deduplicated, deterministic ──


def _record_with_sources(query, sources, status="web_fallback",
                         ts="2026-08-20T10:00:00"):
    rec = _record(query, status, ts=ts)
    rec["source_files"] = sources
    return rec


def test_candidate_sources_show_the_exact_reviewable_url(tmp_path, capsys):
    log = _write_log(tmp_path / "log.jsonl", [
        _record_with_sources(
            "payment architecture patterns",
            ["stripe.com (https://stripe.com/docs/payments/idempotency)"],
        ),
    ])

    rc = kb_gap_report.main(["--log", log])

    out = capsys.readouterr().out
    assert rc == 0
    assert "Candidate sources:" in out
    # Label AND the exact, openable URL — a domain alone is not reviewable.
    assert (
        "- stripe.com — https://stripe.com/docs/payments/idempotency" in out
    )


def test_duplicate_exact_sources_collapse_to_one_line(tmp_path, capsys):
    log = _write_log(tmp_path / "log.jsonl", [
        _record_with_sources(
            "payment architecture patterns",
            ["stripe.com (https://grounding/redirect/abc)"],
        ),
        _record_with_sources(
            "payment architecture patterns",
            ["stripe.com (https://grounding/redirect/abc)"],  # identical
        ),
    ])

    rc = kb_gap_report.main(["--log", log])

    out = capsys.readouterr().out
    assert rc == 0
    assert out.count("- stripe.com — https://grounding/redirect/abc") == 1


def test_same_exact_url_across_equivalent_log_entries_collapses(tmp_path, capsys):
    """THE canonical-dedup contract: the same URL logged once bare and once
    wrapped with a label is ONE candidate — the canonical key is the exact
    URL, not the rendered display string."""
    log = _write_log(tmp_path / "log.jsonl", [
        _record_with_sources(
            "payment architecture patterns",
            ["https://learn.microsoft.com/azure/architecture/guide"],
        ),
        _record_with_sources(
            "payment architecture patterns",
            ["learn.microsoft.com (https://learn.microsoft.com/azure/architecture/guide)"],
        ),
    ])

    rc = kb_gap_report.main(["--log", log])

    out = capsys.readouterr().out
    assert rc == 0
    block = out.split("Candidate sources:")[1].split("microservices")[0]
    candidate_lines = [
        ln for ln in block.splitlines() if ln.startswith("  - ")
    ]
    assert len(candidate_lines) == 1
    # First seen BARE, later seen labelled -> the single line is
    # deterministically UPGRADED to the labelled form.
    assert candidate_lines[0] == (
        "  - learn.microsoft.com — "
        "https://learn.microsoft.com/azure/architecture/guide"
    )


def test_labelled_first_is_not_downgraded_by_a_later_bare_entry(tmp_path, capsys):
    """Reverse order of the upgrade rule: labelled first, bare later — one
    line, and it keeps its label."""
    log = _write_log(tmp_path / "log.jsonl", [
        _record_with_sources(
            "payment architecture patterns",
            ["stripe.com (https://stripe.com/docs/idempotency)"],
        ),
        _record_with_sources(
            "payment architecture patterns",
            ["https://stripe.com/docs/idempotency"],
        ),
    ])

    rc = kb_gap_report.main(["--log", log])

    out = capsys.readouterr().out
    assert rc == 0
    block = out.split("Candidate sources:")[1]
    candidate_lines = [
        ln for ln in block.splitlines() if ln.startswith("  - ")
    ]
    assert candidate_lines == [
        "  - stripe.com — https://stripe.com/docs/idempotency"
    ]


def test_different_urls_on_the_same_domain_both_remain(tmp_path, capsys):
    """THE usability regression this fix exists for: two different articles
    on one domain are two different candidates — collapsing them to the
    domain would hide a reviewable source."""
    log = _write_log(tmp_path / "log.jsonl", [
        _record_with_sources(
            "payment architecture patterns",
            ["stripe.com (https://stripe.com/docs/payments/idempotency)"],
        ),
        _record_with_sources(
            "payment architecture patterns",
            ["stripe.com (https://stripe.com/blog/retries)"],
        ),
    ])

    rc = kb_gap_report.main(["--log", log])

    out = capsys.readouterr().out
    assert rc == 0
    assert "- stripe.com — https://stripe.com/docs/payments/idempotency" in out
    assert "- stripe.com — https://stripe.com/blog/retries" in out


def test_bare_urls_stay_copyable(tmp_path, capsys):
    log = _write_log(tmp_path / "log.jsonl", [
        _record_with_sources(
            "quantum toaster architecture",
            ["https://example.org/papers/toaster-saga"],
        ),
    ])

    rc = kb_gap_report.main(["--log", log])

    out = capsys.readouterr().out
    assert rc == 0
    assert "- https://example.org/papers/toaster-saga" in out


def test_distinct_sources_for_one_topic_all_remain_visible(tmp_path, capsys):
    log = _write_log(tmp_path / "log.jsonl", [
        _record_with_sources(
            "payment architecture patterns",
            ["stripe.com (https://grounding/redirect/abc)",
             "medium.com (https://grounding/redirect/def)"],
        ),
        _record_with_sources(
            "payment architecture patterns",
            ["martinfowler.com (https://grounding/redirect/ghi)"],
        ),
    ])

    rc = kb_gap_report.main(["--log", log])

    out = capsys.readouterr().out
    assert rc == 0
    for domain in ("stripe.com", "medium.com", "martinfowler.com"):
        assert f"- {domain} — https://grounding/redirect/" in out, domain


def test_records_without_sources_render_without_a_candidate_block(
    tmp_path, capsys
):
    """`source_files: []` is the common shape for empty/error/disabled
    fallbacks — no candidate block may be printed for them."""
    log = _write_log(tmp_path / "log.jsonl", [
        _record("strangler fig migration", "web_fallback_empty"),
    ])

    rc = kb_gap_report.main(["--log", log])

    out = capsys.readouterr().out
    assert rc == 0
    assert "strangler fig migration" in out
    assert "Candidate sources:" not in out


def test_malformed_source_entries_degrade_gracefully(tmp_path, capsys):
    """Non-list `source_files` or non-string entries are ignored, and an
    unparseable string is shown truncated — never a crash."""
    log = _write_log(tmp_path / "log.jsonl", [
        _record_with_sources("weird sources", ["x" * 200]),
    ])
    with open(log, encoding="utf-8") as f:  # add one non-list record
        content = f.read()
    path = tmp_path / "log2.jsonl"
    path.write_text(
        content
        + json.dumps(
            {**_record("weird sources", "web_fallback"), "source_files": "not-a-list"}
        ) + "\n",
        encoding="utf-8",
    )

    rc = kb_gap_report.main(["--log", str(path)])

    out = capsys.readouterr().out
    assert rc == 0
    assert "weird sources" in out
    assert "x" * 61 not in out          # the long entry is truncated


def test_candidate_order_is_deterministic(tmp_path, capsys):
    log = _write_log(tmp_path / "log.jsonl", [
        _record_with_sources(
            "payment architecture patterns",
            ["b-second.example (https://g/1)", "a-first.example (https://g/2)"],
        ),
    ])
    outputs = [
        kb_gap_report.main(["--log", log]) or capsys.readouterr().out
        for _ in range(2)
    ]

    assert outputs[0] == outputs[1]      # same log -> identical report
    block = outputs[0].split("Candidate sources:")[1]
    listed = [ln.strip("- ").strip() for ln in block.splitlines() if ln.startswith("  - ")]
    assert listed == [                   # first-seen order, URL kept
        "b-second.example — https://g/1",
        "a-first.example — https://g/2",
    ]


def test_grouping_and_counts_unchanged_by_source_collection(tmp_path, capsys):
    """The candidate feature must not disturb the existing grouping/count
    contract: identical records WITH sources produce the same summary line
    as their source-less twins."""
    with_sources = _write_log(tmp_path / "a.jsonl", [
        _record_with_sources(
            "scale microservices", ["frontegg.com (https://g/x)"],
        ),
        _record_with_sources(
            "How do I scale microservices?",
            ["stripe.com (https://g/y)"],
        ),
    ])
    without = _write_log(tmp_path / "b.jsonl", [
        _record("scale microservices", "web_fallback"),
        _record("How do I scale microservices?", "web_fallback"),
    ])

    kb_gap_report.main(["--log", with_sources])
    summary_with = capsys.readouterr().out.splitlines()[-1]
    kb_gap_report.main(["--log", without])
    summary_without = capsys.readouterr().out.splitlines()[-1]

    assert summary_with == summary_without == "2 web-fallback queries across 1 topic."
