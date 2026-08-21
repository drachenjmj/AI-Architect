"""kb_gap_report.py - Knowledge-base gap report from the RAG log.

Scans rag_log.jsonl for web-search fallback queries (any "web_fallback*"
status — success, empty, error, or disabled), i.e. topics the internal
knowledge base could not answer, normalises and groups them, and prints a
frequency table so recurring gaps become visible. A topic that hits the
threshold is flagged with "Consider adding to KB box 1/2".

All four fallback statuses count as a KB gap: the external fallback's own
outcome (it succeeded, returned nothing, failed, or was switched off) does not
change the fact that the internal KB had no answer. The pre-fallback statuses
("no_results"/"below_threshold") are NOT counted because every fallback query
also logs one of them — counting both would double-report each gap.

Rule-based only - no LLM calls, no third-party packages.

Usage:
    python kb_gap_report.py [--log rag_log.jsonl] [--threshold 3]
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

DEFAULT_LOG = "rag_log.jsonl"
DEFAULT_THRESHOLD = 3

# Records that mark a knowledge-base gap: the retrieval fell through to the
# web-search fallback. Mirrors the statuses documented in rag_logger.py; keep
# the two in sync. Every KB miss produces exactly one web_fallback* record.
WEB_FALLBACK_STATUSES = frozenset({
    "web_fallback",          # fallback succeeded with grounded chunks
    "web_fallback_empty",    # fallback ran, no usable grounded chunks
    "web_fallback_error",    # fallback raised (API/network failure)
    "web_fallback_disabled", # fallback skipped via kill switch
})

# Generic function words + obvious filler removed before grouping so that
# phrasing variants collapse onto the same topic.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "for", "to", "in", "on", "at",
    "with", "without", "is", "are", "was", "were", "be", "been", "being",
    "what", "whats", "which", "who", "how", "why", "when", "where",
    "do", "does", "did", "can", "could", "should", "would", "will", "shall",
    "may", "might", "must", "i", "you", "we", "they", "it", "this", "that",
    "these", "those", "my", "our", "your", "their", "as", "by", "from", "into",
    "about", "than", "then", "so", "if", "no", "not", "me", "us",
    "best", "practice", "practices", "using", "used", "use", "uses",
    "recommended", "good", "top", "vs", "like", "get", "via",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalize(query: str) -> str:
    """Lowercase, drop punctuation + stopwords, return sorted significant tokens.

    Token order is normalised (sorted) so differently worded queries about the
    same topic collapse into one group. Falls back to the raw lowercased query
    when nothing significant remains.
    """
    tokens = _TOKEN_RE.findall(query.lower())
    sig = sorted({t for t in tokens if t not in STOPWORDS and len(t) > 1})
    if not sig:
        return query.strip().lower() or query.strip()
    return " ".join(sig)


def _fmt_ts(ts: str) -> str:
    """Render an ISO timestamp as YYYY-MM-DD HH:MM, or '-' if unparseable."""
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return ts[:16] if ts else "-"


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "\u2026"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="KB gap report from the RAG log.")
    parser.add_argument(
        "--log",
        default=DEFAULT_LOG,
        help=f"Path to the JSON-Lines RAG log (default: {DEFAULT_LOG}).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help=f"Count that triggers the 'add to KB' hint (default: {DEFAULT_THRESHOLD}).",
    )
    args = parser.parse_args(argv)

    try:
        with open(args.log, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as e:
        print(f"kb_gap_report: cannot read {args.log}: {e}", file=sys.stderr)
        return 1

    counts: dict[str, int] = defaultdict(int)
    latest: dict[str, str] = defaultdict(str)
    raw_queries: dict[str, Counter] = defaultdict(Counter)

    for ln, raw in enumerate(lines, 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            print(f"kb_gap_report: skipping malformed line {ln}", file=sys.stderr)
            continue
        if rec.get("status") not in WEB_FALLBACK_STATUSES:
            continue
        query = rec.get("query", "") or ""
        key = _normalize(query)
        counts[key] += 1
        raw_queries[key][query] += 1
        ts = rec.get("timestamp", "") or ""
        if ts > latest[key]:
            latest[key] = ts

    if not counts:
        print("No web-fallback queries found in the log.")
        return 0

    # Sort by frequency desc, then most-recent desc, then topic for stable order.
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], latest[kv[0]], kv[0]))

    topic_w = max(
        len("Query-Thema"),
        max(len(_truncate(raw_queries[k].most_common(1)[0][0], 45)) for k, _ in rows),
    )
    count_w = max(len("Anzahl"), len(str(max(counts.values()))))

    header = f"{'Query-Thema':<{topic_w}}  {'Anzahl':>{count_w}}  {'zuletzt':<16}"
    print(header)
    print("-" * len(header))
    for key, n in rows:
        topic = _truncate(raw_queries[key].most_common(1)[0][0], 45)
        line = f"{topic:<{topic_w}}  {n:>{count_w}}  {_fmt_ts(latest[key]):<16}"
        if n >= args.threshold:
            line += "  <- Consider adding to KB box 1/2."
        print(line)
    print(
        f"\n{sum(counts.values())} web-fallback quer{'y' if sum(counts.values()) == 1 else 'ies'}"
        f" across {len(counts)} topic{'s' if len(counts) != 1 else ''}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
