"""probe_retrieval.py - Raw Chroma distance probe for DISTANCE_THRESHOLD tuning.

NOT a pytest test — a standalone measurement script. Run it directly:

    python probe_retrieval.py [-k N]

For each representative query it calls similarity_search_with_score() directly
on the Chroma store and prints min / max / mean of the RAW distances. Unlike a
retrieve_chunks()-based probe, nothing is filtered here: DISTANCE_THRESHOLD is
never applied, so distances above the current threshold stay visible — which
is the whole point, because a threshold cannot be calibrated from output that
was already filtered by that threshold.

The probe also never triggers the web-search fallback and never calls a
generative Gemini model. Embedding the query through the configured embedding
model is part of the raw Chroma measurement and is expected.

Rule-based only - no LLM calls.
"""
import argparse
from pathlib import Path
from statistics import mean

import architect

QUERIES = [
    "What are the benefits of a microservices architecture?",
    "When should I choose a monolith over microservices?",
    "How does event-driven architecture improve scalability?",
    "What is the strangler fig pattern for migrating legacy systems?",
    "Which AWS services support serverless architectures?",
    "How to design for high availability and fault tolerance?",
    "What are the responsibilities of an API gateway?",
    "How to handle data consistency in distributed systems?",
    "What is the CQRS pattern and when should it be used?",
    "How to choose between relational SQL and NoSQL databases?",
]

_MAX_SOURCES_SHOWN = 3


def probe_query(vs, query: str, k: int = 5) -> dict:
    """Measure RAW Chroma distances for one query directly on the store.

    Calls vs.similarity_search_with_score(query, k) — no DISTANCE_THRESHOLD
    filtering, no web fallback, no generative model. Returns
    {query, n, min, max, mean, sources} where the distance stats are None
    when the store returns no results.
    """
    results = vs.similarity_search_with_score(query, k=k)
    distances = [dist for _doc, dist in results]
    sources: list[str] = []
    for doc, _dist in results:
        src = Path(doc.metadata.get("source", "unknown")).name
        if src not in sources:
            sources.append(src)
    return {
        "query": query,
        "n": len(results),
        "min": min(distances) if distances else None,
        "max": max(distances) if distances else None,
        "mean": mean(distances) if distances else None,
        "sources": sources,
    }


def format_row(i: int, row: dict) -> str:
    """Render one probe_query() result as a single table line."""
    def _fmt(v):
        return f"{v:.4f}" if v is not None else "--"

    line = (
        f"{i:>2}  {row['query']}\n"
        f"     n={row['n']}  min={_fmt(row['min'])}  "
        f"max={_fmt(row['max'])}  mean={_fmt(row['mean'])}"
    )
    if row["sources"]:
        shown = ", ".join(row["sources"][:_MAX_SOURCES_SHOWN])
        more = len(row["sources"]) - _MAX_SOURCES_SHOWN
        suffix = f" (+{more})" if more > 0 else ""
        line += f"  [{shown}{suffix}]"
    return line


HEADER = (
    f"{'#':>2}  query (raw, unfiltered Chroma distances; lower = better)"
)


def footer_text() -> str:
    """Static explanation printed under the table (kept testable)."""
    return (
        "\nValues are RAW, unfiltered Chroma distances: lower distance = "
        "better match.\n"
        "This probe does NOT apply DISTANCE_THRESHOLD "
        f"(currently {architect.DISTANCE_THRESHOLD}) and never triggers the "
        "web-search fallback, so distances above the threshold stay visible.\n"
        "Pick/adjust DISTANCE_THRESHOLD in architect.py manually from this "
        "table — the probe is a measurement tool only."
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Raw Chroma distance probe (no threshold filtering)."
    )
    parser.add_argument(
        "-k", type=int, default=5,
        help="Raw results fetched per query (default: 5).",
    )
    args = parser.parse_args(argv)

    vs = architect.get_vectorstore()

    print(HEADER)
    print("-" * 78)
    for i, q in enumerate(QUERIES, 1):
        row = probe_query(vs, q, k=args.k)
        print(format_row(i, row))
    print(footer_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
