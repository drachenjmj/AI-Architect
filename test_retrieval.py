"""test_retrieval.py - Distance-distribution probe for DISTANCE_THRESHOLD tuning.

Runs 10 hardcoded architecture queries through retrieve_chunks() and prints a
table of min / max / mean Chroma distance per query (lower = better match).
Rule-based only - no LLM calls.

Usage:
    python test_retrieval.py [-k N]
"""
import argparse
from statistics import mean

from architect import retrieve_chunks

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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="RAG distance distribution probe.")
    parser.add_argument(
        "-k", type=int, default=5,
        help="Number of chunks retrieved per query (default: 5).",
    )
    args = parser.parse_args(argv)

    header = f"{'#':>2}  {'min':>7} {'max':>7} {'mean':>7}  {'n':>2}  query"
    print(header)
    print("-" * len(header))
    for i, q in enumerate(QUERIES, 1):
        chunks, _origin = retrieve_chunks(q, k=args.k)
        dists = [c["distance"] for c in chunks]
        if dists:
            print(f"{i:>2}  {min(dists):>7.4f} {max(dists):>7.4f} {mean(dists):>7.4f}  {len(dists):>2}  {q}")
        else:
            print(f"{i:>2}  {'--':>7} {'--':>7} {'--':>7}  {0:>2}  {q}  [no results]")

    print("\nLower distance = better match. Pick DISTANCE_THRESHOLD in architect.py from the max/mean gap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
