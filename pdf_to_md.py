"""pdf_to_md.py - Convert a PDF to clean Markdown for RAG ingestion.

Deterministic, rule-based extraction with pypdf. No LLM, no network.

Cleanup rules:
  - Running headers/footers (lines recurring in the top/bottom margin of
    >= --min-repeat pages) are removed.
  - Trailing whitespace per line is stripped.
  - Runs of blank lines are collapsed to at most one blank line.
  - Page boundaries become horizontal rules ("---").

Usage:
    python pdf_to_md.py input.pdf -o "Rag Database/box1_patterns/"
"""
import argparse
import os
import re
import sys
from collections import Counter

from pypdf import PdfReader


def _extract_pages(pdf_path: str) -> list[str]:
    reader = PdfReader(pdf_path)
    return [(page.extract_text() or "") for page in reader.pages]


def _recurring_lines(pages: list[str], margin: int = 3, min_repeat: int = 3) -> set[str]:
    """Stripped lines that recur in the top/bottom *margin* lines of at least
    *min_repeat* pages (typical running headers/footers)."""
    counter = Counter()
    for text in pages:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            continue
        head = lines[:margin]
        tail = lines[-margin:] if len(lines) > margin else []
        for ln in set(head) | set(tail):
            counter[ln] += 1
    return {ln for ln, n in counter.items() if n >= min_repeat}


def _clean_page(text: str, recurring: set[str]) -> str:
    kept = [ln for ln in text.splitlines() if ln.strip() not in recurring]
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def convert_pages_to_markdown(pages: list[str], margin: int = 3, min_repeat: int = 3) -> str:
    recurring = _recurring_lines(pages, margin, min_repeat)
    parts = []
    for text in pages:
        cleaned = _clean_page(text, recurring)
        if cleaned:
            parts.append(cleaned)
    return "\n\n---\n\n".join(parts)


def pdf_to_markdown(pdf_path: str, margin: int = 3, min_repeat: int = 3) -> str:
    return convert_pages_to_markdown(_extract_pages(pdf_path), margin, min_repeat)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a PDF to clean Markdown for RAG ingestion."
    )
    parser.add_argument("pdf", help="Path to the input PDF file.")
    parser.add_argument(
        "-o", "--outdir", default=".",
        help="Output directory (default: current working directory).",
    )
    parser.add_argument(
        "--margin", type=int, default=3,
        help="Header/footer scan depth in lines (default: 3).",
    )
    parser.add_argument(
        "--min-repeat", type=int, default=3,
        help="Pages a margin line must recur on to be dropped (default: 3).",
    )
    args = parser.parse_args(argv)

    if not os.path.isfile(args.pdf):
        print(f"Error: input PDF not found: {args.pdf}", file=sys.stderr)
        return 1

    pages = _extract_pages(args.pdf)
    markdown = convert_pages_to_markdown(pages, args.margin, args.min_repeat)

    base = os.path.splitext(os.path.basename(args.pdf))[0]
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, base + ".md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"Wrote {out_path} ({len(markdown)} chars from {len(pages)} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
