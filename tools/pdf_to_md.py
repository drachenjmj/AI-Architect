"""pdf_to_md.py - Convert a PDF to clean Markdown for RAG ingestion.

Deterministic, rule-based extraction with pypdf. No LLM, no network, no
external ChatGPT step — this script IS the repo-owned preparation stage of
the standalone ingestion workflow:

    PDF source
    -> python pdf_to_md.py input.pdf -o "Rag Database/<box folder>"
    -> clean Markdown with page boundaries preserved as "---" separators
    -> REVIEW the .md by hand (nothing becomes permanent KB content unseen)
    -> rerun notebooks/Rag_Setup.ipynb, which chunks (1000/200), embeds, and
       persists to chroma_db/ via the existing pipeline.

Box selection is purely the output DIRECTORY:
    - "Rag Database/box1_patterns/"  -> box 1 (general architecture patterns)
    - "Rag Database/box2_domain/"    -> box 2 (domain knowledge)

Future Box-2 domains need no code change: the loader keys on the two box
folder names only, never on domain names — additional domains are added as
flat, domain-prefixed files (the existing convention, e.g.
"ecommerce_*.md", "healthcare_*.md") inside box2_domain/.

Page provenance: each PDF page becomes one block separated by a horizontal
rule ("---"), so the ORIGINAL page boundaries stay visible inline in the
prepared Markdown and in the chunks built from it. Chunk metadata is
assigned by the ingestion loader, NOT here, and differs by source type:
direct PDFs (PyPDFLoader in notebooks/Rag_Setup.ipynb) get real per-page `source` +
`page` metadata; prepared Markdown (TextLoader) keeps `source=<filename>`
but sets `page=0` — exact original PDF page numbers are NOT reconstructed
as chunk metadata. That limitation is acceptable for this prototype and can
be improved later if page-level Markdown citation ever becomes necessary.

Source-file provenance: the file written to disk (not the in-memory
`pdf_to_markdown()` return value used by callers/tests) is prefixed with a
two-line header naming the originating PDF, e.g. "Source PDF: report.pdf".
This keeps a generated file identifiable on its own even outside git.

Safety: an existing output `.md` file is never silently replaced — this
tool is expected to run against a curated knowledge base where hand-edited
Markdown must not be clobbered by a rerun. Pass --overwrite to replace an
existing file intentionally. The final file is written atomically (temp
file + os.replace) so a failure mid-write never leaves a partial file at
the destination path.

Limits: text extraction only — scanned/image-only PDFs have no extractable
text and fail with a clear error (no OCR; the project does not depend on
any OCR library).

Cleanup rules:
  - Running headers/footers (lines recurring in the top/bottom margin of
    >= --min-repeat pages) are removed.
  - Trailing whitespace per line is stripped.
  - Runs of blank lines are collapsed to at most one blank line.
  - Page boundaries become horizontal rules ("---").

Usage:
    python pdf_to_md.py input.pdf -o "Rag Database/box1_patterns/"
    python pdf_to_md.py input.pdf -o "Rag Database/box1_patterns/" --overwrite
"""
import argparse
import os
import re
import sys
import tempfile
from collections import Counter

from pypdf import PdfReader
from pypdf.errors import PdfReadError


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


def _write_markdown_atomic(content: str, out_path: str) -> None:
    """Write *content* to *out_path* atomically (temp file + os.replace), so a
    failure mid-write never leaves a partial file at the destination path."""
    out_dir = os.path.dirname(out_path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=out_dir, prefix=".pdf_to_md-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, out_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


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
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace an existing output .md file (default: refuse and exit 1).",
    )
    args = parser.parse_args(argv)

    if not os.path.isfile(args.pdf):
        print(f"Error: input PDF not found: {args.pdf}", file=sys.stderr)
        return 1

    base = os.path.splitext(os.path.basename(args.pdf))[0]
    out_path = os.path.join(args.outdir, base + ".md")
    if os.path.exists(out_path) and not args.overwrite:
        print(
            f"Error: output already exists: {out_path} "
            "(pass --overwrite to replace it intentionally).",
            file=sys.stderr,
        )
        return 1

    try:
        pages = _extract_pages(args.pdf)
    except (PdfReadError, ValueError, OSError) as e:
        print(
            f"Error: cannot read {args.pdf} as a PDF: {e}", file=sys.stderr
        )
        return 1

    markdown = convert_pages_to_markdown(pages, args.margin, args.min_repeat)
    if not markdown:
        # An empty result would otherwise be written as a silent 0-byte
        # "success". Text extraction found nothing — typically a scanned,
        # image-only PDF (no OCR in this project) — so fail loudly instead
        # of producing a source that would ingest as nothing.
        print(
            f"Error: no extractable text found in {args.pdf} — it is empty or "
            "a scanned/image-only PDF (OCR is not supported).",
            file=sys.stderr,
        )
        return 1

    source_name = os.path.basename(args.pdf)
    header = f"Source PDF: {source_name}\nConversion: generated by tools/pdf_to_md.py\n\n"
    os.makedirs(args.outdir, exist_ok=True)
    _write_markdown_atomic(header + markdown, out_path)

    print(f"Wrote {out_path} ({len(markdown)} chars from {len(pages)} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
