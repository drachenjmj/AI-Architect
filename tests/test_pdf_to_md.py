"""test_pdf_to_md.py - Offline regression tests for pdf_to_md.py.

Deterministic helpers (_recurring_lines, _clean_page,
convert_pages_to_markdown, main) are unit-tested directly with plain page
strings. Real-PDF I/O is exercised through a tiny fixture built as raw,
uncompressed minimal-PDF bytes (text drawn with a standard Type1 font) — no
pypdf writer internals, no network, no OCR, no new packages. Assertions
avoid exact-whitespace coupling except where whitespace normalisation IS the
contract.
"""

from __future__ import annotations

import pytest

from tools import pdf_to_md as p2m

HEADER = "Running Header Inc."
FOOTER = "Page footer — confidential"


def _pages(n=3, body_fmt="Body line {i} of page {p}.", header=HEADER,
           footer=FOOTER):
    """Page strings with a running header/footer around real body text."""
    out = []
    for p in range(1, n + 1):
        lines = [header]
        lines += [body_fmt.format(i=i, p=p) for i in range(1, 4)]
        lines += [footer]
        out.append("\n".join(lines))
    return out


# ── _recurring_lines: header/footer detection ───────────────────────────────

def test_recurring_header_and_footer_are_detected():
    rec = p2m._recurring_lines(_pages(3), margin=3, min_repeat=3)
    assert HEADER in rec
    assert FOOTER in rec


def test_body_lines_are_not_flagged_as_recurring():
    rec = p2m._recurring_lines(_pages(3), margin=3, min_repeat=3)
    # Body differs per page, so none of it survives the min_repeat=3 filter.
    assert not any(ln.startswith("Body line") for ln in rec)


def test_min_repeat_controls_the_threshold():
    pages = _pages(2)  # header/footer appear on only 2 pages
    assert p2m._recurring_lines(pages, margin=3, min_repeat=3) == set()
    assert HEADER in p2m._recurring_lines(pages, margin=3, min_repeat=2)


def test_margin_controls_scan_depth():
    # Header sits BELOW the top-3 margin lines, with enough body after it
    # that it also escapes the bottom margin -> invisible at margin=3.
    pages = [
        "\n".join(
            ["filler-a", "filler-b", "filler-c", HEADER,
             f"body {i}-one", f"body {i}-two", f"body {i}-three", FOOTER]
        )
        for i in range(3)
    ]
    assert HEADER not in p2m._recurring_lines(pages, margin=3, min_repeat=3)
    assert HEADER in p2m._recurring_lines(pages, margin=4, min_repeat=3)


def test_blank_pages_are_ignored_by_detection():
    rec = p2m._recurring_lines(_pages(3) + ["", "   \n"], margin=3, min_repeat=3)
    assert HEADER in rec  # blank pages neither crash nor skew the count


# ── _clean_page: removal + whitespace normalisation ─────────────────────────

def test_clean_page_drops_recurring_lines_and_keeps_body():
    cleaned = p2m._clean_page(_pages(1)[0], {HEADER, FOOTER})
    assert HEADER not in cleaned
    assert FOOTER not in cleaned
    assert "Body line 1 of page 1." in cleaned
    assert "Body line 3 of page 1." in cleaned


def test_clean_page_strips_trailing_whitespace_and_collapses_blanks():
    raw = "keep me   \t\n\n\n\nalso keep\n"
    cleaned = p2m._clean_page(raw, set())
    assert cleaned == "keep me\n\nalso keep"  # trailing ws gone, <=1 blank line


def test_clean_page_of_only_recurring_lines_becomes_empty():
    assert p2m._clean_page(f"{HEADER}\n{FOOTER}", {HEADER, FOOTER}) == ""


# ── convert_pages_to_markdown: page separation ──────────────────────────────

def test_pages_are_joined_by_horizontal_rules():
    md = p2m.convert_pages_to_markdown(_pages(3))
    parts = md.split("\n\n---\n\n")
    assert len(parts) == 3  # page boundaries become "---" separators
    assert HEADER not in md and FOOTER not in md
    for p in (1, 2, 3):
        assert f"Body line 2 of page {p}." in md


def test_empty_pages_are_skipped_not_blank_separated():
    md = p2m.convert_pages_to_markdown(
        ["body A", "", "   \n", "body B"]
    )
    assert md == "body A\n\n---\n\nbody B"


def test_all_empty_pages_yield_empty_string():
    assert p2m.convert_pages_to_markdown(["", " "]) == ""


# ── real PDF end-to-end with a minimal raw-bytes PDF fixture ────────────────

def _make_pdf(path, page_texts):
    """Write a tiny but fully valid multi-page PDF as raw bytes.

    Each page draws one line of text (Helvetica) via a plain uncompressed
    content stream, so pypdf's extract_text() can read it back. Object
    layout: 1=Catalog, 2=Pages, then per page i: (2+2i)=Page,
    (3+2i)=Contents; the font is object 0-replaced by a shared last object.
    """
    n = len(page_texts)
    font_num = 3 + 2 * n  # shared /Font object comes after all pages

    objects = {}  # number -> bytes (without "N 0 obj"/"endobj" wrapper)

    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode()
    for i, text in enumerate(page_texts):
        page_num = 3 + 2 * i
        content_num = page_num + 1
        safe = text.encode("latin-1", errors="replace").decode("latin-1")
        stream = (
            "BT /F1 10 Tf 20 270 Td "
            f"({safe.replace(chr(40), chr(92) + chr(40)).replace(chr(41), chr(92) + chr(41))}) "
            "Tj ET"
        ).encode("latin-1")
        objects[page_num] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] "
            f"/Resources << /Font << /F1 {font_num} 0 R >> >> "
            f"/Contents {content_num} 0 R >>"
        ).encode()
        objects[content_num] = (
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
            + stream + b"\nendstream"
        )
    objects[font_num] = (
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for num in sorted(objects):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + objects[num] + b"\nendobj\n"
    xref_pos = len(out)
    total = max(objects) + 1
    out += f"xref\n0 {total}\n".encode()
    out += b"0000000000 65535 f \n"
    for num in range(1, total):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {total} /Root 1 0 R >>\nstartxref\n"
        f"{xref_pos}\n%%EOF"
    ).encode()
    path.write_bytes(bytes(out))


def test_pdf_to_markdown_reads_real_pdf_and_keeps_body(tmp_path):
    pdf = tmp_path / "sample.pdf"
    _make_pdf(pdf, ["alpha body text", "beta body text", "gamma body text"])

    md = p2m.pdf_to_markdown(str(pdf))
    for word in ("alpha", "beta", "gamma"):
        assert word in md
    assert md.count("\n\n---\n\n") == 2  # three pages -> two separators


def test_main_missing_input_returns_error(tmp_path, capsys):
    rc = p2m.main([str(tmp_path / "missing.pdf")])

    assert rc == 1
    assert "not found" in capsys.readouterr().err


def test_main_writes_markdown_named_after_input(tmp_path, capsys):
    pdf = tmp_path / "Report 2026.pdf"
    _make_pdf(pdf, ["only page"])
    outdir = tmp_path / "out"
    outdir.mkdir()

    rc = p2m.main([str(pdf), "-o", str(outdir)])

    assert rc == 0
    out = outdir / "Report 2026.md"
    assert out.is_file()
    assert "only page" in out.read_text(encoding="utf-8")
    assert "Wrote" in capsys.readouterr().out


# ── standalone-workflow hardening: deterministic, provenance, failure ──────


def test_conversion_is_deterministic_for_the_same_file(tmp_path):
    pdf = tmp_path / "sample.pdf"
    _make_pdf(pdf, ["alpha body text", "beta body text", "gamma body text"])

    first = p2m.pdf_to_markdown(str(pdf))
    second = p2m.pdf_to_markdown(str(pdf))

    assert first == second


def test_page_provenance_survives_as_separators(tmp_path):
    """Each PDF page becomes exactly one block; the '---' separators are the
    page boundaries the ingestion pipeline's chunks will carry inline."""
    pdf = tmp_path / "pages.pdf"
    _make_pdf(pdf, ["page one text", "page two text", "page three text"])

    md = p2m.pdf_to_markdown(str(pdf))
    blocks = md.split("\n\n---\n\n")

    assert len(blocks) == 3
    assert blocks[0].strip() == "page one text"
    assert blocks[2].strip() == "page three text"


def test_main_fails_clearly_on_unreadable_pdf(tmp_path, capsys):
    """A non-PDF file (here: plain text) must fail with a clear error, not a
    traceback or a bogus empty output."""
    bogus = tmp_path / "not-a-real.pdf"
    bogus.write_bytes(b"this is definitely not a pdf")

    rc = p2m.main([str(bogus)])

    assert rc == 1
    assert "cannot read" in capsys.readouterr().err
    assert not (tmp_path / "not-a-real.md").exists()


def test_main_fails_clearly_on_textless_pdf(tmp_path, capsys):
    """A structurally VALID PDF whose pages contain no extractable text
    (the scanned/image-only shape): a silent 0-byte .md would ingest as
    nothing — fail loudly instead. The fixture builds a real one-page PDF
    with an EMPTY content stream."""
    scanned = tmp_path / "scanned.pdf"
    _make_pdf(scanned, [""])  # valid page, no text drawn

    rc = p2m.main([str(scanned)])

    assert rc == 1
    err = capsys.readouterr().err
    assert "no extractable text" in err
    assert "OCR is not supported" in err
    assert not (tmp_path / "scanned.md").exists()


def test_single_bad_page_does_not_drop_the_document(tmp_path):
    """One empty page among real ones: the document still converts, the
    empty page is skipped rather than aborting everything."""
    pdf = tmp_path / "mixed.pdf"
    _make_pdf(pdf, ["real page one", "", "real page three"])

    md = p2m.pdf_to_markdown(str(pdf))

    assert "real page one" in md and "real page three" in md
    assert md.count("\n\n---\n\n") == 1  # the empty page leaves no separator


# ── the prepared output feeds the EXISTING ingestion path unchanged ────────


def test_ingestion_constants_are_pinned():
    """The accepted chunking/retrieval design lives in
    notebooks/Rag_Setup.ipynb. This parses the notebook source (offline, no
    execution) and pins the values this workflow must NOT change: chunk
    size/overlap and the Box folder mapping."""
    import json
    from pathlib import Path

    nb = json.loads(
        (Path(__file__).resolve().parents[1] / "notebooks" / "Rag_Setup.ipynb")
        .read_text(encoding="utf-8")
    )
    src = "\n".join("".join(c["source"]) for c in nb["cells"])

    assert "chunk_size = 1000" in src.replace(", ", " = ").replace(
        "chunk_size=1000", "chunk_size = 1000"
    ) or "chunk_size = 1000" in src or "chunk_size=1000" in src
    assert "chunk_overlap = 200" in src or "chunk_overlap=200" in src
    assert '"box1_patterns": 1' in src
    assert '"box2_domain": 2' in src


# ── safe-rebuild hardening: overwrite protection, atomic write, provenance ─


def test_convert_pages_to_markdown_has_no_provenance_header():
    """The pure content function (what callers/tests use, and what the
    ingestion loader's page-block boundaries are built from) stays
    header-free; only the CLI file-write path adds the provenance header."""
    md = p2m.convert_pages_to_markdown(["only page"])
    assert "Source PDF" not in md


def test_write_markdown_atomic_writes_utf8_and_leaves_no_temp_file(tmp_path):
    out_path = tmp_path / "out.md"
    content = "Café façade — 你好\n"

    p2m._write_markdown_atomic(content, str(out_path))

    assert out_path.read_text(encoding="utf-8") == content
    assert list(tmp_path.iterdir()) == [out_path]  # no stray .tmp file


def test_write_markdown_atomic_cleans_up_temp_file_on_failure(tmp_path, monkeypatch):
    out_path = tmp_path / "out.md"
    monkeypatch.setattr(p2m.os, "replace", lambda *a, **kw: (_ for _ in ()).throw(OSError("boom")))

    with pytest.raises(OSError):
        p2m._write_markdown_atomic("content", str(out_path))

    assert not out_path.exists()
    assert list(tmp_path.iterdir()) == []  # no partial/temp file left behind


def test_main_writes_provenance_header_identifying_source_pdf(tmp_path):
    pdf = tmp_path / "Whitepaper.pdf"
    _make_pdf(pdf, ["body text"])
    outdir = tmp_path / "out"
    outdir.mkdir()

    rc = p2m.main([str(pdf), "-o", str(outdir)])

    assert rc == 0
    content = (outdir / "Whitepaper.md").read_text(encoding="utf-8")
    assert "Source PDF: Whitepaper.pdf" in content
    assert "generated by tools/pdf_to_md.py" in content
    assert "body text" in content


def test_main_refuses_to_overwrite_existing_output_by_default(tmp_path, capsys):
    pdf = tmp_path / "sample.pdf"
    _make_pdf(pdf, ["first version"])
    outdir = tmp_path / "out"
    outdir.mkdir()
    existing = outdir / "sample.md"
    existing.write_text("curated content — do not touch", encoding="utf-8")

    rc = p2m.main([str(pdf), "-o", str(outdir)])

    assert rc == 1
    assert "already exists" in capsys.readouterr().err
    assert existing.read_text(encoding="utf-8") == "curated content — do not touch"


def test_main_overwrite_flag_replaces_existing_output(tmp_path):
    pdf = tmp_path / "sample.pdf"
    _make_pdf(pdf, ["new version text"])
    outdir = tmp_path / "out"
    outdir.mkdir()
    existing = outdir / "sample.md"
    existing.write_text("old content", encoding="utf-8")

    rc = p2m.main([str(pdf), "-o", str(outdir), "--overwrite"])

    assert rc == 0
    assert "new version text" in existing.read_text(encoding="utf-8")


def test_main_help_documents_overwrite_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        p2m.main(["--help"])

    assert exc.value.code == 0
    assert "--overwrite" in capsys.readouterr().out


def test_generated_output_has_no_machine_specific_absolute_path(tmp_path):
    pdf = tmp_path / "sample.pdf"
    _make_pdf(pdf, ["alpha body text"])
    outdir = tmp_path / "out"
    outdir.mkdir()

    rc = p2m.main([str(pdf), "-o", str(outdir)])

    assert rc == 0
    content = (outdir / "sample.md").read_text(encoding="utf-8")
    assert str(tmp_path) not in content


def test_prepared_output_matches_the_box2_domain_convention():
    """B6: nothing in the converter's LOGIC hard-codes a domain — the module
    docstring may name the existing file-naming convention as documentation,
    but the functions themselves must stay domain-agnostic. Future Box-2
    domains are flat, domain-prefixed .md files inside box2_domain/, which
    the converter's output (basename + '.md') fits as-is."""
    import inspect

    for function in (
        p2m._extract_pages, p2m._recurring_lines, p2m._clean_page,
        p2m.convert_pages_to_markdown, p2m.pdf_to_markdown, p2m.main,
    ):
        source = inspect.getsource(function)
        for domain in ("ecommerce", "healthcare", "finance"):
            assert domain not in source, (function.__name__, domain)
