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

import pdf_to_md as p2m

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
