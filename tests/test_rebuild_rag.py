"""test_rebuild_rag.py - offline tests for the canonical RAG rebuild pipeline
(tools/rebuild_rag.py).

Everything here is offline: source discovery/chunking reads real files
already committed to the repo (no network), and every SQLite/manifest
fixture is hand-built. No test calls Gemini, Chroma remote services,
KI Connect, or the web, and no test mutates the real bundled chroma_db/ or
Rag Database/ - swap/interruption logic is exercised against tmp_path
directories via the parameterized helpers, never the module-level
STAGING_DIR/BACKUP_DIR/CHROMA_DIR constants.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from tools import rebuild_rag as rr

REPO_ROOT = rr.REPO_ROOT


# ── source discovery ────────────────────────────────────────────────────────

def test_discover_sources_is_box1_and_box2_only_count_11():
    sources = rr.discover_sources()
    assert len(sources) == 11
    assert {b for _p, b in sources} == {1, 2}


def test_discover_sources_ignores_raw_archive():
    sources = rr.discover_sources()
    for path, _box in sources:
        assert "raw_source_archive" not in path.parts
        assert "Archive" not in path.parts


def test_discover_sources_is_deterministic():
    assert rr.discover_sources() == rr.discover_sources()


def test_discover_sources_supports_pdf_and_markdown():
    suffixes = {p.suffix.lower() for p, _box in rr.discover_sources()}
    assert suffixes == {".pdf", ".md"}


def test_discover_sources_box_assignment_matches_directory():
    for path, box in rr.discover_sources():
        if "box1_patterns" in path.parts:
            assert box == 1
        elif "box2_domain" in path.parts:
            assert box == 2
        else:
            pytest.fail(f"unexpected source directory: {path}")


def test_discover_sources_only_scans_named_box_dirs(tmp_path):
    data_dir = tmp_path / "Rag Database"
    (data_dir / "box1_patterns").mkdir(parents=True)
    (data_dir / "box2_domain").mkdir(parents=True)
    (data_dir / "raw_source_archive").mkdir(parents=True)
    (data_dir / "box1_patterns" / "a.md").write_text("a", encoding="utf-8")
    (data_dir / "box2_domain" / "b.pdf").write_bytes(b"%PDF-1.4 fake")
    (data_dir / "raw_source_archive" / "c.md").write_text("c", encoding="utf-8")
    (data_dir / "root_level.md").write_text("root", encoding="utf-8")

    sources = rr.discover_sources(data_dir)

    names = {p.name for p, _b in sources}
    assert names == {"a.md", "b.pdf"}


# ── metadata normalization ──────────────────────────────────────────────────

def test_relative_source_path_is_repo_relative_posix():
    path, _box = rr.discover_sources()[0]
    rel = rr.relative_source_path(path)
    assert rel.startswith("Rag Database/")
    assert "\\" not in rel
    assert ":" not in rel  # no Windows drive letter
    assert not rel.startswith("/")


def test_relative_source_path_matches_for_all_active_sources():
    for path, _box in rr.discover_sources():
        rel = rr.relative_source_path(path)
        assert (REPO_ROOT / rel).resolve() == path.resolve()


def test_loaded_documents_have_normalized_source_metadata():
    sources = rr.discover_sources()
    md_sources = [(p, b) for p, b in sources if p.suffix.lower() == ".md"][:1]
    docs = rr.load_documents(md_sources)
    assert docs
    for doc in docs:
        assert doc.metadata["source"] == rr.relative_source_path(md_sources[0][0])
        assert "\\" not in doc.metadata["source"]
        assert doc.metadata["box"] == md_sources[0][1]


# ── chunk plan (reproducibility contract) ───────────────────────────────────

def test_chunking_constants_are_the_accepted_values():
    assert rr.CHUNK_SIZE == 1000
    assert rr.CHUNK_OVERLAP == 200
    assert rr.EMBEDDING_MODEL == "models/gemini-embedding-2"
    assert rr.BATCH_SIZE == 100
    assert rr.MAX_TRIES == 6
    assert rr.COLLECTION_NAME == "langchain"


@pytest.fixture(scope="module")
def real_plan():
    return rr.build_chunk_plan()


def test_current_corpus_matches_baseline_503_440_63(real_plan):
    assert real_plan.source_count == 11
    assert real_plan.chunk_count == 503
    assert real_plan.box_counts == {1: 440, 2: 63}
    assert rr.verify_expected_corpus(real_plan) == []


def test_per_source_chunk_counts_match_baseline(real_plan):
    expected = {
        "Rag Database/box1_patterns/architecture_patterns_v2.md": 119,
        "Rag Database/box1_patterns/microservices-on-aws.pdf": 86,
        "Rag Database/box1_patterns/wellarchitected-serverless-applications-lens.pdf": 235,
        "Rag Database/box2_domain/ecommerce_eshop_event_choreography_hrbatovic.md": 15,
        "Rag Database/box2_domain/ecommerce_inventory_optimistic_locking_aws.md": 4,
        "Rag Database/box2_domain/ecommerce_inventory_reservation_stripe.md": 3,
        "Rag Database/box2_domain/ecommerce_microservices_challenges_ibrahim_luong.md": 9,
        "Rag Database/box2_domain/ecommerce_migration_event_driven_bulus.md": 13,
        "Rag Database/box2_domain/ecommerce_payment_idempotency_stripe.md": 3,
        "Rag Database/box2_domain/ecommerce_polyglot_persistence_microsoft.md": 10,
        "Rag Database/box2_domain/ecommerce_search_opensearch_aws.md": 6,
    }
    assert real_plan.per_source_chunk_count == expected


def test_verify_expected_corpus_flags_mismatch():
    plan = rr.ChunkPlan(sources=[], chunks=[object()] * 10, per_source_chunk_count={}, box_counts={1: 10})
    problems = rr.verify_expected_corpus(plan)
    assert problems  # source count, chunk count, and box counts all wrong
    assert any("source count" in p for p in problems)
    assert any("chunk count" in p for p in problems)
    assert any("box" in p for p in problems)


# ── manifest ─────────────────────────────────────────────────────────────

def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib
    f = tmp_path / "x.txt"
    f.write_bytes(b"hello world")
    assert rr.sha256_file(f) == hashlib.sha256(b"hello world").hexdigest()


# ── canonical cross-platform source hashing (source_sha256) ────────────────
# core.autocrlf=true (the Windows default) rewrites LF -> CRLF on checkout,
# so a fresh Windows clone's Markdown files are byte-different from the same
# logical content checked out on macOS/Linux (or edited directly on a
# machine that never re-checked them out). Raw-byte hashing would then see
# every text source as "changed" and report a false STALE. source_sha256()
# exists specifically to make that impossible for .md sources while still
# hashing binary PDFs exactly as before.

_MD_SAMPLE_LF = "# Title\n\nSome body text.\n\n- item one\n- item two\n"


def test_markdown_hash_identical_for_lf_crlf_cr(tmp_path):
    lf = tmp_path / "lf.md"
    crlf = tmp_path / "crlf.md"
    cr = tmp_path / "cr.md"
    lf.write_bytes(_MD_SAMPLE_LF.encode("utf-8"))
    crlf.write_bytes(_MD_SAMPLE_LF.replace("\n", "\r\n").encode("utf-8"))
    cr.write_bytes(_MD_SAMPLE_LF.replace("\n", "\r").encode("utf-8"))

    h_lf = rr.source_sha256(lf)
    h_crlf = rr.source_sha256(crlf)
    h_cr = rr.source_sha256(cr)

    assert h_lf == h_crlf == h_cr


def test_markdown_hash_changes_on_real_textual_change(tmp_path):
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text(_MD_SAMPLE_LF, encoding="utf-8")
    b.write_text(_MD_SAMPLE_LF.replace("Some body text.", "Some OTHER body text."), encoding="utf-8")
    assert rr.source_sha256(a) != rr.source_sha256(b)


def test_markdown_hash_changes_on_trailing_space_change(tmp_path):
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text(_MD_SAMPLE_LF, encoding="utf-8")
    b.write_text(_MD_SAMPLE_LF.replace("item one\n", "item one \n"), encoding="utf-8")
    assert rr.source_sha256(a) != rr.source_sha256(b)


def test_markdown_hash_changes_on_missing_final_newline():
    """A missing/extra trailing newline is a REAL content difference (a
    different number of logical lines), not a newline-ENCODING difference -
    it must still change the hash even though only newline canonicalization
    is applied."""
    with_nl = rr.canonicalize_markdown_text(_MD_SAMPLE_LF)
    without_nl = rr.canonicalize_markdown_text(_MD_SAMPLE_LF.rstrip("\n"))
    assert with_nl != without_nl


def test_markdown_hash_is_deterministic_for_unicode_content(tmp_path):
    # write_bytes(), not write_text(): Path.write_text()'s default universal-
    # newline mode translates \n -> os.linesep on WRITE on Windows, which
    # would silently re-normalize the deliberately-CRLF fixture back toward
    # the host's own newline style and defeat the point of this test.
    text = "# Café façade\n\n你好，世界。\n\n- Ångström\n"
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_bytes(text.encode("utf-8"))
    b.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    assert rr.source_sha256(a) == rr.source_sha256(b) == rr.source_sha256(a)


def test_markdown_hash_does_not_depend_on_machine_path(tmp_path):
    dir_a = tmp_path / "dir_a" / "nested"
    dir_b = tmp_path / "totally" / "different" / "path"
    dir_a.mkdir(parents=True)
    dir_b.mkdir(parents=True)
    (dir_a / "same.md").write_text(_MD_SAMPLE_LF, encoding="utf-8")
    (dir_b / "same.md").write_text(_MD_SAMPLE_LF, encoding="utf-8")
    assert rr.source_sha256(dir_a / "same.md") == rr.source_sha256(dir_b / "same.md")


def test_canonicalize_markdown_text_only_touches_newlines():
    text = "line one  \nline two\r\nline three\r\n\r\ntrailing blank above\n"
    canonical = rr.canonicalize_markdown_text(text)
    assert "\r" not in canonical
    assert canonical == "line one  \nline two\nline three\n\ntrailing blank above\n"
    # trailing spaces on "line one" are untouched by canonicalization
    assert "line one  \n" in canonical


# ── PDF hashing: raw bytes, no text normalization ───────────────────────────

def test_pdf_hash_identical_bytes_identical_hash(tmp_path):
    data = b"%PDF-1.4\r\nfake binary content\r\nwith embedded CRLF bytes\r\n%%EOF"
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    a.write_bytes(data)
    b.write_bytes(data)
    assert rr.source_sha256(a) == rr.source_sha256(b)


def test_pdf_hash_one_changed_byte_changes_hash(tmp_path):
    data = bytearray(b"%PDF-1.4\r\nfake binary content\r\n%%EOF")
    a = tmp_path / "a.pdf"
    a.write_bytes(bytes(data))
    data[20] ^= 0xFF
    b = tmp_path / "b.pdf"
    b.write_bytes(bytes(data))
    assert rr.source_sha256(a) != rr.source_sha256(b)


def test_pdf_hash_no_newline_normalization(tmp_path):
    """A .pdf containing CRLF byte sequences must hash as raw bytes, exactly
    like sha256_file() - unlike .md, nothing is ever canonicalized."""
    data = b"%PDF-1.4\r\nsome bytes\r\nthat happen to look like CRLF text\r\n%%EOF"
    p = tmp_path / "sample.pdf"
    p.write_bytes(data)
    assert rr.source_sha256(p) == rr.sha256_file(p)
    # And explicitly NOT equal to what the .md canonicalization path would
    # produce for the same bytes decoded as text.
    canonical_as_if_text = rr.canonicalize_markdown_text(data.decode("utf-8")).encode("utf-8")
    import hashlib
    assert rr.source_sha256(p) != hashlib.sha256(canonical_as_if_text).hexdigest()


# ── manifest / validator regression against a CRLF checkout simulation ─────
#
# Fully isolated from the real Rag Database/ and chroma_db/: REPO_ROOT and
# discover_sources() are monkeypatched to a synthetic tmp_path fixture built
# with the real Strategy-A layout, then run through the REAL
# build_chunk_plan()/build_manifest()/validate_offline() pipeline - no test
# here ever opens a file under the real repository tree.

def _isolated_md_fixture(tmp_path, monkeypatch, text=_MD_SAMPLE_LF):
    """One box1_patterns/*.md source under an isolated tmp "repo root",
    wired so build_chunk_plan()/build_manifest()/validate_offline() (which
    reads REPO_ROOT and calls discover_sources() with no arguments) resolve
    entirely inside tmp_path."""
    repo_root = tmp_path / "fake_repo"
    data_dir = repo_root / "Rag Database"
    (data_dir / "box1_patterns").mkdir(parents=True)
    md_path = data_dir / "box1_patterns" / "sample.md"
    md_path.write_text(text, encoding="utf-8")

    monkeypatch.setattr(rr, "REPO_ROOT", repo_root)
    # discover_sources() is called with NO arguments inside validate_offline
    # (its "data_dir: Path = DATA_DIR" default was already bound to the REAL
    # DATA_DIR at import time, so patching rr.DATA_DIR alone would not
    # retroactively change it) - replace the function itself instead.
    monkeypatch.setattr(rr, "discover_sources", lambda data_dir=None: [(md_path, 1)])
    return data_dir, md_path


def test_validate_offline_tolerates_crlf_checkout_of_same_markdown(tmp_path, monkeypatch):
    """Simulates PHASE 6's scenario end-to-end through the real validator:
    build a manifest from LF source, rewrite ONLY the isolated tmp copy
    (never the real Rag Database/) to CRLF, and confirm validation still
    reports VALID."""
    data_dir, md_path = _isolated_md_fixture(tmp_path, monkeypatch)
    plan = rr.build_chunk_plan(data_dir=data_dir)
    manifest = rr.build_manifest(plan)
    chroma_dir = _fake_chroma_dir(tmp_path, manifest)

    # Simulate a fresh Windows checkout (core.autocrlf=true: LF -> CRLF).
    text = md_path.read_text(encoding="utf-8")
    md_path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))

    result = rr.validate_offline(chroma_dir)
    assert result.ok, result.messages


def test_validate_offline_still_detects_real_content_change(tmp_path, monkeypatch):
    data_dir, md_path = _isolated_md_fixture(tmp_path, monkeypatch)
    plan = rr.build_chunk_plan(data_dir=data_dir)
    manifest = rr.build_manifest(plan)
    chroma_dir = _fake_chroma_dir(tmp_path, manifest)

    md_path.write_text(_MD_SAMPLE_LF + "AN ACTUAL CONTENT CHANGE, NOT JUST A NEWLINE STYLE\n", encoding="utf-8")

    result = rr.validate_offline(chroma_dir)
    assert not result.ok
    assert any("hash changed" in m for m in result.messages)


def test_build_manifest_is_deterministic(real_plan):
    m1 = rr.build_manifest(real_plan)
    m2 = rr.build_manifest(real_plan)
    assert json.dumps(m1, sort_keys=True) == json.dumps(m2, sort_keys=True)
    assert list(m1.keys()) == list(m2.keys())
    assert [s["path"] for s in m1["sources"]] == [s["path"] for s in m2["sources"]]


def test_manifest_has_no_timestamp_or_machine_path(real_plan):
    import re

    manifest = rr.build_manifest(real_plan)
    blob = json.dumps(manifest)
    assert "timestamp" not in manifest
    assert str(REPO_ROOT) not in blob
    assert "\\" not in blob
    assert re.search(r"[A-Za-z]:[/\\]", blob) is None  # no Windows drive-letter path


def test_manifest_schema_and_hashes(real_plan):
    manifest = rr.build_manifest(real_plan)
    assert manifest["schema_version"] == 1
    assert manifest["embedding_model"] == rr.EMBEDDING_MODEL
    assert manifest["chunk_size"] == rr.CHUNK_SIZE
    assert manifest["chunk_overlap"] == rr.CHUNK_OVERLAP
    assert manifest["collection_name"] == rr.COLLECTION_NAME
    assert manifest["vector_count"] == 503
    assert manifest["source_count"] == 11
    assert manifest["box_counts"] == {"1": 440, "2": 63}
    for entry in manifest["sources"]:
        actual = rr.source_sha256(REPO_ROOT / entry["path"])
        assert entry["sha256"] == actual


def test_write_manifest_round_trips(tmp_path, real_plan):
    manifest = rr.build_manifest(real_plan)
    path = rr.write_manifest(manifest, tmp_path)
    assert path == tmp_path / rr.MANIFEST_NAME
    assert json.loads(path.read_text(encoding="utf-8")) == manifest


# ── offline validation fixtures ─────────────────────────────────────────────

def _fake_chroma_dir(tmp_path, manifest, *, hnsw=True,
                      db_vector_count=None, db_sources=None, db_box_counts=None,
                      write_sqlite=True, write_manifest_file=True,
                      corrupt_manifest=False) -> Path:
    """Build a minimal-but-schema-faithful fake chroma_db/ under tmp_path
    from *manifest* (as produced by build_manifest()). Individual keyword
    overrides let a test introduce exactly one deliberate mismatch between
    the manifest and the "on-disk" SQLite content it is meant to describe.
    """
    chroma_dir = tmp_path / f"chroma_{uuid.uuid4().hex[:8]}"
    chroma_dir.mkdir()

    if write_manifest_file:
        manifest_path = chroma_dir / rr.MANIFEST_NAME
        if corrupt_manifest:
            manifest_path.write_text("{not valid json", encoding="utf-8")
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not write_sqlite:
        return chroma_dir

    collection_id = "11111111-1111-1111-1111-111111111111"
    metadata_segment_id = "22222222-2222-2222-2222-222222222222"
    vector_segment_id = "33333333-3333-3333-3333-333333333333"

    con = sqlite3.connect(str(chroma_dir / "chroma.sqlite3"))
    cur = con.cursor()
    cur.execute("CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT, dimension INTEGER)")
    cur.execute("CREATE TABLE segments (id TEXT PRIMARY KEY, type TEXT, scope TEXT, collection TEXT)")
    cur.execute(
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, segment_id TEXT, embedding_id TEXT, seq_id BLOB)"
    )
    cur.execute("CREATE TABLE embedding_metadata (id INTEGER, key TEXT, string_value TEXT, int_value INTEGER)")

    cur.execute("INSERT INTO collections VALUES (?, ?, ?)", (collection_id, rr.COLLECTION_NAME, 3072))
    cur.execute(
        "INSERT INTO segments VALUES (?, 'urn:chroma:segment/metadata/sqlite', 'METADATA', ?)",
        (metadata_segment_id, collection_id),
    )
    cur.execute(
        "INSERT INTO segments VALUES (?, 'urn:chroma:segment/vector/hnsw-local-persisted', 'VECTOR', ?)",
        (vector_segment_id, collection_id),
    )

    sources_for_db = db_sources if db_sources is not None else [s["path"] for s in manifest["sources"] for _ in range(s["chunk_count"])]
    boxes_for_db = db_box_counts
    if boxes_for_db is None:
        boxes_for_db = []
        for s in manifest["sources"]:
            boxes_for_db.extend([s["box"]] * s["chunk_count"])

    count = db_vector_count if db_vector_count is not None else len(sources_for_db)
    row_id = 1
    for i in range(count):
        cur.execute(
            "INSERT INTO embeddings VALUES (?, ?, ?, ?)",
            (row_id, metadata_segment_id, str(uuid.uuid4()), b""),
        )
        src = sources_for_db[i % len(sources_for_db)] if sources_for_db else "unknown"
        box = boxes_for_db[i % len(boxes_for_db)] if boxes_for_db else 1
        cur.execute(
            "INSERT INTO embedding_metadata VALUES (?, 'source', ?, NULL)", (row_id, src)
        )
        cur.execute(
            "INSERT INTO embedding_metadata VALUES (?, 'box', NULL, ?)", (row_id, box)
        )
        row_id += 1

    con.commit()
    con.close()

    if hnsw:
        seg_dir = chroma_dir / vector_segment_id
        seg_dir.mkdir()
        for name in ("data_level0.bin", "header.bin", "length.bin", "link_lists.bin"):
            (seg_dir / name).write_bytes(b"")

    return chroma_dir


@pytest.fixture(scope="module")
def real_manifest(real_plan):
    return rr.build_manifest(real_plan)


def test_validate_offline_passes_for_a_matching_fixture(tmp_path, real_manifest):
    chroma_dir = _fake_chroma_dir(tmp_path, real_manifest)
    result = rr.validate_offline(chroma_dir)
    assert result.ok, result.messages


def test_validate_offline_missing_chroma_dir(tmp_path):
    result = rr.validate_offline(tmp_path / "does_not_exist")
    assert not result.ok
    assert any("MISSING" in m for m in result.messages)


def test_validate_offline_missing_manifest(tmp_path, real_manifest):
    chroma_dir = _fake_chroma_dir(tmp_path, real_manifest, write_manifest_file=False)
    result = rr.validate_offline(chroma_dir)
    assert not result.ok
    assert any("manifest" in m.lower() for m in result.messages)


def test_validate_offline_corrupt_manifest_json(tmp_path, real_manifest):
    chroma_dir = _fake_chroma_dir(tmp_path, real_manifest, corrupt_manifest=True)
    result = rr.validate_offline(chroma_dir)
    assert not result.ok
    assert any("json" in m.lower() for m in result.messages)


def test_validate_offline_missing_sqlite(tmp_path, real_manifest):
    chroma_dir = _fake_chroma_dir(tmp_path, real_manifest, write_sqlite=False)
    result = rr.validate_offline(chroma_dir)
    assert not result.ok
    assert any("chroma.sqlite3" in m for m in result.messages)


def test_validate_offline_wrong_vector_count(tmp_path, real_manifest):
    chroma_dir = _fake_chroma_dir(tmp_path, real_manifest, db_vector_count=1)
    result = rr.validate_offline(chroma_dir)
    assert not result.ok
    assert any("vector count" in m for m in result.messages)


def test_validate_offline_wrong_source_set(tmp_path, real_manifest):
    chroma_dir = _fake_chroma_dir(
        tmp_path, real_manifest,
        db_sources=["Rag Database/box1_patterns/architecture_patterns_v2.md"] * real_manifest["vector_count"],
    )
    result = rr.validate_offline(chroma_dir)
    assert not result.ok
    assert any("source set" in m for m in result.messages)


def test_validate_offline_wrong_box_counts(tmp_path, real_manifest):
    n = real_manifest["vector_count"]
    chroma_dir = _fake_chroma_dir(tmp_path, real_manifest, db_box_counts=[1] * n)
    result = rr.validate_offline(chroma_dir)
    assert not result.ok
    assert any("box counts" in m for m in result.messages)


def test_validate_offline_missing_hnsw_files(tmp_path, real_manifest):
    chroma_dir = _fake_chroma_dir(tmp_path, real_manifest, hnsw=False)
    result = rr.validate_offline(chroma_dir)
    assert not result.ok
    assert any("HNSW" in m for m in result.messages)


def test_validate_offline_stale_source_hash(tmp_path, real_manifest):
    manifest = json.loads(json.dumps(real_manifest))
    manifest["sources"][0]["sha256"] = "0" * 64
    chroma_dir = _fake_chroma_dir(tmp_path, manifest)
    result = rr.validate_offline(chroma_dir)
    assert not result.ok
    assert any("hash changed" in m for m in result.messages)


def test_validate_offline_parameter_mismatch(tmp_path, real_manifest):
    manifest = json.loads(json.dumps(real_manifest))
    manifest["chunk_size"] = 999
    chroma_dir = _fake_chroma_dir(tmp_path, manifest)
    result = rr.validate_offline(chroma_dir)
    assert not result.ok
    assert any("chunk_size" in m for m in result.messages)


def test_validate_offline_missing_source_entry(tmp_path, real_manifest):
    manifest = json.loads(json.dumps(real_manifest))
    manifest["sources"].append({
        "path": "Rag Database/box1_patterns/does_not_exist.md",
        "sha256": "0" * 64,
        "box": 1,
        "chunk_count": 1,
    })
    manifest["source_count"] = len(manifest["sources"])
    chroma_dir = _fake_chroma_dir(tmp_path, manifest)
    result = rr.validate_offline(chroma_dir)
    assert not result.ok
    assert any("not found on disk" in m for m in result.messages)


def test_validate_offline_does_not_mutate_files(tmp_path, real_manifest):
    chroma_dir = _fake_chroma_dir(tmp_path, real_manifest)
    before = {
        p: (p.stat().st_mtime_ns, p.stat().st_size)
        for p in chroma_dir.rglob("*") if p.is_file()
    }
    rr.validate_offline(chroma_dir)
    after = {
        p: (p.stat().st_mtime_ns, p.stat().st_size)
        for p in chroma_dir.rglob("*") if p.is_file()
    }
    assert before == after


# ── safe swap ────────────────────────────────────────────────────────────

def _staged_dir(tmp_path, real_manifest, name="staging"):
    d = tmp_path / name
    built = _fake_chroma_dir(tmp_path, real_manifest)
    built.rename(d)
    return d


def test_swap_in_moves_staging_into_target_and_removes_backup(tmp_path, real_manifest):
    staging = _staged_dir(tmp_path, real_manifest, "staging")
    target = tmp_path / "target"
    backup = tmp_path / "backup"
    target.mkdir()
    (target / "old_marker.txt").write_text("old", encoding="utf-8")

    rc = rr.swap_in(staging, target, backup)

    assert rc == 0
    assert target.is_dir()
    assert (target / rr.MANIFEST_NAME).is_file()
    assert not (target / "old_marker.txt").exists()
    assert not staging.exists()
    assert not backup.exists()


def test_swap_in_with_no_prior_target(tmp_path, real_manifest):
    staging = _staged_dir(tmp_path, real_manifest, "staging")
    target = tmp_path / "target_new"
    backup = tmp_path / "backup_new"

    rc = rr.swap_in(staging, target, backup)

    assert rc == 0
    assert target.is_dir()
    assert not backup.exists()


def test_swap_in_rolls_back_when_post_swap_validation_fails(tmp_path, real_manifest):
    # A staging dir that will fail validate_offline once installed (missing
    # HNSW files) must never leave target_dir replaced - the previous
    # (good) target content must come back exactly.
    broken_staging = _fake_chroma_dir(tmp_path, real_manifest, hnsw=False)
    staging = tmp_path / "broken_staging"
    broken_staging.rename(staging)

    target = tmp_path / "target"
    backup = tmp_path / "backup"
    target.mkdir()
    (target / "old_marker.txt").write_text("still here", encoding="utf-8")

    rc = rr.swap_in(staging, target, backup)

    assert rc != 0
    assert target.is_dir()
    assert (target / "old_marker.txt").read_text(encoding="utf-8") == "still here"
    assert not backup.exists()
    assert not staging.exists()  # it was renamed into target's old slot, then rolled back out


def test_interrupted_rebuild_message_none_when_clean(tmp_path):
    assert rr.interrupted_rebuild_message(tmp_path / "staging", tmp_path / "backup") is None


def test_interrupted_rebuild_message_reports_existing_staging(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    msg = rr.interrupted_rebuild_message(staging, tmp_path / "backup")
    assert msg is not None
    assert "interrupted" in msg.lower()


def test_interrupted_rebuild_message_reports_existing_backup(tmp_path):
    backup = tmp_path / "backup"
    backup.mkdir()
    msg = rr.interrupted_rebuild_message(tmp_path / "staging", backup)
    assert msg is not None


# ── CLI ──────────────────────────────────────────────────────────────────

def test_cli_help_lists_public_flags_only(capsys):
    with pytest.raises(SystemExit) as exc:
        rr.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--validate-only" in out
    assert "--yes" in out
    assert "_build-worker" not in out
    assert "_smoke-worker" not in out


def test_cli_validate_only_never_calls_resolve_api_key(monkeypatch, tmp_path):
    def _boom(interactive):
        raise AssertionError("resolve_api_key must not be called for --validate-only")

    monkeypatch.setattr(rr, "resolve_api_key", _boom)
    monkeypatch.setattr(rr, "CHROMA_DIR", tmp_path / "nonexistent")

    rc = rr.main(["--validate-only"])

    assert rc == 1  # nonexistent dir -> INVALID, but no crash and no key prompt


def test_resolve_api_key_fails_clearly_when_missing_and_non_interactive(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(rr, "REPO_ROOT", rr.REPO_ROOT)  # unchanged, .env still loadable
    monkeypatch.chdir(rr.REPO_ROOT)

    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)

    with pytest.raises(SystemExit) as exc:
        rr.resolve_api_key(interactive=False)
    assert "GEMINI_API_KEY" in str(exc.value)


def test_confirm_true_when_yes_flag_set():
    assert rr._confirm(yes=True) is True


def test_confirm_false_when_non_interactive_and_not_yes(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert rr._confirm(yes=False) is False


def test_spawn_worker_never_puts_api_key_in_argv(monkeypatch, tmp_path, capsys):
    secret = "SECRET_TOKEN_XYZ"
    captured = {}

    class _FakeCompletedProcess:
        returncode = 0
        stdout = rr._RESULT_MARKER + json.dumps({"ok": True}) + "\n"
        stderr = ""

    def _fake_run(args, cwd, env, capture_output, text):
        captured["args"] = args
        captured["env"] = env
        return _FakeCompletedProcess()

    monkeypatch.setattr(subprocess, "run", _fake_run)

    rc, result, out = rr._spawn_worker(rr._BUILD_WORKER_FLAG, tmp_path, secret)

    assert rc == 0
    assert result == {"ok": True}
    assert secret not in captured["args"]
    assert captured["env"]["GEMINI_API_KEY"] == secret
    printed = capsys.readouterr()
    assert secret not in printed.out
    assert secret not in printed.err


# ── wrapper (REBUILD_RAG.bat / scripts/rebuild_rag.ps1) ─────────────────────

def test_bat_wrapper_exists_and_delegates_to_powershell_script():
    bat = REPO_ROOT / "REBUILD_RAG.bat"
    assert bat.is_file()
    content = bat.read_text(encoding="utf-8")
    assert "rebuild_rag.ps1" in content


def test_ps1_wrapper_exists_and_calls_canonical_module():
    ps1 = REPO_ROOT / "scripts" / "rebuild_rag.ps1"
    assert ps1.is_file()
    content = ps1.read_text(encoding="utf-8")
    assert "tools.rebuild_rag" in content
    assert "requirements.txt" in content


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell wrapper is Windows-only")
def test_ps1_wrapper_forwards_help_flag():
    ps1 = REPO_ROOT / "scripts" / "rebuild_rag.ps1"
    proc = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-File", str(ps1), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "--validate-only" in proc.stdout
    assert "--yes" in proc.stdout
