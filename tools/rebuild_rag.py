"""rebuild_rag.py - canonical, safe, reproducible RAG rebuild pipeline.

This is the single source of truth for building the Chroma knowledge base
that architect.py reads at ./chroma_db. It replaces the old notebook-driven
"delete the live collection, then rebuild in place" workflow with a staged
build: a brand-new index is built in a throwaway staging directory, fully
validated (offline manifest checks, a raw retrieval smoke test), and only
THEN swapped in for the currently bundled chroma_db/. The previous index is
kept as a backup until the swap itself is confirmed good, so a failure at
any point leaves the bundled KB exactly as it was.

Entry points:
    python -m tools.rebuild_rag                  # staged rebuild (confirms first)
    python -m tools.rebuild_rag --yes             # staged rebuild, no confirmation
    python -m tools.rebuild_rag --validate-only   # offline check, no API key, no network

Only the Gemini embedding endpoint is ever called here (models/gemini-embedding-2,
via GoogleGenerativeAIEmbeddings). No generative model, no KI Connect, no web
search, no Architect/Reviewer/Clarifier code path is touched.

Sources are discovered ONLY under the two active Strategy-A folders:
    Rag Database/box1_patterns/   -> box 1
    Rag Database/box2_domain/     -> box 2
`Rag Database/raw_source_archive/` (and any future root-level file) is never
indexed - see docs/rag_source_strategy notes for why.

Windows file-lock safety: chromadb/HNSW hold OS file handles for as long as
the process that opened them is alive, and merely dropping a Python
reference does not guarantee those handles are released in time for a
following os.rename() on Windows. So the actual embedding build and the
retrieval smoke test each run in a short-lived CHILD process (via
`--_build-worker` / `--_smoke-worker`, both hidden from --help - they are
not a supported public CLI) that does its work and exits completely; only
after each child has exited does this (parent) process touch the
filesystem. The API key is passed to the child via its environment, never
as a command-line argument, and is never written to any file this script
controls.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── accepted ingestion constants (single source of truth) ──────────────────
EMBEDDING_MODEL = "models/gemini-embedding-2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 100
MAX_TRIES = 6
COLLECTION_NAME = "langchain"

# The current committed corpus is a reproducibility contract: if source
# discovery/chunking ever silently drifts from this, the rebuild must stop
# BEFORE any embedding API call rather than quietly index something else.
EXPECTED_SOURCE_COUNT = 11
EXPECTED_CHUNK_COUNT = 503
EXPECTED_BOX_COUNTS = {1: 440, 2: 63}

DATA_DIR = REPO_ROOT / "Rag Database"
CHROMA_DIR = REPO_ROOT / "chroma_db"
STAGING_DIR = REPO_ROOT / ".chroma_db_rebuild_staging"
BACKUP_DIR = REPO_ROOT / ".chroma_db_rebuild_backup"
MANIFEST_NAME = "index_manifest.json"

# Directory name (directly under DATA_DIR) -> box number. Nothing else is
# ever scanned for indexing - in particular raw_source_archive/ and Archive/
# are deliberately not in this map.
BOX_DIRS = {
    "box1_patterns": 1,
    "box2_domain": 2,
}
ACTIVE_EXTENSIONS = {".pdf", ".md"}

# Retryable failure signatures (mirrors the accepted notebook behaviour).
# Anything else (auth/config errors, bad requests, ...) is not retried.
TRANSIENT_MARKERS = (
    "429", "500", "502", "503", "504",
    "RESOURCE_EXHAUSTED", "Bad Gateway",
    "getaddrinfo", "ConnectError", "timed out", "timeout",
)

SMOKE_QUERIES = [
    "What are the benefits of microservices architecture?",
    "How should an e-commerce system handle inventory consistency?",
    "How can payment operations be made idempotent?",
    "What are migration patterns from a monolith to microservices?",
]
SMOKE_K = 3

_BUILD_WORKER_FLAG = "--_build-worker"
_SMOKE_WORKER_FLAG = "--_smoke-worker"
_RESULT_MARKER = "REBUILD_RAG_WORKER_RESULT:"


# ──────────────────────────────────────────────
# SOURCE DISCOVERY / LOADING / CHUNKING (pure, offline, no network)
# ──────────────────────────────────────────────
def relative_source_path(path: Path) -> str:
    """Stable, portable, repository-relative POSIX source path.

    Never contains a drive letter or an OS-specific backslash, regardless of
    host OS, so metadata written on Windows and read anywhere else (or vice
    versa) is byte-identical.
    """
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def discover_sources(data_dir: Path = DATA_DIR) -> list[tuple[Path, int]]:
    """Deterministically list active (path, box) pairs.

    Only box1_patterns/ and box2_domain/ are scanned; anything under
    raw_source_archive/, Archive/, or the Rag Database root itself is never
    considered. Sorted by box, then by repository-relative path, so the
    result (and everything derived from it) is fully reproducible.
    """
    # Relative to data_dir's own parent rather than the module-level
    # REPO_ROOT, so this stays deterministic/testable against an arbitrary
    # data_dir (e.g. a tmp_path fixture), not just the real repository tree.
    root = data_dir.resolve().parent
    sources: list[tuple[Path, int]] = []
    for dirname, box in BOX_DIRS.items():
        subdir = data_dir / dirname
        if not subdir.is_dir():
            continue
        for entry in subdir.iterdir():
            if entry.is_file() and entry.suffix.lower() in ACTIVE_EXTENSIONS:
                sources.append((entry, box))
    sources.sort(key=lambda item: (item[1], item[0].resolve().relative_to(root).as_posix()))
    return sources


def load_documents(sources: list[tuple[Path, int]]):
    """Load PDFs (per-page, PyPDFLoader) and Markdown (TextLoader, page=0).

    Every returned Document gets a repository-relative POSIX `source` and
    the correct `box`, matching the previously accepted metadata shape.
    """
    from langchain_community.document_loaders import PyPDFLoader, TextLoader

    documents = []
    for path, box in sources:
        rel = relative_source_path(path)
        if path.suffix.lower() == ".pdf":
            docs = PyPDFLoader(str(path)).load()
            for doc in docs:
                doc.metadata["source"] = rel
        else:
            docs = TextLoader(str(path), encoding="utf-8").load()
            for doc in docs:
                doc.metadata["source"] = rel
                doc.metadata["page"] = 0
        for doc in docs:
            doc.metadata["box"] = box
        documents.extend(docs)
    return documents


def chunk_documents(documents):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


@dataclass
class ChunkPlan:
    sources: list[tuple[Path, int]]
    chunks: list
    per_source_chunk_count: dict[str, int]
    box_counts: dict[int, int]

    @property
    def source_count(self) -> int:
        return len(self.sources)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


def build_chunk_plan(data_dir: Path = DATA_DIR) -> ChunkPlan:
    """Discover + load + chunk, entirely offline (no embedding call)."""
    sources = discover_sources(data_dir)
    documents = load_documents(sources)
    chunks = chunk_documents(documents)

    per_source: dict[str, int] = {relative_source_path(p): 0 for p, _ in sources}
    box_counts: dict[int, int] = {}
    for chunk in chunks:
        src = chunk.metadata["source"]
        per_source[src] = per_source.get(src, 0) + 1
        box = chunk.metadata["box"]
        box_counts[box] = box_counts.get(box, 0) + 1

    return ChunkPlan(sources, chunks, per_source, box_counts)


def verify_expected_corpus(plan: ChunkPlan) -> list[str]:
    """Compare a freshly computed plan against the committed baseline.

    Returns a list of human-readable mismatch descriptions (empty = OK).
    Called BEFORE any embedding API call so an unexpected corpus change is
    caught while it is still free and offline to investigate.
    """
    problems = []
    if plan.source_count != EXPECTED_SOURCE_COUNT:
        problems.append(
            f"active source count = {plan.source_count}, expected {EXPECTED_SOURCE_COUNT}"
        )
    if plan.chunk_count != EXPECTED_CHUNK_COUNT:
        problems.append(
            f"total chunk count = {plan.chunk_count}, expected {EXPECTED_CHUNK_COUNT}"
        )
    if plan.box_counts != EXPECTED_BOX_COUNTS:
        problems.append(
            f"box chunk counts = {plan.box_counts}, expected {EXPECTED_BOX_COUNTS}"
        )
    return problems


# ──────────────────────────────────────────────
# MANIFEST
# ──────────────────────────────────────────────
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_manifest(plan: ChunkPlan) -> dict:
    """Deterministic manifest dict: no timestamp, no machine path, no key."""
    sources_manifest = []
    for path, box in plan.sources:
        rel = relative_source_path(path)
        sources_manifest.append({
            "path": rel,
            "sha256": sha256_file(path),
            "box": box,
            "chunk_count": plan.per_source_chunk_count.get(rel, 0),
        })
    # already sorted (box, path) by discover_sources; keep it explicit here
    sources_manifest.sort(key=lambda s: (s["box"], s["path"]))

    return {
        "schema_version": 1,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "collection_name": COLLECTION_NAME,
        "vector_count": plan.chunk_count,
        "source_count": plan.source_count,
        "box_counts": {str(k): v for k, v in sorted(plan.box_counts.items())},
        "sources": sources_manifest,
    }


def write_manifest(manifest: dict, chroma_dir: Path) -> Path:
    manifest_path = chroma_dir / MANIFEST_NAME
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest_path


# ──────────────────────────────────────────────
# OFFLINE VALIDATION (read-only SQLite - no chromadb/langchain import,
# no network, never mutates chroma_dir)
# ──────────────────────────────────────────────
@dataclass
class ValidationResult:
    ok: bool
    messages: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.messages.append(msg)


def _ro_connect(sqlite_path: Path):
    import sqlite3
    uri = f"file:{sqlite_path.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def validate_offline(chroma_dir: Path = CHROMA_DIR) -> ValidationResult:
    """Validate a Chroma index directory without an API key, network call,
    embedding call, or generative call, and without mutating chroma_dir.

    Uses read-only SQLite access rather than a LangChain/Chroma client so
    that merely checking the index can never itself create runtime drift.
    """
    result = ValidationResult(ok=True)
    chroma_dir = Path(chroma_dir)

    if not chroma_dir.is_dir():
        result.fail(f"MISSING: chroma directory does not exist: {chroma_dir}")
        return result

    manifest_path = chroma_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        result.fail(f"MISSING: manifest not found: {manifest_path}")
        return result
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        result.fail(f"INVALID: manifest is not valid JSON: {e}")
        return result

    if manifest.get("embedding_model") != EMBEDDING_MODEL:
        result.fail(
            f"STALE: manifest embedding_model={manifest.get('embedding_model')!r}, "
            f"code expects {EMBEDDING_MODEL!r}"
        )
    if manifest.get("chunk_size") != CHUNK_SIZE:
        result.fail(f"STALE: manifest chunk_size={manifest.get('chunk_size')!r}, code expects {CHUNK_SIZE!r}")
    if manifest.get("chunk_overlap") != CHUNK_OVERLAP:
        result.fail(
            f"STALE: manifest chunk_overlap={manifest.get('chunk_overlap')!r}, "
            f"code expects {CHUNK_OVERLAP!r}"
        )
    if manifest.get("collection_name") != COLLECTION_NAME:
        result.fail(
            f"STALE: manifest collection_name={manifest.get('collection_name')!r}, "
            f"code expects {COLLECTION_NAME!r}"
        )

    manifest_sources = manifest.get("sources", [])
    manifest_source_set = {s["path"] for s in manifest_sources}
    manifest_box_counts = {int(k): v for k, v in manifest.get("box_counts", {}).items()}

    # 4/5: every manifest source exists on disk and its hash still matches.
    for entry in manifest_sources:
        src_path = REPO_ROOT / entry["path"]
        if not src_path.is_file():
            result.fail(f"MISSING: source listed in manifest not found on disk: {entry['path']}")
            continue
        actual_hash = sha256_file(src_path)
        if actual_hash != entry.get("sha256"):
            result.fail(f"STALE: source hash changed since last rebuild: {entry['path']}")

    # 6: current active source set == manifest source set (nothing added/removed).
    current_sources = {relative_source_path(p) for p, _ in discover_sources()}
    if current_sources != manifest_source_set:
        added = current_sources - manifest_source_set
        removed = manifest_source_set - current_sources
        detail = []
        if added:
            detail.append(f"on disk but not in manifest: {sorted(added)}")
        if removed:
            detail.append(f"in manifest but not on disk: {sorted(removed)}")
        result.fail("STALE: active source set differs from manifest (" + "; ".join(detail) + ")")

    if manifest.get("source_count") != len(manifest_sources):
        result.fail("INVALID: manifest source_count does not match its own sources list")
    if manifest.get("vector_count") != sum(s.get("chunk_count", 0) for s in manifest_sources):
        result.fail("INVALID: manifest vector_count does not match sum of per-source chunk_count")

    sqlite_path = chroma_dir / "chroma.sqlite3"
    if not sqlite_path.is_file():
        result.fail(f"MISSING: {sqlite_path}")
        return result

    try:
        con = _ro_connect(sqlite_path)
        try:
            cur = con.cursor()

            cur.execute("SELECT id FROM collections WHERE name = ?", (COLLECTION_NAME,))
            row = cur.fetchone()
            if row is None:
                result.fail(f"MISSING: collection {COLLECTION_NAME!r} not found in {sqlite_path}")
                return result
            collection_id = row[0]

            cur.execute(
                "SELECT id, type FROM segments WHERE collection = ? AND scope = 'METADATA'",
                (collection_id,),
            )
            meta_row = cur.fetchone()
            if meta_row is None:
                result.fail(f"INVALID: no METADATA segment for collection {COLLECTION_NAME!r}")
                return result
            metadata_segment_id = meta_row[0]

            cur.execute(
                "SELECT COUNT(*) FROM embeddings WHERE segment_id = ?",
                (metadata_segment_id,),
            )
            db_vector_count = cur.fetchone()[0]
            if db_vector_count != manifest.get("vector_count"):
                result.fail(
                    f"STALE: DB vector count = {db_vector_count}, "
                    f"manifest says {manifest.get('vector_count')}"
                )

            cur.execute(
                """
                SELECT DISTINCT em.string_value
                FROM embedding_metadata em
                JOIN embeddings e ON e.id = em.id
                WHERE e.segment_id = ? AND em.key = 'source'
                """,
                (metadata_segment_id,),
            )
            db_source_set = {r[0] for r in cur.fetchall() if r[0] is not None}
            if db_source_set != manifest_source_set:
                result.fail("STALE: DB metadata source set differs from manifest source set")

            cur.execute(
                """
                SELECT em.int_value, COUNT(*)
                FROM embedding_metadata em
                JOIN embeddings e ON e.id = em.id
                WHERE e.segment_id = ? AND em.key = 'box'
                GROUP BY em.int_value
                """,
                (metadata_segment_id,),
            )
            db_box_counts = {int(r[0]): r[1] for r in cur.fetchall() if r[0] is not None}
            if db_box_counts != manifest_box_counts:
                result.fail(
                    f"STALE: DB box counts = {db_box_counts}, manifest says {manifest_box_counts}"
                )

            cur.execute(
                "SELECT id FROM segments WHERE collection = ? AND scope = 'VECTOR'",
                (collection_id,),
            )
            vec_row = cur.fetchone()
            if vec_row is None:
                result.fail(f"INVALID: no VECTOR segment for collection {COLLECTION_NAME!r}")
                return result
            vector_segment_id = vec_row[0]
        finally:
            con.close()
    except Exception as e:  # pragma: no cover - defensive: any sqlite read failure is INVALID
        result.fail(f"INVALID: could not read {sqlite_path}: {e}")
        return result

    segment_dir = chroma_dir / vector_segment_id
    required_files = ("data_level0.bin", "header.bin", "length.bin", "link_lists.bin")
    missing = [name for name in required_files if not (segment_dir / name).is_file()]
    if missing:
        result.fail(f"MISSING: HNSW segment files missing under {segment_dir}: {missing}")

    return result


# ──────────────────────────────────────────────
# API KEY HANDLING (never printed, never logged, never on the command line)
# ──────────────────────────────────────────────
def resolve_api_key(interactive: bool) -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key

    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key

    if not interactive:
        raise SystemExit(
            "GEMINI_API_KEY not found in the environment or .env, and this run is "
            "non-interactive - cannot prompt for a secret. Set GEMINI_API_KEY and retry."
        )

    key = getpass.getpass("Enter GEMINI_API_KEY (input hidden, used only for this rebuild): ").strip()
    if not key:
        raise SystemExit("No API key entered - aborting rebuild.")
    return key


# ──────────────────────────────────────────────
# EMBEDDING BUILD (runs inside the child --_build-worker process)
# ──────────────────────────────────────────────
def _add_with_retry(vectorstore, batch, batch_num: int, total_batches: int) -> None:
    for attempt in range(1, MAX_TRIES + 1):
        try:
            vectorstore.add_documents(batch)
            return
        except Exception as e:
            msg = str(e)
            if any(marker in msg for marker in TRANSIENT_MARKERS) and attempt < MAX_TRIES:
                wait = 30 * attempt
                print(
                    f"   batch {batch_num}/{total_batches}: transient error, "
                    f"waiting {wait}s (attempt {attempt}/{MAX_TRIES - 1})...",
                    flush=True,
                )
                time.sleep(wait)
            else:
                raise


def _build_staging_index(staging_dir: Path, api_key: str) -> dict:
    """Build a brand-new Chroma index at *staging_dir*. Runs inside the
    child --_build-worker process; the caller is expected to exit right
    after this returns so every DB file handle is released for the parent.
    """
    os.environ["GOOGLE_API_KEY"] = api_key
    from langchain_community.vectorstores import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    plan = build_chunk_plan()
    problems = verify_expected_corpus(plan)
    if problems:
        raise RuntimeError("unexpected active corpus before embedding: " + "; ".join(problems))

    staging_dir.mkdir(parents=True, exist_ok=False)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(staging_dir),
    )

    total = plan.chunk_count
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Sources: {plan.source_count}", flush=True)
    print(f"Chunks: {total}", flush=True)
    for i, start in enumerate(range(0, total, BATCH_SIZE), 1):
        batch = plan.chunks[start:start + BATCH_SIZE]
        print(f"Embedding batch {i}/{total_batches}", flush=True)
        _add_with_retry(vectorstore, batch, i, total_batches)

    manifest = build_manifest(plan)
    write_manifest(manifest, staging_dir)
    return manifest


def _query_with_retry(vectorstore, query: str, k: int):
    """similarity_search_with_score with the same transient-error retry as
    the embedding build - a smoke-test query embeds the query text too, so
    it can hit the same free-tier rate limit right after a large batch."""
    for attempt in range(1, MAX_TRIES + 1):
        try:
            return vectorstore.similarity_search_with_score(query, k=k)
        except Exception as e:
            msg = str(e)
            if any(marker in msg for marker in TRANSIENT_MARKERS) and attempt < MAX_TRIES:
                wait = 30 * attempt
                print(f"   query retry: transient error, waiting {wait}s (attempt {attempt}/{MAX_TRIES - 1})...", flush=True)
                time.sleep(wait)
            else:
                raise


def _run_smoke_test(chroma_dir: Path, api_key: str) -> dict:
    """Raw retrieval smoke test against *chroma_dir*. Runs inside the child
    --_smoke-worker process. No DISTANCE_THRESHOLD filtering, no web
    fallback, no generative model - a direct Chroma similarity search only.
    """
    os.environ["GOOGLE_API_KEY"] = api_key
    from langchain_community.vectorstores import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    manifest = json.loads((chroma_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    active_sources = {s["path"] for s in manifest["sources"]}

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(chroma_dir),
    )

    rows = []
    ok = True
    for query in SMOKE_QUERIES:
        results = _query_with_retry(vectorstore, query, SMOKE_K)
        sources = [doc.metadata.get("source") for doc, _dist in results]
        distances = [dist for _doc, dist in results]
        query_ok = (
            len(results) >= 1
            and all(src in active_sources for src in sources)
            and all(isinstance(d, (int, float)) and d == d and abs(d) != float("inf") for d in distances)
        )
        ok = ok and query_ok
        rows.append({"query": query, "n": len(results), "sources": sources, "ok": query_ok})

    return {"ok": ok, "rows": rows}


# ──────────────────────────────────────────────
# CHILD-PROCESS WORKER ENTRY POINTS
# ──────────────────────────────────────────────
def _worker_build(staging_dir: Path) -> int:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("build worker: GEMINI_API_KEY missing from worker environment", file=sys.stderr)
        return 2
    try:
        manifest = _build_staging_index(staging_dir, api_key)
    except Exception as e:
        print(f"build worker failed: {e}", file=sys.stderr)
        return 1
    print(_RESULT_MARKER + json.dumps({"ok": True, "vector_count": manifest["vector_count"]}))
    return 0


def _worker_smoke(chroma_dir: Path) -> int:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("smoke worker: GEMINI_API_KEY missing from worker environment", file=sys.stderr)
        return 2
    try:
        outcome = _run_smoke_test(chroma_dir, api_key)
    except Exception as e:
        print(f"smoke worker failed: {e}", file=sys.stderr)
        return 1
    print(_RESULT_MARKER + json.dumps(outcome))
    return 0 if outcome["ok"] else 1


def _spawn_worker(flag: str, target_dir: Path, api_key: str) -> tuple[int, dict | None, str]:
    """Run this same module as `python -m tools.rebuild_rag <flag> <dir>` in
    a child process with the API key only in that child's environment
    (never as an argv element), wait for it to exit completely, then parse
    its single result line. Returns (returncode, parsed_result_or_None, raw_stdout).
    """
    env = dict(os.environ)
    env["GEMINI_API_KEY"] = api_key
    proc = subprocess.run(
        [sys.executable, "-m", "tools.rebuild_rag", flag, str(target_dir)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)

    parsed = None
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            try:
                parsed = json.loads(line[len(_RESULT_MARKER):])
            except json.JSONDecodeError:
                parsed = None
    return proc.returncode, parsed, proc.stdout


# ──────────────────────────────────────────────
# SAFE STAGED REBUILD (parent process)
# ──────────────────────────────────────────────
def _print_plan(plan: ChunkPlan) -> None:
    print("Rebuild plan:")
    print(f"  active sources : {plan.source_count}")
    print(f"  total chunks   : {plan.chunk_count}")
    print(f"  box counts     : {plan.box_counts}")
    print(f"  embedding model: {EMBEDDING_MODEL}")
    print()
    print("The bundled RAG database is normally sufficient.")
    print("A rebuild is optional and uses the Google Gemini embedding API.")


def _confirm(yes: bool) -> bool:
    if yes:
        return True
    if not sys.stdin.isatty():
        print("Refusing to proceed without --yes in a non-interactive session.")
        return False
    resp = input("Proceed with rebuild? [y/N]: ").strip().lower()
    return resp in ("y", "yes")


def interrupted_rebuild_message(staging_dir: Path, backup_dir: Path) -> str | None:
    """None when neither transient directory exists; otherwise a message
    explaining that a previous rebuild was interrupted and must be inspected
    by hand rather than silently deleted or silently reused."""
    if not staging_dir.exists() and not backup_dir.exists():
        return None
    return (
        "A previous rebuild appears to have been interrupted:\n"
        f"  staging: {staging_dir} {'(exists)' if staging_dir.exists() else ''}\n"
        f"  backup : {backup_dir} {'(exists)' if backup_dir.exists() else ''}\n"
        "Inspect these directories by hand before retrying - if the backup is a good "
        "known-good chroma_db, restore it manually; otherwise remove the stale "
        "directory(ies) once you have confirmed there is nothing worth keeping."
    )


def rebuild(yes: bool = False) -> int:
    interrupted = interrupted_rebuild_message(STAGING_DIR, BACKUP_DIR)
    if interrupted:
        print(interrupted, file=sys.stderr)
        return 1

    print("Computing chunk plan (offline, no network call)...")
    plan = build_chunk_plan()
    problems = verify_expected_corpus(plan)
    if problems:
        print("STOP: active corpus does not match the accepted baseline:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    _print_plan(plan)
    if not _confirm(yes):
        print("Aborted - no changes made.")
        return 1

    api_key = resolve_api_key(interactive=sys.stdin.isatty())

    print("\nBuilding new index in staging (child process)...")
    rc, result, _ = _spawn_worker(_BUILD_WORKER_FLAG, STAGING_DIR, api_key)
    if rc != 0 or not result or not result.get("ok"):
        print("Build worker failed - staging left in place for inspection.", file=sys.stderr)
        return 1

    print("\nOffline-validating staged index...")
    validation = validate_offline(STAGING_DIR)
    if not validation.ok:
        print("Staged index failed offline validation:", file=sys.stderr)
        for m in validation.messages:
            print(f"  - {m}", file=sys.stderr)
        print("chroma_db/ was NOT touched.", file=sys.stderr)
        return 1
    print("  staged index: VALID")

    print("\nRunning retrieval smoke test against staged index (child process)...")
    rc, smoke, _ = _spawn_worker(_SMOKE_WORKER_FLAG, STAGING_DIR, api_key)
    if rc != 0 or not smoke or not smoke.get("ok"):
        print("Smoke test FAILED - chroma_db/ was NOT touched.", file=sys.stderr)
        if smoke:
            for row in smoke["rows"]:
                status = "OK" if row["ok"] else "FAIL"
                print(f"  [{status}] {row['query']} -> {row['n']} result(s)", file=sys.stderr)
        return 1
    print("  smoke test: PASS")
    for row in smoke["rows"]:
        print(f"    [{'OK' if row['ok'] else 'FAIL'}] {row['query']} -> {row['n']} result(s)")

    return swap_in(STAGING_DIR, CHROMA_DIR, BACKUP_DIR)


def swap_in(staging_dir: Path, target_dir: Path, backup_dir: Path) -> int:
    """Atomically (per-OS rename) replace *target_dir* with *staging_dir*,
    keeping *backup_dir* as a rollback copy until post-swap validation
    passes. Never leaves the repo with no chroma_db/, a half-built index,
    or both staging and final ambiguously active: any failure restores the
    previous *target_dir* from *backup_dir* before returning.
    """
    print("\nSwapping staged index into place...")
    had_existing = target_dir.exists()
    if had_existing:
        os.rename(target_dir, backup_dir)
    try:
        os.rename(staging_dir, target_dir)
    except Exception as e:
        print(f"Swap failed while moving staging into place: {e}", file=sys.stderr)
        if had_existing:
            os.rename(backup_dir, target_dir)
            print("Rolled back: previous index restored.", file=sys.stderr)
        return 1

    validation = validate_offline(target_dir)
    if not validation.ok:
        print("Installed index failed post-swap validation - rolling back:", file=sys.stderr)
        for m in validation.messages:
            print(f"  - {m}", file=sys.stderr)
        import shutil
        shutil.rmtree(target_dir, ignore_errors=True)
        if had_existing:
            os.rename(backup_dir, target_dir)
            print("Rolled back: previous index restored.", file=sys.stderr)
        return 1

    if had_existing:
        import shutil
        shutil.rmtree(backup_dir, ignore_errors=True)

    print(f"Swap complete. {target_dir} is now the newly rebuilt index.")
    print("Final validation: VALID")
    return 0


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.rebuild_rag",
        description=(
            "Safely rebuild the bundled Chroma RAG index. The bundled chroma_db/ is "
            "normally sufficient - this is optional and calls the Google Gemini "
            "embedding API (models/gemini-embedding-2)."
        ),
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Offline-check the currently installed chroma_db/ (no API key, no network, no mutation) and exit.",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the interactive confirmation prompt before making Gemini embedding calls.",
    )
    parser.add_argument(_BUILD_WORKER_FLAG, metavar="STAGING_DIR", help=argparse.SUPPRESS)
    parser.add_argument(_SMOKE_WORKER_FLAG, metavar="CHROMA_DIR", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    build_worker_dir = getattr(args, "_build_worker")
    smoke_worker_dir = getattr(args, "_smoke_worker")

    if build_worker_dir:
        return _worker_build(Path(build_worker_dir))
    if smoke_worker_dir:
        return _worker_smoke(Path(smoke_worker_dir))

    if args.validate_only:
        result = validate_offline(CHROMA_DIR)
        if result.ok:
            print("VALID")
            return 0
        print("INVALID:")
        for m in result.messages:
            print(f"  - {m}")
        return 1

    return rebuild(yes=args.yes)


if __name__ == "__main__":
    raise SystemExit(main())
