"""repo_analysis.py — deterministic repo analysis toolbox (Malte). 0 LLM tokens.

Everything in this module is PLAIN CODE. It turns a shallow clone on disk into
the `structure` half of RepoRepresentation; the `behavior` half is the LLM's
job (see agents/repo_ingestor.py). Keeping the two apart is the token-economy
principle: whatever code can derive, code derives.

WHAT LIVES HERE
---------------
  ensure_clone(url)            git clone --depth 1 into .cache/repos/ (reused
                               across runs; .cache is already gitignored).
  build_file_tree(root)        rendered tree, source files annotated with LOC.
  detect_tech_stack(root)      languages (LOC), deps + frameworks from
                               manifests, external services from docker files.
  build_dependency_edges(root) intra-repo import graph (Python-first, via ast;
                               other languages: contribute a parser later).
  render_mermaid(edges)        coarse top-level Mermaid diagram FROM the edges.
  build_repo_map(root, edges)  aider-style map: most-imported files first,
                               signatures only, hard char budget.
  find_integration_interface(root)  condensed OpenAPI/Swagger surface, best
                               effort — "" when the repo ships no spec.

Every function takes the clone ROOT so all of it can also run against any
local directory (that is how the self-test below analyses this very repo).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

from pipeline.state import DependencyEdge, TechStack

# ── constants ─────────────────────────────────────────────────────────────
# Clones live under the project root; `.cache` is covered by .gitignore.
CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "repos"

# Directories that never contribute to any analysis (vendored/generated code).
SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env",
    "__pycache__", ".idea", ".vscode", "dist", "build", ".cache",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", "site-packages",
    ".next", "target", "vendor", "coverage", "htmlcov",
}

LANG_BY_EXT = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".java": "Java", ".go": "Go",
    ".rb": "Ruby", ".rs": "Rust", ".php": "PHP", ".cs": "C#", ".c": "C",
    ".h": "C", ".cpp": "C++", ".hpp": "C++", ".kt": "Kotlin",
    ".swift": "Swift", ".scala": "Scala", ".sql": "SQL", ".html": "HTML",
    ".css": "CSS", ".scss": "CSS", ".vue": "Vue",
}

# dependency-name (lowercase) -> display name. Used to surface frameworks out
# of the flat dependency list. Deliberately small; extend as repos demand.
_FRAMEWORKS = {
    "fastapi": "FastAPI", "flask": "Flask", "django": "Django",
    "streamlit": "Streamlit", "react": "React", "vue": "Vue",
    "angular": "Angular", "express": "Express", "next": "Next.js",
    "svelte": "Svelte", "spring-boot": "Spring Boot", "rails": "Ruby on Rails",
    "langchain": "LangChain", "langgraph": "LangGraph",
    "sqlalchemy": "SQLAlchemy", "celery": "Celery", "torch": "PyTorch",
    "tensorflow": "TensorFlow", "pandas": "pandas", "numpy": "NumPy",
}

# docker image name fragment -> external service it implies.
_SERVICE_IMAGES = {
    "postgres": "PostgreSQL", "mysql": "MySQL", "mariadb": "MariaDB",
    "redis": "Redis", "rabbitmq": "RabbitMQ", "kafka": "Kafka",
    "mongo": "MongoDB", "elasticsearch": "Elasticsearch",
    "memcached": "Memcached", "minio": "MinIO", "nginx": "Nginx",
    "traefik": "Traefik",
}

# ── repo URL recognition (shared: field validation + free-text extraction) ─
# ONE definition of "what counts as a repo URL", used two ways so the two can
# never drift:
#   is_repo_url()    — hard, anchored full-match for the dedicated UI field.
#   REPO_URL_SEARCH  — unanchored search for a URL pasted into free-text prompt.
_REPO_HOSTS = r"(?:github\.com|gitlab\.com|bitbucket\.org)"
REPO_URL_SEARCH = re.compile(rf"https?://(?:www\.)?{_REPO_HOSTS}/[\w.-]+/[\w.-]+")


def is_repo_url(url: str) -> bool:
    """True iff `url` is a complete, well-formed GitHub/GitLab/Bitbucket repo URL."""
    return bool(re.fullmatch(rf"https?://(?:www\.)?{_REPO_HOSTS}/[\w.-]+/[\w.-]+/?", url.strip()))


# ── ingestion: shallow clone ──────────────────────────────────────────────
def ensure_clone(url: str, cache_dir: Path = CACHE_DIR) -> tuple[Path, str]:
    """Shallow-clone `url` into the cache, or reuse an existing clone.

    Returns (clone_path, commit_sha). Raises RuntimeError when git fails —
    the ingestor node treats that as "continue without repo", never as a
    pipeline failure.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tail = url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git") or "repo"
    name = re.sub(r"[^\w.-]", "_", tail)
    digest = hashlib.sha1(url.encode()).hexdigest()[:10]  # same URL -> same dir
    path = cache_dir / f"{name}-{digest}"

    if not (path / ".git").exists():
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, str(path)],
            capture_output=True,
            text=True,
            timeout=300,
            # Never let git open an interactive credential prompt (would hang
            # the pipeline on private/nonexistent repos) — fail fast instead.
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        if result.returncode != 0:
            raise RuntimeError(f"git clone failed: {result.stderr.strip()[:300]}")

    sha = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=30,
    ).stdout.strip()
    return path, sha


# ── shared walking helpers ────────────────────────────────────────────────
def _iter_files(root: Path):
    """All analysable files under root, skipping vendored/generated dirs."""
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        yield p


def _loc(path: Path) -> int:
    """Non-blank lines of code. 0 on unreadable files (binary etc.)."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0
    return sum(1 for line in text.splitlines() if line.strip())


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


# ── structure: file tree ──────────────────────────────────────────────────
def build_file_tree(root: Path, max_lines: int = 400) -> str:
    """Rendered directory tree; source files annotated with their LOC.

    Output is TEXT because its only consumer is the LLM prompt — a drawn tree
    is the most token-compact form. Truncated with a note past `max_lines`.
    """
    lines: list[str] = [f"{root.name}/"]

    def walk(d: Path, prefix: str) -> None:
        try:
            entries = sorted(
                (e for e in d.iterdir() if e.name not in SKIP_DIRS),
                key=lambda e: (e.is_file(), e.name.lower()),
            )
        except OSError:
            return
        for i, e in enumerate(entries):
            last = i == len(entries) - 1
            connector = "└── " if last else "├── "
            if e.is_dir():
                lines.append(f"{prefix}{connector}{e.name}/")
                walk(e, prefix + ("    " if last else "│   "))
            else:
                note = f"  ({_loc(e)} LOC)" if e.suffix in LANG_BY_EXT else ""
                lines.append(f"{prefix}{connector}{e.name}{note}")

    walk(root, "")
    if len(lines) > max_lines:
        omitted = len(lines) - max_lines
        lines = lines[:max_lines] + [f"… ({omitted} more entries omitted)"]
    return "\n".join(lines)


# ── structure: tech stack ─────────────────────────────────────────────────
def _dep_name(spec: str) -> str:
    """'fastapi[all]>=0.100  # comment' -> 'fastapi' (PEP-508-ish prefix)."""
    m = re.match(r"\s*([A-Za-z0-9_.-]+)", spec)
    return m.group(1).lower() if m else ""


def detect_tech_stack(root: Path) -> TechStack:
    """Languages by LOC + direct deps/frameworks/services from manifest files."""
    stack = TechStack()
    deps: set[str] = set()
    services: set[str] = set()

    for f in _iter_files(root):
        lang = LANG_BY_EXT.get(f.suffix)
        if lang:
            stack.languages[lang] = stack.languages.get(lang, 0) + _loc(f)

        name = f.name.lower()
        try:
            if name.startswith("requirements") and f.suffix in (".txt", ".in"):
                for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if line and not line.startswith(("#", "-")):
                        deps.add(_dep_name(line))
            elif name == "pyproject.toml":
                import tomllib
                data = tomllib.loads(f.read_text(encoding="utf-8", errors="ignore"))
                for spec in data.get("project", {}).get("dependencies", []):
                    deps.add(_dep_name(spec))
            elif name == "package.json":
                data = json.loads(f.read_text(encoding="utf-8", errors="ignore"))
                for key in ("dependencies", "devDependencies"):
                    deps.update(k.lower() for k in data.get(key, {}))
            elif name.startswith("docker-compose") and f.suffix in (".yml", ".yaml"):
                import yaml
                data = yaml.safe_load(f.read_text(encoding="utf-8", errors="ignore")) or {}
                for svc in (data.get("services") or {}).values():
                    image = str((svc or {}).get("image", ""))
                    for fragment, display in _SERVICE_IMAGES.items():
                        if fragment in image:
                            services.add(display)
            elif name == "dockerfile":
                for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.strip().lower().startswith("from "):
                        image = line.split()[1].lower()
                        for fragment, display in _SERVICE_IMAGES.items():
                            if fragment in image:
                                services.add(display)
        except Exception:
            continue  # one broken manifest never blocks the rest

    deps.discard("")
    stack.dependencies = sorted(deps)
    stack.frameworks = sorted({_FRAMEWORKS[d] for d in deps if d in _FRAMEWORKS})
    stack.external_services = sorted(services)
    return stack


# ── structure: dependency graph (Python-first) ────────────────────────────
def build_dependency_edges(root: Path) -> list[DependencyEdge]:
    """Intra-repo import edges: source file -> imported repo file.

    Python only for now (stdlib `ast`, fully deterministic). Imports that do
    not resolve to a file INSIDE the repo (stdlib, third-party) are dropped —
    external deps are the tech stack's job, this graph is about coupling.
    """
    py_files = [f for f in _iter_files(root) if f.suffix == ".py"]

    # Module index: dotted module name -> repo-relative file. A package is
    # addressable both as pkg.mod and as pkg (via its __init__.py).
    index: dict[str, str] = {}
    for f in py_files:
        rel = f.relative_to(root)
        index[".".join(rel.with_suffix("").parts)] = rel.as_posix()
        if rel.name == "__init__.py" and rel.parent.parts:
            index[".".join(rel.parent.parts)] = rel.as_posix()

    edges: set[tuple[str, str]] = set()
    for f in py_files:
        rel = _rel(f, root)
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue

        targets: set[str] = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                targets.update(a.name for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                if n.level == 0 and n.module:
                    base_mod = n.module
                elif n.level > 0:
                    # relative import: resolve against this file's package
                    pkg = f.relative_to(root).parent.parts
                    parts = pkg[: len(pkg) - (n.level - 1)]
                    base_mod = ".".join(parts + ((n.module,) if n.module else ()))
                else:
                    continue
                if base_mod:
                    targets.add(base_mod)
                    # `from pkg import x` may import the SUBMODULE pkg.x, not
                    # just a name — index both; unresolvable ones drop out below.
                    targets.update(f"{base_mod}.{a.name}" for a in n.names)

        for target in targets:
            # longest-prefix match: `a.b.c` may resolve via `a.b` (a package)
            candidate = target
            while candidate:
                hit = index.get(candidate)
                if hit and hit != rel:
                    edges.add((rel, hit))
                    break
                candidate = candidate.rpartition(".")[0]

    return [DependencyEdge(source=s, target=t) for s, t in sorted(edges)]


def render_mermaid(edges: list[DependencyEdge]) -> str:
    """Coarse Mermaid diagram: edges aggregated to top-level dirs/files.

    File-level graphs explode visually; the top-level view is the
    "architecture sketch". The fine-grained truth stays in dependency_edges.
    """
    def bucket(path: str) -> str:
        parts = path.split("/")
        return parts[0] if len(parts) > 1 else parts[0]

    agg = {(bucket(e.source), bucket(e.target)) for e in edges}
    agg = {(s, t) for s, t in agg if s != t}
    if not agg:
        return ""

    def node_id(name: str) -> str:
        return re.sub(r"\W", "_", name)

    lines = ["graph LR"]
    lines += [f"    {node_id(s)}[{s}] --> {node_id(t)}[{t}]" for s, t in sorted(agg)]
    return "\n".join(lines)


# ── structure: repo map (aider-style, Python-first) ───────────────────────
def _signatures(path: Path) -> list[str]:
    """Top-level defs/classes (+ method names) — the file's 'table of contents'."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return []

    def args_of(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        return ", ".join(a.arg for a in fn.args.args)

    out: list[str] = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(f"def {n.name}({args_of(n)})")
        elif isinstance(n, ast.ClassDef):
            out.append(f"class {n.name}:")
            out += [
                f"    def {m.name}({args_of(m)})"
                for m in n.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
    return out


def build_repo_map(root: Path, edges: list[DependencyEdge], max_chars: int = 8000) -> str:
    """aider-style map: most-imported files first, signatures only.

    Ranking = import in-degree (how many repo files depend on it), then LOC —
    a cheap, deterministic stand-in for aider's PageRank that orders files by
    how central they are. Hard char budget keeps the artifact injectable.
    """
    in_degree = Counter(e.target for e in edges)
    py_files = sorted(
        (f for f in _iter_files(root) if f.suffix == ".py"),
        key=lambda f: (-in_degree.get(_rel(f, root), 0), -_loc(f)),
    )

    blocks: list[str] = []
    used = 0
    for f in py_files:
        sigs = _signatures(f)
        if not sigs:
            continue
        block = _rel(f, root) + ":\n" + "\n".join(f"    {s}" for s in sigs)
        if used + len(block) > max_chars:
            blocks.append(f"… (map truncated at {max_chars} chars)")
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


# ── structure: integration interface (best effort) ───────────────────────
def find_integration_interface(root: Path, max_chars: int = 4000) -> str:
    """Condensed API surface from an OpenAPI/Swagger spec, if the repo ships one.

    Best effort by design (frozen decision): no spec -> "" — the Architect can
    drill down on demand instead. No framework-specific route parsing here.
    """
    spec_re = re.compile(r"(openapi|swagger)", re.IGNORECASE)
    for f in _iter_files(root):
        if not spec_re.search(f.stem) or f.suffix not in (".json", ".yaml", ".yml"):
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
            if f.suffix == ".json":
                spec = json.loads(text)
            else:
                import yaml
                spec = yaml.safe_load(text)
        except Exception:
            continue
        if not isinstance(spec, dict) or "paths" not in spec:
            continue

        info = spec.get("info", {})
        lines = [f"API: {info.get('title', '?')} v{info.get('version', '?')} (from {_rel(f, root)})"]
        for route, methods in spec["paths"].items():
            if not isinstance(methods, dict):
                continue
            for method, op in methods.items():
                if method.lower() not in ("get", "post", "put", "patch", "delete", "head", "options"):
                    continue
                summary = (op or {}).get("summary", "") if isinstance(op, dict) else ""
                lines.append(f"  {method.upper():6s} {route}" + (f" — {summary}" if summary else ""))
        rendered = "\n".join(lines)
        return rendered[:max_chars] + ("\n… (truncated)" if len(rendered) > max_chars else "")
    return ""


# ── quick self-test: `python -m pipeline.repo_analysis` ──────────────────
# Analyses THIS repo (no clone, no network, no LLM) and prints each artifact
# abridged — proves the whole deterministic layer works on a real codebase.
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]

    tree = build_file_tree(project_root)
    print("── file_tree (first 25 lines) ──")
    print("\n".join(tree.splitlines()[:25]))

    stack = detect_tech_stack(project_root)
    print("\n── tech_stack ──")
    print(f"  languages : {stack.languages}")
    print(f"  frameworks: {stack.frameworks}")
    print(f"  services  : {stack.external_services}")
    print(f"  deps      : {len(stack.dependencies)} found")

    edges = build_dependency_edges(project_root)
    print(f"\n── dependency_edges ({len(edges)}) ──")
    for e in edges[:10]:
        print(f"  {e.source} -> {e.target}")

    print("\n── mermaid ──")
    print(render_mermaid(edges) or "(empty)")

    repo_map = build_repo_map(project_root, edges)
    print("\n── repo_map (first 30 lines) ──")
    print("\n".join(repo_map.splitlines()[:30]))

    api = find_integration_interface(project_root)
    print(f"\n── integration_interface ──\n{api or '(none found — expected here)'}")
