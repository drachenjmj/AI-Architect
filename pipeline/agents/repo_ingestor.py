"""repo_ingestor.py — Repo-Ingestor agent (Malte). Stage 1 of the repo representation.

WHAT IT DOES (in order)
-----------------------
  1. Find a repo URL — DETERMINISTICALLY (regex over the raw prompt + any
     clarification answers). No URL -> greenfield run: write nothing, advance.
     Deliberately NOT read from the ContextRecord: its schema is still a
     placeholder, and the raw prompt is the ground truth anyway.
  2. Shallow-clone it into .cache/repos/ (reused across runs).
  3. Build the STRUCTURE layer — plain code, 0 LLM tokens
     (pipeline/repo_analysis.py). Each part is best-effort: one failing
     analysis leaves its field empty and never blocks the rest.
  4. Build the BEHAVIOR layer — the ONE LLM call here: overview + partition
     summaries, grounded in the deterministic artifacts, validated via
     `response_schema=RepoBehavior`.
  5. Write the write-once `repo_representation`; advance to INGESTING.

Failure philosophy: a dead URL, a broken manifest or a refused LLM call each
degrade the artifact but NEVER fail the run — an architecture pipeline without
repo insight is still more useful than no pipeline at all. Errors are recorded
in `state.errors` (reducer) so nothing is silently swallowed.

Stage 2 (lazy drill-downs -> `state.repo_deep_dives`) is the Architect's job,
not this node's; `meta.clone_path` tells it where to read files from.
"""
from __future__ import annotations

from pipeline.agents.base import make_step, node
from pipeline.llm import llm_call
from pipeline.repo_analysis import (
    REPO_URL_SEARCH,
    build_dependency_edges,
    build_file_tree,
    build_repo_map,
    detect_tech_stack,
    ensure_clone,
    find_integration_interface,
    render_mermaid,
)
from pipeline.state import (
    ArchitectState,
    RepoBehavior,
    RepoMeta,
    RepoRepresentation,
    RepoStructure,
    Stage,
)

# Summarising an unknown codebase correctly matters for everything downstream
# (Researcher queries, Architect design), so — like the Clarifier — this uses
# the stronger model, not the cheap default.
INGESTOR_MODEL = "flash"

# POLICY only; the output SHAPE is enforced by response_schema=RepoBehavior.
INGESTOR_SYSTEM = """\
You are the Repo-Ingestor in an automated software-architecture assistant. You
receive a DETERMINISTIC analysis of a repository (file tree with LOC, tech
stack, dependency graph, code map with signatures). Your only job is to
describe the FUNCTIONAL side of the system:

1. `overview` — a short, plain-language summary of what the whole repository
   does (its purpose, not its file layout).
2. `partitions` — at most 5. Partition the repo along ITS OWN visible
   structure (usually the top-level directories or obvious groupings in the
   file tree); do not impose an external scheme. For each partition give:
   its `name`, the repo-relative `paths` it covers (taken verbatim from the
   file tree), its `role` within the whole system, and its `functionality`
   in plain language.

Ground every statement in the provided analysis. If something is not evident
from it, say so plainly inside the text rather than inventing details. Never
invent paths that do not appear in the file tree.
"""

def _find_repo_url(state: ArchitectState) -> str:
    """Where the repo comes from: the explicit UI field first, prompt as fallback.

    The dedicated URL field (hard-validated in the UI) wins. When it is empty we
    still scan the raw prompt and any clarification answers, so a URL pasted into
    free text — or supplied only in a later answer — keeps working.
    """
    explicit = state.initial_request.repo_url.strip()
    if explicit:
        return explicit
    texts = [state.initial_request.raw_prompt, *state.clarification_answers.values()]
    for text in texts:
        m = REPO_URL_SEARCH.search(text)
        if m:
            return m.group(0).rstrip(".,;")
    return ""


def _build_behavior_prompt(structure: RepoStructure) -> str:
    """The deterministic artifacts ARE the prompt — nothing else is injected."""
    parts = [f"FILE TREE (with LOC):\n{structure.file_tree}"]
    stack = structure.tech_stack
    parts.append(
        "TECH STACK:\n"
        f"  languages: {stack.languages}\n"
        f"  frameworks: {stack.frameworks}\n"
        f"  external services: {stack.external_services}\n"
        f"  direct dependencies: {', '.join(stack.dependencies) or '(none found)'}"
    )
    if structure.repo_map:
        parts.append(f"CODE MAP (most-imported files first, signatures only):\n{structure.repo_map}")
    if structure.architecture_diagram:
        parts.append(f"TOP-LEVEL DEPENDENCY DIAGRAM (mermaid):\n{structure.architecture_diagram}")
    if structure.integration_interface:
        parts.append(f"API SURFACE (from OpenAPI spec):\n{structure.integration_interface}")
    return "\n\n".join(parts)


@node("repo_ingestor")
def repo_ingestor_node(state: ArchitectState) -> dict:
    # 1. URL — deterministic. None found ⇒ greenfield: skip, advance, done.
    url = _find_repo_url(state)
    if not url:
        step = make_step(
            "repo_ingestor", state.stage, Stage.INGESTING,
            "no repo URL in request — greenfield run, nothing to ingest",
        )
        return {"stage": Stage.INGESTING, "history": [step]}

    # 2. Shallow clone. Unreachable repo degrades, never fails the run.
    try:
        clone_path, sha = ensure_clone(url)
    except Exception as e:
        step = make_step(
            "repo_ingestor", state.stage, Stage.INGESTING,
            f"clone of {url} failed — continuing without repo insight",
        )
        return {
            "stage": Stage.INGESTING,
            "history": [step],
            "errors": [f"repo_ingestor: clone failed: {e}"],
        }

    # 3. STRUCTURE — deterministic, each part best-effort.
    errors: list[str] = []
    structure = RepoStructure()
    for field, builder in (
        ("file_tree", lambda: build_file_tree(clone_path)),
        ("tech_stack", lambda: detect_tech_stack(clone_path)),
        ("dependency_edges", lambda: build_dependency_edges(clone_path)),
    ):
        try:
            setattr(structure, field, builder())
        except Exception as e:
            errors.append(f"repo_ingestor: {field}: {e}")
    try:
        structure.architecture_diagram = render_mermaid(structure.dependency_edges)
    except Exception as e:
        errors.append(f"repo_ingestor: architecture_diagram: {e}")
    try:
        structure.repo_map = build_repo_map(clone_path, structure.dependency_edges)
    except Exception as e:
        errors.append(f"repo_ingestor: repo_map: {e}")
    try:
        structure.integration_interface = find_integration_interface(clone_path)
    except Exception as e:
        errors.append(f"repo_ingestor: integration_interface: {e}")

    # 4. BEHAVIOR — the one LLM call. A refusal leaves the layer empty.
    behavior = RepoBehavior()
    try:
        behavior = llm_call(
            state,
            _build_behavior_prompt(structure),
            system=INGESTOR_SYSTEM,
            model=INGESTOR_MODEL,
            response_schema=RepoBehavior,
        )
    except Exception as e:
        errors.append(f"repo_ingestor: behavior LLM call failed: {e}")

    # 5. Write the write-once artifact and advance.
    representation = RepoRepresentation(
        meta=RepoMeta(url=url, commit_sha=sha, clone_path=str(clone_path)),
        structure=structure,
        behavior=behavior,
    )
    step = make_step(
        "repo_ingestor", state.stage, Stage.INGESTING,
        f"ingested {url}@{sha[:7]}: {len(structure.dependency_edges)} import edge(s), "
        f"{len(behavior.partitions)} partition(s)"
        + (f", {len(errors)} part(s) degraded" if errors else ""),
    )
    out: dict = {
        "repo_representation": representation,
        "stage": Stage.INGESTING,
        "history": [step],
    }
    if errors:
        out["errors"] = errors
    return out


# ── Live smoke test: `python -m pipeline.agents.repo_ingestor` ────────────
# Clones a real public repo and makes ONE real LLM call (needs GEMINI_API_KEY).
# Not a unit test — it checks what mocks can't: clone + deterministic layer +
# schema-valid RepoBehavior end-to-end. Deterministic tests: test_repo_ingestor.py.
if __name__ == "__main__":
    from pipeline.state import new_run

    s = new_run(
        "Our team inherited this Flask-based service and needs it re-architected. "
        "Repo: https://github.com/pallets/flask"
    )
    out = repo_ingestor_node(s)
    rep: RepoRepresentation | None = out.get("repo_representation")
    print(f"stage   : {out['stage'].value}")
    print(f"note    : {out['history'][0].note}")
    if rep is not None:
        print(f"clone   : {rep.meta.clone_path} @ {rep.meta.commit_sha[:7]}")
        print(f"langs   : {rep.structure.tech_stack.languages}")
        print(f"edges   : {len(rep.structure.dependency_edges)}")
        print(f"overview: {rep.behavior.overview[:300]}")
        for p in rep.behavior.partitions:
            print(f"  partition {p.name!r}: paths={p.paths} — {p.role[:80]}")
    if out.get("errors"):
        print("degraded:", out["errors"])
    print(f"tokens  : {s.input_tokens}/{s.output_tokens}")
