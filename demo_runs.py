"""demo_runs.py — bundled, curated evaluation demo runs (Kush).

WHAT THIS IS
------------
On the `eval-demo-runs` branch a small, curated set of FINISHED,
successful full-pipeline runs ships inside the repository (under
`demo_runs/<run_id>/`) so a teammate can:

    git clone ...
    git switch eval-demo-runs
    streamlit run ui.py

and immediately see those runs in History / Switch run / Resume — no live
Gemini/Claude call, no RAG rebuild, no manual import step.

This is NOT a second history subsystem. `pipeline.persistence.list_runs`
and `load_state`, `run_history.py`'s own listing, and every screen built on
them (`ui.py`'s resume picker and run switcher, the History workspace view)
already work purely off whatever is physically sitting under
`persistence.runs_dir()`. So the only thing this module does is put the
bundled, sanitized checkpoint files there, once, before that code ever
runs — after that, a bundled run is indistinguishable from a run the
teammate produced themselves, and every existing code path already knows
how to show it. See `EVAL_DEMO_RUNS.md` for what each bundled run is and
why it was chosen.

NOT PIPELINE CODE, NOT A TEST FIXTURE. Nothing here makes an LLM call.
`seed_bundled_demo_runs` only copies files that are already checked into
the repository under `demo_runs/`; on a branch/checkout that has no
`demo_runs/` directory (the normal `experiment/claude-opus5` source
branch, for instance) every bundled id is simply absent and this is a
silent no-op.

IDEMPOTENT AND NON-DESTRUCTIVE
-------------------------------
`seed_bundled_demo_runs` is called on every app start (see ui.py) and must
therefore be cheap and safe to call over and over:

  * a run id already present under `runs_dir()` — whether from a PRIOR
    seed, or because the teammate's own new run happened to land there
    (astronomically unlikely: run ids are a UTC timestamp plus a random
    suffix) — is left completely alone. Nothing here ever overwrites an
    existing run directory.
  * seeding one run is itself an atomic publish (build in a sibling temp
    directory, then one `os.replace`), the same shape
    `persistence.save_state` uses one level down for a single file — so a
    reader (`list_runs`, mid-copy) never sees a half-seeded run directory.

SANITIZATION HAPPENS ONCE, AT AUTHORING TIME
----------------------------------------------
`sanitize_checkpoint` is what produced the files under `demo_runs/` in the
first place — run once, by hand, against a real local checkpoint, with the
sanitized JSON committed as the bundled artifact. It is exported here
(rather than left as a throwaway script) so it stays testable and so the
exact transform a bundled run went through is documented in code, not only
in a commit message. `seed_bundled_demo_runs` does NOT re-sanitize on
every copy — the committed file already IS the sanitized, final artifact;
re-deriving it from a live local run on every teammate's machine would
defeat the point of bundling a fixed, reviewed snapshot.
"""
from __future__ import annotations

import copy
import os
import shutil
import tempfile
from pathlib import Path

from pipeline.persistence import runs_dir

# Repo-root-relative source of the bundled snapshots — a plain copy of the
# real `.cache/runs/<run_id>/` layout `persistence.py` already uses, so
# seeding is nothing more than "copy this directory if it is not already
# there". Anchored on this file, the same way `persistence._DEFAULT_RUNS_DIR`
# anchors on itself, so it resolves the same regardless of cwd.
BUNDLE_ROOT = Path(__file__).resolve().parent / "demo_runs"

# The ONLY run ids this module will ever place on disk. A single, explicit
# allowlist (not "whatever happens to be under demo_runs/") so a stray file
# added to that directory can never silently start seeding — see
# EVAL_DEMO_RUNS.md for what each one is, its provider/model, and why it
# was selected.
BUNDLED_RUN_IDS: tuple[str, ...] = (
    "20260824T141045Z-e1cdb3ee",  # Gemini flash-lite — e-commerce monolith, PASS
)


def _bundle_dir(run_id: str) -> Path:
    return BUNDLE_ROOT / run_id


def sanitize_checkpoint(data: dict) -> dict:
    """Return a portable copy of one parsed checkpoint dict. Pure.

    Narrowly targeted at the ONE field a real recorded run can carry that is
    genuinely machine-local and purely informational:
    `repo_representation.meta.clone_path`, the absolute local path a
    brownfield run's shallow clone lived at (see `pipeline.state.RepoMeta`
    and `pipeline.agents.repo_ingestor`) — read only for a live repo
    drill-down against a clone that will not exist on a teammate's machine
    either way, so replacing it costs a bundled run nothing. Kept portable
    the same way `ui_demo.py`'s own demo fixture already does
    (`.cache/repos/bugged-shop`): the trailing directory name survives (so
    the value still reads as a real clone path), the machine-specific
    prefix does not.

    Deliberately NOT a blanket regex substitution across the whole JSON
    tree: architecture prose, ADR text, or a repository's own file paths
    are real recorded output, and a "looks like a path" rule run over all
    of it risks corrupting genuine content rather than a machine artifact.
    Anything else that needs sanitizing belongs here as its own explicit,
    narrow rule — never a generic scrub — and gets checked in by the
    security-scan test alongside this run's own bundled file.
    """
    out = copy.deepcopy(data)
    meta = (out.get("repo_representation") or {}).get("meta")
    if isinstance(meta, dict) and meta.get("clone_path"):
        name = Path(str(meta["clone_path"])).name
        meta["clone_path"] = f".cache/repos/{name}" if name else ""
    return out


def _copy_run_into_place(src: Path, dest: Path) -> None:
    """Publish one bundled run directory at `dest`, atomically.

    `dest`'s PARENT existing and `dest` itself NOT existing are both the
    caller's (`seed_bundled_demo_runs`'s) responsibility to have checked;
    this only does the copy-then-replace. Only `*.json` checkpoint files are
    copied — a bundled run carries nothing else — so an incidental stray
    file under a bundle directory (a README, say) is never published into
    the live runs store.
    """
    tmp_dir = Path(tempfile.mkdtemp(dir=dest.parent, prefix=f".{dest.name}-seed-"))
    try:
        for item in sorted(src.glob("*.json")):
            shutil.copy2(item, tmp_dir / item.name)
        os.replace(tmp_dir, dest)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def seed_bundled_demo_runs() -> list[str]:
    """Copy every bundled run into the LIVE `runs_dir()` if not already
    there. Safe to call on every app start — see the module docstring's
    IDEMPOTENT AND NON-DESTRUCTIVE section.

    Returns the run ids actually copied THIS call — empty on every call
    after the first, once seeding has already happened, and empty on a
    checkout with no `demo_runs/` directory at all (the normal source
    branch).
    """
    seeded: list[str] = []
    for run_id in BUNDLED_RUN_IDS:
        src = _bundle_dir(run_id)
        if not src.is_dir():
            continue  # not on this branch/checkout — nothing to seed
        dest = runs_dir() / run_id
        if dest.exists():
            continue  # already seeded, or the id is otherwise claimed
        dest.parent.mkdir(parents=True, exist_ok=True)
        _copy_run_into_place(src, dest)
        seeded.append(run_id)
    return seeded
