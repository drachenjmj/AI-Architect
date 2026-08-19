"""persistence.py — state on disk: checkpoint every transition, resume any run (Kati).

WHAT THIS IS
------------
The state object already survives a JSON round-trip (`ArchitectState` is a
Pydantic model — see the self-test at the bottom of state.py). This module is
the thin, *deterministic* layer that puts those round-trips on disk so a run
outlives the process that started it: close the browser mid-clarification,
reopen it, pick the run, keep going.

There is NO LLM call in here and there never should be. Persistence is
plumbing, and plumbing belongs to code — the same rule the orchestrator lives
by.

LAYOUT (no manifest, no index, no database)
-------------------------------------------
    .cache/runs/<run_id>/001_created.json
                         002_ingesting.json
                         003_awaiting_input.json   <- newest = the resume point

One file per transition, sequence-numbered and zero-padded so a plain
alphabetical listing is chronological. The directory IS the index: `list_runs`
globs it, `load_state` reads the highest-numbered file. Nothing to keep in
sync, nothing to corrupt separately from the data, and the earlier files are a
free append-only audit trail of exactly how the run got where it is.

DURABILITY
----------
Every write is atomic: content goes to a temp file in the *same* directory,
gets flushed, and is then moved into place with `os.replace`. A reader
therefore only ever sees a whole checkpoint or no checkpoint — never a
half-written one. If anything goes wrong the temp file is removed, so a
`.tmp` never survives a failed write.

IMMUTABILITY
------------
`save_state` reads its argument and returns a `Path`. It never mutates the
state it is handed — the same contract `run_pipeline` keeps.

WHERE THE FILES GO
------------------
`<repo root>/.cache/runs`, which is already gitignored: checkpoints are local
artifacts, not repo content. Set `AI_ARCHITECT_RUNS_DIR` to point somewhere
else (tests do exactly that, so a test run never touches the real cache).
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from pipeline.state import ArchitectState

log = logging.getLogger(__name__)

# `.cache/` sits at the repo root, next to ui.py. Anchoring on THIS file rather
# than the cwd means checkpoints land in one place whether the caller is
# `streamlit run ui.py`, `python -m pipeline.run`, or pytest from anywhere.
_DEFAULT_RUNS_DIR = Path(__file__).resolve().parent.parent / ".cache" / "runs"

# `001_awaiting_input.json` — the sequence number is what orders a run's files;
# the stage is there so a human can read the trail without opening anything.
_CHECKPOINT_RE = re.compile(r"^(\d+)_([a-z_]+)\.json$")

_EXCERPT_CHARS = 100


class CheckpointError(RuntimeError):
    """A checkpoint could not be read back. Raised only by `load_state`."""


class RunSummary(NamedTuple):
    """One line for the resume picker.

    A NamedTuple, so callers can unpack it positionally
    (`run_id, stage, updated_at, excerpt = summary`) or read it by name.
    """

    run_id: str
    stage: str
    updated_at: str
    raw_prompt_excerpt: str


# ── where ────────────────────────────────────────────────────────────────
def runs_dir() -> Path:
    """Root directory holding one sub-directory per run.

    Resolved on every call (never cached) so tests can redirect it with
    `monkeypatch.setenv` without reloading the module.
    """
    override = os.environ.get("AI_ARCHITECT_RUNS_DIR")
    return Path(override) if override else _DEFAULT_RUNS_DIR


def _run_dir(run_id: str) -> Path:
    """Directory for one run, with `run_id` validated as a plain directory name.

    A run_id reaches this function from a UI selection or a CLI argument, so it
    is treated as untrusted input: anything with a separator or `..` in it is
    rejected rather than resolved into a path outside the runs directory.
    """
    if not run_id or run_id in {".", ".."} or set(run_id) & set("/\\") or os.path.isabs(run_id):
        raise CheckpointError(f"Invalid run_id: {run_id!r}")
    return runs_dir() / run_id


def _checkpoints(run_id: str) -> list[tuple[int, Path]]:
    """Every valid checkpoint file of one run as (sequence, path), oldest first.

    Files that do not match the naming convention (strays, leftovers, anything
    a human dropped in) are ignored rather than treated as checkpoints.
    """
    directory = _run_dir(run_id)
    if not directory.is_dir():
        return []
    found = []
    for path in directory.glob("*.json"):
        match = _CHECKPOINT_RE.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


# ── write ────────────────────────────────────────────────────────────────
def save_state(state: ArchitectState) -> Path:
    """Append one checkpoint for `state` and return the file it was written to.

    The sequence number continues from whatever is already on disk, so calling
    this after every transition builds the audit trail; the file written last
    is by definition the resume point. Does not mutate `state`.
    """
    directory = _run_dir(state.run_id)
    directory.mkdir(parents=True, exist_ok=True)

    existing = _checkpoints(state.run_id)
    seq = (existing[-1][0] + 1) if existing else 1
    target = directory / f"{seq:03d}_{state.stage.value}.json"

    payload = state.model_dump_json(indent=2)

    # Atomic publish: full content lands in a temp file in the SAME directory
    # (so the final move is a rename, not a cross-device copy), then replaces
    # the target in one indivisible step. Readers never see a partial file.
    handle, tmp_name = tempfile.mkstemp(dir=directory, prefix=f".{seq:03d}_", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())  # content is on disk before the rename
        os.replace(tmp_path, target)
    except BaseException:
        # Failed write leaves nothing behind — no .tmp, no half-checkpoint.
        tmp_path.unlink(missing_ok=True)
        raise
    return target


def checkpoint(state: ArchitectState) -> Path | None:
    """`save_state` that can never take the pipeline down.

    This is the form the orchestrator calls. Losing a checkpoint costs us the
    ability to resume; it must never cost us the run itself, so a full disk or
    a read-only directory is logged and swallowed. Returns the path written, or
    None if the checkpoint was lost.
    """
    try:
        return save_state(state)
    except Exception as exc:  # noqa: BLE001 — deliberately total
        log.warning("checkpoint failed for run %s: %s", state.run_id, exc)
        return None


# ── read ─────────────────────────────────────────────────────────────────
def load_state(run_id: str) -> ArchitectState:
    """Rebuild the state of `run_id` from its newest checkpoint.

    Raises `CheckpointError` — with the offending path in the message — when
    the run is unknown or its newest checkpoint cannot be parsed. Loudly, on
    purpose: the caller asked for one specific run, and silently handing back
    an older or empty state would be worse than saying so.
    """
    files = _checkpoints(run_id)
    if not files:
        raise CheckpointError(f"No checkpoints found for run {run_id!r} in {runs_dir()}")

    path = files[-1][1]
    try:
        return ArchitectState.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CheckpointError(f"Checkpoint {path} is unreadable or corrupt: {exc}") from exc


def list_runs() -> list[RunSummary]:
    """Summarise every resumable run, newest first — the resume picker's data.

    Reads only the newest checkpoint per run directory (the resume point).
    A run whose newest checkpoint is corrupt or unreadable is SKIPPED rather
    than raised on: one damaged file must never stop the user from seeing and
    resuming their other runs.
    """
    root = runs_dir()
    if not root.is_dir():
        return []

    summaries: list[RunSummary] = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        files = _checkpoints(directory.name)
        if not files:
            continue
        path = files[-1][1]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            prompt = data["initial_request"]["raw_prompt"]
            stage = data["stage"]
            updated_at = datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat(timespec="seconds")
        except Exception as exc:  # noqa: BLE001 — one bad file, not a bad list
            log.warning("skipping unreadable checkpoint %s: %s", path, exc)
            continue
        summaries.append(
            RunSummary(directory.name, str(stage), updated_at, _excerpt(prompt))
        )

    # Newest first. run_id starts with a UTC timestamp, so it is a meaningful
    # tie-break when two runs share an mtime second.
    summaries.sort(key=lambda s: (s.updated_at, s.run_id), reverse=True)
    return summaries


def _excerpt(raw_prompt: str) -> str:
    """One-line, length-capped version of the prompt, for a picker label."""
    flat = " ".join(str(raw_prompt).split())
    return flat if len(flat) <= _EXCERPT_CHARS else flat[: _EXCERPT_CHARS - 1] + "…"


# ── quick self-test: `python -m pipeline.persistence` ────────────────────
if __name__ == "__main__":
    from pipeline.state import Stage, new_run

    s = new_run("We need an event-driven redesign of our checkout.")
    print("saved:", save_state(s))
    s.log_step("clarifier", Stage.AWAITING_HUMAN, "asked 2 questions")
    print("saved:", save_state(s))

    assert load_state(s.run_id) == s
    print("round-trip OK")

    for row in list_runs()[:5]:
        print(f"  {row.updated_at}  {row.stage:15s} {row.raw_prompt_excerpt}")
