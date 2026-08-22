"""run_history.py — READ-ONLY browsing of saved runs (Kati).

WHAT THIS IS
------------
The History view's entire data layer. The orchestrator already checkpoints
every transition of every run to `.cache/runs/<run_id>/NNN_stage.json`
(see pipeline/persistence.py); this module reads those same files back and
NOTHING else. There is no second persistence system here — no SQLite, no
vector store, no index to keep in sync — because the directory IS the index,
which is the whole design of the checkpoint layout.

THE ONE RULE: READ-ONLY
-----------------------
Nothing here writes, repairs, deletes or migrates a checkpoint; nothing here
calls the pipeline, an LLM, a repository ingest or Chroma. History answers
"what has this machine already architected?" from disk, and a historical run
it hands back is a display object, never a resume handle — the UI keeps it
strictly separate from `st.session_state["state"]` (see ui_workspace.py).

DISCOVERY AND "LATEST USABLE"
-----------------------------
`list_history_runs` does the cheap pass: for each run directory it reads ONLY
the newest checkpoint, as plain JSON (no Pydantic validation), and extracts
summary metadata. `persistence.load_state` deliberately RAISES when a run's
newest checkpoint is corrupt; History instead walks backwards through the
audit trail until a checkpoint parses — a damaged last write must not hide
the fifteen good ones below it. A run with no usable checkpoint at all is
skipped, never raised on: one malformed folder must not cost the user the
whole list.

CHRONOLOGY
----------
Runs are ordered by when they were last ACTIVE, newest first. The primary
source is the SAVED timestamp of the run's last trace step
(`history[-1].timestamp`) — a fact the checkpoint states about itself, which
survives file copies that reset mtimes. Filesystem mtime is used only as a
fallback for checkpoints with no trace (e.g. a lone `001_created.json`),
and the summary records which of the two it used (`from_saved_timestamp`).
`run_id` — which starts with a UTC timestamp — breaks ties.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pipeline.persistence import runs_dir
from pipeline.state import ArchitectState

log = logging.getLogger(__name__)

# Mirrors persistence._CHECKPOINT_RE (kept private there; this module must
# not modify pipeline code, so the naming convention is restated here). If
# the checkpoint NAMING ever changes, persistence.save_state is the writer
# to check — this pattern must accept exactly what that writes.
_CHECKPOINT_RE = re.compile(r"^(\d+)_([a-z_]+)\.json$")

_EXCERPT_CHARS = 100


class HistoryError(RuntimeError):
    """A historical run could not be read back. Raised only by `load_history_run`."""


@dataclass(frozen=True)
class HistorySummary:
    """One History list row — compact metadata from the run's newest usable
    checkpoint. Every field is what the checkpoint states; nothing is
    fabricated, and missing values degrade to neutral fallbacks
    ("" / 0 / False) rather than guesses.
    """

    run_id: str
    updated_at: str            # ISO — last trace step's timestamp, else file mtime
    from_saved_timestamp: bool # False => updated_at came from filesystem mtime
    project_label: str         # context/blueprint project name, else prompt excerpt
    repo_name: str             # short repo name; "" for greenfield
    stage: str                 # pipeline stage value, e.g. "done"
    verdict: str               # review overall_status; "" when never reviewed
    accepted: bool             # stage is accepted / accepted_at recorded
    feature_count: int
    adr_count: int
    component_count: int
    kb_chunk_count: int


# ── directory layout ──────────────────────────────────────────────────────
def _run_dir(run_id: str) -> Path:
    """Directory for one run, with `run_id` validated as a plain directory
    name — same rule as `persistence._run_dir`: a run_id selected in the UI
    is untrusted input and must not resolve outside the runs directory."""
    if (
        not run_id
        or run_id in {".", ".."}
        or set(run_id) & set("/\\")
        or os.path.isabs(run_id)
    ):
        raise HistoryError(f"Invalid run_id: {run_id!r}")
    return runs_dir() / run_id


def _checkpoints(run_id: str) -> list[tuple[int, Path]]:
    """Every valid checkpoint file of one run as (sequence, path), oldest
    first. Stray files that do not match the naming convention are ignored,
    exactly as `persistence._checkpoints` ignores them."""
    directory = _run_dir(run_id)
    if not directory.is_dir():
        return []
    found = []
    for path in directory.glob("*.json"):
        match = _CHECKPOINT_RE.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


# ── the cheap discovery pass ──────────────────────────────────────────────
def list_history_runs() -> list[HistorySummary]:
    """Summarise every readable run, NEWEST FIRST — the History list's data.

    Reads only the newest USABLE checkpoint per run directory, as plain JSON.
    A run whose checkpoints are all unreadable is skipped rather than raised
    on; a missing runs directory is simply no runs, which is why the History
    empty state is an empty state and not an error.
    """
    root = runs_dir()
    if not root.is_dir():
        return []

    summaries: list[HistorySummary] = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        summary = _summarize_run(directory.name)
        if summary is not None:
            summaries.append(summary)

    summaries.sort(key=_chronology_key, reverse=True)
    return summaries


def _summarize_run(run_id: str) -> HistorySummary | None:
    """One run's summary from its newest usable checkpoint, or None to skip.

    Tolerant by design: a checkpoint only has to parse as a JSON object to be
    usable here (full schema validation happens on the load path), and every
    field is read through neutral fallbacks so a checkpoint from an older
    schema version still produces a row rather than an exception.
    """
    for _seq, path in reversed(_checkpoints(run_id)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — one bad file, not a bad list
            log.warning("history: skipping unreadable checkpoint %s: %s", path, exc)
            continue
        if not isinstance(data, dict):
            continue
        return _summary_from_checkpoint(run_id, data, path)
    return None


def _summary_from_checkpoint(
    run_id: str, data: dict, path: Path
) -> HistorySummary:
    """Extract the row's metadata from one parsed checkpoint dict."""

    request = _as_dict(data.get("initial_request"))
    record = _as_dict(data.get("context_record"))
    blueprint = _as_dict(data.get("blueprint"))
    repo = _as_dict(data.get("repo_representation"))
    repo_meta = _as_dict(repo.get("meta"))
    review = _as_dict(data.get("review"))
    history = data.get("history") or []

    # Last activity: the run's own statement about itself first (the last
    # trace step's timestamp); filesystem mtime only when there is no trace
    # (a lone created-checkpoint, e.g. a crash before the first transition).
    updated_at, from_saved = "", False
    if history and isinstance(history[-1], dict):
        updated_at = str(history[-1].get("timestamp") or "")
        from_saved = bool(updated_at)
    if not updated_at:
        updated_at = datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(timespec="seconds")

    prompt = str(request.get("raw_prompt") or "")
    return HistorySummary(
        run_id=run_id,
        updated_at=updated_at,
        from_saved_timestamp=from_saved,
        project_label=(
            str(record.get("project_name") or "").strip()
            or str(blueprint.get("project_name") or "").strip()
            or _excerpt(prompt)
            or "Untitled run"
        ),
        repo_name=_repo_short_name(
            str(repo_meta.get("url") or "").strip()
            or str(request.get("repo_url") or "").strip()
        ),
        stage=str(data.get("stage") or ""),
        verdict=str(review.get("overall_status") or ""),
        accepted=(
            str(data.get("stage") or "") == "accepted"
            or bool(str(data.get("accepted_at") or ""))
        ),
        feature_count=len(data.get("features") or []),
        adr_count=len(data.get("adrs") or []),
        component_count=len(data.get("components") or []),
        kb_chunk_count=len(data.get("retrieved_knowledge") or []),
    )


def _chronology_key(summary: HistorySummary) -> tuple:
    """Sortable (instant, run_id) — parsed timestamp, not string compare,
    because ISO strings with and without microseconds do not order
    lexicographically. An unparseable timestamp sorts oldest rather than
    crashing the list."""
    try:
        when = datetime.fromisoformat(summary.updated_at)
    except ValueError:
        when = datetime.min.replace(tzinfo=timezone.utc)
    return (when, summary.run_id)


# ── the on-demand load ────────────────────────────────────────────────────
def load_history_run(run_id: str) -> ArchitectState:
    """Rebuild one historical run from its newest USABLE checkpoint.

    Unlike `persistence.load_state` (the RESUME path, which demands the
    newest file and fails loudly when it is corrupt), this walks backwards
    through the audit trail until a checkpoint validates — a historical run
    with a damaged last write is still worth reading at the last good state.
    Raises `HistoryError` when nothing in the directory validates.
    """
    checkpoints = _checkpoints(run_id)
    if not checkpoints:
        raise HistoryError(f"No checkpoints found for run {run_id!r} in {runs_dir()}")

    for _seq, path in reversed(checkpoints):
        try:
            return ArchitectState.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception as exc:  # noqa: BLE001 — try the next-oldest
            log.warning(
                "history: checkpoint %s unreadable, walking back: %s", path, exc
            )
    raise HistoryError(
        f"No readable checkpoint for run {run_id!r} in {runs_dir()}"
    )


# ── small helpers ─────────────────────────────────────────────────────────
def _as_dict(value: object) -> dict:
    """`value` if it is a dict, else {} — checkpoints from older schema
    versions may omit whole sections."""
    return value if isinstance(value, dict) else {}


def _repo_short_name(url: str) -> str:
    """The repository's short name from its URL; '' when there is none."""
    return url.rstrip("/").rsplit("/", 1)[-1] if url else ""


def _excerpt(raw_prompt: str) -> str:
    """One-line, length-capped prompt excerpt — the project label of last
    resort for a run that never produced a Context Record."""
    flat = " ".join(str(raw_prompt).split())
    return flat if len(flat) <= _EXCERPT_CHARS else flat[: _EXCERPT_CHARS - 1] + "…"
