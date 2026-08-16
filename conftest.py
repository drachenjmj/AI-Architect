"""conftest.py — keep the test suite out of the real checkpoint cache (Kati).

WHY THIS EXISTS
---------------
State-on-disk is ON by default: `run_pipeline` checkpoints every transition to
`.cache/runs/` (see pipeline/persistence.py). That is what we want in the app —
but it means any test that drives the pipeline would leave a real run behind,
and the UI's "Resume run" picker reads that same directory. Without this file,
running the suite fills a developer's picker with runs about selling sneakers.

So: point every test at one throwaway directory for the session and delete it
afterwards. Nothing is mocked or disabled — checkpointing still runs for real,
it just writes somewhere disposable, which is also what makes the assertions in
test_persistence.py trustworthy.

Tests that need their own isolated runs directory (test_persistence.py does)
simply set the same environment variable again; `persistence.runs_dir()` reads
it on every call, so the innermost setting wins.
"""
from __future__ import annotations

import os
import shutil
import tempfile

_ENV_VAR = "AI_ARCHITECT_RUNS_DIR"
_tmp_dir: str | None = None


def pytest_configure(config):  # noqa: ARG001 — pytest hook signature
    """Redirect checkpoints to a temp directory before any test runs."""
    global _tmp_dir
    _tmp_dir = tempfile.mkdtemp(prefix="aiarch_pytest_runs_")
    os.environ[_ENV_VAR] = _tmp_dir


def pytest_unconfigure(config):  # noqa: ARG001 — pytest hook signature
    """Throw the temp checkpoints away once the session ends."""
    if _tmp_dir:
        shutil.rmtree(_tmp_dir, ignore_errors=True)
    os.environ.pop(_ENV_VAR, None)
