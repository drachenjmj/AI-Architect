"""setup_app.py - canonical one-click application setup + launcher.

This is the testable logic behind SETUP_AND_RUN.bat / scripts/setup_and_run.ps1,
runnable directly once a Python 3.12 virtualenv with the pinned dependencies
already exists:

    python -m tools.setup_app

It offline-validates the bundled RAG index (reusing tools.rebuild_rag - never
duplicating that logic), guides runtime-provider selection (Google Gemini or
University of Cologne KI Connect), securely collects only the credential(s)
still missing, updates the setup-managed keys in the local .env file (leaving
every other line untouched), and launches Streamlit.

Nothing here makes a generative or embedding API call, and nothing here talks
to KI Connect - the only network activity this module can ever trigger is an
explicit RAG rebuild handoff to tools.rebuild_rag (Gemini embeddings only),
which itself never runs without the user choosing it.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

from tools import rebuild_rag

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"
ENV_EXAMPLE_PATH = REPO_ROOT / ".env.example"

# The runtime-provider identifiers this repo actually supports, read from the
# single existing source of truth rather than duplicated here.
from pipeline.llm import PROVIDERS  # noqa: E402 - after REPO_ROOT/local setup above

PROVIDER_LABELS: dict[str, str] = {
    "google": "Google Gemini",
    "kiconnect": "University of Cologne KI Connect",
}

GEMINI_THINKING_HIGH = "high"
# Matches the commented default in .env.example - reused verbatim rather than
# invented, per the "use existing defaults" setup policy.
KICONNECT_DEFAULT_MODEL = "OpenAI GPT OSS 120b KI:Inferenz.nrw"

# The ONLY .env keys this setup tool is allowed to add/change/remove. Every
# other line (comments, unrelated settings) is preserved byte-for-byte.
SETUP_MANAGED_KEYS = (
    "GEMINI_API_KEY",
    "KICONNECT_API_KEY",
    "ARCHITECT_LLM_PROVIDER",
    "ARCHITECT_LLM_MODEL",
    "REVIEWER_LLM_PROVIDER",
    "REVIEWER_LLM_MODEL",
    "ARCHITECT_GEMINI_THINKING_LEVEL",
    "REVIEWER_GEMINI_THINKING_LEVEL",
)


# ──────────────────────────────────────────────
# .env FILE MANAGEMENT (atomic, preserves unrelated content)
# ──────────────────────────────────────────────
def read_env_dict(path: Path = ENV_PATH) -> dict[str, str]:
    """Current key/value pairs in *path* (empty dict if it doesn't exist).

    Never logs or prints anything - callers are responsible for keeping any
    secret value out of output themselves.
    """
    if not path.is_file():
        return {}
    from dotenv import dotenv_values
    return {k: v for k, v in dotenv_values(path).items() if v is not None}


_ENV_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=")


def update_env_file(updates: dict[str, str | None], path: Path = ENV_PATH) -> None:
    """Atomically set/remove only the given keys in *path*.

    updates[key] = a string  -> that key's line is set/replaced (added at the
                                 end if it was not already present).
    updates[key] = None      -> that key's line is removed entirely.
    Every other line (comments, blank lines, unrelated keys, ordering) is
    preserved exactly. Written via temp-file + os.replace so a crash mid-write
    can never corrupt .env or leave a secret-bearing partial file behind.
    """
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        m = _ENV_KEY_RE.match(line)
        if m and m.group(1) in updates:
            key = m.group(1)
            seen.add(key)
            value = updates[key]
            if value is not None:
                out.append(f"{key}={value}")
            # value is None -> line dropped (unset)
        else:
            out.append(line)
    for key, value in updates.items():
        if key not in seen and value is not None:
            out.append(f"{key}={value}")

    content = ("\n".join(out) + "\n") if out else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".env-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def env_is_gitignored(repo_root: Path = REPO_ROOT) -> bool:
    """Whether .gitignore has an exact-match ``.env`` rule."""
    gitignore = repo_root / ".gitignore"
    if not gitignore.is_file():
        return False
    return any(
        line.strip() == ".env"
        for line in gitignore.read_text(encoding="utf-8").splitlines()
    )


# ──────────────────────────────────────────────
# CREDENTIAL HANDLING (never printed, never on argv)
# ──────────────────────────────────────────────
def resolve_credential(
    var_name: str, label: str, existing_env: dict[str, str], interactive: bool
) -> str | None:
    """`var_name`'s value from the process environment, then .env, then (only
    if interactive) a hidden prompt. Returns None if still unresolved."""
    val = os.environ.get(var_name, "").strip()
    if val:
        return val
    val = existing_env.get(var_name, "").strip()
    if val:
        return val
    if not interactive:
        return None
    val = getpass.getpass(f"Enter {label} (input hidden): ").strip()
    return val or None


def credential_present(var_name: str, existing_env: dict[str, str]) -> bool:
    return bool(os.environ.get(var_name, "").strip() or existing_env.get(var_name, "").strip())


def report_credential_status(var_name: str, existing_env: dict[str, str]) -> bool:
    """Print only WHETHER a credential was found - never its value."""
    found = credential_present(var_name, existing_env)
    if found:
        print(f"  {var_name}: found (reusing existing credential)")
    return found


# ──────────────────────────────────────────────
# PROVIDER CONFIGURATION
# ──────────────────────────────────────────────
def detect_existing_provider(env: dict[str, str]) -> str | None:
    """Which provider (if any) `.env` is currently FULLY configured to use.

    KI Connect needs BOTH credentials to be a complete, working config
    (Architect/Reviewer -> KI Connect, but Clarifier + RAG query embeddings
    always need Gemini) - a KI Connect routing pin without a Gemini key is
    not reported as "configured", so setup asks for the missing key instead
    of silently reusing a broken configuration.
    """
    arch_provider = env.get("ARCHITECT_LLM_PROVIDER", "").strip().lower()
    if (
        arch_provider == "kiconnect"
        and env.get("KICONNECT_API_KEY", "").strip()
        and env.get("GEMINI_API_KEY", "").strip()
    ):
        return "kiconnect"
    if arch_provider in ("", "google") and env.get("GEMINI_API_KEY", "").strip():
        return "google"
    return None


def apply_gemini_config(
    updates: dict[str, str | None], gemini_key: str | None, existing_env: dict[str, str]
) -> None:
    if gemini_key:
        updates["GEMINI_API_KEY"] = gemini_key
    # Google is the routing DEFAULT: role_model_override() falls back to
    # Gemini whenever {ROLE}_LLM_MODEL is unset. So selecting Gemini only
    # needs to CLEAR a role that currently points at KI Connect - a role
    # already explicitly pinned to Gemini (or unset) is left exactly as the
    # user set it, per "do not silently overwrite unrelated .env settings".
    if existing_env.get("ARCHITECT_LLM_PROVIDER", "").strip().lower() == "kiconnect":
        updates["ARCHITECT_LLM_PROVIDER"] = None
        updates["ARCHITECT_LLM_MODEL"] = None
    if existing_env.get("REVIEWER_LLM_PROVIDER", "").strip().lower() == "kiconnect":
        updates["REVIEWER_LLM_PROVIDER"] = None
        updates["REVIEWER_LLM_MODEL"] = None
    updates["ARCHITECT_GEMINI_THINKING_LEVEL"] = GEMINI_THINKING_HIGH
    updates["REVIEWER_GEMINI_THINKING_LEVEL"] = GEMINI_THINKING_HIGH


def apply_kiconnect_config(
    updates: dict[str, str | None],
    existing_env: dict[str, str],
    kiconnect_key: str | None,
    gemini_key: str | None = None,
) -> None:
    """KI Connect redirects ONLY the Architect/Reviewer generation calls
    (ARCHITECT_LLM_PROVIDER/REVIEWER_LLM_PROVIDER). The Clarifier/advisor
    (pipeline/agents/clarifier.py: CLARIFIER_MODEL/ADVISOR_MODEL = a bare
    "flash-lite" registry name, never routed through role_model_override)
    and every RAG query-time embedding call (architect.get_vectorstore() ->
    GoogleGenerativeAIEmbeddings, used by every retrieve_chunks() /
    similarity_search_with_score() call) always go through Gemini regardless
    of this routing - so a Gemini key is required for KI Connect too, not
    only for Google."""
    if kiconnect_key:
        updates["KICONNECT_API_KEY"] = kiconnect_key
    if gemini_key:
        updates["GEMINI_API_KEY"] = gemini_key
    updates["ARCHITECT_LLM_PROVIDER"] = "kiconnect"
    updates["REVIEWER_LLM_PROVIDER"] = "kiconnect"
    # Preserve an already-configured custom model override; only fill in the
    # repository's documented default when nothing is set yet.
    if not existing_env.get("ARCHITECT_LLM_MODEL", "").strip():
        updates["ARCHITECT_LLM_MODEL"] = KICONNECT_DEFAULT_MODEL
    if not existing_env.get("REVIEWER_LLM_MODEL", "").strip():
        updates["REVIEWER_LLM_MODEL"] = KICONNECT_DEFAULT_MODEL
    # Gemini thinking level is a Google-only seam - never combined with KI
    # Connect routing (see .env.example / pipeline/llm.py).
    updates["ARCHITECT_GEMINI_THINKING_LEVEL"] = None
    updates["REVIEWER_GEMINI_THINKING_LEVEL"] = None
    # KICONNECT_BASE_URL / MAX_TOKENS / TIMEOUT_SECONDS deliberately untouched:
    # pipeline/llm.py already has repository defaults for all three, and the
    # setup policy is to never ask for values that already have a good default.


def resolve_provider(args: argparse.Namespace, env: dict[str, str], interactive: bool) -> str | None:
    """The provider to configure this run, or None if none could be determined."""
    if args.provider:
        return args.provider

    existing = detect_existing_provider(env)
    if not interactive:
        return existing  # None if nothing configured yet - caller reports the failure

    if existing:
        print(f"\nExisting configuration found: {PROVIDER_LABELS[existing]}")
        print("[Enter] Use existing configuration")
        print("[C]     Change provider")
        choice = input("Selection: ").strip().lower()
        if choice != "c":
            return existing

    print("\nChoose the runtime LLM provider:\n")
    print("  [1] Google Gemini (recommended)")
    print("      General setup; requires a Gemini API key.\n")
    print("  [2] University of Cologne KI Connect")
    print("      Requires University of Cologne access.\n")
    while True:
        choice = input("Selection: ").strip()
        if choice == "1":
            return "google"
        if choice == "2":
            return "kiconnect"
        print("Please enter 1 or 2.")


# ──────────────────────────────────────────────
# RAG STATUS (offline - reuses tools.rebuild_rag entirely)
# ──────────────────────────────────────────────
def check_rag_status(chroma_dir: Path | None = None) -> tuple[str, dict | None]:
    """("VALID"|"STALE"|"MISSING_OR_INVALID", manifest-dict-or-None).

    Pure offline check via tools.rebuild_rag.validate_offline() - no API key,
    no network call, no mutation. STALE means the index directory/manifest/DB
    are all structurally present but out of sync (e.g. a source changed);
    MISSING_OR_INVALID covers everything more broken than that (missing
    directory, missing DB, corrupt manifest, ...).
    """
    chroma_dir = chroma_dir or rebuild_rag.CHROMA_DIR
    result = rebuild_rag.validate_offline(chroma_dir)
    if result.ok:
        manifest = None
        try:
            manifest = json.loads((chroma_dir / rebuild_rag.MANIFEST_NAME).read_text(encoding="utf-8"))
        except Exception:
            pass
        return "VALID", manifest

    structural = any(m.startswith(("MISSING:", "INVALID:")) for m in result.messages)
    if structural:
        return "MISSING_OR_INVALID", None
    return "STALE", None


def format_rag_status_line(status: str, manifest: dict | None) -> str:
    if status == "VALID" and manifest:
        return (
            f"Bundled RAG database: VALID "
            f"({manifest['source_count']} sources, {manifest['vector_count']} vectors)"
        )
    if status == "VALID":
        return "Bundled RAG database: VALID"
    if status == "STALE":
        return "Bundled RAG database: STALE (structurally usable, but out of date)"
    return "Bundled RAG database: MISSING or INVALID"


def prompt_stale_rag_choice(interactive: bool) -> str:
    """R/C/Q. Non-interactive default is "C" - the safe, explicit choice:
    the index is still structurally usable, so continuing spends nothing."""
    if not interactive:
        return "C"
    print("\n[R] Rebuild now")
    print("[C] Continue with the bundled index")
    print("[Q] Quit")
    choice = input("Selection [C]: ").strip().upper() or "C"
    return choice if choice in ("R", "C", "Q") else "C"


def prompt_missing_rag_choice(interactive: bool) -> str:
    """R/Q. Non-interactive default is "Q": never spend on an embedding
    rebuild without an explicit human choosing it."""
    if not interactive:
        return "Q"
    print("\n[R] Rebuild RAG database")
    print("[Q] Quit setup")
    choice = input("Selection [Q]: ").strip().upper() or "Q"
    return choice if choice in ("R", "Q") else "Q"


def handle_rebuild() -> bool:
    """Hand off to the canonical rebuild pipeline; True iff it ends VALID.

    Never duplicates source discovery/build/swap/validation logic - this is
    a thin call into tools.rebuild_rag, which owns all of that (including its
    own Gemini-key resolution for the embedding calls it makes).
    """
    gemini_present = credential_present("GEMINI_API_KEY", read_env_dict())
    if not gemini_present:
        print(
            "\nRebuilding the RAG database uses Gemini embeddings and "
            "therefore requires a Gemini API key."
        )
    rc = rebuild_rag.rebuild(yes=True)
    if rc != 0:
        return False
    return rebuild_rag.validate_offline(rebuild_rag.CHROMA_DIR).ok


# ──────────────────────────────────────────────
# LOCAL PRE-LAUNCH CHECKS (no network)
# ──────────────────────────────────────────────
def resolve_venv_python(repo_root: Path = REPO_ROOT) -> Path:
    windows = repo_root / ".venv" / "Scripts" / "python.exe"
    if windows.is_file():
        return windows
    posix = repo_root / ".venv" / "bin" / "python"
    if posix.is_file():
        return posix
    return Path(sys.executable)


def prelaunch_checks(repo_root: Path, provider: str | None, env: dict[str, str]) -> list[str]:
    problems: list[str] = []

    venv_python = resolve_venv_python(repo_root)
    if not venv_python.is_file():
        problems.append(f"venv Python not found at {venv_python}")

    if not (repo_root / "ui.py").is_file():
        problems.append("ui.py not found")

    import importlib.util
    if importlib.util.find_spec("streamlit") is None:
        problems.append("streamlit is not importable in this environment")

    if provider not in PROVIDERS:
        problems.append(f"no supported runtime provider selected (must be one of {PROVIDERS})")
    elif provider == "google" and not env.get("GEMINI_API_KEY", "").strip():
        problems.append("GEMINI_API_KEY is not set")
    elif provider == "kiconnect":
        if not env.get("KICONNECT_API_KEY", "").strip():
            problems.append("KICONNECT_API_KEY is not set")
        # KI Connect only redirects Architect/Reviewer generation - the
        # Clarifier and every RAG query-time embedding call still go through
        # Gemini (see apply_kiconnect_config's docstring), so a missing
        # Gemini key is just as fatal here as it is for the Google provider.
        if not env.get("GEMINI_API_KEY", "").strip():
            problems.append("GEMINI_API_KEY is not set (still required for the Clarifier and RAG query embeddings)")

    return problems


# ──────────────────────────────────────────────
# STREAMLIT LAUNCH
# ──────────────────────────────────────────────
def find_free_port(preferred: int = 8501, attempts: int = 20) -> int:
    port = preferred
    for _ in range(attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    raise RuntimeError(f"no free local port found in range {preferred}-{port - 1}")


def build_streamlit_command(python_exe: Path, repo_root: Path, port: int) -> list[str]:
    return [
        str(python_exe), "-m", "streamlit", "run", str(repo_root / "ui.py"),
        "--server.port", str(port),
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false",
    ]


def _suppress_streamlit_first_run_prompt() -> None:
    """Pre-seed an empty credentials.toml so Streamlit's first-run email
    prompt never blocks the evaluator. Never touches an existing file."""
    cred_path = Path.home() / ".streamlit" / "credentials.toml"
    if cred_path.exists():
        return
    try:
        cred_path.parent.mkdir(parents=True, exist_ok=True)
        cred_path.write_text('[general]\nemail = ""\n', encoding="utf-8")
    except OSError:
        pass  # best-effort convenience only - never block launch over it


def launch_streamlit(repo_root: Path = REPO_ROOT) -> int:
    python_exe = resolve_venv_python(repo_root)
    port = find_free_port()
    _suppress_streamlit_first_run_prompt()
    cmd = build_streamlit_command(python_exe, repo_root, port)
    print(f"\nStarting AI-Architect on http://localhost:{port} ...")
    proc = subprocess.run(cmd, cwd=str(repo_root))
    return proc.returncode


# ──────────────────────────────────────────────
# ORCHESTRATION
# ──────────────────────────────────────────────
def run_setup(args: argparse.Namespace, *, interactive: bool | None = None) -> int:
    if interactive is None:
        interactive = sys.stdin.isatty()

    print("AI-Architect Setup\n")

    status, manifest = check_rag_status()
    print(format_rag_status_line(status, manifest))

    if status == "STALE":
        choice = prompt_stale_rag_choice(interactive)
        if choice == "Q":
            print("Setup cancelled.")
            return 1
        if choice == "R":
            if not handle_rebuild():
                print("Rebuild failed or did not complete - not launching with an unverified RAG index.")
                return 1
    elif status == "MISSING_OR_INVALID":
        choice = prompt_missing_rag_choice(interactive)
        if choice != "R":
            print("Setup cancelled.")
            return 1
        if not handle_rebuild():
            print("Rebuild failed or did not complete - cannot launch without a valid RAG index.")
            return 1

    env = read_env_dict()
    provider = resolve_provider(args, env, interactive)
    if provider is None or provider not in PROVIDERS:
        print("\nNo runtime provider selected - aborting setup.")
        return 1

    updates: dict[str, str | None] = {}
    if provider == "google":
        report_credential_status("GEMINI_API_KEY", env)
        key = resolve_credential("GEMINI_API_KEY", "Gemini API key", env, interactive)
        if not key and not credential_present("GEMINI_API_KEY", env):
            print("\nA Gemini API key is required for the Google Gemini runtime provider.")
            return 1
        apply_gemini_config(updates, key, env)
        update_env_file(updates)
        print("\nRuntime provider: Google Gemini")
        print("Architect thinking: HIGH")
        print("Reviewer thinking: HIGH")
    else:
        # KI Connect only redirects Architect/Reviewer generation. The
        # Clarifier and every RAG query-time embedding call always go
        # through Gemini regardless of that routing (see
        # apply_kiconnect_config's docstring) - so BOTH credentials are
        # required, not just KICONNECT_API_KEY.
        report_credential_status("KICONNECT_API_KEY", env)
        ki_key = resolve_credential("KICONNECT_API_KEY", "KI Connect API key", env, interactive)
        if not ki_key and not credential_present("KICONNECT_API_KEY", env):
            print("\nA KI Connect API key is required for the University of Cologne KI Connect provider.")
            return 1

        report_credential_status("GEMINI_API_KEY", env)
        gem_key = resolve_credential("GEMINI_API_KEY", "Gemini API key", env, interactive)
        if not gem_key and not credential_present("GEMINI_API_KEY", env):
            print(
                "\nA Gemini API key is also required for KI Connect: the Clarifier "
                "and RAG query embeddings always use Gemini, independent of the "
                "Architect/Reviewer routing."
            )
            return 1

        apply_kiconnect_config(updates, env, ki_key, gem_key)
        update_env_file(updates)
        print("\nRuntime provider: University of Cologne KI Connect")
        print()
        print("KI Connect is used for Architect and Reviewer.")
        print("A Gemini API key is also required for the Clarifier and RAG query embeddings.")
        print("The bundled RAG database is already included, so no rebuild is required.")

    env = read_env_dict()
    problems = prelaunch_checks(REPO_ROOT, provider, env)
    if problems:
        print("\nSetup is incomplete:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("\nSetup complete.")

    if args.setup_only or args.no_launch:
        return 0

    return launch_streamlit(REPO_ROOT)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.setup_app",
        description="One-click AI-Architect application setup and launcher.",
    )
    parser.add_argument(
        "--provider", choices=PROVIDERS,
        help="Select the runtime provider non-interactively (for testing/advanced use).",
    )
    parser.add_argument(
        "--setup-only", action="store_true",
        help="Run setup/validation only; do not launch Streamlit.",
    )
    parser.add_argument(
        "--no-launch", action="store_true",
        help="Alias for --setup-only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_setup(args)


if __name__ == "__main__":
    raise SystemExit(main())
