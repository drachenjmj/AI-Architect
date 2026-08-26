"""test_setup_app.py - offline tests for the one-click application setup
helper (tools/setup_app.py).

Every .env fixture is a tmp_path file; no test reads or writes the real
`.env`, and no test calls a generative/embedding/KI-Connect/web endpoint.
`getpass.getpass` and `input` are always monkeypatched before any code path
that could reach them - on Windows, `getpass.getpass` reads directly from
the console via msvcrt and ignores redirected/closed stdin, so leaving it
un-mocked can HANG the test process rather than raise cleanly.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tools import setup_app as sa
from tools import rebuild_rag as rr

REPO_ROOT = sa.REPO_ROOT


# ── provider configuration ──────────────────────────────────────────────

def test_only_google_and_kiconnect_are_supported():
    assert sa.PROVIDERS == ("google", "kiconnect")


def test_no_anthropic_path_exists():
    assert "anthropic" not in sa.PROVIDERS
    assert "anthropic" not in sa.PROVIDER_LABELS
    with pytest.raises(SystemExit):
        sa.build_arg_parser().parse_args(["--provider", "anthropic"])


def test_unsupported_provider_cli_flag_fails_clearly(capsys):
    with pytest.raises(SystemExit) as exc:
        sa.build_arg_parser().parse_args(["--provider", "bogus"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_apply_gemini_config_sets_thinking_high_and_key():
    updates: dict = {}
    sa.apply_gemini_config(updates, "sk-fake-gemini", existing_env={})
    assert updates["GEMINI_API_KEY"] == "sk-fake-gemini"
    assert updates["ARCHITECT_GEMINI_THINKING_LEVEL"] == "high"
    assert updates["REVIEWER_GEMINI_THINKING_LEVEL"] == "high"


def test_apply_gemini_config_clears_prior_kiconnect_routing():
    existing = {
        "ARCHITECT_LLM_PROVIDER": "kiconnect",
        "ARCHITECT_LLM_MODEL": "OpenAI GPT OSS 120b KI:Inferenz.nrw",
        "REVIEWER_LLM_PROVIDER": "kiconnect",
        "REVIEWER_LLM_MODEL": "OpenAI GPT OSS 120b KI:Inferenz.nrw",
    }
    updates: dict = {}
    sa.apply_gemini_config(updates, None, existing_env=existing)
    assert updates["ARCHITECT_LLM_PROVIDER"] is None
    assert updates["ARCHITECT_LLM_MODEL"] is None
    assert updates["REVIEWER_LLM_PROVIDER"] is None
    assert updates["REVIEWER_LLM_MODEL"] is None


def test_apply_gemini_config_preserves_existing_explicit_google_pin():
    """A pre-existing explicit ARCHITECT_LLM_MODEL that already points at
    Google must survive - selecting Gemini is a no-op for it, not a reset."""
    existing = {
        "ARCHITECT_LLM_PROVIDER": "google",
        "ARCHITECT_LLM_MODEL": "gemini-3.1-flash-lite",
    }
    updates: dict = {}
    sa.apply_gemini_config(updates, None, existing_env=existing)
    assert "ARCHITECT_LLM_PROVIDER" not in updates
    assert "ARCHITECT_LLM_MODEL" not in updates


def test_apply_kiconnect_config_selects_kiconnect_route():
    updates: dict = {}
    sa.apply_kiconnect_config(updates, existing_env={}, kiconnect_key="sk-fake-ki")
    assert updates["KICONNECT_API_KEY"] == "sk-fake-ki"
    assert updates["ARCHITECT_LLM_PROVIDER"] == "kiconnect"
    assert updates["REVIEWER_LLM_PROVIDER"] == "kiconnect"
    assert updates["ARCHITECT_LLM_MODEL"] == sa.KICONNECT_DEFAULT_MODEL
    assert updates["REVIEWER_LLM_MODEL"] == sa.KICONNECT_DEFAULT_MODEL
    # No Gemini-only concept invented for KI Connect.
    assert updates["ARCHITECT_GEMINI_THINKING_LEVEL"] is None
    assert updates["REVIEWER_GEMINI_THINKING_LEVEL"] is None


def test_apply_kiconnect_config_preserves_custom_model_override():
    existing = {"ARCHITECT_LLM_MODEL": "some-custom-model"}
    updates: dict = {}
    sa.apply_kiconnect_config(updates, existing_env=existing, kiconnect_key=None)
    assert "ARCHITECT_LLM_MODEL" not in updates  # left as the user set it
    assert updates["REVIEWER_LLM_MODEL"] == sa.KICONNECT_DEFAULT_MODEL


def test_detect_existing_provider_google():
    assert sa.detect_existing_provider({"GEMINI_API_KEY": "x"}) == "google"


def test_detect_existing_provider_kiconnect():
    env = {"ARCHITECT_LLM_PROVIDER": "kiconnect", "KICONNECT_API_KEY": "x"}
    assert sa.detect_existing_provider(env) == "kiconnect"


def test_detect_existing_provider_none_when_unconfigured():
    assert sa.detect_existing_provider({}) is None


# ── .env file management ────────────────────────────────────────────────

def test_update_env_file_creates_new_file(tmp_path):
    path = tmp_path / ".env"
    sa.update_env_file({"GEMINI_API_KEY": "abc123"}, path=path)
    assert path.read_text(encoding="utf-8") == "GEMINI_API_KEY=abc123\n"


def test_update_env_file_preserves_unrelated_lines_and_comments(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "# a comment\n"
        "SOME_OTHER_SETTING=keep-me\n"
        "\n"
        "GEMINI_API_KEY=old-value\n",
        encoding="utf-8",
    )
    sa.update_env_file({"GEMINI_API_KEY": "new-value"}, path=path)
    content = path.read_text(encoding="utf-8")
    assert "# a comment" in content
    assert "SOME_OTHER_SETTING=keep-me" in content
    assert "GEMINI_API_KEY=new-value" in content
    assert "old-value" not in content


def test_update_env_file_removes_key_on_none(tmp_path):
    path = tmp_path / ".env"
    path.write_text("ARCHITECT_LLM_PROVIDER=kiconnect\nGEMINI_API_KEY=x\n", encoding="utf-8")
    sa.update_env_file({"ARCHITECT_LLM_PROVIDER": None}, path=path)
    content = path.read_text(encoding="utf-8")
    assert "ARCHITECT_LLM_PROVIDER" not in content
    assert "GEMINI_API_KEY=x" in content


def test_update_env_file_is_atomic_no_temp_file_left(tmp_path):
    path = tmp_path / ".env"
    path.write_text("GEMINI_API_KEY=x\n", encoding="utf-8")
    sa.update_env_file({"GEMINI_API_KEY": "y"}, path=path)
    leftovers = [p for p in tmp_path.iterdir() if p.name != ".env"]
    assert leftovers == []


def test_update_env_file_cleans_up_temp_file_on_write_failure(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("GEMINI_API_KEY=x\n", encoding="utf-8")

    def _boom(*a, **k):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(sa.os, "replace", _boom)
    with pytest.raises(OSError):
        sa.update_env_file({"GEMINI_API_KEY": "y"}, path=path)

    leftovers = [p for p in tmp_path.iterdir() if p.name != ".env"]
    assert leftovers == []  # no orphaned secret-bearing .env-*.tmp file
    assert path.read_text(encoding="utf-8") == "GEMINI_API_KEY=x\n"  # untouched


def test_read_env_dict_missing_file_returns_empty(tmp_path):
    assert sa.read_env_dict(tmp_path / "nope.env") == {}


def test_env_is_gitignored_true_for_real_repo():
    assert sa.env_is_gitignored(REPO_ROOT) is True


def test_env_is_gitignored_false_without_rule(tmp_path):
    (tmp_path / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
    assert sa.env_is_gitignored(tmp_path) is False


# ── credential safety ───────────────────────────────────────────────────

def test_resolve_credential_reuses_env_var_without_prompting(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-process-env")

    def _boom(*a, **k):
        raise AssertionError("must not prompt when already in os.environ")

    monkeypatch.setattr(sa.getpass, "getpass", _boom)
    result = sa.resolve_credential("GEMINI_API_KEY", "Gemini API key", {}, interactive=True)
    assert result == "from-process-env"


def test_resolve_credential_reuses_dotenv_value_without_prompting(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def _boom(*a, **k):
        raise AssertionError("must not prompt when already in .env")

    monkeypatch.setattr(sa.getpass, "getpass", _boom)
    result = sa.resolve_credential(
        "GEMINI_API_KEY", "Gemini API key", {"GEMINI_API_KEY": "from-dotenv"}, interactive=True
    )
    assert result == "from-dotenv"


def test_resolve_credential_prompts_securely_when_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    calls = []

    def _fake_getpass(prompt=""):
        calls.append(prompt)
        return "typed-secret"

    monkeypatch.setattr(sa.getpass, "getpass", _fake_getpass)
    result = sa.resolve_credential("GEMINI_API_KEY", "Gemini API key", {}, interactive=True)
    assert result == "typed-secret"
    assert len(calls) == 1


def test_resolve_credential_non_interactive_never_prompts(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def _boom(*a, **k):
        raise AssertionError("must not prompt when non-interactive")

    monkeypatch.setattr(sa.getpass, "getpass", _boom)
    result = sa.resolve_credential("GEMINI_API_KEY", "Gemini API key", {}, interactive=False)
    assert result is None


def test_report_credential_status_never_prints_the_value(capsys):
    found = sa.report_credential_status("GEMINI_API_KEY", {"GEMINI_API_KEY": "super-secret-value"})
    assert found is True
    out = capsys.readouterr().out
    assert "super-secret-value" not in out
    assert "found" in out.lower()


def test_report_credential_status_silent_when_absent(capsys, monkeypatch):
    # Other tests in the full suite call load_dotenv() (via architect.py /
    # pipeline.llm), which sets os.environ["GEMINI_API_KEY"] from the real
    # .env for the rest of the process (dotenv fills unset vars, it just
    # never overrides ones already set) - delenv here so this test's
    # "absent" premise holds regardless of run order.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    found = sa.report_credential_status("GEMINI_API_KEY", {})
    assert found is False
    assert capsys.readouterr().out == ""


# ── RAG integration ──────────────────────────────────────────────────────

def _fake_manifest(vector_count=503, source_count=11):
    return {"schema_version": 1, "vector_count": vector_count, "source_count": source_count}


def test_check_rag_status_valid(monkeypatch, tmp_path):
    ok_result = rr.ValidationResult(ok=True)
    monkeypatch.setattr(rr, "validate_offline", lambda d: ok_result)
    manifest_path = tmp_path / rr.MANIFEST_NAME
    manifest_path.write_text(json.dumps(_fake_manifest()), encoding="utf-8")
    status, manifest = sa.check_rag_status(tmp_path)
    assert status == "VALID"
    assert manifest["vector_count"] == 503


def test_check_rag_status_stale(monkeypatch, tmp_path):
    stale = rr.ValidationResult(ok=False, messages=["STALE: source hash changed since last rebuild: x"])
    monkeypatch.setattr(rr, "validate_offline", lambda d: stale)
    status, manifest = sa.check_rag_status(tmp_path)
    assert status == "STALE"
    assert manifest is None


def test_check_rag_status_missing(monkeypatch, tmp_path):
    missing = rr.ValidationResult(ok=False, messages=["MISSING: chroma directory does not exist"])
    monkeypatch.setattr(rr, "validate_offline", lambda d: missing)
    status, _ = sa.check_rag_status(tmp_path)
    assert status == "MISSING_OR_INVALID"


def test_prompt_stale_default_is_continue_non_interactive():
    assert sa.prompt_stale_rag_choice(interactive=False) == "C"


def test_prompt_missing_default_is_quit_non_interactive():
    assert sa.prompt_missing_rag_choice(interactive=False) == "Q"


def test_prompt_stale_reads_interactive_choice(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "R")
    assert sa.prompt_stale_rag_choice(interactive=True) == "R"


def test_prompt_missing_reads_interactive_choice(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "R")
    assert sa.prompt_missing_rag_choice(interactive=True) == "R"


def test_handle_rebuild_calls_canonical_rebuild_module(monkeypatch):
    calls = []
    monkeypatch.setattr(sa.rebuild_rag, "rebuild", lambda yes: calls.append(yes) or 0)
    monkeypatch.setattr(sa.rebuild_rag, "validate_offline", lambda d: rr.ValidationResult(ok=True))
    monkeypatch.setattr(sa, "read_env_dict", lambda path=sa.ENV_PATH: {"GEMINI_API_KEY": "x"})
    assert sa.handle_rebuild() is True
    assert calls == [True]


def test_handle_rebuild_reports_failure_when_rebuild_fails(monkeypatch):
    monkeypatch.setattr(sa.rebuild_rag, "rebuild", lambda yes: 1)
    monkeypatch.setattr(sa, "read_env_dict", lambda path=sa.ENV_PATH: {"GEMINI_API_KEY": "x"})
    assert sa.handle_rebuild() is False


def test_handle_rebuild_explains_gemini_requirement_when_key_missing(monkeypatch, capsys):
    monkeypatch.setattr(sa.rebuild_rag, "rebuild", lambda yes: 0)
    monkeypatch.setattr(sa.rebuild_rag, "validate_offline", lambda d: rr.ValidationResult(ok=True))
    monkeypatch.setattr(sa, "read_env_dict", lambda path=sa.ENV_PATH: {})
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    sa.handle_rebuild()
    out = capsys.readouterr().out
    assert "Gemini" in out and "embed" in out.lower()


def test_run_setup_valid_rag_never_calls_rebuild(monkeypatch, tmp_path):
    """Normal setup with a VALID bundled RAG makes no embedding call - the
    rebuild path must never even be reached."""
    def _boom(*a, **k):
        raise AssertionError("rebuild must not be called when RAG is VALID")

    monkeypatch.setattr(sa, "check_rag_status", lambda: ("VALID", _fake_manifest()))
    monkeypatch.setattr(sa, "handle_rebuild", _boom)
    monkeypatch.setattr(sa, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(sa, "read_env_dict", lambda path=None: {"GEMINI_API_KEY": "existing"})
    monkeypatch.setattr(sa, "update_env_file", lambda updates, path=None: None)
    monkeypatch.setattr(sa, "prelaunch_checks", lambda root, provider, env: [])

    args = sa.build_arg_parser().parse_args(["--provider", "google", "--setup-only"])
    rc = sa.run_setup(args, interactive=False)
    assert rc == 0


def test_run_setup_kiconnect_with_valid_rag_never_touches_gemini_key(monkeypatch, tmp_path):
    monkeypatch.setattr(sa, "check_rag_status", lambda: ("VALID", _fake_manifest()))
    monkeypatch.setattr(sa, "ENV_PATH", tmp_path / ".env")
    env = {"KICONNECT_API_KEY": "existing-ki-key"}  # deliberately NO GEMINI_API_KEY
    monkeypatch.setattr(sa, "read_env_dict", lambda path=None: env)
    captured_updates = {}
    monkeypatch.setattr(
        sa, "update_env_file",
        lambda updates, path=None: captured_updates.update(updates),
    )
    monkeypatch.setattr(sa, "prelaunch_checks", lambda root, provider, env: [])
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    args = sa.build_arg_parser().parse_args(["--provider", "kiconnect", "--setup-only"])
    rc = sa.run_setup(args, interactive=False)

    assert rc == 0
    assert "GEMINI_API_KEY" not in captured_updates


def test_run_setup_stale_rag_non_interactive_continues(monkeypatch, tmp_path):
    monkeypatch.setattr(sa, "check_rag_status", lambda: ("STALE", None))

    def _boom(*a, **k):
        raise AssertionError("rebuild must not run on the non-interactive STALE default")

    monkeypatch.setattr(sa, "handle_rebuild", _boom)
    monkeypatch.setattr(sa, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(sa, "read_env_dict", lambda path=None: {"GEMINI_API_KEY": "x"})
    monkeypatch.setattr(sa, "update_env_file", lambda updates, path=None: None)
    monkeypatch.setattr(sa, "prelaunch_checks", lambda root, provider, env: [])

    args = sa.build_arg_parser().parse_args(["--provider", "google", "--setup-only"])
    rc = sa.run_setup(args, interactive=False)
    assert rc == 0  # continued past the stale warning


def test_run_setup_missing_rag_non_interactive_quits(monkeypatch, tmp_path):
    monkeypatch.setattr(sa, "check_rag_status", lambda: ("MISSING_OR_INVALID", None))

    def _boom(*a, **k):
        raise AssertionError("rebuild must not run on the non-interactive MISSING default")

    monkeypatch.setattr(sa, "handle_rebuild", _boom)
    args = sa.build_arg_parser().parse_args(["--provider", "google", "--setup-only"])
    rc = sa.run_setup(args, interactive=False)
    assert rc == 1


def test_run_setup_missing_credential_fails_without_launching(monkeypatch, tmp_path):
    monkeypatch.setattr(sa, "check_rag_status", lambda: ("VALID", _fake_manifest()))
    monkeypatch.setattr(sa, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(sa, "read_env_dict", lambda path=None: {})
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def _boom(*a, **k):
        raise AssertionError("must not launch Streamlit without a credential")

    monkeypatch.setattr(sa, "launch_streamlit", _boom)
    args = sa.build_arg_parser().parse_args(["--provider", "google"])
    rc = sa.run_setup(args, interactive=False)
    assert rc == 1


# ── launch behavior ──────────────────────────────────────────────────────

def test_resolve_venv_python_windows_layout(tmp_path):
    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_bytes(b"")
    assert sa.resolve_venv_python(tmp_path) == venv_python


def test_resolve_venv_python_falls_back_to_sys_executable(tmp_path):
    assert sa.resolve_venv_python(tmp_path) == Path(sys.executable)


def test_build_streamlit_command_points_at_root_ui_py(tmp_path):
    python_exe = tmp_path / "python.exe"
    cmd = sa.build_streamlit_command(python_exe, tmp_path, 8501)
    assert cmd[0] == str(python_exe)
    assert cmd[1:4] == ["-m", "streamlit", "run"]
    assert cmd[4] == str(tmp_path / "ui.py")
    assert "8501" in cmd


def test_build_streamlit_command_never_contains_a_secret(tmp_path):
    cmd = sa.build_streamlit_command(tmp_path / "python.exe", tmp_path, 8501)
    joined = " ".join(cmd)
    assert "KICONNECT_API_KEY" not in joined
    assert "GEMINI_API_KEY" not in joined


def test_find_free_port_skips_an_occupied_port():
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    occupied_port = blocker.getsockname()[1]
    try:
        port = sa.find_free_port(preferred=occupied_port)
        assert port != occupied_port
        assert port > occupied_port
    finally:
        blocker.close()


def test_setup_only_flag_does_not_launch_streamlit(monkeypatch, tmp_path):
    monkeypatch.setattr(sa, "check_rag_status", lambda: ("VALID", _fake_manifest()))
    monkeypatch.setattr(sa, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(sa, "read_env_dict", lambda path=None: {"GEMINI_API_KEY": "x"})
    monkeypatch.setattr(sa, "update_env_file", lambda updates, path=None: None)
    monkeypatch.setattr(sa, "prelaunch_checks", lambda root, provider, env: [])

    def _boom(*a, **k):
        raise AssertionError("--setup-only must not launch Streamlit")

    monkeypatch.setattr(sa, "launch_streamlit", _boom)
    args = sa.build_arg_parser().parse_args(["--provider", "google", "--setup-only"])
    assert sa.run_setup(args, interactive=False) == 0


def test_no_launch_flag_does_not_launch_streamlit(monkeypatch, tmp_path):
    monkeypatch.setattr(sa, "check_rag_status", lambda: ("VALID", _fake_manifest()))
    monkeypatch.setattr(sa, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(sa, "read_env_dict", lambda path=None: {"GEMINI_API_KEY": "x"})
    monkeypatch.setattr(sa, "update_env_file", lambda updates, path=None: None)
    monkeypatch.setattr(sa, "prelaunch_checks", lambda root, provider, env: [])

    def _boom(*a, **k):
        raise AssertionError("--no-launch must not launch Streamlit")

    monkeypatch.setattr(sa, "launch_streamlit", _boom)
    args = sa.build_arg_parser().parse_args(["--provider", "google", "--no-launch"])
    assert sa.run_setup(args, interactive=False) == 0


# ── pre-launch checks ────────────────────────────────────────────────────

def test_prelaunch_checks_flags_missing_credential(tmp_path):
    (tmp_path / "ui.py").write_text("", encoding="utf-8")
    problems = sa.prelaunch_checks(tmp_path, "google", {})
    assert any("GEMINI_API_KEY" in p for p in problems)


def test_prelaunch_checks_passes_with_everything_present(monkeypatch, tmp_path):
    (tmp_path / "ui.py").write_text("", encoding="utf-8")
    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_bytes(b"")
    problems = sa.prelaunch_checks(tmp_path, "google", {"GEMINI_API_KEY": "x"})
    assert problems == []


def test_prelaunch_checks_flags_missing_ui_py(tmp_path):
    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_bytes(b"")
    problems = sa.prelaunch_checks(tmp_path, "google", {"GEMINI_API_KEY": "x"})
    assert any("ui.py" in p for p in problems)


# ── wrapper / bootstrap ───────────────────────────────────────────────────

def test_bat_wrapper_exists_and_delegates_to_powershell_script():
    bat = REPO_ROOT / "SETUP_AND_RUN.bat"
    assert bat.is_file()
    content = bat.read_text(encoding="utf-8")
    assert "setup_and_run.ps1" in content


def test_ps1_wrapper_exists_and_calls_canonical_module():
    ps1 = REPO_ROOT / "scripts" / "setup_and_run.ps1"
    assert ps1.is_file()
    content = ps1.read_text(encoding="utf-8")
    assert "tools.setup_app" in content
    assert "requirements.txt" in content


def test_ps1_wrapper_has_python312_detection_order():
    content = (REPO_ROOT / "scripts" / "setup_and_run.ps1").read_text(encoding="utf-8")
    assert '"py"' in content
    assert '"python"' in content
    assert '"python3"' in content
    assert "3.12" in content


def test_ps1_wrapper_has_winget_fallback():
    content = (REPO_ROOT / "scripts" / "setup_and_run.ps1").read_text(encoding="utf-8")
    assert "winget" in content
    assert "Python.Python.3.12" in content
    assert "--accept-package-agreements" in content
    assert "--accept-source-agreements" in content
    assert "--scope" in content and "user" in content


def test_ps1_wrapper_creates_venv_when_missing():
    content = (REPO_ROOT / "scripts" / "setup_and_run.ps1").read_text(encoding="utf-8")
    assert "-m venv" in content


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell wrapper is Windows-only")
def test_ps1_wrapper_forwards_help_flag():
    ps1 = REPO_ROOT / "scripts" / "setup_and_run.ps1"
    proc = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-File", str(ps1), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "--setup-only" in proc.stdout
    assert "--provider" in proc.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="BAT wrapper is Windows-only")
def test_bat_wrapper_works_from_a_path_containing_spaces(tmp_path):
    """Copies just the BAT+PS1 pair into a spaced directory and confirms the
    relative hand-off (%~dp0 -> scripts\\setup_and_run.ps1) resolves; does not
    exercise the full bootstrap (no repo/.venv there), only path handling."""
    spaced_dir = tmp_path / "Space Dir"
    (spaced_dir / "scripts").mkdir(parents=True)
    bat_src = (REPO_ROOT / "SETUP_AND_RUN.bat").read_text(encoding="utf-8")
    (spaced_dir / "SETUP_AND_RUN.bat").write_text(bat_src, encoding="utf-8")
    ps1_stub = (
        '$RepoRoot = Split-Path -Parent $PSScriptRoot\n'
        'Write-Host "REACHED_PS1_STUB from $RepoRoot"\n'
        'exit 0\n'
    )
    (spaced_dir / "scripts" / "setup_and_run.ps1").write_text(ps1_stub, encoding="utf-8")

    proc = subprocess.run(
        [str(spaced_dir / "SETUP_AND_RUN.bat")],
        capture_output=True, text=True, timeout=60, shell=True,
    )
    assert "REACHED_PS1_STUB" in proc.stdout
    assert proc.returncode == 0
