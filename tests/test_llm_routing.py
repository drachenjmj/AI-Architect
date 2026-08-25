"""test_llm_routing.py — Google/KI Connect provider routing (Kusha).

Covers the provider-routing seam in `pipeline/llm.py` (see its module
docstring for the design):

  * the Gemini baseline is untouched when no `{ROLE}_LLM_*` env var is set;
  * the Architect and the Reviewer can each be redirected to KI Connect
    independently via `ARCHITECT_LLM_*` / `REVIEWER_LLM_*`;
  * the Clarifier and the advisor have no such seam at all, so they cannot be
    redirected by an Architect/Reviewer env change;
  * misconfiguration — an unknown provider, or a malformed `provider/model`
    string — fails loudly via `LLMError`, never a silent fallback to Gemini;
  * `anthropic` is no longer a supported provider.

No network calls anywhere in this file.
"""
from __future__ import annotations

import pytest

import test_reviewer as trev  # reuse _good_design_state (Kati's helper)
from pipeline import llm
from pipeline.agents import architect as arch
from pipeline.agents import clarifier as clar
from pipeline.agents import reviewer as rev
from pipeline.llm import (
    MODELS,
    LLMError,
    resolve_model_routing,
    role_model_override,
)


ROUTING_ENV_VARS = (
    "ARCHITECT_LLM_PROVIDER", "ARCHITECT_LLM_MODEL",
    "REVIEWER_LLM_PROVIDER", "REVIEWER_LLM_MODEL",
    # the Gemini thinking seam reads the same per-call env shape
    "ARCHITECT_GEMINI_THINKING_LEVEL", "REVIEWER_GEMINI_THINKING_LEVEL",
)


@pytest.fixture(autouse=True)
def _clean_routing_env(monkeypatch):
    """Every test starts from an unconfigured baseline, whatever a developer's
    real `.env` says — role routing is read from `os.environ` per call, so a
    leaked var would silently change which test runs against which provider.
    """
    for var in ROUTING_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# ── 1. today's Gemini defaults are byte-for-byte unchanged ───────────────
def test_default_routing_matches_todays_gemini_baseline():
    assert resolve_model_routing("flash-lite") == ("google", MODELS["flash-lite"])
    assert resolve_model_routing("flash") == ("google", MODELS["flash"])
    assert role_model_override("architect", arch.ARCHITECT_MODEL) == arch.ARCHITECT_MODEL
    assert role_model_override("reviewer", rev.REVIEWER_MODEL) == rev.REVIEWER_MODEL


# ── 2. Architect can be redirected to KI Connect, independently of the Reviewer ─
def test_architect_can_be_configured_for_kiconnect(monkeypatch):
    monkeypatch.setenv("ARCHITECT_LLM_PROVIDER", "kiconnect")
    monkeypatch.setenv("ARCHITECT_LLM_MODEL", "gpt-oss-120b")

    routed = role_model_override("architect", arch.ARCHITECT_MODEL)
    assert resolve_model_routing(routed) == ("kiconnect", "gpt-oss-120b")
    # An Architect-only env change must not leak into the Reviewer's routing.
    assert role_model_override("reviewer", rev.REVIEWER_MODEL) == rev.REVIEWER_MODEL


# ── 3. Reviewer can be redirected to KI Connect, independently of the Architect ─
def test_reviewer_can_be_configured_for_kiconnect(monkeypatch):
    monkeypatch.setenv("REVIEWER_LLM_PROVIDER", "kiconnect")
    monkeypatch.setenv("REVIEWER_LLM_MODEL", "gpt-oss-120b")

    routed = role_model_override("reviewer", rev.REVIEWER_MODEL)
    assert resolve_model_routing(routed) == ("kiconnect", "gpt-oss-120b")
    assert role_model_override("architect", arch.ARCHITECT_MODEL) == arch.ARCHITECT_MODEL


def test_run_reviewer_resolves_the_env_configured_model(monkeypatch):
    """The node-level seam, not just the pure routing function: `run_reviewer`
    with `model=None` must actually pick up `REVIEWER_LLM_*` at call time.
    """
    monkeypatch.setenv("REVIEWER_LLM_PROVIDER", "kiconnect")
    monkeypatch.setenv("REVIEWER_LLM_MODEL", "gpt-oss-120b")

    captured = {}

    def _capture(_state, _prompt, **kwargs):
        captured["model"] = kwargs["model"]
        return trev._all_pass_judgments(), trev.tc.fake_usage()

    monkeypatch.setattr(rev, "llm_call", _capture)
    rev.run_reviewer(trev._good_design_state())

    assert captured["model"] == "kiconnect/gpt-oss-120b"


# ── 4. Clarifier / advisor have no seam to redirect through ──────────────
def test_clarifier_and_advisor_are_unaffected_by_architect_reviewer_routing(monkeypatch):
    monkeypatch.setenv("ARCHITECT_LLM_PROVIDER", "kiconnect")
    monkeypatch.setenv("ARCHITECT_LLM_MODEL", "gpt-oss-120b")
    monkeypatch.setenv("REVIEWER_LLM_PROVIDER", "kiconnect")
    monkeypatch.setenv("REVIEWER_LLM_MODEL", "gpt-oss-120b")

    # No CLARIFIER_LLM_MODEL / ADVISOR_LLM_MODEL seam exists — the Clarifier
    # and the advisor call `llm_call` with a hard-coded model name, so an
    # Architect/Reviewer env change cannot reach them by construction.
    assert clar.CLARIFIER_MODEL == "flash-lite"
    assert clar.ADVISOR_MODEL == "flash-lite"
    assert resolve_model_routing(clar.CLARIFIER_MODEL) == ("google", MODELS["flash-lite"])
    assert resolve_model_routing(clar.ADVISOR_MODEL) == ("google", MODELS["flash-lite"])


# ── misconfiguration must fail loudly, never silently fall back ──────────
def test_unknown_provider_fails_loudly():
    with pytest.raises(LLMError):
        resolve_model_routing("bogus-provider/some-model")


def test_anthropic_is_no_longer_a_supported_provider():
    # Anthropic/Claude support was removed from the final submission — a
    # provider string that used to route to Claude must now fail loudly,
    # exactly like any other unknown provider, never silently fall back.
    assert "anthropic" not in llm.PROVIDERS
    with pytest.raises(LLMError):
        resolve_model_routing("anthropic/claude-opus-5")


def test_provider_without_a_model_is_ignored(monkeypatch):
    monkeypatch.setenv("ARCHITECT_LLM_PROVIDER", "kiconnect")
    # No ARCHITECT_LLM_MODEL — a provider alone names no model, so routing
    # must not guess one; it stays on the frozen Gemini default.
    assert role_model_override("architect", arch.ARCHITECT_MODEL) == arch.ARCHITECT_MODEL




# ═════════════════════════════════════════════════════════════════════════
# Role-specific Gemini thinking levels — the final evaluation runs the
# Gemini arm with an EXPLICIT ThinkingConfig.thinking_level (HIGH) on the
# Architect and Reviewer only. Everything below is offline: the Google
# client is a stub that captures the GenerateContentConfig it was handed.
# ═════════════════════════════════════════════════════════════════════════


class _FakeUsageMeta:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.prompt_token_count = input_tokens
        self.candidates_token_count = output_tokens


class _FakeGenerateContentResponse:
    """The three fields the Google path reads: text, usage_metadata, parsed."""

    def __init__(self, text: str, parsed, input_tokens: int, output_tokens: int):
        self.text = text
        self.parsed = parsed
        self.usage_metadata = _FakeUsageMeta(input_tokens, output_tokens)


class _FakeGoogleModels:
    def __init__(self, response, capture=None):
        self._response = response
        self.capture = capture if capture is not None else []

    def generate_content(self, *, model, contents, config):
        self.capture.append({"model": model, "contents": contents, "config": config})
        return self._response


class _FakeGoogleClient:
    def __init__(self, response, capture=None):
        self.models = _FakeGoogleModels(response, capture)


def _google_stub(monkeypatch, capture, text="ok", parsed=None, in_tok=3, out_tok=1):
    client = _FakeGoogleClient(
        _FakeGenerateContentResponse(text, parsed, in_tok, out_tok), capture,
    )
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    return client


_THINKING_ENV = (
    "ARCHITECT_GEMINI_THINKING_LEVEL", "REVIEWER_GEMINI_THINKING_LEVEL",
)


def _config_thinking_level(config) -> str | None:
    """The effective thinking_level on a captured GenerateContentConfig, as
    the API contract spells it (lowercase), None when not set."""
    tc = getattr(config, "thinking_config", None)
    if tc is None:
        return None
    level = getattr(tc, "thinking_level", None)
    if level is None:
        return None
    # the SDK stores a ThinkingLevel enum member whose value is UPPERCASE
    return getattr(level, "value", str(level)).lower()


def test_architect_env_sends_high_to_google_call(monkeypatch):
    from pipeline.state import new_run

    monkeypatch.setenv("ARCHITECT_GEMINI_THINKING_LEVEL", "high")
    capture = []
    _google_stub(monkeypatch, capture)

    _text, usage = llm.llm_call(
        new_run("p"), "hi",
        model=role_model_override("architect", arch.ARCHITECT_MODEL),
        thinking_level=llm.role_gemini_thinking_level("architect"),
    )

    assert capture[0]["model"] == MODELS["flash-lite"]  # still Google-routed
    assert _config_thinking_level(capture[0]["config"]) == "high"
    assert usage.thinking_level == "high"  # recorded, not just sent


def test_reviewer_env_sends_high_to_google_call(monkeypatch):
    from pipeline.state import new_run

    monkeypatch.setenv("REVIEWER_GEMINI_THINKING_LEVEL", "high")
    capture = []
    _google_stub(monkeypatch, capture)

    _text, usage = llm.llm_call(
        new_run("p"), "hi",
        model=role_model_override("reviewer", rev.REVIEWER_MODEL),
        thinking_level=llm.role_gemini_thinking_level("reviewer"),
    )

    assert _config_thinking_level(capture[0]["config"]) == "high"
    assert usage.thinking_level == "high"


def test_architect_and_reviewer_levels_are_independent(monkeypatch):
    monkeypatch.setenv("ARCHITECT_GEMINI_THINKING_LEVEL", "high")
    # reviewer var unset -> reviewer seam resolves None even with architect set

    assert llm.role_gemini_thinking_level("architect") == "high"
    assert llm.role_gemini_thinking_level("reviewer") is None

    monkeypatch.setenv("REVIEWER_GEMINI_THINKING_LEVEL", "low")
    monkeypatch.delenv("ARCHITECT_GEMINI_THINKING_LEVEL")
    assert llm.role_gemini_thinking_level("architect") is None
    assert llm.role_gemini_thinking_level("reviewer") == "low"


def test_clarifier_and_advisor_have_no_thinking_seam(monkeypatch):
    """No CLARIFIER_/ADVISOR_ env reader exists, so an evaluation-time
    ARCHITECT/REVIEWER setting cannot reach them — same no-seam guarantee
    as the provider routing above."""
    for var in _THINKING_ENV:
        monkeypatch.setenv(var, "high")

    assert llm.role_gemini_thinking_level("clarifier") is None
    assert llm.role_gemini_thinking_level("advisor") is None


def test_unset_env_preserves_existing_gemini_behavior(monkeypatch):
    """Both new vars absent -> NO thinking_config on the request at all, and
    the usage records None: byte-for-byte the pre-seam behavior."""
    from pipeline.state import new_run

    capture = []
    _google_stub(monkeypatch, capture)

    _text, usage = llm.llm_call(new_run("p"), "hi", model="flash-lite")

    assert capture[0]["config"].thinking_config is None
    assert usage.thinking_level is None


def test_all_four_api_levels_are_accepted(monkeypatch):
    from pipeline.state import new_run

    for level in llm.GEMINI_THINKING_LEVELS:
        capture = []
        _google_stub(monkeypatch, capture)

        _text, usage = llm.llm_call(
            new_run("p"), "hi", model="flash-lite", thinking_level=level,
        )
        assert _config_thinking_level(capture[0]["config"]) == level, level
        assert usage.thinking_level == level


def test_invalid_level_fails_loudly(monkeypatch):
    """The SDK itself only WARNS on an unknown level string — the explicit
    allowlist in llm_call is the real validation, and it must raise before
    any request is sent (an evaluation run must never silently run at
    default thinking while believing it ran HIGH)."""
    from pipeline.state import new_run

    capture = []
    _google_stub(monkeypatch, capture)

    with pytest.raises(LLMError, match="ultra"):
        llm.llm_call(new_run("p"), "hi", model="flash-lite", thinking_level="ultra")
    assert capture == []  # nothing was sent


def test_thinking_level_flows_into_step_log_metadata(monkeypatch):
    """Run-trace provenance: the effective level lands on the StepLog via
    make_step, so the final evaluation can prove HIGH per step without
    re-deriving it from `.env`."""
    from pipeline.agents.base import make_step
    from pipeline.llm import LLMUsage, sum_usage
    from pipeline.state import Stage

    usage = LLMUsage(
        model="gemini-3.1-flash-lite", input_tokens=1, output_tokens=1,
        cost_usd=0.0, thinking_level="high",
    )
    step = make_step("reviewer", Stage.DESIGNING, Stage.DONE, "n", usage)
    assert step.thinking_level == "high"

    # sum_usage keeps it across the Architect's two-call fold...
    total = sum_usage([usage, usage.model_copy()])
    assert total.thinking_level == "high"
    # ...mixes visibly rather than silently picking one...
    mixed = sum_usage([usage, LLMUsage(model="x", thinking_level="low")])
    assert mixed.thinking_level == "high, low"
    # ...and stays None when nothing was configured.
    assert sum_usage([LLMUsage(model="x"), LLMUsage(model="x")]).thinking_level is None

    # A step without usage stays "" (old checkpoints deserialize unchanged).
    plain = make_step("researcher", step.stage_in, step.stage_out, "n", None)
    assert plain.thinking_level == ""


def test_run_reviewer_passes_env_thinking_level_to_llm_call(monkeypatch):
    """The node-level seam: run_reviewer must forward the env-resolved level
    into llm_call for its Google call (the counterpart of the model-routing
    test above)."""
    monkeypatch.setenv("REVIEWER_GEMINI_THINKING_LEVEL", "high")

    captured = {}

    def _capture(_state, _prompt, **kwargs):
        captured["thinking_level"] = kwargs.get("thinking_level")
        return trev._all_pass_judgments(), trev.tc.fake_usage()

    monkeypatch.setattr(rev, "llm_call", _capture)
    rev.run_reviewer(trev._good_design_state())

    assert captured["thinking_level"] == "high"


def test_architect_node_passes_env_thinking_level_to_llm_calls(monkeypatch):
    """Both Architect phases forward the env-resolved level (initial pass:
    phase 1 features AND phase 2 design both carry it)."""
    from pipeline.state import Feature

    monkeypatch.setenv("ARCHITECT_GEMINI_THINKING_LEVEL", "high")

    captured = []

    def _fake_call(_state, _prompt, **kwargs):
        captured.append(kwargs.get("thinking_level"))
        if kwargs.get("response_schema") is arch.FeatureDesign:
            return arch.FeatureDesign(features=[Feature(
                id="FEAT-001", name="Survive seasonal peak load",
                scenario="Checkout remains responsive under seasonal peaks.",
                related_requirement_ids=[
                    "Customers complete purchases",
                    "Handle 50k concurrent users",
                ],
            )]), trev.tc.fake_usage()
        return arch.ArchitectureDesign(
            blueprint=arch.Blueprint(stakeholder_view="s", technical_view="t"),
            adrs=[], components=[],
        ), trev.tc.fake_usage()

    monkeypatch.setattr(arch, "llm_call", _fake_call)
    arch.architect_node(trev._good_design_state())

    assert captured == ["high", "high"]  # phase 1 AND phase 2 carried the level
