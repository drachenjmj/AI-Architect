"""test_llm_routing.py — Claude/Anthropic A/B provider routing (Kusha).

Covers the provider-routing seam added to `pipeline/llm.py` for the Claude
Opus 5 A/B experiment (see `pipeline/llm.py` module docstring for the design):

  * the Gemini baseline is untouched when no `{ROLE}_LLM_*` env var is set;
  * the Architect and the Reviewer can each be redirected to Claude
    independently via `ARCHITECT_LLM_*` / `REVIEWER_LLM_*`;
  * the Clarifier and the advisor have no such seam at all, so they cannot be
    redirected by an Architect/Reviewer env change;
  * the Anthropic call path (`_anthropic_call`) returns the exact same
    Pydantic/`str` contract as the Google path, propagates transport and
    schema-validation failures as `LLMError`, and reports provider/model/token
    metadata through the same `LLMUsage` shape;
  * the schema sent for structured output goes through Anthropic's OFFICIAL
    `transform_schema`, never a raw `response_schema.model_json_schema()`;
  * every Anthropic call goes through `messages.stream()` +
    `get_final_message()`, never the non-streaming `messages.create()`,
    which is required at the project's real CLAUDE_MAX_TOKENS default;
  * cost accounting for Claude uses the published per-1M-token rates.

No network calls anywhere in this file — every Anthropic client is a stub.
"""
from __future__ import annotations

import anthropic
import pytest
from pydantic import BaseModel

import test_reviewer as trev  # reuse _good_design_state (Kati's helper)
from pipeline import llm
from pipeline.agents import architect as arch
from pipeline.agents import clarifier as clar
from pipeline.agents import reviewer as rev
from pipeline.llm import (
    MODELS,
    LLMError,
    estimate_cost_usd,
    resolve_model_routing,
    role_model_override,
    usage_from_exception,
)


ROUTING_ENV_VARS = (
    "ARCHITECT_LLM_PROVIDER", "ARCHITECT_LLM_MODEL",
    "REVIEWER_LLM_PROVIDER", "REVIEWER_LLM_MODEL",
)


@pytest.fixture(autouse=True)
def _clean_routing_env(monkeypatch):
    """Every test starts from an unconfigured baseline, whatever a developer's
    real `.env` says — role routing is read from `os.environ` per call, so a
    leaked var would silently change which test runs against which provider.
    """
    for var in ROUTING_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# ── a minimal Anthropic Messages stub — no real SDK object, no network ────
class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeTextBlock:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _FakeMessage:
    def __init__(self, text: str, input_tokens: int, output_tokens: int):
        self.content = [_FakeTextBlock(text)]
        self.usage = _FakeUsage(input_tokens, output_tokens)


class _FakeMessageStream:
    """Stands in for the SDK's `MessageStream` — only `get_final_message()`
    is used by `_anthropic_call`, so that is the only method stubbed.
    """

    def __init__(self, message):
        self._message = message

    def get_final_message(self):
        return self._message


class _FakeStreamManager:
    """Stands in for `MessageStreamManager`. The real SDK only sends the HTTP
    request on `__enter__` (`.stream(...)` itself is lazy) — so a configured
    exception is raised there too, matching where a real transport failure
    would surface inside `with client.messages.stream(...) as stream:`.
    """

    def __init__(self, message=None, exception=None):
        self._message = message
        self._exception = exception

    def __enter__(self):
        if self._exception is not None:
            raise self._exception
        return _FakeMessageStream(self._message)

    def __exit__(self, exc_type, exc, exc_tb):
        return False


class _FakeMessages:
    def __init__(self, response=None, exception=None, capture=None):
        self._response = response
        self._exception = exception
        self.capture = capture if capture is not None else []

    def stream(self, **kwargs):
        self.capture.append(kwargs)
        return _FakeStreamManager(message=self._response, exception=self._exception)

    def create(self, **kwargs):
        # The adapter must go through messages.stream()/get_final_message()
        # for every Anthropic call, never the non-streaming create() path —
        # that path client-side-fails past the SDK's 10-minute non-streaming
        # timeout at the project's real CLAUDE_MAX_TOKENS default. A silent
        # regression back to create() must fail loudly, here, offline.
        raise AssertionError(
            "messages.create() must not be called — the Anthropic adapter "
            "must use messages.stream() + get_final_message()"
        )


class _FakeAnthropicClient:
    def __init__(self, **kw):
        self.messages = _FakeMessages(**kw)


class _Point(BaseModel):
    x: int
    y: int


# ── 1. today's Gemini defaults are byte-for-byte unchanged ───────────────
def test_default_routing_matches_todays_gemini_baseline():
    assert resolve_model_routing("flash-lite") == ("google", MODELS["flash-lite"])
    assert resolve_model_routing("flash") == ("google", MODELS["flash"])
    assert role_model_override("architect", arch.ARCHITECT_MODEL) == arch.ARCHITECT_MODEL
    assert role_model_override("reviewer", rev.REVIEWER_MODEL) == rev.REVIEWER_MODEL


# ── 2. Architect can be redirected to Claude, independently of the Reviewer ─
def test_architect_can_be_configured_for_claude(monkeypatch):
    monkeypatch.setenv("ARCHITECT_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ARCHITECT_LLM_MODEL", "claude-opus-5")

    routed = role_model_override("architect", arch.ARCHITECT_MODEL)
    assert resolve_model_routing(routed) == ("anthropic", "claude-opus-5")
    # An Architect-only env change must not leak into the Reviewer's routing.
    assert role_model_override("reviewer", rev.REVIEWER_MODEL) == rev.REVIEWER_MODEL


# ── 3. Reviewer can be redirected to Claude, independently of the Architect ─
def test_reviewer_can_be_configured_for_claude(monkeypatch):
    monkeypatch.setenv("REVIEWER_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("REVIEWER_LLM_MODEL", "claude-opus-5")

    routed = role_model_override("reviewer", rev.REVIEWER_MODEL)
    assert resolve_model_routing(routed) == ("anthropic", "claude-opus-5")
    assert role_model_override("architect", arch.ARCHITECT_MODEL) == arch.ARCHITECT_MODEL


def test_run_reviewer_resolves_the_env_configured_model(monkeypatch):
    """The node-level seam, not just the pure routing function: `run_reviewer`
    with `model=None` must actually pick up `REVIEWER_LLM_*` at call time.
    """
    monkeypatch.setenv("REVIEWER_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("REVIEWER_LLM_MODEL", "claude-opus-5")

    captured = {}

    def _capture(_state, _prompt, **kwargs):
        captured["model"] = kwargs["model"]
        return trev._all_pass_judgments(), trev.tc.fake_usage()

    monkeypatch.setattr(rev, "llm_call", _capture)
    rev.run_reviewer(trev._good_design_state())

    assert captured["model"] == "anthropic/claude-opus-5"


# ── 4. Clarifier / advisor have no seam to redirect through ──────────────
def test_clarifier_and_advisor_are_unaffected_by_architect_reviewer_routing(monkeypatch):
    monkeypatch.setenv("ARCHITECT_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ARCHITECT_LLM_MODEL", "claude-opus-5")
    monkeypatch.setenv("REVIEWER_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("REVIEWER_LLM_MODEL", "claude-opus-5")

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
        resolve_model_routing("bogus-provider/claude-opus-5")


def test_provider_without_a_model_is_ignored(monkeypatch):
    monkeypatch.setenv("ARCHITECT_LLM_PROVIDER", "anthropic")
    # No ARCHITECT_LLM_MODEL — a provider alone names no model, so routing
    # must not guess one; it stays on the frozen Gemini default.
    assert role_model_override("architect", arch.ARCHITECT_MODEL) == arch.ARCHITECT_MODEL


def test_claude_effort_defaults_to_high():
    # CLAUDE_EFFORT is read once at import time — this pins the documented
    # default (see .env.example) for whoever hasn't overridden it locally.
    assert llm.CLAUDE_EFFORT == "high"


# ── 5 & 7. Claude structured output -> exact Pydantic type + usage metadata ─
def test_claude_structured_output_becomes_the_requested_pydantic_type(monkeypatch):
    capture = []
    fake_client = _FakeAnthropicClient(
        response=_FakeMessage('{"x": 1, "y": 2}', input_tokens=10, output_tokens=4),
        capture=capture,
    )
    monkeypatch.setattr(llm, "_get_anthropic_client", lambda: fake_client)

    result, usage = llm._anthropic_call(
        "prompt", system="sys", model_id="claude-opus-5", response_schema=_Point,
    )

    assert isinstance(result, _Point)
    assert result == _Point(x=1, y=2)
    assert usage.model == "claude-opus-5"
    assert usage.input_tokens == 10
    assert usage.output_tokens == 4
    assert usage.cost_usd == pytest.approx(estimate_cost_usd("claude-opus-5", 10, 4))

    sent = capture[0]
    assert sent["model"] == "claude-opus-5"
    assert sent["max_tokens"] == llm.CLAUDE_MAX_TOKENS
    assert sent["messages"] == [{"role": "user", "content": "prompt"}]
    assert sent["system"] == "sys"
    assert sent["output_config"]["effort"] == llm.CLAUDE_EFFORT

    # THE regression this pins: the schema on the wire must be Anthropic's
    # OFFICIAL `transform_schema` output, never raw `model_json_schema()`.
    # `additionalProperties: false` is the concrete, API-required shape that
    # `transform_schema` adds and a raw Pydantic schema never has — so this
    # also proves the transform actually ran, not just that some dict was sent.
    raw_schema = _Point.model_json_schema()
    assert "additionalProperties" not in raw_schema
    sent_schema = sent["output_config"]["format"]["schema"]
    assert sent_schema != raw_schema
    assert sent_schema["additionalProperties"] is False
    assert sent["output_config"]["format"] == {
        "type": "json_schema",
        "schema": anthropic.transform_schema(_Point),
    }


def test_claude_plain_text_reply_and_usage_metadata(monkeypatch):
    fake_client = _FakeAnthropicClient(
        response=_FakeMessage("hello", input_tokens=5, output_tokens=2)
    )
    monkeypatch.setattr(llm, "_get_anthropic_client", lambda: fake_client)

    text, usage = llm._anthropic_call("prompt", system="", model_id="claude-opus-5")

    assert text == "hello"
    assert usage.model == "claude-opus-5"
    assert usage.input_tokens == 5
    assert usage.output_tokens == 2


def test_llm_call_dispatches_explicit_anthropic_routing(monkeypatch):
    """The public `llm_call` entry point, not just the private helper."""
    from pipeline.state import new_run

    fake_client = _FakeAnthropicClient(
        response=_FakeMessage("ok", input_tokens=3, output_tokens=1)
    )
    monkeypatch.setattr(llm, "_get_anthropic_client", lambda: fake_client)

    text, usage = llm.llm_call(
        new_run("test prompt"), "hi", model="anthropic/claude-opus-5",
    )

    assert text == "ok"
    assert usage.model == "claude-opus-5"


# ── 6. transport and schema-validation failures surface as LLMError ──────
def test_anthropic_transport_error_becomes_llmerror(monkeypatch):
    fake_client = _FakeAnthropicClient(exception=RuntimeError("connection reset"))
    monkeypatch.setattr(llm, "_get_anthropic_client", lambda: fake_client)

    with pytest.raises(LLMError, match="connection reset"):
        llm._anthropic_call("prompt", system="", model_id="claude-opus-5")


def test_claude_schema_invalid_output_raises_with_billed_usage_attached(monkeypatch):
    fake_client = _FakeAnthropicClient(
        response=_FakeMessage("not valid json", input_tokens=7, output_tokens=3)
    )
    monkeypatch.setattr(llm, "_get_anthropic_client", lambda: fake_client)

    with pytest.raises(LLMError) as exc_info:
        llm._anthropic_call(
            "prompt", system="", model_id="claude-opus-5", response_schema=_Point,
        )

    usage = usage_from_exception(exc_info.value)
    assert usage is not None
    assert usage.input_tokens == 7
    assert usage.output_tokens == 3


# ── transport: always streaming, never the non-streaming create() path ───
def test_anthropic_call_uses_streaming_transport(monkeypatch):
    """`_anthropic_call` must go through `messages.stream()` +
    `get_final_message()` — `_FakeMessages.create()` raises AssertionError
    unconditionally (see its definition above), so this only passes if the
    adapter never reaches for the non-streaming path.
    """
    capture = []
    fake_client = _FakeAnthropicClient(
        response=_FakeMessage("ok", input_tokens=1, output_tokens=1),
        capture=capture,
    )
    monkeypatch.setattr(llm, "_get_anthropic_client", lambda: fake_client)

    text, usage = llm._anthropic_call("prompt", system="", model_id="claude-opus-5")

    assert text == "ok"
    assert usage.input_tokens == 1
    assert len(capture) == 1  # exactly one messages.stream() call, no create()


def test_claude_max_tokens_default_goes_through_streaming_transport(monkeypatch):
    """Regression for the exact failure this task fixes: at the project's
    real CLAUDE_MAX_TOKENS=32000 default, the non-streaming create() path
    fails client-side with 'Streaming is required for operations that may
    take longer than 10 minutes' before any request is sent. This proves
    max_tokens=32000 is sent through messages.stream(), not create().
    """
    assert llm.CLAUDE_MAX_TOKENS == 32000  # the real, unchanged project default

    capture = []
    fake_client = _FakeAnthropicClient(
        response=_FakeMessage("ok", input_tokens=1, output_tokens=1),
        capture=capture,
    )
    monkeypatch.setattr(llm, "_get_anthropic_client", lambda: fake_client)

    llm._anthropic_call("prompt", system="", model_id="claude-opus-5")

    assert capture[0]["max_tokens"] == 32000


# ── 8. cost accounting for Claude, a deterministic synthetic example ─────
def test_claude_cost_accounting_matches_published_pricing():
    # $5.00 / 1M input, $25.00 / 1M output — the published rate this
    # experiment uses (see PRICING_USD_PER_MTOK in pipeline/llm.py).
    assert estimate_cost_usd("claude-opus-5", 1_000_000, 1_000_000) == pytest.approx(30.0)
    assert estimate_cost_usd("claude-opus-5", 200_000, 40_000) == pytest.approx(2.0)
    assert estimate_cost_usd("claude-opus-5", 0, 0) == 0


# ═════════════════════════════════════════════════════════════════════════
# Oversized-schema fallback — `ArchitectureDesign` uses prompted JSON on
# Anthropic because a REAL full Architect run failed before generation with
# "The compiled grammar is too large" (400). The policy is an explicit list,
# not a size heuristic; everything else keeps native structured output.
# ═════════════════════════════════════════════════════════════════════════


def _minimal_design_json() -> str:
    """The smallest schema-valid `ArchitectureDesign` document."""
    return (
        '{"blueprint": {"stakeholder_view": "s", "technical_view": "t"},'
        ' "adrs": [], "components": []}'
    )


def test_architecture_design_is_on_the_prompted_json_list():
    assert llm._anthropic_uses_prompted_json(arch.ArchitectureDesign) is True
    # ...and nothing else is: the same NAME in another module never
    # triggers it (the module pin is part of the identity).
    class ArchitectureDesign(BaseModel):
        x: int

    assert llm._anthropic_uses_prompted_json(ArchitectureDesign) is False
    assert llm._anthropic_uses_prompted_json(_Point) is False
    assert llm._anthropic_uses_prompted_json(None) is False
    assert ("pipeline.agents.architect", "ArchitectureDesign") in (
        llm._PROMPTED_JSON_SCHEMAS
    )


def test_architecture_design_sends_no_native_format_but_full_prompt_contract(
    monkeypatch,
):
    capture = []
    fake_client = _FakeAnthropicClient(
        response=_FakeMessage(_minimal_design_json(), 100, 2000),
        capture=capture,
    )
    monkeypatch.setattr(llm, "_get_anthropic_client", lambda: fake_client)

    llm._anthropic_call(
        "design prompt", system="sys", model_id="claude-opus-5",
        response_schema=arch.ArchitectureDesign,
    )

    sent = capture[0]
    # No native grammar for the demonstrated-oversized schema...
    assert "format" not in sent["output_config"]
    assert sent["output_config"] == {"effort": llm.CLAUDE_EFFORT}  # effort kept
    assert sent["max_tokens"] == llm.CLAUDE_MAX_TOKENS             # budget kept
    assert sent["system"] == "sys"                                  # untouched
    # ...and the prompt carries the COMPLETE contract exactly once.
    sent_prompt = sent["messages"][0]["content"]
    assert sent_prompt.startswith("design prompt")
    assert sent_prompt.count("<json_schema>") == 1
    assert sent_prompt.count("</json_schema>") == 1
    import json

    schema_text = json.dumps(
        arch.ArchitectureDesign.model_json_schema(), indent=2,
        ensure_ascii=False,
    )
    assert sent_prompt.count(schema_text) == 1  # full schema, exactly once
    # JSON-only requirements are explicit.
    for phrase in ("exactly one JSON document", "Do not use Markdown fences",
                   "Do not include any commentary",
                   "Do not omit required fields",
                   "Do not invent fields"):
        assert phrase in sent_prompt, phrase


def test_architecture_design_prompted_json_returns_real_pydantic_instance(
    monkeypatch,
):
    fake_client = _FakeAnthropicClient(
        response=_FakeMessage(_minimal_design_json(), 100, 2000)
    )
    monkeypatch.setattr(llm, "_get_anthropic_client", lambda: fake_client)

    result, usage = llm._anthropic_call(
        "p", system="", model_id="claude-opus-5",
        response_schema=arch.ArchitectureDesign,
    )

    assert isinstance(result, arch.ArchitectureDesign)
    assert result.blueprint.stakeholder_view == "s"
    # Token/cost accounting identical to the native path.
    assert usage.model == "claude-opus-5"
    assert usage.input_tokens == 100
    assert usage.output_tokens == 2000
    assert usage.cost_usd == pytest.approx(
        estimate_cost_usd("claude-opus-5", 100, 2000)
    )


def test_architecture_design_invalid_json_fails_loudly_with_billed_usage(
    monkeypatch,
):
    fake_client = _FakeAnthropicClient(
        response=_FakeMessage("```json\n{oops}\n```", 80, 1500)
    )
    monkeypatch.setattr(llm, "_get_anthropic_client", lambda: fake_client)

    with pytest.raises(LLMError) as exc_info:
        llm._anthropic_call(
            "p", system="", model_id="claude-opus-5",
            response_schema=arch.ArchitectureDesign,
        )

    usage = usage_from_exception(exc_info.value)
    assert usage is not None
    assert usage.input_tokens == 80
    assert usage.output_tokens == 1500


def test_architecture_design_uses_streaming_transport_too(monkeypatch):
    """`_FakeMessages.create()` raises unconditionally, so reaching the
    reply at all proves messages.stream() + get_final_message() — the
    prompted-JSON path shares the same transport as the native path."""
    capture = []
    fake_client = _FakeAnthropicClient(
        response=_FakeMessage(_minimal_design_json(), 1, 1), capture=capture,
    )
    monkeypatch.setattr(llm, "_get_anthropic_client", lambda: fake_client)

    llm._anthropic_call(
        "p", system="", model_id="claude-opus-5",
        response_schema=arch.ArchitectureDesign,
    )

    assert len(capture) == 1  # exactly one stream() call, zero create()


def test_smaller_schemas_keep_native_structured_output(monkeypatch):
    """The fallback is ArchitectureDesign-ONLY: a small schema (the tiny
    smoke shape) still sends the native grammar via transform_schema."""
    capture = []
    fake_client = _FakeAnthropicClient(
        response=_FakeMessage('{"x": 1, "y": 2}', 10, 4), capture=capture,
    )
    monkeypatch.setattr(llm, "_get_anthropic_client", lambda: fake_client)

    llm._anthropic_call(
        "p", system="", model_id="claude-opus-5", response_schema=_Point,
    )

    sent = capture[0]
    assert sent["output_config"]["format"] == {
        "type": "json_schema",
        "schema": anthropic.transform_schema(_Point),
    }
    # And no prompted contract leaked into the small-schema prompt.
    assert "<json_schema>" not in sent["messages"][0]["content"]


def test_reviewer_schema_keeps_native_structured_output(monkeypatch):
    """The Reviewer's `LLMJudgments` is NOT on the oversized list — it keeps
    native structured output, so Reviewer behavior is unchanged."""
    capture = []
    fake_client = _FakeAnthropicClient(
        response=_FakeMessage(
            '{"repo_grounding": {"passed": true, "reason": "ok"}}', 5, 5,
        ),
        capture=capture,
    )
    monkeypatch.setattr(llm, "_get_anthropic_client", lambda: fake_client)

    llm._anthropic_call(
        "p", system="", model_id="claude-opus-5",
        response_schema=rev.LLMJudgments,
    )

    assert "format" in capture[0]["output_config"]
