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
  * a schema-invalid Anthropic response reports the SDK's own final
  `stop_reason`, THIS call's input/output tokens, the response length in
  characters and the actual Pydantic validation reason — without dumping
  the generated content — while billed usage stays attached;
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
    """Carries the fields `_anthropic_call` reads off a final Message:
    content blocks, usage, and `stop_reason` (the SDK's own value —
    'end_turn' by default, 'max_tokens' for the truncation tests below).
    """

    def __init__(
        self,
        text: str,
        input_tokens: int,
        output_tokens: int,
        stop_reason: str | None = "end_turn",
    ):
        self.content = [_FakeTextBlock(text)]
        self.usage = _FakeUsage(input_tokens, output_tokens)
        self.stop_reason = stop_reason


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


def test_claude_effort_defaults_to_medium():
    # CLAUDE_EFFORT is read once at import time — this pins the documented
    # default (see .env.example) for whoever hasn't overridden it locally.
    assert llm.CLAUDE_EFFORT == "medium"


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
    # The reproducible experiment default: medium effort on the wire.
    assert sent["output_config"]["effort"] == "medium"

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

    with pytest.raises(LLMError, match="connection reset") as exc_info:
        llm._anthropic_call("prompt", system="", model_id="claude-opus-5")

    # No final Message was obtained, so NOTHING is invented: no stop reason
    # in the message, no billed usage attached (nothing was received).
    assert "stop_reason" not in str(exc_info.value)
    assert usage_from_exception(exc_info.value) is None


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


# ── completion diagnostics: the failed-run post-mortem, offline ───────────
# The real Opus E2E died with a bare "returned no schema-valid JSON" and
# node-level token totals that could not separate truncation from a schema
# mismatch. These pin the per-call facts a failed run must now report:
# the SDK's own stop reason, THIS call's billed tokens, the response length
# in characters, and the actual Pydantic validation reason.
def test_schema_invalid_normal_completion_reports_stop_reason_and_validation(
    monkeypatch,
):
    fake_client = _FakeAnthropicClient(
        response=_FakeMessage(
            '{"x": 1}', input_tokens=42, output_tokens=17, stop_reason="end_turn",
        )
    )
    monkeypatch.setattr(llm, "_get_anthropic_client", lambda: fake_client)

    with pytest.raises(LLMError) as exc_info:
        llm._anthropic_call(
            "prompt", system="", model_id="claude-opus-5", response_schema=_Point,
        )

    msg = str(exc_info.value)
    assert "claude-opus-5" in msg            # provider/model
    assert "_Point" in msg                   # response schema name
    assert "end_turn" in msg                 # normal completion: NOT truncation
    assert "input_tokens=42" in msg          # per-CALL, not node totals
    assert "output_tokens=17" in msg
    assert "response_chars=8" in msg         # len('{"x": 1}')
    # The model's own validation reason (y missing), not a guessed diagnosis.
    assert "y" in msg
    assert "Field required" in msg
    # Billed usage for THIS call remains attached.
    usage = usage_from_exception(exc_info.value)
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens) == (42, 17)


def test_schema_invalid_max_tokens_stop_reason_is_explicit(monkeypatch):
    """The truncation case: a max_tokens stop reason must be impossible to
    miss in the error, with the per-call budget consumption attached."""
    fake_client = _FakeAnthropicClient(
        response=_FakeMessage(
            '{"x": 1', input_tokens=42, output_tokens=32000,
            stop_reason="max_tokens",
        )
    )
    monkeypatch.setattr(llm, "_get_anthropic_client", lambda: fake_client)

    with pytest.raises(LLMError) as exc_info:
        llm._anthropic_call(
            "prompt", system="", model_id="claude-opus-5", response_schema=_Point,
        )

    msg = str(exc_info.value)
    assert "max_tokens" in msg
    assert "output_tokens=32000" in msg
    usage = usage_from_exception(exc_info.value)
    assert usage is not None
    assert usage.output_tokens == 32000


def test_schema_invalid_error_never_dumps_the_raw_response(monkeypatch):
    """Diagnostics quote locations/messages/types only — the model's
    generated content (here a 2000-char value Pydantic rejects) must not
    leak into the error, even though Pydantic's error dicts carry it."""
    long_bad_value = "A" * 2000
    fake_client = _FakeAnthropicClient(
        response=_FakeMessage(
            f'{{"x": "{long_bad_value}", "y": 2}}',
            input_tokens=1, output_tokens=2, stop_reason="end_turn",
        )
    )
    monkeypatch.setattr(llm, "_get_anthropic_client", lambda: fake_client)

    with pytest.raises(LLMError) as exc_info:
        llm._anthropic_call(
            "prompt", system="", model_id="claude-opus-5", response_schema=_Point,
        )

    msg = str(exc_info.value)
    assert "AAAA" not in msg
    assert len(msg) < 1000
    # ...but the actual validation reason is still visible.
    assert "valid integer" in msg


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
    assert sent["output_config"] == {"effort": "medium"}  # effort kept (medium)
    assert sent["max_tokens"] == llm.CLAUDE_MAX_TOKENS         # budget kept
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


def test_claude_calls_reject_gemini_thinking_level(monkeypatch):
    """A Gemini thinking level on an Anthropic-routed call is a
    misconfiguration (Claude effort is CLAUDE_EFFORT) — fail loudly, and
    never put thinking config into an Anthropic request."""
    from pipeline.state import new_run

    fake_client = _FakeAnthropicClient(
        response=_FakeMessage("ok", input_tokens=3, output_tokens=1)
    )
    monkeypatch.setattr(llm, "_get_anthropic_client", lambda: fake_client)

    with pytest.raises(LLMError, match="CLAUDE_EFFORT"):
        llm.llm_call(
            new_run("p"), "hi", model="anthropic/claude-opus-5",
            thinking_level="high",
        )
    # and no Anthropic request left the building
    assert fake_client.messages.capture == []


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
