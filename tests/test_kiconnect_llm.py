"""Offline contract tests for the KI Connect NRW provider adapter."""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from pipeline import llm
from pipeline.llm import LLMError, usage_from_exception
from pipeline.state import new_run


MODEL = "OpenAI GPT OSS 120b KI:Inferenz.nrw"


class _Point(BaseModel):
    x: int
    y: int


class _FakeResponse:
    def __init__(self, body: dict, *, error: Exception | None = None):
        self._body = body
        self._error = error

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error

    def json(self) -> dict:
        return self._body


class _FakeSession:
    def __init__(self, response: _FakeResponse):
        self.response = response
        self.calls: list[dict] = []

    def post(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.response


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setenv("KICONNECT_API_KEY", "test-secret")
    monkeypatch.setenv("KICONNECT_BASE_URL", "https://example.test/api/v1/")
    monkeypatch.setenv("KICONNECT_MAX_TOKENS", "4096")
    monkeypatch.setenv("KICONNECT_TIMEOUT_SECONDS", "17")


def _body(content: str, *, prompt_tokens: int = 10, completion_tokens: int = 4):
    return {
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def test_explicit_kiconnect_routing():
    assert llm.resolve_model_routing(f"kiconnect/{MODEL}") == (
        "kiconnect", MODEL,
    )


def test_role_override_can_select_kiconnect(monkeypatch):
    monkeypatch.setenv("REVIEWER_LLM_PROVIDER", "kiconnect")
    monkeypatch.setenv("REVIEWER_LLM_MODEL", MODEL)
    assert llm.role_model_override("reviewer", "flash-lite") == (
        f"kiconnect/{MODEL}"
    )


def test_kiconnect_plain_text_reply_and_usage(monkeypatch):
    session = _FakeSession(_FakeResponse(_body("hello", prompt_tokens=7, completion_tokens=2)))
    monkeypatch.setattr(llm, "_get_kiconnect_session", lambda: session)

    text, usage = llm.llm_call(
        new_run("test"), "prompt", system="system",
        model=f"kiconnect/{MODEL}",
    )

    assert text == "hello"
    assert usage.model == MODEL
    assert (usage.input_tokens, usage.output_tokens) == (7, 2)
    assert usage.cost_usd is None
    sent = session.calls[0]
    assert sent["url"] == "https://example.test/api/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer test-secret"
    assert sent["json"]["max_tokens"] == 4096
    assert sent["timeout"] == 17.0
    assert sent["json"]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "prompt"},
    ]
    assert "response_format" not in sent["json"]


def test_kiconnect_structured_output_uses_prompt_contract_and_validates(monkeypatch):
    session = _FakeSession(_FakeResponse(_body('{"x": 1, "y": 2}')))
    monkeypatch.setattr(llm, "_get_kiconnect_session", lambda: session)

    result, usage = llm._kiconnect_call(
        "prompt", system="", model_id=MODEL, response_schema=_Point,
    )

    assert result == _Point(x=1, y=2)
    assert usage.input_tokens == 10
    sent_prompt = session.calls[0]["json"]["messages"][-1]["content"]
    assert sent_prompt.startswith("prompt")
    assert "<json_schema>" in sent_prompt
    assert "Do not use Markdown fences" in sent_prompt
    assert '"required": [' in sent_prompt


@pytest.mark.parametrize(
    "content",
    [
        "```json\n{\"x\": 1, \"y\": 2}\n```",
        '{"x": 1}',
        "not json",
    ],
)
def test_kiconnect_never_repairs_invalid_structured_output(
    monkeypatch, content,
):
    session = _FakeSession(
        _FakeResponse(_body(content, prompt_tokens=12, completion_tokens=6))
    )
    monkeypatch.setattr(llm, "_get_kiconnect_session", lambda: session)

    with pytest.raises(LLMError) as exc_info:
        llm._kiconnect_call(
            "prompt", system="", model_id=MODEL, response_schema=_Point,
        )

    message = str(exc_info.value)
    assert MODEL in message
    assert "finish_reason='stop'" in message
    assert "input_tokens=12" in message
    assert "output_tokens=6" in message
    assert content not in message
    usage = usage_from_exception(exc_info.value)
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens) == (12, 6)


def test_kiconnect_transport_error_is_llmerror_without_usage(monkeypatch):
    session = _FakeSession(
        _FakeResponse({}, error=RuntimeError("service unavailable"))
    )
    monkeypatch.setattr(llm, "_get_kiconnect_session", lambda: session)

    with pytest.raises(LLMError, match="service unavailable") as exc_info:
        llm._kiconnect_call("prompt", system="", model_id=MODEL)

    assert usage_from_exception(exc_info.value) is None


def test_missing_kiconnect_key_fails_before_network(monkeypatch):
    monkeypatch.setenv("KICONNECT_API_KEY", "")

    with pytest.raises(LLMError, match="KICONNECT_API_KEY"):
        llm._kiconnect_call("prompt", system="", model_id=MODEL)
