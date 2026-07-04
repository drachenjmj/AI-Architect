"""llm.py — the single door every agent uses to talk to an LLM.

Why this module exists (see pipeline-skeleton notes):
  * ONE place to load the API key, build the client, and read token usage —
    so agents don't each reinvent it and standards can't drift.
  * Agents request a model BY NAME ("flash-lite" / "flash"), not by ID —
    this is the token-efficiency lever (cheapest model for the reviewer,
    stronger for the architect).
  * Every call funnels token counts into state, matching the "token
    efficiency as a design input" principle.

Public surface: agents import ONLY `llm_call`. The factory `_get_client`
is private — kept underneath in case a special case (e.g. Google-Search
grounding) needs the raw client.

SDK: uses the current `google.genai` package (the old `google.generativeai`
is end-of-life). Here the system prompt is passed PER CALL via the request
config, so — unlike the old SDK — we cache a single client for the whole
process instead of one model object per (model, system_prompt).
"""
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from pipeline.state import ArchitectState


# ── Model registry: friendly NAME -> real model ID ───────────────────────
# This is the cheap/strong lever. Agents say model="flash"; only this line
# changes if we swap the underlying model. To add Claude/GPT later, this
# maps to (provider, model_id) and llm_call dispatches on provider.
MODELS: dict[str, str] = {
    "flash-lite": "gemini-2.5-flash-lite",  # cheapest / fastest — default
    "flash":      "gemini-2.5-flash",       # stronger — for harder tasks
}
DEFAULT_MODEL = "flash-lite"
# NB: gemini-2.5-pro is deliberately NOT here — it has a zero free-tier quota
# and this is a university project with no billing. Both models above are
# free-tier accessible. Add "pro" back only if a billing-enabled key appears.


class LLMError(RuntimeError):
    """Raised when an LLM call fails. base.run() catches it and marks FAILED."""


# ── Client: one per process, key loaded once ─────────────────────────────
# The new SDK holds the API key on the Client, not on each model. The model
# ID and system prompt are chosen per request, so a single cached client
# serves every agent and every model name.
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise LLMError("GEMINI_API_KEY not found in .env!")
        _client = genai.Client(api_key=api_key)
    return _client


# ── The one public function ──────────────────────────────────────────────
def llm_call(
    state: ArchitectState,
    prompt: str,
    *,
    system: str = "",
    model: str = DEFAULT_MODEL,
    response_schema: type | None = None,
) -> Any:
    """Send one prompt to the named model; return the reply.

    Side effect: adds this call's token usage into `state` (input/output).
    Raises LLMError on failure — the agent's node wrapper turns that into a
    clean FAILED step instead of crashing the pipeline.

    `system` is passed PER CALL so each agent's prompt lives with the agent,
    not scattered in a registry.

    `response_schema` (optional): a Pydantic model class. When given, the model
    is asked for JSON constrained to that schema and this returns a VALIDATED
    instance of the class (via the SDK's `resp.parsed`) — no manual parsing.
    When omitted, behaviour is unchanged and a plain `str` is returned, so every
    existing caller keeps working.
    """
    if model not in MODELS:
        raise LLMError(f"Unknown model '{model}'. Known: {list(MODELS)}")
    client = _get_client()
    config = types.GenerateContentConfig(
        system_instruction=system or None,
    )
    if response_schema is not None:
        config.response_mime_type = "application/json"
        config.response_schema = response_schema
    try:
        resp = client.models.generate_content(
            model=MODELS[model],
            contents=prompt,
            config=config,
        )
    except Exception as e:
        raise LLMError(f"{model} call failed: {e}") from e

    # Record tokens at the single chokepoint (usage_metadata may be absent).
    usage = getattr(resp, "usage_metadata", None)
    if usage is not None:
        state.input_tokens += getattr(usage, "prompt_token_count", 0) or 0
        state.output_tokens += getattr(usage, "candidates_token_count", 0) or 0

    if response_schema is not None:
        parsed = getattr(resp, "parsed", None)
        if parsed is None:  # model returned something unparseable to the schema
            raise LLMError(f"{model} returned no schema-valid JSON for {response_schema.__name__}")
        return parsed
    return resp.text


# ── Self-test: `python -m pipeline.llm` ──────────────────────────────────
# Makes ONE real API call to prove the key + registry + token wiring work.
# Not a unit test (it costs a token or two and needs the network); it's a
# smoke check you run by hand after setting up your .env.
if __name__ == "__main__":
    from pipeline.state import new_run

    s = new_run(raw_prompt="(llm smoke test)")
    print(f"Calling model '{DEFAULT_MODEL}' …")
    reply = llm_call(
        s,
        prompt="Reply with exactly the word: OK",
        system="You are a terse test probe.",
        model=DEFAULT_MODEL,
    )
    print(f"  reply         : {reply!r}")
    print(f"  input_tokens  : {s.input_tokens}")
    print(f"  output_tokens : {s.output_tokens}")
    assert s.input_tokens > 0, "no input tokens recorded — token wiring broken"
    print("✓ API key works and token usage was recorded into state.")
