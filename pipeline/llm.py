"""llm.py — the single door every agent uses to talk to an LLM.

Why this module exists (see pipeline-skeleton notes):
  * ONE place to load the API key, build the client, and read token usage —
    so agents don't each reinvent it and standards can't drift.
  * Agents request a model BY NAME ("flash-lite" / "flash"), not by ID —
    this is the token-efficiency lever (cheapest model for the reviewer,
    stronger for the architect).
  * Every call REPORTS its token count back to the caller, matching the
    "token efficiency as a design input" principle.

TOKEN ACCOUNTING (why `llm_call` returns a tuple)
-------------------------------------------------
This module used to add tokens straight into `state.input_tokens` /
`state.output_tokens`. Under LangGraph that was silently dead: the graph
persists ONLY what a node RETURNS (see agents/base.py), so every count was
thrown away at the end of the node and the run-cost guardrail in
refine_gate.py compared 0 against its budget forever.

So `llm_call` is now PURE with respect to `state` and returns
`(reply, LLMUsage)`. The calling node sums the usage of its own calls and puts
the totals in its returned dict, where the `operator.add` reducers on
`ArchitectState` accumulate them across the whole run.

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
from typing import Any, NamedTuple, Sequence

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from pipeline.state import ArchitectState


# ── Model registry: friendly NAME -> real model ID ───────────────────────
# This is the cheap/strong lever. Agents say model="flash"; only this line
# changes if we swap the underlying model. To add Claude/GPT later, this
# maps to (provider, model_id) and llm_call dispatches on provider.
MODELS: dict[str, str] = {
    "flash-lite": "gemini-3.1-flash-lite",  # cheapest / fastest — default
    "flash":      "gemini-3.5-flash",       # stronger — for harder tasks
}
DEFAULT_MODEL = "flash-lite"
# NB: gemini-2.5-pro is deliberately NOT here — it has a zero free-tier quota
# and this is a university project with no billing. Both models above are
# free-tier accessible. Add "pro" back only if a billing-enabled key appears.

# ── Provider routing (the controlled A/B seam) ────────────────────────────
# `llm_call`'s `model=` argument accepts three forms, resolved by
# `resolve_model_routing`:
#   * a friendly NAME from MODELS          -> the Google path, exactly as
#                                              before the seam existed;
#   * "provider/model-id" (e.g.            -> that provider explicitly;
#     "anthropic/claude-opus-5")
#   * a bare real model ID whose family     -> that provider (a claude-*
#     prefix is known ("claude-opus-5")        id is Anthropic, a gemini-*
#                                              id is Google).
# Roles pick their routing per call via `role_model_override`, so the
# Architect and the Reviewer can be pointed at Claude by environment while
# the Clarifier, the advisor, RAG and the web fallback stay on Gemini.
PROVIDERS: tuple[str, ...] = ("google", "anthropic")
_PROVIDER_BY_PREFIX: dict[str, str] = {
    "gemini": "google",
    "claude": "anthropic",
}


def resolve_model_routing(model: str) -> tuple[str, str]:
    """`(provider, model_id)` for one `model=` argument. Pure.

    Raises LLMError for an unknown friendly name AND an unknown explicit
    provider — misconfiguration must fail loudly, never fall back to a
    different model than the caller asked for.
    """
    if "/" in model:
        provider, _, model_id = model.partition("/")
        provider = provider.strip().lower()
        model_id = model_id.strip()
        if provider not in PROVIDERS or not model_id:
            raise LLMError(
                f"Unknown LLM provider or malformed routing '{model}'. "
                f"Known providers: {', '.join(PROVIDERS)} "
                f"(form: 'provider/model-id')."
            )
        return provider, model_id
    if model in MODELS:
        return "google", MODELS[model]
    prefix = model.split("-", 1)[0].strip().lower()
    if prefix in _PROVIDER_BY_PREFIX:
        return _PROVIDER_BY_PREFIX[prefix], model
    raise LLMError(
        f"Unknown model '{model}'. Known friendly names: {list(MODELS)}; "
        "or pass a real model id / 'provider/model-id'."
    )


def role_model_override(role: str, default: str) -> str:
    """The `model=` routing string for one agent role, this call. Pure.

    Reads `{ROLE}_LLM_MODEL` (and optional `{ROLE}_LLM_PROVIDER`) from the
    environment on EVERY call, so a run can be redirected without code
    changes and tests can redirect without reloading modules:

        ARCHITECT_LLM_PROVIDER=anthropic
        ARCHITECT_LLM_MODEL=claude-opus-5     -> "anthropic/claude-opus-5"
        ARCHITECT_LLM_MODEL=claude-opus-5     -> "claude-opus-5" (prefix-inferred)
        neither set                           -> `default` (today's Gemini)

    A provider set without a model is ignored: a provider alone names no
    model, and guessing one would break the "same model every run" premise
    of the A/B experiment.
    """
    model_id = os.environ.get(f"{role.upper()}_LLM_MODEL", "").strip()
    if not model_id:
        return default
    provider = os.environ.get(f"{role.upper()}_LLM_PROVIDER", "").strip().lower()
    return f"{provider}/{model_id}" if provider else model_id


class LLMError(RuntimeError):
    """Raised when an LLM call fails. base.run() catches it and marks FAILED."""


# ── Cost model ───────────────────────────────────────────────────────────
# HONESTY NOTE, because these numbers go into a graded business case:
# this project runs on FREE-TIER API keys. Nothing computed here is money
# actually spent. It is the LIST-PRICE EQUIVALENT of the tokens consumed —
# what the run WOULD have cost on the paid tier. Report it with that wording,
# never as spend.
#
# Prices are Google's official Gemini API list prices, PAID tier, USD per 1M
# tokens, looked up 2026-08-16 from:
#   https://ai.google.dev/gemini-api/docs/pricing
#     gemini-3.1-flash-lite : $0.25 in  / $1.50 out
#         (that input price is the text/image/video one; audio input is $0.50,
#          and we only ever send text, so the text price is the right one)
#     gemini-3.5-flash      : $1.50 in  / $9.00 out
# Reported in USD, the currency Google publishes — no FX conversion, so there is
# no second, separately-rotting assumption sitting between the source and the
# report. A model whose price we cannot verify maps to None; `estimate_cost_usd`
# then returns None for it rather than inventing a number for the report.
class ModelPrice(NamedTuple):
    """List price of one model, split by direction (USD per 1M tokens)."""

    input_usd_per_mtok: float
    output_usd_per_mtok: float


PRICING_USD_PER_MTOK: dict[str, ModelPrice | None] = {
    "gemini-3.1-flash-lite": ModelPrice(0.25, 1.50),
    "gemini-3.5-flash":      ModelPrice(1.50, 9.00),
    # Standard API pricing for the Claude A/B experiment (Anthropic's
    # published list prices at experiment time, USD per 1M tokens). Same
    # honesty rule as the Gemini entries: list-price equivalent only.
    "claude-opus-5":         ModelPrice(5.00, 25.00),
}


def estimate_cost_usd(
    model_id: str, input_tokens: int, output_tokens: int
) -> float | None:
    """List-price-equivalent cost of one call, in USD.

    Returns None when the model has no verified price — an unknown model ID, or
    an entry deliberately left None. Callers must treat None as "unknown", not
    as zero.
    """
    price = PRICING_USD_PER_MTOK.get(model_id)
    if price is None:
        return None
    return (
        input_tokens * price.input_usd_per_mtok
        + output_tokens * price.output_usd_per_mtok
    ) / 1_000_000


class LLMUsage(BaseModel):
    """What ONE `llm_call` consumed — the value the caller must pass upward.

    Deliberately NOT written into state by this module: the node that made the
    call returns these numbers, and LangGraph's reducers accumulate them.
    """

    model: str = ""            # real model ID, e.g. "gemini-3.1-flash-lite"
    input_tokens: int = 0
    output_tokens: int = 0
    # None = no verified list price for `model`. Kept nullable rather than
    # defaulting to 0.0 so an unpriced model reads as "unknown" instead of free.
    cost_usd: float | None = None


def sum_usage(usages: Sequence[LLMUsage]) -> LLMUsage:
    """Fold several calls' usage into one per-node total.

    The Architect makes two calls per run, so its node reports the sum. If ANY
    component cost is None the total is None: better an honest "unknown" than a
    total that silently omits an unpriced call.
    """
    models = sorted({u.model for u in usages if u.model})
    costs = [u.cost_usd for u in usages]
    return LLMUsage(
        model=models[0] if len(models) == 1 else ", ".join(models),
        input_tokens=sum(u.input_tokens for u in usages),
        output_tokens=sum(u.output_tokens for u in usages),
        cost_usd=None if any(c is None for c in costs) else sum(costs),
    )


# ── Partial usage on the error path ──────────────────────────────────────
# A node can raise AFTER a call has already been billed — the Architect
# validates its phase-1 result and may raise before phase 2 ever runs. Those
# tokens were really spent, so losing them would understate the run.
#
# The node attaches what it has to the in-flight exception; the `@node` wrapper
# (agents/base.py) reads it back and reports it on the FAILED step. Attaching an
# ATTRIBUTE (rather than wrapping in a new exception type) keeps the original
# exception's type and message intact, so error reporting is unchanged.
_USAGE_ATTR = "llm_usage"


def attach_usage(exc: BaseException, usage: LLMUsage) -> None:
    """Record usage-so-far on an exception, ADDING to anything already there.

    Additive because `llm_call` attaches its own usage when a call is billed but
    unusable, and the node then wraps its whole body — merging means neither
    layer clobbers the other and neither counts twice.
    """
    existing = usage_from_exception(exc)
    setattr(exc, _USAGE_ATTR, sum_usage([existing, usage]) if existing else usage)


def usage_from_exception(exc: BaseException) -> LLMUsage | None:
    """Usage attached to a failed node's exception, or None if it spent nothing."""
    return getattr(exc, _USAGE_ATTR, None)


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
) -> tuple[Any, LLMUsage]:
    """Send one prompt to the named model; return `(reply, usage)`.

    PURE with respect to `state`: this function neither reads from nor writes
    into it. Token accounting is RETURNED instead, because LangGraph persists
    only what a node returns — mutating `state` here was silently discarded.
    The caller sums the usage of its calls and puts the totals in its returned
    dict. (`state` stays in the signature so every agent call site and its test
    stubs keep their shape; it is intentionally unused today.)

    Raises LLMError on failure — the agent's node wrapper turns that into a
    clean FAILED step instead of crashing the pipeline. When a call was billed
    but its output unusable, the usage is attached to the raised error (see
    `attach_usage`) so those real tokens are still counted.

    `system` is passed PER CALL so each agent's prompt lives with the agent,
    not scattered in a registry.

    `response_schema` (optional): a Pydantic model class. When given, the model
    is asked for JSON constrained to that schema and the reply is a VALIDATED
    instance of the class — no manual parsing outside the provider seam. When
    omitted, the reply is a plain `str`.

    PROVIDER ROUTING: `model` is resolved by `resolve_model_routing`; the
    Google path is byte-for-byte the behaviour that existed before the
    seam, and the Anthropic path (`_anthropic_call`) returns the SAME
    downstream types under the SAME contract — a validated Pydantic
    instance or a plain str, plus an `LLMUsage`. There is NO cross-provider
    fallback: a configured Claude call either succeeds as Claude or fails
    as Claude, so an A/B run can never silently mix providers.
    """
    provider, model_id = resolve_model_routing(model)
    if provider == "anthropic":
        return _anthropic_call(
            prompt, system=system, model_id=model_id,
            response_schema=response_schema,
        )

    client = _get_client()
    config = types.GenerateContentConfig(
        system_instruction=system or None,
    )
    if response_schema is not None:
        config.response_mime_type = "application/json"
        config.response_schema = response_schema
    try:
        resp = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=config,
        )
    except Exception as e:
        raise LLMError(f"{model} call failed: {e}") from e

    # Read tokens at the single chokepoint (usage_metadata may be absent).
    meta = getattr(resp, "usage_metadata", None)
    input_tokens = (getattr(meta, "prompt_token_count", 0) or 0) if meta else 0
    output_tokens = (getattr(meta, "candidates_token_count", 0) or 0) if meta else 0
    usage = LLMUsage(
        model=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=estimate_cost_usd(model_id, input_tokens, output_tokens),
    )

    if response_schema is not None:
        parsed = getattr(resp, "parsed", None)
        if parsed is None:  # model returned something unparseable to the schema
            err = LLMError(f"{model} returned no schema-valid JSON for {response_schema.__name__}")
            attach_usage(err, usage)  # billed but unusable — still real tokens
            raise err
        return parsed, usage
    return resp.text, usage


# ── Anthropic provider (the A/B experiment path) ──────────────────────────
# Claude Opus 5 for Architect + Reviewer, selected per role via
# `role_model_override`. One client per process, key loaded once — the same
# caching discipline as the Google client above.
_anthropic_client: Any = None

# Output budget: Opus 5 REQUIRES an explicit max_tokens and adaptive
# thinking spends from it, so the ceiling must comfortably exceed the
# largest schema output the agents request (Blueprint + ADRs + Components
# round-trips measured at ~10k tokens in real runs). 32k sits well inside
# Opus 5's supported range with headroom for thinking; overridable for
# experiments without code changes.
CLAUDE_MAX_TOKENS = int(os.environ.get("CLAUDE_MAX_TOKENS", "32000"))
# Effort is explicit on purpose (never SDK-default): the A/B experiment
# must be able to state exactly which reasoning effort produced a run.
CLAUDE_EFFORT = os.environ.get("CLAUDE_EFFORT", "high")


def _get_anthropic_client() -> Any:
    global _anthropic_client
    if _anthropic_client is None:
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError(
                "anthropic package is not installed — run "
                "`pip install anthropic` to use the Claude routing."
            ) from exc
        load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY not found in .env!")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def _anthropic_call(
    prompt: str,
    *,
    system: str,
    model_id: str,
    response_schema: type | None = None,
) -> tuple[Any, LLMUsage]:
    """One Messages-API call to Claude; `(reply, LLMUsage)` like the Google path.

    Structured output uses Claude's native JSON-schema enforcement
    (`output_config.format`); the returned JSON text is validated into the
    SAME Pydantic class the Google path returns (`model_validate_json`), so
    no Claude-shaped object ever leaves this boundary. Transport errors and
    schema-invalid output raise the same `LLMError` contract, with billed
    tokens attached on the unusable-output path — no silent JSON repair, no
    cross-provider fallback.

    The schema itself goes through Anthropic's OFFICIAL `transform_schema`
    (not a hand-rolled sanitizer) before it is sent — a raw
    `response_schema.model_json_schema()` omits API-required shape like
    `additionalProperties: false` on objects, which is exactly what
    `transform_schema` adds. `client.messages.parse()` applies this same
    transform internally, but its own schema-validation failure raises
    directly out of the SDK call with no response object recoverable — which
    would silently drop the billed-usage-on-failure guarantee above. Calling
    `transform_schema` explicitly and keeping our own `model_validate_json`
    step keeps that guarantee intact.
    """
    client = _get_anthropic_client()
    from anthropic import transform_schema

    output_config: dict[str, Any] = {"effort": CLAUDE_EFFORT}
    if response_schema is not None:
        output_config["format"] = {
            "type": "json_schema",
            "schema": transform_schema(response_schema),
        }
    request: dict[str, Any] = {
        "model": model_id,
        "max_tokens": CLAUDE_MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": output_config,
    }
    if system:
        request["system"] = system
    try:
        resp = client.messages.create(**request)
    except Exception as e:
        raise LLMError(f"anthropic/{model_id} call failed: {e}") from e

    meta = getattr(resp, "usage", None)
    input_tokens = (getattr(meta, "input_tokens", 0) or 0) if meta else 0
    output_tokens = (getattr(meta, "output_tokens", 0) or 0) if meta else 0
    usage = LLMUsage(
        model=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=estimate_cost_usd(model_id, input_tokens, output_tokens),
    )

    text = "".join(
        getattr(block, "text", "")
        for block in (getattr(resp, "content", None) or [])
        if getattr(block, "type", "") == "text"
    )
    if response_schema is not None:
        try:
            return response_schema.model_validate_json(text), usage
        except Exception as e:
            err = LLMError(
                f"anthropic/{model_id} returned no schema-valid JSON "
                f"for {response_schema.__name__}"
            )
            attach_usage(err, usage)  # billed but unusable — still real tokens
            raise err from e
    return text, usage


# ── Self-test: `python -m pipeline.llm` ──────────────────────────────────
# Makes ONE real API call to prove the key + registry + token wiring work.
# Not a unit test (it costs a token or two and needs the network); it's a
# smoke check you run by hand after setting up your .env.
if __name__ == "__main__":
    from pipeline.state import new_run

    s = new_run(raw_prompt="(llm smoke test)")
    print(f"Calling model '{DEFAULT_MODEL}' …")
    reply, usage = llm_call(
        s,
        prompt="Reply with exactly the word: OK",
        system="You are a terse test probe.",
        model=DEFAULT_MODEL,
    )
    # Read the RETURNED usage, not `s` — llm_call no longer touches state.
    print(f"  reply         : {reply!r}")
    print(f"  model         : {usage.model}")
    print(f"  input_tokens  : {usage.input_tokens}")
    print(f"  output_tokens : {usage.output_tokens}")
    print(f"  cost_usd      : {usage.cost_usd} (list-price equivalent; free-tier key)")
    assert usage.input_tokens > 0, "no input tokens reported — token wiring broken"
    print("✓ API key works and token usage was reported back to the caller.")
