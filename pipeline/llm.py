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

import json
import os
from typing import Any, NamedTuple, Sequence

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from pipeline.state import ArchitectState


# ── Model registry: friendly NAME -> real model ID ───────────────────────
# This is the cheap/strong lever. Agents say model="flash"; only this line
# changes if we swap the underlying model. Adding another provider maps this
# to (provider, model_id) and llm_call dispatches on provider.
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
#     "kiconnect/OpenAI GPT OSS 120b ...")
#   * a bare real model ID whose family     -> that provider (a gemini-*
#     prefix is known                          id is Google).
# Roles pick their routing per call via `role_model_override`, so the
# Architect and the Reviewer can be pointed at KI Connect by environment
# while the Clarifier, the advisor, RAG and the web fallback stay on Gemini.
PROVIDERS: tuple[str, ...] = ("google", "kiconnect")
_PROVIDER_BY_PREFIX: dict[str, str] = {
    "gemini": "google",
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

        ARCHITECT_LLM_PROVIDER=kiconnect
        ARCHITECT_LLM_MODEL=<kiconnect model> -> "kiconnect/<kiconnect model>"
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


# ── Role-specific Gemini thinking level (final 2×2 evaluation) ────────────
# The Gemini arm of the final evaluation must run the Architect and the
# Reviewer with an EXPLICIT thinking level (HIGH), not the model's silent
# default — while the cheap calls (Clarifier, advisor, Ask-AI chat, repo
# ingestor) keep today's default behavior. Same per-call env-read shape as
# `role_model_override` above, so the two seams cannot drift apart.
#
# Values are the Gemini API's own `ThinkingLevel` contract (verified against
# the installed google-genai SDK's enum). NOTE: the SDK itself only WARNS on
# an unknown string and would send it anyway — so the explicit allowlist
# below is the real validation, and an invalid value must fail LOUDLY here
# (never a silent default-thinking run the evaluation believes was HIGH).
GEMINI_THINKING_LEVELS: tuple[str, ...] = ("minimal", "low", "medium", "high")


def role_gemini_thinking_level(role: str) -> str | None:
    """`{ROLE}_GEMINI_THINKING_LEVEL` for one agent role, this call. Pure.

    Returns None when unset — the caller then omits the setting entirely and
    the provider/model's existing default behavior is preserved exactly (the
    safe default: HIGH is opted into through `.env`, never global). A value
    is returned LOWERCASED but otherwise UNVALIDATED here; `llm_call`
    validates at the single chokepoint where both env-readers and direct
    callers converge.

    Deliberately NO role outside architect/reviewer is read anywhere: the
    Clarifier, advisor, Ask-AI chat and repo ingestor have no
    `{ROLE}_GEMINI_THINKING_LEVEL` seam at all, so an evaluation-time
    ARCHITECT/REVIEWER setting cannot reach them by construction.
    """
    raw = os.environ.get(f"{role.upper()}_GEMINI_THINKING_LEVEL", "").strip()
    return raw.lower() or None


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
    # The EFFECTIVE Gemini thinking level this call ran with (e.g. "high"),
    # or None when none was requested / this is not a Google call. Additive
    # (default None): old checkpoints and other-provider usages deserialize
    # unchanged, and the final evaluation can PROVE what setting a run used
    # instead of re-deriving it from `.env`.
    thinking_level: str | None = None


def sum_usage(usages: Sequence[LLMUsage]) -> LLMUsage:
    """Fold several calls' usage into one per-node total.

    The Architect makes two calls per run, so its node reports the sum. If ANY
    component cost is None the total is None: better an honest "unknown" than
    a total that silently omits an unpriced call.
    """
    models = sorted({u.model for u in usages if u.model})
    costs = [u.cost_usd for u in usages]
    levels = sorted({u.thinking_level for u in usages if u.thinking_level})
    return LLMUsage(
        model=models[0] if len(models) == 1 else ", ".join(models),
        input_tokens=sum(u.input_tokens for u in usages),
        output_tokens=sum(u.output_tokens for u in usages),
        cost_usd=None if any(c is None for c in costs) else sum(costs),
        # Same merge convention as `model`: one level reads as itself, a mix
        # (a node whose calls disagreed) reads as the sorted joined list —
        # visible, never silently dropped to the first value.
        thinking_level=levels[0] if len(levels) == 1 else (", ".join(levels) or None),
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
    thinking_level: str | None = None,
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

    `thinking_level` (optional, GOOGLE calls only): the Gemini
    `ThinkingConfig.thinking_level` for this call — the role-specific seam is
    `role_gemini_thinking_level("architect" | "reviewer")`, and passing the
    resolved value here is the ONLY sanctioned way to set it. None (the
    default) omits the setting entirely and preserves the model's existing
    behavior exactly. Validated against the API's own ThinkingLevel contract
    (`GEMINI_THINKING_LEVELS`) at this chokepoint: an invalid value raises
    LLMError rather than silently running at default thinking — an
    evaluation run that believes it ran HIGH must never discover it did not.
    Model support for the requested level is the provider's own contract: a
    model that rejects it fails through the same LLMError wrapping below,
    never a silent fallback.

    PROVIDER ROUTING: `model` is resolved by `resolve_model_routing`; the
    Google path is byte-for-byte the behaviour that existed before the
    seam. The KI Connect path returns the SAME downstream types under the
    SAME contract — a validated Pydantic instance or a plain str, plus an
    `LLMUsage`. There is NO cross-provider fallback: a configured call
    either succeeds on that provider or fails there, so an evaluation run
    can never silently mix providers.
    """
    provider, model_id = resolve_model_routing(model)
    if provider == "kiconnect":
        return _kiconnect_call(
            prompt, system=system, model_id=model_id,
            response_schema=response_schema,
        )

    if thinking_level is not None and thinking_level not in GEMINI_THINKING_LEVELS:
        raise LLMError(
            f"Invalid Gemini thinking_level {thinking_level!r} for {model_id}. "
            f"Valid levels: {', '.join(GEMINI_THINKING_LEVELS)}."
        )

    client = _get_client()
    config = types.GenerateContentConfig(
        system_instruction=system or None,
    )
    if thinking_level is not None:
        # Verified against the installed google-genai SDK: the field is
        # `thinking_config.thinking_level` (there is no top-level
        # thinking_level on GenerateContentConfig).
        config.thinking_config = types.ThinkingConfig(
            thinking_level=thinking_level,
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
        thinking_level=thinking_level,
    )

    if response_schema is not None:
        parsed = getattr(resp, "parsed", None)
        if parsed is None:  # model returned something unparseable to the schema
            err = LLMError(f"{model} returned no schema-valid JSON for {response_schema.__name__}")
            attach_usage(err, usage)  # billed but unusable — still real tokens
            raise err
        return parsed, usage
    return resp.text, usage


def _prompted_json_contract(response_schema: type) -> str:
    """The provider-local output-format instruction appended to the prompt
    so the model returns schema-shaped JSON without native structured-output
    support (KI Connect).

    The COMPLETE Pydantic JSON schema is embedded exactly once, serialized
    deterministically (`model_json_schema` + `json.dumps`), unmutated — the
    final validator remains the original Pydantic class, this text is only
    the contract the model is asked to satisfy.
    """
    schema_text = json.dumps(
        response_schema.model_json_schema(), indent=2, ensure_ascii=False
    )
    return (
        "\n\nReturn exactly one JSON document matching the following "
        "JSON Schema.\n"
        "Do not use Markdown fences.\n"
        "Do not include any commentary before or after the JSON.\n"
        "Do not omit required fields.\n"
        "Do not invent fields outside the schema.\n"
        f"<json_schema>\n{schema_text}\n</json_schema>"
    )


# ── KI Connect NRW provider ──────────────────────────────────────────────
# The University of Cologne provides institution-funded access through an
# OpenAI-compatible endpoint. Live probes on 2026-08-24 established two
# important boundaries that are encoded here rather than hidden:
#   * /v1/chat/completions works and reports token usage;
#   * response_format/json_schema is accepted but NOT enforced, and the
#     available GPT-OSS model does not support /v1/responses.
# Therefore every structured call uses the complete prompt-embedded schema
# above and the original Pydantic class remains the only acceptance gate.
# No fence stripping, JSON repair, retry, or provider fallback is allowed.
KICONNECT_DEFAULT_BASE_URL = "https://chat.kiconnect.nrw/api/v1"
KICONNECT_DEFAULT_MAX_TOKENS = 32000
KICONNECT_DEFAULT_TIMEOUT_SECONDS = 180.0
_kiconnect_session: requests.Session | None = None


def _get_kiconnect_session() -> requests.Session:
    global _kiconnect_session
    if _kiconnect_session is None:
        _kiconnect_session = requests.Session()
    return _kiconnect_session


def _kiconnect_settings() -> tuple[str, str, int, float]:
    """Return validated KI Connect settings after loading the local .env."""
    load_dotenv()
    api_key = os.getenv("KICONNECT_API_KEY", "").strip()
    if not api_key:
        raise LLMError("KICONNECT_API_KEY not found in .env!")
    base_url = os.getenv(
        "KICONNECT_BASE_URL", KICONNECT_DEFAULT_BASE_URL
    ).strip().rstrip("/")
    if not base_url:
        raise LLMError("KICONNECT_BASE_URL is empty in .env!")
    try:
        max_tokens = int(os.getenv(
            "KICONNECT_MAX_TOKENS", str(KICONNECT_DEFAULT_MAX_TOKENS)
        ))
        timeout = float(os.getenv(
            "KICONNECT_TIMEOUT_SECONDS",
            str(KICONNECT_DEFAULT_TIMEOUT_SECONDS),
        ))
    except ValueError as exc:
        raise LLMError(
            "KICONNECT_MAX_TOKENS and KICONNECT_TIMEOUT_SECONDS must be numeric"
        ) from exc
    if max_tokens <= 0 or timeout <= 0:
        raise LLMError(
            "KICONNECT_MAX_TOKENS and KICONNECT_TIMEOUT_SECONDS must be positive"
        )
    return api_key, base_url, max_tokens, timeout


def _kiconnect_call(
    prompt: str,
    *,
    system: str,
    model_id: str,
    response_schema: type | None = None,
) -> tuple[Any, LLMUsage]:
    """One KI Connect chat-completion call under the shared LLM contract."""
    api_key, base_url, max_tokens, timeout = _kiconnect_settings()
    request_prompt = (
        prompt + _prompted_json_contract(response_schema)
        if response_schema is not None else prompt
    )
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": request_prompt})
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    try:
        resp = _get_kiconnect_session().post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise LLMError(f"kiconnect/{model_id} call failed: {exc}") from exc

    choices = data.get("choices") if isinstance(data, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    text = message.get("content") if isinstance(message, dict) else None
    if not isinstance(text, str):
        raise LLMError(
            f"kiconnect/{model_id} returned no text completion"
        )

    meta = data.get("usage") if isinstance(data, dict) else None
    input_tokens = (
        int(meta.get("prompt_tokens", 0) or 0)
        if isinstance(meta, dict) else 0
    )
    output_tokens = (
        int(meta.get("completion_tokens", 0) or 0)
        if isinstance(meta, dict) else 0
    )
    usage = LLMUsage(
        model=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=estimate_cost_usd(model_id, input_tokens, output_tokens),
    )

    if response_schema is not None:
        try:
            return response_schema.model_validate_json(text), usage
        except Exception as exc:
            finish_reason = (
                choice.get("finish_reason")
                if isinstance(choice, dict) else None
            )
            err = LLMError(
                f"kiconnect/{model_id} returned no schema-valid JSON "
                f"for {response_schema.__name__} "
                f"(finish_reason={finish_reason!r}, "
                f"input_tokens={input_tokens}, "
                f"output_tokens={output_tokens}, "
                f"response_chars={len(text)}): "
                f"{_validation_summary(exc)}"
            )
            attach_usage(err, usage)
            raise err from exc
    return text, usage


def _validation_summary(exc: BaseException) -> str:
    """Concise summary of a Pydantic/JSON validation failure, for an LLMError.

    Built ONLY from each error's location, message and type. Pydantic's
    `errors()` dicts also carry `input`/`ctx` — for this module that is the
    model's GENERATED content, which a diagnostic must never dump. Falls
    back to a length-capped single-line `str(exc)` for non-Pydantic
    exceptions, so the underlying reason is always the model's own report,
    never a guessed diagnosis.
    """
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            details = errors()
        except Exception:
            details = None
        if isinstance(details, list) and details:
            parts = []
            for d in details[:3]:
                if not isinstance(d, dict):
                    continue
                loc = ".".join(str(p) for p in (d.get("loc") or ())) or "<root>"
                parts.append(f"{loc}: {d.get('msg', '?')} [type={d.get('type', '?')}]")
            summary = "; ".join(parts)
            count = getattr(exc, "error_count", None)
            total = count() if callable(count) else len(details)
            if isinstance(total, int) and total > 3:
                summary += f" (+{total - 3} more)"
            if summary:
                return summary
    text = " ".join(str(exc).split())
    return text[:500] + "..." if len(text) > 500 else text


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
