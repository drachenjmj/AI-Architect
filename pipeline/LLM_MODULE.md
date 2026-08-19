# `pipeline/llm.py` — LLM access for agents

The **single door** every agent uses to call an LLM. Import one function; the
module handles the SDK, the API key, model selection, token counting, cost, and
errors. Don't call the Gemini SDK directly from an agent.

## Use it

```python
from pipeline.llm import llm_call

text, usage = llm_call(state, prompt, system="...", model="flash-lite")
```

Inside a node function:

```python
@node("clarifier")
def clarifier_node(state):
    reply, usage = llm_call(
        state,
        prompt=f"Extract the constraints from: {state.initial_request.raw_prompt}",
        system="You are a requirements clarifier. Reply in JSON.",
        model="flash-lite",
    )
    step = make_step("clarifier", state.stage, Stage.CLARIFYING, "note", usage)
    return {                       # RETURN the usage — see below
        "context_record": ...,
        "stage": Stage.CLARIFYING,
        "history": [step],
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }
```

## Signature

`llm_call(state, prompt, *, system="", model="flash-lite", response_schema=None) -> tuple[Any, LLMUsage]`

| Arg | What it is |
|-----|-----------|
| `state` | the shared `ArchitectState`. **Not read from and not written to** — kept in the signature so call sites and test stubs keep their shape |
| `prompt` | the user/task message |
| `system` | your agent's system prompt, passed **per call** (keep it in your agent) |
| `model` | a **name** from the registry, not a model ID |
| `response_schema` | optional Pydantic class; when given, the reply is a validated instance of it instead of `str` |

Returns `(reply, usage)`. `system`, `model` and `response_schema` are keyword-only.

`LLMUsage` carries `model` (the real ID), `input_tokens`, `output_tokens`, and
`cost_usd` (`None` when that model has no verified price).

## Models (the cheap/strong lever)

| Name | Model ID | Use for |
|------|----------|---------|
| `flash-lite` | gemini-3.1-flash-lite | default — cheapest/fastest (e.g. reviewer) |
| `flash` | gemini-3.5-flash | stronger, harder reasoning (e.g. architect) |

Pick by cost/quality per call. `gemini-2.5-pro` is intentionally excluded
(zero free-tier quota; no billing on this project).

## Token accounting — RETURNED, not automatic

**This changed, and it is the one thing to get right.** `llm_call` used to add
usage straight into `state.input_tokens` / `state.output_tokens`. Under LangGraph
that was silently dead: **the graph persists only what a node RETURNS**, so
every count was discarded and the run-cost cap in `refine_gate.py` compared 0
against its budget forever.

So `llm_call` is now pure with respect to `state` and hands the numbers back:

1. Capture the `usage` your call returns.
2. If your node calls the LLM more than once, sum them with `sum_usage(...)`
   (the Architect does — it must report the total of both phases).
3. Pass the usage to `make_step(...)` **and** put the same numbers in your
   returned dict as `input_tokens` / `output_tokens`.

Those two state fields are `Annotated[int, operator.add]` reducers, so return
only YOUR OWN delta — never a running total, or it double-counts. Because each
`StepLog` also carries its own usage, per-agent attribution is free:
`state.usage_by_agent()` and `state.total_cost_usd()`.

If your node raises after a call was already billed, attach what you have with
`attach_usage(exc, usage)` before re-raising; the `@node` wrapper reports it on
the FAILED step so the tokens are not lost.

### If you call the LLM from OUTSIDE the graph

There is one such call today: `clarifier.ask_advisor`, the read-only question a
human may ask while the run is paused at the context lock. It is not a node, so
nothing returns its usage and no reducer accumulates it — which means the two
steps above are the caller's job instead:

1. Append a `StepLog` to `state.history` under **its own agent name**, carrying
   the usage. That is what keeps `usage_by_agent()` complete.
2. Add the same numbers to `state.input_tokens` / `state.output_tokens` by hand.

Both, not one. `usage_by_agent()` promises to sum to the run totals, and a call
that lands in only one of the two ledgers quietly breaks that promise — after
which the per-agent cost table is wrong and nothing says so. If you add another
out-of-graph call, copy that pattern; do not invent a second one.

## Cost — list-price equivalent, not spend

`estimate_cost_usd(model_id, input_tokens, output_tokens)` prices a call from
`PRICING_USD_PER_MTOK` — Google's published Gemini API list prices (paid tier,
per 1M tokens), in the currency Google publishes, so there is no FX assumption
between the source and the report. The table is commented with its source URL
and lookup date in `llm.py`.

**We run on free-tier keys, so no dollar figure here is money actually spent** —
it is what the run *would* have cost on the paid tier. Label it that way
wherever it appears. A model with no verified price maps to `None`, and
`estimate_cost_usd` returns `None` for it rather than inventing a number;
treat `None` as *unknown*, never as zero.

## Errors — automatic

On any failure `llm_call` raises `LLMError`. You don't need to catch it: the
`@node` wrapper in `agents/base.py` already catches exceptions and logs a clean
`FAILED` step, so a bad call won't crash the pipeline.

## Rules

- **Do** route every LLM call through `llm_call`.
- **Do** RETURN the usage you get back — a node that only mutates `state` loses
  it, which is exactly the bug this module was fixed for.
- **Don't** import `google.genai` or build your own client in an agent — that
  breaks token counting and the determinism story.
- **Don't** hard-code a model ID; use a registry name.
- **Don't** add a price you have not verified against Google's pricing page;
  leave the entry `None` instead. These numbers go into a graded business case.
- Keep your system prompt **in your agent**, not in `llm.py`.

## Setup (once, per developer)

1. `cp .env.example .env`
2. Paste **your own** Gemini key into `.env` after `GEMINI_API_KEY=`
   (each developer uses their own key; `.env` is gitignored — never commit it).
3. Smoke-test: `python -m pipeline.llm` → prints `OK` + token counts.

## Adding a model / provider later

Add a line to `MODELS` in `llm.py` (name → model ID). To add Claude/GPT,
the map becomes name → (provider, id) and `llm_call` branches on provider —
**no agent changes needed**. Contact Kati before editing the registry.
