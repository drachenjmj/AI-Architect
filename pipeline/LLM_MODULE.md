# `pipeline/llm.py` — LLM access for agents

The **single door** every agent uses to call an LLM. Import one function; the
module handles the SDK, the API key, model selection, token counting, and
errors. Don't call the Gemini SDK directly from an agent.

## Use it

```python
from pipeline.llm import llm_call

text = llm_call(state, prompt, system="...", model="flash-lite")
```

Inside an agent's `_act(self, state)`:

```python
def _act(self, state):
    reply = llm_call(
        state,
        prompt=f"Extract the constraints from: {state.initial_request.raw_prompt}",
        system="You are a requirements clarifier. Reply in JSON.",
        model="flash-lite",
    )
    # ... parse reply, write it into your own state field ...
    return "extracted draft context record"   # one-line note for the log
```

## Signature

`llm_call(state, prompt, *, system="", model="flash-lite") -> str`

| Arg | What it is |
|-----|-----------|
| `state` | the shared `ArchitectState` (required — tokens are added to it) |
| `prompt` | the user/task message |
| `system` | your agent's system prompt, passed **per call** (keep it in your agent) |
| `model` | a **name** from the registry, not a model ID |

Returns the reply text. `system` and `model` are keyword-only.

## Models (the cheap/strong lever)

| Name | Model | Use for |
|------|-------|---------|
| `flash-lite` | gemini-2.5-flash-lite | default — cheapest/fastest (e.g. reviewer) |
| `flash` | gemini-2.5-flash | stronger, harder reasoning (e.g. architect) |

Pick by cost/quality per call. `gemini-2.5-pro` is intentionally excluded
(zero free-tier quota; no billing on this project).

## Token accounting — automatic

Every call adds usage into `state.input_tokens` / `state.output_tokens`.
You don't do anything; just call `llm_call` and the totals are tracked.

## Errors — automatic

On any failure `llm_call` raises `LLMError`. You don't need to catch it: the
`Agent.base.run()` wrapper already catches exceptions and logs a clean
`FAILED` step, so a bad call won't crash the pipeline.

## Rules

- **Do** route every LLM call through `llm_call`.
- **Don't** import `google.genai` or build your own client in an agent — that
  breaks token counting and the determinism story.
- **Don't** hard-code a model ID; use a registry name.
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
