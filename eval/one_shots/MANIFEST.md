# One-shot baseline artifacts

These are **NOT** AI-Architect pipeline runs. A one-shot baseline is a
single prompt/response pair sent directly to a model with no clarifier, no
retrieval, no reviewer, no refinement loop — the comparison point for what
the multi-agent pipeline adds.

Because they are not pipeline runs, they must never be made discoverable
through `persistence.list_runs()` / `load_state()`, and must never appear
in History, Switch run, or the resume picker. They live here, in their own
plain folder, specifically so nothing under `pipeline.persistence.runs_dir()`
ever has to know they exist. See `EVAL_DEMO_RUNS.md` for how the curated
pipeline runs are bundled instead — a deliberately different mechanism for
a deliberately different kind of artifact.

## Status: PENDING

Neither one-shot baseline has been generated yet. Nothing is bundled here
yet, and nothing here is fabricated to look like a result — nothing in
this task made a live model call. When they exist, each is a `.json` (or
`.md`) file placed directly in this folder, with its row added to the
table below.

| File | Model / provider | Input prompt | Output (summary/link) | Timestamp | Purpose |
|---|---|---|---|---|---|
| _(pending)_ | Google / `gemini-3.1-flash-lite` | — | — | — | Gemini one-shot baseline for the pipeline-vs-one-shot comparison. |
| _(pending)_ | Anthropic / strongest chosen Claude/Opus configuration | — | — | — | Claude one-shot baseline for the pipeline-vs-one-shot comparison. |

## Sanitization

Same rules as bundled pipeline runs (see `demo_runs.py`'s module
docstring): no API keys, no absolute local paths, no `.env` values,
authentic model output only — never edited to look better.
