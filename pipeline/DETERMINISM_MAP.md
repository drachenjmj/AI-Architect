# Determinism map

Purpose: a complete inventory of every operation in the pipeline, each classified
by *how* it produces its output — deterministic **code**, stochastic **LLM**, a
**hybrid** of the two, or an external **human** step. For every non-deterministic
step it also records the guardrail that constrains it and the way it can fail.

This is the concrete evidence for the project's "determinism by default" principle:
it shows the LLM is used *only* where genuine reasoning is required, and never
touches control flow. The single source of truth for routing is the
`STAGE_TO_NODE` table in `orchestrator.py` — the arrow labels below are its keys.

## Flowchart

The boxes are where reasoning happens; the **arrows and start/end are code** —
that is the whole claim of the map. Colour encodes determinism class.

```mermaid
flowchart TD
    start([start]) -->|CREATED| clarifier
    clarifier -->|CLARIFYING| researcher
    clarifier <-->|AWAITING_INPUT / resume| human[human input]
    researcher -->|RESEARCHING| architect
    architect -->|DESIGNING| reviewer
    reviewer -->|pass| done([end])
    reviewer -.->|refine loop, W3 todo| architect

    classDef code fill:#EAF3DE,stroke:#639922,color:#173404;
    classDef hybrid fill:#FAEEDA,stroke:#BA7517,color:#412402;
    classDef llm fill:#FCEBEB,stroke:#E24B4A,color:#501313;
    classDef human fill:#F1EFE8,stroke:#888780,color:#2C2C2A;

    class start,done code;
    class clarifier hybrid;
    class researcher,architect,reviewer llm;
    class human human;
```

Legend: **code** (green) — routing, control flow, state; deterministic and
reproducible. **hybrid** (amber) — LLM judgment wrapped in a code + human gate.
**llm** (red) — generative reasoning; stochastic. **human** (gray) — external
input, not automated.

## Table

| # | Step | Type | What produces the output | Guardrail (what tames it) | Failure mode |
|---|------|------|--------------------------|---------------------------|--------------|
| 1 | `run.py` entry / `new_run` | CODE | Wraps the raw prompt into an `ArchitectState` | Pydantic v2 validation | none (pure) |
| 2 | Orchestrator `_route` / `_entry_route` / `STAGE_TO_NODE` | CODE | Picks the next node from `state.stage` | Static table; unknown stage → END; `recursion_limit` cap | mis-wired route → fails loudly, never hangs |
| 3 | `llm.py` `llm_call` | HYBRID | The wrapper is code; the model response is stochastic | Model registry, per-call system prompt, token accounting | API/network error, token overrun |
| 4 | Clarifier `_act` | HYBRID | LLM judges assume-vs-ask; the human answers | Deterministic gate + HITL `interrupt`; writes only `ContextRecord` | hallucinated assumption, poor question |
| 5 | Human input | HUMAN | User supplies answers on resume | — (external) | missing / ambiguous answer |
| 6 | Researcher `_act` | LLM | Reasons over RAG-retrieved KB chunks | RAG grounding (retrieval is code), citations | retrieval miss, ungrounded claim |
| 7 | Architect `_act` | LLM | Generates the blueprint / ADRs / component descriptions | Output-schema validation, retry cap | schema-invalid output, drift |
| 8 | Reviewer PASS/FAIL judgment | LLM | Scores the artifact against the eval rubric | Eval rubric | wrong verdict |
| 9 | Reviewer → REFINING routing | CODE | Sets `stage`; `_route` reads it (W3) | Same static table + retry cap | — |
| 10 | `persistence.checkpoint` (state on disk) | CODE | Serialises each emitted state to `.cache/runs/<run_id>/<NNN>_<stage>.json` | Atomic write (temp file + `os.replace`); `checkpoint` swallows every error | lost checkpoint (run continues), corrupt file (skipped by `list_runs`, raises in `load_state`) |

Rows 8 and 9 are split on purpose: the reviewer's *judgment* (is this good?) is
stochastic LLM, but the *action* taken on that judgment (loop back vs. finish) is
pure code in the router. That separation is exactly what proves the LLM never
decides control flow.

Row 10 is code for the same reason: persisting a state is plumbing, not a
judgment. It is hooked on a single line in `run_pipeline_streaming`, so it
observes transitions without participating in routing.

## Failure semantics

`run_pipeline` never mutates its input, on **any** path. Normal transitions come
back as freshly validated states from the graph; the `GraphRecursionError`
(step-cap) path marks a deep copy FAILED rather than writing into the caller's
object. So a caller may always keep the state it passed in — to retry from, or
to compare against — and trust that it is unchanged.

One consequence for persistence: the checkpoint written on the step-cap path
comes from the `finally` block in `run_pipeline_streaming`, not from a stream
emission, because that FAILED state is born from an exception and never reaches
the graph. It is the only checkpoint in the system with that provenance, and
`test_persistence.test_step_cap_failure_is_checkpointed` pins it.