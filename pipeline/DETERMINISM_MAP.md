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
| 3 | `llm.py` `llm_call` | HYBRID | The wrapper is code; the model response is stochastic | Model registry, per-call system prompt; usage **returned** per call as `LLMUsage`, summed by the `input_tokens`/`output_tokens` reducers, then enforced by `refine_gate.MAX_TOTAL_TOKENS` | API/network error; token overrun now stops the refine loop (but the cap itself is still untuned — see below) |
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

### Correction: row 3's token guardrail was dead until now

This row previously claimed "token accounting" as a working guardrail. It was
not one. `llm_call` recorded usage by mutating `state.input_tokens` /
`state.output_tokens` in place, and no node ever RETURNED those fields — and
LangGraph persists only what a node returns. Every count was therefore
discarded: an 11-call run finished reporting `input_tokens=0, output_tokens=0`,
so `refine_gate.evaluate_caps` compared 0 against its budget on every visit and
the run-cost cap could never trip. Only the iteration cap was doing any work.

Fixed by making `llm_call` pure with respect to state and returning
`(reply, LLMUsage)`; each node sums its own calls and returns the totals, which
the `operator.add` reducers accumulate. `test_token_accounting.py` pins this at
the full-graph level, which is where the bug lived — `llm_call` and the nodes
were each individually fine and only the wiring between them was broken.

Two honest caveats for the report:

* `MAX_TOTAL_TOKENS = 500_000` was chosen while counting was broken, so it has
  never been measured against a real run. It is a placeholder ceiling, not an
  evidence-based budget.
* Costs are computed from Google's published USD list prices, but this project
  runs on free-tier keys. Every dollar figure is a **list-price equivalent** —
  what the run would have cost on the paid tier — never money actually spent.

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