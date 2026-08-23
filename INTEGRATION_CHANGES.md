# Integration Changes

This file documents later integration and E2E hardening across team-owned areas.
Original ownership remains unchanged.

All changes below were driven by failures observed in real end-to-end runs
(each retains its evidence in commit messages, `pipeline/DETERMINISM_MAP.md`,
or the persisted run checkpoints). Nothing here reassigns ownership: the
owner's original structure and naming were kept, and the integration work
sits as localized additions around it. Retained cross-owner additions carry
a short `Integration note (Kush):` comment at their boundary in code.

Status values: `Keep` · `Simplified` · `Experiment layer` · `No action`

## Kati — Orchestration / UI / Clarifier / State

| File / Area | Original contribution | Kush integration change | Why | Status |
|---|---|---|---|---|
| `pipeline/agents/clarifier.py` | assume-vs-ask clarifier, context gate | ask-once `_can_ask`, critical-slot relevance tables, safe-assumption parsing, optional-context resolution | unbounded question loops and silently-assumed gaps observed in E2E | Keep |
| `pipeline/state.py` | state schema, reducers | `MigrationStep` model; `OPTIONAL_CONTEXT` pending decision + entry-route guard; review field tuples shared with the gate | brownfield contract; headless callers hung on the new pause | Keep |
| `pipeline/orchestrator.py` | stage routing | optional-context entry guard | keep headless runs non-blocking | Keep |
| `pipeline/run.py` | CLI driver | `resolve_optional_gate`, context workflow commands | same | Keep |
| `pipeline/refine_gate.py` | caps, best-so-far, user rounds | `MAX_REFINE_ITERATIONS` 2→3 | Claude A/B evaluation headroom | Experiment layer |
| `pipeline/llm.py` | single-door Gemini wrapper | Anthropic adapter: routing seam, structured output, streaming, prompted-JSON fallback, failure diagnostics, Claude pricing | Claude/Gemini A/B comparison | Experiment layer |
| `ui.py`, `ui_sections.py` | UI surface | run switcher, new-run reset, context-record versioning, optional-context UI, shared flow renderer, architecture overview sections | state-isolation bugs found in E2E; new read-only surfaces | Keep |
| `ui_workspace.py`, `run_history.py`, `architecture_chat.py` (new) | — | run history, read-only architecture chat with grounding, workspace layout | product features added during integration | Keep |

## Maheen — Architect / Schemas

| File / Area | Original contribution | Kush integration change | Why | Status |
|---|---|---|---|---|
| `pipeline/agents/architect.py` | two-phase structured Architect | output-boundary canonicalization (`sanitize_adr_sources`, `canonicalize_data_flow_endpoints`), requirement-catalog prompt + coverage validation, brownfield/migration prompt discipline, detected-stack block, component identity rule | fabricated sources, phantom flow endpoints, dropped requirements, SLO invention observed in E2E runs | Keep (manifest Simplified) |
| `pipeline/state.py` schema section | Feature/Blueprint/ADR/Component | `MigrationStep` schema | migration-sequence contract for the checks below | Keep |

## Waqar — Reviewer / Evaluation

| File / Area | Original contribution | Kush integration change | Why | Status |
|---|---|---|---|---|
| `pipeline/review_checks.py` | two-layer deterministic checks | new invariants: target-service ownership, flow participants/directionality, migration disposition/targets, technology drift, requirement-feature coverage; actionable constraint failures; shared `resolve_source_reference` | cross-artifact violations observed in E2E; judge and renderer must share one grammar | Keep |
| `pipeline/agents/reviewer.py` | LLM verdict layer | brownfield verdict adjustments; `model=None` + role override hook | verdict consistency; A/B routing | Keep |
| `pipeline/flow_syntax.py` (new) | — | single directional-flow grammar shared by renderer + Reviewer | deduplicated the two drifting parsers | Keep |
| `eval/scenarios.py` | eval scenarios | scenario sync with the new invariants | keep each scenario's labeled flaw valid | Keep |

## Malte — Repo Tooling

| File / Area | Original contribution | Kush integration change | Why | Status |
|---|---|---|---|---|
| `pipeline/repo_analysis.py`, `pipeline/agents/repo_ingestor.py`, `RepoRepresentation` | repo ingestion and representation | none structurally; his fields are read (never rewritten) by the migration-disposition checks | brownfield evidence source | No action |

## Kush — KB / RAG

| File / Area | Original contribution | Kush integration change | Why | Status |
|---|---|---|---|---|
| `Rag Database/`, `chroma_db/`, `chunk_schema.json` | curated KB + indexed corpus | final curation, index versioning, gap report | KB quality for retrieval | Keep |
| `kb_gap_report.py`, `pdf_to_md.py`, `probe_retrieval.py`, `rag_logger.py` | KB maintenance tooling | hardening + tests | reproducible KB upkeep | Keep |
| `pipeline/agents/researcher.py` | RAG retrieval agent | retrieval validation | grounding guarantees | Keep |
| root `architect.py`, `app.py`, notebooks | week-1 single-agent prototype | — | still documented in README/SETUP as reference | No action |
