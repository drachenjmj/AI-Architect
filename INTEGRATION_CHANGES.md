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
| `ui.py`, `ui_sections.py` | UI surface | run switcher, new-run reset, context-record versioning, optional-context UI, shared flow renderer, architecture overview sections; Architecture-diagram Overview/Detailed view split with a Business/Events/Data/Observability flow filter (`build_overview_edges`, `filter_edges_by_category`) | state-isolation bugs found in E2E; new read-only surfaces; a ~15-component/~44-flow holdout diagram was real information but visually unreadable as one undifferentiated graph | Keep |
| `ui_workspace.py`, `run_history.py`, `architecture_chat.py` (new) | — | run history, read-only architecture chat with grounding, workspace layout | product features added during integration | Keep |

## Maheen — Architect / Schemas

| File / Area | Original contribution | Kush integration change | Why | Status |
|---|---|---|---|---|
| `pipeline/agents/architect.py` | two-phase structured Architect | output-boundary canonicalization (`sanitize_adr_sources`, `canonicalize_data_flow_endpoints`), requirement-catalog prompt + coverage validation, brownfield/migration prompt discipline, detected-stack block, component identity rule; `<decision_evidence>` prompt block + `sanitize_adr_evidence_ids` boundary + DECISION EVIDENCE GROUNDING prompt rule | fabricated sources, phantom flow endpoints, dropped requirements, SLO invention observed in E2E runs; ADRs need to be traceable to literature actually retrieved this run, not just the model's own training knowledge | Keep (manifest Simplified) |
| `pipeline/state.py` schema section | Feature/Blueprint/ADR/Component | `MigrationStep` schema; `ADR.evidence_ids`, `KBChunk.evidence_id`, `DecisionTopic` model, `ArchitectState.decision_topics` | migration-sequence contract for the checks below; stable per-run evidence identity for decision-level literature grounding | Keep |

## Waqar — Reviewer / Evaluation

| File / Area | Original contribution | Kush integration change | Why | Status |
|---|---|---|---|---|
| `pipeline/review_checks.py` | two-layer deterministic checks | new invariants: target-service ownership, flow participants/directionality, migration disposition/targets, technology drift, requirement-feature coverage; actionable constraint failures; shared `resolve_source_reference`; decision-level KB literature-grounding gate (`qualifying_kb_evidence_ids`, `_check_kb_evidence_grounding`, `kb_evidence_grounding` rubric score) | cross-artifact violations observed in E2E; judge and renderer must share one grammar; an ADR must not be able to claim literature support that was never actually retrieved this run | Keep |
| `pipeline/agents/reviewer.py` | LLM verdict layer | brownfield verdict adjustments; `model=None` + role override hook; wired `kb_evidence_grounding` into `RubricScores` | verdict consistency; A/B routing; literature-grounding verdict | Keep |
| `pipeline/flow_syntax.py` (new) | — | single directional-flow grammar shared by renderer + Reviewer; `classify_flow` — one centralized, display-only Business/Events/Data/Observability classifier reused by the diagram's Detailed-view filter | deduplicated the two drifting parsers; the diagram filter must classify the exact same edges the renderer draws, never a second grammar | Keep |
| `eval/scenarios.py` | eval scenarios | scenario sync with the new invariants | keep each scenario's labeled flaw valid | Keep |
| `docs/prompt_quality/06_reviewer_report_schema.json` | frozen report schema | added `kb_evidence_grounding` to `rubric_scores` | schema must track the new rubric field | Keep |

## Malte — Repo Tooling

| File / Area | Original contribution | Kush integration change | Why | Status |
|---|---|---|---|---|
| `pipeline/repo_analysis.py`, `pipeline/agents/repo_ingestor.py`, `RepoRepresentation` | repo ingestion and representation | none structurally; his fields are read (never rewritten) by the migration-disposition checks | brownfield evidence source | No action |

## Kush — KB / RAG

| File / Area | Original contribution | Kush integration change | Why | Status |
|---|---|---|---|---|
| `Rag Database/`, `chroma_db/`, `chunk_schema.json` | curated KB + indexed corpus | final curation, index versioning, gap report | KB quality for retrieval | Keep |
| `kb_gap_report.py`, `pdf_to_md.py`, `probe_retrieval.py`, `rag_logger.py` | KB maintenance tooling | hardening + tests | reproducible KB upkeep | Keep |
| `pipeline/agents/researcher.py` | RAG retrieval agent | replaced one global Top-3 query with bounded, case-derived decision-topic planning (`plan_decision_topics`, 4–8 topics) and per-topic retrieval (still Top-3 per topic, same threshold/logging/web-fallback path via the existing `retrieve_chunks`), cross-topic dedup, a deterministic total cap (`MAX_TOTAL_EVIDENCE = 15`), and stable per-run evidence IDs assigned only to curated-KB chunks | a strong model can write plausible ADRs from three chunks and its own training knowledge; the project claim requires every ADR to be traceable to literature THIS run actually retrieved from OUR curated KB, and the existing per-topic gap machinery (RagLogger → `kb_gap_report.py`) already surfaces coverage gaps with no changes needed once retrieval is topic-scoped | Keep |
| `ui_sections.py` (ADR rendering) | full ADR/component detail views | `_render_literature_evidence`: compact, expandable "Literature evidence" block per ADR, resolving `evidence_ids` to source/page — never raw chunk text by default, never a web-fallback item shown as curated literature | evidence traceability needs to be visible to a human, not just enforced in the pipeline | Keep |
| root `architect.py`, `app.py`, notebooks | week-1 single-agent prototype | — | still documented in README/SETUP as reference | No action |
