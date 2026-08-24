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
| `pipeline/agents/clarifier.py` | assume-vs-ask clarifier, context gate | ask-once `_can_ask`, critical-slot relevance tables, safe-assumption parsing, optional-context resolution; `discuss_field`/`FieldDiscussionResponse`/`field_purpose_hint` — a per-field "Ask AI" LLM call, session-only (never touches `state.advisory_turns`/`history`), reusing `ADVISOR_MODEL` and `_format_repo_context` (whose signature was widened to take `RepoRepresentation \| None` directly instead of `ArchitectState`, so it also works before any run exists) | unbounded question loops and silently-assumed gaps observed in E2E; BCG feedback: the user should be able to discuss a field before committing an answer, at every clarification/ground-truth decision point | Keep |
| `pipeline/state.py` | state schema, reducers | `MigrationStep` model; `OPTIONAL_CONTEXT` pending decision + entry-route guard; review field tuples shared with the gate | brownfield contract; headless callers hung on the new pause | Keep |
| `pipeline/orchestrator.py` | stage routing | optional-context entry guard | keep headless runs non-blocking | Keep |
| `pipeline/run.py` | CLI driver | `resolve_optional_gate`, context workflow commands | same | Keep |
| `pipeline/refine_gate.py` | caps, best-so-far, user rounds | `MAX_REFINE_ITERATIONS` 2→3 | Claude A/B evaluation headroom | Experiment layer |
| `pipeline/llm.py` | single-door Gemini wrapper | Anthropic adapter: routing seam, structured output, streaming, prompted-JSON fallback, failure diagnostics, Claude pricing | Claude/Gemini A/B comparison | Experiment layer |
| `ui.py`, `ui_sections.py` | UI surface | run switcher, new-run reset, context-record versioning, optional-context UI, shared flow renderer, architecture overview sections; Architecture-diagram Overview/Detailed view split with a Business/Events/Data/Observability flow filter (`build_overview_edges`, `filter_edges_by_category`); `ui_sections.render_ask_ai` — a compact per-field "Ask AI" popover reused by every clarification/approval/intake form. The required-questions form, the optional-context form, the Context Record approval form, and the pre-run intake form each draw the popover+label and the field's own input as PLAIN (non-form) widgets, one interleaved unit per field — NOT inside `st.form(...)`. A prior revision wrapped the input alone in `st.form` (label/popover outside, input inside via `form.text_input(...)`); that batched the input until one submit, which (a) fed `discuss_field` a stale, last-submitted draft instead of whatever was currently typed when Ask AI was opened before submitting, and (b) painted the form's single reserved slot — every input, then Submit — as one block AFTER every label+popover, instead of each question being one unit. Each screen's own single explicit action ("Submit answers" / "Skip"+"Continue" / "Save changes"+"Approve and continue" / "Start", all plain `st.button`s now) stays the only commit boundary into the pipeline; typing and opening Ask AI never do | state-isolation bugs found in E2E; new read-only surfaces; a ~15-component/~44-flow holdout diagram was real information but visually unreadable as one undifferentiated graph; BCG feedback (context-aware clarification discussions); UX finalization pass fixing the reported input/label layout regression and the stale-draft risk `st.form` introduced | Keep |
| `ui_workspace.py`, `run_history.py`, `architecture_chat.py` (new) | — | run history, read-only architecture chat with grounding, workspace layout | product features added during integration | Keep |
| `field_discussion.py` (new) | — | session-only per-(run/pre-run-scope, field) "Ask AI" discussion history + the REACT half (`ask`) that calls `clarifier.discuss_field`; same three-way DRAW/REACT/store split as `architecture_chat.py`/`ui_sections.py`/`ui.py`. `pre_run_scope()` mints one id per not-yet-submitted intake draft (cached in `st.session_state`, replacing the single fixed `PRE_RUN_SCOPE` sentinel reused for the whole browser session) so a fresh draft after "New run" clears the session never inherits an abandoned draft's discussion, while an ordinary rerun of the same draft keeps its history | keeps `ui_sections.py`'s DRAW-only rule intact (it never calls the LLM itself — see that module's docstring) while giving every screen one shared, run-and-field-isolated history store | Keep |

## Maheen — Architect / Schemas

| File / Area | Original contribution | Kush integration change | Why | Status |
|---|---|---|---|---|
| `pipeline/agents/architect.py` | two-phase structured Architect | output-boundary canonicalization (`sanitize_adr_sources`, `canonicalize_data_flow_endpoints`), requirement-catalog prompt + coverage validation, brownfield/migration prompt discipline, detected-stack block, component identity rule; `<decision_evidence>` prompt block + `sanitize_adr_evidence_ids` boundary + DECISION EVIDENCE GROUNDING prompt rule; fail-fast `_validate_no_invented_quantitative_targets` on generated Features (before phase 2 ever runs); QUANTITATIVE TARGETS rule added to `FEATURE_SYSTEM_PROMPT`; DECISION-TOPIC MAPPING prompt rule for `ADR.related_decision_topic_ids` | fabricated sources, phantom flow endpoints, dropped requirements, SLO invention observed in E2E runs; ADRs need to be traceable to literature actually retrieved this run, not just the model's own training knowledge; a live Gemini E2E showed invented numeric Feature targets (`under 200ms`, `10x traffic`) surviving to a PASS, and material decisions (a Strangler-fig migration) with no governing ADR at all | Keep (manifest Simplified) |
| `pipeline/state.py` schema section | Feature/Blueprint/ADR/Component | `MigrationStep` schema; `ADR.evidence_ids`, `KBChunk.evidence_id`, `DecisionTopic` model, `ArchitectState.decision_topics`; additive `ADR.related_decision_topic_ids` (default `[]`, backward-compatible) | migration-sequence contract for the checks below; stable per-run evidence identity for decision-level literature grounding; exact, non-fuzzy ADR-to-DecisionTopic mapping for material-decision coverage | Keep |

## Waqar — Reviewer / Evaluation

| File / Area | Original contribution | Kush integration change | Why | Status |
|---|---|---|---|---|
| `pipeline/review_checks.py` | two-layer deterministic checks | new invariants: target-service ownership, flow participants/directionality, migration disposition/targets, technology drift, requirement-feature coverage; actionable constraint failures; shared `resolve_source_reference`; decision-level KB literature-grounding gate (`qualifying_kb_evidence_ids`, `_check_kb_evidence_grounding`, `kb_evidence_grounding` rubric score); `find_unauthorized_quantitative_targets` (shared by the architect's fail-fast gate and this module's defense-in-depth `_check_quantitative_targets` scan of Blueprint/ADR/Component/migration-step prose); material-decision coverage gate (`MATERIAL_DECISION_TOPICS`, `MIGRATION_MATERIAL_TOPIC`, `_check_material_decision_coverage`, `_check_adr_evidence_topic_provenance`) — folded into the existing `traceability`/`kb_evidence_grounding` scores, no new rubric field | cross-artifact violations observed in E2E; judge and renderer must share one grammar; an ADR must not be able to claim literature support that was never actually retrieved this run; a live E2E showed invented numeric targets and un-ADR'd material decisions (migration strategy, service boundaries) surviving to a PASS | Keep |
| `pipeline/agents/reviewer.py` | LLM verdict layer | brownfield verdict adjustments; `model=None` + role override hook; wired `kb_evidence_grounding` into `RubricScores`; `best_practice_grounding` question extended to also judge ADR topic-mapping accuracy and material recommendations hiding outside the ADR trail (no new score field — `<deterministic_check_results>` already carries the new checks via `checks.model_dump_json()`) | verdict consistency; A/B routing; literature-grounding verdict; material-decision semantic review | Keep |
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
| `ui_sections.py` (ADR rendering) | full ADR/component detail views | `_render_literature_evidence`: compact, expandable "Literature evidence" block per ADR, resolving `evidence_ids` to source/page — never raw chunk text by default, never a web-fallback item shown as curated literature; one added `_chips("Decision topics", adr.related_decision_topic_ids)` line, same pattern as the existing "Related features"/"Related components" chips | evidence traceability needs to be visible to a human, not just enforced in the pipeline; material-decision topic mapping needs the same minimal visibility | Keep |
| root `architect.py`, `app.py`, notebooks | week-1 single-agent prototype | — | still documented in README/SETUP as reference | No action |
