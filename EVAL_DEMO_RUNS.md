# Evaluation demo runs

This branch (`eval-demo-runs`) ships a small, curated set of **finished,
successful full-pipeline runs** inside the repository, under
`demo_runs/<run_id>/`. `ui.py` seeds them into the local run store
(`.cache/runs/`, same as any run you produce yourself) the first time you
start the app — see `demo_runs.py` for exactly how.

**These are curated successful demonstration runs.** They exist so
teammates and reviewers can inspect final artifacts — the Context Record,
Blueprint, ADRs, Component Descriptions, Reviewer report, RAG evidence, run
trace, and token/cost accounting — **without spending API quota or
re-running the pipeline**.

> These bundled runs are curated qualitative examples. They are not the final quantitative 2×2 evaluation sample because they were not all generated on the identical final code version.

Nothing bundled is scored or editorialized — each run is shown exactly as
the pipeline produced it, under the code version that existed when it ran.
Historical runs are NOT "upgraded" to current-schema behavior: their
artifacts stay byte-faithful to what the model actually produced at the
time (only machine-local clone paths are sanitized for portability — see
`demo_runs.sanitize_checkpoint`).

## Final clean Gemini demo

- `20260824T141045Z-e1cdb3ee`
- Google / `gemini-3.1-flash-lite`
- `harsh020/ecommerce-monolith`
- PASS, 1 refinement round
- generated AFTER the final material-grounding / invented-numeric-target /
  DecisionTopic-ADR-evidence hardening (started `2026-08-24T14:10:45Z`,
- after `60966c3`, `6211ed7`, `afe6162`, `033e2e4` — inferred from
  timestamp ordering, since the pipeline does not stamp the source commit
  into a run)
- suitable as the current final Gemini qualitative example

## Historical Claude demo runs

All three are real successful Opus runs, kept for qualitative inspection of
strong-model behavior. All were generated BEFORE the final
material-grounding / numeric-target safeguards landed, so their artifacts
may exhibit behavior the current code would constrain — that is exactly why
they are labeled historical and must not be read as final-code output.

### `20260823T171738Z-935663e4`
- Anthropic / `claude-opus-5` (Architect + Reviewer; Clarifier/ingestor on
  Gemini flash-lite per the A/B routing)
- `harsh020/ecommerce-monolith`
- successful: DONE, PASS, 0 refinement rounds
- generated before final material-grounding/numeric-target safeguards
- retained because it is a useful strong-model qualitative example

### `20260823T192225Z-5e6ac35c`
- Anthropic / `claude-opus-5`
- `harsh020/ecommerce-monolith`
- successful robustness run: DONE, PASS, 0 refinement rounds
- same historical caveat

### `20260824T084650Z-4447a662`
- Anthropic / `claude-opus-5`
- `ttulka/ddd-example-ecommerce` — the HOLDOUT repository, not the
  canonical monolith scenario
- successful: DONE, PASS, 2 refinement rounds (full refine-loop trace with
  best-design selection preserved in the bundled history)
- demonstrates behavior on a different modular-monolith/DDD repository
- same historical caveat where applicable

## Bundled-run summary table

| Run ID | Provider/model | Repository | Verdict | Refinements | Classification |
|---|---|---|---|---|---|
| `20260824T141045Z-e1cdb3ee` | Google / `gemini-3.1-flash-lite` | `harsh020/ecommerce-monolith` | PASS | 1 | Final clean Gemini demo |
| `20260823T171738Z-935663e4` | Anthropic / `claude-opus-5` | `harsh020/ecommerce-monolith` | PASS | 0 | Historical Claude monolith |
| `20260823T192225Z-5e6ac35c` | Anthropic / `claude-opus-5` | `harsh020/ecommerce-monolith` | PASS | 0 | Historical Claude robustness |
| `20260824T084650Z-4447a662` | Anthropic / `claude-opus-5` | `ttulka/ddd-example-ecommerce` | PASS | 2 | Historical Claude holdout |

### Deliberately NOT bundled

- `20260824T130359Z-b1ba2568` (Gemini) — pre-final-hardening run with the
  unsupported-numeric-target issue; historically useful, not a final demo.
- `20260823T155114Z-d1bc5c19` (Gemini) — pre-final-safeguards system
  behavior; same reason.

Both remain untouched in the local cache; they are simply not curated into
`demo_runs/`.

## PENDING — formal evaluation runs

Kept strictly separate from the curated demos above:

- **Fresh final Claude full-pipeline run** (current code, canonical
  scenario): **PENDING** — the historical Claude runs above must not be
  substituted for it.
- **Gemini one-shot baseline**: **PENDING** — see
  `eval/one_shots/MANIFEST.md`.
- **Claude one-shot baseline**: **PENDING** — same.

## One-shot baselines

Separate from pipeline runs entirely — see `eval/one_shots/MANIFEST.md`.
Both one-shot baselines are **PENDING**; none exist yet, so none are
bundled or fabricated. They are not pipeline runs and must never appear in
History / Switch run / Resume — see that file's own note on why.

## How seeding works, briefly

`ui.py` calls `demo_runs.seed_bundled_demo_runs()` on every app start. It
copies each id in `demo_runs.BUNDLED_RUN_IDS` from `demo_runs/<run_id>/`
into `persistence.runs_dir()` **only if that run id is not already
there** — so it costs nothing on the second and subsequent starts, and it
never overwrites a run you produced yourself (including, in the
astronomically unlikely case, one that happened to land on the same run
id). Seeding is disabled under the `AI_ARCHITECT_RUNS_DIR` test override,
exactly as designed. From that point on, the bundled run is an ordinary
entry in `.cache/runs/` and every existing code path
(`persistence.list_runs`, `persistence.load_state`, `run_history.py`, the
sidebar run switcher, the resume picker) already knows how to show it —
nothing about History, Switch run, or Resume needed to change.

Opening a bundled run only reads it (`load_state`/`load_history_run`);
nothing in the normal browsing flow writes a new checkpoint, so it stays
exactly as bundled no matter how many times it is viewed.
