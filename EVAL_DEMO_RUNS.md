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

**They are NOT the quantitative evaluation sample.** Formal evaluation
(the Gemini/Claude comparison, refinement metrics, the agreed 2×2 design)
must use pre-specified fresh runs generated for that purpose, not these
bundled convenience artifacts. Nothing here is scored or editorialized —
each run is shown exactly as the pipeline produced it.

## Bundled full-pipeline runs

| Run ID | Model / provider | Repository | Scenario | Final verdict | Refinement count | Purpose |
|---|---|---|---|---|---|---|
| `20260824T141045Z-e1cdb3ee` | Google / `gemini-3.1-flash-lite` | `harsh020/ecommerce-monolith` | Modernize an e-commerce monolith: scalability/maintainability under peak traffic, incremental evolution without disrupting the running business. | PASS | 1 | Gemini side of the Gemini/Claude side-by-side inspection demo. |

Code freshness: this run started at `2026-08-24T14:10:45Z`, after every
material-grounding / numeric-target hardening commit on
`experiment/claude-opus5` (`60966c3`, `6211ed7`, `afe6162`, `033e2e4` —
all landed before `2026-08-24T14:03:10Z`). The pipeline does not stamp a
run with the exact source commit it ran under, so this is inferred from
timestamp ordering rather than read directly off the artifact — stated
here so the inference is auditable rather than assumed silently.

### PENDING — final Claude run

**No fresh Claude full-pipeline run exists yet and none is bundled.**

The most recent locally available Claude run
(`20260824T084650Z-4447a662`, PASS, 2 refinements) predates the
material-grounding hardening (`afe6162` @ `13:41:26Z`, `033e2e4` @
`14:03:10Z` — the run started at `08:46:50Z`, hours earlier) **and** used a
different repository (`ttulka/ddd-example-ecommerce`, not the canonical
`harsh020/ecommerce-monolith` scenario the Gemini run above uses). Per
this task's own instruction, a stale or off-scenario run must not be
substituted for the missing final one.

**To fill this slot:** run the pipeline once, live, against
`harsh020/ecommerce-monolith` with the same prompt as the Gemini run
above, routed to the strongest chosen Claude/Opus configuration, on the
current `experiment/claude-opus5` HEAD or later. Once it reaches `DONE`
with a passing verdict, sanitize it with `demo_runs.sanitize_checkpoint`
(inspect the result for anything the sanitizer doesn't already cover — see
its docstring), commit the sanitized checkpoint under
`demo_runs/<run_id>/`, add its id to `demo_runs.BUNDLED_RUN_IDS`, and add
its row to the table above.

## One-shot baselines

Separate from pipeline runs entirely — see `eval/one_shots/MANIFEST.md`.
Both the Gemini and Claude one-shot baselines are **PENDING**; none exist
yet, so none are bundled or fabricated. They are not pipeline runs and
must never appear in History / Switch run / Resume — see that file's own
note on why.

## How seeding works, briefly

`ui.py` calls `demo_runs.seed_bundled_demo_runs()` on every app start. It
copies each id in `demo_runs.BUNDLED_RUN_IDS` from `demo_runs/<run_id>/`
into `persistence.runs_dir()` **only if that run id is not already
there** — so it costs nothing on the second and subsequent starts, and it
never overwrites a run you produced yourself (including, in the
astronomically unlikely case, one that happened to land on the same run
id). From that point on, the bundled run is an ordinary entry in
`.cache/runs/` and every existing code path (`persistence.list_runs`,
`persistence.load_state`, `run_history.py`, the sidebar run switcher, the
resume picker) already knows how to show it — nothing about History,
Switch run, or Resume needed to change.

Opening a bundled run only reads it (`load_state`/`load_history_run`);
nothing in the normal browsing flow writes a new checkpoint, so it stays
exactly as bundled no matter how many times it is viewed.
