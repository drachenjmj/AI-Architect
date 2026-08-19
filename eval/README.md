# Reviewer Evaluation Harness

`eval/` is a development tool. It is not an agent and the production pipeline
does not import it.

## What it measures

For every case, the harness compares:

- expected versus actual final verdict;
- all five deterministic rubric scores;
- all five qualitative Reviewer judgments.

This prevents a failure for the wrong reason from counting as success. Reports
show complete-case, verdict, rubric-field, and per-field agreement, plus explicit
false passes and false failures. Live results also record Reviewer-only token
usage and latency for each repeat.

## Seed cases

`scenarios.py` contains seven cross-domain seed cases covering:

- sound and symptom-patching brownfield shop designs;
- sound and non-compliant greenfield healthcare designs;
- a repository-mismatched warehouse design;
- a fabricated source reference; and
- a sound modular monolith that guards against microservice pattern bias.

These labels are **provisional**. They exercise the harness and support error
analysis, but they are not a human-validated benchmark.

## Run modes

Offline tests mock the LLM:

```bash
python -m pytest -q
```

A live run uses the production Reviewer and needs `GEMINI_API_KEY` in the local,
gitignored `.env`:

```bash
python -m eval.harness --model flash-lite --repeats 3 \
  --output eval/results/flash-lite.json
```

The command exits `1` if any verdict or rubric field disagrees with its label.
Repeated runs expose model instability; JSON output preserves the model, time,
label rationale, token use, latency, full report, reasons, issues, and all
comparisons.

## Evaluate real pipeline outputs

The harness accepts one or more labeled JSON case files:

```bash
python -m eval.harness --case eval/cases/case-001.json \
  --case eval/cases/case-002.json --repeats 3 --output eval/results/real.json
```

Each file contains:

```json
{
  "name": "case-001",
  "domain": "healthcare",
  "description": "Initial Architect output",
  "expected_pass": false,
  "expected_code_scores": {
    "all_artifacts_present": 2,
    "constraint_coverage": 2,
    "traceability": 2,
    "adr_presence": 2,
    "source_integrity": 2
  },
  "expected_judgments": {
    "repo_grounding": true,
    "flaw_detection": false,
    "adr_soundness": false,
    "best_practice_grounding": true,
    "refinement_readiness": true
  },
  "label_status": "human_reviewed",
  "label_rationale": "Reviewed independently by two team members.",
  "state": {}
}
```

`state` must be a serialized `ArchitectState`, ideally captured immediately
before the Reviewer node. Do not commit cases containing private repository or
client data.

## Interpretation

The seed set is a smoke/error-analysis set, not statistical validation. A
credible reliability claim additionally needs diverse real outputs, at least
two human labelers, recorded disagreements, repeated live runs, and a held-out
test set that was not used to revise the prompt.

`flaw_detection` is retained as a persisted field name for compatibility with
the UI and saved runs. Its current meaning is general: whether the design solves
the problem stated in that run's request, Context Record, and repository. It no
longer refers to a built-in shop flaw. `refinement_readiness` is recorded and
evaluated for judge analysis, but remains advisory in the production verdict.
