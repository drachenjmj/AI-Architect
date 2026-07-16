# Reviewer Evaluation Harness

This directory is a development tool. It is not an agent and is not imported by
the production pipeline or orchestrator.

After the repository's `repo_ingestor` import is repaired and a local
`GEMINI_API_KEY` is configured, run:

```bash
python -m eval.harness
```

The harness sends two labeled use-case #1 designs through the real Reviewer:

- a formally complete design with the seeded architectural flaw, expected to fail;
- a queue-decoupled sound design, expected to pass.

It prints agreement, true-positive and true-negative rates, the confusion
matrix, and all per-question reasons for disagreements. The command exits with
status `1` when any label and verdict disagree.

Offline metric behavior is covered by `test_eval_harness.py`; those tests mock
the Reviewer and never call Gemini.
