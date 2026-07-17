# Reviewer Evaluation Harness

This directory is a development tool. It is not an agent and is not imported by
the production pipeline or orchestrator.

A live harness run calls Gemini through the production Reviewer and therefore
requires a local `GEMINI_API_KEY`:

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
the Reviewer and never call Gemini. The complete offline suite can be run with:

```bash
python -m pytest -q
```

The labeled harness calibrates Reviewer behavior against crafted designs. It
does not replace the Week 3 requirement to evaluate a real end-to-end pipeline
output for use-case #1.
