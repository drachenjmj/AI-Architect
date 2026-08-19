# Evaluation Rubric v2

> Historical specification. New reports use rubric v3, documented in
> `08_eval_rubric_v3.md`. The v2 fields remain loadable for stored-run replay.

Rubric v2 removes the numeric pass threshold. It preserves diagnostic scores
for checks that code can measure and makes every qualitative judgment an atomic
yes/no question. Python assembles the report and owns the final verdict.

## Code-owned checks

Each deterministic item retains a `0-2` diagnostic score. A passing design must
score `2` on every item.

| Item | Full-score rule |
|---|---|
| Artifact completeness | Context Record, Blueprint, ADRs, and Components exist and required fields are populated |
| Constraint coverage | Every constraint stated in the Context Record is addressed in the Blueprint, ADRs, or Components |
| Structured traceability | Blueprint, Component, and ADR references are present and every reference resolves |
| ADR presence | ADRs are present, well formed, uniquely numbered, and linked to Components |

The Context Record identifies which constraints apply. It is not evidence that
the generated design addresses them.

## LLM-owned questions

The Reviewer answers each question with `passed: true|false`, an evidence-backed
reason, and a concrete suggested fix when false.

1. **Repository grounding:** If repository context exists, is the design
   demonstrably consistent with it rather than generic? A documented greenfield
   run passes as not applicable.
2. **Flaw detection:** Does the submitted design itself identify and
   structurally address the ground-truth flaw rather than patch symptoms?
3. **ADR soundness:** Are the ADR rationales, alternatives, and trade-offs
   internally sound and supported by locked context or retrieved evidence?
4. **Best-practice grounding:** Are recommendations supported by retrieved
   knowledge or explicit source references?
5. **Refinement readiness:** Are any remaining shortcomings specific enough for
   the Architect to correct without guessing? If all preceding questions pass,
   this passes.

The LLM does not emit scores, status, routing, issues, or the final verdict.
Python converts failed judgments into report issues.

## Verdict rule

The report passes only when all three conditions hold:

- every code-owned item scores `2`;
- every LLM-owned question passes;
- no high-severity issue exists.

There is no `pass_with_minor_issues` status and no summed threshold. Any failed
condition produces `overall_status = "fail"`, `requires_refinement = true`, and
an actionable refinement instruction assembled in code.

## Evaluation harness labels

- **Positive label:** a sound design expected to pass.
- **Negative label:** a seeded-flaw design expected to fail.
- **TP:** sound design passed.
- **TN:** flawed design failed.
- **FP:** flawed design passed.
- **FN:** sound design failed.

The development-only harness reports agreement and every disagreement with the
per-question Reviewer reasons. It is not part of the pipeline or state machine.
