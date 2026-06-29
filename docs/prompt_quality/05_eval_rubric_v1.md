# Evaluation Rubric v1

Eight items, scored 0–2 each. Maximum 16.

- 0 = missing or wrong
- 1 = partially present
- 2 = good enough

`[code]` items are computed deterministically in Python and passed to the Reviewer.
`[LLM]` items are judged by the Reviewer.

| # | Item | Check | What "2" means |
|---|------|-------|----------------|
| 1 | All artifacts present | [code] | Context Record, Blueprint, ADRs, Component Descriptions all exist |
| 2 | Constraint coverage | [code] | Cloud, budget, scalability, compliance, and existing-system constraints are each addressed |
| 3 | Repo grounding | [LLM] | The design reflects the actual repo, not generic assumptions |
| 4 | Flaw detection | [LLM] | The ground-truth architectural flaw is identified and given a structural fix |
| 5 | Traceability | [code] | Requirements map to features, features to components, components and major choices to ADRs |
| 6 | ADR quality | [code] presence + [LLM] soundness | Each ADR has context, options, decision, trade-offs, and the rationale holds |
| 7 | Best-practice grounding | [LLM] | Recommendations are supported by retrieved patterns or KB entries |
| 8 | Refinement readiness | [LLM] | Reviewer issues are concrete enough for the Architect to act on |

## Thresholds

- 13–16 → pass
- 10–12 → pass with minor issues
- below 10 → fail, refine

## Feeding the loop

The Reviewer sets `requires_refinement = true` when the total is below 13 or any high-severity
issue exists. The orchestrator then routes `refinement_instruction` back to the Architect/Writer
node. The loop is bounded by the retry cap and the cost cap; on exhaustion the run returns the
last artifacts with the open issues attached.

## Definition of done (use-case #1)

A run passes when items 1, 2, and 5 score 2 (no missing fields, every constraint addressed,
traceability intact), item 4 scores 2 (flaw matches ground truth), and the run completes within
the cost cap.

