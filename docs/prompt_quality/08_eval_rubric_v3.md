# Evaluation Rubric v3

Rubric v3 removes the use-case-specific answer from the production Reviewer.
The standard comes from each run's initial request, locked Context Record,
optional repository representation, retrieved knowledge, and generated
artifacts. It does not prescribe microservices, queues, or any other pattern.

## Code-owned checks

Every item retains a `0-2` diagnostic score and must equal `2` to pass.

| Item | Full-score rule |
|---|---|
| Artifact completeness | Context, Features, Blueprint, ADRs, and Components contain required decision information |
| Constraint coverage | Every structured applicable constraint has evidence in generated design artifacts |
| Structured traceability | All Feature, Blueprint, Component, and ADR links are present and resolve |
| ADR completeness | ADRs are uniquely numbered and contain context, rationale, alternatives, and positive and negative consequences |
| Source integrity | Every ADR citation resolves to supplied KB or repository evidence |

Python also raises a high-severity issue when a repository was requested but no
repository representation is available. Input constraints are never counted as
proof that the generated design addressed them.

## LLM-owned questions

One structured call answers five binary questions:

1. **Repository grounding:** Is a brownfield design consistent with supplied
   repository facts? A genuine greenfield request is recorded as not applicable.
2. **Problem/flaw resolution (`flaw_detection`):** Does the design solve the
   problem stated in this run and address any repository-evidenced cause?
3. **ADR soundness:** Are decisions, alternatives, rationales, and trade-offs
   internally sound and supported?
4. **Best-practice/evidence grounding:** Are major recommendations supported by
   supplied evidence or explicitly labelled assumptions and open risks?
5. **Refinement readiness:** Are remaining shortcomings actionable? This answer
   is recorded for judge evaluation but is advisory in the production verdict.

The `flaw_detection` name is retained for checkpoint and UI compatibility; its
meaning is now run-specific. Every yes requires a non-empty reason. Every no
should include a correction, with deterministic fallback text if omitted.

## Verdict rule

Python passes only when:

- all five code-owned scores equal `2`;
- all verdict-bearing LLM judgments pass, excluding explicitly not-applicable
  criteria and advisory `refinement_readiness`; and
- no high-severity issue remains.

There is no summed threshold, and the LLM never emits status or routing.

## Evaluation method

The harness compares the expected and actual final verdict, every deterministic
score, and every qualitative judgment. A correct failure for the wrong reason
is therefore still a disagreement. Results use explicit labels such as
`correct_pass`, `correct_fail`, `false_pass`, and `false_fail`.

The seven bundled cross-domain cases are provisional error-analysis fixtures,
not a reliability benchmark. A reliability claim requires independently
reviewed labels, real saved pipeline outputs, repeated live runs, and a held-out
set that was not used to revise the prompt.
