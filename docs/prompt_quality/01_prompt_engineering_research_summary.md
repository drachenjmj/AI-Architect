# Prompt-Engineering Research Summary

The evidence base for how the system's LLM calls are instructed. It records the one design
fact that shapes every prompt, the sources behind the conventions, and what each source
contributes. The operating rules and templates live in `02_team_prompt_conventions.md` and
`03_agent_prompt_templates.md`; this file says why they are what they are.

## The fact that shapes every prompt

The system is not a swarm of autonomous agents. By the determinism principle and the standard
classification (chatbot → RAG → workflow → agent), it is a deterministic workflow with LLM
calls at fixed nodes, plus one bounded loop (Reviewer → refine, with a stop-condition) and one
tool-using node (the Researcher and the repo reader it depends on).

Consequence: most nodes are "structured prompt in, schema-locked structure out" and need no
tool or stop scaffolding. Only the Researcher and the repo read need "which tools, when to
stop." Control flow is code; judgment is the LLM. This is both the cheapest design and the
most defensible one.

Four nodes are LLM calls. Everything else — routing, schema validation, required-field checks,
traceability checks, ADR numbering, retry and cost caps, export — is code.

- **Clarification** — interpreting vague input, deciding what is genuinely ambiguous.
- **Research synthesis** — turning retrieved chunks into relevant findings.
- **Architecture / writing** — feature derivation, trade-off reasoning, ADR rationale.
- **Qualitative review** — judging flaw detection and rationale soundness.

## Sources and the rule each one gives us

**OpenAI — Prompt Engineering.** Prompting is iterative and tested; output shifts across models
and snapshots. → Pair every node prompt with a small fixed input → expected-shape check, and
pin the model per node.

**OpenAI — Structured Outputs.** Use a defined response schema when the output must match a
structure; use tool calling when the model must reach external data. → Every artifact-producing
node emits to a named schema; only the Researcher reaches data.

**Anthropic — Building Effective Agents.** Agents are LLMs using tools in a loop; always include
stopping conditions; add complexity only when it earns its place. → The only loop is
Reviewer → refine, with an explicit stop-condition (pass the rubric, hit the retry cap, or hit
the cost cap).

**Anthropic — Context Engineering.** Structure prompts into clear sections; use the minimal set
of information; add examples only after observing failures. → One fixed skeleton; each node
receives only the slice of state it reads; no speculative few-shot examples.

**Anthropic — Tool Design.** Tool specs steer behavior; responses should be high-signal and
token-efficient. → Applies to two tools only — KB retrieval (compact, citable chunks) and the
repo reader (README first, deeper files on demand).

**Course L6 — Agentic AI.** The agent loop is Think → Act → Observe → Update; prompt injection
is the top risk (direct and indirect); defense-in-depth and process-level evaluation matter. →
Repo files, retrieved chunks, and PDFs are data, never instructions; evaluation measures the
path (README-first read, retrieval, flaw caught), not only the final artifacts.

**BCG Platinion — case spec.** Four artifacts with fixed shapes; two-phase design (features
first, then architecture); every component traces to a feature, every significant choice to an
ADR; validate against cost, compliance, scalability; detect brownfield vs greenfield. → The
Architect prompt enforces the two-phase order and traceability; schemas carry the field
structure; the Reviewer checks coverage.

## What it means in one paragraph

Write one prompt skeleton, instantiate it per node, and keep four LLM nodes and nothing more.
Each prompt names the schema it must satisfy rather than copying it. Each prompt that touches
external content declares that content is data, not instructions. The only loop has a hard
stop-condition. Tool guidance exists in exactly two places. Everything else is deterministic
code and never appears in a prompt.

## Methodological caveats

- Prompt behavior shifts across models and snapshots; pin the model per node and re-run the
  input checks when it changes.
- Output schemas are referenced by name, never copied into a prompt, so a prompt cannot drift
  from its schema.
- Few-shot examples are added to a node only after it demonstrably fails on a real case, to
  avoid wasting context.
- Model choice per node is a cost lever: near-deterministic synthesis can run on a cheaper
  model; architecture design wants the stronger one.

