# AI-Architect

An agentic **AI solution architect**. Given a business use case and technical
constraints in natural language, it produces a Context Record, an Architecture
Blueprint (stakeholder + technical views), Architecture Decision Records (ADRs),
and Component Descriptions — grounded in a curated knowledge base via RAG.

## How it works

A single shared state object travels through a pipeline of specialised agents.
Agents never call each other; they read and write fields on the state
(blackboard pattern). A deterministic orchestrator — plain code, no LLM — routes
the state between agents based on its `stage`. LLM reasoning lives only inside
the agents. This keeps the system auditable and deterministic wherever it can be.

- **Orchestrator** — deterministic router; runs the gate/loop/retry logic.
- **Clarifier** — turns the raw request into a structured Context Record, asking
  follow-up questions when information is missing.
- **Researcher** — retrieves relevant knowledge from the RAG knowledge base.
- **Architect** — derives features and writes the Blueprint, ADRs, and Components.
- **Reviewer** — checks quality and can send work back for refinement.

## Reviewer quality gate

Reviewer rubric v2 separates checks by who can evaluate them reliably:

- Python checks artifact completeness, constraint coverage, structured
  traceability, and ADR presence.
- One LLM call answers five atomic yes/no questions covering repository
  grounding, flaw detection, ADR soundness, best-practice grounding, and
  refinement readiness.
- Python assembles the issues and owns the final pass/fail verdict. A design
  passes only when every deterministic check is full, every qualitative
  judgment passes, and no high-severity issue remains.

See the [Reviewer prompt](docs/prompt_quality/04_reviewer_agent_prompt.md),
[report schema](docs/prompt_quality/06_reviewer_report_schema.json), and
[evaluation rubric v2](docs/prompt_quality/07_eval_rubric_v2.md) for the full
contract.

## Evaluation

The development-only [evaluation harness](eval/README.md) sends two labeled
use-case #1 designs through the production Reviewer: a seeded-flaw design that
must fail and a sound design that must pass. It reports agreement, true-positive
and true-negative rates, the confusion matrix, and every disagreement. The
`eval/` package is not an agent and is never imported by the production
pipeline.

```bash
python -m pytest -q          # offline suite; LLM calls are mocked
python -m eval.harness       # real Reviewer call; requires GEMINI_API_KEY
```

Current status: rubric v2 and the labeled harness are implemented and covered
by offline tests. Applying the rubric to a real end-to-end pipeline output for
use-case #1 remains the Week 3 evaluation exit gate.

## Layout

```
pipeline/
  state.py         shared ArchitectState — the contract every agent uses
  orchestrator.py  deterministic router (stage -> agent)
  llm.py           single door for all LLM calls (see LLM_MODULE.md)
  review_checks.py deterministic Reviewer checks
  run.py           entry point / end-to-end trace
  agents/          base class + one file per agent
eval/              development-only labeled evaluation harness
architect.py       single-agent prototype (reference for prompt + RAG wiring)
Rag Database/      knowledge base assets
```

## Quick start

Full environment instructions are in **[SETUP.md](SETUP.md)** (Python 3.12,
pinned `requirements.txt`, `.env`). In short, from the `AI-Architect/` folder:

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # Windows: py -3.12 -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env          # then paste your own Gemini API key into .env
python -m pipeline.run        # run the pipeline end to end
```

Each developer uses their **own** Gemini API key (`GEMINI_API_KEY` in the
gitignored `.env`). All agents call the LLM through `pipeline/llm.py`; see
**[pipeline/LLM_MODULE.md](pipeline/LLM_MODULE.md)** for the interface.

## Conventions

- One writer per state field; changing a field's name or type needs team agreement.
- The orchestrator stays LLM-free; reasoning belongs in an agent's `_act`.
- Every LLM call goes through `llm_call`; never call the SDK directly in an agent.
- Add dependencies via `requirements.in`, then recompile the lock (see SETUP.md).
