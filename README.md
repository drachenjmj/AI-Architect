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

## Layout

```
pipeline/
  state.py         shared ArchitectState — the contract every agent uses
  orchestrator.py  deterministic router (stage -> agent)
  llm.py           single door for all LLM calls (see LLM_MODULE.md)
  run.py           entry point / end-to-end trace
  agents/          base class + one file per agent
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
