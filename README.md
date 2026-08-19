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

The human is in the loop at three points, and none of them adds a routing
decision an LLM makes: the Clarifier's questions, the Context Record approval
gate, and — on the finished run — two feedback boxes. A correction typed into
the requirements box re-opens the record as a new *version*; a directive typed
into the design box goes to the Architect ranked above the Reviewer's own
instruction. Which box the text was typed into IS the route, so no classifier
stands between a person and what they asked for
(`pipeline/user_feedback.py`).

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
  refine_gate.py   cost-cap gate: loop-vs-stop, best-so-far, human-round budget
  user_feedback.py caller-side write path for feedback on a finished run
  persistence.py   state on disk: a checkpoint per transition, resume any run
  run.py           CLI entry point: answer loop + full plain-text report
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
python -m pipeline.run        # drive one run from the terminal (see below)
```

Each developer uses their **own** Gemini API key (`GEMINI_API_KEY` in the
gitignored `.env`). All agents call the LLM through `pipeline/llm.py`; see
**[pipeline/LLM_MODULE.md](pipeline/LLM_MODULE.md)** for the interface.

## Running from the command line

`python -m pipeline.run` drives one complete run and prints the live step
trace followed by every artifact as plain text: Context Record, Features,
Blueprint, ADRs, Components, the review report, the retrieved knowledge, the
token/cost breakdown, and the run id with its checkpoint directory.

The pipeline pauses mid-run by returning with `stage == AWAITING_HUMAN`, and
`state.pending_decision` says what is owed. The command resolves both kinds:

* **Clarifying questions** — it asks each one, merges the answers into the
  state, and resumes, up to three clarification rounds. Answering interactively
  is the default; `--answers` makes a run reproducible without a keyboard.
* **The context lock** (`--approve-context`, off by default) — once the Context
  Record is frozen, the run stops and shows it for approval *before* any
  research, design or review token is spent on it. Approve it, edit a field,
  strike an assumption you do not accept, ask the clarifier to recommend a value
  you do not know, or ask a read-only question about it.

```bash
python -m pipeline.run                                   # default example prompt
python -m pipeline.run "Design an order pipeline for 10k rps."
python -m pipeline.run --repo-url https://github.com/pallets/flask
python -m pipeline.run --answers answers.json --no-input  # unattended
python -m pipeline.run --approve-context                  # approve the record first
```

| Flag | What it does |
|------|--------------|
| `PROMPT` | The system to architect. Defaults to the built-in `EXAMPLE_PROMPT`, whose repo URL is fictional — the ingestor records the failed clone and the run continues without repository insight. |
| `--repo-url URL` | Existing codebase to re-architect, passed to `new_run()`. Omit for greenfield. |
| `--answers FILE` | JSON answers to the Clarifier's questions. An **object** maps question text to answer (exact match); an **array** is consumed positionally in the order asked. A question with no match is asked on stdin, or fails the run under `--no-input`. |
| `--no-input` | Never prompt — fail instead. For unattended runs. |
| `--approve-context` | Pause at the context lock and show the frozen Context Record for approval before design begins. Needs a terminal, so it cannot be combined with `--no-input`: a gate with nobody to work it would just rubber-stamp its own ground truth. Off by default, so unattended runs behave exactly as before. |
| `--max-steps N` | Graph step cap per invocation (default 20). |

Exit codes: `0` the run reached DONE, `1` it reached FAILED, `2` it could not be
driven to a terminal stage (an unanswerable question, the round cap, or a pause
that carried no questions). `python -m pipeline.run --help` is the full
reference.

The Streamlit UI (`streamlit run ui.py`) drives the same contract with a form
instead of a terminal.

## Conventions

- One writer per state field; changing a field's name or type needs team agreement.
- The orchestrator stays LLM-free; reasoning belongs in an agent's `_act`.
- Every LLM call goes through `llm_call`; never call the SDK directly in an agent.
- Add dependencies via `requirements.in`, then recompile the lock (see SETUP.md).
