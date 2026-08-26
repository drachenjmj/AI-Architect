# AI-Architect

An agentic **AI solution architect**. Given a business use case and technical
constraints in natural language, it produces a Context Record, an Architecture
Blueprint (stakeholder + technical views), Architecture Decision Records (ADRs),
and Component Descriptions — grounded in a curated knowledge base via RAG.

## Quick Start on Windows

1. Download or clone the repository.
2. Double-click `SETUP_AND_RUN.bat`.
3. Choose Google Gemini (recommended, works outside the university) or
   University of Cologne KI Connect (requires University of Cologne access).
4. Enter the requested credential(s) when prompted — a Gemini API key for
   Gemini, or both a KI Connect key and a Gemini key for KI Connect (see
   below). Nothing is ever displayed or logged; both are saved locally in
   the git-ignored `.env`.
5. The AI-Architect UI opens automatically in your browser.

Gemini runtime automatically sets the Architect and Reviewer to `HIGH`
thinking. KI Connect redirects only the Architect and Reviewer generation
calls — the Clarifier and every RAG query-time embedding call always use
Gemini regardless of runtime provider, so **a Gemini API key is required
for both providers**, not only Gemini. The bundled `chroma_db/` knowledge
base is included and validated offline on every launch — a normal start
never rebuilds it and never spends an API call doing so. Rebuilding is
optional and always uses Gemini embeddings (`models/gemini-embedding-2`),
regardless of which runtime provider you chose, via its own one-click entry
point: `REBUILD_RAG.bat` (see [tools/rebuild_rag.py](tools/rebuild_rag.py)).
Manual setup remains available — see [SETUP.md](SETUP.md).

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

The human is in the loop at four points, and none of them adds a routing
decision an LLM makes: the Clarifier's questions, the Context Record approval
gate, the two feedback boxes on the finished run, and the sign-off that closes
it. A correction typed into the requirements box re-opens the record as a new
*version*; a directive typed into the design box goes to the Architect ranked
above the Reviewer's own instruction. Which box the text was typed into IS the
route, so no classifier stands between a person and what they asked for
(`pipeline/user_feedback.py`).

## Sign-off — what was decided, and what it was decided against

`DONE` means the pipeline stopped. `ACCEPTED` means a person took the design,
and only the second is a fact anyone downstream can act on, so the run records
both (`pipeline/sign_off.py`).

- A design the Reviewer did **not** pass may be accepted — that is the normal
  case, since most runs end on the refine budget with findings still open.
- Accepting against open findings records a **waiver**: every finding it covers,
  their severities, the Reviewer's verdict at the time, and an optional note.
  They are listed above the button, highest severity first, and a high-severity
  finding needs a second, deliberate confirmation. A clean sign-off records no
  waiver, and that absence is itself information.
- Anything the person typed and never re-ran is surfaced on the same screen and
  marked `abandoned` on confirmation — never left looking outstanding on a
  closed run, and never a reason to block the sign-off.
- When the Architect cannot build a directive as stated it builds the closest
  feasible variant and records an **objection** — what was asked, why it does
  not work, what it built instead — shown against the artifacts.

None of it ever enters an agent prompt. The Reviewer grades artifacts, not
intentions; a waived finding is not a solved one; and the Architect never reads
its own objections back. That is asserted in `test_sign_off.py`, not trusted.

## Reviewer quality gate

Reviewer rubric v3 separates checks by who can evaluate them reliably:

- Python checks artifact completeness, constraint coverage, structured
  traceability, ADR completeness, repository availability, and source integrity.
- One LLM call answers five atomic yes/no questions covering repository
  grounding, run-specific problem resolution, ADR soundness, evidence
  grounding, and refinement readiness.
- Python assembles the issues and owns the final pass/fail verdict. A design
  passes only when every deterministic check is full, every verdict-bearing
  judgment passes, and no high-severity issue remains. Refinement readiness is
  retained as an advisory measurement rather than a second veto.

See the [Reviewer prompt](docs/prompt_quality/04_reviewer_agent_prompt.md),
[report schema](docs/prompt_quality/06_reviewer_report_schema.json), and
[evaluation rubric v3](docs/prompt_quality/08_eval_rubric_v3.md) for the full
contract.

## Evaluation

The development-only [evaluation harness](eval/README.md) compares the final
verdict, every deterministic score, and every qualitative judgment against
labeled cases. It includes seven provisional cross-domain seed cases and can
load human-labeled saved pipeline states. `eval/replay_reviews.py` separately
replays the deterministic verdict rule over historical runs. Neither tool is
an agent or part of production routing.

```bash
python -m pytest -q          # offline suite; LLM calls are mocked
python -m eval.harness --repeats 3 --output eval/results/run.json
```

Current status: rubric v3 and the criterion-level harness are implemented and
covered by offline tests. The bundled labels remain provisional; human review
and repeated live evaluation on saved real outputs are still required before
making a reliability claim.

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
Rag Database/      curated knowledge base. Only two folders are ever indexed:
                   box1_patterns/ (general architecture: 2 AWS whitepaper PDFs +
                   curated Markdown) and box2_domain/ (8 curated e-commerce
                   Markdown sources). raw_source_archive/ is unindexed raw
                   upstream material, kept for audit only. Web-search grounding
                   (box 3) is the live fallback in architect.py when the KB has
                   no usable match. See docs/setup_report.md.
tools/rebuild_rag.py  canonical RAG rebuild pipeline (see REBUILD_RAG.bat below)
```

### KB maintenance workflow (PDF → Markdown → KB)

Adding a new source is fully standalone — repository code only, no external
ChatGPT/manual transformation, no LLM call:

```powershell
# 1. Prepare: deterministic PDF → clean Markdown (pypdf; recurring
#    headers/footers removed; page boundaries kept as '---' separators).
python tools/pdf_to_md.py path\to\source.pdf -o "Rag Database/box1_patterns/"   # box 1
python tools/pdf_to_md.py path\to\source.pdf -o "Rag Database/box2_domain/"     # box 2
```

The output **directory is the box selector** — nothing else. Box 2 keeps a
flat, domain-prefixed naming convention (`ecommerce_*.md`; a future domain
would be e.g. `healthcare_*.md`) — no code change needed. Scanned/image-only
PDFs (no extractable text) fail with a clear error; OCR is not supported.

Safety: an existing output `.md` is **never silently replaced** — most of
`Rag Database/box1_patterns/` and `box2_domain/` is hand-curated (condensed,
re-structured, citation-annotated excerpts, not raw conversions), so a rerun
refuses and exits non-zero if the target file already exists. Pass
`--overwrite` to replace one intentionally. The file is written atomically
(temp file + rename), so a failed run never leaves a partial `.md` behind.

Provenance: prepared Markdown is prefixed with a `Source PDF: <filename>`
header identifying the originating PDF, and keeps `source=<filename>`
metadata plus the original PDF page boundaries inline as `---` separators,
but the Markdown loader sets `page=0` — exact original page numbers are not
reconstructed as chunk metadata (direct PDF ingestion via PyPDFLoader still
has per-page metadata). Acceptable for the prototype.

2. **Review the prepared `.md` by hand** — nothing becomes permanent KB
   content unseen. Treat it as a first draft: most existing Box 1/2 sources
   are curated, condensed excerpts of the raw conversion, not the raw
   conversion itself.
3. **Rebuild the index** with the canonical rebuild pipeline:

   ```powershell
   .\REBUILD_RAG.bat                # Windows one-click
   python -m tools.rebuild_rag      # direct CLI
   ```

   The bundled `chroma_db/` is included in the repo and normally needs no
   rebuild — this step is only needed after adding/changing a source. It is
   a safe, staged rebuild (chunks at 1000/200, embeds with
   `models/gemini-embedding-2`, builds into a throwaway staging directory,
   validates the result, then swaps it in) — a failure at any point leaves
   `chroma_db/` exactly as it was. It requires a Gemini API key (from
   `.env` or an interactive prompt) because embeddings always go through
   Gemini, even on a KI Connect run — KI Connect is a runtime choice for the
   Architect/Reviewer LLM calls, not a substitute for the Gemini embedding
   endpoint used to build the KB. `python -m tools.rebuild_rag
   --validate-only` checks the installed index offline (no API key, no
   network). `notebooks/Rag_Setup.ipynb` remains as a read-only inspection
   notebook only. This is a separate, manual step — running `pdf_to_md.py`
   never touches `chroma_db/` on its own.

When Box 1/2 cannot answer a query, Box 3 performs a grounded web search and
the gap plus its candidate sources are logged. Recurring gaps and their
candidate sources are surfaced for human review — nothing from Box 3 is ever
auto-ingested:

```powershell
python tools/kb_gap_report.py      # grouped gaps + candidate Box-3 sources
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
