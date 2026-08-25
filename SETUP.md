# Environment Setup

Everyone runs the **same Python and the same pinned packages** so the repo behaves
identically on every machine. Two files guarantee this:

- `.python-version` → Python **3.12** (the version the lock was resolved for)
- `requirements.txt` → fully pinned, cross-resolved dependency lock

The lock includes `pytest`, so once you are set up the test suite runs with
`python -m pytest -q`.

You need a **Gemini API key** before the app will run. Each developer uses their
**own** key (independent free-tier quotas) — get one free at
[Google AI Studio](https://aistudio.google.com/apikey).

---

## Setup (pip + venv — works everywhere)

From inside the `AI-Architect/` folder:

**1. Install Python 3.12** if you don't have it (python.org, pyenv, or `winget install Python.Python.3.12`).

**2. Create and activate a virtual environment**

Windows (PowerShell):
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**3. Install the locked dependencies**
```bash
pip install -r requirements.txt
```

**4. Add the API key**

Copy `.env.example` to `.env` and paste your own key:
```bash
cp .env.example .env        # Windows: copy .env.example .env
```
Then edit `.env` so it reads `GEMINI_API_KEY=<your key>`.
(`.env` is gitignored — never commit it.)

**Optional: University of Cologne KI Connect.** If you have KI Connect
access, you can redirect the Architect and/or Reviewer to it instead of
Gemini — see the commented block in `.env.example` and the "KI Connect A/B
routing" section of [pipeline/LLM_MODULE.md](pipeline/LLM_MODULE.md) for the
environment variables. Not required for a standard setup.

**5. Run**
```bash
python -m pipeline.run      # run the pipeline end to end
```
(The old single-agent prototype is still runnable with `streamlit run app.py`.)

---

## Faster alternative (uv)

If you have [uv](https://docs.astral.sh/uv/) installed, it handles the Python
version and the venv in one step:
```bash
uv venv --python 3.12
uv pip install -r requirements.txt
uv run python -m pipeline.run
```

---

## Adding a dependency later

Don't hand-edit `requirements.txt`. Add the top-level package to `requirements.in`,
then recompile the lock and commit both files:
```bash
uv pip compile requirements.in --python-version 3.12 -o requirements.txt
```
(No uv? `pip install pip-tools` then `pip-compile --python-version 3.12 requirements.in`.)

This keeps everyone on an identical, reproducible environment.
