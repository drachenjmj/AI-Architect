# Environment Setup

Everyone runs the **same Python and the same pinned packages** so the repo behaves
identically on every machine. Two files guarantee this:

- `.python-version` → Python **3.12** (the version the lock was resolved for)
- `requirements.txt` → fully pinned, cross-resolved dependency lock

You need a **Gemini API key** (shared by the team) before the app will run.

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

Copy `.env.example` to `.env` and paste the shared key:
```bash
cp .env.example .env        # Windows: copy .env.example .env
```
Then edit `.env` so it reads `GEMINI_API_KEY=<the key>`.

**5. Run**
```bash
streamlit run app.py
```

---

## Faster alternative (uv)

If you have [uv](https://docs.astral.sh/uv/) installed, it handles the Python
version and the venv in one step:
```bash
uv venv --python 3.12
uv pip install -r requirements.txt
uv run streamlit run app.py
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
