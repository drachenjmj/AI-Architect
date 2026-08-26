# rebuild_rag.ps1 - environment bootstrap for the canonical RAG rebuild.
#
# No RAG business logic lives here - this script only makes sure a working
# Python 3.12 virtualenv with the pinned dependencies exists, then hands off
# to the single source of truth:
#
#     python -m tools.rebuild_rag
#
# All CLI options (--validate-only, --yes, --help, ...) are passed straight
# through via $args - this script never inspects or duplicates them.

# Left at the default ("Continue") deliberately: this script shells out to
# native python.exe repeatedly and checks $LASTEXITCODE itself. Windows
# PowerShell 5.1 turns a native command's stderr output into a terminating
# NativeCommandError when $ErrorActionPreference is "Stop", which would
# abort the script on a harmless warning rather than on an actual failure.
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "No .venv found - looking for Python 3.12 to create one..."

    $Candidate = $null
    $PyLauncher = Get-Command "py" -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        & py -3.12 -c "import sys" | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $Candidate = "py -3.12"
        }
    }
    if (-not $Candidate) {
        $Python312 = Get-Command "python3.12" -ErrorAction SilentlyContinue
        if ($Python312) {
            $Candidate = $Python312.Source
        }
    }
    if (-not $Candidate) {
        $PythonCmd = Get-Command "python" -ErrorAction SilentlyContinue
        if ($PythonCmd) {
            $version = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
            if ($LASTEXITCODE -eq 0 -and $version.Trim() -eq "3.12") {
                $Candidate = $PythonCmd.Source
            }
        }
    }

    if (-not $Candidate) {
        Write-Error "Python 3.12 was not found (tried 'py -3.12', 'python3.12', 'python'). Install Python 3.12 and add it to PATH, then re-run REBUILD_RAG.bat."
        exit 1
    }

    Write-Host "Creating .venv with $Candidate ..."
    if ($Candidate -eq "py -3.12") {
        & py -3.12 -m venv (Join-Path $RepoRoot ".venv")
    } else {
        & $Candidate -m venv (Join-Path $RepoRoot ".venv")
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create .venv."
        exit 1
    }
}

# Verify the pinned packages this rebuild needs are importable; if not
# (fresh or incomplete venv), install requirements.txt.
& $VenvPython -c "import chromadb, langchain_community, langchain_google_genai, langchain_text_splitters, pypdf, dotenv" | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing pinned dependencies from requirements.txt ..."
    & $VenvPython -m pip install -r (Join-Path $RepoRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) {
        Write-Error "pip install -r requirements.txt failed."
        exit 1
    }
}

& $VenvPython -m tools.rebuild_rag @args
exit $LASTEXITCODE
