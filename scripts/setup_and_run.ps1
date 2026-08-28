# setup_and_run.ps1 - environment bootstrap for the one-click professor setup.
#
# No application/setup logic lives here - this script only makes sure Python
# 3.12 and a working virtualenv with the pinned dependencies exist, then
# hands off to the single source of truth:
#
#     python -m tools.setup_app
#
# All CLI options (--provider, --setup-only, --no-launch, --help, ...) are
# passed straight through via $args - this script never inspects or
# duplicates them. Mirrors scripts/rebuild_rag.ps1's bootstrap shape (same
# .venv-detection / pinned-install pattern) rather than a second,
# possibly-drifting implementation, and adds the Python-3.12
# detection/winget-install step that a fresh machine needs before any
# venv can exist at all.

# Left at the default ("Continue") deliberately: this script shells out to
# native python.exe repeatedly and checks $LASTEXITCODE itself. Windows
# PowerShell 5.1 turns a native command's stderr output into a terminating
# NativeCommandError when $ErrorActionPreference is "Stop", which would
# abort the script on a harmless warning rather than on an actual failure.
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"

function Test-Python312 {
    param([string]$Cmd, [string[]]$CmdArgs)
    try {
        $out = & $Cmd @CmdArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($LASTEXITCODE -eq 0 -and $out.Trim() -eq "3.12") {
            return $true
        }
    } catch {}
    return $false
}

function Find-Python312 {
    # Detection order per the project's setup contract: the "py" launcher's
    # -3.12 selector first (most reliable when multiple Pythons are
    # installed), then a bare "python"/"python3" whose OWN version happens
    # to already be 3.12 - never assumed, always verified.
    if (Get-Command "py" -ErrorAction SilentlyContinue) {
        if (Test-Python312 -Cmd "py" -CmdArgs @("-3.12")) {
            return [PSCustomObject]@{ Cmd = "py"; Args = @("-3.12") }
        }
    }
    if (Get-Command "python" -ErrorAction SilentlyContinue) {
        if (Test-Python312 -Cmd "python" -CmdArgs @()) {
            return [PSCustomObject]@{ Cmd = "python"; Args = @() }
        }
    }
    if (Get-Command "python3" -ErrorAction SilentlyContinue) {
        if (Test-Python312 -Cmd "python3" -CmdArgs @()) {
            return [PSCustomObject]@{ Cmd = "python3"; Args = @() }
        }
    }
    return $null
}

function Install-Python312ViaWinget {
    Write-Host "Installing Python 3.12 via winget (current user, no admin required) ..."
    # IMPORTANT: piped through Out-Host rather than left bare. A PowerShell
    # function implicitly captures EVERYTHING written to its output stream as
    # its return value - without this pipe, winget's own (unredirected)
    # progress text would be captured alongside `return $LASTEXITCODE` into
    # one mixed array at the call site, silently corrupting the caller's
    # `-ne 0` exit-code check (an array of non-"0" strings plus a trailing 0
    # reads as "not equal to 0" -> a successful install would misreport as
    # failed). Out-Host sends winget's text straight to the console instead,
    # so the function's output stream carries only the real exit code.
    & winget install --id Python.Python.3.12 --scope user --source winget `
        --accept-package-agreements --accept-source-agreements -e | Out-Host
    return $LASTEXITCODE
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "No .venv found - looking for Python 3.12 ..."
    $Found = Find-Python312

    if (-not $Found) {
        Write-Host "Python 3.12 was not found on this machine."
        $Winget = Get-Command "winget" -ErrorAction SilentlyContinue
        if ($Winget) {
            $Answer = Read-Host "Install Python 3.12 for your user account now via winget? [Y/n]"
            if ([string]::IsNullOrWhiteSpace($Answer) -or $Answer.Trim().ToLower() -eq "y") {
                $InstallExit = Install-Python312ViaWinget
                if ($InstallExit -ne 0) {
                    Write-Error "winget install failed (exit $InstallExit). Install Python 3.12 manually from https://www.python.org/downloads/ and re-run SETUP_AND_RUN.bat."
                    exit 1
                }
                # PATH may need a refresh after a fresh winget install; pull the
                # current machine+user PATH into this process before re-probing.
                $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + `
                            [System.Environment]::GetEnvironmentVariable("Path", "Machine")
                $Found = Find-Python312
            } else {
                Write-Error "Python 3.12 is required. Install it from https://www.python.org/downloads/ and re-run SETUP_AND_RUN.bat."
                exit 1
            }
        } else {
            Write-Error "Python 3.12 was not found and winget is not available on this machine. Install Python 3.12 from https://www.python.org/downloads/ and re-run SETUP_AND_RUN.bat."
            exit 1
        }
    }

    if (-not $Found) {
        Write-Error "Python 3.12 installation could not be verified in this session. Open a NEW terminal (so PATH refreshes) and re-run SETUP_AND_RUN.bat, or install Python 3.12 manually from https://www.python.org/downloads/."
        exit 1
    }

    Write-Host "Creating .venv with $($Found.Cmd) $($Found.Args -join ' ') ..."
    & $Found.Cmd @($Found.Args) -m venv (Join-Path $RepoRoot ".venv")
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create .venv."
        exit 1
    }
}

# Verify the pinned packages this app needs are importable; if not (fresh or
# incomplete venv), install requirements.txt. Same try-import-then-install
# shape as scripts/rebuild_rag.ps1 - see its comment for why stderr from
# this native python.exe call is deliberately left un-redirected.
& $VenvPython -c "import streamlit, langgraph, chromadb, google.genai, dotenv, requests, fpdf" | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing pinned dependencies from requirements.txt ..."
    & $VenvPython -m pip install -r (Join-Path $RepoRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) {
        Write-Error "pip install -r requirements.txt failed."
        exit 1
    }
}

& $VenvPython -m tools.setup_app @args
exit $LASTEXITCODE
