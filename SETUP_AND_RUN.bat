@echo off
REM One-click setup + launch wrapper for a professor/evaluator on Windows.
REM All actual logic lives in scripts\setup_and_run.ps1 / tools\setup_app.py -
REM this file only locates the repo root, invokes PowerShell, and keeps the
REM window open on failure so the error is readable when launched by
REM double-click.
setlocal

cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_and_run.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo SETUP_AND_RUN failed with exit code %EXIT_CODE%.
    echo Press any key to close this window...
    pause >nul
)

exit /b %EXIT_CODE%
