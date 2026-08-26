@echo off
REM One-click wrapper for the canonical RAG rebuild pipeline.
REM All actual logic lives in scripts\rebuild_rag.ps1 / tools\rebuild_rag.py -
REM this file only locates the repo root, invokes PowerShell, and keeps the
REM window open so the result is readable when launched by double-click.
setlocal

cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\rebuild_rag.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
    echo REBUILD_RAG finished successfully.
) else (
    echo REBUILD_RAG failed with exit code %EXIT_CODE%.
)
echo Press any key to close this window...
pause >nul

exit /b %EXIT_CODE%
