# Activate the dedicated venv and launch the PyQt UI.
# Windows / PowerShell.
#
# Use this instead of `python ui\app.py` so you can't accidentally
# run against your global / Anaconda Python (which has been the
# source of multiple numpy/scipy version conflicts).
#
# First time? Run .\scripts\setup_env.ps1 first.

$ErrorActionPreference = "Stop"

$EnvDir = ".venv"

if (-not (Test-Path $EnvDir)) {
    Write-Host "ERROR: $EnvDir\ doesn't exist." -ForegroundColor Red
    Write-Host "Run .\scripts\setup_env.ps1 first to create the dedicated env."
    exit 1
}

& "$EnvDir\Scripts\Activate.ps1"

# Defense-in-depth: enable Python's fault handler so any native
# crash writes a C stack trace instead of dying silently.
python -X faulthandler ui\app.py @args
