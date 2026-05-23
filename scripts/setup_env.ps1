# Set up the dedicated Python venv for the detector PyQt + Streamlit
# UIs. Windows / PowerShell version. macOS / Linux users: setup_env.sh.
#
# Why a dedicated env at all: sharing Anaconda's base environment
# with other tools means any other project's pip install / conda
# update can silently break this project's pinned numpy / scipy
# versions. We've been bitten by exactly that more than once. A venv
# isolates the project entirely.
#
# Run once from the repo root:
#     .\scripts\setup_env.ps1
#
# Activate later (every new shell):
#     .\.venv\Scripts\Activate.ps1
#
# Launch the UI:
#     .\.venv\Scripts\Activate.ps1; python ui\app.py

$ErrorActionPreference = "Stop"

$EnvDir = ".venv"
$PyBin = if ($env:PYTHON) { $env:PYTHON } else { "python" }

# ----- Pre-flight -----------------------------------------------------

if (-not (Get-Command $PyBin -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: $PyBin not found on PATH." -ForegroundColor Red
    Write-Host ""
    Write-Host "Install Python 3.12 from https://www.python.org/downloads/"
    Write-Host "Make sure 'Add to PATH' is checked during install."
    Write-Host ""
    Write-Host "If you have Python 3.12 under a different name, set"
    Write-Host "the PYTHON env var first:"
    Write-Host "    `$env:PYTHON = 'C:\Path\To\python.exe'; .\scripts\setup_env.ps1"
    exit 1
}

$pyVerLine = & $PyBin -c "import sys; print('{}.{}'.format(*sys.version_info[:2]))"
if ($pyVerLine -ne "3.12") {
    Write-Host "WARNING: $PyBin is Python $pyVerLine (project requires 3.12)." -ForegroundColor Yellow
    $ans = Read-Host "Continue anyway? [y/N]"
    if ($ans -ne "y" -and $ans -ne "Y") {
        exit 0
    }
}

# ----- Existing env handling -----------------------------------------

if (Test-Path $EnvDir) {
    Write-Host "Existing $EnvDir\ found."
    $ans = Read-Host "Re-create from scratch? [Y/n]"
    if ($ans -eq "n" -or $ans -eq "N") {
        Write-Host "Reusing existing env. Run with 'Remove-Item -Recurse -Force $EnvDir'"
        Write-Host "first if you want a clean slate."
        exit 0
    }
    Remove-Item -Recurse -Force $EnvDir
}

# ----- Create + install ----------------------------------------------

Write-Host ""
Write-Host "Creating venv at $EnvDir\ using $PyBin ..."
& $PyBin -m venv $EnvDir

# Activate (sets $env:VIRTUAL_ENV + prepends to PATH for THIS session).
& "$EnvDir\Scripts\Activate.ps1"

Write-Host ""
Write-Host "Upgrading pip ..."
python -m pip install --upgrade pip

Write-Host ""
Write-Host "Installing detector-pyqt + transitive deps (pyproject.toml) ..."
python -m pip install -e .

# detector-core lives inside this repo as a submodule and is imported
# via sys.path injection in conftest.py / ui/app.py. Install editable
# so `from detector import ...` works from anywhere in the env.
if (Test-Path "detector-core") {
    Write-Host ""
    Write-Host "Installing detector-core (editable) ..."
    python -m pip install -e detector-core
}

# Streamlit isn't in detector-pyqt's pyproject (PyQt is the headline
# UI), but the same env should cover both.
Write-Host ""
Write-Host "Installing optional Streamlit UI deps ..."
python -m pip install "streamlit>=1.36" "plotly>=5.20" "plotly-resampler>=0.11" "streamlit-plotly-events>=0.0.6"

# TDT block reader for preprocessing (optional).
Write-Host ""
Write-Host "Installing preprocessing deps (TDT block reader) ..."
python -m pip install "tdt>=0.7"

# Confirm everything imports cleanly + verify numpy/scipy work.
Write-Host ""
Write-Host "Verifying installation ..."
python -c @"
import sys
print('python', sys.executable)
print('python version', sys.version.split()[0])
import numpy, scipy, pandas, sklearn, h5py, lightgbm, matplotlib
print('numpy', numpy.__version__)
print('scipy', scipy.__version__)
print('pandas', pandas.__version__)
print('lightgbm', lightgbm.__version__)
print('h5py', h5py.__version__)
print('matplotlib', matplotlib.__version__)
from detector import review  # noqa: F401
print('detector.review import OK')
import PySide6.QtCore
print('PySide6 Qt', PySide6.QtCore.qVersion())
A = numpy.array([[1., 2.], [3., 4.]])
print('numpy.linalg.solve OK:', numpy.linalg.solve(A, numpy.array([1., 2.])))
from scipy import signal
sos = signal.butter(4, [10, 100], btype='band', fs=1000, output='sos')
x = numpy.random.randn(10000).astype(numpy.float32)
y = signal.sosfiltfilt(sos, x)
print('scipy.signal.sosfiltfilt OK, output shape:', y.shape)
print()
print('All imports + functional checks passed.')
"@

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Activate this env in every new shell:"
Write-Host "    .\$EnvDir\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Then launch the PyQt UI:"
Write-Host "    python ui\app.py"
Write-Host ""
Write-Host "Or the Streamlit UI:"
Write-Host "    streamlit run ui\streamlit_app.py"
Write-Host ""
Write-Host "Deactivate when done:"
Write-Host "    deactivate"
