#!/usr/bin/env bash
# Set up the dedicated Python venv for the detector PyQt + Streamlit
# UIs. macOS / Linux version. Windows users: scripts/setup_env.ps1.
#
# Why a dedicated env at all: sharing Anaconda's base environment
# with other tools means any other project's `pip install` /
# `conda update` can silently break this project's pinned numpy /
# scipy versions. We've been bitten by exactly that more than once
# (Bus error inside scipy.signal on Mac; silent process kill on
# Windows). A venv isolates the project entirely.
#
# Run once from the repo root:
#     bash scripts/setup_env.sh
#
# Activate later (every new shell):
#     source .venv/bin/activate
#
# Launch the UI:
#     source .venv/bin/activate && python ui/app.py

set -e

ENV_DIR=".venv"
PY_BIN="${PYTHON:-python3.12}"

# ----- Pre-flight ------------------------------------------------------

if ! command -v "$PY_BIN" >/dev/null 2>&1; then
    echo "ERROR: $PY_BIN not found on PATH."
    echo
    echo "Install Python 3.12 from https://www.python.org/downloads/"
    echo "(or with Homebrew: brew install python@3.12)"
    echo
    echo "If you have Python 3.12 under a different name, set the"
    echo "PYTHON env var first:"
    echo "    PYTHON=/path/to/python3.12 bash scripts/setup_env.sh"
    exit 1
fi

PY_VERSION="$("$PY_BIN" -c 'import sys; print("{}.{}".format(*sys.version_info[:2]))')"
if [ "$PY_VERSION" != "3.12" ]; then
    echo "WARNING: $PY_BIN is Python $PY_VERSION (project requires 3.12)."
    echo "Continue anyway? [y/N]"
    read -r ans
    if [ "$ans" != "y" ] && [ "$ans" != "Y" ]; then
        exit 0
    fi
fi

# ----- Existing env handling ------------------------------------------

if [ -d "$ENV_DIR" ]; then
    echo "Existing $ENV_DIR/ found."
    echo "Re-create from scratch? [Y/n]"
    read -r ans
    if [ "$ans" = "n" ] || [ "$ans" = "N" ]; then
        echo "Reusing existing env. Run with 'rm -rf $ENV_DIR' first if"
        echo "you want a clean slate."
        exit 0
    fi
    rm -rf "$ENV_DIR"
fi

# ----- Create + install -----------------------------------------------

echo
echo "Creating venv at $ENV_DIR/ using $PY_BIN ..."
"$PY_BIN" -m venv "$ENV_DIR"

# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

echo
echo "Upgrading pip ..."
python -m pip install --upgrade pip

echo
echo "Installing detector-pyqt + transitive deps (pyproject.toml) ..."
python -m pip install -e .

# detector-core lives inside this repo as a submodule and is imported
# via sys.path injection in conftest.py / ui/app.py. We install it as
# editable too so `from detector import ...` works from anywhere in
# the env, not just from this repo's cwd.
if [ -d "detector-core" ]; then
    echo
    echo "Installing detector-core (editable) ..."
    python -m pip install -e detector-core
fi

# Streamlit isn't in detector-pyqt's pyproject (PyQt is the headline
# UI), but Andrea uses both UIs interchangeably. Install it here so
# this one env covers both.
echo
echo "Installing optional Streamlit UI deps ..."
python -m pip install "streamlit>=1.36" "plotly>=5.20" "plotly-resampler>=0.11" "streamlit-plotly-events>=0.0.6"

# TDT block reader for preprocessing (optional).
echo
echo "Installing preprocessing deps (TDT block reader) ..."
python -m pip install "tdt>=0.7"

# Confirm everything imports cleanly + verify numpy/scipy work.
echo
echo "Verifying installation ..."
python -c "
import sys
print('python', sys.executable)
print('python version', sys.version.split()[0])
import numpy, scipy, pandas, sklearn, h5py, lightgbm
print('numpy', numpy.__version__)
print('scipy', scipy.__version__)
print('pandas', pandas.__version__)
print('lightgbm', lightgbm.__version__)
print('h5py', h5py.__version__)
import PySide6.QtCore
print('PySide6 Qt', PySide6.QtCore.qVersion())
# Functional check: the exact code path that has crashed before.
A = numpy.array([[1., 2.], [3., 4.]])
print('numpy.linalg.solve OK:', numpy.linalg.solve(A, numpy.array([1., 2.])))
from scipy import signal
sos = signal.butter(4, [10, 100], btype='band', fs=1000, output='sos')
x = numpy.random.randn(10000).astype(numpy.float32)
y = signal.sosfiltfilt(sos, x)
print('scipy.signal.sosfiltfilt OK, output shape:', y.shape)
print()
print('All imports + functional checks passed.')
"

echo
echo "============================================================"
echo "Setup complete."
echo "============================================================"
echo
echo "Activate this env in every new shell:"
echo "    source $ENV_DIR/bin/activate"
echo
echo "Then launch the PyQt UI:"
echo "    python ui/app.py"
echo
echo "Or the Streamlit UI:"
echo "    streamlit run ui/streamlit_app.py"
echo
echo "Deactivate when done:"
echo "    deactivate"
