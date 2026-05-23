#!/usr/bin/env bash
# Activate the dedicated venv and launch the PyQt UI.
# macOS / Linux.
#
# Use this instead of `python ui/app.py` so you can't accidentally
# run against your global / Anaconda Python (which has been the
# source of multiple numpy/scipy version conflicts).
#
# First time? Run scripts/setup_env.sh first.

set -e

ENV_DIR=".venv"

if [ ! -d "$ENV_DIR" ]; then
    echo "ERROR: $ENV_DIR/ doesn't exist."
    echo "Run scripts/setup_env.sh first to create the dedicated env."
    exit 1
fi

# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

# Defense-in-depth: enable Python's fault handler so any native
# crash (segfault / bus error) writes a C stack trace instead of
# dying silently.
exec python -X faulthandler ui/app.py "$@"
