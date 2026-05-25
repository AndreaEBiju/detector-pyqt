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

# ---------------------------------------------------------------
# Apple Silicon / Accelerate thread-safety workaround.
#
# On macOS, numpy ships linked against Apple's Accelerate
# framework, whose LAPACK is NOT safe under concurrent calls
# from multiple Python threads. Symptom: SIGBUS (Bus error) deep
# inside scipy.signal.sosfiltfilt -> numpy.linalg.solve while the
# UI paint thread is concurrently running numpy.nanmin (for
# pyqtgraph bounds). Process dies with no Python exception.
#
# Forcing every BLAS-ish backend to single-thread mode serializes
# the calls per-thread, which is enough to dodge the crash. The
# *proper* fix is reinstalling numpy against OpenBLAS, but this
# costs nothing and works.
#
# These ONLY affect threading inside BLAS/LAPACK kernels; the
# Python-level workers in ui/workers/ still run in parallel.
# ---------------------------------------------------------------
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---------------------------------------------------------------
# h5py + CloudStorage (Drive) concurrency workaround.
#
# When multiple multiprocessing.Pool workers concurrently open
# large h5py files hosted on macOS CloudStorage (Google Drive /
# OneDrive / iCloud), h5py's default file-locking conflicts with
# the CloudStorage layer and some workers see a misleading
# "OSError: Unable to synchronously open file (file signature
# not found)" -- even when the file is fully synced and the
# magic bytes are correct.
#
# Disabling h5py's locking with this env var lets concurrent
# read-only access succeed on Drive paths. Safe for our usage
# (we never WRITE to the source recording files; outputs land
# in separate files).
#
# The retrain subprocess already sets this; the PyQt app didn't
# until Andrea hit it during bulk inference on 18 notched
# recordings (9 failed, all the larger stim_rec_ files).
# ---------------------------------------------------------------
export HDF5_USE_FILE_LOCKING=FALSE

# Defense-in-depth: enable Python's fault handler so any native
# crash (segfault / bus error) writes a C stack trace instead of
# dying silently.
exec python -X faulthandler ui/app.py "$@"
