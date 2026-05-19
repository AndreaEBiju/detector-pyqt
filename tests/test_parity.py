"""M6.1 output-parity test suite.

Verifies that the PyQt UI and the Streamlit UI produce identical
outputs for the same input, by checking that:

1. Both UIs ultimately invoke the same backend code paths (no
   secret divergence in import paths).
2. The shared backend functions (`detect_bad_with_progress`,
   `Manifest`, `save_native`/`save_matlab_compatible`, `RetrainJob` /
   `start_retrain`, `detector.review.extract_disagreements`) produce
   bit-identical results when called with the same inputs.
3. Review JSON written by the PyQt `ReviewPanel.write_review_json`
   matches the schema written by Streamlit's
   `disagreement_review.save_review_json`.

This file is the contract. Any future divergence between the UIs
fails one of these tests immediately.

Most tests don't require a model artifact on disk; they exercise
the shared-import contract + small synthetic data. The
inference-output test skips when `~/.detector/artifacts/model_v0.1.0/`
isn't reachable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
_DETECTOR_CORE = ROOT / "detector-core"
if _DETECTOR_CORE.exists() and str(_DETECTOR_CORE) not in sys.path:
    sys.path.insert(0, str(_DETECTOR_CORE))


# ----------------------------------------------------------------------
# Contract: PyQt UI imports the lifted backend, not a local copy
# ----------------------------------------------------------------------

def test_pyqt_inference_worker_imports_lifted_backend():
    """If PyQt's worker stops calling detector.predict.detect_bad_with_progress
    (e.g. someone copies the function into ui/ as a quick fix), the
    two UIs will silently diverge. Pin this contract."""
    from ui.workers import inference_worker
    src = Path(inference_worker.__file__).read_text()
    assert "from detector.predict import detect_bad_with_progress" in src, (
        "PyQt inference worker must import detect_bad_with_progress from "
        "detector.predict (the lifted shared backend). Found local copy?"
    )


def test_pyqt_retrain_worker_imports_lifted_backend():
    """Same contract for the retrain subprocess plumbing."""
    from ui.workers import retrain_worker
    src = Path(retrain_worker.__file__).read_text()
    assert "from detector.retrain_subprocess import" in src, (
        "PyQt retrain worker must import from detector.retrain_subprocess "
        "(the lifted shared backend)."
    )


def test_pyqt_uses_detector_recording_io_not_streamlit_loader():
    """The PyQt main window must read recordings via
    `detector.recording_io.load_recording`, NOT via the Streamlit
    `ui/data/loaders.py` (which is a re-export shim and would
    conflict with the PyQt-side `ui/`)."""
    from ui.windows import main_window
    src = Path(main_window.__file__).read_text()
    assert "from detector.recording_io import" in src


def test_pyqt_uses_detector_labeled_save():
    from ui.windows import main_window
    src = Path(main_window.__file__).read_text()
    for sym in ("save_native", "save_matlab_compatible",
                 "save_segment_table"):
        assert sym in src, f"main_window must call {sym}"
    assert "from detector.labeled_save import" in src


# ----------------------------------------------------------------------
# Bit-identity: detect_bad_with_progress
# ----------------------------------------------------------------------

def _synth_recording():
    from detector import features as F
    fs = 24414.0625
    n_long = F._samples_from_ms(F.DEFAULT_WINDOW_MS["long"], fs)
    rng = np.random.default_rng(0)
    y = rng.standard_normal((n_long * 2, 5)).astype(np.float32) * 1e-4
    return y, fs


def _model_artifact_or_skip():
    artifact_dir = Path.home() / ".detector" / "artifacts" / "model_v0.1.0"
    booster = artifact_dir / "booster.txt"
    if not booster.exists() or not booster.is_file():
        pytest.skip(f"model_v0.1.0 not on local disk at {booster}")
    try:
        open(booster, "rb").read(1)
    except OSError as e:
        pytest.skip(f"booster unreadable ({e})")
    import os
    os.environ["DETECTOR_ARTIFACTS"] = str(artifact_dir.parent)
    from detector.model_artifact import ModelArtifact
    return ModelArtifact.load(artifact_dir)


def test_detect_bad_with_progress_result_shape_is_stable():
    """The dict shape returned by `detect_bad_with_progress` is the
    contract between the backend and both UIs. Pin the keys so a
    future refactor that drops one fails loudly."""
    artifact = _model_artifact_or_skip()
    from detector.predict import detect_bad_with_progress
    y, fs = _synth_recording()
    res = detect_bad_with_progress(
        y, fs, artifact, return_probs=True, return_features=True,
    )
    expected_keys = {
        "status", "intervals", "bad_fraction", "threshold_used",
        "fs", "n_samples", "model_version", "stride_ms", "windows_ms",
        "min_gap_ms", "min_bad_ms", "max_bad_fraction", "created_at",
        "per_window_prob", "first_position_sample", "position_samples",
        "features",
    }
    assert expected_keys.issubset(res.keys()), (
        f"missing keys: {expected_keys - set(res.keys())}"
    )


def test_inference_two_calls_bit_identical():
    """Two consecutive calls to detect_bad_with_progress on the same
    inputs must produce identical intervals (no hidden RNG / state).
    This is the bit-identity guarantee both UIs depend on."""
    artifact = _model_artifact_or_skip()
    from detector.predict import detect_bad_with_progress
    y, fs = _synth_recording()
    a = detect_bad_with_progress(y, fs, artifact, return_probs=True)
    b = detect_bad_with_progress(y, fs, artifact, return_probs=True)
    np.testing.assert_array_equal(a["intervals"], b["intervals"])
    assert a["status"] == b["status"]
    assert a["threshold_used"] == b["threshold_used"]
    assert a["bad_fraction"] == pytest.approx(b["bad_fraction"])


# ----------------------------------------------------------------------
# Manifest round-trip (M6.1: manifest mutation visible to both UIs)
# ----------------------------------------------------------------------

def test_manifest_round_trip(tmp_path, monkeypatch):
    """Add a recording to a fresh manifest. Re-load it. Verify the
    entry survives. (Both UIs use the same Manifest class; this is
    a self-test confirming the shared module works.)"""
    monkeypatch.setenv("DETECTOR_HOME", str(tmp_path))
    from detector.manifest import Manifest
    from detector import paths as P
    manifest_path = P.get_manifest_path()
    # Manifest() takes defaults for everything — the dataclass keyword
    # is `schema_version`, not `version` (this is the canonical empty
    # manifest the `detector init` CLI writes).
    m = Manifest()
    m.save(manifest_path)

    rec = {
        "recording_id": "test_rec",
        "source_path": "/tmp/nonexistent.mat",
        "splitter_clean_path": "/tmp/nonexistent_clean.h5",
        "splitter_bad_path": "/tmp/nonexistent_bad.h5",
        "rec_type": "baseline",
        "fs": 24414.0625,
        "n_samples": 1_000_000,
        "n_channels": 5,
        "label_source": "human",
        "added_by": "parity_test",
        "notes": "",
        "model_version_last_trained_on": None,
    }
    m = Manifest.load(manifest_path)
    m.add_recording(rec, by="parity_test")
    m.save(manifest_path)

    m2 = Manifest.load(manifest_path)
    assert any(r["recording_id"] == "test_rec" for r in m2.list_recordings())


# ----------------------------------------------------------------------
# Review JSON schema parity (M6.1)
# ----------------------------------------------------------------------

def test_review_json_schema_matches_streamlit():
    """Verify both UIs' review-JSON writers produce the same schema.

    We can't easily exec the Streamlit writer from this test
    environment (it has package-relative imports that break under
    `importlib.util.spec_from_file_location`). Instead we verify
    parity by source inspection — both source files must build a
    payload with the same canonical keys.

    The canonical schema (per `detector/review.py`'s doc):
      top-level: {fold_id, model_version, threshold_used,
                  reviewed_at, reviewer, scores}
      per-score: {position_sample, model_prob, score, notes}

    If either UI drops or adds a key, the standalone Phase-5
    aggregator can't pool reviews from both, so this contract is
    load-bearing.
    """
    expected_top = {"fold_id", "model_version", "threshold_used",
                     "reviewed_at", "reviewer", "scores"}
    expected_score = {"position_sample", "model_prob", "score", "notes"}

    # Streamlit side: detector-core/ui/components/disagreement_review.py
    sl_src = (_DETECTOR_CORE / "ui" / "components"
              / "disagreement_review.py").read_text()
    for key in expected_top:
        assert f'"{key}"' in sl_src, (
            f"Streamlit review writer missing key {key!r}"
        )
    for key in expected_score:
        assert f'"{key}"' in sl_src, (
            f"Streamlit review writer missing score-field {key!r}"
        )

    # PyQt side: ui/widgets/review_panel.py
    py_src = (ROOT / "ui" / "widgets" / "review_panel.py").read_text()
    for key in expected_top:
        assert f'"{key}"' in py_src, (
            f"PyQt review writer missing key {key!r}"
        )
    for key in expected_score:
        assert f'"{key}"' in py_src, (
            f"PyQt review writer missing score-field {key!r}"
        )


# ----------------------------------------------------------------------
# Retrain-subprocess shared plumbing
# ----------------------------------------------------------------------

def test_retrain_subprocess_re_export_paths():
    """Streamlit's `ui.workers.training_worker` re-exports from
    `detector.retrain_subprocess`. Both modules must expose the same
    names so future Streamlit code keeps working unchanged."""
    sys.path.insert(0, str(_DETECTOR_CORE))
    try:
        from detector.retrain_subprocess import (
            RetrainJob, start_retrain, status, cancel, list_jobs, JOBS_DIR,
        )
    finally:
        sys.path.remove(str(_DETECTOR_CORE))
    assert callable(start_retrain)
    assert callable(status)
    assert callable(cancel)
    assert callable(list_jobs)
    assert RetrainJob.__name__ == "RetrainJob"
    assert isinstance(JOBS_DIR, Path)


# ----------------------------------------------------------------------
# Native HDF5 save → load round-trip (M6.1)
# ----------------------------------------------------------------------

def test_save_native_then_reload_matches(tmp_path):
    """save_native writes splitter clean.h5+bad.h5; load_recording
    reads it back. Both UIs use the same code paths."""
    from detector.recording_io import Recording, load_recording
    from detector.labeled_save import save_native

    fs = 1000.0
    n = 10_000
    y = np.zeros((n, 5), dtype=np.float32)
    y[:] = np.linspace(0, 1, n)[:, None]
    rec = Recording(
        y=y, fs=fs, recording_id="ptest", rec_type="baseline",
        source_path=tmp_path / "ptest.mat",
    )
    bad = np.array([[100, 200], [500, 700]], dtype=np.int64)
    clean = tmp_path / "ptest_clean.h5"
    save_native(rec, bad, output_clean_path=clean)
    assert clean.exists()
    rec2 = load_recording(clean)
    # load_recording derives recording_id from the file stem, so a
    # _clean.h5 round-trips with the "_clean" suffix preserved. The
    # important parity property is that the bad-intervals come back
    # bit-identical, not the id-shape.
    assert rec2.recording_id == "ptest_clean"
    np.testing.assert_array_equal(rec2.existing_bad_intervals, bad)
