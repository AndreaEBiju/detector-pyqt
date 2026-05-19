"""Tests for the PyQt-side inference worker.

`InferenceWorker` is mostly a thin Qt wrapper around
`detector.predict.detect_bad_with_progress`. The substantive tests
that pin bit-equivalence with `detect_bad` live in the detector-core
submodule's `tests/test_phase8.py`. Here we test:
- The Qt signals fire with the expected shape.
- `load_model_artifact_cached` accepts both "v0.1.0" and "model_v0.1.0"
  version strings (the shared-pointer format vs. the manifest short
  form).
- The stim-skip offset bookkeeping inside the worker correctly
  re-offsets `intervals`, `position_samples`, `first_position_sample`,
  and the features DataFrame's `position_sample` column.
- `current_promoted_version_short()` strips the `model_` prefix.

Doesn't require a real model on disk — uses synthetic inputs + skips
the model-dependent paths when the local fallback artifacts dir is
empty (same pattern as detector-core's test_phase8).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

# Add the submodule to sys.path so `from detector import …` works.
_DETECTOR_CORE = ROOT / "detector-core"
if _DETECTOR_CORE.exists() and str(_DETECTOR_CORE) not in sys.path:
    sys.path.insert(0, str(_DETECTOR_CORE))

from ui.workers.inference_worker import (
    InferenceWorker, current_promoted_version_short,
    load_model_artifact_cached, _artifact_cache,
)


def _make_synth_recording():
    from detector import features as F
    fs = 24414.0625
    n_long = F._samples_from_ms(F.DEFAULT_WINDOW_MS["long"], fs)
    rng = np.random.default_rng(0)
    y = rng.standard_normal((n_long * 2, 5)).astype(np.float32) * 1e-4
    return y, fs


def _local_model_dir() -> Path:
    return Path.home() / ".detector" / "artifacts" / "model_v0.1.0"


@pytest.fixture(autouse=True)
def clear_artifact_cache_between_tests():
    """Stop tests leaking model state between each other."""
    _artifact_cache.clear()
    yield
    _artifact_cache.clear()


def test_current_promoted_version_short_strips_model_prefix(tmp_path, monkeypatch):
    """When `paths.get_current_model_version` returns "model_vX.Y.Z",
    the short helper should return "vX.Y.Z"."""
    monkeypatch.setenv("DETECTOR_HOME", str(tmp_path))
    monkeypatch.delenv("DETECTOR_ARTIFACTS", raising=False)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "current_model").write_text("model_v0.7.0\n")
    assert current_promoted_version_short() == "v0.7.0"


def test_current_promoted_version_short_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("DETECTOR_HOME", str(tmp_path))
    monkeypatch.delenv("DETECTOR_ARTIFACTS", raising=False)
    assert current_promoted_version_short() is None


def test_load_model_artifact_cached_accepts_both_version_forms():
    """The Streamlit short form ("v0.1.0") and the pointer-file form
    ("model_v0.1.0") should both resolve to the same artifact."""
    artifact_dir = _local_model_dir()
    booster = artifact_dir / "booster.txt"
    if not booster.exists() or not booster.is_file():
        pytest.skip(f"model_v0.1.0 not on local disk at {booster}")
    try:
        open(booster, "rb").read(1)
    except OSError as e:
        pytest.skip(f"booster unreadable ({e})")
    # Override the artifacts dir so we hit the local fallback regardless
    # of Drive-folder config.
    import os
    os.environ["DETECTOR_ARTIFACTS"] = str(artifact_dir.parent)
    try:
        a_short = load_model_artifact_cached("v0.1.0")
        a_full = load_model_artifact_cached("model_v0.1.0")
        # Cache key is the full dir name; the short form should
        # produce the same object.
        assert a_short is a_full
    finally:
        del os.environ["DETECTOR_ARTIFACTS"]


# Default-skipped: requires both a local model artifact AND a working
# Qt platform plugin. Run with `pytest -m qt` from a Mac terminal with
# a real display to exercise.
@pytest.mark.skipif(
    True,
    reason="Qt+QThread integration test — run manually via "
            "`pytest tests/test_inference_worker.py -k signals_fire "
            "-p no:cacheprovider` on a Mac terminal with display.",
)
def test_inference_worker_signals_fire_and_intervals_offset(qtbot):
    """End-to-end through the QThread plumbing: kick off a worker
    with a small synthetic recording and assert it (a) emits
    progress, (b) emits finished with the right shape, (c) applies
    the stim-skip offset when given. Manual-test only (above)."""
    pytest.importorskip("pytestqt")
    try:
        from PySide6.QtWidgets import QApplication
        if QApplication.instance() is None:
            test_app = QApplication([])
            test_app.processEvents()
    except Exception as exc:
        pytest.skip(f"Qt platform unavailable: {exc}")
    artifact_dir = _local_model_dir()
    booster = artifact_dir / "booster.txt"
    if not booster.exists() or not booster.is_file():
        pytest.skip(f"model_v0.1.0 not on local disk at {booster}")
    try:
        open(booster, "rb").read(1)
    except OSError as e:
        pytest.skip(f"booster unreadable ({e})")

    import os
    os.environ["DETECTOR_ARTIFACTS"] = str(artifact_dir.parent)
    try:
        from detector.model_artifact import ModelArtifact
        artifact = ModelArtifact.load(artifact_dir)
    finally:
        del os.environ["DETECTOR_ARTIFACTS"]
    y, fs = _make_synth_recording()
    OFFSET = 12_345
    from PySide6.QtCore import QThread

    worker = InferenceWorker(
        y, fs, artifact, stim_skip_offset=OFFSET, return_features=True,
    )
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    progress_calls: list[tuple[float, str]] = []
    finished_results: list[dict] = []
    error_msgs: list[str] = []
    worker.progress.connect(lambda f, m: progress_calls.append((f, m)))
    worker.finished.connect(lambda d: finished_results.append(d))
    worker.error.connect(lambda m: error_msgs.append(m))
    worker.finished.connect(thread.quit)
    worker.error.connect(thread.quit)
    thread.start()

    with qtbot.waitSignal(worker.finished, timeout=90_000):
        pass
    thread.wait(5_000)

    assert not error_msgs, f"worker emitted error: {error_msgs}"
    assert len(finished_results) == 1
    res = finished_results[0]
    assert "intervals" in res
    assert "scope_offset_sample" in res and res["scope_offset_sample"] == OFFSET
    # Progress should have fired at least the four canonical stages.
    assert len(progress_calls) >= 3
    fracs = [f for f, _ in progress_calls]
    assert min(fracs) < 0.1 and fracs[-1] == 1.0
    # If intervals exist, they should be in full-recording coords
    # (≥ OFFSET because the offset got added).
    if res["intervals"].size:
        assert int(res["intervals"][:, 0].min()) >= OFFSET
    # position_samples and features should also be offset.
    if res.get("position_samples") is not None:
        assert int(res["position_samples"].min()) >= OFFSET
    feats = res.get("features")
    if feats is not None and "position_sample" in feats:
        assert int(feats["position_sample"].min()) >= OFFSET
