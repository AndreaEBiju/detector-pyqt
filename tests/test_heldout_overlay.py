"""Smoke test for the held-out overlay window (Phase 3).

We don't have a way to assert pixel output here, so we exercise the
construction path: synthesise a tiny HDF5 recording + a matching
splitter_bad.h5, build a HeldoutOverlayWindow with a stub cache
entry, and verify that:
  - The window opens without crashing.
  - The LazyRecording loaded successfully (no fallback error path).
  - `viewer.bad_intervals` reflects what we passed via the human
    bad intervals path.
  - The model overlay reaches the viewer (verified by reading the
    viewer's internal model-region-items list count).

Closing the window after the test releases the h5py file handle so
tmp_path cleanup is clean.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def synth_recording_pair(tmp_path):
    """Make a (source_path, splitter_bad_path) pair with known
    contents the overlay window can load.

    Source: (n_ch, N) HDF5 with a `/y` and `/fs`. Tiny so opening
    + draw is millisecond-cheap.
    Splitter bad: the schema `load_bad_intervals` expects -- one
    group per interval with `start_idx` + `end_idx` datasets.
    """
    src = tmp_path / "synth_notched.mat"
    bad = tmp_path / "synth_splitter_bad.h5"
    n_ch, n_samples, fs = 5, 60_000, 24414.0625
    y = np.stack([
        k * 1e3 + np.arange(n_samples, dtype=np.float64)
        for k in range(n_ch)
    ], axis=0)  # (n_ch, N)
    with h5py.File(str(src), "w") as f:
        f.create_dataset("y", data=y)
        f.create_dataset("fs", data=np.array([[fs]]))
    # Two human bad intervals (1-based inclusive).
    with h5py.File(str(bad), "w") as f:
        for i, (s, e) in enumerate([(1000, 2000), (5000, 5500)]):
            g = f.create_group(f"interval_{i:03d}")
            g.create_dataset("start_idx", data=s)
            g.create_dataset("end_idx", data=e)
    return src, bad, n_samples, fs


def test_overlay_window_opens_with_cache(qapp, synth_recording_pair):
    src, bad, n_samples, fs = synth_recording_pair
    # Cached model intervals from a hypothetical eval run.
    model_intervals = np.array([[3000, 3800], [7000, 8000]], dtype=np.int64)
    recording_entry = {
        "recording_id": "synth_001",
        "source_path": str(src),
        "splitter_bad_path": str(bad),
    }
    cache_entry = {
        "model": model_intervals,
        "human": np.zeros((0, 2), dtype=np.int64),
        "fs": fs,
        "n_samples": n_samples,
    }
    from ui.windows.heldout_overlay_window import HeldoutOverlayWindow

    win = HeldoutOverlayWindow(recording_entry, cache_entry)
    try:
        assert win._viewer is not None, "viewer should be constructed"
        # Human intervals came from the splitter_bad.h5 file (NOT the
        # cache_entry["human"]) -- the window deliberately re-loads
        # from disk to make ground-truth the source of truth, never
        # whatever the worker happened to cache.
        assert win._viewer.bad_intervals.shape[0] == 2
        # Model overlay was applied via set_model_intervals.
        assert hasattr(win._viewer, "_model_region_items")
        # 2 intervals × n_channels stacked plots.
        total_model_items = sum(
            len(items) for items in win._viewer._model_region_items
        )
        assert total_model_items == 2 * win._lazy.n_channels
    finally:
        win.close()


def test_overlay_window_opens_without_cache(qapp, synth_recording_pair):
    """When the cache is missing, the model overlay should be empty
    but the window should still open (no crash, no exception)."""
    src, bad, _, _ = synth_recording_pair
    recording_entry = {
        "recording_id": "synth_002",
        "source_path": str(src),
        "splitter_bad_path": str(bad),
    }
    from ui.windows.heldout_overlay_window import HeldoutOverlayWindow

    win = HeldoutOverlayWindow(recording_entry, None)  # no cache, no artifact
    try:
        assert win._viewer is not None
        # Human bands still load from disk.
        assert win._viewer.bad_intervals.shape[0] == 2
        # No model bands -- _model_region_items is initialised but empty.
        total_model_items = sum(
            len(items) for items in getattr(
                win._viewer, "_model_region_items", []
            )
        )
        assert total_model_items == 0
    finally:
        win.close()


def test_overlay_window_missing_source_shows_error(qapp, tmp_path,
                                                    monkeypatch):
    """If the recording's source_path is unreachable, the window must
    not crash. It surfaces a QMessageBox and schedules itself to
    close, leaving the parent dialog alive."""
    from PySide6.QtWidgets import QMessageBox

    # Replace QMessageBox.critical so the test doesn't pop a modal.
    captured = {}

    def _fake_critical(parent, title, msg):
        captured["title"] = title
        captured["msg"] = msg
        return QMessageBox.Ok

    monkeypatch.setattr(QMessageBox, "critical", _fake_critical)

    from ui.windows.heldout_overlay_window import HeldoutOverlayWindow

    recording_entry = {
        "recording_id": "missing",
        "source_path": str(tmp_path / "does_not_exist.mat"),
        "splitter_bad_path": str(tmp_path / "also_missing.h5"),
    }
    win = HeldoutOverlayWindow(recording_entry, None)
    try:
        assert "title" in captured, "error dialog should have been shown"
        assert win._viewer is None, (
            "no viewer should be built when source open fails"
        )
    finally:
        win.close()
