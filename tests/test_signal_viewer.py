"""Unit tests for `ui.widgets.signal_viewer.LazyRecording`.

Pure HDF5 + numpy assertions — no Qt dependency, so these run in CI
without a display server. The `MultiChannelViewer` itself is tested
interactively (M1.3 benchmark + manual UX check) since exercising
pyqtgraph paint paths from pytest requires `pytest-qt` + a working
display; we defer that to M2 when the labeling UI lands.
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

from ui.widgets.signal_viewer import LazyRecording


# ---------------------------------------------------------------------
# Fixtures — synthesise HDF5 files in both supported layouts.
# ---------------------------------------------------------------------

@pytest.fixture
def matlab_v73_recording(tmp_path):
    """A `.mat`-style file: `/y` shape (n_ch, N) float64, `/fs` scalar.
    Mirrors the existing `_notched.mat` schema we read in production."""
    path = tmp_path / "synth_notched.mat"
    n_ch, n_samples, fs = 5, 240_000, 24414.0625
    rng = np.random.default_rng(0)
    # Use a recognisable signal per channel so we can detect transpose
    # bugs: channel k's samples are k * 1e6 + arange.
    y = np.stack([
        k * 1e6 + np.arange(n_samples, dtype=np.float64)
        for k in range(n_ch)
    ], axis=0)
    assert y.shape == (n_ch, n_samples)
    with h5py.File(str(path), "w") as f:
        f.create_dataset("y", data=y)
        f.create_dataset("fs", data=np.array([[fs]]))
    return path, n_ch, n_samples, fs


@pytest.fixture
def flat_h5_recording(tmp_path):
    """A flat HDF5 in the (N, n_ch) convention."""
    path = tmp_path / "synth_flat.h5"
    n_ch, n_samples, fs = 5, 240_000, 24414.0625
    y = np.stack([
        k * 1e6 + np.arange(n_samples, dtype=np.float64)
        for k in range(n_ch)
    ], axis=1)
    assert y.shape == (n_samples, n_ch)
    with h5py.File(str(path), "w") as f:
        f.create_dataset("y", data=y)
        f.create_dataset("fs", data=fs)
    return path, n_ch, n_samples, fs


@pytest.fixture
def yout_recording(tmp_path):
    """`_blankmotion.mat` style: `/yOut` shape (n_ch, N) float32."""
    path = tmp_path / "synth_blankmotion.mat"
    n_ch, n_samples, fs = 5, 120_000, 24414.0625
    rng = np.random.default_rng(7)
    y = rng.standard_normal((n_ch, n_samples)).astype(np.float32) * 1e-3
    with h5py.File(str(path), "w") as f:
        f.create_dataset("yOut", data=y)
        f.create_dataset("fs", data=np.array([[fs]]))
    return path, n_ch, n_samples, fs


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_matlab_v73_layout_is_detected_and_transposed(matlab_v73_recording):
    path, n_ch, n_samples, fs = matlab_v73_recording
    rec = LazyRecording(path)
    assert rec.n_channels == n_ch
    assert rec.n_samples == n_samples
    assert rec.fs == pytest.approx(fs)
    # get_range returns (n, n_ch) — caller-facing shape regardless of
    # disk layout
    out = rec.get_range(0.0, 0.5)  # ~0.5s × fs samples
    expected_n = min(n_samples, int(round(0.5 * fs)))
    assert out.shape == (expected_n, n_ch)
    # Channel signature preserved through the transpose
    expected_ch0 = np.arange(expected_n, dtype=np.float32)
    np.testing.assert_allclose(out[:, 0], expected_ch0, atol=1e-3)
    expected_ch4 = 4e6 + np.arange(expected_n, dtype=np.float32)
    np.testing.assert_allclose(out[:, 4], expected_ch4, atol=1.0)


def test_flat_layout_is_not_transposed(flat_h5_recording):
    path, n_ch, n_samples, fs = flat_h5_recording
    rec = LazyRecording(path)
    assert rec.n_channels == n_ch
    assert rec.n_samples == n_samples
    out = rec.get_range(1.0, 1.5)
    n_expected = int(round(1.5 * fs)) - int(round(1.0 * fs))
    assert out.shape == (n_expected, n_ch)


def test_yout_dataset_is_picked_up(yout_recording):
    path, n_ch, n_samples, fs = yout_recording
    rec = LazyRecording(path)
    assert rec.n_channels == n_ch
    assert rec.n_samples == n_samples
    assert rec.duration_sec == pytest.approx(n_samples / fs)


def test_get_range_returns_float32(matlab_v73_recording):
    path, _, _, _ = matlab_v73_recording
    rec = LazyRecording(path)
    out = rec.get_range(0.0, 0.1)
    # Underlying dataset is float64; reader downcasts.
    assert out.dtype == np.float32


def test_get_range_clips_to_recording_bounds(matlab_v73_recording):
    path, n_ch, n_samples, fs = matlab_v73_recording
    rec = LazyRecording(path)
    dur = n_samples / fs
    # Request a window that runs past the end.
    out = rec.get_range(dur - 0.1, dur + 1.0)
    assert out.shape[0] > 0
    # Returned samples should not exceed n_samples.
    assert out.shape[0] <= int(round(0.1 * fs)) + 1


def test_get_range_returns_empty_when_window_past_end(matlab_v73_recording):
    path, n_ch, n_samples, fs = matlab_v73_recording
    rec = LazyRecording(path)
    dur = n_samples / fs
    out = rec.get_range(dur + 1.0, dur + 2.0)
    assert out.shape == (0, n_ch)


def test_get_range_returns_empty_for_inverted_window(matlab_v73_recording):
    path, n_ch, _, _ = matlab_v73_recording
    rec = LazyRecording(path)
    out = rec.get_range(5.0, 5.0)
    assert out.shape == (0, n_ch)


def test_missing_file_raises_filenotfounderror(tmp_path):
    with pytest.raises(FileNotFoundError):
        LazyRecording(tmp_path / "does_not_exist.mat")


def test_missing_y_dataset_raises(tmp_path):
    path = tmp_path / "no_y.h5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("fs", data=1000.0)
        f.create_dataset("not_y", data=np.zeros((10, 5)))
    with pytest.raises(ValueError, match="no `/y` or `/yOut`"):
        LazyRecording(path)


def test_missing_fs_raises(tmp_path):
    path = tmp_path / "no_fs.h5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("y", data=np.zeros((100, 5)))
    with pytest.raises(ValueError, match="no `/fs`"):
        LazyRecording(path)


def test_context_manager_closes_file(matlab_v73_recording):
    path, _, _, _ = matlab_v73_recording
    with LazyRecording(path) as rec:
        out = rec.get_range(0.0, 0.1)
        assert out.shape[0] > 0
    # After __exit__, the h5 file should be closed. Touching it should
    # raise; we just check we can open a fresh handle without lock
    # conflicts.
    with h5py.File(str(path), "r") as f:
        assert "y" in f


def test_repr_includes_metadata(matlab_v73_recording):
    path, n_ch, n_samples, fs = matlab_v73_recording
    rec = LazyRecording(path)
    r = repr(rec)
    assert path.name in r
    assert str(n_ch) in r
    assert "Hz" in r


@pytest.fixture
def flat_h5_with_overview(tmp_path):
    """A flat-HDF5 recording WITH a /y_overview dataset, mirroring
    what scripts/m1_ingest.py writes. Verifies the M2.0 read path."""
    path = tmp_path / "synth_with_overview.h5"
    n_ch, n_samples, fs = 5, 240_000, 24414.0625
    bucket_size = 1000
    n_buckets = (n_samples + bucket_size - 1) // bucket_size
    y = np.stack([
        k * 1e3 + np.arange(n_samples, dtype=np.float64) / fs
        for k in range(n_ch)
    ], axis=1)
    # Build the (2 * n_buckets, n_ch) min/max-pair overview by hand.
    ov_rows = 2 * n_buckets
    ov = np.zeros((ov_rows, n_ch), dtype=np.float32)
    for b in range(n_buckets):
        bs = b * bucket_size
        be = min(bs + bucket_size, n_samples)
        window = y[bs:be, :]
        ov[2 * b, :] = window.min(axis=0)
        ov[2 * b + 1, :] = window.max(axis=0)
    with h5py.File(str(path), "w") as f:
        f.create_dataset("y", data=y.astype(np.float32))
        f.create_dataset("y_overview", data=ov)
        f.create_dataset("fs", data=fs)
        f.attrs["overview_bucket_size"] = bucket_size
        f.attrs["overview_n_buckets"] = n_buckets
    return path, n_ch, n_samples, fs, bucket_size


def test_has_overview_detects_dataset(flat_h5_with_overview, flat_h5_recording):
    path_with, *_ = flat_h5_with_overview
    path_without, *_ = flat_h5_recording
    assert LazyRecording(path_with).has_overview is True
    assert LazyRecording(path_without).has_overview is False


def test_get_overview_for_range_returns_min_max_pairs(flat_h5_with_overview):
    path, n_ch, n_samples, fs, bucket_size = flat_h5_with_overview
    rec = LazyRecording(path)
    # Span buckets 5..10
    t_start = 5 * bucket_size / fs
    t_end = 10 * bucket_size / fs
    t, data = rec.get_overview_for_range(t_start, t_end)
    # 5 buckets × 2 rows = 10 rows
    assert data.shape == (10, n_ch)
    assert t.shape == (10,)
    # Each pair of rows shares the same t (the bucket midpoint).
    assert t[0] == t[1] and t[2] == t[3]
    # Rows 0..n_ch should be min < max for an increasing-per-channel signal.
    for ch in range(n_ch):
        assert data[0, ch] < data[1, ch], f"ch {ch}: min should be < max"


def test_get_overview_raises_when_absent(flat_h5_recording):
    path, *_ = flat_h5_recording
    rec = LazyRecording(path)
    with pytest.raises(RuntimeError, match="no /y_overview"):
        rec.get_overview_for_range(0.0, 1.0)


def test_lazy_read_does_not_load_whole_file(matlab_v73_recording):
    """Reading a small window should touch only the requested range —
    we approximate this by checking the function returns quickly even
    when n_samples is large (the synth fixture is 240k samples, a
    proxy). Strictly speaking 'lazy' is a property of h5py + libhdf5;
    this test is a sanity check that we don't accidentally materialise
    `rec._y[:]` somewhere in our wrapper."""
    import time
    path, _, _, _ = matlab_v73_recording
    rec = LazyRecording(path)
    t0 = time.perf_counter()
    rec.get_range(0.0, 0.05)
    dt = time.perf_counter() - t0
    # The synth file is small (~10MB on disk) so this is mostly a smoke
    # test, but if a future refactor materialises the whole dataset
    # we'd see times growing with n_samples not viewport.
    assert dt < 0.5, f"get_range was slow ({dt*1000:.1f}ms) — possible eager load"
