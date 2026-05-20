"""Autosave integration tests for the MainWindow.

Verifies that PyQt mutation paths actually call into
`detector.autosave.save_snapshot` so a crash before Save loses
nothing. Uses an isolated DETECTOR_HOME so the test never touches
the user's real autosave directory.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
_DETECTOR_CORE = ROOT / "detector-core"
if _DETECTOR_CORE.exists() and str(_DETECTOR_CORE) not in sys.path:
    sys.path.insert(0, str(_DETECTOR_CORE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("DETECTOR_HOME", str(tmp_path))
    yield tmp_path


def test_autosave_module_round_trip_from_pyqt_side(isolated_home, tmp_path):
    """Sanity: detector.autosave is importable from the pyqt side
    (submodule pin includes it), and round-trips. Catches the
    common 'forgot to bump submodule' regression."""
    from detector import autosave
    rec = tmp_path / "rec.h5"
    rec.write_bytes(b"")
    autosave.save_snapshot(rec, np.array([[1, 2]]), ["user"])
    loaded = autosave.load_snapshot(rec)
    assert loaded is not None
    assert loaded["bad_intervals"] == [[1, 2]]


def test_main_window_autosave_now_writes_snapshot(qapp, isolated_home, tmp_path):
    """When MainWindow has a recording loaded, `_autosave_now()` must
    actually write a snapshot file. This is the core crash-recovery
    promise — any mutation triggers a write."""
    from PySide6.QtWidgets import QApplication
    from ui.windows.main_window import MainWindow
    from detector.recording_io import Recording
    from detector import autosave

    mw = MainWindow()
    try:
        # Fake a loaded recording without going through file I/O.
        rec_path = tmp_path / "subj01.h5"
        rec_path.write_bytes(b"")
        rec = Recording(
            y=np.zeros((100, 5), dtype=np.float32),
            fs=1000.0,
            recording_id="subj01",
            rec_type="baseline",
            source_path=rec_path,
            existing_bad_intervals=None,
        )
        mw._recording = rec
        mw._bad_intervals = np.array([[10, 20], [50, 80]], dtype=np.int64)
        mw._bad_sources = ["user", "user"]
        mw._stim_end_idx = None

        # No snapshot yet
        assert autosave.load_snapshot(rec_path) is None

        # Trigger autosave
        mw._autosave_now()

        loaded = autosave.load_snapshot(rec_path)
        assert loaded is not None
        assert loaded["bad_intervals"] == [[10, 20], [50, 80]]
        assert loaded["bad_sources"] == ["user", "user"]
    finally:
        mw.close()


def test_main_window_autosave_silent_when_no_recording(qapp, isolated_home):
    """`_autosave_now()` must be a safe no-op when no recording is
    loaded — called from random handlers during dialogs etc."""
    from ui.windows.main_window import MainWindow
    mw = MainWindow()
    try:
        assert mw._recording is None
        # No exception, no side effect.
        mw._autosave_now()
    finally:
        mw.close()


def test_main_window_clear_autosave_removes_file(qapp, isolated_home, tmp_path):
    """After explicit Save, _clear_autosave drops the sidecar so the
    next open of the same file doesn't offer to restore stale state."""
    from ui.windows.main_window import MainWindow
    from detector.recording_io import Recording
    from detector import autosave

    mw = MainWindow()
    try:
        rec_path = tmp_path / "subj01.h5"
        rec_path.write_bytes(b"")
        rec = Recording(
            y=np.zeros((100, 5), dtype=np.float32),
            fs=1000.0,
            recording_id="subj01",
            rec_type="baseline",
            source_path=rec_path,
        )
        mw._recording = rec
        mw._bad_intervals = np.array([[1, 2]], dtype=np.int64)
        mw._bad_sources = ["user"]
        mw._stim_end_idx = None
        mw._autosave_now()
        assert autosave.load_snapshot(rec_path) is not None

        mw._clear_autosave()
        assert autosave.load_snapshot(rec_path) is None
    finally:
        mw.close()
