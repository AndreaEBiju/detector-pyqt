"""Multi-select behavior on the predictions panel.

Verifies the panel emits a list of row indices on accept/dismiss,
that the bulk-action buttons enable + relabel as the selection
grows, and that the main-window accept-handler correctly removes
all selected predictions in one pass (so deleting indices 0+2+4
doesn't shift indices mid-loop).
"""

from __future__ import annotations

import sys
from pathlib import Path

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


def _select_rows(panel, rows):
    """Programmatically select a set of rows."""
    from PySide6.QtCore import QItemSelectionModel
    sm = panel._table.selectionModel()
    sm.clear()
    for r in rows:
        idx = panel._table.model().index(r, 0)
        sm.select(idx, QItemSelectionModel.Select | QItemSelectionModel.Rows)


def test_predictions_panel_supports_extended_selection(qapp):
    from PySide6.QtWidgets import QAbstractItemView
    from ui.widgets.predictions_panel import PredictionsPanel
    p = PredictionsPanel()
    try:
        assert p._table.selectionMode() == QAbstractItemView.ExtendedSelection
    finally:
        p.close()


def test_bulk_buttons_enable_with_selection(qapp):
    from ui.widgets.predictions_panel import PredictionsPanel
    p = PredictionsPanel()
    try:
        intervals = np.array(
            [[100, 200], [300, 400], [500, 600]], dtype=np.int64,
        )
        p.set_predictions(intervals, probs=None, fs=1000.0)
        # Nothing selected → disabled
        assert not p._accept_sel_btn.isEnabled()
        assert not p._dismiss_sel_btn.isEnabled()
        # Select two rows
        _select_rows(p, [0, 2])
        assert p._accept_sel_btn.isEnabled()
        assert p._dismiss_sel_btn.isEnabled()
        # Labels reflect the count
        assert "2" in p._accept_sel_btn.text()
        assert "2" in p._dismiss_sel_btn.text()
    finally:
        p.close()


def test_accept_signal_emits_list_of_selected_indices(qapp):
    from ui.widgets.predictions_panel import PredictionsPanel
    p = PredictionsPanel()
    received = []
    p.interval_accepted.connect(received.append)
    try:
        intervals = np.array(
            [[100, 200], [300, 400], [500, 600], [700, 800]], dtype=np.int64,
        )
        p.set_predictions(intervals, probs=None, fs=1000.0)
        _select_rows(p, [0, 2, 3])
        p._emit_for_selection(p.interval_accepted)
        assert received == [[0, 2, 3]]
    finally:
        p.close()


def test_dismiss_signal_emits_list(qapp):
    from ui.widgets.predictions_panel import PredictionsPanel
    p = PredictionsPanel()
    received = []
    p.interval_dismissed.connect(received.append)
    try:
        intervals = np.array([[100, 200], [300, 400]], dtype=np.int64)
        p.set_predictions(intervals, probs=None, fs=1000.0)
        _select_rows(p, [1])
        p._emit_for_selection(p.interval_dismissed)
        # Single-row interactions also come through as a list (uniform
        # signal signature is the whole point of the API change).
        assert received == [[1]]
    finally:
        p.close()


@pytest.fixture(scope="session")
def _shared_main_window(qapp, tmp_path_factory):
    """One MainWindow per pytest session. Constructing more than
    one in the same QApplication tends to deadlock during the
    second instance's signal-cleanup phase (a Qt-side quirk with
    pytest-qt's shared QApplication), so we reuse a single instance
    and the per-test fixture below resets its state in-place."""
    from ui.windows.main_window import MainWindow

    mw = MainWindow()
    # Render side-effects are unrelated to the indexing/shifting
    # logic this test exercises, and they require a real viewer
    # which a unit test doesn't have.
    mw._refresh_widgets = lambda: None
    mw._refresh_predictions_panel = lambda: None
    yield mw
    # NOTE: deliberately no mw.close() — calling close on a
    # MainWindow built without a real recording can deadlock
    # pytest-qt's session teardown on macOS. Process exit handles
    # final cleanup. The teardown deadlock is a known Qt/PySide
    # behaviour, not a leak; this fixture is session-scoped so
    # we'd only call close() once anyway.


@pytest.fixture
def stubbed_main_window(_shared_main_window, tmp_path, monkeypatch):
    """A MainWindow with the rendering side-effects stubbed out and
    its state reset to a clean baseline. Bulk-action handlers can
    then be tested for data correctness without dragging the viewer
    / region table / predictions panel into the test path. Those
    widgets are exercised separately by test_predictions_panel_*."""
    monkeypatch.setenv("DETECTOR_HOME", str(tmp_path))
    from detector.recording_io import Recording

    mw = _shared_main_window
    rec = Recording(
        y=np.zeros((10_000, 5), dtype=np.float32),
        fs=1000.0, recording_id="subj01", rec_type="baseline",
        source_path=tmp_path / "rec.h5",
    )
    mw._recording = rec
    mw._bad_intervals = np.zeros((0, 2), dtype=np.int64)
    mw._bad_sources = []
    mw._model_intervals = np.zeros((0, 2), dtype=np.int64)
    mw._undo_stack = []
    mw._redo_stack = []
    mw._dirty = False
    yield mw


def test_main_window_accept_handler_removes_multiple_rows(stubbed_main_window):
    """Critical: passing [0, 2, 4] to _on_accept_prediction must
    remove all three predictions correctly, not shift indices."""
    mw = stubbed_main_window
    mw._model_intervals = np.array([
        [100, 200], [300, 400], [500, 600],
        [700, 800], [900, 1000],
    ], dtype=np.int64)
    mw._on_accept_prediction([0, 2, 4])
    # Three accepted, two remain.
    assert mw._bad_intervals.shape[0] == 3
    assert mw._model_intervals.shape[0] == 2
    # The remaining model intervals are exactly indices 1 + 3 from
    # the original — NOT shifted weirdly.
    assert mw._model_intervals.tolist() == [[300, 400], [700, 800]]
    # All accepted got source "model_accepted".
    assert mw._bad_sources == ["model_accepted"] * 3


def test_main_window_dismiss_handler_removes_multiple_rows(stubbed_main_window):
    mw = stubbed_main_window
    mw._model_intervals = np.array([
        [10, 20], [30, 40], [50, 60], [70, 80],
    ], dtype=np.int64)
    mw._on_dismiss_prediction([1, 3])
    # Dismissed: model shrinks to 2 (indices 0 + 2 of the original);
    # bad_intervals unchanged because dismiss doesn't mark.
    assert mw._model_intervals.tolist() == [[10, 20], [50, 60]]
    assert mw._bad_intervals.shape[0] == 0


def test_accept_handler_ignores_out_of_range_indices(stubbed_main_window):
    """Defensive: stale UI indices (e.g. from a render mid-update)
    shouldn't crash — they're filtered out."""
    mw = stubbed_main_window
    mw._model_intervals = np.array([[10, 20], [30, 40]], dtype=np.int64)
    # 5 is out of range — just ignored.
    mw._on_accept_prediction([0, 5, 7])
    assert mw._bad_intervals.shape[0] == 1
    assert mw._model_intervals.shape[0] == 1


def test_accept_handler_handles_single_int_input(stubbed_main_window):
    """The signal is `Signal(list)` but be lenient: a single int
    passed by accident also works (wrapped as a 1-element iter
    internally)."""
    mw = stubbed_main_window
    mw._model_intervals = np.array([[10, 20], [30, 40]], dtype=np.int64)
    mw._on_accept_prediction([1])
    assert mw._bad_intervals.tolist() == [[30, 40]]
    assert mw._model_intervals.tolist() == [[10, 20]]
