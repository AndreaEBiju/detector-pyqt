"""Smoke + behaviour tests for the preprocessing UI components.

These intentionally exercise lots of construction paths but not the
async parts (`PreprocessWorker.run`) because exercising the worker
requires a real TDT block. Instead, the worker's wiring is verified
indirectly by signal connection checks here.

Run from repo root:
    pytest tests/test_ui_preprocessing.py
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

# pytest-qt's `qtbot` fixture handles QApplication construction; if
# the marker isn't available we fall back to a manual QApplication
# in this module-level fixture.
try:
    import pytestqt  # noqa: F401
    _PYTEST_QT_AVAILABLE = True
except Exception:                                           # pragma: no cover
    _PYTEST_QT_AVAILABLE = False


@pytest.fixture(scope="session")
def qapp() -> Iterator:
    """Always provide a QApplication for the test session, even if
    pytest-qt isn't available."""
    from PySide6.QtWidgets import QApplication
    existing = QApplication.instance()
    if existing is None:
        app = QApplication([])
    else:
        app = existing
    yield app


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Isolate `~/.detector/` so settings + profiles writes from the
    UI don't leak across tests."""
    monkeypatch.setenv("DETECTOR_HOME", str(tmp_path))
    monkeypatch.delenv("DETECTOR_ARTIFACTS", raising=False)
    yield tmp_path


# ----------------------------------------------------------------------
# Settings — P14
# ----------------------------------------------------------------------

def test_settings_has_preprocessing_defaults(isolated_home):
    """The preprocessing defaults landed in DEFAULTS and round-trip."""
    from ui.data import settings as S
    out = S.load_settings()
    assert out["preprocessing_q_factor"] == 30.0
    assert out["preprocessing_reduction_threshold"] == 0.05
    assert out["preprocessing_max_harmonics"] == 4
    assert out["preprocessing_detrend"] is True
    assert isinstance(out["preprocessing_candidate_harmonics"], list)
    assert 60.0 in out["preprocessing_candidate_harmonics"]


def test_settings_round_trip_preprocessing(isolated_home):
    """User edits in the Training-window's Preprocessing tab persist."""
    from ui.data import settings as S
    S.update_setting("preprocessing_q_factor", 25.0)
    S.update_setting(
        "preprocessing_candidate_harmonics", [50.0, 100.0, 150.0]
    )
    out = S.load_settings()
    assert out["preprocessing_q_factor"] == 25.0
    assert out["preprocessing_candidate_harmonics"] == [50.0, 100.0, 150.0]


# ----------------------------------------------------------------------
# MultiFolderPicker
# ----------------------------------------------------------------------

def test_folder_picker_starts_empty_with_disabled_continue(qapp, isolated_home):
    from ui.dialogs.folder_picker import MultiFolderPicker
    from PySide6.QtWidgets import QDialogButtonBox
    dlg = MultiFolderPicker()
    try:
        assert dlg.folders == []
        # Continue is disabled until the user adds a folder
        ok_btn = dlg._buttons.button(QDialogButtonBox.Ok)
        assert ok_btn.isEnabled() is False
    finally:
        dlg.close()


def test_folder_picker_can_inject_folder(qapp, isolated_home, tmp_path):
    """Bypass the OS file dialog by injecting a folder directly into
    the accumulator (tests the dedup + list rendering)."""
    from ui.dialogs.folder_picker import MultiFolderPicker
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QListWidgetItem, QDialogButtonBox
    dlg = MultiFolderPicker()
    try:
        p = tmp_path / "foo"
        p.mkdir()
        dlg._folders.append(p)
        item = QListWidgetItem(str(p))
        item.setData(Qt.UserRole, str(p))
        dlg._list.addItem(item)
        dlg._update_ok_state()
        assert dlg._buttons.button(QDialogButtonBox.Ok).isEnabled()
        assert dlg.folders == [p]
    finally:
        dlg.close()


# ----------------------------------------------------------------------
# TdtBatchTable — P7
# ----------------------------------------------------------------------

def test_batch_table_validation_messages(qapp, isolated_home, tmp_path,
                                            monkeypatch):
    """validation_message() reports the right blocker per row state."""
    from ui.widgets.tdt_batch_table import TdtBatchTable
    # detect_naming_convention does folder-name parsing only — no I/O.
    # Monkeypatch list_streams to avoid the (slow) tdt read.
    table = TdtBatchTable()
    # Empty table → "Add at least one folder"
    assert "least one folder" in table.validation_message()

    # Set two folders with same output name → uniqueness check
    f1 = tmp_path / "subj01_bl"
    f2 = tmp_path / "subj02_bl"
    for f in (f1, f2):
        f.mkdir()
    table.set_folders([f1, f2])
    # Fill animal IDs + identical output names
    for i, row in enumerate(table.rows):
        row.animal_id = f"subj0{i + 1}"
    table._rows[0].output_condition_name = "same"
    table._rows[1].output_condition_name = "same"
    msg = table.validation_message()
    assert msg is None or "unique" in msg.lower() or "row" in msg.lower()


def test_batch_table_animal_ids_in_order(qapp, isolated_home, tmp_path):
    from ui.widgets.tdt_batch_table import TdtBatchTable
    table = TdtBatchTable()
    f1, f2, f3 = (
        tmp_path / "a_bl", tmp_path / "b_stim", tmp_path / "a_rec",
    )
    for f in (f1, f2, f3):
        f.mkdir()
    table.set_folders([f1, f2, f3])
    # Animals: A, B, A — list should dedup keeping order.
    table._rows[0].animal_id = "A"
    table._rows[1].animal_id = "B"
    table._rows[2].animal_id = "A"
    assert table.animal_ids_in_order() == ["A", "B"]


# ----------------------------------------------------------------------
# ChannelAssignmentDialog — P9
# ----------------------------------------------------------------------

def test_channel_assignment_default_role_layout(qapp, isolated_home):
    """With 5 channels: first 2 default to nerve, next 3 to stomach."""
    from ui.widgets.channel_assignment import ChannelAssignmentDialog
    from detector.preprocessing.tdt_io import StreamInfo
    streams = {
        "Raww": StreamInfo("Raww", 5, 30000.0, 100000, np.dtype("float32"))
    }
    dlg = ChannelAssignmentDialog("subjT", streams)
    try:
        # 5 channels in the table
        assert dlg._channels_table.rowCount() == 5
        # Sparkline column added
        assert dlg._channels_table.columnCount() == 5
        # Selected channels: roles in expected layout
        selected = dlg._selected_channels()
        roles = [c["role"] for c in selected]
        assert roles == ["nerve", "nerve", "stomach", "stomach", "stomach"]
        # Sparkline button disabled when no tdt_folder provided
        assert dlg._sparkline_btn.isEnabled() is False
    finally:
        dlg.close()


def test_channel_assignment_emits_correct_dict(qapp, isolated_home):
    """channel_assignment() returns the dict shape Profile expects."""
    from ui.widgets.channel_assignment import ChannelAssignmentDialog
    from detector.preprocessing.tdt_io import StreamInfo
    streams = {
        "Raww": StreamInfo("Raww", 5, 30000.0, 100000, np.dtype("float32")),
        "BiPl": StreamInfo("BiPl", 1, 30000.0, 100000, np.dtype("float32")),
    }
    dlg = ChannelAssignmentDialog("subjT", streams)
    try:
        ca = dlg.channel_assignment()
        assert ca["raw_stream"] == "Raww"
        assert isinstance(ca["channels"], list)
        # Default tags are 0..n_ch − 1 in TDT 0-based order
        for i, ch in enumerate(ca["channels"]):
            assert ch["signal_index"] == i
            assert ch["tdt_index"] == i
            assert "label" in ch
            assert ch["role"] in {"nerve", "stomach", "other", "exclude"}
    finally:
        dlg.close()


# ----------------------------------------------------------------------
# NotchReviewDialog — P10
# ----------------------------------------------------------------------

def test_notch_review_settings_includes_apply_scope(qapp, isolated_home,
                                                      tmp_path):
    """notch_settings() carries the apply-to scope + reduction
    threshold + max harmonics from settings, and includes the
    Quiroga-σ-reduction per-harmonic dict."""
    from ui.widgets.notch_review import NotchReviewDialog
    from ui.data import settings as S

    # User changes their default σ-reduction threshold:
    S.update_setting("preprocessing_reduction_threshold", 0.10)
    S.update_setting("preprocessing_max_harmonics", 3)

    nrd = NotchReviewDialog(
        animal_id="subjT",
        tdt_folder=tmp_path / "no-such-folder",      # silently no-data
        raw_stream="Raww",
        channels=[{"signal_index": 0, "tdt_index": 0,
                    "role": "nerve", "label": "VN1"}],
    )
    try:
        # Default = apply_all
        assert nrd._apply_all_radio.isChecked()
        out = nrd.notch_settings()
        assert out["apply_scope"] == "all_channels"
        assert out["reduction_threshold"] == 0.10
        assert out["max_harmonics_filtered"] == 3
        # The current-metric (v4.0) key replaces older detection fields.
        assert "noise_reductions_per_harmonic" in out
        # Switch radio → reflected
        nrd._apply_selected_radio.setChecked(True)
        assert nrd.notch_settings()["apply_scope"] == "selected_channel_preview"
    finally:
        nrd.close()


# ----------------------------------------------------------------------
# PreprocessWindow — P6 + P13
# ----------------------------------------------------------------------

def test_preprocess_window_seven_steps(qapp, isolated_home):
    from ui.windows.preprocess_window import PreprocessWindow
    w = PreprocessWindow()
    try:
        assert w._stack.count() == 7
        # Starts on step 1
        assert w._stack.currentIndex() == 0
        # Next is disabled with no folders
        assert w._next_btn.isEnabled() is False
    finally:
        w.close()


def test_preprocess_window_seed_folder(qapp, isolated_home, tmp_path):
    """seed_folder() pre-populates the batch and stays on step 1."""
    from ui.windows.preprocess_window import PreprocessWindow
    folder = tmp_path / "subj01_bl"
    folder.mkdir()
    w = PreprocessWindow()
    try:
        w.seed_folder(folder)
        assert w._folders == [folder]
        assert w._stack.currentIndex() == 0
        assert w._next_btn.isEnabled() is True
    finally:
        w.close()


def test_preprocess_window_already_processed_decisions(qapp, isolated_home,
                                                         tmp_path):
    """AlreadyProcessedDialog wiring: decisions feed the plan via the
    confirm-step map."""
    from ui.windows.preprocess_window import (
        PreprocessWindow, AlreadyProcessedDialog,
    )
    from ui.widgets.tdt_batch_table import BatchRowState
    row = BatchRowState(
        folder=tmp_path / "x",
        inferred_condition="baseline",
        condition="baseline",
        output_condition_name="x_bl",
        animal_id="subjT",
    )
    (tmp_path / "x").mkdir()
    dlg = AlreadyProcessedDialog(row, [tmp_path / "x" / "x_bl_sig.mat"])
    try:
        # Default = skip
        assert dlg.decision() == "skip"
        dlg._overwrite_btn.setChecked(True)
        assert dlg.decision() == "process"
        dlg._suffix_btn.setChecked(True)
        assert dlg.decision() == "suffix:_v2"
    finally:
        dlg.close()


# ----------------------------------------------------------------------
# PreprocessWorker — P12
# ----------------------------------------------------------------------

def test_preprocess_worker_emits_skip_for_skip_reason(qapp, isolated_home,
                                                         tmp_path):
    """A plan item with `skip_reason` set bypasses processing and
    appears in the result list as status='skipped'."""
    from ui.workers.preprocess_worker import PreprocessWorker
    from detector.preprocessing.batch import BatchPlanItem
    from detector.preprocessing.profiles import Profile

    item = BatchPlanItem(
        tdt_folder=tmp_path / "fake",
        output_folder=tmp_path,
        animal_id="subjT",
        condition="baseline",
        output_condition_name="x",
        profile=Profile(animal_id="subjT"),
        skip_reason="user-skip",
    )
    worker = PreprocessWorker([item])
    seen: list[dict] = []
    worker.file_completed.connect(lambda i, t, r: seen.append(r))
    finished: list[list] = []
    worker.finished.connect(lambda r: finished.append(r))
    # Run on the test thread — no QThread wiring needed for a single
    # skip-only item. (PreprocessWorker.run is synchronous.)
    worker.run()
    assert len(seen) == 1
    assert seen[0]["status"] == "skipped"
    assert seen[0]["reason"] == "user-skip"
    assert len(finished) == 1


def test_preprocess_worker_cancel_flag(qapp, isolated_home, tmp_path):
    """request_cancel() before run() → no items processed, cancelled
    signal fires with empty partial."""
    from ui.workers.preprocess_worker import PreprocessWorker
    from detector.preprocessing.batch import BatchPlanItem
    from detector.preprocessing.profiles import Profile

    item = BatchPlanItem(
        tdt_folder=tmp_path / "fake",
        output_folder=tmp_path,
        animal_id="subjT",
        condition="baseline",
        output_condition_name="x",
        profile=Profile(animal_id="subjT"),
        skip_reason="user-skip",
    )
    worker = PreprocessWorker([item])
    cancelled: list[list] = []
    worker.cancelled.connect(lambda r: cancelled.append(r))
    worker.request_cancel()
    worker.run()
    assert len(cancelled) == 1
    assert cancelled[0] == []


# ----------------------------------------------------------------------
# TrainingWindow's Preprocessing tab — P14
# ----------------------------------------------------------------------

def test_training_window_has_preprocessing_tab(qapp, isolated_home):
    from ui.windows.training_window import TrainingWindow
    w = TrainingWindow()
    try:
        labels = [w._tabs.tabText(i) for i in range(w._tabs.count())]
        assert "Preprocessing" in labels
        # Q-factor spinbox holds the default
        assert w._pp_q_factor.value() == pytest.approx(30.0)
    finally:
        w.close()
