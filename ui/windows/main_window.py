"""Main window — Browser & Label (M2.1 + ties everything together).

Wires the M2 widgets:
- Central widget: `MultiChannelViewer` (lazy h5py-backed multi-channel plot).
- Right dock: `RegionTable` (the marked-intervals list).
- Bottom dock: `OverviewStrip` (whole-recording navigator).
- Menu / toolbar / status bar.

State model:
- `self.recording_for_save`: the `detector.recording_io.Recording`
  dataclass loaded for the active file. Used for save flow.
- `self.bad_intervals`: `(k, 2)` 1-based-inclusive sample indices.
- `self.bad_sources`: parallel list of `"user"/"existing"/"unknown"`.
- `self.stim_end_idx`: 1-based stim/recovery boundary, or None.

A 20-deep undo stack snapshots `(bad_intervals, bad_sources)` before
every mutation. M3 will extend this state with model predictions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Self-contained sys.path setup so `from detector …` works whether
# this module is imported via ui/app.py or directly. The repo-vs-
# submodule order is critical — same lesson as scripts/m1_benchmark.py:
# detector-core has its own `ui/` package (Streamlit) so it must come
# AFTER detector-pyqt in sys.path or it'll shadow our ui.widgets.
_repo_root = Path(__file__).resolve().parent.parent.parent
_detector_core = _repo_root / "detector-core"
if _detector_core.exists():
    if str(_detector_core) not in sys.path:
        sys.path.insert(0, str(_detector_core))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QComboBox, QDockWidget, QFileDialog, QInputDialog, QLabel, QMainWindow,
    QMessageBox, QProgressDialog, QPushButton, QStatusBar, QTabWidget,
    QToolBar, QWidget,
)

from detector import paths as detector_paths
from detector import review as detector_review
from detector.recording_io import Recording, load_recording
from detector.labeled_save import (
    save_native, save_matlab_compatible, save_period_split,
    save_segment_table,
)
from detector.predict import list_available_versions

from ui.data import settings as ui_settings
from ui.widgets.signal_viewer import LazyRecording, MultiChannelViewer
from ui.widgets.region_table import RegionTable
from ui.widgets.overview_strip import OverviewStrip
from ui.widgets.predictions_panel import PredictionsPanel
from ui.widgets.review_panel import ReviewPanel
from ui.workers.inference_worker import (
    InferenceWorker, current_promoted_version_short,
    load_model_artifact_cached,
)


UNDO_DEPTH = 20


class MainWindow(QMainWindow):
    def __init__(self, recording_path: Optional[Path] = None):
        super().__init__()
        self.setWindowTitle("Detector — PyQt")
        self.resize(1400, 800)

        # Two pieces of state: a `LazyRecording` (h5py-backed, lazy
        # reads — what the viewer uses) AND a `Recording` dataclass
        # (eager load into memory — what the save flow needs because
        # labeled_save.save_native walks the full signal). We do
        # both because the save flow is once-per-session and a few
        # GB of in-memory float32 is acceptable; the viewer's
        # constant-time pan needs the lazy version.
        self._lazy: Optional[LazyRecording] = None
        self._recording: Optional[Recording] = None

        # Edit state
        self._bad_intervals: np.ndarray = np.zeros((0, 2), dtype=np.int64)
        self._bad_sources: list[str] = []
        self._stim_end_idx: Optional[int] = None
        self._undo_stack: list[tuple[np.ndarray, list[str], Optional[int]]] = []
        self._redo_stack: list[tuple[np.ndarray, list[str], Optional[int]]] = []
        # Dirty bit so we can warn before closing without saving.
        self._dirty: bool = False

        # M3 inference state.
        self._model_intervals: Optional[np.ndarray] = None
        self._model_per_window_probs: Optional[np.ndarray] = None
        self._model_position_samples: Optional[np.ndarray] = None
        self._model_features = None                # pandas DataFrame
        self._model_version: Optional[str] = None
        self._model_threshold_used: Optional[float] = None
        self._inference_thread: Optional[QThread] = None
        self._inference_worker: Optional[InferenceWorker] = None
        self._inference_progress: Optional[QProgressDialog] = None
        # Settings shape: {"model_version", "auto_run_on_open",
        # "inference_skip_stim", "last_recording_dir"}.
        self._settings = ui_settings.load_settings()

        # Widgets — placeholders until a recording opens.
        self._viewer: Optional[MultiChannelViewer] = None
        self._overview: Optional[OverviewStrip] = None
        self._region_table = RegionTable()
        self._region_table.interval_jumped.connect(self._on_jump_to_interval)
        self._region_table.interval_deleted.connect(self._on_delete_interval)
        self._predictions_panel = PredictionsPanel()
        self._predictions_panel.interval_accepted.connect(
            self._on_accept_prediction
        )
        self._predictions_panel.interval_dismissed.connect(
            self._on_dismiss_prediction
        )
        self._predictions_panel.interval_jumped.connect(
            self._on_jump_to_prediction
        )
        self._predictions_panel.accept_all_requested.connect(
            self._on_accept_all_predictions
        )
        # Review-panel dock comes and goes on demand.
        self._review_panel: Optional[ReviewPanel] = None
        self._review_dock: Optional[QDockWidget] = None

        self._build_menus()
        self._build_toolbar()
        self.setStatusBar(QStatusBar())
        self._update_status_bar()

        # Right dock for the tabbed Marked / Model panel.
        self._right_tabs = QTabWidget()
        self._right_tabs.addTab(self._region_table, "🟥 Marked (0)")
        self._right_tabs.addTab(self._predictions_panel, "🟧 Model")
        self._table_dock = QDockWidget("Intervals", self)
        self._table_dock.setObjectName("intervals_dock")
        self._table_dock.setWidget(self._right_tabs)
        self._table_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.RightDockWidgetArea, self._table_dock)

        # If a recording path was passed on the command line, open it now.
        if recording_path is not None:
            QTimer.singleShot(0, lambda: self._open_path(Path(recording_path)))

    # ------------------------------------------------------------------
    # UI scaffolding
    # ------------------------------------------------------------------

    def _build_menus(self) -> None:
        mb = self.menuBar()
        # File menu
        file_menu = mb.addMenu("&File")
        self._action_open = QAction("&Open recording…", self)
        self._action_open.setShortcut(QKeySequence.Open)
        self._action_open.triggered.connect(self._on_open_clicked)
        file_menu.addAction(self._action_open)

        self._action_save = QAction("&Save", self)
        self._action_save.setShortcut(QKeySequence.Save)
        self._action_save.triggered.connect(self._on_save_clicked)
        self._action_save.setEnabled(False)
        file_menu.addAction(self._action_save)

        self._action_save_as = QAction("Save &As…", self)
        self._action_save_as.setShortcut(QKeySequence.SaveAs)
        self._action_save_as.triggered.connect(self._on_save_as_clicked)
        self._action_save_as.setEnabled(False)
        file_menu.addAction(self._action_save_as)

        file_menu.addSeparator()
        action_quit = QAction("&Quit", self)
        action_quit.setShortcut(QKeySequence.Quit)
        action_quit.triggered.connect(self.close)
        file_menu.addAction(action_quit)

        # Edit menu
        edit_menu = mb.addMenu("&Edit")
        self._action_undo = QAction("&Undo", self)
        self._action_undo.setShortcut(QKeySequence.Undo)
        self._action_undo.triggered.connect(self._on_undo)
        self._action_undo.setEnabled(False)
        edit_menu.addAction(self._action_undo)

        self._action_redo = QAction("&Redo", self)
        self._action_redo.setShortcut(QKeySequence("Ctrl+Shift+Z"))
        self._action_redo.triggered.connect(self._on_redo)
        self._action_redo.setEnabled(False)
        edit_menu.addAction(self._action_redo)

        # View menu
        view_menu = mb.addMenu("&View")
        self._action_reset_zoom = QAction("Reset zoom (0–60s)", self)
        self._action_reset_zoom.setShortcut("R")
        self._action_reset_zoom.triggered.connect(self._on_reset_zoom)
        view_menu.addAction(self._action_reset_zoom)
        self._action_fit_all = QAction("Fit all", self)
        self._action_fit_all.setShortcut("F")
        self._action_fit_all.triggered.connect(self._on_fit_all)
        view_menu.addAction(self._action_fit_all)
        self._action_zoom_in = QAction("Zoom in (×0.5)", self)
        self._action_zoom_in.setShortcut("Z")
        self._action_zoom_in.triggered.connect(lambda: self._zoom(0.5))
        view_menu.addAction(self._action_zoom_in)
        self._action_zoom_out = QAction("Zoom out (×2)", self)
        self._action_zoom_out.setShortcut("X")
        self._action_zoom_out.triggered.connect(lambda: self._zoom(2.0))
        view_menu.addAction(self._action_zoom_out)
        view_menu.addSeparator()
        self._action_step_prev = QAction("Previous interval", self)
        self._action_step_prev.setShortcut("Left")
        self._action_step_prev.triggered.connect(lambda: self._step_interval(-1))
        view_menu.addAction(self._action_step_prev)
        self._action_step_next = QAction("Next interval", self)
        self._action_step_next.setShortcut("Right")
        self._action_step_next.triggered.connect(lambda: self._step_interval(+1))
        view_menu.addAction(self._action_step_next)

        # Edit > Set stim/recovery boundary
        edit_menu.addSeparator()
        self._action_set_boundary = QAction("Set stim/recovery boundary…", self)
        self._action_set_boundary.triggered.connect(self._on_set_boundary)
        self._action_set_boundary.setEnabled(False)
        edit_menu.addAction(self._action_set_boundary)

        # Inference preferences (persisted via ui.data.settings)
        edit_menu.addSeparator()
        self._action_auto_run = QAction("Auto-run inference on open", self)
        self._action_auto_run.setCheckable(True)
        self._action_auto_run.setChecked(bool(self._settings.get("auto_run_on_open", True)))
        self._action_auto_run.toggled.connect(
            lambda checked: self._set_setting("auto_run_on_open", checked)
        )
        edit_menu.addAction(self._action_auto_run)

        self._action_skip_stim = QAction(
            "Skip stim region in inference (recovery only)", self,
        )
        self._action_skip_stim.setCheckable(True)
        self._action_skip_stim.setChecked(bool(self._settings.get("inference_skip_stim", True)))
        self._action_skip_stim.toggled.connect(
            lambda checked: self._set_setting("inference_skip_stim", checked)
        )
        edit_menu.addAction(self._action_skip_stim)

    def _set_setting(self, key: str, value) -> None:
        self._settings[key] = value
        ui_settings.save_settings(self._settings)

        # Tools menu — Training Management (M4) + Queue (M5).
        tools_menu = mb.addMenu("&Tools")
        self._action_training_window = QAction("Training management…", self)
        self._action_training_window.setShortcut("Ctrl+T")
        self._action_training_window.triggered.connect(self._open_training_window)
        tools_menu.addAction(self._action_training_window)
        self._action_show_queue = QAction("Recording queue", self)
        self._action_show_queue.setShortcut("Ctrl+Q")
        self._action_show_queue.setCheckable(True)
        self._action_show_queue.toggled.connect(self._on_toggle_queue_panel)
        tools_menu.addAction(self._action_show_queue)

        # Help menu
        help_menu = mb.addMenu("&Help")
        action_about = QAction("About", self)
        action_about.triggered.connect(self._on_about)
        help_menu.addAction(action_about)

        # Cached child window reference so repeated open clicks reuse
        # the same instance (and a running retrain doesn't drop).
        self._training_window: Optional[QWidget] = None

    def _open_training_window(self) -> None:
        from ui.windows.training_window import TrainingWindow
        if self._training_window is None:
            self._training_window = TrainingWindow(self)
            # If the user adds a recording or rolls back, refresh the
            # main window's model-version dropdown.
            self._training_window.manifest_or_versions_changed.connect(
                self._refresh_version_combo
            )
        self._training_window.show()
        self._training_window.raise_()
        self._training_window.activateWindow()

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setObjectName("main_toolbar")
        tb.setMovable(False)
        self.addToolBar(tb)
        tb.addAction(self._action_open)
        tb.addAction(self._action_save)
        tb.addSeparator()
        tb.addAction(self._action_reset_zoom)
        tb.addAction(self._action_fit_all)
        tb.addAction(self._action_zoom_in)
        tb.addAction(self._action_zoom_out)
        tb.addSeparator()

        # M3 inference controls
        tb.addWidget(QLabel("Model:"))
        self._version_combo = QComboBox()
        self._version_combo.setMinimumWidth(120)
        self._refresh_version_combo()
        self._version_combo.currentTextChanged.connect(
            self._on_version_changed
        )
        tb.addWidget(self._version_combo)

        self._action_run_inference = QAction("▶ Run inference", self)
        self._action_run_inference.setShortcut("Ctrl+R")
        self._action_run_inference.triggered.connect(self._on_run_inference)
        self._action_run_inference.setEnabled(False)
        tb.addAction(self._action_run_inference)

        self._action_clear_predictions = QAction("Clear predictions", self)
        self._action_clear_predictions.triggered.connect(self._on_clear_predictions)
        self._action_clear_predictions.setEnabled(False)
        tb.addAction(self._action_clear_predictions)

        self._action_review_mode = QAction("🔍 Review", self)
        self._action_review_mode.setCheckable(True)
        self._action_review_mode.toggled.connect(self._on_toggle_review_mode)
        self._action_review_mode.setEnabled(False)
        tb.addAction(self._action_review_mode)

    def _refresh_version_combo(self) -> None:
        """Repopulate the version dropdown with whatever's on disk."""
        self._version_combo.blockSignals(True)
        self._version_combo.clear()
        versions = list_available_versions()
        if not versions:
            self._version_combo.addItem("(no models)", None)
            self._version_combo.setEnabled(False)
        else:
            promoted = current_promoted_version_short()
            preferred = self._settings.get("model_version") or promoted
            for v in versions:
                # Mark the promoted version in the dropdown.
                label = f"{v} ★" if v == promoted else v
                self._version_combo.addItem(label, v)
            # Select the preferred version if present.
            if preferred:
                for i in range(self._version_combo.count()):
                    if self._version_combo.itemData(i) == preferred:
                        self._version_combo.setCurrentIndex(i)
                        break
            self._version_combo.setEnabled(True)
        self._version_combo.blockSignals(False)

    def _on_version_changed(self, text: str) -> None:
        v = self._version_combo.currentData()
        if v is None:
            return
        self._settings["model_version"] = v
        ui_settings.save_settings(self._settings)

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _on_open_clicked(self) -> None:
        if not self._maybe_discard_unsaved():
            return
        start_dir = self._settings.get("last_recording_dir") or ""
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open recording",
            start_dir,
            "Recording files (*.mat *.h5);;All files (*)",
        )
        if not path_str:
            return
        self._open_path(Path(path_str))

    def _open_path(self, path: Path) -> None:
        """Open `path`. We load TWICE because the two readers serve
        different needs (see __init__).
        """
        if not path.exists():
            QMessageBox.warning(self, "Open failed",
                                  f"File not found:\n{path}")
            return
        try:
            # 1. Lazy reader for the viewer
            lazy = LazyRecording(path)
            # 2. Eager Recording for the save flow. This loads the
            # full signal into RAM — slow on large recordings but
            # only happens once per file open. Future M2 enhancement:
            # make labeled_save accept a LazyRecording so this is
            # avoided.
            rec = load_recording(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed",
                                   f"Could not open {path}:\n{exc}")
            return

        # Close any previously-open recording's resources.
        if self._lazy is not None:
            try:
                self._lazy.close()
            except Exception:
                pass

        self._lazy = lazy
        self._recording = rec

        # Replace the central widget with a new viewer for this file.
        self._viewer = MultiChannelViewer(lazy)
        self.setCentralWidget(self._viewer)
        self._viewer.bad_interval_added.connect(self._on_user_marked_region)
        self._viewer.stim_boundary_moved.connect(self._on_boundary_moved)

        # Build / re-build the bottom overview strip.
        if self._overview is not None:
            self.removeDockWidget(self._overview_dock)
        self._overview = OverviewStrip(lazy)
        self._overview_dock = QDockWidget("Overview", self)
        self._overview_dock.setObjectName("overview_dock")
        self._overview_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self._overview_dock.setWidget(self._overview)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._overview_dock)
        self._overview.viewport_changed.connect(self._on_overview_dragged)

        # Wire viewer↔overview viewport-tracking.
        self._viewer.plots[0][0].sigRangeChanged.connect(
            self._sync_overview_from_viewer
        )

        # Seed edit state from the recording's loaded `existing_bad_intervals`.
        if rec.existing_bad_intervals is not None and len(rec.existing_bad_intervals):
            self._bad_intervals = np.asarray(
                rec.existing_bad_intervals, dtype=np.int64,
            )
            self._bad_sources = ["existing"] * int(self._bad_intervals.shape[0])
        else:
            self._bad_intervals = np.zeros((0, 2), dtype=np.int64)
            self._bad_sources = []
        self._stim_end_idx = rec.stim_end_idx
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._dirty = False

        # Clear M3 inference state from any previous file.
        self._on_clear_predictions()

        self._refresh_widgets()
        self._action_save.setEnabled(True)
        self._action_save_as.setEnabled(True)
        self._action_set_boundary.setEnabled(rec.rec_type == "stim_rec")
        self._action_run_inference.setEnabled(
            self._version_combo.currentData() is not None
        )
        self._update_status_bar()
        self.setWindowTitle(f"Detector — PyQt   |   {path.name}")

        # Remember the directory for the next Open dialog.
        self._settings["last_recording_dir"] = str(path.parent)
        ui_settings.save_settings(self._settings)

        # M3: auto-run inference on open if the user has it enabled.
        # Fires once per file open. The latch is the simple
        # _auto_run_attempted_for path that records what we've already
        # tried so a re-open of the SAME file doesn't loop.
        if (
            self._settings.get("auto_run_on_open", True)
            and self._version_combo.currentData() is not None
        ):
            QTimer.singleShot(50, self._on_run_inference)

    # ------------------------------------------------------------------
    # Region mutations (with undo)
    # ------------------------------------------------------------------

    def _push_undo(self) -> None:
        snap = (self._bad_intervals.copy(), list(self._bad_sources), self._stim_end_idx)
        self._undo_stack.append(snap)
        if len(self._undo_stack) > UNDO_DEPTH:
            self._undo_stack.pop(0)
        # New action invalidates the redo stack.
        self._redo_stack.clear()
        self._action_undo.setEnabled(True)
        self._action_redo.setEnabled(False)

    def _on_undo(self) -> None:
        if not self._undo_stack:
            return
        cur = (self._bad_intervals.copy(), list(self._bad_sources), self._stim_end_idx)
        self._redo_stack.append(cur)
        intervals, sources, stim_idx = self._undo_stack.pop()
        self._bad_intervals = intervals
        self._bad_sources = sources
        self._stim_end_idx = stim_idx
        self._refresh_widgets()
        self._action_undo.setEnabled(bool(self._undo_stack))
        self._action_redo.setEnabled(True)

    def _on_redo(self) -> None:
        if not self._redo_stack:
            return
        cur = (self._bad_intervals.copy(), list(self._bad_sources), self._stim_end_idx)
        self._undo_stack.append(cur)
        intervals, sources, stim_idx = self._redo_stack.pop()
        self._bad_intervals = intervals
        self._bad_sources = sources
        self._stim_end_idx = stim_idx
        self._refresh_widgets()
        self._action_undo.setEnabled(True)
        self._action_redo.setEnabled(bool(self._redo_stack))

    def _on_user_marked_region(self, start_sec: float, end_sec: float) -> None:
        if self._recording is None:
            return
        fs = self._recording.fs
        n_samples = self._recording.n_samples
        s = max(1, int(np.floor(start_sec * fs)) + 1)
        e = min(int(n_samples), int(np.ceil(end_sec * fs)) + 1)
        if e <= s:
            return
        self._push_undo()
        new = np.array([[s, e]], dtype=np.int64)
        combined = (np.concatenate([self._bad_intervals, new], axis=0)
                    if self._bad_intervals.size else new)
        order = np.argsort(combined[:, 0], kind="stable")
        self._bad_intervals = combined[order]
        sources_combined = list(self._bad_sources) + ["user"]
        self._bad_sources = [sources_combined[int(i)] for i in order]
        self._dirty = True
        self._refresh_widgets()

    def _on_delete_interval(self, idx: int) -> None:
        if idx < 0 or idx >= self._bad_intervals.shape[0]:
            return
        self._push_undo()
        self._bad_intervals = np.delete(self._bad_intervals, idx, axis=0)
        self._bad_sources = [
            s for i, s in enumerate(self._bad_sources) if i != idx
        ]
        self._dirty = True
        self._refresh_widgets()

    def _on_jump_to_interval(self, idx: int) -> None:
        if idx < 0 or idx >= self._bad_intervals.shape[0] or self._viewer is None:
            return
        s, e = self._bad_intervals[idx]
        fs = self._recording.fs
        target = ((int(s) + int(e)) / 2 - 1) / fs
        cur_s, cur_e = self._viewer.time_range
        width = max(cur_e - cur_s, 1.0)
        new_s = max(0.0, target - width / 2)
        new_e = min(self._recording.duration_sec, new_s + width)
        self._viewer.set_viewport(new_s, new_e)

    def _on_boundary_moved(self, new_idx: int) -> None:
        if self._stim_end_idx != new_idx:
            self._push_undo()
            self._stim_end_idx = new_idx
            self._dirty = True
            self._update_status_bar()

    def _on_set_boundary(self) -> None:
        if self._recording is None:
            return
        cur_sec = ((self._stim_end_idx - 1) / self._recording.fs
                    if self._stim_end_idx else 0.0)
        new_sec, ok = QInputDialog.getDouble(
            self, "Stim/recovery boundary",
            "Stim end (seconds):",
            cur_sec, 0.0, self._recording.duration_sec, 3,
        )
        if not ok:
            return
        self._push_undo()
        self._stim_end_idx = max(
            1, min(self._recording.n_samples,
                    int(round(new_sec * self._recording.fs)) + 1)
        )
        self._dirty = True
        self._refresh_widgets()

    # ------------------------------------------------------------------
    # View actions
    # ------------------------------------------------------------------

    def _on_reset_zoom(self) -> None:
        if self._viewer is None:
            return
        end = min(60.0, self._recording.duration_sec)
        self._viewer.set_viewport(0.0, end)

    def _on_fit_all(self) -> None:
        if self._viewer is None:
            return
        self._viewer.set_viewport(0.0, self._recording.duration_sec)

    def _zoom(self, factor: float) -> None:
        if self._viewer is None:
            return
        s, e = self._viewer.time_range
        center = (s + e) / 2
        width = (e - s) * factor
        new_s = max(0.0, center - width / 2)
        new_e = min(self._recording.duration_sec, new_s + width)
        self._viewer.set_viewport(new_s, new_e)

    def _step_interval(self, direction: int) -> None:
        """Step the viewport to the previous (-1) or next (+1) bad
        interval. Centres the viewport on it preserving current width.
        """
        if self._bad_intervals.shape[0] == 0 or self._viewer is None:
            return
        cur_s, cur_e = self._viewer.time_range
        cur_center = (cur_s + cur_e) / 2
        fs = self._recording.fs
        centers_sec = (
            (self._bad_intervals[:, 0].astype(np.float64)
             + self._bad_intervals[:, 1].astype(np.float64))
            / 2.0 - 1
        ) / fs
        if direction > 0:
            candidates = np.where(centers_sec > cur_center + 1e-3)[0]
            if len(candidates) == 0:
                return
            target_idx = int(candidates[0])
        else:
            candidates = np.where(centers_sec < cur_center - 1e-3)[0]
            if len(candidates) == 0:
                return
            target_idx = int(candidates[-1])
        self._on_jump_to_interval(target_idx)
        self._region_table.selectRow(target_idx)

    def _sync_overview_from_viewer(self, _view_box, ranges) -> None:
        """Viewer panned/zoomed → mirror in overview's viewport indicator."""
        if self._overview is None or self._recording is None:
            return
        x_min, x_max = ranges[0]
        x_min = max(0.0, float(x_min))
        x_max = min(self._recording.duration_sec, float(x_max))
        if x_max > x_min:
            self._overview.set_viewport_range(x_min, x_max)

    def _on_overview_dragged(self, t_start: float, t_end: float) -> None:
        """Overview region dragged → mirror in viewer."""
        if self._viewer is None:
            return
        self._viewer.set_viewport(t_start, t_end)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _on_save_clicked(self) -> None:
        if self._recording is None:
            return
        # Default output paths next to the source recording.
        base = self._recording.source_path
        clean_path = base.with_name(f"{self._recording.recording_id}_clean.h5")
        mat_path = base.with_name(f"{self._recording.recording_id}_blankmotion.mat")
        seg_path = base.with_name(f"{self._recording.recording_id}_segments.json")
        self._save_to(clean_path, mat_path, seg_path)

    def _on_save_as_clicked(self) -> None:
        if self._recording is None:
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save as — choose the base path",
            str(self._recording.source_path.with_name(
                f"{self._recording.recording_id}_clean.h5"
            )),
            "Splitter HDF5 (*_clean.h5);;All files (*)",
        )
        if not path_str:
            return
        clean_path = Path(path_str)
        stem = clean_path.stem
        if stem.endswith("_clean"):
            stem = stem[: -len("_clean")]
        mat_path = clean_path.with_name(f"{stem}_blankmotion.mat")
        seg_path = clean_path.with_name(f"{stem}_segments.json")
        self._save_to(clean_path, mat_path, seg_path)

    def _save_to(self, clean_path: Path, mat_path: Path, seg_path: Path) -> None:
        rec = self._recording
        intervals = self._bad_intervals
        sources = self._bad_sources
        # Anything still in predictions at save time is the
        # "model_unsure" set — predictions the user didn't accept or
        # dismiss explicitly. Streamlit Phase 8 parity: include them
        # in the .mat as `removedSegmentIdx_model_unsure` and in the
        # segment table JSON as `segments_unsure`.
        model_unsure = (
            self._model_intervals if (self._model_intervals is not None
                                      and self._model_intervals.size > 0)
            else None
        )
        try:
            r_native = save_native(
                rec, intervals,
                output_clean_path=clean_path,
                stim_end_idx=self._stim_end_idx,
            )
            r_mat = save_matlab_compatible(
                rec, intervals, mat_path,
                stim_end_idx=self._stim_end_idx,
                bad_sources=sources,
                model_unsure_intervals=model_unsure,
                write_yout=True,
            )
            r_seg = save_segment_table(
                rec, intervals, sources,
                output_path=seg_path,
                model_unsure_intervals=model_unsure,
                stim_end_idx=self._stim_end_idx,
                model_version=self._model_version,
                threshold_used=self._model_threshold_used,
            )
            # Per-period split if a boundary is set.
            split_msg = ""
            if self._stim_end_idx is not None and rec.rec_type == "stim_rec":
                r_split = save_period_split(
                    rec, intervals, output_path_base=mat_path,
                    stim_end_idx=self._stim_end_idx,
                    bad_sources=sources,
                    model_unsure_intervals=model_unsure,
                    write_yout=True,
                )
                split_msg = (
                    f"  ·  split: stim={r_split['stim_n_intervals']} + "
                    f"recovery={r_split['recovery_n_intervals']}"
                )
            unsure_msg = (
                f"  ·  unsure={int(model_unsure.shape[0])}"
                if model_unsure is not None else ""
            )
            self._dirty = False
            self.statusBar().showMessage(
                f"saved: clean.h5={r_native['n_clean_chunks']} chunks, "
                f".mat={r_mat['n_intervals']} intervals "
                f"(user={r_mat['n_user']}, "
                f"model_accepted={r_mat['n_model_accepted']}, "
                f"existing={r_mat['n_existing']}){unsure_msg}{split_msg}",
                12_000,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", f"{exc}")

    # ------------------------------------------------------------------
    # House-keeping
    # ------------------------------------------------------------------

    def _refresh_widgets(self) -> None:
        """Push the current edit state into the viewer + table + overview."""
        if self._viewer is not None:
            self._viewer.set_bad_intervals(self._bad_intervals)
            self._viewer.set_stim_boundary(self._stim_end_idx)
            self._viewer.set_model_intervals(self._model_intervals)
        if self._overview is not None:
            self._overview.set_bad_intervals(self._bad_intervals)
            self._overview.set_stim_boundary(self._stim_end_idx)
            self._overview.set_model_intervals(self._model_intervals)
        if self._recording is not None:
            self._region_table.set_intervals(
                self._bad_intervals, self._bad_sources, self._recording.fs,
            )
        # Tab labels reflect counts even without an inference run.
        n_marked = int(self._bad_intervals.shape[0])
        n_pred = (int(self._model_intervals.shape[0])
                   if self._model_intervals is not None else 0)
        self._right_tabs.setTabText(0, f"🟥 Marked ({n_marked})")
        self._right_tabs.setTabText(1, f"🟧 Model ({n_pred})")
        self._update_status_bar()

    def _update_status_bar(self) -> None:
        if self._recording is None:
            self.statusBar().showMessage("Open a recording to start.")
            return
        n_bad = self._bad_intervals.shape[0]
        total_bad_samples = (
            int(np.sum(self._bad_intervals[:, 1] - self._bad_intervals[:, 0] + 1))
            if n_bad else 0
        )
        bad_frac = total_bad_samples / max(self._recording.n_samples, 1)
        bd = (f"  ·  stim_end={self._stim_end_idx}"
              if self._stim_end_idx else "")
        dirty = "  ·  ●" if self._dirty else ""
        self.statusBar().showMessage(
            f"{self._recording.recording_id}  ·  "
            f"fs={self._recording.fs:.1f} Hz  ·  "
            f"{self._recording.n_samples:,} samples "
            f"({self._recording.duration_sec:.1f}s)  ·  "
            f"marked: {n_bad} ({bad_frac:.3f}){bd}{dirty}"
        )

    def _maybe_discard_unsaved(self) -> bool:
        """Returns True to proceed (no unsaved data, or user said yes
        to discard); False to cancel the action."""
        if not self._dirty:
            return True
        resp = QMessageBox.question(
            self, "Unsaved changes",
            "You have unsaved changes. Discard them?",
            QMessageBox.Discard | QMessageBox.Cancel,
        )
        return resp == QMessageBox.Discard

    def closeEvent(self, event) -> None:
        if self._maybe_discard_unsaved():
            event.accept()
        else:
            event.ignore()

    # ------------------------------------------------------------------
    # M3 — inference
    # ------------------------------------------------------------------

    def _on_run_inference(self) -> None:
        """Start a background inference job. UI disables the Run
        Inference action and shows a QProgressDialog until the worker
        emits finished or error.
        """
        if self._recording is None or self._lazy is None:
            return
        if self._inference_thread is not None and self._inference_thread.isRunning():
            QMessageBox.information(
                self, "Inference busy",
                "Inference is already running. Wait for it to finish.",
            )
            return
        version = self._version_combo.currentData()
        if version is None:
            QMessageBox.warning(
                self, "No model",
                "No model artifacts available. Configure the shared "
                "model folder via `detector init` (CLI).",
            )
            return
        try:
            artifact = load_model_artifact_cached(version)
        except Exception as exc:
            QMessageBox.critical(
                self, "Failed to load model",
                f"Could not load {version}:\n{exc}",
            )
            return

        # Determine input + stim-skip offset. The Recording dataclass
        # holds the in-RAM signal so we slice cheaply.
        skip_stim = bool(
            self._settings.get("inference_skip_stim", True)
            and self._stim_end_idx
            and self._recording.rec_type == "stim_rec"
        )
        if skip_stim:
            offset = int(self._stim_end_idx)
            y_in = self._recording.y[offset:, :]
        else:
            offset = 0
            y_in = self._recording.y

        self._inference_thread = QThread()
        self._inference_worker = InferenceWorker(
            y_in, self._recording.fs, artifact,
            stim_skip_offset=offset,
            return_features=True,
        )
        self._inference_worker.moveToThread(self._inference_thread)
        self._inference_thread.started.connect(self._inference_worker.run)
        self._inference_worker.progress.connect(self._on_inference_progress)
        self._inference_worker.finished.connect(self._on_inference_finished)
        self._inference_worker.error.connect(self._on_inference_error)
        # Clean up the thread when the worker finishes or errors.
        self._inference_worker.finished.connect(self._inference_thread.quit)
        self._inference_worker.error.connect(self._inference_thread.quit)
        self._inference_thread.finished.connect(
            self._inference_worker.deleteLater
        )
        self._inference_thread.finished.connect(
            self._inference_thread.deleteLater
        )

        # Progress dialog
        self._inference_progress = QProgressDialog(
            "Running inference …", "Cancel", 0, 100, self,
        )
        self._inference_progress.setWindowModality(Qt.WindowModal)
        self._inference_progress.setMinimumDuration(0)
        # Cancel is best-effort — the worker doesn't poll a flag, so
        # cancel just hides the dialog and stops listening. Inference
        # finishes either way.
        self._inference_progress.canceled.connect(
            lambda: self._action_run_inference.setEnabled(True)
        )
        self._inference_progress.setValue(0)
        self._action_run_inference.setEnabled(False)

        self._inference_thread.start()
        self.statusBar().showMessage(
            f"running inference with model {version} "
            f"(scope = {'recovery_only' if offset else 'full'}) …"
        )

    def _on_inference_progress(self, frac: float, msg: str) -> None:
        if self._inference_progress is not None:
            self._inference_progress.setValue(int(frac * 100))
            self._inference_progress.setLabelText(f"{msg} ({frac:.0%})")

    def _on_inference_finished(self, result: dict) -> None:
        if self._inference_progress is not None:
            self._inference_progress.setValue(100)
            self._inference_progress.close()
            self._inference_progress = None
        self._action_run_inference.setEnabled(True)
        # Store + integrate
        self._model_intervals = result.get("intervals")
        self._model_per_window_probs = result.get("per_window_prob")
        self._model_position_samples = result.get("position_samples")
        self._model_features = result.get("features")
        self._model_version = result.get("model_version")
        self._model_threshold_used = result.get("threshold_used")
        self._action_clear_predictions.setEnabled(True)
        self._action_review_mode.setEnabled(True)
        self._refresh_predictions_panel()
        self._refresh_widgets()
        n = int(self._model_intervals.shape[0]) if self._model_intervals is not None else 0
        scope_note = (" · scope = recovery-only"
                       if result.get("scope_offset_sample") else "")
        self.statusBar().showMessage(
            f"inference done: {n} predictions · "
            f"thr={result.get('threshold_used'):.3f}{scope_note}",
            8_000,
        )

    def _on_inference_error(self, message: str) -> None:
        if self._inference_progress is not None:
            self._inference_progress.close()
            self._inference_progress = None
        self._action_run_inference.setEnabled(True)
        QMessageBox.critical(self, "Inference failed", message)

    def _on_clear_predictions(self) -> None:
        self._model_intervals = None
        self._model_per_window_probs = None
        self._model_position_samples = None
        self._model_features = None
        self._model_threshold_used = None
        self._action_clear_predictions.setEnabled(False)
        self._action_review_mode.setEnabled(False)
        if self._action_review_mode.isChecked():
            self._action_review_mode.setChecked(False)
        self._refresh_predictions_panel()
        self._refresh_widgets()

    def _refresh_predictions_panel(self) -> None:
        """Sync the right-dock model tab + per-prediction probabilities."""
        # Compute per-interval probability — the MAX per-window prob
        # whose position_sample falls within the interval. (Streamlit
        # Phase 8 used the same heuristic for display.)
        per_int_probs: Optional[np.ndarray] = None
        if (
            self._model_intervals is not None
            and self._model_per_window_probs is not None
            and self._model_position_samples is not None
        ):
            probs = np.zeros(int(self._model_intervals.shape[0]),
                              dtype=np.float32)
            for i, (s, e) in enumerate(self._model_intervals):
                mask = (
                    (self._model_position_samples >= int(s))
                    & (self._model_position_samples <= int(e))
                )
                if mask.any():
                    probs[i] = float(self._model_per_window_probs[mask].max())
            per_int_probs = probs
        fs = self._recording.fs if self._recording else 1.0
        self._predictions_panel.set_predictions(
            self._model_intervals if self._model_intervals is not None else
            np.zeros((0, 2), dtype=np.int64),
            per_int_probs, fs,
        )
        # Update tab labels with counts
        n_marked = int(self._bad_intervals.shape[0])
        n_pred = (int(self._model_intervals.shape[0])
                   if self._model_intervals is not None else 0)
        self._right_tabs.setTabText(0, f"🟥 Marked ({n_marked})")
        self._right_tabs.setTabText(1, f"🟧 Model ({n_pred})")

    # ----- Prediction action handlers -----

    def _on_accept_prediction(self, idx: int) -> None:
        if self._model_intervals is None or idx < 0 or idx >= self._model_intervals.shape[0]:
            return
        s, e = self._model_intervals[idx]
        fs = self._recording.fs
        self._push_undo()
        new = np.array([[int(s), int(e)]], dtype=np.int64)
        combined = (np.concatenate([self._bad_intervals, new], axis=0)
                    if self._bad_intervals.size else new)
        order = np.argsort(combined[:, 0], kind="stable")
        self._bad_intervals = combined[order]
        sources_combined = list(self._bad_sources) + ["model_accepted"]
        self._bad_sources = [sources_combined[int(i)] for i in order]
        # Remove from predictions
        self._model_intervals = np.delete(self._model_intervals, idx, axis=0)
        self._dirty = True
        self._refresh_predictions_panel()
        self._refresh_widgets()

    def _on_dismiss_prediction(self, idx: int) -> None:
        if self._model_intervals is None or idx < 0 or idx >= self._model_intervals.shape[0]:
            return
        self._model_intervals = np.delete(self._model_intervals, idx, axis=0)
        self._refresh_predictions_panel()
        self._refresh_widgets()

    def _on_jump_to_prediction(self, idx: int) -> None:
        if self._model_intervals is None or idx < 0 or idx >= self._model_intervals.shape[0]:
            return
        s, e = self._model_intervals[idx]
        fs = self._recording.fs
        target = ((int(s) + int(e)) / 2 - 1) / fs
        cur_s, cur_e = self._viewer.time_range
        width = max(cur_e - cur_s, 1.0)
        new_s = max(0.0, target - width / 2)
        new_e = min(self._recording.duration_sec, new_s + width)
        self._viewer.set_viewport(new_s, new_e)

    def _on_accept_all_predictions(self) -> None:
        if self._model_intervals is None or self._model_intervals.size == 0:
            return
        self._push_undo()
        new = np.asarray(self._model_intervals, dtype=np.int64)
        combined = (np.concatenate([self._bad_intervals, new], axis=0)
                    if self._bad_intervals.size else new)
        order = np.argsort(combined[:, 0], kind="stable")
        self._bad_intervals = combined[order]
        sources_combined = list(self._bad_sources) + ["model_accepted"] * int(new.shape[0])
        self._bad_sources = [sources_combined[int(i)] for i in order]
        self._model_intervals = np.zeros((0, 2), dtype=np.int64)
        self._dirty = True
        self._refresh_predictions_panel()
        self._refresh_widgets()

    # ----- Review-mode toggle -----

    def _on_toggle_review_mode(self, checked: bool) -> None:
        if not checked:
            self._close_review_panel()
            return
        if (
            self._model_intervals is None
            or self._model_features is None
            or self._lazy is None
            or self._recording is None
        ):
            self._action_review_mode.setChecked(False)
            return
        # Build disagreements list using detector.review.
        try:
            artifact = load_model_artifact_cached(self._model_version)
            n_short = int(round(0.100 * self._recording.fs))
            half = n_short // 2
            positions = self._model_features["position_sample"].to_numpy(np.int64)
            labels = np.zeros(len(self._model_features), dtype=np.uint8)
            for s, e in self._bad_intervals:
                in_hum = (
                    (positions >= int(s) + half)
                    & (positions <= int(e) - half)
                )
                labels[in_hum] = 1
            disagreements = detector_review.extract_disagreements(
                recording_id=self._recording.recording_id,
                df_features=self._model_features,
                probs=self._model_per_window_probs,
                labels=labels,
                threshold=float(artifact.threshold),
                fs=self._recording.fs,
                n_total_samples=self._recording.n_samples,
                top_k=20,
                context_seconds=2.0,
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Review setup failed", f"{exc}",
            )
            self._action_review_mode.setChecked(False)
            return

        if not disagreements:
            QMessageBox.information(
                self, "No disagreements",
                "No predicted-bad windows fall outside your marked "
                "intervals. Nothing to review.",
            )
            self._action_review_mode.setChecked(False)
            return

        self._review_panel = ReviewPanel(
            self._lazy, artifact, disagreements, self._model_features,
            parent=self,
        )
        self._review_panel.mark_wider_requested.connect(
            self._on_review_mark_wider
        )
        self._review_panel.save_requested.connect(self._on_save_review_json)
        self._review_panel.exit_requested.connect(
            lambda: self._action_review_mode.setChecked(False)
        )
        self._review_dock = QDockWidget("Disagreement review", self)
        self._review_dock.setObjectName("review_dock")
        self._review_dock.setWidget(self._review_panel)
        self._review_dock.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )
        self._review_dock.visibilityChanged.connect(
            self._on_review_dock_visibility
        )
        # Place on the right; tabify with the intervals dock so the
        # user can flip between them.
        self.addDockWidget(Qt.RightDockWidgetArea, self._review_dock)
        self.tabifyDockWidget(self._table_dock, self._review_dock)
        self._review_dock.raise_()

    def _close_review_panel(self) -> None:
        if self._review_dock is not None:
            self.removeDockWidget(self._review_dock)
            self._review_dock.deleteLater()
            self._review_dock = None
            self._review_panel = None

    def _on_review_dock_visibility(self, visible: bool) -> None:
        # If the user closes the dock via its X, keep the toolbar
        # checkbox in sync.
        if not visible and self._action_review_mode.isChecked():
            self._action_review_mode.setChecked(False)

    def _on_review_mark_wider(
        self, start_sec: float, end_sec: float, position_sample: int,
    ) -> None:
        """User drag-marked a wider region in the review panel — apply
        the same three-step handling Streamlit had: append as user
        interval, drop the matching model prediction, advance."""
        self._on_user_marked_region_with_source(start_sec, end_sec, "user")
        # Remove model prediction containing position_sample
        if self._model_intervals is not None and len(self._model_intervals):
            mi = np.asarray(self._model_intervals, dtype=np.int64)
            keep = ~((mi[:, 0] <= position_sample) & (mi[:, 1] >= position_sample))
            self._model_intervals = (
                mi[keep] if keep.any() else np.zeros((0, 2), dtype=np.int64)
            )
            self._refresh_predictions_panel()
        if self._review_panel is not None:
            self._review_panel.remove_current_after_mark()
        self._refresh_widgets()

    def _on_user_marked_region_with_source(
        self, start_sec: float, end_sec: float, source: str,
    ) -> None:
        if self._recording is None:
            return
        fs = self._recording.fs
        n_samples = self._recording.n_samples
        s = max(1, int(np.floor(start_sec * fs)) + 1)
        e = min(int(n_samples), int(np.ceil(end_sec * fs)) + 1)
        if e <= s:
            return
        self._push_undo()
        new = np.array([[s, e]], dtype=np.int64)
        combined = (np.concatenate([self._bad_intervals, new], axis=0)
                    if self._bad_intervals.size else new)
        order = np.argsort(combined[:, 0], kind="stable")
        self._bad_intervals = combined[order]
        sources_combined = list(self._bad_sources) + [source]
        self._bad_sources = [sources_combined[int(i)] for i in order]
        self._dirty = True

    def _on_save_review_json(self) -> None:
        if self._review_panel is None or self._recording is None:
            return
        out_path = (
            detector_paths.get_reviews_dir()
            / f"{self._recording.recording_id}_review.json"
        )
        try:
            written = self._review_panel.write_review_json(
                out_path,
                model_version=self._model_version or "(unknown)",
                threshold_used=self._model_threshold_used or 0.5,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save review failed", f"{exc}")
            return
        self.statusBar().showMessage(f"review JSON → {written}", 6_000)

    # ------------------------------------------------------------------
    # M5 — queue workflow
    # ------------------------------------------------------------------

    def _on_toggle_queue_panel(self, checked: bool) -> None:
        if checked:
            self._show_queue_panel()
        else:
            self._hide_queue_panel()

    def _show_queue_panel(self) -> None:
        if getattr(self, "_queue_dock", None) is not None:
            self._queue_dock.show()
            return
        from ui.widgets.queue_panel import QueuePanel
        self._queue_panel = QueuePanel()
        self._queue_panel.open_recording.connect(self._on_queue_open_recording)
        self._queue_panel.bulk_inference_requested.connect(self._on_bulk_inference)
        self._queue_dock = QDockWidget("Recording queue", self)
        self._queue_dock.setObjectName("queue_dock")
        self._queue_dock.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )
        self._queue_dock.setWidget(self._queue_panel)
        self._queue_dock.visibilityChanged.connect(self._on_queue_dock_visibility)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._queue_dock)

    def _hide_queue_panel(self) -> None:
        if getattr(self, "_queue_dock", None) is None:
            return
        self.removeDockWidget(self._queue_dock)
        self._queue_dock.deleteLater()
        self._queue_dock = None
        self._queue_panel = None

    def _on_queue_dock_visibility(self, visible: bool) -> None:
        if not visible and self._action_show_queue.isChecked():
            self._action_show_queue.setChecked(False)

    def _on_queue_open_recording(self, path: str) -> None:
        """User clicked a queue row — load that recording into the
        main viewer. Save-on-close will be wired to mark "done"."""
        if not self._maybe_discard_unsaved():
            return
        self._open_path(Path(path))

    def _on_bulk_inference(self) -> None:
        """Sequentially run inference on all queue items with
        status='pending'. Results land alongside each recording as
        `<rid>_modelblank.h5` (via detector.predict.detect_bad's
        normal output path). Progress dialog tracks per-file."""
        if getattr(self, "_queue_panel", None) is None:
            return
        pending = [
            it for it in self._queue_panel.queue.items
            if it.status == "pending"
        ]
        if not pending:
            return
        # Build a sequential plan and walk through it via a small
        # state machine (QTimer-driven so we yield to the event loop
        # between recordings).
        self._bulk_queue: list = list(pending)
        self._bulk_total = len(pending)
        self._bulk_done = 0
        self._bulk_progress = QProgressDialog(
            f"Bulk inference: {self._bulk_total} recordings",
            "Cancel", 0, self._bulk_total, self,
        )
        self._bulk_progress.setWindowModality(Qt.WindowModal)
        self._bulk_progress.setMinimumDuration(0)
        self._bulk_progress.canceled.connect(self._on_bulk_inference_cancel)
        self._bulk_cancelled = False
        self._bulk_step()

    def _bulk_step(self) -> None:
        """Process the next queue item."""
        if self._bulk_cancelled or not self._bulk_queue:
            self._on_bulk_inference_done()
            return
        item = self._bulk_queue.pop(0)
        path = Path(item.path)
        self._bulk_progress.setLabelText(
            f"[{self._bulk_done + 1}/{self._bulk_total}] {path.name}"
        )
        # We open the recording into the main viewer + auto-fire
        # inference, then on finish we mark + step.
        try:
            self._open_path(path)
        except Exception as exc:
            QMessageBox.warning(
                self, "Bulk inference",
                f"Skipping {path.name}: {exc}",
            )
            self._bulk_done += 1
            self._bulk_progress.setValue(self._bulk_done)
            QTimer.singleShot(0, self._bulk_step)
            return

        # Hook a one-shot listener for the inference finish.
        if self._inference_worker is None:
            # Auto-run is OFF? Force a manual run.
            QTimer.singleShot(50, self._on_run_inference)
        # Wait for finish via the worker's `finished` signal.
        # We re-connect each step because the worker is replaced
        # each run.
        def _hook(_result):
            try:
                if self._recording is not None:
                    # Save the predictions so they persist.
                    self._on_save_clicked()
                # Update queue status
                if getattr(self, "_queue_panel", None) is not None:
                    self._queue_panel.mark_status(item.path, "done")
            finally:
                self._bulk_done += 1
                self._bulk_progress.setValue(self._bulk_done)
                # The next step runs after the event loop yields so the
                # progress dialog can repaint.
                QTimer.singleShot(50, self._bulk_step)

        # Wait a tick for the inference worker to spawn (it's set up
        # inside _open_path's auto-run logic via QTimer.singleShot).
        def _wait_for_worker():
            if self._inference_worker is None:
                # Auto-run disabled — call _on_run_inference directly.
                self._on_run_inference()
            if self._inference_worker is not None:
                self._inference_worker.finished.connect(_hook)
                # If error, treat as done-with-warning and continue.
                self._inference_worker.error.connect(
                    lambda msg: (self._bulk_skip_with_warning(item.path, msg))
                )
            else:
                # Couldn't start; skip
                self._bulk_done += 1
                self._bulk_progress.setValue(self._bulk_done)
                QTimer.singleShot(50, self._bulk_step)

        QTimer.singleShot(100, _wait_for_worker)

    def _bulk_skip_with_warning(self, path: str, msg: str) -> None:
        QMessageBox.warning(self, "Bulk inference",
                              f"{Path(path).name} failed: {msg}")
        self._bulk_done += 1
        self._bulk_progress.setValue(self._bulk_done)
        QTimer.singleShot(50, self._bulk_step)

    def _on_bulk_inference_cancel(self) -> None:
        self._bulk_cancelled = True

    def _on_bulk_inference_done(self) -> None:
        if self._bulk_progress is not None:
            self._bulk_progress.close()
            self._bulk_progress = None
        self.statusBar().showMessage(
            f"bulk inference: processed {self._bulk_done}/{self._bulk_total} "
            f"recording(s)",
            10_000,
        )

    def _on_about(self) -> None:
        QMessageBox.information(
            self, "About",
            "Detector — PyQt UI\n\n"
            "Phase M2 (browser + manual labeling).\n"
            "Plotly-free, GPU-friendly via pyqtgraph.\n\n"
            "Shift+drag to mark a bad region.\n"
            "Drag the overview region or click to navigate.\n"
        )
