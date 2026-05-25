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
    QComboBox, QDockWidget, QFileDialog, QInputDialog, QLabel,
    QMainWindow, QMessageBox, QProgressDialog, QPushButton, QStatusBar,
    QTabWidget, QToolBar, QWidget,
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
from ui.workers.heldout_eval_worker import (
    HeldoutEvalWorker, HeldoutEvalMultiModelWorker,
)
from ui.workers.baseline_worker import BaselineWorker


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
        # 1-based channel index where the cleanest heartbeat was
        # observed — written as `hrChanIdx` into the saved .mat
        # (mirrors the manual MATLAB pipeline). None = don't write
        # the field. User sets this via Edit → Set best heartbeat
        # channel…. Default is 1 (first nerve channel) but the
        # user is expected to confirm + override per recording.
        self._hr_chan_idx: Optional[int] = None
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
        # Held-out evaluation -- separate worker/thread/dialog refs so
        # the user can still use the labeling UI while it runs in the
        # background.
        self._heldout_thread: Optional[QThread] = None
        self._heldout_worker: Optional[HeldoutEvalWorker] = None
        self._heldout_progress: Optional[QProgressDialog] = None
        # Multi-model variant uses its own slots so a single-model and
        # multi-model run can't collide. Two separate "trains" are
        # never queued concurrently because the menu actions get
        # disabled while a run is in flight.
        self._heldout_multi_thread: Optional[QThread] = None
        self._heldout_multi_worker: Optional[HeldoutEvalMultiModelWorker] = None
        self._heldout_multi_progress: Optional[QProgressDialog] = None
        # Background baseline.h5 computation (runs after Save). Only
        # one in flight at a time — a second save while the prior
        # baseline is still computing is blocked at _save_to entry
        # rather than queued, since they'd race on the same on-disk
        # baseline.h5 path.
        self._baseline_thread: Optional[QThread] = None
        self._baseline_worker: Optional[BaselineWorker] = None
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

        # Folder-picker sibling to Open recording. Qt has no single
        # native dialog that lets the user pick EITHER a file OR a
        # folder, so we expose folder selection separately. Routes
        # through _handle_tdt_folder_open: load outputs if already
        # preprocessed, else seed the Preprocess window.
        self._action_open_tdt_folder = QAction("Open &TDT folder…", self)
        self._action_open_tdt_folder.setShortcut(
            QKeySequence("Ctrl+Shift+O")
        )
        self._action_open_tdt_folder.triggered.connect(
            self._on_open_tdt_folder_clicked
        )
        file_menu.addAction(self._action_open_tdt_folder)

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
        # Preprocessing workflow entry point. Opens the
        # PreprocessWindow modally; the user picks TDT folders,
        # configures the batch, reviews channels + notch, and runs
        # the pipeline. Outputs go alongside the source TDT folders,
        # which the user can then open via this same File menu.
        self._action_preprocess = QAction("&Preprocess TDT data…", self)
        self._action_preprocess.triggered.connect(self._on_preprocess_clicked)
        file_menu.addAction(self._action_preprocess)

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

        # Edit > Set best heartbeat channel — writes `hrChanIdx`
        # into the saved .mat, matching the manual MATLAB pipeline.
        self._action_set_hr_chan = QAction(
            "Set best heartbeat channel…", self,
        )
        self._action_set_hr_chan.triggered.connect(self._on_set_hr_chan)
        self._action_set_hr_chan.setEnabled(False)
        edit_menu.addAction(self._action_set_hr_chan)

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
        # Held-out evaluation: runs the current model against every
        # manifest entry tagged held_out=True (which are excluded from
        # training), reports per-sample model-vs-human agreement. The
        # honest generalization metric: "how well does the model do
        # on animals/recordings it has never seen?"
        tools_menu.addSeparator()
        self._action_heldout_eval = QAction("Held-out evaluation…", self)
        self._action_heldout_eval.triggered.connect(self._run_heldout_eval)
        tools_menu.addAction(self._action_heldout_eval)
        # Multi-model comparison: runs 2-3 models against the same
        # held-out set in one pass (parallelized across recordings).
        # Same metrics as single-model held-out eval, side-by-side
        # per model, plus a per-recording table and an overlay viewer
        # whose model dropdown swaps the predicted-bad band instantly.
        self._action_heldout_multi = QAction(
            "Compare models on held-out…", self,
        )
        self._action_heldout_multi.triggered.connect(
            self._run_heldout_multi_model
        )
        tools_menu.addAction(self._action_heldout_multi)

        # Help menu. On macOS, Qt's `TextHeuristicRole` (default) auto-
        # moves actions named "About" / "Preferences" / "Quit" into
        # the Application Menu. If the only entry in Help is About,
        # the menu ends up empty AND hidden by macOS — exactly the
        # bug that made the Help menu invisible in the M4 commit.
        # Force NoRole so About stays put, and add a second entry
        # (Keyboard shortcuts) so the menu has substance.
        help_menu = mb.addMenu("&Help")
        action_shortcuts = QAction("Keyboard shortcuts", self)
        action_shortcuts.setMenuRole(QAction.MenuRole.NoRole)
        action_shortcuts.setShortcut(QKeySequence("F1"))
        action_shortcuts.triggered.connect(self._on_show_shortcuts)
        help_menu.addAction(action_shortcuts)
        help_menu.addSeparator()
        action_about = QAction("About Detector — PyQt", self)
        action_about.setMenuRole(QAction.MenuRole.NoRole)
        action_about.triggered.connect(self._on_about)
        help_menu.addAction(action_about)

        # Cached child window reference so repeated open clicks reuse
        # the same instance (and a running retrain doesn't drop).
        self._training_window: Optional[QWidget] = None
        # Cached preprocess window — re-used across menu clicks so a
        # mid-run batch isn't dropped on accidental dismissal.
        self._preprocess_window: Optional[QWidget] = None

    def _set_setting(self, key: str, value) -> None:
        self._settings[key] = value
        ui_settings.save_settings(self._settings)

    def _on_preprocess_clicked(self) -> None:
        """Open (or re-show) the preprocess window.

        Re-uses a single PreprocessWindow instance so a long batch
        survives an accidental close-and-reopen of the menu item.
        The window keeps its own thread; this main window just
        spawns and forgets it.

        Listens for `batch_completed` so the newest produced
        `_notched.mat` auto-loads into the labeler (P13).
        """
        from ui.windows.preprocess_window import PreprocessWindow
        if self._preprocess_window is None:
            start_dir = self._settings.get("last_recording_dir") or ""
            self._preprocess_window = PreprocessWindow(
                self, start_dir=start_dir,
            )
            # When the window is closed, drop the reference so the
            # next open builds a fresh state.
            self._preprocess_window.destroyed.connect(
                lambda *_: setattr(self, "_preprocess_window", None)
            )
            self._preprocess_window.batch_completed.connect(
                self._on_preprocess_finished
            )
        self._preprocess_window.show()
        self._preprocess_window.raise_()
        self._preprocess_window.activateWindow()

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

    # ------------------------------------------------------------------
    # Held-out evaluation
    # ------------------------------------------------------------------

    def _run_heldout_eval(self) -> None:
        """Kick off `detector heldout_eval.evaluate_heldout` in a
        background thread. Shows a model picker first (defaults to the
        toolbar dropdown's current selection), then a progress dialog
        while running, then a results dialog when done."""
        from PySide6.QtCore import QThread, Qt
        from PySide6.QtWidgets import (
            QInputDialog, QProgressDialog, QMessageBox,
        )
        from detector import paths as _detector_paths
        from detector.manifest import Manifest as _Manifest

        manifest_path = _detector_paths.get_manifest_path()
        if not manifest_path.exists():
            QMessageBox.warning(
                self, "No manifest",
                f"No training manifest at:\n{manifest_path}\n\n"
                "Run `detector init` or add at least one recording first.",
            )
            return

        # Sanity-check that at least one recording is marked held_out=True
        # before spinning up the worker. Otherwise the dialog just shows
        # "0 recordings evaluated" -- friendlier to catch it here.
        try:
            m = _Manifest.load(manifest_path)
        except Exception as e:
            QMessageBox.critical(
                self, "Manifest load failed",
                f"Could not load manifest:\n{e}",
            )
            return
        held = [r for r in m.list_recordings(include_held_out=True)
                if r.get("held_out", False)]
        if not held:
            QMessageBox.information(
                self, "No held-out recordings",
                "No recordings in the manifest are marked held_out=True.\n\n"
                "Hold-out a recording by checking 'Hold out from training' "
                "in Training Management → Add recording.",
            )
            return

        # ------------------------------------------------------------
        # Model picker
        # ------------------------------------------------------------
        # Show a chooser populated from the same versions list the
        # toolbar dropdown uses. Default the highlighted item to
        # whatever the toolbar currently has selected -- the user is
        # most likely evaluating "the model I'm working with right
        # now". The ★ marker on the promoted version is preserved so
        # it's obvious which one is currently in production.
        versions = list_available_versions()
        if not versions:
            QMessageBox.critical(
                self, "No model versions",
                "No model versions found on disk. Train at least one "
                "model before running held-out evaluation.",
            )
            return
        promoted = current_promoted_version_short()
        toolbar_choice = (
            self._version_combo.currentData()
            if hasattr(self, "_version_combo") and self._version_combo
            else None
        )
        labels = [f"{v} ★" if v == promoted else v for v in versions]
        # Map label -> raw version (the ★ suffix is display-only).
        label_to_version = dict(zip(labels, versions))
        default_version = toolbar_choice or promoted or versions[0]
        try:
            default_idx = versions.index(default_version)
        except ValueError:
            default_idx = 0

        chosen_label, ok = QInputDialog.getItem(
            self,
            "Choose model for held-out evaluation",
            (f"Evaluate {len(held)} held-out recording(s) against "
             "which model?\n\n(★ = currently promoted)"),
            labels,
            current=default_idx,
            editable=False,
        )
        if not ok:
            return   # user cancelled
        version_short = label_to_version[chosen_label]

        try:
            artifact = load_model_artifact_cached(version_short)
        except Exception as e:
            QMessageBox.critical(
                self, "Model load failed",
                f"Could not load model artifact {version_short}:\n{e}",
            )
            return

        # Background worker + progress dialog.
        self._heldout_thread = QThread()
        self._heldout_worker = HeldoutEvalWorker(
            manifest_path, artifact,
        )
        self._heldout_worker.moveToThread(self._heldout_thread)
        self._heldout_thread.started.connect(self._heldout_worker.run)
        self._heldout_worker.progress.connect(self._on_heldout_progress)
        self._heldout_worker.finished.connect(self._on_heldout_finished)
        self._heldout_worker.error.connect(self._on_heldout_error)
        self._heldout_worker.finished.connect(self._heldout_thread.quit)
        self._heldout_worker.error.connect(self._heldout_thread.quit)
        self._heldout_thread.finished.connect(
            self._heldout_worker.deleteLater
        )
        self._heldout_thread.finished.connect(
            self._heldout_thread.deleteLater
        )

        # Progress is per-recording (0 .. n_held); start at 0.
        self._heldout_progress = QProgressDialog(
            f"Evaluating held-out recordings (0/{len(held)}) …",
            "Cancel", 0, len(held), self,
        )
        self._heldout_progress.setWindowTitle(
            f"Held-out evaluation — model {version_short or '(unknown)'}"
        )
        self._heldout_progress.setWindowModality(Qt.WindowModal)
        self._heldout_progress.setMinimumDuration(0)
        # Cancel is best-effort: detector.predict has no cancel hook,
        # so closing the dialog stops the UI updates but the eval
        # finishes in the background. Same UX as the inference worker.
        self._heldout_progress.canceled.connect(
            lambda: self._action_heldout_eval.setEnabled(True)
        )

        self._action_heldout_eval.setEnabled(False)
        self._heldout_thread.start()

    def _on_heldout_progress(self, idx: int, total: int, rid: str) -> None:
        if self._heldout_progress is None:
            return
        self._heldout_progress.setMaximum(total)
        self._heldout_progress.setValue(idx)
        self._heldout_progress.setLabelText(
            f"Evaluating held-out recordings ({idx+1}/{total}) …\n{rid}"
        )

    def _on_heldout_finished(self, report: dict, interval_cache: dict) -> None:
        if self._heldout_progress is not None:
            self._heldout_progress.close()
            self._heldout_progress = None
        self._action_heldout_eval.setEnabled(True)
        # Lazy import to avoid circulars on startup.
        from ui.dialogs.heldout_eval_dialog import HeldoutEvalDialog
        from detector import paths as _detector_paths
        dlg = HeldoutEvalDialog(
            report,
            interval_cache=interval_cache,
            manifest_path=_detector_paths.get_manifest_path(),
            parent=self,
        )
        dlg.exec()

    def _on_heldout_error(self, msg: str) -> None:
        from PySide6.QtWidgets import QMessageBox
        if self._heldout_progress is not None:
            self._heldout_progress.close()
            self._heldout_progress = None
        self._action_heldout_eval.setEnabled(True)
        QMessageBox.critical(self, "Held-out evaluation failed", msg)

    # ------------------------------------------------------------------
    # Multi-model held-out comparison
    # ------------------------------------------------------------------

    def _run_heldout_multi_model(self) -> None:
        """Tools -> Compare models on held-out: pick 2-3 models, run
        them in parallel against the held-out set, show side-by-side
        results dialog."""
        from PySide6.QtCore import QThread, Qt
        from PySide6.QtWidgets import QProgressDialog, QMessageBox
        from detector import paths as _detector_paths
        from detector.manifest import Manifest as _Manifest

        manifest_path = _detector_paths.get_manifest_path()
        if not manifest_path.exists():
            QMessageBox.warning(
                self, "No manifest",
                f"No training manifest at:\n{manifest_path}\n\n"
                "Run `detector init` or add at least one recording first.",
            )
            return

        try:
            m = _Manifest.load(manifest_path)
        except Exception as e:
            QMessageBox.critical(
                self, "Manifest load failed",
                f"Could not load manifest:\n{e}",
            )
            return
        held = [r for r in m.list_recordings(include_held_out=True)
                if r.get("held_out", False)]
        if not held:
            QMessageBox.information(
                self, "No held-out recordings",
                "No recordings in the manifest are marked held_out=True.\n\n"
                "Hold-out a recording by checking 'Hold out from training' "
                "in Training Management → Add recording.",
            )
            return

        # Model picker
        versions = list_available_versions()
        if len(versions) < 2:
            QMessageBox.information(
                self, "Need at least 2 models",
                "Multi-model comparison needs at least 2 trained models "
                f"on disk; found {len(versions)}.",
            )
            return
        from ui.dialogs.heldout_multi_model_dialog import (
            HeldoutMultiModelPicker,
        )
        promoted = current_promoted_version_short()
        toolbar_choice = (
            self._version_combo.currentData()
            if hasattr(self, "_version_combo") and self._version_combo
            else None
        )
        # Default: toolbar selection + promoted (if different), else
        # just promoted (or first two versions if no promoted).
        default_selected: list[str] = []
        if toolbar_choice:
            default_selected.append(toolbar_choice)
        if promoted and promoted not in default_selected:
            default_selected.append(promoted)
        if not default_selected:
            default_selected = versions[:2]
        picker = HeldoutMultiModelPicker(
            versions,
            promoted=promoted,
            default_selected=default_selected,
            parent=self,
        )
        if picker.exec() != HeldoutMultiModelPicker.Accepted:
            return
        chosen = picker.chosen_versions()
        chosen_parallelize = picker.chosen_parallelize()
        if len(chosen) < 2:
            return   # safety net; picker already validates

        # Resolve picked versions -> on-disk artifact dirs.
        try:
            artifacts_dir = _detector_paths.get_artifacts_dir()
        except Exception as e:
            QMessageBox.critical(
                self, "Artifacts dir not configured",
                f"{e}",
            )
            return
        artifact_paths = []
        for v in chosen:
            d = (artifacts_dir / v if v.startswith("model_")
                 else artifacts_dir / f"model_{v}")
            if not (d / "booster.txt").exists():
                QMessageBox.critical(
                    self, "Model artifact missing",
                    f"Expected booster.txt at:\n{d}\n\n"
                    "The version may not be fully synced from Drive yet.",
                )
                return
            artifact_paths.append(d)

        # Background worker.
        self._heldout_multi_thread = QThread()
        self._heldout_multi_worker = HeldoutEvalMultiModelWorker(
            manifest_path, artifact_paths,
            parallelize=chosen_parallelize,
        )
        self._heldout_multi_worker.moveToThread(self._heldout_multi_thread)
        self._heldout_multi_thread.started.connect(
            self._heldout_multi_worker.run
        )
        self._heldout_multi_worker.progress.connect(
            self._on_heldout_multi_progress
        )
        self._heldout_multi_worker.finished.connect(
            self._on_heldout_multi_finished
        )
        self._heldout_multi_worker.error.connect(self._on_heldout_multi_error)
        self._heldout_multi_worker.finished.connect(
            self._heldout_multi_thread.quit
        )
        self._heldout_multi_worker.error.connect(
            self._heldout_multi_thread.quit
        )
        self._heldout_multi_thread.finished.connect(
            self._heldout_multi_worker.deleteLater
        )
        self._heldout_multi_thread.finished.connect(
            self._heldout_multi_thread.deleteLater
        )

        self._heldout_multi_progress = QProgressDialog(
            f"Evaluating {len(held)} recording(s) × {len(chosen)} model(s) …",
            "Cancel", 0, len(held), self,
        )
        self._heldout_multi_progress.setWindowTitle(
            f"Multi-model held-out: {' vs '.join(chosen)}"
        )
        self._heldout_multi_progress.setWindowModality(Qt.WindowModal)
        self._heldout_multi_progress.setMinimumDuration(0)
        # Stash the chosen versions so we can display them in the
        # finished/error handlers if needed.
        self._heldout_multi_chosen = list(chosen)
        self._heldout_multi_progress.canceled.connect(
            lambda: self._action_heldout_multi.setEnabled(True)
        )

        self._action_heldout_multi.setEnabled(False)
        self._heldout_multi_thread.start()

    def _on_heldout_multi_progress(
        self, idx: int, total: int, rid: str,
    ) -> None:
        if self._heldout_multi_progress is None:
            return
        self._heldout_multi_progress.setMaximum(total)
        self._heldout_multi_progress.setValue(idx)
        self._heldout_multi_progress.setLabelText(
            f"Evaluated {idx+1}/{total} recordings (last: {rid})"
        )

    def _on_heldout_multi_finished(
        self, report: dict, interval_cache: dict,
    ) -> None:
        if self._heldout_multi_progress is not None:
            self._heldout_multi_progress.close()
            self._heldout_multi_progress = None
        self._action_heldout_multi.setEnabled(True)
        from ui.dialogs.heldout_multi_model_dialog import (
            HeldoutMultiModelDialog,
        )
        from detector import paths as _detector_paths
        dlg = HeldoutMultiModelDialog(
            report,
            interval_cache=interval_cache,
            manifest_path=_detector_paths.get_manifest_path(),
            parent=self,
        )
        dlg.exec()

    def _on_heldout_multi_error(self, msg: str) -> None:
        from PySide6.QtWidgets import QMessageBox
        if self._heldout_multi_progress is not None:
            self._heldout_multi_progress.close()
            self._heldout_multi_progress = None
        self._action_heldout_multi.setEnabled(True)
        QMessageBox.critical(
            self, "Multi-model held-out evaluation failed", msg,
        )

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
        """Open a recording file (.mat / .h5).

        Uses the native file dialog — it matches the user's OS look &
        feel and the Files-vs-Drive-folder navigation works as
        expected. TDT folders use a separate menu entry
        (`_on_open_tdt_folder_clicked`) since Qt doesn't have a
        usable "either file or folder" dialog on macOS — `AnyFile`
        treats folder clicks as descend-into rather than select-this.
        """
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

    def _on_open_tdt_folder_clicked(self) -> None:
        """Open a TDT block folder.

        Uses the native folder-picker (`getExistingDirectory`). The
        selected folder is routed through `_handle_tdt_folder_open`
        which (a) loads the newest `_notched.mat` if outputs exist
        in the folder, or (b) opens the Preprocess window with the
        folder seeded if it looks like a TDT block but hasn't been
        processed yet.
        """
        if not self._maybe_discard_unsaved():
            return
        start_dir = self._settings.get("last_recording_dir") or ""
        path_str = QFileDialog.getExistingDirectory(
            self, "Open TDT folder", start_dir,
        )
        if not path_str:
            return
        self._handle_tdt_folder_open(Path(path_str))

    def _handle_tdt_folder_open(self, folder: Path) -> None:
        """The user picked a folder from the Open dialog. If it looks
        like a TDT block, route them appropriately:

        - Pipeline outputs already in the folder
          (`*_notched.mat`) → load the newest one directly.
        - Looks like a TDT block (`.tsq` inside) but no outputs →
          open the Preprocess window with this folder pre-populated.
        - Doesn't look like a TDT block → tell the user.

        Heuristic for "looks like a TDT block": presence of any
        `.tsq` file. This is the same check the backend uses in
        `tdt_io.list_streams`.
        """
        # 1. Already-processed outputs win — load directly.
        notched_candidates = sorted(folder.glob("*_notched.mat"))
        if notched_candidates:
            # Newest by mtime (so the most recently preprocessed run
            # wins if the user re-ran with a suffix).
            newest = max(notched_candidates, key=lambda p: p.stat().st_mtime)
            self._open_path(newest)
            return

        # 2. Looks like a TDT block? Open Preprocess pre-populated.
        if any(folder.glob("*.tsq")):
            resp = QMessageBox.question(
                self, "Preprocess this TDT folder?",
                f"<b>{folder.name}</b> looks like a TDT block but "
                "has no pipeline outputs yet.\n\n"
                "Open the preprocessing workflow with this folder "
                "pre-populated?",
            )
            if resp != QMessageBox.Yes:
                return
            self._open_preprocess_with_folder(folder)
            return

        # 3. Neither — bail loudly.
        QMessageBox.warning(
            self, "Not a recognized folder",
            f"<b>{folder.name}</b> isn't a .mat/.h5 recording and "
            "doesn't look like a TDT block (no .tsq file inside).\n\n"
            "Either pick a recording file directly, or pick a TDT "
            "block folder.",
        )

    def _open_preprocess_with_folder(self, folder: Path) -> None:
        """Open (or reuse) the Preprocess window and seed it with
        `folder` already in the batch list. Called from the
        smart-Open code path."""
        from ui.windows.preprocess_window import PreprocessWindow
        if self._preprocess_window is None:
            self._preprocess_window = PreprocessWindow(
                self, start_dir=str(folder.parent),
            )
            self._preprocess_window.destroyed.connect(
                lambda *_: setattr(self, "_preprocess_window", None)
            )
            # P13: when the batch finishes, load the newest produced
            # notched output into the main window so the user lands
            # straight in the labeler.
            self._preprocess_window.batch_completed.connect(
                self._on_preprocess_finished
            )
        # Seed the folder so the user doesn't have to click "Add".
        if hasattr(self._preprocess_window, "seed_folder"):
            self._preprocess_window.seed_folder(folder)
        self._preprocess_window.show()
        self._preprocess_window.raise_()
        self._preprocess_window.activateWindow()

    def _on_preprocess_finished(self, results: list) -> None:
        """Called when a Preprocess batch completes.

        Pick the newest `_notched.mat` from the successful results
        and load it into the main window. Multiple successful items
        → load the newest by mtime (typical case is the user processed
        baseline + stim_rec for the same animal and wants to see the
        most recent).
        """
        notched_paths: list[Path] = []
        for r in results or []:
            if r.get("status") != "ok":
                continue
            outputs = r.get("output_paths") or {}
            p = outputs.get("notched")
            if p is not None and Path(p).exists():
                notched_paths.append(Path(p))
        if not notched_paths:
            return
        newest = max(notched_paths, key=lambda p: p.stat().st_mtime)
        # Don't barge — ask before replacing whatever the user is
        # currently editing.
        if self._recording is not None:
            resp = QMessageBox.question(
                self, "Open preprocessed output?",
                f"Preprocessing finished. Open <b>{newest.name}</b> "
                "in the labeler?",
            )
            if resp != QMessageBox.Yes:
                return
        self._open_path(newest)

    def _open_path(self, path: Path) -> None:
        """Open `path`. We load TWICE because the two readers serve
        different needs (see __init__).

        Folder selections are routed through `_handle_tdt_folder_open`
        which decides between loading existing outputs vs opening the
        Preprocess window.
        """
        if not path.exists():
            QMessageBox.warning(self, "Open failed",
                                  f"File not found:\n{path}")
            return
        if path.is_dir():
            self._handle_tdt_folder_open(path)
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

        # Autosave restore prompt. If the previous session crashed
        # before the user clicked Save, the autosave sidecar still
        # has their work. Offer to restore it before any mutations
        # overwrite the in-memory state.
        self._maybe_restore_autosave(path)

        # Clear M3 inference state from any previous file.
        self._on_clear_predictions()

        self._refresh_widgets()
        self._action_save.setEnabled(True)
        self._action_save_as.setEnabled(True)
        # Boundary-setting is enabled for ANY loaded recording. The
        # filename-based rec_type ("stim_rec" vs "baseline" vs
        # "unknown") is a hint, not a constraint — the user might
        # want to mark a boundary on a file whose name doesn't match
        # the _stim_rec_ convention. save_period_split() validates
        # the boundary value at save time, so a bad value gets a
        # clear error rather than a silently dropped split.
        self._action_set_boundary.setEnabled(True)
        self._action_set_hr_chan.setEnabled(True)
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
    # Autosave (per-recording crash-recovery sidecar)
    # ------------------------------------------------------------------

    def _autosave_now(self) -> None:
        """Snapshot current label state to the per-recording autosave
        sidecar. Bounded by the number of marked intervals (~80 bytes
        per row in JSON), so safe to call on every mutation.

        Best-effort: any failure is swallowed so a write error never
        blocks the user's editing flow. The next Save (which goes
        through `save_native` / `save_matlab_compatible`) is the
        authoritative persistence path; autosave only exists for
        crash recovery.
        """
        if self._recording is None:
            return
        try:
            from detector import autosave as _A
            _A.save_snapshot(
                self._recording.source_path,
                bad_intervals=self._bad_intervals,
                bad_sources=self._bad_sources,
                stim_end_idx=self._stim_end_idx,
                model_intervals=self._model_intervals,
            )
        except Exception:
            pass

    def _clear_autosave(self) -> None:
        """Drop the autosave file. Called after a successful explicit
        Save so the next open doesn't offer to restore stale state."""
        if self._recording is None:
            return
        try:
            from detector import autosave as _A
            _A.clear_snapshot(self._recording.source_path)
        except Exception:
            pass

    def _maybe_restore_autosave(self, path: Path) -> None:
        """If an autosave sidecar exists for `path` and contains MORE
        state than the in-memory load, offer to restore it.

        Called from `_open_path` after the file's existing state is
        loaded; the prompt fires only if the autosave's interval
        count differs from what's in memory (otherwise there's
        nothing useful to restore)."""
        try:
            from detector import autosave as _A
            snapshot = _A.load_snapshot(path)
        except Exception:
            snapshot = None
        if snapshot is None:
            return
        as_intervals = np.asarray(
            snapshot.get("bad_intervals", []), dtype=np.int64,
        )
        if as_intervals.size == 0:
            as_intervals = as_intervals.reshape(0, 2)
        if as_intervals.shape[0] == self._bad_intervals.shape[0]:
            # Same count as what's on disk — nothing useful to
            # restore. Drop the autosave silently (it can only be
            # equal or stale at this point).
            self._clear_autosave()
            return
        age = _A.snapshot_age_seconds(path)
        age_str = _A.format_age(age) if age is not None else "unknown age"
        resp = QMessageBox.question(
            self, "Restore unsaved work?",
            f"<b>Autosave found</b> for {path.name} ({age_str}).<br><br>"
            f"It has <b>{as_intervals.shape[0]}</b> marked interval(s); "
            f"the file on disk has <b>{self._bad_intervals.shape[0]}</b>.<br><br>"
            "Restore the autosave, or keep what's on disk?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if resp == QMessageBox.Yes:
            self._bad_intervals = as_intervals
            self._bad_sources = list(snapshot.get("bad_sources", []))
            sei = snapshot.get("stim_end_idx")
            if sei is not None:
                self._stim_end_idx = int(sei)
            mi = snapshot.get("model_intervals")
            if mi:
                self._model_intervals = np.asarray(mi, dtype=np.int64)
            self._dirty = True       # user must Save to make it disk-of-truth
            self._refresh_widgets()
        else:
            self._clear_autosave()

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
        self._autosave_now()
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
        self._autosave_now()
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
        self._autosave_now()
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
        self._autosave_now()
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
            self._autosave_now()
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
        self._autosave_now()
        self._refresh_widgets()

    def _on_set_hr_chan(self) -> None:
        """Pop a dialog asking which channel carries the cleanest
        heartbeat. The selection is written as `hrChanIdx` (1-based)
        into the saved .mat, matching the manual MATLAB pipeline.
        Cancel/None means the field is omitted on save.
        """
        if self._recording is None:
            return
        n_ch = int(self._recording.y.shape[1])
        # Label format: "Ch 1 (VN1)" — matches the Streamlit picker.
        from ui.widgets.signal_viewer import CHANNEL_NAMES_DEFAULT
        labels = [
            f"Ch {i + 1} ({CHANNEL_NAMES_DEFAULT[i] if i < len(CHANNEL_NAMES_DEFAULT) else f'Ch{i + 1}'})"
            for i in range(n_ch)
        ]
        # Prepend an opt-out option so the user can clear a stale
        # selection without restarting the app.
        options = ["(none — don't write hrChanIdx)"] + labels
        # Pre-select whatever the recording currently has, or the
        # first nerve channel if unset.
        if self._hr_chan_idx is None or not (1 <= self._hr_chan_idx <= n_ch):
            cur_index = 1   # "Ch 1 (…)"
        else:
            cur_index = self._hr_chan_idx   # 1-based maps to options index
        choice, ok = QInputDialog.getItem(
            self, "Best heartbeat channel",
            ("Which channel shows the cleanest heartbeat?\n"
             "Saved as `hrChanIdx` in the .mat for the downstream "
             "MATLAB HR-rate detector."),
            options, cur_index, False,
        )
        if not ok:
            return
        if choice.startswith("(none"):
            self._hr_chan_idx = None
        else:
            # Pull the 1-based int between "Ch " and " (".
            try:
                self._hr_chan_idx = int(choice.split(" ", 2)[1])
            except (IndexError, ValueError):
                self._hr_chan_idx = None
        self._update_status_bar()

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
        # Block re-entry while a prior baseline is still computing in
        # the background -- otherwise two BaselineWorkers would race
        # on the same on-disk baseline.h5. The clean/bad/.mat/seg
        # writes themselves are fast (sub-second) and would succeed,
        # but the second compute_baseline would step on the first's
        # output. Simpler to block until the prior one lands.
        if (self._baseline_thread is not None
                and self._baseline_thread.isRunning()):
            QMessageBox.information(
                self, "Baseline still computing",
                "The previous Save's baseline.h5 is still being "
                "computed in the background. Wait for the status "
                "bar to show \"baseline ready\" before saving again.",
            )
            return

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
                hr_chan_idx=self._hr_chan_idx,
            )
            r_seg = save_segment_table(
                rec, intervals, sources,
                output_path=seg_path,
                model_unsure_intervals=model_unsure,
                stim_end_idx=self._stim_end_idx,
                model_version=self._model_version,
                threshold_used=self._model_threshold_used,
            )
            # Per-period split whenever a boundary is set, regardless
            # of filename-derived rec_type. The user may have a
            # generic .mat file (no _stim_rec_ in the name) but still
            # want the signal split at a manually-entered boundary.
            # save_period_split validates 1 <= stim_end_idx < N at
            # the backend, so an out-of-range value surfaces a clear
            # ValueError via the QMessageBox below.
            split_msg = ""
            if self._stim_end_idx is not None:
                try:
                    r_split = save_period_split(
                        rec, intervals, output_path_base=mat_path,
                        stim_end_idx=self._stim_end_idx,
                        bad_sources=sources,
                        model_unsure_intervals=model_unsure,
                        write_yout=True,
                        hr_chan_idx=self._hr_chan_idx,
                    )
                    split_msg = (
                        f"  ·  split: stim={r_split['stim_n_intervals']} + "
                        f"recovery={r_split['recovery_n_intervals']}"
                    )
                except ValueError as exc:
                    QMessageBox.warning(
                        self, "Period split skipped",
                        f"Could not split at sample {self._stim_end_idx}: "
                        f"{exc}\n\nThe full _blankmotion.mat was still "
                        "saved; only the per-period split files were "
                        "skipped.",
                    )
            unsure_msg = (
                f"  ·  unsure={int(model_unsure.shape[0])}"
                if model_unsure is not None else ""
            )
            self._dirty = False
            # The on-disk file is now the source of truth — drop the
            # autosave so the next open doesn't offer to restore the
            # stale snapshot.
            self._clear_autosave()
            # Co-locate the per-recording baseline.h5 next to the
            # just-written clean.h5 ("Option A" baseline layout). The
            # compute is offloaded to a Qt worker thread so the UI
            # doesn't freeze for the 5-15s it takes to scan every
            # clean chunk on a multi-GB recording. The status-bar line
            # below shows the save success combined with "computing
            # baseline (background)…"; the BaselineWorker's
            # finished/error slots overwrite it when the baseline
            # lands or fails.
            #
            # Naming: <stem>_baseline.h5 where <stem> matches the
            # clean.h5's stem with "_clean" stripped. Mirrors the
            # save_native bad.h5 derivation.
            stem_for_baseline = clean_path.stem
            if stem_for_baseline.endswith("_clean"):
                stem_for_baseline = stem_for_baseline[: -len("_clean")]
            baseline_path = clean_path.with_name(
                f"{stem_for_baseline}_baseline.h5"
            )
            self._start_baseline_worker(
                rec.recording_id, clean_path, baseline_path, float(rec.fs),
            )
            self.statusBar().showMessage(
                f"saved: clean.h5={r_native['n_clean_chunks']} chunks, "
                f".mat={r_mat['n_intervals']} intervals "
                f"(user={r_mat['n_user']}, "
                f"model_accepted={r_mat['n_model_accepted']}, "
                f"existing={r_mat['n_existing']})"
                f"{unsure_msg}{split_msg}"
                f"  ·  computing baseline (background) …",
                0,    # no timeout -- replaced when the baseline worker lands
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", f"{exc}")

    # ------------------------------------------------------------------
    # Background baseline.h5 computation
    # ------------------------------------------------------------------

    def _start_baseline_worker(
        self,
        recording_id: str,
        clean_path: Path,
        baseline_path: Path,
        fs: float,
    ) -> None:
        """Spawn the background BaselineWorker for a just-saved
        clean.h5. Wires the standard finished -> thread.quit ->
        deleteLater chain so the QThread doesn't leak when it
        completes. Re-entry guarding lives in `_save_to` -- by the
        time we reach this method, any prior baseline has finished.
        """
        self._baseline_thread = QThread()
        self._baseline_worker = BaselineWorker(
            recording_id, clean_path, baseline_path, fs,
            force=True,    # clean.h5 just changed; don't reuse stale cache
        )
        self._baseline_worker.moveToThread(self._baseline_thread)
        self._baseline_thread.started.connect(self._baseline_worker.run)
        self._baseline_worker.finished.connect(self._on_baseline_finished)
        self._baseline_worker.error.connect(self._on_baseline_error)
        self._baseline_worker.finished.connect(self._baseline_thread.quit)
        self._baseline_worker.error.connect(self._baseline_thread.quit)
        self._baseline_thread.finished.connect(
            self._baseline_worker.deleteLater
        )
        self._baseline_thread.finished.connect(
            self._baseline_thread.deleteLater
        )
        self._baseline_thread.start()

    def _on_baseline_finished(self, baseline_path: str) -> None:
        self.statusBar().showMessage(
            f"baseline ready: {Path(baseline_path).name}",
            12_000,
        )

    def _on_baseline_error(self, message: str) -> None:
        # Match the inline-baseline error UX from commit f8bbfe3 --
        # status-bar note + QMessageBox.warning with remediation
        # instructions. The clean/bad/.mat/seg files are already on
        # disk so this only affects baseline-dependent retrains.
        self.statusBar().showMessage(
            f"baseline FAILED: {message}",
            12_000,
        )
        QMessageBox.warning(
            self, "Baseline computation failed",
            "The clean.h5 / bad.h5 / .mat / segments.json were "
            "saved successfully, but the per-recording "
            "baseline.h5 could not be computed:\n\n"
            f"{message}\n\n"
            "You can either rerun Save (overwrites all files) "
            "or manually run detector.baseline.compute_baseline "
            "on the clean.h5 to produce the baseline.",
        )

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
        hr_tag = (f"  ·  hrCh={self._hr_chan_idx}"
                   if self._hr_chan_idx else "")
        dirty = "  ·  ●" if self._dirty else ""
        self.statusBar().showMessage(
            f"{self._recording.recording_id}  ·  "
            f"fs={self._recording.fs:.1f} Hz  ·  "
            f"{self._recording.n_samples:,} samples "
            f"({self._recording.duration_sec:.1f}s)  ·  "
            f"marked: {n_bad} ({bad_frac:.3f}){bd}{hr_tag}{dirty}"
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
        if not self._maybe_discard_unsaved():
            event.ignore()
            return
        # If a background baseline is still computing, ask the user
        # whether to wait. Quitting without waiting orphans the
        # QThread -- on the next event loop turn it'd try to emit
        # `finished` into a torn-down main window. Blocking on wait()
        # here freezes the close for up to 5-15s on big recordings,
        # but the user just asked to close, so a brief block is
        # acceptable.
        if (self._baseline_thread is not None
                and self._baseline_thread.isRunning()):
            resp = QMessageBox.question(
                self, "Baseline still computing",
                "A baseline.h5 is still being computed in the "
                "background. Wait for it to finish before closing?\n\n"
                "(Choosing 'No' will close the window now; the "
                "baseline will be abandoned and may leave a partial "
                "file on disk.)",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            )
            if resp == QMessageBox.Cancel:
                event.ignore()
                return
            if resp == QMessageBox.Yes:
                self._baseline_thread.wait()
        event.accept()

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

    def _on_accept_prediction(self, indices) -> None:
        """Accept one or more predictions. `indices` is a list of row
        indices into `self._model_intervals` (the panel always emits a
        list; a single-row interaction is wrapped as `[i]`)."""
        if self._model_intervals is None or self._model_intervals.size == 0:
            return
        # Defensive: filter to in-range indices, dedupe, sort.
        idx_arr = np.array(sorted({
            int(i) for i in (indices if hasattr(indices, "__iter__")
                              else [indices])
            if 0 <= int(i) < self._model_intervals.shape[0]
        }), dtype=np.int64)
        if idx_arr.size == 0:
            return
        self._push_undo()
        new = self._model_intervals[idx_arr].astype(np.int64, copy=True)
        combined = (np.concatenate([self._bad_intervals, new], axis=0)
                    if self._bad_intervals.size else new)
        order = np.argsort(combined[:, 0], kind="stable")
        self._bad_intervals = combined[order]
        sources_combined = (
            list(self._bad_sources)
            + ["model_accepted"] * int(new.shape[0])
        )
        self._bad_sources = [sources_combined[int(i)] for i in order]
        # Remove ALL accepted rows from predictions in one slice — np.delete
        # accepts an array of indices and handles the bookkeeping (the
        # alternative, deleting one-by-one, would invalidate later indices
        # as the array shrinks).
        self._model_intervals = np.delete(
            self._model_intervals, idx_arr, axis=0,
        )
        self._dirty = True
        self._autosave_now()
        self._refresh_predictions_panel()
        self._refresh_widgets()

    def _on_dismiss_prediction(self, indices) -> None:
        """Dismiss one or more predictions (remove from model list
        without marking)."""
        if self._model_intervals is None or self._model_intervals.size == 0:
            return
        idx_arr = np.array(sorted({
            int(i) for i in (indices if hasattr(indices, "__iter__")
                              else [indices])
            if 0 <= int(i) < self._model_intervals.shape[0]
        }), dtype=np.int64)
        if idx_arr.size == 0:
            return
        self._model_intervals = np.delete(
            self._model_intervals, idx_arr, axis=0,
        )
        # Autosave so dismissed predictions stay dismissed across
        # crashes. Even though bad_intervals didn't change, the
        # model_intervals state needs to persist.
        self._autosave_now()
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
        self._autosave_now()
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
        self._autosave_now()

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
            "Phases M0–M5 complete: lazy multichannel viewer, manual "
            "labeling, inference + disagreement review, training "
            "management, and recording queue.\n\n"
            "Plotly-free, GPU-friendly via pyqtgraph.\n\n"
            "Shift+drag to mark a bad region. Use the right-panel "
            "tabs to switch between user marks and model predictions.",
        )

    def _on_show_shortcuts(self) -> None:
        """Modal listing every keyboard shortcut wired up in the app
        — useful both as user documentation and as a self-check that
        nothing got disconnected during the migration."""
        rows = [
            ("File",   "Ctrl+O",        "Open recording"),
            ("File",   "Ctrl+S",        "Save"),
            ("File",   "Ctrl+Shift+S",  "Save as…"),
            ("File",   "Ctrl+Q",        "Quit"),
            ("Edit",   "Ctrl+Z",        "Undo"),
            ("Edit",   "Ctrl+Shift+Z",  "Redo"),
            ("View",   "R",             "Reset zoom (0–60s)"),
            ("View",   "F",             "Fit all"),
            ("View",   "Z",             "Zoom in (×0.5)"),
            ("View",   "X",             "Zoom out (×2)"),
            ("View",   "←",             "Previous interval"),
            ("View",   "→",             "Next interval"),
            ("M3",     "Ctrl+R",        "Run inference"),
            ("M3",     "1 / 2 / 3",     "Score true / borderline / FP "
                                          "(in review panel)"),
            ("M4",     "Ctrl+T",        "Training management"),
            ("M5",     "Ctrl+Q",        "Toggle recording queue panel"),
            ("Viewer", "Shift+drag",    "Mark a bad region"),
            ("Help",   "F1",            "This dialog"),
        ]
        # Render as a monospace block so columns line up nicely.
        widths = (max(len(r[0]) for r in rows) + 1,
                  max(len(r[1]) for r in rows) + 2,
                  max(len(r[2]) for r in rows))
        lines = []
        for menu, sc, desc in rows:
            lines.append(f"{menu:<{widths[0]}}  "
                          f"{sc:<{widths[1]}}  {desc}")
        text = "\n".join(lines)
        # Use QMessageBox.about() variant so the user can dismiss easily,
        # but we set the text manually so the formatting renders cleanly.
        box = QMessageBox(self)
        box.setWindowTitle("Keyboard shortcuts")
        box.setIcon(QMessageBox.Information)
        box.setText("<b>Keyboard shortcuts</b>")
        box.setInformativeText(f"<pre>{text}</pre>")
        box.setStandardButtons(QMessageBox.Ok)
        box.exec()
