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
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget, QFileDialog, QInputDialog, QMainWindow, QMessageBox,
    QStatusBar, QToolBar, QWidget,
)

from detector.recording_io import Recording, load_recording
from detector.labeled_save import (
    save_native, save_matlab_compatible, save_period_split,
    save_segment_table,
)

from ui.widgets.signal_viewer import LazyRecording, MultiChannelViewer
from ui.widgets.region_table import RegionTable
from ui.widgets.overview_strip import OverviewStrip


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

        # Widgets — placeholders until a recording opens.
        self._viewer: Optional[MultiChannelViewer] = None
        self._overview: Optional[OverviewStrip] = None
        self._region_table = RegionTable()
        self._region_table.interval_jumped.connect(self._on_jump_to_interval)
        self._region_table.interval_deleted.connect(self._on_delete_interval)

        self._build_menus()
        self._build_toolbar()
        self.setStatusBar(QStatusBar())
        self._update_status_bar()

        # Right dock for the region table.
        self._table_dock = QDockWidget("Marked intervals", self)
        self._table_dock.setObjectName("region_table_dock")
        self._table_dock.setWidget(self._region_table)
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

        # Help menu
        help_menu = mb.addMenu("&Help")
        action_about = QAction("About", self)
        action_about.triggered.connect(self._on_about)
        help_menu.addAction(action_about)

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

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _on_open_clicked(self) -> None:
        if self._maybe_discard_unsaved() == False:
            return
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open recording",
            "",
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

        self._refresh_widgets()
        self._action_save.setEnabled(True)
        self._action_save_as.setEnabled(True)
        self._action_set_boundary.setEnabled(rec.rec_type == "stim_rec")
        self._update_status_bar()
        self.setWindowTitle(f"Detector — PyQt   |   {path.name}")

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
                write_yout=True,
            )
            r_seg = save_segment_table(
                rec, intervals, sources,
                output_path=seg_path,
                stim_end_idx=self._stim_end_idx,
            )
            # Per-period split if a boundary is set.
            split_msg = ""
            if self._stim_end_idx is not None and rec.rec_type == "stim_rec":
                r_split = save_period_split(
                    rec, intervals, output_path_base=mat_path,
                    stim_end_idx=self._stim_end_idx,
                    bad_sources=sources,
                    write_yout=True,
                )
                split_msg = (
                    f"\nsplit: stim={r_split['stim_n_intervals']} + "
                    f"recovery={r_split['recovery_n_intervals']}"
                )
            self._dirty = False
            self.statusBar().showMessage(
                f"saved: clean.h5={r_native['n_clean_chunks']} chunks, "
                f".mat={r_mat['n_intervals']} intervals, "
                f"segments.json={r_seg['n_accepted']}{split_msg}",
                10_000,
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
        if self._overview is not None:
            self._overview.set_bad_intervals(self._bad_intervals)
            self._overview.set_stim_boundary(self._stim_end_idx)
        if self._recording is not None:
            self._region_table.set_intervals(
                self._bad_intervals, self._bad_sources, self._recording.fs,
            )
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

    def _on_about(self) -> None:
        QMessageBox.information(
            self, "About",
            "Detector — PyQt UI\n\n"
            "Phase M2 (browser + manual labeling).\n"
            "Plotly-free, GPU-friendly via pyqtgraph.\n\n"
            "Shift+drag to mark a bad region.\n"
            "Drag the overview region or click to navigate.\n"
        )
