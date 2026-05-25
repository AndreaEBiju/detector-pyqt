"""Per-recording overlay viewer for held-out evaluation results.

Opens from a row in the HeldoutEvalDialog table. Shows the
recording's signal in the same MultiChannelViewer used by the
labeling UI, with two overlays:

  * Human ground-truth bad intervals (red translucent bands) --
    read from the recording's `splitter_bad_path` h5 file.
  * Model predicted bad intervals (orange translucent bands) --
    read from the eval run's interval cache when available, or
    re-computed on the fly via `detector.predict.detect_bad` when
    the cache is empty (e.g. an old report dict).

The window is intentionally read-only -- no Shift+drag wiring, no
boundary edits -- because changing labels here would silently
contaminate the held-out ground truth (the whole point of held-out
is that the human labels are frozen). The user gets pan / zoom / fit
controls only.

Drive-hosted recordings (Andrea's setup) are loaded via h5py the
same way `main_window._open_path` does, but in a try/except wrapper
so that a missing file or h5 read failure shows a friendly error
instead of crashing the dialog along with the window.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QLabel, QMainWindow, QMessageBox, QToolBar, QWidget,
)

# Mirror the worker's sys.path setup so the window works whether it
# was launched from the main app or from a standalone script.
_repo_root = Path(__file__).resolve().parent.parent.parent
_detector_core = _repo_root / "detector-core"
if _detector_core.exists() and str(_detector_core) not in sys.path:
    sys.path.insert(0, str(_detector_core))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from ui.widgets.signal_viewer import LazyRecording, MultiChannelViewer  # noqa: E402


class HeldoutOverlayWindow(QMainWindow):
    """Read-only overlay viewer for a single held-out recording.

    Construct with a manifest record dict (so we have source_path +
    splitter_bad_path) and the cached model intervals from the eval
    run. If the cache entry is missing, the overlay window can still
    open -- it just won't show the orange model band layer (and will
    log a warning in the title bar).
    """

    def __init__(
        self,
        recording_entry: dict,
        cache_entry: Optional[dict],
        *,
        artifact=None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        rid = recording_entry.get("recording_id", "?")
        self.setWindowTitle(f"Held-out overlay — {rid}")
        self.resize(1200, 720)
        self._lazy: Optional[LazyRecording] = None
        self._viewer: Optional[MultiChannelViewer] = None

        source_path = Path(recording_entry["source_path"])
        bad_path = Path(recording_entry["splitter_bad_path"])

        # Open the signal lazily (h5py mmap). Wrap in a friendly error
        # path so a Drive read failure doesn't take the dialog down.
        try:
            self._lazy = LazyRecording(source_path)
        except Exception as exc:
            self._show_error(
                "Could not open recording",
                f"Failed to open\n{source_path}\n\n{type(exc).__name__}: {exc}",
            )
            return

        # Build viewer. Note we deliberately don't connect the
        # bad_interval_added / stim_boundary_moved signals -- this
        # window is read-only by design.
        self._viewer = MultiChannelViewer(self._lazy)
        self.setCentralWidget(self._viewer)

        # Load human bad intervals.
        human_intervals = self._load_human_intervals(bad_path)
        # Resolve model intervals (cache first, fallback to detect_bad
        # if a model artifact was passed in).
        model_intervals = self._resolve_model_intervals(
            cache_entry, artifact, rid,
        )

        self._viewer.set_bad_intervals(human_intervals)
        self._viewer.set_model_intervals(model_intervals)

        self._build_toolbar(rid, human_intervals, model_intervals)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_human_intervals(self, bad_path: Path) -> np.ndarray:
        try:
            from detector.evaluate import load_bad_intervals
            return load_bad_intervals(str(bad_path))
        except Exception as exc:
            # Surface in the title bar but don't kill the window --
            # the user might still want to inspect the signal alone.
            self.statusBar().showMessage(
                f"Could not load ground-truth bad intervals: "
                f"{type(exc).__name__}: {exc}",
            )
            return np.zeros((0, 2), dtype=np.int64)

    def _resolve_model_intervals(
        self,
        cache_entry: Optional[dict],
        artifact,
        rid: str,
    ) -> np.ndarray:
        if cache_entry is not None and "model" in cache_entry:
            return np.asarray(cache_entry["model"], dtype=np.int64)
        if artifact is None:
            self.statusBar().showMessage(
                "No cached model intervals and no artifact passed -- "
                "model overlay will be empty.",
            )
            return np.zeros((0, 2), dtype=np.int64)
        # Fallback: re-run detect_bad. Slow (same as the original eval
        # for this recording) but lets the user open an overlay window
        # from a stale report dict in a future workflow.
        try:
            from detector import predict as Pr
            from detector.evaluate import load_y
            y, fs = load_y(str(self._lazy.path))
            res = Pr.detect_bad(y, fs, artifact)
            return np.asarray(res["intervals"], dtype=np.int64)
        except Exception as exc:
            self.statusBar().showMessage(
                f"Could not compute model intervals for {rid}: "
                f"{type(exc).__name__}: {exc}",
            )
            return np.zeros((0, 2), dtype=np.int64)

    def _build_toolbar(
        self,
        rid: str,
        human: np.ndarray,
        model: np.ndarray,
    ) -> None:
        tb = QToolBar("Overlay")
        tb.setMovable(False)
        self.addToolBar(tb)

        act_fit = QAction("Fit all", self)
        act_fit.triggered.connect(self._fit_all)
        tb.addAction(act_fit)

        act_first60 = QAction("First 60s", self)
        act_first60.triggered.connect(self._first_60s)
        tb.addAction(act_first60)

        tb.addSeparator()

        legend = QLabel(
            f"  <b>{rid}</b>  |  "
            f"<span style='color:#d62728;'>&#9608;</span> human bad "
            f"({len(human)} interval{'s' if len(human) != 1 else ''})  "
            f"<span style='color:#ff7f0e;'>&#9608;</span> model bad "
            f"({len(model)} interval{'s' if len(model) != 1 else ''})  "
            f"<i>(read-only)</i>"
        )
        legend.setTextFormat(Qt.RichText)
        legend.setContentsMargins(12, 0, 12, 0)
        tb.addWidget(legend)

    def _fit_all(self) -> None:
        if self._viewer is None or self._lazy is None:
            return
        self._viewer.set_viewport(0.0, self._lazy.duration_sec)

    def _first_60s(self) -> None:
        if self._viewer is None or self._lazy is None:
            return
        end = min(60.0, self._lazy.duration_sec)
        self._viewer.set_viewport(0.0, end)

    def _show_error(self, title: str, msg: str) -> None:
        QMessageBox.critical(self, title, msg)
        # Defer closing until the event loop runs so __init__ can
        # return cleanly before the window is reaped.
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self.close)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        if self._lazy is not None:
            try:
                self._lazy.close()
            except Exception:
                pass
            self._lazy = None
        super().closeEvent(event)
