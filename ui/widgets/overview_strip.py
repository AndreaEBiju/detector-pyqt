"""Overview strip — thin pyqtgraph widget showing the whole recording
on a single channel, with a draggable viewport indicator and red
overlays for bad regions.

Reads from `LazyRecording.get_overview_for_range(0, duration)` so the
strip paints in ~10 ms regardless of recording length. Falls back to a
heavily-decimated full-data read for older files that don't have
`/y_overview`.

Interactions:
- Draggable LinearRegionItem shows current viewport. Drag it → emit
  `viewport_changed(t_start, t_end)`. Main window then calls
  `signal_viewer.set_viewport(...)` to keep both widgets in sync.
- Click anywhere on the strip → centre viewport on click position
  (preserves the current viewport width).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal

from .signal_viewer import LazyRecording


class OverviewStrip(pg.PlotWidget):
    """A single-row pyqtgraph plot tracking the whole recording."""

    # The user moved the viewport-indicator region. Args are
    # (t_start_sec, t_end_sec) in recording-time coordinates.
    viewport_changed = Signal(float, float)

    def __init__(
        self,
        recording: LazyRecording,
        channel: int = 0,
        parent: Optional[object] = None,
    ):
        super().__init__(parent)
        self.setBackground("#0e1117")  # match Streamlit dark theme
        self.recording = recording
        self.channel = max(0, min(channel, recording.n_channels - 1))

        self._plot_item = self.getPlotItem()
        self._plot_item.setMouseEnabled(x=False, y=False)
        self._plot_item.hideAxis("left")
        self._plot_item.hideAxis("bottom")
        self._plot_item.showGrid(x=False, y=False)
        self._plot_item.setMenuEnabled(False)
        # Suppress range autoscale flicker when bad-region items get added.
        self._plot_item.setMouseEnabled(False, False)

        # Render the full-recording overview curve.
        t, data = self._read_full_overview()
        self._overview_curve = self._plot_item.plot(
            t, data[:, self.channel],
            pen=pg.mkPen(color="#aaaaaa", width=0.7),
        )
        self._plot_item.setXRange(0.0, recording.duration_sec, padding=0)

        # The viewport indicator — a draggable horizontal-direction
        # LinearRegionItem. Spans whatever range the main signal viewer
        # is currently showing.
        self._viewport_region = pg.LinearRegionItem(
            values=(0.0, min(60.0, recording.duration_sec)),
            orientation="vertical",
            movable=True,
            brush=pg.mkBrush(78, 163, 255, 60),       # blue, translucent
            pen=pg.mkPen(color="#4ea3ff", width=1.5),
        )
        self._viewport_region.setZValue(20)
        # Clamp drag inside the recording's time bounds.
        self._viewport_region.setBounds([0.0, recording.duration_sec])
        self._viewport_region.sigRegionChangeFinished.connect(
            self._on_region_changed
        )
        self._plot_item.addItem(self._viewport_region)

        # Bad-region overlay state: list of LinearRegionItem we created.
        self._bad_region_items: list[pg.LinearRegionItem] = []

        # Suppress signal echo when set_viewport_range() is called
        # programmatically.
        self._suppress_signal = False

        # Slim widget (about as tall as a few rows of the main viewer).
        self.setFixedHeight(80)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read_full_overview(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (t, data) for the whole recording — uses the
        pre-computed /y_overview if available, otherwise decimates a
        full /y read."""
        if self.recording.has_overview:
            return self.recording.get_overview_for_range(
                0.0, self.recording.duration_sec,
            )
        # Fallback: decimate a full read. This is OK for short
        # recordings; longer ones will be slow on first open but
        # the strip is cached implicitly via the curve's autoDownsample.
        data = self.recording.get_range(0.0, self.recording.duration_sec)
        n = data.shape[0]
        t = np.linspace(0.0, self.recording.duration_sec, n, endpoint=False)
        return t, data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_viewport_range(self, t_start: float, t_end: float) -> None:
        """Move the indicator region (programmatic; no signal echo)."""
        self._suppress_signal = True
        try:
            self._viewport_region.setRegion((float(t_start), float(t_end)))
        finally:
            self._suppress_signal = False

    def set_bad_intervals(self, intervals: np.ndarray) -> None:
        """Replace the red overlay rectangles with `intervals`."""
        for it in self._bad_region_items:
            self._plot_item.removeItem(it)
        self._bad_region_items = []
        if intervals is None or len(intervals) == 0:
            return
        for s, e in np.asarray(intervals, dtype=np.int64):
            s_sec = (int(s) - 1) / self.recording.fs
            e_sec = (int(e) - 1) / self.recording.fs
            region = pg.LinearRegionItem(
                values=(s_sec, e_sec),
                orientation="vertical",
                movable=False,
                brush=pg.mkBrush(214, 39, 40, 70),  # red, slightly more opaque
                pen=pg.mkPen(None),
            )
            region.setZValue(5)
            self._plot_item.addItem(region)
            self._bad_region_items.append(region)

    def set_model_intervals(self, intervals: Optional[np.ndarray]) -> None:
        """Replace the orange model-prediction overlays."""
        if not hasattr(self, "_model_region_items"):
            self._model_region_items: list[pg.LinearRegionItem] = []
        for it in self._model_region_items:
            self._plot_item.removeItem(it)
        self._model_region_items = []
        if intervals is None or len(intervals) == 0:
            return
        for s, e in np.asarray(intervals, dtype=np.int64):
            s_sec = (int(s) - 1) / self.recording.fs
            e_sec = (int(e) - 1) / self.recording.fs
            region = pg.LinearRegionItem(
                values=(s_sec, e_sec),
                orientation="vertical",
                movable=False,
                brush=pg.mkBrush(255, 127, 14, 60),  # orange
                pen=pg.mkPen(None),
            )
            region.setZValue(3)  # behind red overlays
            self._plot_item.addItem(region)
            self._model_region_items.append(region)

    def set_stim_boundary(self, stim_end_idx: Optional[int]) -> None:
        """Draw a dashed line at the stim/recovery boundary."""
        # Clear any existing
        for it in list(getattr(self, "_boundary_items", [])):
            self._plot_item.removeItem(it)
        self._boundary_items = []
        if stim_end_idx is None:
            return
        from PySide6.QtCore import Qt
        sec = (int(stim_end_idx) - 1) / self.recording.fs
        line = pg.InfiniteLine(
            pos=sec, angle=90, movable=False,
            pen=pg.mkPen(color="#cccccc", width=1.5, style=Qt.DashLine),
        )
        line.setZValue(15)
        self._plot_item.addItem(line)
        self._boundary_items.append(line)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_region_changed(self, region_item: pg.LinearRegionItem) -> None:
        if self._suppress_signal:
            return
        t_start, t_end = region_item.getRegion()
        self.viewport_changed.emit(float(t_start), float(t_end))
