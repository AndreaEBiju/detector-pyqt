"""M1 spike — lazy-loading multi-channel signal viewer.

Two classes:

`LazyRecording`
    Memory-mapped multi-channel HDF5 reader. Reads only the requested
    time range from the underlying file via h5py's lazy slicing.
    Supports the existing `_notched.mat` schema (MATLAB v7.3, which is
    HDF5 internally) and a future flat HDF5 schema interchangeably.
    Reading a 60s window out of a 2-hour file touches roughly 60/7200
    of the underlying bytes, not the whole thing.

`MultiChannelViewer`
    Stacked pyqtgraph plots, one per channel, x-axes linked. Pulls
    fresh data from the LazyRecording on every `set_viewport()` call
    and feeds it to pyqtgraph's built-in MinMax (peak) downsampler.
    Never holds more than the current viewport's data in memory.

M1.3 measures pan/zoom latency against these — see
`scripts/m1_benchmark.py`. M2 will build the labeling UI on top.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, Signal


# Mirror the Streamlit UI's 5-channel naming so the two stay
# visually consistent. The PyQt viewer's labels match the Streamlit
# config (`ui/config.py::CHANNEL_NAMES`).
CHANNEL_NAMES_DEFAULT = ("VN1", "VN2", "Ant1", "Ant2", "Ant3")
CHANNEL_COLORS_DEFAULT = (
    "#4ea3ff",  # VN1   blue
    "#2ca02c",  # VN2   green
    "#ff7f0e",  # Ant1  orange
    "#d62728",  # Ant2  red
    "#9467bd",  # Ant3  purple
)


class LazyRecording:
    """Memory-mapped multi-channel recording.

    On-disk schemas accepted:
      - MATLAB v7.3 (`.mat`) — produced by the existing notch step.
        `/y` shape `(n_ch, N)` float64, `/fs` scalar.
      - MATLAB-style `yOut` (`_blankmotion.mat` family) —
        `/yOut` shape `(n_ch, N)` float32, `/fs` scalar.
      - Flat HDF5 (`.h5`) — `/y` shape `(N, n_ch)` (any float dtype),
        `/fs` scalar.

    The (n_ch, N) vs (N, n_ch) layout is auto-detected from the shape
    (n_ch is conventionally ≤32, much smaller than N). The reader
    transposes on the fly so callers always see `(n, n_ch)` slices.

    Returned arrays are always `float32` — the heavy float64 from
    MATLAB is downcast at the read boundary to keep memory + plotting
    cheap. Visual fidelity at scope range is unaffected.
    """

    def __init__(self, path: Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        self.path = path
        # h5py opens lazily — no data read until we slice the dataset.
        self._h5 = h5py.File(str(path), "r")

        # Probe the dataset key. MATLAB blankmotion files use yOut;
        # MATLAB notched files use y; future flat HDF5 we'll write
        # also uses y. yOut takes precedence if both exist.
        if "yOut" in self._h5:
            self._y = self._h5["yOut"]
        elif "y" in self._h5:
            self._y = self._h5["y"]
        else:
            raise ValueError(
                f"{path}: no `/y` or `/yOut` dataset found. Available "
                f"top-level keys: {list(self._h5.keys())}"
            )

        # Auto-detect orientation. MATLAB-saved (n_ch, N) has small
        # first dim. We treat anything with first dim < 32 as the
        # channel axis. Adjust this heuristic if recordings ever come
        # in with hundreds of channels.
        a, b = self._y.shape
        if a <= 32 and b > 32:
            self._transposed = True
            self.n_channels, self.n_samples = a, b
        else:
            self._transposed = False
            self.n_samples, self.n_channels = a, b

        if "fs" not in self._h5:
            raise ValueError(f"{path}: no `/fs` dataset found")
        self.fs = float(np.asarray(self._h5["fs"][()]).squeeze())
        if self.fs <= 0:
            raise ValueError(f"{path}: invalid fs={self.fs}")
        self.duration_sec = self.n_samples / self.fs

        # M2.0 overview: a pre-downsampled min/max-pair summary
        # written by scripts/m1_ingest.py. None when the file
        # predates M2.0 ingest (old flat.h5 or raw MATLAB) — in
        # which case wide viewports fall back to the full /y read.
        self._y_overview = self._h5["y_overview"] if "y_overview" in self._h5 else None
        self._overview_bucket_size = int(
            self._h5.attrs.get("overview_bucket_size", 0)
        ) if self._y_overview is not None else 0

    def get_range(self, t_start: float, t_end: float) -> np.ndarray:
        """Return samples in `[t_start, t_end)` as `(n, n_ch)` float32.

        Out-of-range indices are clipped to `[0, n_samples]`. If the
        requested window is entirely outside the recording, returns an
        empty `(0, n_ch)` array — callers can detect with `.size == 0`.

        Perf note: for the (n_ch, N) MATLAB layout this read involves
        an h5py lazy slice + transpose + dtype downcast. The transpose
        is a view (free), but `astype(np.float32)` is a copy. We skip
        the explicit `order='C'` ascontiguous step because pyqtgraph's
        curve.setData() handles strided arrays internally and the
        explicit copy added ~4x overhead in M1 spike timings. For the
        truly fast path, use `flat_h5_path()` to ingest the file
        once into a (N, n_ch) flat HDF5 — that lets each viewport
        read be a single contiguous chunk read.
        """
        i0 = max(0, int(round(t_start * self.fs)))
        i1 = min(self.n_samples, int(round(t_end * self.fs)))
        if i0 >= i1:
            return np.zeros((0, self.n_channels), dtype=np.float32)
        if self._transposed:
            arr = self._y[:, i0:i1].T  # view, no copy
        else:
            arr = self._y[i0:i1, :]
        # astype() does the dtype copy; we deliberately don't add a
        # second copy for C-contiguous layout.
        return arr.astype(np.float32, copy=False)

    @property
    def has_overview(self) -> bool:
        """True if the file has a pre-computed `/y_overview` dataset
        (written by `scripts/m1_ingest.py` since M2.0). False for raw
        MATLAB / older flat HDF5 files."""
        return self._y_overview is not None

    def get_overview_for_range(
        self, t_start: float, t_end: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return `(t, y_overview)` for the requested time window,
        sampled from the pre-downsampled `/y_overview` dataset.

        Use this at wide viewports (where reading full /y would be
        bandwidth-bound; see M1.3 results). Output is `(rows, n_ch)`
        float32; `rows ≪ get_range()` would return.

        The overview stores min/max pairs per source bucket; output
        rows alternate min and max so the rendered curve preserves
        extremes (motion-artifact spikes won't be averaged away).

        Raises `RuntimeError` if the file has no overview. Caller is
        expected to check `has_overview` first or fall back to
        `get_range()`.
        """
        if self._y_overview is None:
            raise RuntimeError(
                f"{self.path}: no /y_overview dataset. Re-run "
                "scripts/m1_ingest.py to produce one."
            )
        bucket_size = self._overview_bucket_size or 1
        # Source-sample range we care about.
        i0 = max(0, int(round(t_start * self.fs)))
        i1 = min(self.n_samples, int(round(t_end * self.fs)))
        if i0 >= i1:
            return (np.zeros(0, dtype=np.float64),
                    np.zeros((0, self.n_channels), dtype=np.float32))
        # Map to overview-row range. The overview has 2 rows per
        # bucket (min then max); each bucket covers bucket_size source
        # samples. Bucket b spans source indices [b*bucket_size, (b+1)*bucket_size).
        bb0 = i0 // bucket_size
        bb1 = (i1 + bucket_size - 1) // bucket_size
        n_ov_rows = 2 * (bb1 - bb0)
        # h5py lazy read of exactly those rows.
        data = np.asarray(
            self._y_overview[2 * bb0 : 2 * bb0 + n_ov_rows, :],
            dtype=np.float32,
        )
        # Time axis: each pair of rows shares the same source time
        # (the bucket midpoint). We emit (min, max) at the same t
        # so pyqtgraph draws the vertical span.
        bucket_centers_s = (
            (np.arange(bb0, bb1, dtype=np.float64) + 0.5)
            * bucket_size / self.fs
        )
        # Duplicate each t for the min/max pair → same length as data.
        t = np.repeat(bucket_centers_s, 2)
        return t, data

    def close(self) -> None:
        try:
            self._h5.close()
        except Exception:
            pass

    def __enter__(self) -> "LazyRecording":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"LazyRecording({self.path.name!s}, fs={self.fs:.1f} Hz, "
            f"n_channels={self.n_channels}, n_samples={self.n_samples:,}, "
            f"duration={self.duration_sec:.1f}s)"
        )


class MultiChannelViewer(pg.GraphicsLayoutWidget):
    """N-channel stacked pyqtgraph viewer with linked x-axes.

    On each `set_viewport()` call we:
      1. Pull samples for the requested window from the LazyRecording.
      2. Feed them to pyqtgraph's plot curve with `autoDownsample=True,
         downsampleMethod="peak"` — pyqtgraph's MinMax-in-C downsampler.
         The on-screen pixel count caps the work regardless of viewport
         span.
      3. Hard-set the x-range so pyqtgraph's auto-fit doesn't fight us.

    The plot is never given more data than the viewport contains — no
    full-recording buffer, no incremental update of a giant array.
    """

    # Qt signals — surface user interactions to the main window.
    # `bad_interval_added(start_sec, end_sec)` fires when the user
    # finishes a Shift+drag mark.
    bad_interval_added = Signal(float, float)
    # `stim_boundary_moved(new_idx)` fires when the user drags the
    # stim/recovery InfiniteLine. Value is in 1-based sample index.
    stim_boundary_moved = Signal(int)

    def __init__(
        self,
        recording: LazyRecording,
        parent: Optional[object] = None,
        channel_names: Optional[tuple[str, ...]] = None,
        channel_colors: Optional[tuple[str, ...]] = None,
    ):
        super().__init__(parent)
        self.setBackground("#0e1117")  # match Streamlit dark theme
        self.recording = recording
        names = channel_names or CHANNEL_NAMES_DEFAULT
        colors = channel_colors or CHANNEL_COLORS_DEFAULT

        # Guard for the set_viewport → setXRange → sigRangeChanged →
        # set_viewport infinite loop. Flipped True during programmatic
        # range changes so the user-pan handler skips them.
        self._suppress_range_signal = False

        # Bad-region rendering state. _bad_intervals is the
        # authoritative list (k, 2) 1-based-inclusive sample indices.
        # _bad_region_items[ch] is the list of LinearRegionItem
        # instances currently shown on plot[ch] — kept in sync with
        # _bad_intervals by `set_bad_intervals()`.
        self._bad_intervals: np.ndarray = np.zeros((0, 2), dtype=np.int64)
        self._bad_region_items: list[list[pg.LinearRegionItem]] = [
            [] for _ in range(recording.n_channels)
        ]
        # Pending region (during Shift+drag) on each plot.
        self._pending_items: list[Optional[pg.LinearRegionItem]] = [
            None for _ in range(recording.n_channels)
        ]
        # Stim/recovery boundary InfiniteLine on each plot.
        self._boundary_lines: list[Optional[pg.InfiniteLine]] = [
            None for _ in range(recording.n_channels)
        ]
        # Suppress sigPositionChanged echo when we set the boundary
        # programmatically from set_stim_boundary().
        self._suppress_boundary_signal = False

        # Per-plot Shift+drag tracking. The Shift modifier is checked
        # on mouse-press; if held, we suppress pyqtgraph's default
        # pan and instead grow a green pending region. Released =
        # commit; bad_interval_added emitted.
        self._mark_active: bool = False
        self._mark_start_sec: Optional[float] = None
        self._mark_active_plot_idx: Optional[int] = None

        self.plots: list[tuple[pg.PlotItem, pg.PlotDataItem]] = []
        for ch in range(recording.n_channels):
            p = self.addPlot(row=ch, col=0)
            p.setMouseEnabled(x=True, y=False)
            p.showGrid(x=True, y=False, alpha=0.15)
            # Clamp pan/zoom to the recording's time bounds so the user
            # can't drag past the start or end.
            p.setLimits(
                xMin=0.0, xMax=recording.duration_sec,
                minXRange=1.0 / recording.fs,           # at least 1 sample wide
                maxXRange=recording.duration_sec,
            )
            label = names[ch] if ch < len(names) else f"Ch{ch + 1}"
            p.setLabel("left", label)
            # Hide x tick labels on all but the bottom row to save space.
            if ch < recording.n_channels - 1:
                p.getAxis("bottom").setStyle(showValues=False)
            color = colors[ch % len(colors)]
            curve = p.plot(pen=pg.mkPen(color=color, width=1))
            if self.plots:
                # Link this plot's x-axis to the first plot's so a pan
                # or zoom on any one channel moves all five together.
                p.setXLink(self.plots[0][0])
            # Install our mouse hooks on the ViewBox so we can intercept
            # Shift+drag before pyqtgraph's default pan handler runs.
            ch_index = ch
            vb = p.getViewBox()
            vb.installEventFilter(
                _ShiftDragFilter(self, ch_index, parent=vb)
            )
            self.plots.append((p, curve))
        self.plots[-1][0].setLabel("bottom", "Time (s)")

        # Subscribe the master plot's range-changed signal so mouse
        # pan / scroll-wheel zoom / pinch-to-zoom trigger a re-fetch
        # from the LazyRecording. Without this, user gestures move
        # the ViewBox visually but the displayed curve stays bound
        # to the initial 60s of data.
        self.plots[0][0].sigRangeChanged.connect(self._on_view_range_changed)

        # Initial time range: first 60 seconds, or the whole recording
        # if shorter. NB: we use the name `_time_range` rather than
        # `_viewport` because `viewport()` is an inherited
        # QGraphicsView method (returns the underlying QWidget) and
        # the parent's __init__ calls it before ours has finished —
        # shadowing it with a property breaks pyqtgraph.
        init_end = min(60.0, recording.duration_sec)
        self._time_range: tuple[float, float] = (0.0, init_end)
        self.set_viewport(*self._time_range)

    @property
    def time_range(self) -> tuple[float, float]:
        """Currently displayed `(t_start, t_end)` in seconds."""
        return self._time_range

    def set_viewport(self, t_start: float, t_end: float) -> None:
        """Move the viewport to `[t_start, t_end)` programmatically and
        refresh the plots. No-op if `t_end <= t_start` or the range
        slices to an empty window.

        Used by the toolbar (Reset / Fit all) and by the benchmark.
        Mouse-driven pan/zoom from the user comes through
        `_on_view_range_changed` instead — both call `_refresh_data`.
        """
        if t_end <= t_start:
            return
        self._refresh_data(t_start, t_end)
        # Suppress the sigRangeChanged echo so we don't recurse.
        self._suppress_range_signal = True
        try:
            self.plots[0][0].setXRange(t_start, t_end, padding=0)
        finally:
            self._suppress_range_signal = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _on_view_range_changed(self, view_box, ranges) -> None:
        """pyqtgraph sigRangeChanged handler. `ranges` is
        `[[xmin, xmax], [ymin, ymax]]`. We refresh the curve data for
        the new x-window so panning/zooming with the mouse keeps
        showing real data (not the initial buffer).
        """
        if self._suppress_range_signal:
            return
        x_min, x_max = ranges[0]
        x_min = max(0.0, float(x_min))
        x_max = min(self.recording.duration_sec, float(x_max))
        if x_max <= x_min:
            return
        # Skip negligible changes — avoids re-fetching on micro-drag
        # jitter from a trackpad.
        cur_s, cur_e = self._time_range
        if abs(x_min - cur_s) < 1e-3 and abs(x_max - cur_e) < 1e-3:
            return
        self._refresh_data(x_min, x_max)

    # Viewport span (seconds) above which we switch from reading
    # full /y data to the pre-downsampled /y_overview. Below the
    # threshold full data is fast enough (<200ms read on flat HDF5)
    # and gives the user single-sample fidelity. Above the threshold
    # full reads grow bandwidth-bound — the M1.3 spike measured
    # ~800ms at 600s viewport — so we serve the overview instead.
    # Visual quality at wide zoom is preserved because the overview
    # is min/max-pair downsampling, not naive subsample.
    OVERVIEW_THRESHOLD_SEC = 120.0

    def _refresh_data(self, t_start: float, t_end: float) -> None:
        """Pull samples for `[t_start, t_end)` and push them into the
        per-channel curves. Updates `_time_range`. Does NOT touch the
        view-box range — the caller decides whether to also call
        `setXRange()` (programmatic) or not (the user already moved
        the view via mouse and we're just catching up the data).

        Two read paths:
          - **Narrow viewport** (< OVERVIEW_THRESHOLD_SEC): read full
            samples from /y for single-sample fidelity.
          - **Wide viewport** (≥ OVERVIEW_THRESHOLD_SEC, and the
            recording has /y_overview): read from the pre-downsampled
            min/max-pair summary. ~1000× faster at 7200s viewport;
            visually indistinguishable at zoom levels where the eye
            couldn't resolve individual samples anyway.
        """
        viewport_sec = t_end - t_start
        use_overview = (
            viewport_sec >= self.OVERVIEW_THRESHOLD_SEC
            and self.recording.has_overview
        )
        if use_overview:
            t, data = self.recording.get_overview_for_range(t_start, t_end)
            if data.shape[0] == 0:
                return
            for ch, (_plot, curve) in enumerate(self.plots):
                # No autoDownsample needed — we already gave pyqtgraph
                # ~2 rows per visible pixel.
                curve.setData(t, data[:, ch])
        else:
            data = self.recording.get_range(t_start, t_end)
            if data.shape[0] == 0:
                return
            n = data.shape[0]
            t = np.linspace(
                t_start, t_start + n / self.recording.fs, n, endpoint=False,
            )
            for ch, (_plot, curve) in enumerate(self.plots):
                curve.setData(
                    t, data[:, ch],
                    autoDownsample=True,
                    downsampleMethod="peak",
                )
        self._time_range = (t_start, t_end)

    # ------------------------------------------------------------------
    # Bad-interval rendering (M2.5)
    # ------------------------------------------------------------------

    def set_bad_intervals(self, intervals: np.ndarray) -> None:
        """Replace the displayed bad-region overlays with `intervals`.

        Intervals are `(k, 2)` 1-based inclusive sample indices —
        same convention used throughout the detector backend. Each
        interval is rendered as a red translucent `LinearRegionItem`
        on every channel's plot. The bands stay anchored to time
        coordinates so they move with pan/zoom.
        """
        intervals = (np.asarray(intervals, dtype=np.int64)
                      if intervals is not None
                      else np.zeros((0, 2), dtype=np.int64))
        # Clear existing region items.
        for ch, items in enumerate(self._bad_region_items):
            plot = self.plots[ch][0]
            for it in items:
                plot.removeItem(it)
            self._bad_region_items[ch] = []
        # Render new ones. Color matches the Streamlit UI's bad-region
        # red (#d62728 at ~22% opacity).
        for s, e in intervals:
            s_sec = (int(s) - 1) / self.recording.fs
            e_sec = (int(e) - 1) / self.recording.fs
            for ch, (plot, _curve) in enumerate(self.plots):
                region = pg.LinearRegionItem(
                    values=(s_sec, e_sec),
                    orientation="vertical",
                    movable=False,
                    brush=pg.mkBrush(214, 39, 40, 56),    # (#d62728, alpha 56/255)
                    pen=pg.mkPen(None),
                )
                # Push behind the curve so the line stays visible.
                region.setZValue(-10)
                plot.addItem(region)
                self._bad_region_items[ch].append(region)
        self._bad_intervals = intervals

    @property
    def bad_intervals(self) -> np.ndarray:
        return self._bad_intervals

    def set_model_intervals(self, intervals: Optional[np.ndarray]) -> None:
        """Replace the displayed model-prediction overlays with
        `intervals`. Orange translucent bands (#ff7f0e at ~18% opacity),
        rendered BEHIND the red bad bands so user marks dominate
        visually when they overlap.

        `intervals=None` (or empty) clears the model overlay.
        """
        if not hasattr(self, "_model_region_items"):
            # First call — initialise per-channel lists. (Avoids
            # bloating __init__ with M3-only state.)
            self._model_region_items: list[list[pg.LinearRegionItem]] = [
                [] for _ in range(self.recording.n_channels)
            ]
        for ch, items in enumerate(self._model_region_items):
            plot = self.plots[ch][0]
            for it in items:
                plot.removeItem(it)
            self._model_region_items[ch] = []
        if intervals is None or len(intervals) == 0:
            return
        intervals = np.asarray(intervals, dtype=np.int64)
        for s, e in intervals:
            s_sec = (int(s) - 1) / self.recording.fs
            e_sec = (int(e) - 1) / self.recording.fs
            for ch, (plot, _curve) in enumerate(self.plots):
                region = pg.LinearRegionItem(
                    values=(s_sec, e_sec),
                    orientation="vertical",
                    movable=False,
                    brush=pg.mkBrush(255, 127, 14, 46),    # (#ff7f0e, alpha 46/255)
                    pen=pg.mkPen(None),
                )
                region.setZValue(-15)                       # behind bad bands
                plot.addItem(region)
                self._model_region_items[ch].append(region)

    # ------------------------------------------------------------------
    # Pending-region rendering (during Shift+drag)
    # ------------------------------------------------------------------

    def _start_pending_region(self, plot_idx: int, x_sec: float) -> None:
        """Create the green pending region on every channel,
        seeded at `x_sec`. Called on Shift+mouse-press."""
        self._clear_pending_regions()
        for ch, (plot, _curve) in enumerate(self.plots):
            region = pg.LinearRegionItem(
                values=(x_sec, x_sec),
                orientation="vertical",
                movable=False,
                brush=pg.mkBrush(44, 160, 44, 80),    # green
                pen=pg.mkPen(None),
            )
            region.setZValue(-5)
            plot.addItem(region)
            self._pending_items[ch] = region
        self._mark_active = True
        self._mark_start_sec = x_sec
        self._mark_active_plot_idx = plot_idx

    def _update_pending_region(self, x_sec: float) -> None:
        """Resize the pending region as Shift+drag continues."""
        if not self._mark_active or self._mark_start_sec is None:
            return
        lo = min(self._mark_start_sec, x_sec)
        hi = max(self._mark_start_sec, x_sec)
        for region in self._pending_items:
            if region is not None:
                region.setRegion((lo, hi))

    def _commit_pending_region(self, x_sec: float) -> None:
        """Mouse-release ends the drag. Emit bad_interval_added with
        the (start_sec, end_sec) of the resulting region, then clear
        the pending overlay. Main window appends to its bad-intervals
        model and calls set_bad_intervals() to re-render."""
        if not self._mark_active or self._mark_start_sec is None:
            return
        lo = min(self._mark_start_sec, x_sec)
        hi = max(self._mark_start_sec, x_sec)
        self._clear_pending_regions()
        self._mark_active = False
        self._mark_start_sec = None
        self._mark_active_plot_idx = None
        # Drop zero-width drags (probably a misfire).
        if hi - lo < 1.0 / self.recording.fs:
            return
        self.bad_interval_added.emit(float(lo), float(hi))

    def _cancel_pending_region(self) -> None:
        self._clear_pending_regions()
        self._mark_active = False
        self._mark_start_sec = None
        self._mark_active_plot_idx = None

    def _clear_pending_regions(self) -> None:
        for ch, region in enumerate(self._pending_items):
            if region is not None:
                self.plots[ch][0].removeItem(region)
                self._pending_items[ch] = None

    # ------------------------------------------------------------------
    # Stim/recovery boundary (M2.7)
    # ------------------------------------------------------------------

    def set_stim_boundary(self, stim_end_idx: Optional[int]) -> None:
        """Show or hide a draggable vertical line at the stim/recovery
        boundary. `stim_end_idx` is 1-based inclusive (matches the
        detector backend's convention); `None` hides the line."""
        # Remove existing
        for ch, line in enumerate(self._boundary_lines):
            if line is not None:
                self.plots[ch][0].removeItem(line)
                self._boundary_lines[ch] = None
        if stim_end_idx is None:
            return
        sec = (int(stim_end_idx) - 1) / self.recording.fs
        for ch, (plot, _curve) in enumerate(self.plots):
            line = pg.InfiniteLine(
                pos=sec,
                angle=90,                                # vertical
                movable=True,
                pen=pg.mkPen(color="#cccccc", width=1.5, style=Qt.DashLine),
                hoverPen=pg.mkPen(color="#ffffff", width=2),
            )
            line.setZValue(10)  # above curves
            # Sync drag on any one line to all the others so the
            # boundary stays consistent across channels.
            line.sigPositionChanged.connect(self._on_boundary_dragged)
            plot.addItem(line)
            self._boundary_lines[ch] = line

    def _on_boundary_dragged(self, dragged_line: pg.InfiniteLine) -> None:
        if self._suppress_boundary_signal:
            return
        new_sec = float(dragged_line.value())
        new_idx = max(1, min(self.recording.n_samples,
                              int(round(new_sec * self.recording.fs)) + 1))
        # Mirror the new position to the other boundary lines.
        self._suppress_boundary_signal = True
        try:
            for line in self._boundary_lines:
                if line is not None and line is not dragged_line:
                    line.setValue((new_idx - 1) / self.recording.fs)
        finally:
            self._suppress_boundary_signal = False
        self.stim_boundary_moved.emit(new_idx)


# ----------------------------------------------------------------------
# Shift+drag event filter
# ----------------------------------------------------------------------

class _ShiftDragFilter(QObject):
    """Installed on each plot's ViewBox to intercept mouse events. When
    Shift is held on mouse-press, we suppress pyqtgraph's default pan
    and instead start a "mark" gesture that grows a green region as
    the user drags. On release, the viewer emits
    `bad_interval_added(start_sec, end_sec)`.

    Without the modifier, events pass through to pyqtgraph and pan/zoom
    works as usual.
    """

    def __init__(self, viewer: "MultiChannelViewer", plot_idx: int, parent=None):
        super().__init__(parent)
        self._viewer = viewer
        self._plot_idx = plot_idx
        # We don't claim the press until we confirm Shift is held;
        # this flag tells us we own subsequent moves/releases.
        self._owning = False

    def eventFilter(self, obj, event) -> bool:
        from PySide6.QtCore import QEvent
        # Mouse events on a QGraphicsView ViewBox come as
        # QGraphicsSceneMouseEvent (not QMouseEvent), so we match on
        # event type rather than isinstance.
        et = event.type()
        if et == QEvent.GraphicsSceneMousePress:
            mods = event.modifiers()
            if mods & Qt.ShiftModifier and event.button() == Qt.LeftButton:
                # Map the scene-space click to a data-space x-coord.
                x_sec = self._scene_x_to_data(event.scenePos())
                if x_sec is None:
                    return False
                self._viewer._start_pending_region(self._plot_idx, x_sec)
                self._owning = True
                event.accept()
                return True
            return False
        if et == QEvent.GraphicsSceneMouseMove and self._owning:
            x_sec = self._scene_x_to_data(event.scenePos())
            if x_sec is not None:
                self._viewer._update_pending_region(x_sec)
            event.accept()
            return True
        if et == QEvent.GraphicsSceneMouseRelease and self._owning:
            x_sec = self._scene_x_to_data(event.scenePos())
            if x_sec is not None:
                self._viewer._commit_pending_region(x_sec)
            else:
                self._viewer._cancel_pending_region()
            self._owning = False
            event.accept()
            return True
        return False

    def _scene_x_to_data(self, scene_pos) -> Optional[float]:
        """Convert a scene-coordinate point to data x. Returns None
        if the conversion can't be done (e.g. event from outside the
        view)."""
        plot_item = self._viewer.plots[self._plot_idx][0]
        vb = plot_item.getViewBox()
        try:
            data_pt = vb.mapSceneToView(scene_pos)
            return float(data_pt.x())
        except Exception:
            return None
