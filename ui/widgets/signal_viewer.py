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

        self.plots: list[tuple[pg.PlotItem, pg.PlotDataItem]] = []
        for ch in range(recording.n_channels):
            p = self.addPlot(row=ch, col=0)
            p.setMouseEnabled(x=True, y=False)
            p.showGrid(x=True, y=False, alpha=0.15)
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
            self.plots.append((p, curve))
        self.plots[-1][0].setLabel("bottom", "Time (s)")

        # Initial viewport: first 60 seconds, or the whole recording
        # if shorter.
        init_end = min(60.0, recording.duration_sec)
        self._viewport: tuple[float, float] = (0.0, init_end)
        self.set_viewport(*self._viewport)

    @property
    def viewport(self) -> tuple[float, float]:
        return self._viewport

    def set_viewport(self, t_start: float, t_end: float) -> None:
        """Move the viewport to `[t_start, t_end)` and refresh the
        plots. No-op if `t_end <= t_start` or the requested range is
        empty.
        """
        if t_end <= t_start:
            return
        data = self.recording.get_range(t_start, t_end)
        if data.shape[0] == 0:
            return
        # The slice may be shorter than requested if t_end exceeds the
        # recording's duration — derive the actual time axis from the
        # returned sample count rather than assuming it matches.
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
        # Lock the x-range so pyqtgraph doesn't auto-fit past our
        # viewport on the next paint cycle.
        self.plots[0][0].setXRange(t_start, t_end, padding=0)
        self._viewport = (t_start, t_end)
