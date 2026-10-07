"""The two optional panels of the candidate adjudication screen (task 16 Change 1, hum).

:class:`SpectrumPanel` draws a :class:`ui.adjudicate.hum.SpectrumResult`: the Welch PSD
of the core's ±1 s window (yellow) and of its reference window (grey), µV²/Hz on a log
axis against Hz, 0-3000 Hz by default (mouse wheel zooms in frequency; the button puts
0-3000 Hz back). Mains lines at 60·k Hz are dotted blue; k×HR lines (k <= 20) dashed
green, only when a stored beat train gives a heart rate. Everything the panel cannot show it SAYS,
in the text above the plot (no beat train, a reference window that overlaps another
core or leaves the region, a signal that could not be built).

:class:`ZoomPanel` shows every raw channel over exactly 100 ms around the core's peak
time (or its centre), at the main plot's vertical scale with the same clipping marks
(:class:`ui.widgets.clipped_viewer.ClippedChannelViewer`), the core extent shaded.

Neither panel computes anything itself; the window decides when (only while its toggle
is on) and where (off the GUI thread for the spectrum).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.adjudicate.hum import (
    F_MAX_HZ,
    HR_K_MAX,
    HR_WINDOW_S,
    PEAK_SIGNAL,
    ChannelScale,
    HeartRate,
    SpectrumResult,
    hr_lines,
    mains_lines,
)
from ui.audit.array_recording import ArrayRecording
from ui.widgets.clipped_viewer import ClippedChannelViewer

NO_BEATS: Final = "no beat train — k×HR not shown"
_CORE_PEN: Final = pg.mkPen("#ffe600", width=2)
_REF_PEN: Final = pg.mkPen("#9aa4b2", width=1.5)
_MAINS_PEN: Final = pg.mkPen("#4c9be8", width=1, style=Qt.PenStyle.DotLine)
_HR_PEN: Final = pg.mkPen("#5cc98a", width=1, style=Qt.PenStyle.DashLine)


def _vertical_pairs(freqs: np.ndarray, y0: float, y1: float) -> tuple[np.ndarray, np.ndarray]:
    """``connect='pairs'`` data drawing one vertical line per frequency, ``y0`` to ``y1``."""
    x = np.repeat(np.asarray(freqs, dtype=np.float64), 2)
    y = np.tile(np.array([y0, y1], dtype=np.float64), freqs.size)
    return x, y


class SpectrumPanel(QWidget):
    """Welch PSD of the core window against its reference window, with hum markers."""

    def __init__(self, channel_names: Sequence[str] = (), parent: QWidget | None = None,
                 ) -> None:
        super().__init__(parent)
        self.channel = QComboBox()
        self.channel.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # never steal the judging keys
        self.channel.setToolTip("The signal analysed. 'Core's peak signal' uses the queue "
                                "row's peak_signal (e.g. L_T, ANT1) when it has one.")
        self.set_channels(channel_names)
        reset = QPushButton("0-3000 Hz")
        reset.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        reset.clicked.connect(self.reset_range)
        top = QHBoxLayout()
        top.addWidget(QLabel("Signal:"))
        top.addWidget(self.channel, 1)
        top.addWidget(reset)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#0e1117")
        self.plot.setLogMode(x=False, y=True)
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        self.plot.setLabel("left", "PSD (µV²/Hz)")
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.addLegend(offset=(-10, 10))
        self.core_curve = self.plot.plot(pen=_CORE_PEN, name="core ±1 s")
        self.ref_curve = self.plot.plot(pen=_REF_PEN, name="reference")
        self.mains_item = pg.PlotDataItem(pen=_MAINS_PEN, connect="pairs")
        self.hr_item = pg.PlotDataItem(pen=_HR_PEN, connect="pairs")
        for item in (self.mains_item, self.hr_item):
            item.setZValue(-1)
            self.plot.addItem(item, ignoreBounds=True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addLayout(top)
        lay.addWidget(self.info)
        lay.addWidget(self.plot, 1)
        self.result: SpectrumResult | None = None
        self.mains_x = np.zeros(0)
        """The frequencies of the 60·k lines drawn (Hz)."""
        self.hr_x = np.zeros(0)
        """The frequencies of the k×HR lines drawn (Hz); empty without a beat train."""
        self._y = (1e-6, 1.0)
        self.reset_range()

    def set_channels(self, names: Sequence[str]) -> None:
        """Offer :data:`PEAK_SIGNAL` and each raw channel; keep the current choice if listed."""
        current = self.channel.currentText() or PEAK_SIGNAL
        self.channel.blockSignals(True)
        self.channel.clear()
        self.channel.addItems([PEAK_SIGNAL, *names])
        i = self.channel.findText(current)
        self.channel.setCurrentIndex(max(i, 0))
        self.channel.blockSignals(False)

    @property
    def selected(self) -> str:
        """The selector's choice: :data:`PEAK_SIGNAL` or a raw channel name."""
        return self.channel.currentText() or PEAK_SIGNAL

    def reset_range(self) -> None:
        """Back to 0-3000 Hz."""
        self.plot.setXRange(0.0, F_MAX_HZ, padding=0)

    @property
    def message(self) -> str:
        """The panel's text, as plain text."""
        return self.info.text()

    def show_pending(self, what: str) -> None:
        """Say the spectrum is being computed; keep the old curves off screen."""
        self.result = None
        self.core_curve.setData([], [])
        self.ref_curve.setData([], [])
        self.mains_item.setData([], [])
        self.hr_item.setData([], [])
        self.mains_x = self.hr_x = np.zeros(0)
        self.info.setText(what)

    def show_result(self, res: SpectrumResult, hr: HeartRate | None,
                    beats_note: str | None) -> None:
        """Draw ``res``; ``hr`` from the recording's stored beats (``None``: no train)."""
        self.result = res
        ys = []
        for curve, psd in ((self.core_curve, res.core), (self.ref_curve, res.ref)):
            if psd is None:
                curve.setData([], [])
                continue
            keep = psd.power > 0  # a log axis cannot show 0 (a flat signal)
            curve.setData(psd.freqs[keep], psd.power[keep])
            ys.append(psd.power[keep & (psd.freqs <= F_MAX_HZ)])
        pos = np.concatenate(ys) if ys else np.zeros(0)
        if pos.size:
            lo, hi = float(np.log10(pos.min())) - 0.5, float(np.log10(pos.max())) + 0.5
            self.plot.setYRange(lo, hi, padding=0)
            self._y = (10.0 ** lo, 10.0 ** hi)
        self.mains_x = mains_lines()
        self.mains_item.setData(*_vertical_pairs(self.mains_x, *self._y))
        self.set_hr(hr, beats_note)

    def set_hr(self, hr: HeartRate | None, beats_note: str | None) -> None:
        """Draw (or withdraw) the k×HR lines and rewrite the text."""
        res = self.result
        if res is None:
            return
        if hr is not None and hr.hz is not None:
            self.hr_x = hr_lines(hr.hz)
            refused = f", {hr.n_refused} refused" if hr.n_refused else ""
            hr_text = (f"HR {hr.hz:.2f} Hz ({60 * hr.hz:.0f} bpm): 1/median of {hr.n_rr} "
                       f"valid RR intervals within ±{HR_WINDOW_S:.0f} s{refused}"
                       f"{'; ' + beats_note if beats_note else ''}. k×HR lines (k ≤ "
                       f"{HR_K_MAX}) dashed green")
        else:
            self.hr_x = np.zeros(0)
            why = (hr.reason if hr is not None else None) or beats_note
            hr_text = NO_BEATS + (f" ({why})" if why else "")
        self.hr_item.setData(*_vertical_pairs(self.hr_x, *self._y))
        segs = [f"{p.n_used}/{p.n_segments}" if p is not None else "none"
                for p in (res.core, res.ref)]
        w = res.ref_window
        ref = ("" if w is None else
               f"reference {w.start_s:.2f}-{w.stop_s:.2f} s ({w.offset_s:.1f} s {w.side} "
               "the core)")
        head = (f"<b>{res.signal}</b>, core window "
                f"{res.window_s[0]:.2f}-{res.window_s[1]:.2f} s" + (f"; {ref}" if ref else ""))
        welch = (f"Welch: nperseg {res.nperseg}, Hann, 50% overlap; segments used core "
                 f"{segs[0]}, reference {segs[1]}. 60·k Hz dotted blue.")
        lines = [head, welch, hr_text]
        lines += [f"<span style='color:#e8834c'>{n}</span>" for n in res.notes]
        self.info.setText("<br>".join(lines))


class ZoomPanel(QWidget):
    """Every raw channel over 100 ms around the core, at the main plot's scale."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(4, 4, 4, 4)
        self._lay.addWidget(self.info)
        self.viewer: ClippedChannelViewer | None = None
        self._recording: Any = None
        self._core_items: list[pg.LinearRegionItem] = []

    def set_recording(self, recording: Any, names: Sequence[str],
                      colours: Sequence[str]) -> None:
        """Build the zoom's viewer for ``recording`` (same array, no copy)."""
        self.release()
        self._recording = recording
        arr = ArrayRecording(recording.data, recording.fs, copy_to_float32=False)
        self.viewer = ClippedChannelViewer(arr, channel_names=tuple(names),
                                           channel_colors=tuple(colours))
        self._lay.addWidget(self.viewer, 1)

    def clear(self) -> None:
        """Show nothing (a new core is coming, or none); keep the viewer for reuse."""
        if self.viewer is not None:
            self.viewer.hide()
        self.info.setText("")

    @property
    def showing(self) -> bool:
        """Whether a core is on show (not cleared, not released)."""
        return self.viewer is not None and not self.viewer.isHidden()

    def release(self) -> None:
        """Drop the viewer and the recording it holds."""
        if self.viewer is not None:
            v, self.viewer = self.viewer, None
            v.plots = []
            v.clear()
            v.recording = None  # type: ignore[assignment]
            v.setParent(None)
            v.deleteLater()
        self._recording = None
        self._core_items = []

    @property
    def recording(self) -> Any:
        """The recording the zoom shows (``None`` before :meth:`set_recording`)."""
        return self._recording

    def show_core(self, window: tuple[float, float, float, str], core: tuple[float, float],
                  scales: Sequence[ChannelScale | None]) -> None:
        """Show ``window`` = ``(centre, lo, hi, basis)`` with ``core`` shaded."""
        assert self.viewer is not None
        centre, lo, hi, basis = window
        for (plot, _c), item in zip(self.viewer.plots, self._core_items, strict=False):
            plot.removeItem(item)
        self._core_items = []
        self.viewer.set_scales(scales)
        self.viewer.set_viewport(lo, hi)
        self.viewer.show()
        for plot, _curve in self.viewer.plots:
            item = pg.LinearRegionItem(values=core, orientation="vertical", movable=False,
                                       brush=pg.mkBrush(255, 230, 0, 70),
                                       pen=pg.mkPen("#ffe600", width=1))
            item.setZValue(-5)
            plot.addItem(item)
            self._core_items.append(item)
        self.info.setText(f"100 ms around the {basis} ({centre:.4f} s): {lo:.4f}-{hi:.4f} s, "
                          "every channel at the main plot's scale; core shaded yellow")
