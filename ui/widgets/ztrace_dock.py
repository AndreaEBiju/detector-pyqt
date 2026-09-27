"""Six band z-traces, shown only after the reveal.

One plot per band, each showing the per-frame MAXIMUM of z over **every** signal
including the raw contacts (invariant 6 - motion appears on them too, and a
reduction that skipped them could hide the evidence the audit is looking for),
with ``z_enter`` drawn as a horizontal line.

Six rather than fifty-four because a labeller cannot read fifty-four. Which
signal carried the maximum is kept per frame and shown on click, not by default:
the first question is "did anything see this?", and the second is "what".

**Empty until the reveal.** The traces are part of what the reveal reveals - a
dock that populated during blind marking would leak the detector's opinion just
as surely as the candidate overlay, and more subtly, because z is not a
candidate and might look harmless.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ui.audit.diagnose import diagnose

BAND_ORDER = ("300-3000", "100-300", "10-150", "2-50", "0.5-3", "0-2")
"""Fast to slow, so the eye travels the same way as the consumer table."""


class ZTraceDock(QWidget):
    """Stacked per-band z-traces with the threshold drawn, plus a verdict line."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._traces: list[Any] = []
        self._plots: dict[str, pg.PlotItem] = {}
        self._curves: dict[str, pg.PlotDataItem] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self._verdict = QLabel("z-traces appear after you commit your marks.")
        self._verdict.setWordWrap(True)
        self._verdict.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._verdict)

        self._detail = QLabel("")
        self._detail.setWordWrap(True)
        layout.addWidget(self._detail)

        self._glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw, stretch=1)

        first: pg.PlotItem | None = None
        for row, band in enumerate(BAND_ORDER):
            plot = self._glw.addPlot(row=row, col=0)
            plot.setLabel("left", band)
            plot.showGrid(x=False, y=True, alpha=0.2)
            plot.setMouseEnabled(x=True, y=False)
            if first is None:
                first = plot
            else:
                plot.setXLink(first)
            if row < len(BAND_ORDER) - 1:
                plot.getAxis("bottom").setStyle(showValues=False)
            self._plots[band] = plot
            self._curves[band] = plot.plot([], [], pen=pg.mkPen("#1f77b4", width=1))
        if first is not None:
            first.setLabel("bottom", "time (s)")

    # ------------------------------------------------------------------
    def set_traces(self, traces: list[Any]) -> None:
        """Populate, or clear when given an empty list.

        Clearing on empty is what keeps the dock honest across spans: a dock
        reused for the next span must not still be showing the last one's z.
        """
        self._traces = list(traces)
        by_band = {t.band: t for t in self._traces}
        for band in BAND_ORDER:
            plot = self._plots[band]
            curve = self._curves[band]
            for item in list(plot.items):
                if isinstance(item, pg.InfiniteLine):
                    plot.removeItem(item)
            tr = by_band.get(band)
            if tr is None:
                curve.setData([], [])
                continue
            t = np.arange(tr.z_max.size) * tr.grid_s
            curve.setData(t, np.nan_to_num(tr.z_max, nan=0.0))
            line = pg.InfiniteLine(
                pos=tr.z_enter, angle=0,
                pen=pg.mkPen("#d62728", width=1, style=Qt.DashLine),
            )
            plot.addItem(line)
        if not self._traces:
            self._verdict.setText("z-traces appear after you commit your marks.")
            self._detail.setText("")

    def n_curves_with_data(self) -> int:
        """How many band curves actually hold points. For the offscreen test."""
        return sum(
            1 for c in self._curves.values()
            if c.xData is not None and len(c.xData) > 0
        )

    def show_span(self, t_start_s: float, t_stop_s: float) -> str:
        """Diagnose a marked span and display the verdict. Returns the headline."""
        if not self._traces:
            return ""
        d = diagnose(self._traces, t_start_s, t_stop_s)
        self._verdict.setText(d.headline)
        self._detail.setText(
            "blind spot -> needs a new band or feature; "
            "threshold too high -> lower z_enter; "
            "already above threshold -> the fault is downstream of z"
        )
        for band, plot in self._plots.items():
            plot.setTitle(
                "<b>peak</b>" if band == d.peak_band else None
            )
        return d.headline
