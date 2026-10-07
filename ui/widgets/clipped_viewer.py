"""A ``MultiChannelViewer`` with a fixed vertical scale per channel and clipping marked.

Used by the candidate adjudication screen (task 16 Change 1). The scale for each
channel comes from ``ui.adjudicate.hum.channel_scales`` (the ±1 s around the core) and
is set with :meth:`ClippedChannelViewer.set_scales`; panning and zooming in time never
change it. A sample beyond a channel's limits is drawn AT the limit, and every clipped
run is marked by a thick red segment along that limit (:data:`CLIP_COLOUR`), so a
flat-topped stretch is never mistaken for a real plateau. Each channel also carries a
small label in its top-right corner: the half-range (``±S µV``), and ``CLIPPED (n)`` in
red while ``n`` samples in view are clipped. A channel with no scale (no finite sample
near the core) auto-ranges and says so.

Samples are placed at their exact times ``i / fs`` (the base viewer anchors them at the
requested start), so a window that begins before 0 s - a 100 ms zoom at the start of a
file - still draws each sample where it is. NaN is drawn as a gap and never clipped.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Final

import numpy as np
import pyqtgraph as pg

from ui.adjudicate.hum import ChannelScale, clip_to_scale, clipped_segments
from ui.widgets.signal_viewer import MultiChannelViewer

CLIP_COLOUR: Final = "#ff3b3b"
_PLAIN_COLOUR: Final = "#9aa4b2"
_Y_PADDING: Final = 0.05
"""Fraction of the range left above and below the limits, so the marks are visible."""


class ClippedChannelViewer(MultiChannelViewer):
    """Stacked channels with :class:`ChannelScale` limits; clipped samples marked."""

    def __init__(self, recording: Any, *, initial_range: tuple[float, float] | None = None,
                 **kw: Any) -> None:
        """``initial_range`` is drawn once the marks exist; the base class's default
        first 60 s is never rendered (it was replaced at once anyway)."""
        super().__init__(recording, **kw)
        n = len(self.plots)
        self._scales: list[ChannelScale | None] = [None] * n
        self.clipped_in_view: list[int] = [0] * n
        """Per channel, how many samples in the current view are clipped."""
        self.clip_items: list[pg.PlotDataItem] = []
        self.scale_labels: list[pg.LabelItem] = []
        for plot, _curve in self.plots:
            item = pg.PlotDataItem(pen=pg.mkPen(CLIP_COLOUR, width=4), connect="pairs")
            item.setZValue(10)
            plot.addItem(item, ignoreBounds=True)
            self.clip_items.append(item)
            label = pg.LabelItem("", size="8pt", justify="right")
            label.setParentItem(plot.getViewBox())
            label.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-4, 2))
            self.scale_labels.append(label)
        self._ready = True
        if initial_range is not None:
            self.set_viewport(*initial_range)

    # -- the scale ------------------------------------------------------------

    @property
    def scales(self) -> list[ChannelScale | None]:
        """The limits in force, per channel (``None``: auto-ranged)."""
        return list(self._scales)

    def set_scales(self, scales: Sequence[ChannelScale | None], *, redraw: bool = True) -> None:
        """Fix each channel's vertical limits and redraw the current view clipped to them.

        ``redraw=False`` when the caller moves the viewport next (it redraws anyway).
        """
        if len(scales) != len(self.plots):
            msg = f"{len(scales)} scales for {len(self.plots)} channels"
            raise ValueError(msg)
        self._scales = list(scales)
        for (plot, _curve), sc in zip(self.plots, self._scales, strict=True):
            if sc is None:
                plot.enableAutoRange(axis="y")
            else:
                plot.setYRange(sc.lo, sc.hi, padding=_Y_PADDING)
        if redraw:
            self._refresh_data(*self._time_range)

    # -- drawing --------------------------------------------------------------

    def set_viewport(self, t_start: float, t_end: float) -> None:
        """As the base class; ignored until the viewer is fully built."""
        if not getattr(self, "_ready", False):
            self._time_range = (t_start, t_end)
            return
        super().set_viewport(t_start, t_end)

    def _refresh_data(self, t_start: float, t_end: float) -> None:
        if not getattr(self, "_ready", False):
            return
        rec = self.recording
        fs = float(rec.fs)
        data = rec.get_range(t_start, t_end)
        if data.shape[0] == 0:
            return
        i0 = max(0, math.floor(t_start * fs))
        t = (i0 + np.arange(data.shape[0], dtype=np.float64)) / fs
        for ch, (_plot, curve) in enumerate(self.plots):
            sc = self._scales[ch]
            y = data[:, ch]
            yc, mask = clip_to_scale(y, sc)
            curve.setData(t, yc, autoDownsample=True, downsampleMethod="peak")
            xs, ys = clipped_segments(t, y, sc, 1.0 / fs)
            self.clip_items[ch].setData(xs, ys, connect="pairs")
            self.clipped_in_view[ch] = int(mask.sum())
            self._label(ch)
        self._time_range = (t_start, t_end)

    def _label(self, ch: int) -> None:
        sc = self._scales[ch]
        if sc is None:
            text = "no data within ±1 s of the core: auto-ranged"
        elif sc.flat:
            text = f"flat near the core: ±{sc.half_range_uv:.3g} µV"
        else:
            text = f"±{sc.half_range_uv:.3g} µV"
        n = self.clipped_in_view[ch]
        if n:
            text += f" &nbsp;<b><span style='color:{CLIP_COLOUR}'>CLIPPED ({n})</span></b>"
        self.scale_labels[ch].setText(text, color=_PLAIN_COLOUR)
