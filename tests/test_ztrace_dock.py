"""The z-trace dock: empty while blind, six bands on reveal, verdict in one line."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg

from ui.audit.bridge import BandTrace
from ui.audit.session import SpanSession
from ui.widgets.ztrace_dock import BAND_ORDER, ZTraceDock

ENTER = 3.0


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def _traces() -> list[BandTrace]:
    return [
        BandTrace(b, np.full(50, 0.4 + i), tuple(["RVN1"] * 50), 0.010, ENTER)
        for i, b in enumerate(BAND_ORDER)
    ]


def test_the_dock_is_empty_until_the_reveal(qapp, tmp_path) -> None:
    """z is part of what the reveal reveals. A dock populated during blind
    marking leaks the detector's opinion as surely as the candidate overlay,
    and more subtly, because z is not a candidate and looks harmless."""
    dock = ZTraceDock()
    s = SpanSession("s0", "rec", 0.0, 10.0)

    dock.set_traces(s.traces)          # BLIND -> session returns []

    assert dock.n_curves_with_data() == 0
    assert "after you commit" in dock._verdict.text()


def test_all_six_bands_are_drawn_after_the_reveal(qapp, tmp_path) -> None:
    dock = ZTraceDock()
    s = SpanSession("s0", "rec", 0.0, 10.0)
    s.commit(tmp_path)
    s.reveal(candidates=np.zeros((0, 2)), traces=_traces())

    dock.set_traces(s.traces)

    assert dock.n_curves_with_data() == 6
    assert sorted(dock._plots) == sorted(BAND_ORDER)


def test_z_enter_is_drawn_on_every_band(qapp) -> None:
    """Without the line, 'z was low' and 'z was high but under it' look the same,
    and those are the two diagnoses the dock exists to separate."""
    dock = ZTraceDock()
    dock.set_traces(_traces())

    for band, plot in dock._plots.items():
        lines = [i for i in plot.items if isinstance(i, pg.InfiniteLine)]
        assert len(lines) == 1, f"{band} has {len(lines)} threshold lines"
        assert lines[0].value() == pytest.approx(ENTER)


def test_the_verdict_is_one_line_naming_the_fix(qapp) -> None:
    """'In one glance' means a sentence, not six curves to interpret."""
    dock = ZTraceDock()
    dock.set_traces(_traces())

    head = dock.show_span(0.0, 0.2)

    assert head
    assert "peak z=" in head
    assert "enter" in head


def test_reusing_the_dock_for_a_new_span_clears_the_last_one(qapp) -> None:
    """A dock carried across spans must not still show the previous span's z
    while the next span is being marked blind."""
    dock = ZTraceDock()
    dock.set_traces(_traces())
    assert dock.n_curves_with_data() == 6

    dock.set_traces([])

    assert dock.n_curves_with_data() == 0
    for plot in dock._plots.values():
        assert not [i for i in plot.items if isinstance(i, pg.InfiniteLine)]
