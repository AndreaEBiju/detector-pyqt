"""A missed artifact must be classifiable as blind spot vs threshold, computed."""

from __future__ import annotations

import numpy as np
import pytest

from ui.audit.bridge import BandTrace
from ui.audit.diagnose import SUB_THRESHOLD_FLOOR, Verdict, diagnose

GRID = 0.010
ENTER = 3.0


def _trace(band: str, z: list[float], who: str = "RVN1") -> BandTrace:
    a = np.asarray(z, dtype=np.float64)
    return BandTrace(band, a, tuple([who] * a.size), GRID, ENTER)


def test_low_z_in_every_band_is_a_generator_blind_spot() -> None:
    """No threshold change finds this. It needs a new band or feature, and
    telling someone to lower z_enter would waste the finding."""
    traces = [_trace("300-3000", [0.2, 0.3]), _trace("10-150", [0.4, 0.1])]

    d = diagnose(traces, 0.0, 0.02)

    assert d.verdict is Verdict.BLIND_SPOT
    assert d.peak_z == pytest.approx(0.4)


def test_z_that_rose_but_did_not_cross_is_a_threshold_problem() -> None:
    """Visible to the existing bands; a lower z_enter would catch it."""
    traces = [_trace("300-3000", [0.2, 2.6])]

    d = diagnose(traces, 0.0, 0.02)

    assert d.verdict is Verdict.SUB_THRESHOLD
    assert d.peak_z == pytest.approx(2.6)
    assert "300-3000" in d.headline


def test_z_above_enter_points_downstream_not_at_the_threshold() -> None:
    """If z crossed and there is still no candidate, the fault is in the merge,
    duration or cardiac-suppression rules - not in z_enter."""
    d = diagnose([_trace("100-300", [5.0])], 0.0, 0.01)

    assert d.verdict is Verdict.DETECTED


def test_the_peak_is_taken_over_every_band_not_one() -> None:
    """A miss is only a blind spot if NO band saw it. One quiet band while
    another shouts is a routing question, not a blind spot."""
    traces = [_trace("0-2", [0.1, 0.1]), _trace("300-3000", [0.1, 4.0], who="T")]

    d = diagnose(traces, 0.0, 0.02)

    assert d.verdict is Verdict.DETECTED
    assert d.peak_band == "300-3000"
    assert d.peak_signal == "T"


def test_all_nan_is_unassessable_not_a_blind_spot() -> None:
    """The opposite claim: nothing was measured, so no conclusion about the
    detector can be drawn. Reporting it as low z would blame the generator for
    a stretch it was never shown."""
    d = diagnose([_trace("0-2", [np.nan, np.nan])], 0.0, 0.02)

    assert d.verdict is Verdict.UNASSESSABLE
    assert np.isnan(d.peak_z)
    assert "UNASSESSABLE" in d.headline


def test_a_nan_frame_does_not_hide_a_valid_one_in_the_same_span() -> None:
    traces = [_trace("2-50", [np.nan, 2.8, np.nan])]

    d = diagnose(traces, 0.0, 0.03)

    assert d.verdict is Verdict.SUB_THRESHOLD
    assert d.peak_z == pytest.approx(2.8)


def test_the_blind_spot_boundary_is_the_declared_floor() -> None:
    """Half of z_enter is a convention, not a measurement - so it is asserted
    here and reported with the verdict rather than buried."""
    just_under = SUB_THRESHOLD_FLOOR * ENTER - 0.01
    just_over = SUB_THRESHOLD_FLOOR * ENTER + 0.01

    assert diagnose([_trace("0-2", [just_under])], 0.0, 0.01).verdict is Verdict.BLIND_SPOT
    assert diagnose([_trace("0-2", [just_over])], 0.0, 0.01).verdict is Verdict.SUB_THRESHOLD


def test_the_headline_names_the_band_and_signal_for_click_to_expand() -> None:
    """Which signal carried the max is the next question a labeller asks."""
    d = diagnose([_trace("100-300", [2.0], who="ANT2")], 0.0, 0.01)

    assert "ANT2" in d.headline
    assert "100-300" in d.headline
