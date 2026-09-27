"""The bridge must reduce correctly and must not lose NaN meaning."""

from __future__ import annotations

import numpy as np
import pytest

from ui.audit.bridge import BandTrace, reduce_to_band_traces


def test_six_bands_collapse_to_six_traces_not_fifty_four() -> None:
    """Nine signals x six bands is unreadable; the panel shows one per band."""
    bands = ("0-2", "0.5-3", "2-50", "10-150", "100-300", "300-3000")
    signals = [f"S{i}" for i in range(9)]
    z = {(s, b): np.zeros(100) for s in signals for b in bands}

    traces = reduce_to_band_traces(z, z_enter=3.0, grid_s=0.010)

    assert len(z) == 54
    assert len(traces) == 6
    assert sorted(t.band for t in traces) == sorted(bands)


def test_the_reduction_is_a_maximum_over_every_signal() -> None:
    """Including raw contacts, per invariant 6 - motion appears on them too, and
    a reduction that skipped them could hide the evidence being looked for."""
    z = {
        ("RVN1", "10-150"): np.array([1.0, 5.0, 2.0]),
        ("T", "10-150"): np.array([4.0, 2.0, 2.0]),
        ("ANT1", "10-150"): np.array([0.0, 0.0, 9.0]),
    }

    t = reduce_to_band_traces(z, z_enter=3.0, grid_s=0.010)[0]

    np.testing.assert_allclose(t.z_max, [4.0, 5.0, 9.0])
    assert t.winner == ("T", "RVN1", "ANT1")


def test_which_signal_won_is_kept_per_frame_for_click_to_expand() -> None:
    """Discarding it would make the panel unable to answer 'which channel?'."""
    z = {("A", "2-50"): np.array([1.0, 7.0]), ("B", "2-50"): np.array([6.0, 2.0])}

    t = reduce_to_band_traces(z, z_enter=3.0, grid_s=0.010)[0]

    assert t.at(0.000) == (6.0, "B")
    assert t.at(0.010) == (7.0, "A")


def test_a_nan_on_one_signal_does_not_poison_the_frame() -> None:
    """NaN means 'not assessable here', not 'low z'. A plain max would turn a
    frame that is perfectly valid on eight signals into NaN and hide it."""
    z = {
        ("A", "0-2"): np.array([np.nan, 2.0]),
        ("B", "0-2"): np.array([3.0, np.nan]),
    }

    t = reduce_to_band_traces(z, z_enter=3.0, grid_s=0.010)[0]

    np.testing.assert_allclose(t.z_max, [3.0, 2.0])
    assert t.winner == ("B", "A")


def test_a_frame_invalid_on_every_signal_stays_nan_with_no_winner() -> None:
    """It must not be reported as a low-z frame - that is the opposite claim."""
    z = {("A", "0-2"): np.array([np.nan]), ("B", "0-2"): np.array([np.nan])}

    t = reduce_to_band_traces(z, z_enter=3.0, grid_s=0.010)[0]

    assert np.isnan(t.z_max[0])
    assert t.winner == ("",)


def test_z_enter_travels_with_the_trace() -> None:
    """The threshold has to be drawable on the plot: 'z was low' and 'z was high
    but under the line' need different fixes and are the audit's whole point."""
    t = reduce_to_band_traces({("A", "0-2"): np.zeros(3)}, z_enter=2.5, grid_s=0.01)[0]

    assert t.z_enter == 2.5


def test_a_time_outside_the_trace_returns_nan_rather_than_wrapping() -> None:
    """Negative indexing would silently report the end of the record."""
    t = BandTrace("0-2", np.array([1.0, 2.0]), ("A", "A"), 0.010, 3.0)

    assert np.isnan(t.at(-1.0)[0])
    assert np.isnan(t.at(99.0)[0])
