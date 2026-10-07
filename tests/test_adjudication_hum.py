"""Change 1: the vertical scale from ±1 s around the core, clipping marks, hum panels.

Offscreen, seeded, hermetic: every recording comes from ``tests/conftest.py``'s
generators, queues and judgements live in ``tmp_path``, and the windows run their
panel jobs synchronously so each assertion sees a finished computation.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import gc
import math
import threading
import time
import weakref
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from scipy.io import savemat
from scipy.signal import welch

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from ui.adjudicate import hum, loaders
from ui.adjudicate.loaders import BeatSource
from ui.widgets.hum_panels import NO_BEATS

FS = 2000.0


# ---------------------------------------------------------------------------
# A. the vertical scale
# ---------------------------------------------------------------------------


def test_the_estimator_is_the_documented_one(make_recording) -> None:
    rec = make_recording(seed=1)
    i0, i1 = hum.core_window(30.0, 30.05, FS, rec.n_samples)
    assert (i0, i1) == (math.floor(29.0 * FS), math.ceil(31.05 * FS))
    x = rec.data[i0:i1, 3]
    med = np.median(x)
    s = np.percentile(np.abs(x - med), 99.5)
    sc = hum.channel_scales(rec.data, FS, 30.0, 30.05)[3]
    assert sc is not None and not sc.flat
    assert hum.SCALE_HEADROOM == 1.5  # Andrea, 2026-10-08
    assert sc.centre_uv == pytest.approx(med) and sc.half_range_uv == pytest.approx(1.5 * s)
    assert (sc.lo, sc.hi) == pytest.approx((med - 1.5 * s, med + 1.5 * s))


def test_the_scale_ignores_a_huge_event_1_5_s_away(make_recording, add_burst) -> None:
    quiet = make_recording(seed=2)
    rec = make_recording(seed=2)
    add_burst(rec, 30.0, 0.05, 60.0)            # the core: a modest burst
    add_burst(quiet, 30.0, 0.05, 60.0)
    add_burst(rec, 31.55, 0.1, 5000.0)          # 1.5 s after the core ends, in the 4 s view
    with_event = hum.channel_scales(rec.data, FS, 30.0, 30.05)
    without = hum.channel_scales(quiet.data, FS, 30.0, 30.05)
    for a, b in zip(with_event, without, strict=True):
        assert a is not None and b is not None
        assert a.half_range_uv == pytest.approx(b.half_range_uv)
        assert a.half_range_uv < 100.0
    # the whole 4 s view would have been dominated by the event
    view = rec.data[round(28.0 * FS):round(32.05 * FS), 0]
    assert np.percentile(np.abs(view - np.median(view)), 99.5) > 10 * with_event[0].half_range_uv


def test_clipping_is_at_the_limit_and_nan_stays_nan() -> None:
    sc = hum.ChannelScale(0.0, 10.0, flat=False)
    y = np.array([0.0, 5.0, 25.0, 26.0, np.nan, -30.0, 3.0, 11.0])
    yc, mask = hum.clip_to_scale(y, sc)
    assert np.array_equal(mask, [False, False, True, True, False, True, False, True])
    assert yc[[2, 3, 7]] == pytest.approx([10.0, 10.0, 10.0]) and yc[5] == -10.0
    assert np.isnan(yc[4]) and yc[1] == 5.0
    t = np.arange(y.size) / FS
    xs, ys = hum.clipped_segments(t, y, sc, 1 / FS)
    # three runs: [2, 3] top, [7] top, [5] bottom - each a segment at its limit
    segs = sorted(zip(xs[0::2], xs[1::2], ys[0::2], strict=True))
    assert segs == pytest.approx([(2 / FS, 4 / FS, 10.0), (5 / FS, 6 / FS, -10.0),
                                  (7 / FS, 8 / FS, 10.0)])
    assert np.all(ys[0::2] == ys[1::2])
    assert hum.clipped_segments(t, np.zeros(8), sc, 1 / FS)[0].size == 0


def test_flat_and_all_nan_channels_and_partial_nan(make_recording) -> None:
    rec = make_recording(seed=3)
    rec.data[:, 0] = 7.5                     # flat
    rec.data[:, 1] = np.nan                  # nothing finite
    rec.data[round(29.5 * FS):round(30.5 * FS), 2] = np.nan  # a masked second
    sc = hum.channel_scales(rec.data, FS, 30.0, 30.05)
    assert sc[0] is not None and sc[0].flat
    assert (sc[0].centre_uv, sc[0].half_range_uv) == (7.5, hum.FLAT_HALF_RANGE_UV)
    assert sc[1] is None
    i0, i1 = hum.core_window(30.0, 30.05, FS, rec.n_samples)
    x = rec.data[i0:i1, 2]
    expect = hum.scale_of(x[np.isfinite(x)])
    assert sc[2] == expect


@pytest.mark.parametrize("core", [(0.2, 0.25), (59.9, 59.95)])
def test_near_a_file_edge_the_window_is_truncated_never_padded(make_recording,
                                                               core) -> None:
    rec = make_recording(seed=4)
    rec.data[:200, 0] = 0.0  # zeros that WOULD be what padding looks like... at the start
    i0, i1 = hum.core_window(*core, FS, rec.n_samples)
    assert i0 >= 0 and i1 <= rec.n_samples
    assert (i0 == 0) if core[0] < 1 else (i1 == rec.n_samples)
    got = hum.channel_scales(rec.data, FS, *core)[3]
    assert got == hum.scale_of(rec.data[i0:i1, 3])


def test_the_window_applies_the_scale_and_marks_the_clipping(make_hum_window, make_recording,
                                                             add_burst, queue_row) -> None:
    rec = make_recording(seed=5, session="r1")
    add_burst(rec, 30.0, 0.05, 60.0)
    add_burst(rec, 31.55, 0.1, 5000.0, channels=[0])
    w = make_hum_window([queue_row("r1", 30.0, 30.05), queue_row("r1", 45.0, 45.02)],
                        {"r1": rec})
    v = w.viewer
    sc = hum.channel_scales(rec.data, FS, 30.0, 30.05)
    assert v.scales == sc
    (_x, (ylo, yhi)) = v.plots[0][0].getViewBox().viewRange()
    assert ylo < sc[0].lo < sc[0].hi < yhi and (yhi - ylo) < 1.2 * (sc[0].hi - sc[0].lo)
    # the huge event is in view, clipped, and marked on channel 0 only
    lo, hi = v.time_range
    assert lo <= 31.55 < hi
    assert v.clipped_in_view[0] > 100
    xs, ys = v.clip_items[0].getData()
    assert xs.size and np.all((ys == sc[0].hi) | (ys == sc[0].lo))
    assert np.any((xs >= 31.55) & (xs <= 31.66))
    assert "CLIPPED" in v.scale_labels[0].text
    curve_y = v.plots[0][1].yData
    assert np.nanmax(curve_y) <= sc[0].hi and np.nanmin(curve_y) >= sc[0].lo


def test_panning_keeps_the_scale_home_reapplies_it_and_a_new_core_recomputes_it(
        make_hum_window, make_recording, add_burst, queue_row) -> None:
    rec = make_recording(seed=6, session="r1")
    add_burst(rec, 45.0, 0.05, 400.0)   # the second core is much larger
    w = make_hum_window([queue_row("r1", 30.0, 30.05), queue_row("r1", 45.0, 45.05)],
                        {"r1": rec})
    v = w.viewer
    first = v.scales
    v.set_viewport(43.0, 47.0)          # pan onto the big burst
    assert v.scales == first
    assert v.plots[0][0].getViewBox().viewRange()[1][1] < 2 * first[0].hi + 50
    v.plots[0][0].setYRange(-1e4, 1e4)  # anything that moved the y-range ...
    w.recentre()                         # ... Home puts the core's scale back
    ylo, yhi = v.plots[0][0].getViewBox().viewRange()[1]
    assert ylo < first[0].lo and yhi > first[0].hi and yhi - ylo < 1.2 * (first[0].hi
                                                                          - first[0].lo)
    w.press("2")
    assert w.viewer.scales == hum.channel_scales(rec.data, FS, 45.0, 45.05)
    assert w.viewer.scales[0].half_range_uv > 3 * first[0].half_range_uv


# ---------------------------------------------------------------------------
# B. panels
# ---------------------------------------------------------------------------


def test_panels_are_off_by_default_and_then_nothing_is_computed(
        make_hum_window, make_recording, queue_row, monkeypatch) -> None:
    calls: list[str] = []
    real_welch, real_spec, real_zoom = hum.welch_psd, hum.compute_spectrum, hum.zoom_window
    monkeypatch.setattr(hum, "welch_psd", lambda *a, **k: calls.append("welch")
                        or real_welch(*a, **k))
    monkeypatch.setattr(hum, "compute_spectrum", lambda *a, **k: calls.append("spectrum")
                        or real_spec(*a, **k))
    monkeypatch.setattr(hum, "zoom_window", lambda *a, **k: calls.append("zoom")
                        or real_zoom(*a, **k))
    beats_calls: list[str] = []

    def beats_fn(row, _rec):
        beats_calls.append(row["recording"])
        return BeatSource(None, None, "none")

    rec = make_recording(seed=7, session="r1")
    w = make_hum_window([queue_row("r1", 10.0, 10.05), queue_row("r1", 20.0, 20.05),
                         queue_row("r1", 30.0, 30.05)], {"r1": rec}, beats_fn=beats_fn)
    assert not w.spectrum_box.isChecked() and not w.zoom_box.isChecked()
    assert w._spectrum_dock.isHidden() and w._zoom_dock.isHidden()
    w.press("1")
    w.skip()
    w.recentre()
    assert calls == [] and beats_calls == []
    assert w.zoom_panel.viewer is None and w.spectrum.result is None
    w.spectrum_box.setChecked(True)
    assert calls.count("spectrum") == 1 and calls.count("welch") == 2  # core + reference
    assert beats_calls == ["r1"]
    w.zoom_box.setChecked(True)
    assert "zoom" in calls
    w.spectrum_box.setChecked(False)
    w.zoom_box.setChecked(False)
    n = len(calls)
    w.press("1")
    assert len(calls) == n and w.zoom_panel.viewer is None


def test_mains_and_heart_rate_lines_sit_at_60k_and_k_hr(make_hum_window, make_recording,
                                                         queue_row) -> None:
    rec = make_recording(seed=8, session="r1", fs=8000.0)
    hr = 6.0
    beats = np.arange(0.05, 60.0, 1 / hr)

    def beats_fn(_row, _rec):
        return BeatSource(beats, "r1_HRBR.mat", None)

    w = make_hum_window([queue_row("r1", 30.0, 30.05)], {"r1": rec}, beats_fn=beats_fn)
    w.spectrum_box.setChecked(True)
    sp = w.spectrum
    assert sp.mains_x == pytest.approx(60.0 * np.arange(1, 51))
    assert sp.hr_x.size == hum.HR_K_MAX == 20  # 6, 12, ..., 120 Hz: capped at k = 20
    assert sp.hr_x == pytest.approx(hr * np.arange(1, 21))
    x, _y = sp.hr_item.getData()
    assert np.array_equal(np.unique(x), np.unique(sp.hr_x))
    assert "r1_HRBR.mat" in sp.message and NO_BEATS not in sp.message


def test_without_a_beat_train_the_panel_says_so(make_hum_window, make_recording,
                                                queue_row) -> None:
    rec = make_recording(seed=9, session="r1")
    w = make_hum_window([queue_row("r1", 30.0, 30.05)], {"r1": rec},
                        beats_fn=lambda _r, _x: BeatSource(None, None,
                                                           loaders.NEW_COHORT_NO_BEATS))
    w.spectrum_box.setChecked(True)
    assert NO_BEATS in w.spectrum.message
    assert loaders.NEW_COHORT_NO_BEATS in w.spectrum.message
    drawn = w.spectrum.hr_item.getData()[0]
    assert w.spectrum.hr_x.size == 0 and (drawn is None or drawn.size == 0)
    assert w.spectrum.mains_x.size == 50


def test_heart_rate_is_the_median_of_valid_rr_not_a_mean_across_a_blanked_gap() -> None:
    """The review's example: beats dropped in a blanked stretch around a core at 30 s."""
    beats = np.array([29.0, 29.15, 30.9, 31.0])
    old_mean = (beats.size - 1) / (beats[-1] - beats[0])
    assert old_mean == pytest.approx(1.5)            # what the first version reported
    hr = hum.heart_rate(beats, 30.0 - hum.HR_WINDOW_S, 30.05 + hum.HR_WINDOW_S)
    assert hr.hz == pytest.approx(1 / 0.125)          # median of 0.15 and 0.10
    assert (hr.n_rr, hr.n_refused) == (2, 1)          # 1.75 s is outside 80-500 ms
    tagged = hum.heart_rate(beats, 20.0, 40.0, gap_after=[True, False, False, False])
    assert tagged.hz is None and tagged.n_rr == 1
    assert "1 valid RR" in (tagged.reason or "") and "k×HR not shown" in (tagged.reason or "")


def test_heart_rate_refuses_gaps_blank_spans_and_implausible_rr() -> None:
    reg = np.round(np.arange(20.0, 40.0, 0.15), 6)
    assert hum.heart_rate(reg, 20.0, 40.0).hz == pytest.approx(1 / 0.15)
    # a blank span refuses every interval it overlaps, even a plausible one
    span = [[30.0, 30.2]]
    across = int(np.sum((reg[:-1] < 30.2) & (reg[1:] > 30.0)))
    b = hum.heart_rate(reg, 20.0, 40.0, blank_spans=span)
    assert b.n_refused == across >= 2 and b.hz == pytest.approx(1 / 0.15)
    # gap tags follow the beats as given, even unsorted
    shuffled = reg[::-1].copy()
    tags = np.zeros(reg.size, dtype=bool)
    tags[-1] = True                                   # shuffled[-1] is reg[0]
    g = hum.heart_rate(shuffled, 20.0, 40.0, gap_after=tags)
    assert g.n_refused == 1
    # an implausible interval (600 ms) and beats outside the window are ignored
    odd = np.concatenate([reg[reg < 25.0], reg[reg >= 25.6]])
    o = hum.heart_rate(odd, 20.0, 40.0)
    assert o.n_refused == 1 and o.hz == pytest.approx(1 / 0.15)
    # an untagged missed beat (one 300 ms interval) cannot pull the median down
    missed = np.delete(reg, 40)
    m = hum.heart_rate(missed, 20.0, 40.0)
    assert m.n_refused == 0 and m.hz == pytest.approx(1 / 0.15)
    far = np.concatenate([reg, np.arange(50.0, 60.0, 0.4)])
    assert hum.heart_rate(far, 20.0, 40.0).hz == pytest.approx(1 / 0.15)
    assert hum.hr_lines(float("nan")).size == 0
    assert hum.hr_lines(6.0, k_max=100).size == 100 and hum.hr_lines(6.0).size == 20


# -- the reference window -----------------------------------------------------


def _ref(start=100.0, stop=100.1, length=2.1, region=(0.0, 600.0), dur=600.0, others=()):
    return hum.place_reference(start, stop, length, region, dur, others)


def test_reference_is_5_s_after_the_core_when_that_is_clean() -> None:
    w = _ref()
    assert (w.side, w.offset_s) == ("after", 5.0) and w.note is None
    assert (w.start_s, w.stop_s) == pytest.approx((105.1, 107.2))


def test_reference_goes_before_when_after_leaves_the_region() -> None:
    w = _ref(region=(0.0, 106.0))
    assert (w.side, w.offset_s, w.in_region) == ("before", 5.0, True)
    assert (w.start_s, w.stop_s) == pytest.approx((92.9, 95.0))


def test_reference_avoids_other_cores_in_the_fixed_order() -> None:
    # after@5 and before@5 both overlap a core; after@5.5 is clear
    w = _ref(others=[(105.2, 105.3), (94.0, 94.05)])
    assert (w.side, w.offset_s, w.overlaps) == ("after", 5.5, 0)
    # every in-region candidate overlaps: the overlap is SAID, not silent
    w2 = _ref(region=(90.0, 118.0), others=[(91.0, 98.0), (104.0, 118.0)])
    assert w2.in_region and w2.overlaps and "overlaps" in (w2.note or "")
    # nothing fits the region: a clean window outside it, and the panel says so
    w3 = _ref(region=(99.0, 101.0))
    assert not w3.in_region and not w3.overlaps and "region" in (w3.note or "")
    assert _ref(dur=7.0, start=0.5, stop=0.6, region=(0.0, 7.0)) is None


@pytest.mark.parametrize("seed", range(20))
def test_reference_is_always_same_length_and_5_to_10_s_away(seed) -> None:
    rng = np.random.default_rng(seed)
    start = float(rng.uniform(0, 100))
    stop = start + float(rng.uniform(0.01, 0.5))
    others = [(float(a), float(a) + 0.05) for a in rng.uniform(0, 120, 6)]
    w = _ref(start, stop, 2.0 + stop - start, (0.0, 120.0), 120.0, others)
    if w is None:
        return
    assert w.stop_s - w.start_s == pytest.approx(2.0 + stop - start)
    gap = w.start_s - stop if w.side == "after" else start - w.stop_s
    assert 5.0 - 1e-9 <= gap <= 10.0 + 1e-9 and gap == pytest.approx(w.offset_s)
    assert 0.0 <= w.start_s and w.stop_s <= 120.0


def test_the_panel_names_a_missing_reference(make_hum_window, make_recording,
                                             queue_row) -> None:
    rec = make_recording(seed=10, session="r1", dur_s=8.0)
    w = make_hum_window([queue_row("r1", 3.0, 3.05, region=(0.0, 8.0))], {"r1": rec})
    w.spectrum_box.setChecked(True)
    assert "no reference window: it needs 7.0 s clear on one side" in w.spectrum.message
    assert "5.0 s after it and 3.0 s before it" in w.spectrum.message
    assert w.spectrum.result.ref is None and w.spectrum.result.core is not None


def test_a_masked_reference_window_is_named_as_such(make_hum_window, make_recording,
                                                    queue_row) -> None:
    rec = make_recording(seed=17, session="r1")
    rec.data[round(25.0 * FS):round(27.2 * FS), :] = np.nan   # where the reference goes
    w = make_hum_window([queue_row("r1", 20.0, 20.05)], {"r1": rec})
    w.spectrum_box.setChecked(True)
    assert "reference window found, but it has no NaN-free Welch segment" in w.spectrum.message
    assert "no reference window:" not in w.spectrum.message


# -- the spectrum's signal and Welch ------------------------------------------


def test_welch_matches_scipy_and_drops_nan_segments(make_recording) -> None:
    rec = make_recording(seed=11)
    x = rec.data[:4200, 0].copy()
    got = hum.welch_psd(x, FS, 512)
    f, p = welch(x, FS, nperseg=512)
    assert got is not None and got.n_used == got.n_segments
    assert got.freqs == pytest.approx(f) and got.power == pytest.approx(p)
    x[1000] = np.nan
    nan_psd = hum.welch_psd(x, FS, 512)
    assert nan_psd is not None and nan_psd.n_used == nan_psd.n_segments - 2
    assert np.all(np.isfinite(nan_psd.power))
    assert hum.welch_nperseg(24414.0625) == 8192 and hum.welch_nperseg(FS) == 512


def test_the_peak_signal_is_built_as_detection_builds_it(make_recording) -> None:
    rec = make_recording(seed=12)
    lt = hum.signal_window(rec, 100, 300, "L_T")
    v1, v2, v3 = (rec.data[100:300, i] for i in (3, 4, 5))
    assert lt == pytest.approx(0.5 * v1 + 0.5 * v3 - v2)
    assert hum.signal_window(rec, 100, 300, "R_V2") == pytest.approx(rec.data[100:300, 1])
    assert hum.signal_window(rec, 100, 300, "ANT2") == pytest.approx(rec.data[100:300, 7])
    ref = rec.data[100:300, 6:9]
    assert hum.signal_window(rec, 100, 300, "stomach_ref") == pytest.approx(
        ref[:, 0] - ref.mean(axis=1))
    assert hum.signal_window(rec, 100, 300, "X_T") is None
    old = make_recording(seed=13, cohort="old")
    assert hum.signal_window(old, 0, 50, "R_T") == pytest.approx(old.data[:50, 0])


def test_spectrum_uses_the_peak_signal_else_the_selector(make_hum_window, make_recording,
                                                         queue_row, add_tone) -> None:
    rec = make_recording(seed=14, session="r1")
    add_tone(rec, 120.0, 40.0, channels=[7], seed=1)   # hum on ANT2 only
    w = make_hum_window([queue_row("r1", 30.0, 30.05, peak_signal="ANT2"),
                         queue_row("r1", 40.0, 40.05)], {"r1": rec})
    w.spectrum_box.setChecked(True)
    res = w.spectrum.result
    assert res.signal == "ANT2" and not res.notes
    k = int(np.argmin(np.abs(res.core.freqs - 120.0)))
    assert res.core.power[k] > 100 * np.median(res.core.power)
    w.press("4")                                       # next core has no peak_signal
    assert w.spectrum.result.signal == "RVN1"
    assert "no peak_signal" in w.spectrum.message
    w.spectrum.channel.setCurrentText("ANT3")          # the selector decides
    assert w.spectrum.result.signal == "ANT3"


# -- the zoom -----------------------------------------------------------------


def test_zoom_window_is_exactly_100_ms_around_the_stated_centre() -> None:
    c, lo, hi, basis = hum.zoom_window({"start_s": 10.0, "stop_s": 10.3})
    assert (c, basis) == (pytest.approx(10.15), "core centre")
    assert hi - lo == pytest.approx(0.100) and (lo + hi) / 2 == pytest.approx(c)
    c2, lo2, hi2, basis2 = hum.zoom_window({"start_s": 10.0, "stop_s": 10.3, "peak_s": 10.27})
    assert (c2, lo2, hi2, basis2) == (10.27, pytest.approx(10.22), pytest.approx(10.32), "peak")
    for bad in (11.0, float("nan"), None, "x"):
        assert hum.zoom_window({"start_s": 10.0, "stop_s": 10.3, "peak_s": bad})[3] == (
            "core centre")


def test_the_zoom_panel_shows_100_ms_at_the_main_scale(make_hum_window, make_recording,
                                                       add_burst, queue_row) -> None:
    rec = make_recording(seed=15, session="r1")
    add_burst(rec, 20.0, 0.03, 300.0)
    w = make_hum_window([queue_row("r1", 20.0, 20.03, peak_s=20.012),
                         queue_row("r1", 0.01, 0.02)], {"r1": rec})
    w.zoom_box.setChecked(True)
    z = w.zoom_panel.viewer
    assert z.time_range == pytest.approx((20.012 - 0.05, 20.012 + 0.05))
    assert z.scales == w.viewer.scales
    assert len(z.plots) == 9
    assert w.zoom_panel._core_items[0].getRegion() == pytest.approx((20.0, 20.03))
    t = z.plots[0][1].xData
    assert t[0] == pytest.approx(math.floor(19.962 * FS) / FS)
    assert "peak" in w.zoom_panel.info.text()
    w.press("1")                                  # a core 10 ms from the file start
    lo, hi = w.zoom_panel.viewer.time_range
    assert (lo, hi) == pytest.approx((0.015 - 0.05, 0.015 + 0.05))
    t2 = w.zoom_panel.viewer.plots[0][1].xData
    assert t2[0] == 0.0 and np.all(np.diff(t2) == pytest.approx(1 / FS))  # nothing padded


# ---------------------------------------------------------------------------
# old-cohort beat files
# ---------------------------------------------------------------------------


def _write_hrbr_v5(path: Path, heartlocs: np.ndarray, n: int, *, t_fs: float = FS,
                   t0: float = 0.0) -> None:
    savemat(path, {"heartlocs": heartlocs.reshape(-1, 1).astype(float),
                   "t": (t0 + np.arange(n) / t_fs).reshape(-1, 1)})


def _write_hrbr_v73(path: Path, heartlocs: np.ndarray, n: int, *, t_fs: float = FS,
                    t0: float = 0.0, empty: bool = False) -> None:
    with h5py.File(path, "w", userblock_size=512) as f:
        if empty:  # MATLAB writes [] as its shape, flagged MATLAB_empty
            ds = f.create_dataset("heartlocs", data=np.array([0, 1], dtype=np.uint64))
            ds.attrs["MATLAB_empty"] = np.uint8(1)
        else:
            f["heartlocs"] = heartlocs.reshape(1, -1).astype(float)
        f["t"] = (t0 + np.arange(n) / t_fs).reshape(1, -1)
    header = b"MATLAB 7.3 MAT-file, Platform: x".ljust(116, b" ") + b"\0" * 8 + b"\x00\x02IM"
    with path.open("r+b") as fh:
        fh.write(header)


@pytest.mark.parametrize("writer", [_write_hrbr_v5, _write_hrbr_v73])
def test_old_beats_read_the_whole_recording_hrbr_and_refuse_an_epoch_file(tmp_path,
                                                                          writer) -> None:
    folder = tmp_path / "root" / "f" / "rec1"
    folder.mkdir(parents=True)
    n = 20_000
    writer(folder / "rec1_notched_v0.2.2_recovery_HRBR.mat", np.array([5.0, 9.0]), n // 2)
    got = loaders.old_beats(tmp_path / "root", "f/rec1", "rec1", n_samples=n, fs=FS)
    assert got.beats_s is None and "epoch" in (got.reason or "")
    writer(folder / "REC1_HRBR.mat", np.array([1.0, 341.0, 681.0]), n)  # case differs
    got = loaders.old_beats(tmp_path / "root", "f/rec1", "rec1_notched", n_samples=n, fs=FS)
    assert got.source == "REC1_HRBR.mat"
    assert got.beats_s == pytest.approx(np.array([0.0, 340.0, 680.0]) / FS)  # 1-based
    none = loaders.old_beats(tmp_path / "root", "f/rec1", "other", n_samples=n, fs=FS)
    assert none.beats_s is None and "no other_HRBR.mat" in (none.reason or "")


@pytest.mark.parametrize("writer", [_write_hrbr_v5, _write_hrbr_v73])
def test_old_beats_check_the_file_rate_and_start_and_the_name_prefix(tmp_path,
                                                                    writer) -> None:
    folder = tmp_path / "root" / "f" / "rec1"
    folder.mkdir(parents=True)
    n = 20_000
    hl = np.array([1.0, 341.0, 681.0])
    writer(folder / "rec10_HRBR.mat", hl, n)       # another recording: rec1 + "0"
    got = loaders.old_beats(tmp_path / "root", "f/rec1", "rec1", n_samples=n, fs=FS)
    assert got.beats_s is None and "no rec1_HRBR.mat" in (got.reason or "")
    writer(folder / "rec1_HRBR.mat", hl, n, t_fs=2 * FS)   # same length, another rate
    got = loaders.old_beats(tmp_path / "root", "f/rec1", "rec1", n_samples=n, fs=FS)
    assert got.beats_s is None and "steps 0.00025" in (got.reason or "")
    writer(folder / "rec1_HRBR.mat", hl, n, t0=120.0)      # an epoch that starts later
    got = loaders.old_beats(tmp_path / "root", "f/rec1", "rec1", n_samples=n, fs=FS)
    assert got.beats_s is None and "starts at 120" in (got.reason or "")
    writer(folder / "rec1_HRBR.mat", hl, n)
    got = loaders.old_beats(tmp_path / "root", "f/rec1", "rec1", n_samples=n, fs=FS)
    assert got.source == "rec1_HRBR.mat" and got.beats_s.size == 3


def test_v73_matlab_empty_heartlocs_are_no_beats_not_two(tmp_path) -> None:
    folder = tmp_path / "root" / "f" / "rec1"
    folder.mkdir(parents=True)
    _write_hrbr_v73(folder / "rec1_HRBR.mat", np.zeros(0), 20_000, empty=True)
    got = loaders.old_beats(tmp_path / "root", "f/rec1", "rec1", n_samples=20_000, fs=FS)
    assert got.beats_s is None and "holds no beats" in (got.reason or "")


def test_new_beats_read_the_hrv_grade_train_from_the_store_never_the_mask_grade(
        tmp_path) -> None:
    from gems_blanking_v2.emit.hr_beats import write_hr_beats, write_mask_beats
    from gems_blanking_v2.io.channel_map import meta_path
    from gems_blanking_v2.io.store import GemsStore

    store = GemsStore.initialise(tmp_path / "gems")
    rid, n = "synth_a_rec1", 120_000
    folder = meta_path(store, "A", rid).parent
    folder.mkdir(parents=True)
    beats = np.round(np.arange(1.0, 50.0, 0.15) * FS) / FS
    beats = beats[(beats < 20.0) | (beats >= 21.0)]
    assert loaders.new_beats(store, "A", rid, fs=FS).reason == loaders.NEW_COHORT_NO_BEATS
    write_mask_beats(folder / f"{rid}_peri_r_beats.mat", beats, fs=FS, epoch_start_s=0.0,
                     n_samples=n, channel="L_T", source="test")
    assert loaders.new_beats(store, "A", rid, fs=FS).beats_s is None  # mask grade: never
    gaps = np.zeros(beats.size, dtype=bool)
    gaps[5] = True
    write_hr_beats(folder / f"{rid}_beats.mat", beats, fs=FS, epoch_start_s=0.0, n_samples=n,
                   channel="LVN2-RVN2", source="test", gap_after=gaps,
                   blank_spans_s=[[20.0, 21.0]])
    got = loaders.new_beats(store, "A", rid, fs=FS)
    assert got.source == f"{rid}_beats.mat"
    assert got.beats_s == pytest.approx(beats)
    assert np.array_equal(got.gap_after, gaps)
    assert got.blank_spans == pytest.approx(np.array([[20.0, 21.0]]), abs=1 / FS)
    hr = hum.heart_rate(got.beats_s, 10.0, 30.0, gap_after=got.gap_after,
                        blank_spans=got.blank_spans)
    assert hr.hz == pytest.approx(1 / 0.15, rel=1e-3) and hr.n_refused == 1  # across 20-21
    wrong = loaders.new_beats(store, "A", rid, fs=2 * FS)
    assert wrong.beats_s is None and "Hz" in (wrong.reason or "")


def test_open_queue_wires_the_beat_source_per_cohort(tmp_path, monkeypatch) -> None:
    """New cohort: none, said; old cohort: the HRBR beside the signal."""
    import ui.windows.adjudication_window as aw

    seen: dict[str, Any] = {}

    class Fake:
        def __init__(self, session, **kw):
            seen.update(kw)

    monkeypatch.setattr(aw, "AdjudicationWindow", Fake)
    monkeypatch.setattr(loaders, "resolve_survivals_root", lambda _x=None: tmp_path)
    from gems_blanking_v2.io.store import GemsStore

    from ui.adjudicate.queue import example_queue

    q = tmp_path / "q.parquet"
    example_queue().to_parquet(q, index=False)
    aw.open_queue(q, GemsStore.initialise(tmp_path / "gems"), user="t",
                  journal_dir=tmp_path / "j")
    beats_fn = seen["beats_fn"]

    class Rec:
        data = np.zeros((10, 5))
        fs = FS

    new = beats_fn({"cohort": "new", "animal": "A", "recording": "x"}, Rec())
    assert new.beats_s is None and new.reason == loaders.NEW_COHORT_NO_BEATS
    seen_new: list[tuple[str, str]] = []
    monkeypatch.setattr(loaders, "new_beats", lambda _s, a, r, fs: seen_new.append((a, r))
                        or BeatSource(None, None, "x"))
    beats_fn({"cohort": "new", "animal": "B", "recording": "y"}, Rec())
    assert seen_new == [("B", "y")]
    folder = tmp_path / "synth_folder" / "synth_old_1"
    folder.mkdir(parents=True)
    _write_hrbr_v5(folder / "synth_old_1_HRBR.mat", np.array([3.0]), 10)
    old = beats_fn({"cohort": "old", "recording": "synth_old_1",
                    "folder": "synth_folder/synth_old_1"}, Rec())
    assert old.beats_s == pytest.approx([2.0 / FS])


def test_window_builds_do_not_repeat_the_derivation_warnings(make_recording, caplog) -> None:
    """Silenced for the window build only; the same warning elsewhere still reaches the log."""
    import logging

    from gems_blanking_v2.derive.derivations import build_derivations

    rec = make_recording(seed=16)
    name = "gems_blanking_v2.derive.derivations"
    with caplog.at_level(logging.WARNING, logger=name):
        assert hum.signal_window(rec, 0, 4000, "stomach_ref") is not None
        assert not [r for r in caplog.records if r.name == name]
        build_derivations(rec)
        assert any("stomach_ref" in r.getMessage() for r in caplog.records if r.name == name)



# ---------------------------------------------------------------------------
# review fixes: stale results, held recordings, independent toggles
# ---------------------------------------------------------------------------


def _until(cond, timeout: float = 20.0) -> None:
    from PySide6.QtWidgets import QApplication

    t0 = time.monotonic()
    while not cond():
        QApplication.processEvents()
        time.sleep(0.005)
        assert time.monotonic() - t0 < timeout, "timed out"
    for _ in range(3):
        QApplication.processEvents()


def test_a_spectrum_still_running_never_lands_on_another_core(make_hum_window, make_recording,
                                                             queue_row, monkeypatch) -> None:
    gates: dict[float, threading.Event] = {s: threading.Event() for s in (10.0, 20.0, 30.0)}
    real = hum.compute_spectrum

    def held(rec, row, rows, selected):
        gates[float(row["start_s"])].wait(30)
        return real(rec, row, rows, selected)

    monkeypatch.setattr(hum, "compute_spectrum", held)
    rec = make_recording(seed=18, session="r1")
    w = make_hum_window([queue_row("r1", t, t + 0.05) for t in (10.0, 20.0, 30.0)],
                        {"r1": rec}, async_panels=True)
    try:
        w.spectrum_box.setChecked(True)            # core 10: held
        w.press("1")                               # core 20: held too
        gates[10.0].set()
        _until(lambda: any(j.finished and "|10000000|" in j.token for j in w._panel_jobs))
        assert w.spectrum.result is None           # core 10's result was dropped
        gates[20.0].set()
        _until(lambda: w.spectrum.result is not None)
        assert w.spectrum.result.core_key == w.session.current()["core_key"]
        w.press("1")                               # core 30: held
        w.press("1")                               # queue finished while it runs
        assert w.session.current() is None
        gates[30.0].set()
        _until(lambda: all(j.finished for j in w._panel_jobs))
        assert w.spectrum.result is None and w.spectrum.message == "Queue finished."
    finally:
        for g in gates.values():
            g.set()



def test_failures_and_the_end_of_the_queue_leave_no_stale_panel(make_hum_window,
                                                                make_recording,
                                                                queue_row) -> None:
    recs = {"r1": make_recording(seed=19, session="r1")}
    w = make_hum_window([queue_row("r1", 10.0, 10.05), queue_row("r2", 10.0, 10.05)], recs)
    w.spectrum_box.setChecked(True)
    w.zoom_box.setChecked(True)
    assert w.spectrum.result is not None and w.zoom_panel.showing
    w.press("1")                                   # r2 is not in recs: the load fails
    assert w.last_error and w.spectrum.result is None and not w.zoom_panel.showing
    assert w.spectrum.hr_x.size == 0 and w.spectrum.mains_x.size == 0
    w2 = make_hum_window([queue_row("r1", 10.0, 10.05)],
                         {"r1": make_recording(seed=23, session="r1")})
    w2.spectrum_box.setChecked(True)
    w2.zoom_box.setChecked(True)
    assert w2.spectrum.result is not None and w2.zoom_panel.showing
    w2.press("1")
    assert w2.session.current() is None
    assert w2.spectrum.result is None and not w2.zoom_panel.showing
    assert w2.spectrum.message == "Queue finished."


def test_a_finished_panel_job_does_not_keep_the_recording_alive(make_hum_window,
                                                               make_recording,
                                                               queue_row) -> None:
    recs = {"r1": make_recording(seed=20, session="r1"),
            "r2": make_recording(seed=21, session="r2")}
    w = make_hum_window([queue_row("r1", 10.0, 10.05), queue_row("r2", 10.0, 10.05)], recs,
                        async_panels=True)
    w.spectrum_box.setChecked(True)
    w.zoom_box.setChecked(True)
    _until(lambda: w.spectrum.result is not None and all(j.finished for j in w._panel_jobs))
    first_jobs = list(w._panel_jobs)
    assert first_jobs and all(j.fn is None for j in first_jobs)   # the closure is gone
    data_ref = weakref.ref(recs["r1"].data)  # the ~2 GB; Recording itself has slots
    del recs["r1"]
    w.spectrum_box.setChecked(False)               # so no new job prunes them instead
    w.press("1")                                   # on to r2
    assert w._panel_jobs == []                     # pruned on the recording change
    del first_jobs
    gc.collect()
    assert data_ref() is None


def test_toggling_one_panel_does_not_recompute_the_other(make_hum_window, make_recording,
                                                         queue_row, monkeypatch) -> None:
    calls: list[str] = []
    real_spec, real_zoom = hum.compute_spectrum, hum.zoom_window
    monkeypatch.setattr(hum, "compute_spectrum", lambda *a: calls.append("spectrum")
                        or real_spec(*a))
    monkeypatch.setattr(hum, "zoom_window", lambda *a: calls.append("zoom") or real_zoom(*a))
    w = make_hum_window([queue_row("r1", 10.0, 10.05)],
                        {"r1": make_recording(seed=22, session="r1")})
    w.spectrum_box.setChecked(True)
    assert calls == ["spectrum"]
    w.zoom_box.setChecked(True)
    assert calls == ["spectrum", "zoom"]
    w.spectrum_box.setChecked(False)
    w.spectrum_box.setChecked(True)
    assert calls == ["spectrum", "zoom", "spectrum"]
    w.zoom_box.setChecked(False)
    assert calls == ["spectrum", "zoom", "spectrum"]



def test_the_panel_takes_the_rate_from_10_s_either_side(make_hum_window, make_recording,
                                                        queue_row) -> None:
    """No beat within the core's ±1 s (a blanked stretch); the ±10 s still give the rate."""
    rec = make_recording(seed=24, session="r1")
    beats = np.round(np.arange(20.5, 28.9, 0.15), 6)

    tags = np.zeros(beats.size, dtype=bool)
    tags[3] = True                                  # the file's own gap tag
    span = np.array([[25.0, 25.2]])                 # and a blank span
    across = int(np.sum((beats[:-1] < 25.2) & (beats[1:] > 25.0)))
    w = make_hum_window([queue_row("r1", 30.0, 30.05)], {"r1": rec},
                        beats_fn=lambda _r, _x: BeatSource(beats, "r1_beats.mat", None,
                                                           gap_after=tags, blank_spans=span))
    w.spectrum_box.setChecked(True)
    assert w.spectrum.hr_x[0] == pytest.approx(1 / 0.15)
    n_ok = beats.size - 1 - 1 - across
    assert (f"{n_ok} valid RR intervals within ±10 s, {1 + across} refused"
            in w.spectrum.message)
