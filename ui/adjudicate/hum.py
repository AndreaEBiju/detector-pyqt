"""What the Change 1 screen computes around a core: the vertical scale and the hum panels.

Pure numpy / scipy, no Qt, so every rule here is tested without a display.

**Vertical scale (always on).** Each raw channel's scale is taken from the samples
within ``CORE_MARGIN_S`` (1 s) either side of the core - the half-open sample range
``[floor((start_s - 1) * fs), ceil((stop_s + 1) * fs))`` clipped to the recording -
never from the whole view, so a large event elsewhere cannot flatten the highlight.
The estimator, over the FINITE samples ``x`` of that range (NaN is ignored, never
filled)::

    centre = median(x)
    S      = percentile(|x - centre|, 99.5)          (numpy's default, linear)
    limits = centre +/- SCALE_HEADROOM * S            (SCALE_HEADROOM = 1.5)

A channel whose ``S`` is 0 (flat over the range) gets ``centre +/- FLAT_HALF_RANGE_UV``
(1 µV) and is reported flat; a channel with no finite sample in the range has no scale
(``None``) and is left to auto-range. Near a file edge the range is truncated and the
scale uses what exists: nothing is padded, no zero is invented. Samples beyond the
limits are drawn AT the limit (:func:`clip_to_scale`; NaN stays NaN) and marked.

**Spectrum.** Welch PSD (:func:`welch_psd`) of one signal over the core's ±1 s window,
against a reference window of the same length placed by :func:`place_reference`. Lines
at ``60·k`` Hz (:func:`mains_lines`) and ``k × HR`` for ``k <= 20`` (:func:`hr_lines`), HR
being ``1 / median(RR)`` of a STORED beat train over ``±10 s`` around the core, valid
intervals only (:func:`heart_rate`) - never estimated from the signal.

**Zoom.** :func:`zoom_window`: exactly ``ZOOM_S`` (100 ms) around the core's peak time
when the queue row carries one (``peak_s``), else around the core centre.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from gems_blanking_v2.derive.derivations import build_derivations
from gems_blanking_v2.physio.rpeaks import RR_PLAUSIBLE_RANGE_S
from gems_blanking_v2.types import Recording
from scipy.signal import spectrogram

__all__ = [
    "CORE_MARGIN_S",
    "FLAT_HALF_RANGE_UV",
    "F_MAX_HZ",
    "HR_K_MAX",
    "HR_WINDOW_S",
    "MAINS_HZ",
    "MIN_VALID_RR",
    "PEAK_SIGNAL",
    "PSD_RESOLUTION_HZ",
    "REF_OFFSETS_S",
    "SCALE_HEADROOM",
    "SCALE_PERCENTILE",
    "ZOOM_S",
    "ChannelScale",
    "HeartRate",
    "Psd",
    "RefWindow",
    "SpectrumResult",
    "channel_scales",
    "clip_to_scale",
    "clipped_segments",
    "compute_spectrum",
    "core_window",
    "heart_rate",
    "hr_lines",
    "mains_lines",
    "no_reference_reason",
    "others_in_recording",
    "place_reference",
    "scale_of",
    "signal_window",
    "welch_nperseg",
    "welch_psd",
    "zoom_window",
]

F64 = npt.NDArray[np.float64]

CORE_MARGIN_S: Final = 1.0
"""Seconds either side of the core that set the vertical scale and the spectrum window."""
SCALE_PERCENTILE: Final = 99.5
"""Percentile of ``|x - median|`` over the core window that sets the scale."""
SCALE_HEADROOM: Final = 1.5
"""Multiplier on that percentile: the limits are ``median +/- SCALE_HEADROOM * S``.

Andrea, 2026-10-08: 1.5. Measured on 200 (core, channel) pairs of three real
recordings, the core's own peak clipped in 39.5% of pairs at 1.0, 2.0% at 1.5, 0 at 2.0.
"""
FLAT_HALF_RANGE_UV: Final = 1.0
"""Half-range given to a channel that is flat (S = 0) over the core window, µV."""

PSD_RESOLUTION_HZ: Final = 3.0
"""Target Welch bin width: ``nperseg = 2**round(log2(fs / 3))`` (8192 at 24414 Hz)."""
MAINS_HZ: Final = 60.0
F_MAX_HZ: Final = 3000.0
"""Highest frequency shown by default and marked with 60·k / k×HR lines."""
HR_WINDOW_S: Final = 10.0
"""Seconds either side of the core whose stored beats give the heart rate."""
HR_K_MAX: Final = 20
"""k×HR lines are drawn for ``k = 1 .. 20`` only (about 120-160 Hz at rat heart rates):
higher harmonics crowd the axis and a 1% rate error moves the 500th by 30 Hz."""
MIN_VALID_RR: Final = 2
"""Fewest valid RR intervals the heart rate is taken from."""
REF_OFFSETS_S: Final[tuple[float, ...]] = tuple(5.0 + 0.5 * k for k in range(11))
"""Gaps tried between the core and the reference window: 5.0, 5.5, ..., 10.0 s."""

ZOOM_S: Final = 0.100
"""Width of the zoom panel, seconds."""


# ---------------------------------------------------------------------------
# vertical scale
# ---------------------------------------------------------------------------


def core_window(start_s: float, stop_s: float, fs: float, n_samples: int,
                margin_s: float = CORE_MARGIN_S) -> tuple[int, int]:
    """Sample range ``[i0, i1)`` of ``margin_s`` either side of the core, clipped to the file."""
    i0 = max(0, math.floor((float(start_s) - margin_s) * fs))
    i1 = min(int(n_samples), math.ceil((float(stop_s) + margin_s) * fs))
    return i0, max(i0, i1)


@dataclass(frozen=True)
class ChannelScale:
    """One channel's display limits, µV: ``centre_uv +/- half_range_uv``."""

    centre_uv: float
    half_range_uv: float
    flat: bool
    """True when ``S`` was 0 and :data:`FLAT_HALF_RANGE_UV` was used instead."""

    @property
    def lo(self) -> float:
        """Lower plot limit, µV."""
        return self.centre_uv - self.half_range_uv

    @property
    def hi(self) -> float:
        """Upper plot limit, µV."""
        return self.centre_uv + self.half_range_uv


def scale_of(x: npt.ArrayLike) -> ChannelScale | None:
    """The module docstring's estimator over ``x`` (µV); ``None`` when nothing is finite."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    centre = float(np.median(finite))
    s = float(np.percentile(np.abs(finite - centre), SCALE_PERCENTILE)) * SCALE_HEADROOM
    if not s > 0.0:
        return ChannelScale(centre, FLAT_HALF_RANGE_UV, flat=True)
    return ChannelScale(centre, s, flat=False)


def channel_scales(data: npt.NDArray[Any], fs: float, start_s: float,
                   stop_s: float) -> list[ChannelScale | None]:
    """Per column of ``data`` ``(n, ch)``, the scale from the core's ±1 s window."""
    i0, i1 = core_window(start_s, stop_s, fs, data.shape[0])
    block = data[i0:i1]
    return [scale_of(block[:, ch]) for ch in range(data.shape[1])]


def clip_to_scale(y: npt.ArrayLike, scale: ChannelScale | None,
                  ) -> tuple[F64, npt.NDArray[np.bool_]]:
    """``(y clipped to the limits, which samples were clipped)``. NaN stays NaN, unflagged."""
    arr = np.asarray(y, dtype=np.float64)
    if scale is None:
        return arr, np.zeros(arr.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        mask = (arr < scale.lo) | (arr > scale.hi)
    return np.clip(arr, scale.lo, scale.hi), mask


def _runs(mask: npt.NDArray[np.bool_]) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """``(starts, stops)`` of the runs of True in ``mask``, half-open."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))) != 0)
    return edges[0::2], edges[1::2]


def clipped_segments(t: F64, y: npt.ArrayLike, scale: ChannelScale | None,
                     dt: float) -> tuple[F64, F64]:
    """The clipped runs of ``y`` (unclipped µV) as ``connect='pairs'`` segments.

    Each run of consecutive samples above the upper limit (or below the lower one)
    becomes one segment AT that limit, from its first sample's time to its last
    sample's time plus ``dt`` - so a single clipped sample is still a visible mark.
    Returns ``(x, y)`` with two points per segment (empty when nothing is clipped).
    """
    arr = np.asarray(y, dtype=np.float64)
    if scale is None:
        return np.zeros(0), np.zeros(0)
    xs: list[float] = []
    ys: list[float] = []
    with np.errstate(invalid="ignore"):
        sides = ((arr > scale.hi, scale.hi), (arr < scale.lo, scale.lo))
    for mask, level in sides:
        starts, stops = _runs(mask)
        for a, b in zip(starts, stops, strict=True):
            xs += [float(t[a]), float(t[b - 1]) + dt]
            ys += [level, level]
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------


_QUIET = threading.local()


class _QuietOnThisThread(logging.Filter):
    """Drop the derivation module's records while THIS thread is building a window."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(_QUIET, "on", False)


logging.getLogger("gems_blanking_v2.derive.derivations").addFilter(_QuietOnThisThread())


@contextmanager
def _quiet_derivations() -> Iterator[None]:
    """Silence ``build_derivations``' warnings for one call on this thread only.

    They are diagnostics about a whole recording (a degenerate tripole fit, the
    stomach common average); on a 2 s window they say nothing and would be repeated for
    every core. Warnings from any other thread (a z-trace job) are untouched.
    """
    _QUIET.on = True
    try:
        yield
    finally:
        _QUIET.on = False


def signal_window(rec: Recording, i0: int, i1: int, name: str) -> F64 | None:
    """Samples ``[i0, i1)`` of detection signal ``name`` (a queue row's ``peak_signal``).

    A raw channel name is read directly. Anything else is built exactly as the
    detection chain builds its signal set (``gems_blanking_v2.detect.chain.
    detect_region``: ``build_derivations`` on the slice, plus the raw stomach channels
    by name) - so ``L_T`` is the applied tripole, ``R_V1`` cuff R's contact 1 and
    ``stomach_ref`` the stomach reference. ``None`` when the name is not in that set.
    The derivation module's warnings are silenced for this call (:func:`_quiet_derivations`).
    """
    raw = {c.name: c.index for c in rec.channels}
    if name in raw:
        return np.asarray(rec.data[i0:i1, raw[name]], dtype=np.float64)
    if i1 <= i0:
        return None
    sub = replace(rec, data=rec.data[i0:i1])
    with _quiet_derivations():
        signals, _weights = build_derivations(sub)
    out = signals.get(name)
    return None if out is None else np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# spectrum
# ---------------------------------------------------------------------------


def welch_nperseg(fs: float) -> int:
    """Segment length for :data:`PSD_RESOLUTION_HZ` bins: a power of two (8192 at 24414 Hz)."""
    return int(2 ** round(math.log2(fs / PSD_RESOLUTION_HZ)))


@dataclass(frozen=True)
class Psd:
    """A Welch estimate: ``freqs`` Hz, ``power`` µV²/Hz, and how many segments it used."""

    freqs: F64
    power: F64
    n_used: int
    n_segments: int


def welch_psd(x: npt.ArrayLike, fs: float, nperseg: int) -> Psd | None:
    """Welch PSD of ``x`` (µV): Hann window, ``nperseg``, 50% overlap, constant detrend.

    Density scaling, one-sided, the MEAN over segments - identical to
    ``scipy.signal.welch(x, fs, nperseg=nperseg)`` for finite input. A segment that
    contains any NaN is left out of the mean (nothing is interpolated); ``None`` when
    the input is shorter than ``nperseg`` or no segment is entirely finite.
    """
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size < nperseg or nperseg < 2:
        return None
    f, _t, sxx = spectrogram(arr, fs=fs, window="hann", nperseg=nperseg,
                             noverlap=nperseg // 2, detrend="constant", scaling="density",
                             mode="psd")
    ok = np.isfinite(sxx).all(axis=0)
    if not ok.any():
        return None
    return Psd(np.asarray(f, dtype=np.float64), sxx[:, ok].mean(axis=1),
               int(ok.sum()), int(ok.size))


@dataclass(frozen=True)
class RefWindow:
    """Where the spectrum's reference window went, and what (if anything) is wrong with it."""

    start_s: float
    stop_s: float
    offset_s: float
    """Gap between the core and the window, seconds."""
    side: str
    """``"after"`` or ``"before"`` the core."""
    in_region: bool
    overlaps: int
    """How many other queue cores of the recording the window overlaps."""

    @property
    def note(self) -> str | None:
        """What the panel must say about this window; ``None`` when it is clean."""
        bits = []
        if not self.in_region:
            bits.append("leaves the core's region")
        if self.overlaps:
            bits.append(f"overlaps {self.overlaps} other queue core(s)")
        return None if not bits else "reference window " + " and ".join(bits)


def place_reference(start_s: float, stop_s: float, length_s: float,
                    region: tuple[float, float], duration_s: float,
                    others: Iterable[tuple[float, float]]) -> RefWindow | None:
    """Place the reference window: same length, 5-10 s from the core, deterministically.

    Candidates, in order: gap ``d`` in :data:`REF_OFFSETS_S` (5.0, 5.5, ..., 10.0 s),
    and for each ``d`` first AFTER the core (``[stop_s + d, stop_s + d + length)``) then
    BEFORE it (``[start_s - d - length, start_s - d)``). Every candidate must lie inside
    the recording. The first candidate in that order is taken from the best tier:

    1. inside the core's region and overlapping no other queue core;
    2. inside the region, overlapping other cores (the panel says so);
    3. outside the region, overlapping no other core (the panel says so).

    ``None`` when no candidate fits the recording at all (the panel says why:
    :func:`no_reference_reason`). ``others`` are the OTHER QUEUE cores of the recording,
    ``[start, stop)`` seconds: only cores in this queue are avoided - candidates that were
    not queued, and earlier labels, are not known to the screen.
    """
    r0, r1 = float(region[0]), float(region[1])
    cores = [(float(a), float(b)) for a, b in others]
    cands: list[RefWindow] = []
    for d in REF_OFFSETS_S:
        for side in ("after", "before"):
            a = stop_s + d if side == "after" else start_s - d - length_s
            b = a + length_s
            if a < 0.0 or b > duration_s:
                continue
            n_over = sum(1 for c0, c1 in cores if c0 < b and a < c1)
            cands.append(RefWindow(a, b, d, side, r0 <= a and b <= r1, n_over))
    for tier in (lambda w: w.in_region and not w.overlaps,
                 lambda w: w.in_region,
                 lambda w: not w.overlaps):
        for w in cands:
            if tier(w):
                return w
    return None


def no_reference_reason(start_s: float, stop_s: float, length_s: float,
                        duration_s: float) -> str:
    """Why :func:`place_reference` found nothing: how much room each side of the core has."""
    need = REF_OFFSETS_S[0] + length_s
    after, before = duration_s - stop_s, start_s
    return (f"no reference window: it needs {need:.1f} s clear on one side of the core "
            f"(5 s gap + {length_s:.1f} s), and the recording has {after:.1f} s after it and "
            f"{before:.1f} s before it")


def mains_lines(f_max: float = F_MAX_HZ) -> F64:
    """``60·k`` Hz for ``k = 1 .. floor(f_max / 60)`` (50 lines up to 3000 Hz)."""
    return MAINS_HZ * np.arange(1, math.floor(f_max / MAINS_HZ + 1e-9) + 1, dtype=np.float64)


def hr_lines(hr_hz: float, f_max: float = F_MAX_HZ, k_max: int = HR_K_MAX) -> F64:
    """``k × HR`` Hz for ``k = 1 .. min(k_max, floor(f_max / HR))``."""
    if not (math.isfinite(hr_hz) and hr_hz > 0):
        return np.zeros(0)
    k = min(int(k_max), math.floor(f_max / hr_hz + 1e-9))
    return hr_hz * np.arange(1, k + 1, dtype=np.float64)


@dataclass(frozen=True)
class HeartRate:
    """The heart rate around a core, or why there is none."""

    hz: float | None
    n_rr: int
    """How many valid RR intervals it came from."""
    n_refused: int
    """RR intervals in the window refused (gap-tagged, across a blank span, implausible)."""
    reason: str | None


def heart_rate(beats_s: npt.ArrayLike, t0: float, t1: float, *,
               gap_after: npt.ArrayLike | None = None,
               blank_spans: npt.ArrayLike | None = None) -> HeartRate:
    """``1 / median(RR)`` over the valid consecutive-beat intervals inside ``[t0, t1)``.

    The window is the core ±:data:`HR_WINDOW_S` (the caller passes it). An interval
    between consecutive stored beats ``b[i], b[i+1]`` (both inside the window) is
    REFUSED when the beat file tags it as a gap (``gap_after[i]``, in sorted beat
    order), when a blank span ``[a, b)`` overlaps it (a rejected stretch whose beats
    were removed), or when it lies outside ``gems_blanking_v2.physio.rpeaks.
    RR_PLAUSIBLE_RANGE_S`` (80-500 ms: 750 down to 120 bpm, the range task 05's global
    RR is taken over). The median, not the mean, so a missed beat that is not tagged
    cannot pull the rate down. Fewer than :data:`MIN_VALID_RR` valid intervals: no rate,
    and the reason says so.
    """
    raw = np.asarray(beats_s, dtype=np.float64).ravel()
    order = np.argsort(raw, kind="stable")
    b = raw[order]
    gaps = (np.zeros(b.size, dtype=bool) if gap_after is None
            else np.asarray(gap_after, dtype=bool).ravel()[order])
    keep = (b >= t0) & (b < t1)
    idx = np.flatnonzero(keep[:-1] & keep[1:])
    lo, hi = RR_PLAUSIBLE_RANGE_S
    rr = b[idx + 1] - b[idx]
    ok = (rr >= lo) & (rr <= hi) & ~gaps[idx]
    if blank_spans is not None:
        spans = np.asarray(blank_spans, dtype=np.float64).reshape(-1, 2)
        for s0, s1 in spans:
            ok &= ~((b[idx] < s1) & (s0 < b[idx + 1]))
    valid = rr[ok]
    refused = int(rr.size - valid.size)
    if valid.size < MIN_VALID_RR:
        return HeartRate(None, int(valid.size), refused,
                         f"{valid.size} valid RR interval(s) within ±{HR_WINDOW_S:.0f} s of "
                         f"the core ({refused} refused: gap, blank span or outside "
                         f"{lo * 1000:.0f}-{hi * 1000:.0f} ms) - k×HR not shown")
    return HeartRate(float(1.0 / np.median(valid)), int(valid.size), refused, None)


# ---------------------------------------------------------------------------
# zoom
# ---------------------------------------------------------------------------


def zoom_window(row: dict[str, Any]) -> tuple[float, float, float, str]:
    """``(centre, lo, hi, basis)``: ``ZOOM_S`` seconds centred on the core's peak or centre.

    ``basis`` is ``"peak"`` when the row has a finite ``peak_s`` inside ``[start_s,
    stop_s]``, else ``"core centre"`` (the queues built so far carry no peak time).
    """
    a, b = float(row["start_s"]), float(row["stop_s"])
    peak = row.get("peak_s")
    try:
        p = float(peak) if peak is not None else math.nan
    except (TypeError, ValueError):
        p = math.nan
    if math.isfinite(p) and a <= p <= b:
        c, basis = p, "peak"
    else:
        c, basis = 0.5 * (a + b), "core centre"
    return c, c - ZOOM_S / 2, c + ZOOM_S / 2, basis


def others_in_recording(rows: Sequence[dict[str, Any]], row: dict[str, Any],
                        ) -> list[tuple[float, float]]:
    """The other queue cores of ``row``'s recording, ``[start_s, stop_s)``."""
    rid = str(row["recording"])
    return [(float(r["start_s"]), float(r["stop_s"])) for r in rows
            if str(r["recording"]) == rid and r["core_key"] != row["core_key"]]


# ---------------------------------------------------------------------------
# the spectrum panel's computation (runs off the GUI thread)
# ---------------------------------------------------------------------------

PEAK_SIGNAL: Final = "Core's peak signal"
"""The channel selector's first entry: the queue row's ``peak_signal``."""


@dataclass(frozen=True)
class SpectrumResult:
    """Everything the spectrum panel draws for one core (numbers only, no Qt)."""

    core_key: str
    signal: str
    """The signal analysed (a raw channel name or a detection signal name)."""
    window_s: tuple[float, float]
    nperseg: int
    core: Psd | None
    ref: Psd | None
    ref_window: RefWindow | None
    notes: tuple[str, ...]


def compute_spectrum(rec: Recording, row: dict[str, Any], rows: Sequence[dict[str, Any]],
                     selected: str) -> SpectrumResult:
    """PSDs of the core window and its reference window, for the panel.

    The signal is the row's ``peak_signal`` when ``selected`` is :data:`PEAK_SIGNAL` and
    the row has one this recording can build (:func:`signal_window`); otherwise the
    raw channel ``selected`` names, or the first raw channel - and the notes say which.
    """
    fs, n = float(rec.fs), int(rec.data.shape[0])
    raw_names = [c.name for c in rec.channels]
    notes: list[str] = []
    i0, i1 = core_window(float(row["start_s"]), float(row["stop_s"]), fs, n)
    name: str | None = None
    x: F64 | None = None
    if selected == PEAK_SIGNAL:
        peak = row.get("peak_signal")
        if isinstance(peak, str) and peak:
            x = signal_window(rec, i0, i1, peak)
            if x is None:
                notes.append(f"peak signal {peak} cannot be built from this recording")
            else:
                name = peak
        else:
            notes.append("this core has no peak_signal")
    elif selected in raw_names:
        name = selected
        x = signal_window(rec, i0, i1, selected)
    if x is None or name is None:
        name = raw_names[0]
        x = signal_window(rec, i0, i1, name)
        notes.append(f"showing {name} - pick a channel in the selector")
    assert x is not None
    length_s = (i1 - i0) / fs
    nperseg = min(welch_nperseg(fs), i1 - i0)
    core = welch_psd(x, fs, nperseg)
    if core is None:
        notes.append("no NaN-free Welch segment in the core window")
    ref_w = place_reference(float(row["start_s"]), float(row["stop_s"]), length_s,
                            (float(row["region_start_s"]), float(row["region_stop_s"])),
                            n / fs, others_in_recording(rows, row))
    ref: Psd | None = None
    if ref_w is None:
        notes.append(no_reference_reason(float(row["start_s"]), float(row["stop_s"]),
                                         length_s, n / fs))
    else:
        if ref_w.note:
            notes.append(ref_w.note)
        r0 = math.floor(ref_w.start_s * fs)
        r1 = min(n, r0 + (i1 - i0))
        y = signal_window(rec, r0, r1, name)
        ref = None if y is None else welch_psd(y, fs, nperseg)
        if y is None:
            notes.append(f"reference window found, but {name} could not be built there")
        elif ref is None:
            notes.append("reference window found, but it has no NaN-free Welch segment "
                         "(masked data)")
    return SpectrumResult(str(row["core_key"]), name, (i0 / fs, i1 / fs), nperseg, core, ref,
                          ref_w, tuple(notes))
