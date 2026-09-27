"""The one place this app reaches into ``gems_blanking_v2``.

The recall audit must read **new-cohort** recordings (9-channel ``_sig.mat``,
``RVN1-3``/``LVN1-3``/``ANT1-3``) and must show the candidates and band z-traces
that the gate is actually measured against. Both already exist: task 03's loader
and task 07's ``candidate_report``. Importing them rather than reimplementing is
not a preference - two implementations of "what is a candidate" means the audit
measures one thing and the gate measures another, and the disagreement would be
invisible.

Everything here is a thin adapter. If a function in this file starts computing
something rather than moving it between shapes, it is in the wrong repository.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ``gems_blanking_v2`` is a DECLARED dependency (pyproject ``dependencies``), installed
# into the environment - never located by inserting a sibling checkout into
# ``sys.path``. Invariant 21: a library resolved by path order is an accident; the
# old ``_ensure_path()`` did exactly that and could pick up whichever copy it found.


@dataclass(frozen=True, slots=True)
class BandTrace:
    """One band's z-trace, reduced across signals.

    Attributes
    ----------
    band
        Band name, e.g. ``"300-3000"``.
    z_max
        Per-frame maximum of z over **every** signal, including raw contacts.
        Invariant 6: motion appears on the raw contacts too, and a reduction that
        skipped them could hide the evidence the audit is looking for.
    winner
        Per-frame name of the signal that carried the maximum, so the labeller can
        expand a frame and see which channel it came from.
    grid_s
        Frame step, seconds.
    z_enter
        The threshold in force, drawn on the trace. Without it the labeller cannot
        tell "z was low" from "z was high but under the line", and those are the
        two diagnoses the audit exists to separate.
    """

    band: str
    z_max: np.ndarray
    winner: tuple[str, ...]
    grid_s: float
    z_enter: float

    def at(self, t_s: float) -> tuple[float, str]:
        """Return ``(z, winning signal)`` at a time, for click-to-expand."""
        i = int(t_s / self.grid_s)
        if not 0 <= i < self.z_max.size:
            return float("nan"), ""
        return float(self.z_max[i]), self.winner[i]


def load_new_cohort(path: Path, animal: str, **kw: Any):
    """Task 03's loader. Returns its ``LoadedRecording``."""
    from gems_blanking_v2.io.recording import load_recording

    return load_recording(Path(path), animal, **kw)


def z_by_pair(
    data: np.ndarray, fs: float, names: list[str], bands: tuple[str, ...] | None = None
) -> dict[tuple[str, str], np.ndarray]:
    """Compute z for every ``(signal, band)`` pair, using tasks 06's machinery.

    Each signal is thresholded against **its own** reference (invariant 3), which
    is why this loops rather than pooling: borrowing one signal's sigma for
    another was measured at 1136 true / 19,624 false detections.
    """
    from gems_blanking_v2.bands.envelope import band_envelope_for, log_envelope
    from gems_blanking_v2.bands.reference import epoch_reference
    from gems_blanking_v2.bands.zscore import zscore
    from gems_blanking_v2.constants import BANDS

    use = bands if bands is not None else tuple(BANDS)
    out: dict[tuple[str, str], np.ndarray] = {}
    for col, name in enumerate(names):
        x = np.asarray(data[:, col], dtype=np.float64)
        for band in use:
            env = band_envelope_for(x, fs, band)
            log_env = log_envelope(env)
            ref = epoch_reference(log_env, signal=name, band=band)
            out[(name, band)] = zscore(log_env, ref, signal=name, band=band)
    return out


def reduce_to_band_traces(
    z: dict[tuple[str, str], np.ndarray], z_enter: float, grid_s: float
) -> list[BandTrace]:
    """Collapse 54 traces to six: per band, the maximum over all signals.

    Six because a labeller cannot read 54, and the maximum because the question
    at reveal time is "did ANY signal see this?" - a per-signal view would let a
    missed artifact hide on the one channel not being looked at. Which signal
    carried the maximum is kept per frame rather than discarded, so the detail is
    one click away instead of gone.
    """
    from collections import defaultdict

    by_band: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    for (signal, band), trace in z.items():
        by_band[band].append((signal, np.asarray(trace, dtype=np.float64)))

    traces: list[BandTrace] = []
    for band, entries in by_band.items():
        entries.sort(key=lambda e: e[0])
        names = [e[0] for e in entries]
        stack = np.vstack([e[1] for e in entries])
        # NaN means "not assessable here", not "low z". nanmax keeps a frame that
        # is valid on some signals; a plain max would poison it to NaN.
        # An all-NaN frame is expected, not exceptional: it is a stretch no
        # signal could be assessed on. nanmax warns on it, so the warning is
        # suppressed here rather than left to look like a defect in a log.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            z_max = np.nanmax(stack, axis=0)
            who = np.nanargmax(np.where(np.isnan(stack), -np.inf, stack), axis=0)
        winner = tuple(
            names[w] if np.isfinite(z_max[i]) else "" for i, w in enumerate(who)
        )
        traces.append(
            BandTrace(band=band, z_max=z_max, winner=winner,
                      grid_s=grid_s, z_enter=z_enter)
        )
    traces.sort(key=lambda t: t.band)
    return traces


def candidates_for(
    z: dict[tuple[str, str], np.ndarray], beats: Any, **kw: Any
):
    """Task 07's ``candidate_report``. Returns the report, not a bare list.

    The report, deliberately: task 07 makes ``CandidateReport`` the exported
    contract so a consumer cannot receive the candidate list without the
    ``over_cap`` and ``cardiac_windows`` state that says how to treat it.
    """
    from gems_blanking_v2.detect.candidates import candidate_report

    return candidate_report(z, beats, **kw)


def intervals_from_candidates(report: Any) -> np.ndarray:
    """``(n, 2)`` float seconds, the shape ``MultiChannelViewer`` overlays want."""
    spans = [(c.start_s, c.stop_s) for c in report.candidates]
    if not spans:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(spans, dtype=np.float64)


def reveal_for_region(
    rec: Any, region: tuple[float, float], z_enter: float = 3.0
) -> tuple[np.ndarray, list[BandTrace]]:
    """Candidates and the six band traces for one assessable region.

    Everything is task 03-07 machinery, composed rather than reimplemented: the
    detection signal set is ``build_derivations`` (per cuff V1-V3 and T, plus
    ``stomach_ref`` - ``constants.NERVE_SIGNALS`` / ``STOMACH_SIGNALS``), beats come
    from ``detect_rpeaks`` on the right-cuff tripole, and z uses each signal's own
    whole-epoch reference (invariants 3 and 5). Computed on the REGION (the
    baseline, or the recovery epoch of a stim/recovery file) so the stim epoch
    never enters a reference, then placed on the recording's own timeline: the
    candidate intervals are offset by the region start and each trace is padded
    with NaN frames ("not assessable here") up to it.
    """
    from dataclasses import replace

    from gems_blanking_v2.constants import GRID_S
    from gems_blanking_v2.derive.derivations import build_derivations
    from gems_blanking_v2.physio.rpeaks import detect_rpeaks

    lo, hi = region
    i0, i1 = round(lo * rec.fs), round(hi * rec.fs)
    sub = replace(rec, data=rec.data[i0:i1])
    signals, _weights = build_derivations(sub)
    names = sorted(signals)
    stack = np.column_stack([signals[n] for n in names])
    z = z_by_pair(stack, float(rec.fs), names)
    beats = detect_rpeaks(signals["R_T"], float(rec.fs))
    report = candidates_for(z, beats, z_enter=z_enter)
    intervals = intervals_from_candidates(report) + lo
    pad = round(lo / GRID_S)
    traces = [
        BandTrace(band=t.band, z_max=np.concatenate([np.full(pad, np.nan), t.z_max]),
                  winner=("",) * pad + t.winner, grid_s=t.grid_s, z_enter=t.z_enter)
        for t in reduce_to_band_traces(z, z_enter=z_enter, grid_s=GRID_S)
    ]
    return intervals, traces
