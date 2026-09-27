"""Why was this artifact missed? Blind spot, or threshold too high?

The audit's output is not "recall was 0.83". It is a list of missed artifacts,
each of which needs one of two different fixes:

**generator blind spot** - z is genuinely low in every band. No threshold change
finds this; it needs a new band or a new feature.

**threshold too high** - z rose but stayed under ``z_enter``. The event is
visible to the existing bands and a lower ``z_enter`` would catch it.

These are **indistinguishable from the raw trace alone**, which is why the reveal
shows z-traces at all. Putting the comparison here, as a function over the
traces, means the answer is computed rather than eyeballed off a plot - a
labeller squinting at six overlaid curves at 2 a.m. is exactly how a blind spot
gets recorded as a threshold problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from collections.abc import Sequence
from typing import Final

import numpy as np


class Verdict(StrEnum):
    """What the z-traces say about a missed artifact."""

    BLIND_SPOT = "generator_blind_spot"
    """z stayed low everywhere. Needs a new band or feature, not a lower threshold."""
    SUB_THRESHOLD = "threshold_too_high"
    """z rose but never crossed ``z_enter``. A lower ``z_enter`` would catch it."""
    DETECTED = "already_above_threshold"
    """z crossed. If no candidate exists here the fault is downstream of z - in
    the merge, duration or cardiac-suppression rules, not in the threshold."""
    UNASSESSABLE = "unassessable"
    """Every band is NaN here. Not a low-z finding - the opposite: nothing was
    measured, so no conclusion about the detector can be drawn."""


SUB_THRESHOLD_FLOOR: Final = 0.5
"""Fraction of ``z_enter`` above which a non-crossing counts as sub-threshold.

Below this the band did not meaningfully respond and calling it "threshold too
high" would send someone to lower a threshold that was never close. Half is a
declared convention, not a measurement; it is reported alongside the verdict so
a reader can apply their own.
"""


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """The verdict at one instant, with the evidence that produced it."""

    verdict: Verdict
    peak_z: float
    peak_band: str
    peak_signal: str
    z_enter: float

    @property
    def headline(self) -> str:
        """One line, which is what "in one glance" actually means."""
        if self.verdict is Verdict.UNASSESSABLE:
            return "UNASSESSABLE - no band could be measured here"
        return (
            f"{self.verdict.name}: peak z={self.peak_z:.2f} "
            f"(enter {self.z_enter:.2f}) in {self.peak_band} on {self.peak_signal}"
        )


def diagnose(
    traces: Sequence, t_start_s: float, t_stop_s: float
) -> Diagnosis:
    """Classify a span the human marked but the detector missed.

    The peak is taken over the span and over **every band**, because a miss is
    only a blind spot if *no* band saw it - one band being quiet while another
    shouts is not a blind spot, it is a routing question.
    """
    best_z = -np.inf
    best_band = ""
    best_signal = ""
    z_enter = float("nan")
    any_valid = False

    for tr in traces:
        z_enter = tr.z_enter
        lo = max(0, int(t_start_s / tr.grid_s))
        hi = min(tr.z_max.size, int(np.ceil(t_stop_s / tr.grid_s)))
        if hi <= lo:
            continue
        window = tr.z_max[lo:hi]
        finite = np.isfinite(window)
        if not finite.any():
            continue
        any_valid = True
        i = int(np.nanargmax(np.where(finite, window, -np.inf)))
        if window[i] > best_z:
            best_z = float(window[i])
            best_band = tr.band
            best_signal = tr.winner[lo + i]

    if not any_valid:
        return Diagnosis(Verdict.UNASSESSABLE, float("nan"), "", "", z_enter)
    if best_z >= z_enter:
        v = Verdict.DETECTED
    elif best_z >= SUB_THRESHOLD_FLOOR * z_enter:
        v = Verdict.SUB_THRESHOLD
    else:
        v = Verdict.BLIND_SPOT
    return Diagnosis(v, best_z, best_band, best_signal, z_enter)
