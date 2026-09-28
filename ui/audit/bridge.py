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


def reveal_for_region(
    rec: Any, region: tuple[float, float], z_enter: float | None = None
) -> tuple[np.ndarray, list[BandTrace]]:
    """Candidates and the six band traces for one assessable region.

    The detection itself is ``gems_blanking_v2.detect.chain.detect_region`` - the
    one construction site production shares (invariant 33; moved out of this file
    2026-09-28, and proved to give identical candidates on the 60 budget regions
    and the 5 round-1 regions). This function only places its result on the
    recording's own timeline for display: the candidate intervals already are, and
    each trace is padded with NaN frames ("not assessable here") up to the region
    start. ``z_enter`` defaults to the generator's own.
    """
    from gems_blanking_v2.constants import GRID_S
    from gems_blanking_v2.detect import chain

    found = chain.detect_region(rec, region, z_enter=z_enter)
    pad = round(region[0] / GRID_S)
    traces = [
        BandTrace(band=t.band, z_max=np.concatenate([np.full(pad, np.nan), t.z_max]),
                  winner=("",) * pad + t.winner, grid_s=t.grid_s, z_enter=t.z_enter)
        for t in reduce_to_band_traces(found.z, z_enter=found.report.z_enter, grid_s=GRID_S)
    ]
    return found.intervals, traces


def reveal_sha() -> str:
    """Hash of this module's source - the reveal composition a score or budget used.

    LF-normalised, so Windows and macOS checkouts agree. A budget measured under
    one reveal does not license a round under another (``recall.budget_status``).
    """
    import hashlib
    from pathlib import Path

    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
