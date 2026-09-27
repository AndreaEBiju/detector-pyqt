"""Keeps a :class:`MultiChannelViewer` consistent with an audit's phase.

One function does the whole job, and it is written so that **the viewer's state
is derived from the session rather than tracked alongside it**. A UI that toggles
overlays on its own would have to get every path right - mode change, dock
restore, repaint, undo - and the one path it got wrong would silently show
candidates during blind marking and quietly invalidate the gate.

Here there is nothing to get wrong: :func:`sync_viewer` reads the phase and sets
the overlay to whatever that phase permits. ``SpanSession`` returns an empty
array for candidates until it is revealed, so BLIND cannot paint them even if
this function is called at the wrong moment.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ui.audit.session import Phase, SpanSession


def seconds_to_samples(intervals: np.ndarray, fs: float) -> np.ndarray:
    """``(n, 2)`` seconds to the 1-based inclusive samples the viewer expects.

    ``MultiChannelViewer.set_model_intervals`` renders ``(s - 1) / fs``, so it
    wants 1-based sample indices. Converting here, at the boundary and once,
    rather than leaving each call site to remember - an off-by-one would shift
    every candidate band by a sample and be invisible on screen.
    """
    arr = np.asarray(intervals, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return (np.round(arr * fs) + 1).astype(np.int64)


def sync_viewer(
    viewer: Any,
    session: SpanSession,
    fs: float,
    pending_candidates: np.ndarray | None = None,
) -> None:
    """Make the viewer show exactly what ``session.phase`` allows.

    ``pending_candidates`` is the realistic and dangerous case: the app computes
    candidates once when the span is opened, because they are needed the moment
    the reveal happens, and then holds them in memory THROUGHOUT blind marking.
    They are therefore available to paint at every repaint, and only the phase
    check stops them. Passing them in rather than hiding them from this function
    is what makes the guard testable - a test where no candidate exists anywhere
    proves nothing, since there is nothing that could have been painted.

    Candidates are passed to the viewer as ``None`` while blind, which is what
    ``set_model_intervals`` takes to mean "remove the overlay items", rather than
    an empty array that might leave stale items behind.
    """
    marks = session.marks_array()
    viewer.set_bad_intervals(seconds_to_samples(marks, fs))

    if session.phase is Phase.REVEALED:
        shown = session.candidates
        if shown.size == 0 and pending_candidates is not None:
            shown = np.asarray(pending_candidates, dtype=np.float64)
        viewer.set_model_intervals(seconds_to_samples(shown, fs))
    else:
        # None, not an empty array: None is the documented "clear it" value.
        viewer.set_model_intervals(None)


def model_overlay_count(viewer: Any) -> int:
    """How many candidate overlay items are actually in the scene.

    Counts the items themselves rather than reading a flag, because the failure
    this guards against is precisely a flag saying "hidden" while an item is
    still painted - a stale paint, a signal connected at construction, a dock
    restoring its previous state.
    """
    items = getattr(viewer, "_model_region_items", None)
    if not items:
        return 0
    return sum(len(per_channel) for per_channel in items)
