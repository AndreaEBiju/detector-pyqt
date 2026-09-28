"""The only test that speaks to the gate's validity: nothing is painted in BLIND.

Every other audit test is below the UI. A Qt bug that paints the candidate
overlay during blind marking - a stale paint, a signal connected at
construction, a dock restoring its previous state - defeats the audit entirely
and none of them would catch it.

So this drives the REAL widget and interrogates the REAL scene items, not a
model flag. A flag saying "hidden" while an item is still on screen is exactly
the failure being guarded against.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import h5py
import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from ui.audit.controller import model_overlay_count, seconds_to_samples, sync_viewer
from ui.audit.session import SpanSession
from ui.widgets.signal_viewer import LazyRecording, MultiChannelViewer

FS = 1000.0
N = 20_000
N_CH = 3


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def viewer(qapp, tmp_path_factory):
    """A real MultiChannelViewer over a real (tiny) recording."""
    path = Path(tmp_path_factory.mktemp("rec")) / "r.h5"
    with h5py.File(path, "w") as f:
        rng = np.random.default_rng(0)
        f["y"] = rng.normal(0, 1, size=(N, N_CH)).astype(np.float32)
        f["fs"] = FS
    rec = LazyRecording(path)
    v = MultiChannelViewer(rec)
    v.set_viewport(0.0, 10.0)
    yield v
    # Destroy the widget BEFORE closing the file it reads. Closing the file first
    # left a live viewer over a closed HDF5 dataset until garbage collection, and a
    # later paint read through it: an access violation in 3-4 of 6 full-suite runs
    # once test ordering changed (2026-09-28).
    from PySide6.QtCore import QCoreApplication, QEvent

    v.close()
    v.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    del v
    rec.close()


def _session() -> SpanSession:
    return SpanSession("s0", "rec", 0.0, 20.0)


CANDIDATES = np.array([[2.0, 3.0], [5.0, 6.5]])


def test_no_candidate_item_exists_in_the_scene_while_blind(viewer, tmp_path) -> None:
    """The circularity guard, checked on the rendered scene."""
    s = _session()
    s.add_mark(1.0, 1.5)

    # The candidates EXIST and are in hand - the app computes them when the span
    # opens because it needs them the instant the reveal happens. Only the phase
    # check stops them being painted. A test without them here would prove
    # nothing: there would be nothing that could have been painted.
    sync_viewer(viewer, s, FS, pending_candidates=CANDIDATES)

    assert model_overlay_count(viewer) == 0, (
        "a candidate overlay item exists during BLIND marking; the labeller "
        "would anchor on it and the gate would measure its own output"
    )


def test_candidates_appear_only_after_commit(viewer, tmp_path) -> None:
    """And they do appear - a guard that hides them forever is also broken."""
    s = _session()
    sync_viewer(viewer, s, FS, pending_candidates=CANDIDATES)
    assert model_overlay_count(viewer) == 0

    s.commit(tmp_path)
    sync_viewer(viewer, s, FS, pending_candidates=CANDIDATES)
    assert model_overlay_count(viewer) == 0, "still hidden between commit and reveal"

    s.reveal(candidates=CANDIDATES, traces=[])
    sync_viewer(viewer, s, FS)

    assert model_overlay_count(viewer) == len(CANDIDATES) * N_CH


def test_a_stale_reveal_does_not_survive_into_a_new_blind_span(viewer, tmp_path) -> None:
    """A dock or viewer reused across spans must not carry the last span's
    candidates into the next span's blind phase."""
    done = _session()
    done.commit(tmp_path)
    done.reveal(candidates=CANDIDATES, traces=[])
    sync_viewer(viewer, done, FS)
    assert model_overlay_count(viewer) > 0

    fresh = SpanSession("s1", "rec", 0.0, 20.0)
    sync_viewer(viewer, fresh, FS)

    assert model_overlay_count(viewer) == 0, "previous span's overlay survived"


def test_the_human_marks_are_still_shown_while_blind(viewer, tmp_path) -> None:
    """Blind hides the DETECTOR, not the labeller's own work - they must see
    what they have already marked or they cannot mark coherently."""
    s = _session()
    s.add_mark(1.0, 1.5)
    s.add_mark(4.0, 4.2)

    sync_viewer(viewer, s, FS)

    np.testing.assert_allclose(viewer.bad_intervals, seconds_to_samples(
        np.array([[1.0, 1.5], [4.0, 4.2]]), FS))


def test_seconds_convert_to_one_based_samples() -> None:
    """set_model_intervals renders (s - 1)/fs, so it wants 1-based indices.
    An off-by-one would shift every band by a sample and be invisible."""
    got = seconds_to_samples(np.array([[0.0, 1.0]]), 1000.0)

    np.testing.assert_array_equal(got, [[1, 1001]])
    assert seconds_to_samples(np.zeros((0, 2)), 1000.0).shape == (0, 2)
