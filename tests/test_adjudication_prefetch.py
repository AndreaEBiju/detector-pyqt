"""Change 1: the next recording is prefetched while the current one is judged.

Offscreen, hermetic. The loader is a fake whose every call blocks on a per-recording
gate the test opens, so "prefetch finished" / "still running" are deterministic, and
whose data are seeded. The example queue presents synth_a_rec1 (3 cores), synth_j_rec1,
then synth_old_1.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from gems_blanking_v2.io.store import GemsStore
from gems_blanking_v2.types import ChannelInfo, Recording

from ui.adjudicate.queue import example_queue, load_queue
from ui.adjudicate.session import AdjudicationSession

FS = 1000.0
USER = "tester"
SEEDS = {"synth_a_rec1": 31, "synth_j_rec1": 32, "synth_old_1": 33}
A, J, OLD = "synth_a_rec1", "synth_j_rec1", "synth_old_1"


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def store(tmp_path: Path) -> GemsStore:
    return GemsStore.initialise(tmp_path / "gems")


class GatedLoader:
    """Fake loader: each call waits for its recording's gate; records thread and order."""

    def __init__(self, *, fail_first: set[str] = frozenset(), open_now: set[str] = frozenset()):
        self.gates = {r: threading.Event() for r in SEEDS}
        for r in open_now:
            self.gates[r].set()
        self.calls: list[tuple[str, bool]] = []  # (recording, on the GUI thread?)
        self.fail_first = set(fail_first)
        self.gui = threading.get_ident()
        self.lock = threading.Lock()

    def __call__(self, row: dict[str, Any]) -> Recording:
        rid = str(row["recording"])
        with self.lock:
            self.calls.append((rid, threading.get_ident() == self.gui))
        assert self.gates[rid].wait(10), f"gate for {rid} never opened"
        with self.lock:
            if rid in self.fail_first:
                self.fail_first.discard(rid)
                msg = f"Drive hiccup reading {rid}"
                raise OSError(msg)
        n_ch = 9 if row["cohort"] == "new" else 5
        dur = 200.0 if row["cohort"] == "new" else 60.0
        rng = np.random.default_rng(SEEDS[rid])
        data = rng.standard_normal((int(dur * FS), n_ch)) * 10.0
        chans = [ChannelInfo(i, f"C{i}", "nerve", None, None, None, "independent")
                 for i in range(n_ch)]
        return Recording(fs=FS, data=data, channels=chans, animal=row["animal"], session=rid,
                         path=Path(rid))

    def loads_of(self, rid: str) -> list[bool]:
        return [gui for r, gui in self.calls if r == rid]


def _window(store: GemsStore, tmp_path: Path, loader: GatedLoader):
    from ui.windows.adjudication_window import AdjudicationWindow

    path = tmp_path / "queue.parquet"
    example_queue().to_parquet(path, index=False)
    session = AdjudicationSession(load_queue(path), store, USER, journal_dir=tmp_path / "j",
                                  queue_file=path.name, queue_sha256="0" * 64, app_sha=None)
    loader.gates[A].set()  # the first recording loads normally (the GUI thread waits)
    w = AdjudicationWindow(session, load_fn=loader, async_traces=False, prefetch=True)
    return w


def _until(qapp, cond, timeout: float = 5.0) -> None:
    end = time.monotonic() + timeout
    while not cond():
        qapp.processEvents()
        if time.monotonic() > end:
            msg = "condition not reached"
            raise AssertionError(msg)
        time.sleep(0.01)


def _future_done(w) -> bool:
    return w._pf_future is not None and w._pf_future.done()


def test_the_next_recording_is_prefetched_off_the_gui_thread_and_used(qapp, store, tmp_path
                                                                    ) -> None:
    loader = GatedLoader()
    w = _window(store, tmp_path, loader)
    assert w.prefetch_log == [f"start {J}"]
    loader.gates[J].set()
    _until(qapp, lambda: _future_done(w))
    for _ in range(3):
        w.press("2")  # the three A cores; the third moves to J
    assert w._rec_id == J and w._shown_key == w.session.current()["core_key"]
    assert loader.loads_of(J) == [False]  # loaded once, and not on the GUI thread
    assert loader.loads_of(A) == [True]
    assert f"hit {J}" in w.prefetch_log
    assert w.prefetch_log[-1] == f"start {OLD}"  # and the next one is on its way


def test_arriving_before_the_prefetch_finishes_waits_for_that_one_load(qapp, store, tmp_path
                                                                       ) -> None:
    loader = GatedLoader()
    w = _window(store, tmp_path, loader)
    for _ in range(3):
        w.press("2")
    assert f"wait {J}" in w.prefetch_log
    assert w._shown_key is None and not any(b.isEnabled() for b in w._buttons.values())
    assert w.press("1") is None  # the keys are refused while it loads
    assert "Loading" in w.core_info.text()
    loader.gates[J].set()
    _until(qapp, lambda: w._shown_key is not None)
    assert w._rec_id == J and loader.loads_of(J) == [False]  # no second load
    assert w.session.judgement_of(w.session.rows[3]) is None


def test_a_failed_prefetch_falls_back_to_a_normal_load_with_the_error_shown(qapp, store,
                                                                            tmp_path) -> None:
    loader = GatedLoader(fail_first={J})
    w = _window(store, tmp_path, loader)
    loader.gates[J].set()
    _until(qapp, lambda: _future_done(w))
    for _ in range(3):
        w.press("2")
    assert f"fail {J}" in w.prefetch_log and "Drive hiccup" in (w.last_error or "")
    assert w._rec_id == J and w._shown_key is not None
    assert loader.loads_of(J) == [False, True]  # the prefetch, then a normal load


def test_jumping_elsewhere_discards_the_prefetch(qapp, store, tmp_path) -> None:
    loader = GatedLoader(open_now={OLD})
    w = _window(store, tmp_path, loader)
    assert w._pf_rid == J
    held_at_load: list[str | None] = []
    real = w._load_fn

    def watching(row):  # what is still held when the out-of-order load starts
        held_at_load.append(w._pf_rid)
        return real(row)

    w._load_fn = watching
    w.session.cursor = 4  # synth_old_1, out of order
    w.show_current()
    assert held_at_load == [None]  # discarded BEFORE loading: never two recordings held
    assert f"discard {J}" in w.prefetch_log and w._rec_id == OLD
    assert loader.loads_of(OLD) == [True]  # a normal load: no prefetch was for it
    assert w._pf_rid is None or w._pf_rid != J
    loader.gates[J].set()  # the abandoned load may finish; nothing holds its result


def test_nothing_is_prefetched_past_the_end_of_the_queue(qapp, store, tmp_path) -> None:
    loader = GatedLoader(open_now={J, OLD})
    w = _window(store, tmp_path, loader)
    _until(qapp, lambda: _future_done(w))
    w.skip()  # leave the first A core unjudged: a wrap-around would find it
    w.press("2")
    w.press("2")  # A -> J
    _until(qapp, lambda: w._rec_id == J and w._shown_key is not None)
    w.press("2")  # J -> OLD, the last recording
    _until(qapp, lambda: w._rec_id == OLD and w._shown_key is not None)
    starts = [e for e in w.prefetch_log if e.startswith("start")]
    assert starts == [f"start {J}", f"start {OLD}"]  # nothing past the end, no wrap
    assert w._pf_rid is None


def test_at_most_one_recording_is_prefetched(qapp, store, tmp_path) -> None:
    loader = GatedLoader(open_now={J, OLD})
    w = _window(store, tmp_path, loader)
    _until(qapp, lambda: _future_done(w))
    w.press("2")  # still on A: the same prefetch is kept, not a second one
    assert [e for e in w.prefetch_log if e.startswith("start")] == [f"start {J}"]


def test_prefetch_is_off_unless_asked(qapp, store, tmp_path) -> None:
    from ui.windows.adjudication_window import AdjudicationWindow

    loader = GatedLoader(open_now=set(SEEDS))
    path = tmp_path / "queue.parquet"
    example_queue().to_parquet(path, index=False)
    session = AdjudicationSession(load_queue(path), store, USER, journal_dir=tmp_path / "j",
                                  queue_file=path.name, queue_sha256="0" * 64, app_sha=None)
    w = AdjudicationWindow(session, load_fn=loader, async_traces=False)
    assert w.prefetch_log == [] and w._pool is None


def test_the_previous_recording_is_freed_on_moving_on(qapp, store, tmp_path) -> None:
    """Measured on set A: without an explicit release, each old recording stayed resident."""
    import gc
    import weakref

    loader = GatedLoader(open_now={J, OLD})
    w = _window(store, tmp_path, loader)
    first = weakref.ref(w._recording.data)
    _until(qapp, lambda: _future_done(w))
    for _ in range(3):
        w.press("2")  # -> J
    assert w._rec_id == J
    gc.collect()
    assert first() is None
