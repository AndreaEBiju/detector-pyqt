"""Change 1: widening a core's boundary (Shift+drag) - stored beside the core, never over it.

Hermetic: tmp_path stores and journals, an injected loader, offscreen Qt. The window
tests drive real Shift+drag mouse events and real keystrokes.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from gems_blanking_v2.io.store import GemsStore
from gems_blanking_v2.types import ChannelInfo, Recording

from ui.adjudicate import judgements as jd
from ui.adjudicate.queue import check_queue, core_key, example_queue, load_queue
from ui.adjudicate.session import AdjudicationSession

FS = 2000.0
USER = "tester"
SEEDS = {"synth_a_rec1": 21, "synth_j_rec1": 22, "synth_old_1": 23}


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def store(tmp_path: Path) -> GemsStore:
    return GemsStore.initialise(tmp_path / "gems")


def _session(store: GemsStore, tmp_path: Path) -> AdjudicationSession:
    path = tmp_path / "queue.parquet"
    example_queue().to_parquet(path, index=False)
    return AdjudicationSession(load_queue(path), store, USER, journal_dir=tmp_path / "journal",
                               queue_file=path.name, queue_sha256="0" * 64, app_sha=None)


def _shards(store: GemsStore) -> pd.DataFrame:
    files = sorted((store.root / "labels").rglob("events_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _row(i: int = 0) -> dict:
    return check_queue(example_queue()).iloc[i].to_dict()


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------


def test_a_widened_boundary_sits_beside_the_core_never_over_it() -> None:
    row = _row()
    rec = jd.make_record(row, "motion", user=USER, app_sha=None, queue_file="q",
                         queue_sha256="0", widened=(19.5, 20.8))
    assert (rec["start_s"], rec["stop_s"]) == (row["start_s"], row["stop_s"])
    assert (rec["widened_start_s"], rec["widened_stop_s"]) == (19.5, 20.8)
    assert core_key(rec["recording"], rec["start_s"], rec["stop_s"]) == row["core_key"]


@pytest.mark.parametrize("judgement", ["physiology", "unsure", "line_noise"])
def test_a_widened_boundary_goes_only_with_motion(judgement: str) -> None:
    with pytest.raises(ValueError, match="only with motion"):
        jd.make_record(_row(), judgement, user=USER, app_sha=None, queue_file="q",
                       queue_sha256="0", widened=(19.5, 20.8))


def test_a_widened_boundary_must_contain_the_core() -> None:
    with pytest.raises(ValueError, match="contain the core"):
        jd.make_record(_row(), "motion", user=USER, app_sha=None, queue_file="q",
                       queue_sha256="0", widened=(20.05, 20.8))


def test_the_writer_refuses_a_forged_or_half_widen(store: GemsStore) -> None:
    rec = jd.make_record(_row(), "physiology", user=USER, app_sha=None, queue_file="q",
                         queue_sha256="0")
    with pytest.raises(ValueError, match="only with motion"):
        jd.write_shards(store, [{**rec, "widened_start_s": 19.0, "widened_stop_s": 21.0}], USER)
    mot = jd.make_record(_row(), "motion", user=USER, app_sha=None, queue_file="q",
                         queue_sha256="0")
    with pytest.raises(ValueError, match="both"):
        jd.write_shards(store, [{**mot, "widened_start_s": 19.0}], USER)
    assert not list((store.root / "labels").rglob("*.parquet"))


# ---------------------------------------------------------------------------
# the session: undo, journal, resume, write-once, test set
# ---------------------------------------------------------------------------


def test_widen_round_trips_through_the_journal_and_the_shard(store, tmp_path) -> None:
    s = _session(store, tmp_path)
    s.judge("motion", widened=(19.0, 21.0))
    s.judge("physiology")
    line = s.journal_path.read_text(encoding="utf-8").splitlines()
    first, second = json.loads(line[0]), json.loads(line[1])
    assert (first["widened_start_s"], first["widened_stop_s"]) == (19.0, 21.0)
    assert "widened_start_s" not in second  # absent, never null or NaN
    again = _session(store, tmp_path)  # the app died: the journal restores both
    assert again.restored == 2
    again.close()
    out = _shards(store).sort_values("start_s")
    assert out["widened_start_s"].tolist()[0] == 19.0
    assert np.isnan(out["widened_start_s"].tolist()[1])
    assert out["start_s"].tolist()[0] == 20.0  # the core unchanged


def test_undo_withdraws_the_judgement_and_its_widen(store, tmp_path) -> None:
    s = _session(store, tmp_path)
    s.judge("motion", widened=(19.0, 21.0))
    rec = s.undo()
    assert rec is not None and rec["widened_start_s"] == 19.0
    s.judge("motion")  # judged again, without a widen
    s.close()
    out = _shards(store)
    assert len(out) == 1 and np.isnan(out["widened_start_s"].iloc[0])


def test_a_test_set_row_keeps_label_set_test_when_widened(store, tmp_path) -> None:
    s = _session(store, tmp_path)
    for _ in range(3):
        s.judge("physiology")
    j = s.current()
    assert j is not None and j["animal"] == "J"
    s.judge("motion", widened=(29.0, 31.0))
    s.close()
    out = _shards(store)
    jrow = out[out["animal"] == "J"].iloc[0]
    assert jrow["label_set"] == "test" and jrow["widened_stop_s"] == 31.0


# ---------------------------------------------------------------------------
# the window: real Shift+drag and keystrokes
# ---------------------------------------------------------------------------


def _fake_recording(row: dict) -> Recording:
    n_ch = 9 if row["cohort"] == "new" else 5
    dur = 200.0 if row["cohort"] == "new" else 60.0
    rng = np.random.default_rng(SEEDS[row["recording"]])
    data = rng.standard_normal((int(dur * FS), n_ch)) * 10.0
    chans = [ChannelInfo(i, f"C{i}", "nerve", None, None, None, "independent")
             for i in range(n_ch)]
    return Recording(fs=FS, data=data, channels=chans, animal=row["animal"],
                     session=row["recording"], path=Path(row["recording"]))


def _window(store: GemsStore, tmp_path: Path):
    """The window, after PROBING that this platform delivers shortcut keys at all.

    The probe is Space on a throwaway window (it only moves the cursor). If it fails, the
    platform cannot test keystrokes and the test is skipped for that reason alone; the
    widen behaviour itself is then asserted unconditionally.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from ui.windows.adjudication_window import AdjudicationWindow

    app = QApplication.instance()
    # A window left visible by an earlier test keeps the focus and swallows the shortcuts;
    # hide (not close: no store writes) every other top-level window first.
    for other in app.topLevelWidgets():
        if other.isVisible():
            other.hide()
    (tmp_path / "probe").mkdir(exist_ok=True)
    probe = AdjudicationWindow(_session(store, tmp_path / "probe"), load_fn=_fake_recording,
                               async_traces=False)
    probe.show()
    _activate(probe)
    before = probe.session.cursor
    QTest.keyClick(probe, Qt.Key_Space)
    app.processEvents()
    delivered = probe.session.cursor != before
    probe.session.close = list  # type: ignore[method-assign, assignment]
    probe.close()
    if not delivered:
        pytest.skip("this platform does not deliver shortcut key events (Space probe)")
    w = AdjudicationWindow(_session(store, tmp_path), load_fn=_fake_recording,
                           async_traces=False)
    w.resize(1400, 900)
    w.show()
    _activate(w)
    return w


def _activate(win) -> bool:
    """Make ``win`` the active window. Offscreen, the first activation after the previous
    active window was hidden can be dropped, so it is retried."""
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    for _ in range(5):
        win.raise_()
        win.activateWindow()
        QApplication.instance().processEvents()
        if QTest.qWaitForWindowActive(win, 400):
            return True
    return False


def _shift_drag(qapp, w, x0: float, x1: float) -> None:
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtTest import QTest

    vb = w.viewer.plots[0][0].getViewBox()
    ymid = sum(vb.viewRange()[1]) / 2

    def pos(x):
        return w.viewer.mapFromScene(vb.mapViewToScene(QPointF(x, ymid)))

    vp = w.viewer.viewport()
    QTest.mousePress(vp, Qt.LeftButton, Qt.ShiftModifier, pos(x0))
    QTest.mouseMove(vp, pos(x1))
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.ShiftModifier, pos(x1))
    qapp.processEvents()


def _key(qapp, w, key, mod=None) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    QTest.keyClick(w, key, mod or Qt.NoModifier)
    qapp.processEvents()


def test_shift_drag_then_1_records_motion_with_the_widen(qapp, store, tmp_path) -> None:
    from PySide6.QtCore import Qt

    w = _window(store, tmp_path)
    qapp.processEvents()
    row = w.session.current()  # synth_a_rec1, core 20.00-20.12, view 18.00-22.12
    _shift_drag(qapp, w, 19.0, 20.6)
    widen = w.pending_widen
    assert widen is not None
    assert widen[0] == pytest.approx(19.0, abs=0.05) and widen[1] == pytest.approx(20.6, abs=0.05)
    assert len(w._widen_items) == len(w.viewer.plots)  # drawn on every channel
    _shift_drag(qapp, w, 20.5, 21.5)  # a second drag extends the same widen
    assert w.pending_widen[1] == pytest.approx(21.5, abs=0.05)
    assert w.pending_widen[0] == widen[0]
    _key(qapp, w, Qt.Key_1)
    assert w.session.position()[0] == 1
    assert w.pending_widen is None and w._widen_items == []
    w.close()
    rec = _shards(store).iloc[0]
    assert rec["judgement"] == "motion"
    assert (rec["start_s"], rec["stop_s"]) == (row["start_s"], row["stop_s"])
    assert rec["widened_start_s"] == pytest.approx(19.0, abs=0.05)
    assert rec["widened_stop_s"] == pytest.approx(21.5, abs=0.05)


def test_a_non_motion_key_is_refused_while_a_widen_is_pending(qapp, store, tmp_path) -> None:
    from PySide6.QtCore import Qt

    w = _window(store, tmp_path)
    qapp.processEvents()
    _shift_drag(qapp, w, 19.0, 20.6)
    assert w.pending_widen is not None
    for k in (Qt.Key_2, Qt.Key_3, Qt.Key_4):
        _key(qapp, w, k)
    assert w.press("2") is None and w.session.position()[0] == 0
    assert "only with motion" in w.status.text()
    _key(qapp, w, Qt.Key_Escape)
    assert w.pending_widen is None
    _key(qapp, w, Qt.Key_2)
    assert w.session.judgement_of(w.session.rows[0]) == "physiology"


def test_ctrl_z_clears_a_pending_widen_before_undoing_a_judgement(qapp, store, tmp_path
                                                                  ) -> None:
    from PySide6.QtCore import Qt

    w = _window(store, tmp_path)
    qapp.processEvents()
    w.press("2")  # core 0 judged physiology
    _shift_drag(qapp, w, 54.0, 56.0)  # core 1 (55.5-55.6) widened
    assert w.pending_widen is not None
    _key(qapp, w, Qt.Key_Z, Qt.ControlModifier)
    assert w.pending_widen is None
    assert w.session.judgement_of(w.session.rows[0]) == "physiology"  # not undone yet
    _key(qapp, w, Qt.Key_Z, Qt.ControlModifier)
    assert w.session.judgement_of(w.session.rows[0]) is None


def test_a_widen_does_not_follow_the_labeller_to_the_next_core(qapp, store, tmp_path) -> None:
    w = _window(store, tmp_path)
    qapp.processEvents()
    _shift_drag(qapp, w, 19.0, 20.6)
    w.skip()
    assert w.pending_widen is None
    w.press("1")
    w.close()
    assert _shards(store)["widened_start_s"].isna().all()


def test_a_drag_is_clipped_to_the_cores_region(qapp, store, tmp_path) -> None:
    w = _window(store, tmp_path)
    qapp.processEvents()
    assert w.widen(2.0, 20.5) == pytest.approx((10.0, 20.5))  # region starts at 10 s
    w.clear_widen()
    assert w.widen(19.9, 20.05) == pytest.approx((19.9, 20.12))  # never narrower than the core


@pytest.mark.parametrize("bad", [(float("-inf"), 21.0), (19.0, float("inf")),
                                 (float("nan"), 21.0)])
def test_a_non_finite_widen_is_refused(bad: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="finite"):
        jd.make_record(_row(), "motion", user=USER, app_sha=None, queue_file="q",
                       queue_sha256="0", widened=bad)


def test_a_widen_outside_the_region_is_refused() -> None:
    with pytest.raises(ValueError, match="region"):  # region of row 0 is 10-190 s
        jd.make_record(_row(), "motion", user=USER, app_sha=None, queue_file="q",
                       queue_sha256="0", widened=(5.0, 21.0))


def test_a_drag_inside_the_core_stores_no_widen(qapp, store, tmp_path) -> None:
    w = _window(store, tmp_path)
    qapp.processEvents()
    assert w.widen(20.03, 20.05) is None  # core is 20.00-20.12
    assert w.pending_widen is None and w._widen_items == []


def test_undo_is_enabled_while_a_widen_is_pending(qapp, store, tmp_path) -> None:
    w = _window(store, tmp_path)
    qapp.processEvents()
    assert not w._undo_btn.isEnabled()
    w.widen(19.0, 20.5)
    assert w._undo_btn.isEnabled()
    w._undo_btn.click()
    assert w.pending_widen is None


def test_a_half_widened_journal_line_is_dropped_and_logged(store, tmp_path, caplog) -> None:
    s = _session(store, tmp_path)
    s.judge("motion", widened=(19.0, 21.0))
    line = json.loads(s.journal_path.read_text(encoding="utf-8").splitlines()[0])
    del line["widened_stop_s"]
    s.journal_path.write_text(json.dumps(line) + "\n", encoding="utf-8", newline="\n")
    with caplog.at_level("WARNING"):
        again = _session(store, tmp_path)
    assert again.restored == 0 and again.dropped == [line["judgement_id"]]
    assert "dropped on restore" in caplog.text
