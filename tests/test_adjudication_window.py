"""The adjudication window driven headless (offscreen), and the two recording loaders.

A synthetic queue (``example_queue``: two new-cohort recordings and one old-cohort
one) drives the real window with an injected loader, so no store recording, Drive
path or display is touched. The loader tests write their own tiny files in tmp_path.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from gems_blanking_v2.io.store import GemsStore
from gems_blanking_v2.types import ChannelInfo, Recording
from scipy.io import savemat

from ui.adjudicate import loaders
from ui.adjudicate.queue import example_queue, load_queue
from ui.adjudicate.session import AdjudicationSession
from ui.audit.bridge import BandTrace

FS = 2000.0
USER = "tester"
SEEDS = {"synth_a_rec1": 11, "synth_j_rec1": 12, "synth_old_1": 13}


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def store(tmp_path: Path) -> GemsStore:
    return GemsStore.initialise(tmp_path / "gems")


def _fake_recording(row: dict) -> Recording:
    n_ch = 9 if row["cohort"] == "new" else 5
    dur = 200.0 if row["cohort"] == "new" else 60.0
    rng = np.random.default_rng(SEEDS[row["recording"]])
    data = rng.standard_normal((int(dur * FS), n_ch)) * 10.0
    chans = [ChannelInfo(i, f"C{i}", "nerve", None, None, None, "independent")
             for i in range(n_ch)]
    return Recording(fs=FS, data=data, channels=chans, animal=row["animal"],
                     session=row["recording"], path=Path(row["recording"]))


def _window(store: GemsStore, tmp_path: Path, **kw):
    from ui.windows.adjudication_window import AdjudicationWindow

    path = tmp_path / "queue.parquet"
    example_queue().to_parquet(path, index=False)
    session = AdjudicationSession(load_queue(path), store, USER,
                                  journal_dir=tmp_path / "journal", queue_file=path.name,
                                  queue_sha256="0" * 64, app_sha=None)
    loads: list[str] = []

    def load_fn(row):
        loads.append(row["recording"])
        return _fake_recording(row)

    w = AdjudicationWindow(session, load_fn=load_fn, async_traces=False, **kw)
    w.loads = loads
    return w


def _written(store: GemsStore) -> pd.DataFrame:
    files = sorted((store.root / "labels").rglob("events_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def test_a_synthetic_queue_is_judged_end_to_end_by_keystroke(qapp, store, tmp_path) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    w = _window(store, tmp_path)
    w.resize(1400, 900)
    w.show()
    w.activateWindow()
    qapp.processEvents()
    assert w.last_error is None, w.last_error
    first = w.session.current()
    # the core is in view and highlighted on every channel
    lo, hi = w.viewer.time_range
    assert lo <= first["start_s"] and first["stop_s"] <= hi
    assert len(w._core_items) == len(w.viewer.plots) == 9
    assert w._core_items[0].getRegion() == pytest.approx((first["start_s"], first["stop_s"]))

    QTest.keyClick(w, Qt.Key_1)
    QTest.keyClick(w, Qt.Key_4)
    qapp.processEvents()
    if w.session.position()[0] == 0:  # shortcuts need an active window
        pytest.skip("offscreen platform did not deliver shortcut key events")
    judged = [w.session.judgement_of(r) for r in w.session.rows[:2]]
    assert judged == ["motion", "line_noise"]
    QTest.keyClick(w, Qt.Key_Z, Qt.ControlModifier)  # undo the 4
    qapp.processEvents()
    assert w.session.judgement_of(w.session.rows[1]) is None
    QTest.keyClick(w, Qt.Key_2)
    QTest.keyClick(w, Qt.Key_3)  # third A core
    QTest.keyClick(w, Qt.Key_4)  # J (test set): new recording loaded
    QTest.keyClick(w, Qt.Key_1)  # old cohort: 5 channels
    qapp.processEvents()
    assert w.session.current() is None
    assert w.loads == ["synth_a_rec1", "synth_j_rec1", "synth_old_1"]  # one load each
    assert w.session.pending == 5  # finishing does not write: the last stay undoable
    w.close()
    out = _written(store)  # closing writes everything
    assert sorted(out["judgement"]) == sorted(["motion", "physiology", "unsure",
                                               "line_noise", "motion"])
    assert out.loc[out["animal"] == "J", "label_set"].tolist() == ["test"]


def test_buttons_judge_skip_and_undo_without_a_keyboard(qapp, store, tmp_path) -> None:
    w = _window(store, tmp_path)
    rows = w.session.rows
    w._buttons["4"].click()
    assert w.session.judgement_of(rows[0]) == "line_noise"
    w._skip_btn.click()
    assert w.session.current() is rows[2]
    w._undo_btn.click()
    assert w.session.current() is rows[0] and w.session.judgement_of(rows[0]) is None
    w._buttons["2"].click()
    w._save_btn.click()
    assert list(_written(store)["judgement"]) == ["physiology"]
    assert not w._undo_btn.isEnabled()  # written: undo cannot reach it


def test_moving_to_another_recording_replaces_the_viewer(qapp, store, tmp_path) -> None:
    w = _window(store, tmp_path)
    for _ in range(3):
        w.press("1")
    assert w.session.current()["recording"] == "synth_j_rec1"
    assert len(w.viewer.plots) == 9
    w.press("2")
    assert w.session.current()["cohort"] == "old"
    assert len(w.viewer.plots) == 5 and len(w._core_items) == 5
    # the other queue cores of this recording are shaded for context
    assert sum(len(x) for x in w.viewer._model_region_items) == 5


def test_closing_the_window_writes_pending_judgements(qapp, store, tmp_path) -> None:
    w = _window(store, tmp_path)
    w.press("1")
    w.press("3")
    assert not list((store.root / "labels").rglob("*.parquet"))
    w.close()
    assert list(_written(store)["judgement"]) == ["motion", "unsure"]


def test_a_recording_that_cannot_load_is_shown_not_raised(qapp, store, tmp_path) -> None:
    from ui.windows.adjudication_window import AdjudicationWindow

    path = tmp_path / "queue.parquet"
    example_queue().to_parquet(path, index=False)
    session = AdjudicationSession(load_queue(path), store, USER,
                                  journal_dir=tmp_path / "journal", queue_file=path.name,
                                  queue_sha256="0" * 64, app_sha=None)

    def broken(_row):
        raise FileNotFoundError("no meta.json for A/synth_a_rec1")

    w = AdjudicationWindow(session, load_fn=broken, async_traces=False)
    assert "no meta.json" in (w.last_error or "")


def _traces(region):
    n = int(region[1] / 0.01)
    z = np.zeros(n)
    z[2000:2012] = 9.0
    return [BandTrace(band=b, z_max=z, winner=("x",) * n, grid_s=0.01, z_enter=3.0)
            for b in ("300-3000", "100-300", "10-150", "2-50", "0.5-3", "0-2")]


def test_z_traces_are_off_until_ticked_then_computed_once_per_region(qapp, store,
                                                                     tmp_path) -> None:
    calls: list[tuple[float, float]] = []

    def traces_fn(_rec, region):
        calls.append(region)
        return np.zeros((0, 2)), _traces(region)

    w = _window(store, tmp_path, traces_fn=traces_fn)
    assert calls == [] and w.ztrace.n_curves_with_data() == 0
    w.traces_box.setChecked(True)
    assert calls == [(10.0, 190.0)] and w.ztrace.n_curves_with_data() == 6
    w.press("1")  # same recording and region: cached
    assert calls == [(10.0, 190.0)] and w.ztrace.n_curves_with_data() == 6


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def test_old_cohort_signal_is_declared_volts_converted_to_microvolts(tmp_path) -> None:
    rng = np.random.default_rng(3)
    y_v = rng.standard_normal((20_000, 5)) * 10e-6  # ~10 uV noise, in volts
    folder = tmp_path / "Survivals" / "rat1"
    folder.mkdir(parents=True)
    savemat(folder / "M1_FOO_M1_bl_1200_notched.mat", {"y": y_v, "fs": 24414.0625})
    path = loaders.old_signal_path(tmp_path / "Survivals", "rat1", "M1_FOO_M1_bl_1200")
    rec = loaders.load_old_signal(path, "M1_FOO_M1_bl_1200", "F", units="V")
    assert rec.data == pytest.approx(y_v * 1e6)
    assert [c.name for c in rec.channels] == list(loaders.OLD_COHORT_CHANNELS)
    assert {c.config for c in rec.channels} == {"hw_tripole"}
    assert rec.fs == 24414.0625
    with pytest.raises(ValueError, match="units declared as 'uV'"):
        loaders.load_old_signal(path, "M1_FOO_M1_bl_1200", "F", units="uV")


def test_old_cohort_units_must_be_declared_and_shape_is_never_guessed(tmp_path) -> None:
    rng = np.random.default_rng(5)
    path = tmp_path / "x_notched.mat"
    savemat(path, {"y": rng.standard_normal((20_000, 5)) * 10e-6, "fs": 24414.0625})
    with pytest.raises(TypeError, match="units"):
        loaders.load_old_signal(path, "x", "F")  # type: ignore[call-arg]
    savemat(path, {"y": rng.standard_normal((5, 20_000)) * 10e-6, "fs": 24414.0625})
    with pytest.raises(ValueError, match="expected 5 channels"):
        loaders.load_old_signal(path, "x", "F", units="V")


def test_an_old_cohort_id_ending_in_notched_is_its_own_file_name(tmp_path) -> None:
    p = loaders.old_signal_path(tmp_path, "a/b", "M10_JEL_M10_stim_rec_1144_notched")
    assert p == tmp_path / "a" / "b" / "M10_JEL_M10_stim_rec_1144_notched.mat"
    with pytest.raises(ValueError, match="relative"):
        loaders.old_signal_path(tmp_path, "/abs", "x")
    with pytest.raises(ValueError, match="relative"):
        loaders.old_signal_path(tmp_path, "a/../b", "x")


def test_a_v73_old_cohort_file_is_transposed_not_reshaped(tmp_path) -> None:
    """HDF5 holds MATLAB's (n, 5) as (5, n)."""
    rng = np.random.default_rng(4)
    y_v = rng.standard_normal((5000, 5)) * 10e-6
    path = tmp_path / "x_notched.mat"
    with h5py.File(path, "w", userblock_size=512) as f:
        f["y"] = y_v.T
        f["fs"] = np.array([[24414.0625]])
    header = b"MATLAB 7.3 MAT-file, Platform: x".ljust(116, b" ") + b"\0" * 8 + b"\x00\x02IM"
    with path.open("r+b") as fh:
        fh.write(header)
    rec = loaders.load_old_signal(path, "x", "F", units="V")
    assert rec.data == pytest.approx(y_v * 1e6)


def test_survivals_root_is_explicit_then_env_then_user_config_never_guessed(
    tmp_path, monkeypatch
) -> None:
    cfg = tmp_path / "cfg" / "config.toml"
    monkeypatch.setattr(loaders, "config_path", lambda: cfg)
    monkeypatch.delenv(loaders.SURVIVALS_ENV, raising=False)
    assert loaders.resolve_survivals_root() is None  # nothing configured: no guess
    cfg.parent.mkdir()
    cfg.write_text('gems_root = "x"\nsurvivals_root = "' + (tmp_path / "cfgroot").as_posix()
                   + '"\n', encoding="utf-8")
    assert loaders.resolve_survivals_root() == tmp_path / "cfgroot"
    monkeypatch.setenv(loaders.SURVIVALS_ENV, str(tmp_path / "env"))
    assert loaders.resolve_survivals_root() == tmp_path / "env"
    assert loaders.resolve_survivals_root(tmp_path / "x") == tmp_path / "x"
    assert not hasattr(loaders, "DEFAULT_SURVIVALS_ROOT")


def test_keys_after_a_failed_load_do_not_label_the_unseen_core(qapp, store, tmp_path) -> None:
    """A recording that fails to load leaves its core off screen; keys must not judge it."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from ui.windows.adjudication_window import AdjudicationWindow

    path = tmp_path / "queue.parquet"
    example_queue().to_parquet(path, index=False)
    session = AdjudicationSession(load_queue(path), store, USER,
                                  journal_dir=tmp_path / "journal", queue_file=path.name,
                                  queue_sha256="0" * 64, app_sha=None)

    def load_fn(row):
        if row["recording"] == "synth_j_rec1":
            raise OSError("Drive file not available offline")
        return _fake_recording(row)

    w = AdjudicationWindow(session, load_fn=load_fn, async_traces=False)
    w.show()
    w.activateWindow()
    qapp.processEvents()
    for _ in range(3):
        w.press("1")  # the three A cores, all on screen
    j_row = w.session.current()
    assert j_row["recording"] == "synth_j_rec1"
    assert "offline" in (w.last_error or "")
    assert not any(b.isEnabled() for b in w._buttons.values())
    for key in "1234":
        assert w.press(key) is None
        w._buttons[key].click()  # disabled: does nothing
        QTest.keyClick(w, getattr(Qt, f"Key_{key}"))
    qapp.processEvents()
    assert w.session.judgement_of(j_row) is None and w.session.pending == 3
    w.skip()  # Space still moves past it
    assert w.session.current()["cohort"] == "old"
    assert all(b.isEnabled() for b in w._buttons.values())
    w.press("2")
    assert w.session.judgement_of(j_row) is None
    assert w.session.counts() == {"motion": 3, "physiology": 1}


def test_new_cohort_rows_load_through_meta_json_and_refuse_excluded(store, monkeypatch) -> None:
    from gems_blanking_v2.io.channel_map import meta_path

    import ui.audit.bridge as bridge_mod

    calls = []

    class Loaded:
        recording = "REC"
        provenance: dict = {}

    def fake_load(path, animal, **kw):
        calls.append((path, animal, kw.get("store")))
        return Loaded()

    monkeypatch.setattr(bridge_mod, "load_new_cohort", fake_load)
    meta = meta_path(store, "A", "synth_a_rec1")
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"source_path": "cohort/x/x_sig.mat"}), encoding="utf-8")
    assert loaders.load_new_row(store, "A", "synth_a_rec1") == "REC"
    assert calls == [(store.root / "cohort" / "x" / "x_sig.mat", "A", store)]
    meta.write_text(json.dumps({"source_path": "c/x.mat", "excluded": {"reason": "q"}}),
                    encoding="utf-8")
    with pytest.raises(RuntimeError, match="excluded"):
        loaders.load_new_row(store, "A", "synth_a_rec1")
    with pytest.raises(FileNotFoundError, match="meta.json"):
        loaders.load_new_row(store, "B", "missing")
