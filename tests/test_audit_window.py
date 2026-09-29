"""The blind audit is launchable end to end, on a real (small) GEMS store.

Done, per the spec: Andrea can open the app, pick an eligible new-cohort
recording, and do a blind span with marks written to the store under the session
key - excluded recordings never offered. This builds exactly that world: a store
with a protocol, two TDT blocks with v7 ``_sig.mat`` files and ``meta.json``, one
of them excluded, and drives the real window offscreen.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import struct
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from gems_blanking_v2.io.chanlabels import channels_from_labels
from gems_blanking_v2.io.channel_map import ChannelMap, save_geometry
from gems_blanking_v2.io.stim_split import (
    PROTOCOL_FILENAME,
    default_protocol_book,
    write_protocol_book,
)
from gems_blanking_v2.io.store import GemsStore
from gems_blanking_v2.io.tdt_block import session_key
from scipy.io import savemat

from ui.audit.controller import model_overlay_count
from ui.audit.session import Phase

FS = 8000.0
DURATION_S = 200.0
LABELS = ("RVN1", "RVN2", "RVN3", "LVN1", "LVN2", "LVN3", "ANT1", "ANT2", "ANT3")
COHORT = "August-September Chronic Recordings"


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _block(store: GemsStore, folder: str, epoch_s: float, *, excluded: bool = False,
           fs: float = FS, duration_s: float = DURATION_S) -> str:
    """Write one TDT block (tsq + v7 _sig.mat) and its meta.json; return its key."""
    d = store.root / COHORT / "09152026" / folder
    d.mkdir(parents=True)
    rec = struct.pack("<iiiHHdqif", 40, 0, 0, 0, 0, 0.0, 0, 0, 0.0)
    first = struct.pack("<iiiHHdqif", 40, 0x201, 0, 0, 0, epoch_s, 0, 0, 0.0)
    (d / f"ME_STIM_Andrea-260824-155430_{folder}.tsq").write_bytes(rec + first)
    rng = np.random.default_rng(7)
    y = (rng.standard_normal((int(duration_s * fs), len(LABELS))) * 10e-6).astype(np.float32)
    sig = d / f"{folder}_sig.mat"
    savemat(sig, {"signal": y, "fs": fs, "chanlabels": np.array(LABELS, dtype=object)},
            do_compression=False)
    key = session_key(sig)
    assert key is not None
    animal = folder.split("_")[1].upper()
    cmap = ChannelMap(animal=animal, channels=channels_from_labels(
        LABELS, rostral_end=None, config="independent"), units="V")
    meta = save_geometry(cmap, key, store, mirror_to_profile=False)
    doc = json.loads(meta.read_text(encoding="utf-8"))
    doc.update({"source_path": store.relpath(sig), "folder_name": folder,
                "acquired_at": "2026-09-16T03:03:21+00:00", "duration_s": duration_s})
    if excluded:
        doc["excluded"] = {"reason": "quality_flag", "flags": ["INCOMPLETE"]}
    meta.write_text(json.dumps(doc), encoding="utf-8", newline="\n")
    return key


@pytest.fixture(autouse=True)
def _no_modal_commit_dialog(monkeypatch) -> None:
    """A test must never open the real commit dialog: offscreen it blocks forever.
    One that reaches it fails loudly instead."""
    from ui.windows.audit_window import AuditWindow

    def refuse(self, text: str) -> bool:
        raise AssertionError(f"the commit dialog opened unexpectedly: {text!r}")

    monkeypatch.setattr(AuditWindow, "_confirm_merged", refuse)


@pytest.fixture
def store(tmp_path: Path) -> GemsStore:
    s = GemsStore.initialise(tmp_path / "gems")
    write_protocol_book(default_protocol_book(), s.root / PROTOCOL_FILENAME)
    _within_budget(s)
    return s


def _within_budget(s: GemsStore) -> None:
    """A budget measurement for the generator and reveal under test, within budget."""
    from gems_blanking_v2.detect.recall import (
        BUDGET_MIN_RECORDINGS,
        budget_record,
        write_budget,
    )

    from ui.audit import bridge

    rows = [{"animal": "J", "condition": "baseline", "session": f"s{i}", "folder": f"f{i}",
             "candidates": 500, "assessable_s": 560.0, "covered_s": 30.0}
            for i in range(BUDGET_MIN_RECORDINGS)]
    write_budget(s, budget_record(rows, reveal_sha=bridge.reveal_sha(), sample_rule="test"))


def test_a_blind_span_is_marked_committed_to_the_store_and_revealed(
    qapp, store: GemsStore
) -> None:
    from ui.windows.audit_window import AuditWindow

    key = _block(store, "gems_j_t01_ms3_bl_230315", 1_789_527_801.0)
    _block(store, "gems_j_t02_ms1_bl_171757", 1_789_600_000.0, excluded=True)
    written_before_reveal: list[bool] = []

    def fake_reveal(recording, region):
        # The marks must already be on disk when the reveal is computed.
        written_before_reveal.append(any(store.audit_dir("J", key).glob("*_blind_marks.json")))
        return np.array([[region[0] + 50.0, region[0] + 51.0]]), []

    w = AuditWindow(store, reveal_fn=fake_reveal)
    assert [r.folder_name for r in w.offered] == ["gems_j_t01_ms3_bl_230315"]

    session = w.open_recording(w.offered[0], seed=1234)
    assert session.phase is Phase.BLIND
    assert model_overlay_count(w.viewer) == 0
    w._on_mark(session.start_s + 10.0, session.start_s + 12.0)
    w._on_mark(session.stop_s + 5.0, session.stop_s + 6.0)  # outside: refused
    assert len(session.marks) == 1
    assert model_overlay_count(w.viewer) == 0

    path = w.commit()
    assert path is not None
    assert path.parent == store.audit_dir("J", key)
    assert written_before_reveal == [True]
    marks = json.loads(path.read_text(encoding="utf-8"))
    assert marks["recording_id"] == key and len(marks["marks"]) == 1
    plan = json.loads((path.parent / f"{session.span_id}_plan.json").read_text(encoding="utf-8"))
    assert plan["seed"] == 1234 and plan["session"] == key
    assert plan["stim_epoch_s"] is None and plan["stim_epoch_method"] == "not_stim_recovery"
    assert session.phase is Phase.REVEALED
    assert model_overlay_count(w.viewer) > 0


def test_an_excluded_recording_is_refused_even_if_asked_for_directly(
    qapp, store: GemsStore
) -> None:
    from gems_blanking_v2.io.audit_pool import EligibleRecording

    from ui.windows.audit_window import AuditWindow

    key = _block(store, "gems_j_t02_ms1_bl_171757", 1_789_600_000.0, excluded=True)
    w = AuditWindow(store, reveal_fn=lambda r, g: (np.zeros((0, 2)), []))
    assert w.offered == []
    source = store.root / COHORT / "09152026" / "gems_j_t02_ms1_bl_171757" / \
        "gems_j_t02_ms1_bl_171757_sig.mat"
    rec = EligibleRecording("J", key, "gems_j_t02_ms1_bl_171757", source, "baseline", "")
    with pytest.raises(RuntimeError, match="excluded"):
        w.open_recording(rec, seed=1)


def test_the_five_span_plan_is_recorded_up_front_advanced_and_resumed(
    qapp, store: GemsStore
) -> None:
    """Span k of 5; the plan and its seed written once, before any span is shown."""
    from ui.windows.audit_window import AuditWindow

    for i, folder in enumerate(["gems_j_t01_ms3_bl_230315", "gems_j_t01_ms1_bl_164532",
                                "gems_d_t01_3_2_bl_213219", "gems_d_t01_es1_bl_200359",
                                "gems_j_t02_ms1_sr_164532", "gems_d_t02_es1_sr_210933"]):
        # Both conditions, as the real pool has: an empty pool asks for baseline,
        # stim_recovery, baseline, ... (ruling 2026-09-29).
        _block(store, folder, 1_789_527_801.0 + 3600 * i, fs=1000.0,
               duration_s=900.0 if "_sr_" in folder else 400.0)
    w = AuditWindow(store, reveal_fn=lambda r, g: (np.zeros((0, 2)), []))
    assert w.plan is None
    plan = w.create_plan(seed=99)
    on_disk = json.loads(store.audit_plan_path(plan["plan_id"]).read_text(encoding="utf-8"))
    assert on_disk["seed"] == 99 and on_disk["n_spans"] == 5
    assert {s["animal"] for s in on_disk["spans"]} == {"D", "J"}

    first = w.open_next_span()
    assert first is not None
    assert "span 1 of 5" in w._status.text().lower()
    assert first.span_id == f"{plan['plan_id']}_s1"
    w._on_mark(first.start_s + 5.0, first.start_s + 6.0)
    w.commit()
    assert "1 of 5 spans committed" in w.progress.text()

    # A trial span is recorded as trial and does not advance the plan.
    trial = w.open_recording(w.offered[0], seed=5)
    tpath = w.commit()
    record = json.loads((tpath.parent / f"{trial.span_id}_plan.json").read_text(encoding="utf-8"))
    assert record["trial"] is True
    assert trial.span_id.startswith("trial_") and plan["plan_id"] not in trial.span_id
    assert "plan_id" not in record
    assert "1 of 5 spans committed" in w.progress.text()

    # A fresh window resumes the open plan at span 2.
    w2 = AuditWindow(store, reveal_fn=lambda r, g: (np.zeros((0, 2)), []))
    assert w2.plan is not None and w2.plan["plan_id"] == plan["plan_id"]
    second = w2.open_next_span()
    assert second is not None and second.span_id == f"{plan['plan_id']}_s2"
    assert "span 2 of 5" in w2._status.text().lower()
    with pytest.raises(RuntimeError, match="still open"):
        w2.create_plan(seed=1)


def test_the_buttons_draw_a_random_seed_and_advance_the_plan(qapp, store: GemsStore) -> None:
    """Driven through the real buttons, as Andrea will: ``clicked(bool)`` must not
    become the seed (it did - every plan was seed 0)."""
    from ui.windows.audit_window import AuditWindow

    # 600 s (the shortest real recording): wherever a first 120 s span lands in the
    # 560 s assessable region a second still fits, but a THIRD may not - and with two
    # animals the round-robin gives the first animal three of the five spans. So each
    # animal has two recordings, and no recording ever needs three (1.2% of seeds
    # failed with one D recording; 0 of 3000 with two).
    _small_pool(store)
    seeds = []
    for _ in range(3):
        w = AuditWindow(store, reveal_fn=lambda r, g: (np.zeros((0, 2)), []))
        w._plan_btn.click()
        assert w.last_error is None and w.plan is not None
        seeds.append(w.plan["seed"])
        for path in store.audit_plan_path("x").parent.glob("plan_*.json"):
            path.unlink()  # so the next window may draw a fresh plan
    assert 0 not in seeds and len(set(seeds)) == 3

    w = AuditWindow(store, reveal_fn=lambda r, g: (np.zeros((0, 2)), []))
    w._plan_btn.click()
    w._next_btn.click()
    assert "span 1 of 5" in w._status.text().lower()
    w._commit_btn.click()
    assert "1 of 5 spans committed" in w.progress.text()


def test_an_empty_store_says_why_instead_of_failing_silently(qapp, store: GemsStore) -> None:
    from ui.windows.audit_window import AuditWindow

    w = AuditWindow(store, reveal_fn=lambda r, g: (np.zeros((0, 2)), []))
    assert w.offered == []
    assert "0 eligible recordings" in w._status.text()
    w._plan_btn.click()
    assert w.plan is None
    assert w.last_error is not None and "Could not do that" in w._status.text()





def _small_pool(store: GemsStore) -> None:
    """Two animals, each with two 600 s baselines and one 900 s stim/recovery.

    Both conditions, as the real pool has (ruling 2026-09-29: the planner asks for
    them). With two animals and alternating conditions one animal takes every
    baseline span - three - so each needs two baseline recordings: one recording
    asked for three spans can run out of room (1.2% of seeds).
    """
    for i, folder in enumerate(["gems_j_t01_ms3_bl_230315", "gems_j_t01_ms1_bl_164532",
                                "gems_d_t01_es1_bl_200359", "gems_d_t02_es1_bl_210933",
                                "gems_j_t02_ms1_sr_164532", "gems_d_t02_es2_sr_222933"]):
        _block(store, folder, 1_789_527_801.0 + 3600 * i, fs=1000.0,
               duration_s=900.0 if "_sr_" in folder else 600.0)


def _label_round(store: GemsStore, reveal_fn) -> str:
    from ui.windows.audit_window import AuditWindow

    w = AuditWindow(store, reveal_fn=reveal_fn)
    w._plan_btn.click()
    assert w.last_error is None, w.last_error
    for _ in range(5):
        s = w.open_next_span()
        w._on_mark(s.start_s + 10.0, s.start_s + 11.0)
        w.commit()
    assert "all 5 spans committed" in w.progress.text()
    return w.plan["plan_id"]


def test_a_second_round_waits_for_the_first_to_be_scored_and_clean(
    qapp, store: GemsStore, monkeypatch
) -> None:
    """The sequential rule, through the real window: no new round until the last is
    scored clean. The window and the scorer share ONE reveal, so the digests the
    window wrote verify against the scorer's recompute."""
    from gems_blanking_v2.detect.recall import (
        candidate_digest,
        load_round,
        score_stored_round,
    )

    from ui.audit import bridge, score
    from ui.windows.audit_window import AuditWindow

    def covering(region):  # every mark covered
        return np.array([[region[0], region[1]]]), []

    _small_pool(store)
    plan_id = _label_round(store, lambda r, g: covering(g))
    spans = load_round(store, plan_id)[1]
    for sp in spans:  # the window digested exactly what it revealed
        region = tuple(sp["record"]["assessable_regions_s"][0])
        assert sp["record"]["reveal"]["candidates_sha256"] == candidate_digest(
            covering(region)[0])
    w2 = AuditWindow(store, reveal_fn=lambda r, g: covering(g))
    w2._plan_btn.click()
    assert w2.plan is None and "has not been scored" in (w2.last_error or "")

    seen = []
    monkeypatch.setattr(bridge, "reveal_for_region",
                        lambda rec, region: (seen.append(region), covering(region))[1])
    result = score_stored_round(store, plan_id, score.make_reveal(store))
    assert seen == [tuple(sp["record"]["assessable_regions_s"][0]) for sp in spans]
    assert result.warnings == [] and result.found == 5 and result.covered == 5
    assert set(result.provenance["reveal_digests"].values()) == {"verified"}
    w3 = AuditWindow(store, reveal_fn=lambda r, g: covering(g))
    w3._plan_btn.click()
    assert w3.last_error is None and w3.plan is not None and w3.plan["plan_id"] != plan_id


def test_a_recompute_that_differs_from_the_reveal_is_refused_and_a_legacy_span_warns(
    qapp, store: GemsStore, monkeypatch
) -> None:
    from gems_blanking_v2.detect.recall import RevealMismatchError, score_stored_round

    from ui.audit import bridge, score

    def shown(region):
        return np.array([[region[0] + 1.0, region[0] + 1.5]]), []

    _small_pool(store)
    plan_id = _label_round(store, lambda r, g: shown(g))
    monkeypatch.setattr(bridge, "reveal_for_region",  # a different generator
                        lambda rec, region: (np.array([[region[0] + 2.0, region[0] + 2.5]]), []))
    with pytest.raises(RevealMismatchError, match="do not match"):
        score_stored_round(store, plan_id, score.make_reveal(store))
    assert not store.audit_score_path(plan_id).exists()

    # The same round as an app started before digests would have left it.
    for rec_path in store.root.glob(f"labels/*/blind_audit/*/{plan_id}_s*_plan.json"):
        doc = json.loads(rec_path.read_text(encoding="utf-8"))
        doc.pop("reveal")
        rec_path.write_text(json.dumps(doc), encoding="utf-8")
    legacy = score_stored_round(store, plan_id, score.make_reveal(store))
    assert len(legacy.warnings) == 5
    assert all("without candidate digests" in w for w in legacy.warnings)


def _real_shift_drag(qapp, w, x0: float, x1: float) -> None:
    """A Shift+drag through Qt mouse events on channel 0, from x0 to x1 seconds."""
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


def test_scrolling_off_the_span_is_shaded_refused_clearly_and_one_click_back(
    qapp, store: GemsStore
) -> None:
    """Andrea scrolled to the recording's start, marked at 17.8 s, and was refused
    with no sign of where the span was. Scrolling stays free; the span is visible,
    the refusal says both ranges, and Back to span / Home returns to it."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from ui.windows.audit_window import AuditWindow

    _small_pool(store)
    w = AuditWindow(store, reveal_fn=lambda r, g: (np.zeros((0, 2)), []))
    w.resize(1500, 900)
    w.show()
    qapp.processEvents()
    w._plan_btn.click()
    w._next_btn.click()
    qapp.processEvents()
    s = w._session
    assert w.viewer.time_range == (s.start_s, s.stop_s)

    # the outside of the span is shaded on every channel
    shaded = [tuple(i.getRegion()) for i in w._outside_items]
    assert (0.0, s.start_s) in shaded and len(w._outside_items) == 2 * len(w.viewer.plots)

    # scroll to the recording's start and mark there: refused, and it says why
    w.viewer.set_viewport(0.0, 60.0)
    qapp.processEvents()
    _real_shift_drag(qapp, w, 17.8, 18.0)
    assert s.marks == []
    msg = w._status.text()
    assert "17.8-18.0 s" in msg and f"{s.start_s:.1f}-{s.stop_s:.1f} s" in msg
    assert "Back to span" in msg

    # one click back, and a real mark inside the span is recorded
    w._span_btn.click()
    assert w.viewer.time_range == (s.start_s, s.stop_s)
    _real_shift_drag(qapp, w, s.start_s + 10.0, s.start_s + 12.0)
    assert len(s.marks) == 1 and s.start_s + 9.9 < s.marks[0].start_s < s.start_s + 10.1

    # Home does the same
    w.viewer.set_viewport(0.0, 60.0)
    QTest.keyClick(w, Qt.Key_Home)
    qapp.processEvents()
    assert w.viewer.time_range == (s.start_s, s.stop_s)


def test_an_animal_excluded_mid_round_is_replaced_without_touching_committed_spans(
    qapp, store: GemsStore
) -> None:
    """Round 1 drew two spans from animal D; D was then excluded. The D spans are
    replaced - same condition, a recording not already in the plan - while the
    committed span stays exactly as labelled, and the plan records the change."""
    from ui.windows.audit_window import AuditWindow

    keys = {}
    for i, folder in enumerate(["gems_j_t01_ms3_bl_230315", "gems_j_t01_ms1_bl_164532",
                                "gems_d_t01_es1_bl_200359", "gems_d_t02_es1_bl_210933",
                                "gems_b_t01_es2_bl_214117", "gems_b_t01_ms1_bl_224934",
                                # enough non-D recordings that replacements exist
                                # outside the plan (the real pool has 458)
                                "gems_j_t02_es1_bl_233531", "gems_b_t02_ms2_bl_180847",
                                "gems_j_t02_ms3_bl_203835", "gems_b_t03_ms3_bl_184100",
                                "gems_j_t03_ms1_sr_164532", "gems_b_t03_ms1_sr_194532",
                                "gems_j_t03_ms2_sr_174532", "gems_b_t03_ms2_sr_204532",
                                "gems_d_t03_ms1_sr_184532"]):
        keys[folder] = _block(store, folder, 1_789_527_801.0 + 3600 * i, fs=1000.0,
                              duration_s=600.0)
    for seed in range(200):  # a plan whose span 1 is not D, with D later in it
        for p in store.audit_plan_path("x").parent.glob("plan_*.json"):
            p.unlink()
        w = AuditWindow(store, reveal_fn=lambda r, g: (np.zeros((0, 2)), []))
        plan = w.create_plan(seed=seed)
        animals = [sp["animal"] for sp in plan["spans"]]
        if animals[0] != "D" and "D" in animals[1:]:
            break
    first = w.open_next_span()
    w._on_mark(first.start_s + 5.0, first.start_s + 6.0)
    w.commit()
    committed_span = dict(plan["spans"][0])

    for folder, key in keys.items():  # exclude animal D, as the generator does
        if folder.startswith("gems_d_"):
            meta = store.root / "data" / "D" / key / "meta.json"
            doc = json.loads(meta.read_text(encoding="utf-8"))
            doc["excluded"] = {"reason": "animal_excluded", "animal": "D"}
            meta.write_text(json.dumps(doc), encoding="utf-8", newline="\n")

    w2 = AuditWindow(store, reveal_fn=lambda r, g: (np.zeros((0, 2)), []))
    expected = [k for k, a in enumerate(animals) if a == "D"]
    assert w2.ineligible_spans() == expected
    replaced = w2.replace_ineligible_spans("animal D excluded (test)", seed=7)
    assert replaced == expected
    spans = w2.plan["spans"]
    assert spans[0] == committed_span  # the committed span is untouched
    assert all(sp["animal"] != "D" for sp in spans)
    for k in expected:
        assert spans[k]["condition"] == plan["spans"][k]["condition"]
    assert len({sp["recording_id"] for sp in spans}) >= len({sp["recording_id"] for sp in
                                                              plan["spans"]}) - len(expected)
    assert w2.ineligible_spans() == []
    on_disk = json.loads(store.audit_plan_path(plan["plan_id"]).read_text(encoding="utf-8"))
    amend = on_disk["amendments"][0]
    assert amend["seed"] == 7 and amend["reason"] == "animal D excluded (test)"
    assert [r["span"] for r in amend["replaced"]] == [k + 1 for k in expected]
    assert all(r["old"]["animal"] == "D" for r in amend["replaced"])

    # the plan carries on: the next span opens, from an eligible recording
    nxt = w2.open_next_span()
    assert nxt is not None and "span 2 of 5" in w2._status.text().lower()



def test_no_plan_is_drawn_until_the_generator_is_measured_within_budget(
    qapp, tmp_path: Path
) -> None:
    """Task 09's budget, enforced at the button: round 1 cleared recall by coverage."""
    from gems_blanking_v2.detect.recall import budget_record, write_budget

    from ui.audit import bridge
    from ui.windows.audit_window import AuditWindow

    s = GemsStore.initialise(tmp_path / "gems")
    write_protocol_book(default_protocol_book(), s.root / PROTOCOL_FILENAME)
    _small_pool(s)
    w = AuditWindow(s, reveal_fn=lambda r, g: (np.zeros((0, 2)), []))
    w._plan_btn.click()
    assert w.plan is None and "has not been measured" in (w.last_error or "")

    rows = [{"animal": "J", "condition": "baseline", "session": f"s{i}", "folder": f"f{i}",
             "candidates": 13_000, "assessable_s": 1200.0, "covered_s": 1150.0}
            for i in range(30)]
    write_budget(s, budget_record(rows, reveal_sha=bridge.reveal_sha(), sample_rule="test"))
    w._plan_btn.click()
    assert w.plan is None and "exceeds the candidate budget" in (w.last_error or "")

    _within_budget(s)
    w.last_error = None
    w._plan_btn.click()
    assert w.last_error is None and w.plan is not None


def test_the_budget_is_measured_as_the_window_reveals(qapp, store: GemsStore, monkeypatch) -> None:
    from ui.audit import bridge, budget

    _small_pool(store)
    calls = []

    def fake(recording, region):
        calls.append(region)
        lo = region[0]
        return np.array([[lo + 10.0, lo + 20.0], [lo + 15.0, lo + 30.0]]), []

    monkeypatch.setattr(bridge, "reveal_for_region", fake)
    rec = budget.measure_budget(store, per_cell=5, seed=1, progress=lambda m: None)
    assert rec["n_recordings"] == 6 and len(calls) == 6
    row = rec["recordings"][0]
    assert row["candidates"] == 2 and row["covered_s"] == pytest.approx(20.0)  # union 10-30
    assert rec["within_budget"] is False  # 6 recordings < the minimum: not a measurement


def test_parallel_budget_rows_equal_serial_rows_in_sample_order(
    qapp, store: GemsStore, monkeypatch
) -> None:
    """Workers change the wall clock, never the record - order included."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from ui.audit import bridge, budget

    _small_pool(store)

    calls: list[str] = []
    lock = threading.Lock()

    def fake(recording, region):
        with lock:
            first = not calls
            calls.append(str(recording.path))
        time.sleep(0.3 if first else 0.0)  # the first submitted finishes last
        n = 1 + ("ms3" in str(recording.path))  # differs per recording, so order shows
        lo = region[0]
        return np.array([[lo + 1.0 + i, lo + 1.5 + i] for i in range(n)]), []

    monkeypatch.setattr(bridge, "reveal_for_region", fake)
    serial = budget.measure_budget(store, per_cell=5, seed=1, progress=lambda m: None)
    parallel = budget.measure_budget(store, per_cell=5, seed=1, progress=lambda m: None,
                                     workers=3, executor=ThreadPoolExecutor)
    assert parallel["recordings"] == serial["recordings"]
    assert len({r["candidates"] for r in serial["recordings"]}) > 1


def _open_span(store: GemsStore, **kw):
    from ui.windows.audit_window import AuditWindow

    _small_pool(store)
    w = AuditWindow(store, reveal_fn=lambda r, g: (np.zeros((0, 2)), []), **kw)
    w.create_plan(seed=7)
    return w, w.open_next_span()


def test_overlapping_marks_are_shown_merged_and_cancel_keeps_the_span_blind(
    qapp, store: GemsStore
) -> None:
    """Andrea, 2026-09-28: overlapping or touching marks are one artifact. The window
    shows that, with a count, before she confirms; the file keeps the marks as drawn."""
    from ui.audit.session import Phase

    asked: list[str] = []
    answer = [False]
    w, s = _open_span(store, confirm_commit=lambda text: (asked.append(text), answer[0])[1])
    t0 = s.start_s
    w._on_mark(t0 + 10.0, t0 + 11.0)
    w._on_mark(t0 + 10.8, t0 + 11.5)  # overlaps the first
    w._on_mark(t0 + 11.55, t0 + 12.0)  # 50 ms after: kept separate, but named

    assert w.commit() is None and w._session.phase is Phase.BLIND
    assert not list(store.audit_dir(w._rec.animal, w._rec.session).glob("*_blind_marks.json"))
    assert asked[0].startswith("3 marks -> 2 artifacts.")
    assert "marks 1 + 2 overlap or touch" in asked[0] and "50 ms apart" in asked[0]

    answer[0] = True
    path = w.commit()
    assert path is not None and len(asked) == 2
    saved = json.loads(path.read_text(encoding="utf-8"))["marks"]
    assert len(saved) == 3  # as drawn: the merge is a scoring step, not an edit


def test_marks_that_do_not_touch_commit_without_asking(qapp, store: GemsStore) -> None:
    def refuse(text: str) -> bool:
        raise AssertionError("asked although nothing merges")

    w, s = _open_span(store, confirm_commit=refuse)
    w._on_mark(s.start_s + 10.0, s.start_s + 11.0)
    w._on_mark(s.start_s + 12.0, s.start_s + 13.0)
    assert w.commit() is not None
    assert w.last_merge_summary is not None and w.last_merge_summary.startswith(
        "2 marks -> 2 artifacts.")


def test_the_window_lets_a_mark_run_past_the_viewport(qapp, store: GemsStore) -> None:
    w, _s = _open_span(store)
    assert w.viewer is not None and w.viewer.follow_mark_drag is True


def test_a_new_plan_declares_the_merged_scoring_unit(qapp, store: GemsStore) -> None:
    """Declared before labelling, so the scorer reads the unit from the plan."""
    w, _s = _open_span(store)
    on_disk = json.loads(store.audit_plan_path(w.plan["plan_id"]).read_text(encoding="utf-8"))
    assert on_disk["scoring_unit"] == "merged"


def test_the_plan_draws_the_conditions_the_eligible_pool_needs(
    qapp, store: GemsStore, monkeypatch
) -> None:
    """Ruling 2026-09-29: round 3 was all stim/recovery, so round 4 is all baseline."""
    from gems_blanking_v2.detect import recall

    w, _s = _open_span(store)  # empty eligible pool: baseline first, then alternate
    first = json.loads(store.audit_plan_path(w.plan["plan_id"]).read_text(encoding="utf-8"))
    assert [sp["condition"] for sp in first["spans"]] == \
        ["baseline", "stim_recovery", "baseline", "stim_recovery", "baseline"]
    assert first["conditions_requested"] == [sp["condition"] for sp in first["spans"]]
    assert "never scores" in first["condition_rule"]

    from ui.windows.audit_window import AuditWindow

    asked: list[int] = []

    def five_baselines(st: GemsStore, n: int) -> tuple[str, ...]:
        asked.append(n)
        return ("baseline",) * n

    monkeypatch.setattr(recall, "condition_plan", five_baselines)
    for p in store.audit_plan_path("x").parent.glob("plan_*.json"):
        p.unlink()
    w2 = AuditWindow(store, reveal_fn=lambda r, g: (np.zeros((0, 2)), []))
    plan = w2.create_plan(seed=11)
    assert asked == [5] and {sp["condition"] for sp in plan["spans"]} == {"baseline"}
