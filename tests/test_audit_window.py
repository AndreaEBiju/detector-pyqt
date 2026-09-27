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


@pytest.fixture
def store(tmp_path: Path) -> GemsStore:
    s = GemsStore.initialise(tmp_path / "gems")
    write_protocol_book(default_protocol_book(), s.root / PROTOCOL_FILENAME)
    return s


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
                                "gems_d_t01_3_2_bl_213219", "gems_d_t01_es1_bl_200359"]):
        # 400 s holds two non-overlapping 120 s spans after the 20 s guards;
        # real recordings are 600-1320 s.
        _block(store, folder, 1_789_527_801.0 + 3600 * i, fs=1000.0, duration_s=400.0)
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

    # 600 s (the shortest real recording): wherever a first 120 s span lands in
    # the 560 s assessable region, a second still fits, so any seed can place five.
    for i, folder in enumerate(["gems_j_t01_ms3_bl_230315", "gems_j_t01_ms1_bl_164532",
                                "gems_d_t01_es1_bl_200359"]):
        _block(store, folder, 1_789_527_801.0 + 3600 * i, fs=1000.0, duration_s=600.0)
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





def _three_recordings(store: GemsStore) -> None:
    for i, folder in enumerate(["gems_j_t01_ms3_bl_230315", "gems_j_t01_ms1_bl_164532",
                                "gems_d_t01_es1_bl_200359"]):
        _block(store, folder, 1_789_527_801.0 + 3600 * i, fs=1000.0, duration_s=600.0)


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

    _three_recordings(store)
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

    _three_recordings(store)
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
