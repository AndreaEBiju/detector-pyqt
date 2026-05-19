"""Tests for ui.data.queue.RecordingQueue.

Pure-Python; no Qt needed. The QueuePanel itself is exercised by
the manual UX checklist + future pytest-qt tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
_DETECTOR_CORE = ROOT / "detector-core"
if _DETECTOR_CORE.exists() and str(_DETECTOR_CORE) not in sys.path:
    sys.path.insert(0, str(_DETECTOR_CORE))

from ui.data.queue import RecordingQueue, QueueItem


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Point DETECTOR_HOME at tmp so the queue lives in a known place."""
    monkeypatch.setenv("DETECTOR_HOME", str(tmp_path))
    monkeypatch.delenv("DETECTOR_ARTIFACTS", raising=False)
    yield tmp_path


def test_new_queue_has_no_items(isolated_home):
    q = RecordingQueue.new()
    assert len(q) == 0
    assert q.queue_id  # non-empty timestamp


def test_add_paths_skips_duplicates(isolated_home, tmp_path):
    q = RecordingQueue.new()
    p1 = tmp_path / "a.h5"; p1.touch()
    p2 = tmp_path / "b.h5"; p2.touch()
    added = q.add_paths([p1, p2, p1])      # duplicate p1
    assert added == 2
    assert len(q) == 2


def test_add_folder_uses_glob(isolated_home, tmp_path):
    (tmp_path / "rec1_clean.h5").touch()
    (tmp_path / "rec1.mat").touch()
    (tmp_path / "rec2_clean.h5").touch()
    q = RecordingQueue.new()
    added = q.add_folder(tmp_path, glob="*_clean.h5")
    assert added == 2


def test_save_load_round_trip(isolated_home, tmp_path):
    p1 = tmp_path / "a.h5"; p1.touch()
    q1 = RecordingQueue.new()
    q1.add_paths([p1])
    q1.mark(p1, "in_progress", notes="hello")
    q1.save()
    q2 = RecordingQueue.load_by_id(q1.queue_id)
    assert q2.queue_id == q1.queue_id
    assert len(q2.items) == 1
    assert q2.items[0].status == "in_progress"
    assert q2.items[0].notes == "hello"


def test_next_pending_skips_other_statuses(isolated_home, tmp_path):
    paths = [tmp_path / f"r{i}.h5" for i in range(3)]
    for p in paths:
        p.touch()
    q = RecordingQueue.new()
    q.add_paths(paths)
    q.mark(paths[0], "done")
    q.mark(paths[1], "in_progress")
    nxt = q.next_pending()
    assert nxt is not None
    assert nxt.path == str(paths[2].resolve())


def test_next_pending_returns_none_when_all_handled(isolated_home, tmp_path):
    p = tmp_path / "only.h5"; p.touch()
    q = RecordingQueue.new()
    q.add_paths([p])
    q.mark(p, "done")
    assert q.next_pending() is None


def test_remove_path(isolated_home, tmp_path):
    p1 = tmp_path / "a.h5"; p1.touch()
    p2 = tmp_path / "b.h5"; p2.touch()
    q = RecordingQueue.new()
    q.add_paths([p1, p2])
    assert q.remove_path(p1) is True
    assert len(q) == 1
    assert q.items[0].path == str(p2.resolve())
    assert q.remove_path(p1) is False  # already gone


def test_counts(isolated_home, tmp_path):
    paths = [tmp_path / f"r{i}.h5" for i in range(5)]
    for p in paths:
        p.touch()
    q = RecordingQueue.new()
    q.add_paths(paths)
    q.mark(paths[0], "done")
    q.mark(paths[1], "done")
    q.mark(paths[2], "skipped")
    counts = q.counts()
    assert counts["done"] == 2
    assert counts["skipped"] == 1
    assert counts["pending"] == 2
    assert counts["in_progress"] == 0


def test_list_all_finds_saved_queues(isolated_home, tmp_path):
    p = tmp_path / "x.h5"; p.touch()
    q1 = RecordingQueue.new(queue_id="testq1")
    q1.add_paths([p])
    q1.save()
    q2 = RecordingQueue.new(queue_id="testq2")
    q2.save()
    ids = [qid for qid, _ in RecordingQueue.list_all()]
    assert "testq1" in ids
    assert "testq2" in ids


def test_save_is_atomic(isolated_home, tmp_path):
    q = RecordingQueue.new(queue_id="atomictest")
    q.save()
    saved = RecordingQueue.queues_dir() / "atomictest.json"
    assert saved.exists()
    assert not saved.with_suffix(".tmp").exists()
