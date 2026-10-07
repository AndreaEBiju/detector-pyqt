"""Candidate adjudication (task 16 Change 1): queue schema, key map, shards, session.

Hermetic: every store is a fresh ``GemsStore`` under ``tmp_path`` and every journal is
under ``tmp_path``; nothing reads the real store, the Drive or per-user config.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gems_blanking_v2.io.store import GemsStore
from gems_blanking_v2.model.labels import (
    LABEL_COLUMNS,
    NEGATIVE_JUDGEMENTS,
    training_rows,
)

from ui.adjudicate import judgements as jd
from ui.adjudicate.queue import (
    REQUIRED_COLUMNS,
    QueueSchemaError,
    check_queue,
    core_key,
    example_queue,
    load_queue,
    presentation_order,
)
from ui.adjudicate.session import AdjudicationSession

USER = "tester"


@pytest.fixture
def store(tmp_path: Path) -> GemsStore:
    return GemsStore.initialise(tmp_path / "gems")


def _queue(tmp_path: Path, df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, Path]:
    path = tmp_path / "queue.parquet"
    (example_queue() if df is None else df).to_parquet(path, index=False)
    return load_queue(path), path


def _session(store: GemsStore, tmp_path: Path, df: pd.DataFrame | None = None,
             **kw) -> AdjudicationSession:
    q, path = _queue(tmp_path, df)
    clock_t = [datetime(2026, 10, 8, 14, 0, tzinfo=UTC)]

    def clock() -> datetime:
        clock_t[0] += timedelta(seconds=1)
        return clock_t[0]

    return AdjudicationSession(q, store, USER, journal_dir=tmp_path / "journal",
                               queue_file=path.name, queue_sha256="ab" * 32,
                               app_sha="c0ffee", clock=clock, **kw)


def _shards(store: GemsStore) -> pd.DataFrame:
    files = sorted((store.root / "labels").rglob("events_*.parquet"))
    if not files:
        return pd.DataFrame(columns=list(jd.SHARD_COLUMNS))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


# ---------------------------------------------------------------------------
# the queue schema
# ---------------------------------------------------------------------------


def test_the_example_queue_satisfies_the_schema() -> None:
    q = check_queue(example_queue())
    assert q["core_key"].is_unique
    assert set(q["cohort"]) == {"new", "old"}
    assert set(q["draw"]) == {"random", "uncertainty"}
    assert set(q["label_set"]) == {"train", "test"}


@pytest.mark.parametrize("column", REQUIRED_COLUMNS)
def test_a_missing_required_column_raises_naming_it(column: str) -> None:
    with pytest.raises(QueueSchemaError, match=column):
        check_queue(example_queue().drop(columns=[column]))


def test_a_null_in_a_required_column_raises_naming_it() -> None:
    df = example_queue()
    df.loc[1, "draw"] = None
    with pytest.raises(QueueSchemaError, match="'draw' is null"):
        check_queue(df)


@pytest.mark.parametrize(("column", "value"), [("cohort", "older"), ("draw", "dense"),
                                               ("label_set", "eval")])
def test_an_unknown_enumeration_value_raises(column: str, value: str) -> None:
    df = example_queue()
    df.loc[0, column] = value
    with pytest.raises(QueueSchemaError, match=column):
        check_queue(df)


def test_a_core_outside_its_region_or_reversed_raises() -> None:
    df = example_queue()
    df.loc[0, "region_stop_s"] = 20.05
    with pytest.raises(QueueSchemaError, match="region_start_s <= start_s"):
        check_queue(df)
    df = example_queue()
    df.loc[0, "stop_s"] = df.loc[0, "start_s"]
    with pytest.raises(QueueSchemaError, match="start_s < stop_s"):
        check_queue(df)


def test_old_rows_need_a_relative_posix_folder() -> None:
    with pytest.raises(QueueSchemaError, match="folder"):
        check_queue(example_queue().drop(columns=["folder"]))
    for bad in ("G:/Shared drives/x", "/abs/x", "a/../b", "a\\b"):
        df = example_queue()
        df.loc[df["cohort"] == "old", "folder"] = bad
        with pytest.raises(QueueSchemaError, match="folder"):
            check_queue(df)


def test_a_new_cohort_test_animal_marked_train_is_refused() -> None:
    """R1: I, J and K labels never enter training - a queue saying so is wrong."""
    df = example_queue()
    df.loc[df["animal"] == "J", "label_set"] = "train"
    with pytest.raises(QueueSchemaError, match="R1"):
        check_queue(df)


def test_duplicate_core_keys_are_refused() -> None:
    df = pd.concat([example_queue(), example_queue().iloc[[0]]], ignore_index=True)
    with pytest.raises(QueueSchemaError, match="share a core key"):
        check_queue(df)


def test_presentation_groups_by_recording_in_first_appearance_order() -> None:
    q = presentation_order(check_queue(example_queue()))
    assert list(q["recording"]) == ["synth_a_rec1"] * 3 + ["synth_j_rec1", "synth_old_1"]
    assert list(q.loc[q["recording"] == "synth_a_rec1", "start_s"]) == [20.0, 55.5, 90.0]


def test_core_key_does_not_depend_on_float_formatting() -> None:
    assert core_key("r", 5.0, 5.1) == core_key("r", 5, 5.1000000001)
    assert core_key("r", 0.1 + 0.2, 1.0) == core_key("r", 0.3, 1.0)
    assert core_key("r", 5.0, 5.1) != core_key("r", 5.00001, 5.1)


# ---------------------------------------------------------------------------
# keys
# ---------------------------------------------------------------------------


def test_the_key_map_is_exactly_one_to_four() -> None:
    assert dict(jd.KEY_TO_JUDGEMENT) == {"1": "motion", "2": "physiology",
                                        "3": "unsure", "4": "line_noise"}


@pytest.mark.parametrize(("key", "expected"), [
    ("1", "motion"), ("2", "physiology"), ("3", "unsure"), ("4", "line_noise"),
    ("0", None), ("5", None), ("u", None), ("", None), (" 4", "line_noise"),
])
def test_each_key_maps_to_its_judgement(key: str, expected: str | None) -> None:
    assert jd.judgement_for_key(key) == expected


def test_line_noise_is_a_negative_for_the_motion_classifier() -> None:
    """Ruling (c) item 3: key 4 counts as a negative, not as motion and not dropped."""
    assert jd.judgement_for_key("4") in NEGATIVE_JUDGEMENTS
    assert jd.judgement_for_key("2") in NEGATIVE_JUDGEMENTS
    assert jd.judgement_for_key("1") not in NEGATIVE_JUDGEMENTS


# ---------------------------------------------------------------------------
# the session and the shards
# ---------------------------------------------------------------------------


def test_each_key_writes_its_judgement_with_full_provenance(store, tmp_path) -> None:
    s = _session(store, tmp_path)
    for key in "1234":
        s.judge_key(key)
    s.close()
    out = _shards(store)
    assert list(out["judgement"]) == ["motion", "physiology", "unsure", "line_noise"]
    assert set(LABEL_COLUMNS) <= set(out.columns)
    assert set(out["source"]) == {"human"} and set(out["label_source"]) == {"human"}
    assert set(out["basis"]) == {"adjudicated"}
    assert set(out["by"]) == {USER} and set(out["app_commit"]) == {"c0ffee"}
    assert set(out["queue_sha256"]) == {"ab" * 32}
    assert out["judgement_id"].is_unique
    for at in out["at"]:
        parsed = datetime.fromisoformat(at)
        assert parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0)
    q = s.rows
    assert list(out["draw"]) == [r["draw"] for r in q[:4]]
    assert list(out["start_s"]) == [r["start_s"] for r in q[:4]]


def test_a_non_judging_key_records_nothing(store, tmp_path) -> None:
    s = _session(store, tmp_path)
    first = s.current()
    assert s.judge_key("9") is None and s.judge_key("u") is None
    assert s.current() is first and s.pending == 0


def test_skip_leaves_a_core_unjudged_and_it_comes_round_again(store, tmp_path) -> None:
    s = _session(store, tmp_path)
    first = s.current()
    s.skip()
    assert s.current() is not first
    for _ in range(len(s.rows) - 1):
        s.judge_key("2")
    assert s.current() is first  # the skipped core comes back
    s.close()
    assert core_key(first["recording"], first["start_s"], first["stop_s"]) not in set(
        jd.read_shards(store, USER, ["A", "J", "F"])["core_key"])
    assert "unjudged" not in set(_shards(store)["judgement"])


def test_undo_returns_to_the_core_and_withdraws_the_judgement(store, tmp_path) -> None:
    s = _session(store, tmp_path)
    first = s.current()
    s.judge_key("1")
    second = s.current()
    s.judge_key("4")
    undone = s.undo()
    assert undone is not None and undone["judgement"] == "line_noise"
    assert s.current() is second and s.judgement_of(second) is None
    s.judge_key("2")
    s.close()
    out = _shards(store)
    assert list(out["judgement"]) == ["motion", "physiology"]
    assert s.judgement_of(first) == "motion"


def test_undo_cannot_reach_a_judgement_already_written(store, tmp_path) -> None:
    s = _session(store, tmp_path)
    s.judge_key("1")
    s.flush(force=True)
    assert s.undo() is None
    assert list(_shards(store)["judgement"]) == ["motion"]


def _big_queue(n: int, animal: str = "A", label_set: str = "train") -> pd.DataFrame:
    t = np.arange(n, dtype=np.float64) * 0.5 + 20.0
    return pd.DataFrame({
        "recording": "synth_big", "animal": animal, "cohort": "new", "start_s": t,
        "stop_s": t + 0.1, "region_start_s": 10.0, "region_stop_s": 1000.0,
        "draw": "random", "label_set": label_set})


def test_batches_keep_the_newest_undo_depth_judgements_unwritten(store, tmp_path) -> None:
    s = _session(store, tmp_path, _big_queue(60), undo_depth=5, flush_batch=10)
    for _ in range(14):
        s.judge_key("1")
    assert s.written == [] and s.pending == 14
    s.judge_key("2")  # 15 = 5 + 10 pending: flush all but the newest 5
    assert len(s.written) == 1 and s.pending == 5
    assert len(_shards(store)) == 10
    for _ in range(5):
        assert s.undo() is not None  # the full undo depth is still reachable
    assert s.undo() is None


def test_shards_are_write_once_and_never_rewritten(store, tmp_path) -> None:
    s = _session(store, tmp_path, _big_queue(60), undo_depth=2, flush_batch=3)
    seen: dict[Path, bytes] = {}
    for _ in range(40):
        s.judge_key("1")
        for p in s.written:
            data = p.read_bytes()
            assert seen.setdefault(p, data) == data, f"{p.name} was rewritten"
    s.close()
    assert len(set(s.written)) == len(s.written)  # every flush wrote a new file
    assert len(_shards(store)) == 40
    with pytest.raises(FileExistsError, match="write-once"):
        jd._write_once(s.written[0], b"x")


def test_a_failed_write_leaves_no_partial_shard(store, tmp_path, monkeypatch) -> None:
    s = _session(store, tmp_path)
    s.judge_key("1")

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError, match="disk full"):
        s.flush(force=True)
    labels = store.root / "labels"
    assert not list(labels.rglob("*.parquet")) and not list(labels.rglob("*.tmp"))
    assert s.pending == 1  # still held, and still in the journal
    monkeypatch.undo()
    s.flush(force=True)
    assert list(_shards(store)["judgement"]) == ["motion"]


def test_unjudged_is_never_written(store) -> None:
    rec = jd.make_record(check_queue(example_queue()).iloc[0].to_dict(), "motion",
                        user=USER, app_sha=None, queue_file="q", queue_sha256="0")
    rec["judgement"] = "unjudged"
    with pytest.raises(ValueError, match="unjudged is never written"):
        jd.write_shards(store, [rec], USER)
    with pytest.raises(ValueError, match="judgement must be one of"):
        jd.make_record(check_queue(example_queue()).iloc[0].to_dict(), "unjudged",
                      user=USER, app_sha=None, queue_file="q", queue_sha256="0")


def test_the_test_set_flag_is_preserved_and_kept_out_of_training(store, tmp_path) -> None:
    """R1: J's rows are stored as label_set test, in their own shard, and never train."""
    s = _session(store, tmp_path)
    while s.current() is not None:
        s.judge_key("1")
    s.close()
    out = _shards(store)
    j_rows = out[out["animal"] == "J"]
    assert len(j_rows) == 1 and set(j_rows["label_set"]) == {"test"}
    for p in s.written:
        shard = pd.read_parquet(p)
        assert shard["label_set"].nunique() == 1 and shard["cohort"].nunique() == 1
        assert f"-{shard['label_set'].iloc[0]}-" in p.name
    kept = training_rows(out, old_tiers={"synth_old_1": "1"})
    assert "J" not in set(kept["animal"]) and set(kept["label_set"]) == {"train"}
    assert len(kept) == len(out) - 1


def test_written_shards_feed_the_task_10_loader(store, tmp_path) -> None:
    """line_noise and physiology train as negatives; unsure is dropped."""
    s = _session(store, tmp_path, _big_queue(4))
    for key in "1243":
        s.judge_key(key)
    s.close()
    kept = training_rows(_shards(store))
    assert sorted(zip(kept["judgement"], kept["y"], strict=True)) == [
        ("line_noise", 0), ("motion", 1), ("physiology", 0)]


def test_shards_go_where_the_store_puts_per_user_label_files(store, tmp_path) -> None:
    s = _session(store, tmp_path)
    s.judge_key("1")
    s.close()
    (path,) = s.written
    assert path.parent == store.labels_path("A", USER).parent
    assert path.name.startswith(f"events_{USER}_") and path.suffix == ".parquet"


# ---------------------------------------------------------------------------
# resuming
# ---------------------------------------------------------------------------


def test_reopening_skips_cores_already_in_the_store(store, tmp_path) -> None:
    s = _session(store, tmp_path)
    s.judge_key("1")
    s.judge_key("2")
    s.close()
    again = _session(store, tmp_path)
    assert again.position() == (2, len(again.rows))
    assert again.current() is again.rows[2]
    assert again.restored == 0


def test_the_journal_restores_judgements_that_never_reached_the_store(store, tmp_path) -> None:
    s = _session(store, tmp_path)
    s.judge_key("1")
    s.judge_key("4")
    s.judge_key("2")
    s.undo()  # the undone one must not come back
    # the app dies here: nothing was flushed
    assert not list((store.root / "labels").rglob("*.parquet"))
    again = _session(store, tmp_path)
    assert again.restored == 2 and again.pending == 2
    assert again.position()[0] == 2
    again.close()
    assert list(_shards(store)["judgement"]) == ["motion", "line_noise"]
    # and once written, the journal entries are not restored a second time
    third = _session(store, tmp_path)
    assert third.restored == 0 and third.position()[0] == 2


def test_a_torn_last_journal_line_is_ignored(store, tmp_path) -> None:
    s = _session(store, tmp_path)
    s.judge_key("1")
    with s.journal_path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write('{"op":"judge","recor')
    assert _session(store, tmp_path).restored == 1


def test_journal_lines_are_canonical_ascii_with_absent_missing_keys(store, tmp_path) -> None:
    df = example_queue().drop(columns=["score"])
    s = _session(store, tmp_path, df)
    s.judge_key("1")
    (line,) = s.journal_path.read_text(encoding="utf-8").splitlines()
    assert line.isascii() and '"score"' not in line and "NaN" not in line
    assert line.startswith('{"animal":')  # sorted keys


def test_progress_is_by_animal_and_cohort(store, tmp_path) -> None:
    s = _session(store, tmp_path)
    s.judge_key("1")
    s.judge_key("3")
    prog = {(p.animal, p.cohort): (p.judged, p.total, p.unjudged) for p in s.progress()}
    assert prog == {("A", "new"): (2, 3, 1), ("J", "new"): (0, 1, 1), ("F", "old"): (0, 1, 1)}
    assert s.counts() == {"motion": 1, "unsure": 1}


# ---------------------------------------------------------------------------
# review fixes (2026-10-07): animal letters, R1 by cohort, alias hash, users, retries
# ---------------------------------------------------------------------------

ALIAS = "f" * 64


def _mixed_j_queue() -> pd.DataFrame:
    """Old-cohort J (JEL, a training rat) beside new-cohort J (a test animal)."""
    return pd.DataFrame({
        "recording": ["old_jel_1", "old_jel_1", "new_j_1"], "animal": ["J", "J", "J"],
        "cohort": ["old", "old", "new"], "start_s": [5.0, 7.0, 30.0],
        "stop_s": [5.1, 7.1, 30.2], "region_start_s": [0.0, 0.0, 10.0],
        "region_stop_s": [60.0, 60.0, 190.0], "draw": "random",
        "label_set": ["train", "train", "test"],
        "folder": ["d/old_jel_1", "d/old_jel_1", None],
        "alias_table_sha256": [ALIAS, ALIAS, None]})


@pytest.mark.parametrize("bad", ["j", "J ", " J", "AB", "", "1"])
def test_an_animal_that_is_not_one_upper_case_letter_is_refused(bad: str) -> None:
    df = example_queue()
    df.loc[2, "animal"] = bad
    with pytest.raises(QueueSchemaError, match="one upper-case letter"):
        check_queue(df)


def test_old_j_trains_and_new_j_tests_in_separate_shards(store, tmp_path) -> None:
    """R1 is new-cohort I/J/K only: old JEL ("J") is a training rat (is_test_animal)."""
    s = _session(store, tmp_path, _mixed_j_queue())
    s.judge_key("1")
    s.judge_key("2")
    s.judge_key("1")
    s.close()
    assert len(s.written) == 2
    by_file = {p.name: pd.read_parquet(p) for p in s.written}
    assert {(d["cohort"].iloc[0], d["label_set"].iloc[0], len(d)) for d in by_file.values()} \
        == {("old", "train", 2), ("new", "test", 1)}
    out = _shards(store)
    kept = training_rows(out, old_tiers={"old_jel_1": "1"})
    assert set(zip(kept["cohort"], kept["animal"], strict=True)) == {("old", "J")}
    assert len(kept) == 2


def test_new_j_marked_train_is_refused_but_old_j_train_is_not() -> None:
    df = _mixed_j_queue()
    check_queue(df)  # old J train + new J test: accepted
    df.loc[2, "label_set"] = "train"
    with pytest.raises(QueueSchemaError, match="R1"):
        check_queue(df)


def test_write_shards_refuses_a_test_animal_not_marked_test(store) -> None:
    rec = jd.make_record(check_queue(_mixed_j_queue()).iloc[2].to_dict(), "motion",
                         user=USER, app_sha=None, queue_file="q", queue_sha256="0")
    rec["label_set"] = "train"
    with pytest.raises(ValueError, match="R1"):
        jd.write_shards(store, [rec], USER)
    assert not list((store.root / "labels").rglob("*.parquet"))


def test_old_rows_need_an_alias_table_hash_and_every_old_label_records_it(store,
                                                                       tmp_path) -> None:
    with pytest.raises(QueueSchemaError, match="alias_table_sha256"):
        check_queue(_mixed_j_queue().drop(columns=["alias_table_sha256"]))
    for bad in (None, "abc", "F" * 64):
        df = _mixed_j_queue()
        df.loc[0, "alias_table_sha256"] = bad
        with pytest.raises(QueueSchemaError, match="alias_table_sha256"):
            check_queue(df)
    s = _session(store, tmp_path, _mixed_j_queue())
    while s.current() is not None:
        s.judge_key("1")
    s.close()
    out = _shards(store)
    assert set(out.loc[out["cohort"] == "old", "alias_table_sha256"]) == {ALIAS}
    assert out.loc[out["cohort"] == "new", "alias_table_sha256"].isna().all()


def test_another_users_shards_do_not_mark_this_users_cores(store, tmp_path) -> None:
    """events_tester_* also matches events_tester_smith_*: rows are filtered by ``by``."""
    q, path = _queue(tmp_path)
    other = AdjudicationSession(q, store, "tester_smith", journal_dir=tmp_path / "j2",
                                queue_file=path.name, queue_sha256="cd" * 32, app_sha=None)
    other.judge_key("1")
    other.judge_key("1")
    other.close()
    mine = _session(store, tmp_path)
    assert mine.position() == (0, len(mine.rows))


def test_a_shard_present_twice_counts_once(store, tmp_path) -> None:
    s = _session(store, tmp_path)
    s.judge_key("1")
    s.close()
    (p,) = s.written
    copy = p.with_name(p.name.replace("-000.parquet", "-001.parquet"))
    copy.write_bytes(p.read_bytes())
    assert len(jd.read_shards(store, USER, ["A"])) == 1


def test_a_flush_failing_part_way_never_writes_a_group_twice(store, tmp_path,
                                                             monkeypatch) -> None:
    s = _session(store, tmp_path)
    while s.current() is not None:
        s.judge_key("1")  # A (new, train), J (new, test), F (old, train): three groups
    real = jd.write_shard
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("Drive went away")
        return real(*a, **k)

    monkeypatch.setattr(jd, "write_shard", flaky)
    with pytest.raises(OSError, match="Drive went away"):
        s.flush(force=True)
    assert len(s.written) == 1 and s.pending == len(s.rows) - 3  # the A group landed
    s.flush(force=True)
    out = _shards(store)
    assert len(out) == len(s.rows) and out["judgement_id"].is_unique


def test_restored_journal_records_take_their_fields_from_the_queue(store, tmp_path) -> None:
    import json

    s = _session(store, tmp_path)
    s.judge_key("1")
    lines = s.journal_path.read_text(encoding="utf-8").splitlines()
    ev = json.loads(lines[0])
    ev.update({"label_set": "test", "draw": "uncertainty", "cohort": "old", "score": 0.0})
    forged = dict(ev, judgement_id="f" * 32, by="someone_else")
    s.journal_path.write_text(json.dumps(ev) + "\n" + json.dumps(forged) + "\n",
                              encoding="utf-8", newline="\n")
    again = _session(store, tmp_path)
    assert again.restored == 1  # another user's line is not restored
    again.close()
    out = _shards(store)
    row = check_queue(example_queue()).iloc[0]
    assert (out["label_set"].iloc[0], out["draw"].iloc[0], out["cohort"].iloc[0]) == (
        row["label_set"], row["draw"], row["cohort"])
    assert out["score"].iloc[0] == row["score"]
    assert out["judgement_id"].iloc[0] == ev["judgement_id"]


def test_an_undo_after_a_torn_line_is_not_swallowed(store, tmp_path) -> None:
    s = _session(store, tmp_path)
    s.judge_key("1")
    s.judge_key("4")
    with s.journal_path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write('{"op":"judge","recor')  # crash mid-line
    again = _session(store, tmp_path)  # repairs the torn line on open
    assert again.restored == 2
    again.undo()  # withdraws line_noise; this line must parse
    third = _session(store, tmp_path)
    assert third.restored == 1
    third.close()
    assert list(_shards(store)["judgement"]) == ["motion"]
