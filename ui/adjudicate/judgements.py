"""Keystrokes to judgements, and judgements to per-user, write-once label shards.

Key map (task 16 Change 1; ruling 2026-10-07 (c) item 3)
--------------------------------------------------------
``1`` motion, ``2`` physiology (not motion), ``3`` unsure, ``4`` line noise (not motion;
counts as a negative for the motion classifier - ``NEGATIVE_JUDGEMENTS`` in
``gems_blanking_v2.model.labels``). ``unjudged`` is never written: a core nobody judged
has no row (invariant 9).

Where judgements go
-------------------
``GemsStore.labels_path(animal, user, stamp)`` - the store's per-user, write-once label
file for one animal: ``labels/<animal>/events_<user>_<stamp>.parquet``. No new store
directory. Each flush writes one NEW file per ``(animal, cohort, label_set)`` group and
never rewrites one, so a shard is homogeneous in cohort and label_set; the stamp carries
both (``<utc>-adj-<cohort>-<label_set>-<seq>``) so a test-set shard is visible by name.
Every row carries ``label_set``; test rows (I/J/K, R1) are never merged into training by
``training_rows``, which drops them by ``label_set`` and by animal.

Columns: ``gems_blanking_v2.model.labels.LABEL_COLUMNS`` (``source`` and ``label_source``
are ``"human"``, ``basis`` is ``"adjudicated"``) plus :data:`EXTRA_COLUMNS`: ``draw``,
``region_start_s``, ``region_stop_s``, ``score`` (NaN when the queue had none), ``by``
(user id), ``at`` (UTC ISO-8601 with ``+00:00``), ``app_commit`` (this app's git commit,
null if unknown), ``queue_file``, ``queue_sha256``, ``judgement_id`` (uuid4 hex).

Crash safety
------------
Shards are written in batches, so between flushes each judgement and each undo is also
appended to a LOCAL per-user journal (JSONL, outside the store, never synced). On reopen,
journal judgements not undone and not found in any shard are restored as pending.
Journal lines are canonical ASCII JSON with absent keys for missing values.
"""

from __future__ import annotations

import io
import json
import math
import subprocess
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd
from gems_blanking_v2.io.store import (
    GemsStore,
    append_line,
    atomic_write_bytes,
    read_lines,
    utc_stamp,
    validate_component,
)
from gems_blanking_v2.model.labels import LABEL_COLUMNS

from ui.adjudicate.queue import core_key

__all__ = [
    "BASIS",
    "EXTRA_COLUMNS",
    "JUDGEMENTS",
    "KEY_TO_JUDGEMENT",
    "SHARD_COLUMNS",
    "app_commit",
    "judgement_for_key",
    "make_record",
    "read_journal",
    "read_shards",
    "write_shards",
]

KEY_TO_JUDGEMENT: Final[Mapping[str, str]] = {
    "1": "motion",
    "2": "physiology",
    "3": "unsure",
    "4": "line_noise",
}
"""The only keys that judge. Ruling (c) item 3 adds ``4``."""

JUDGEMENTS: Final[frozenset[str]] = frozenset(KEY_TO_JUDGEMENT.values())
BASIS: Final = "adjudicated"
EXTRA_COLUMNS: Final[tuple[str, ...]] = (
    "draw", "region_start_s", "region_stop_s", "score", "by", "at", "app_commit",
    "queue_file", "queue_sha256", "judgement_id",
)
SHARD_COLUMNS: Final[tuple[str, ...]] = (*LABEL_COLUMNS, *EXTRA_COLUMNS)


def judgement_for_key(key: str) -> str | None:
    """The judgement a key stands for, or ``None`` for a key that does not judge."""
    return KEY_TO_JUDGEMENT.get(str(key).strip())


def app_commit() -> str | None:
    """This app checkout's git commit, or ``None`` when it cannot be read."""
    here = Path(__file__).resolve().parent
    try:
        done = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    sha = done.stdout.strip()
    return sha if done.returncode == 0 and sha else None


def make_record(row: Mapping[str, Any], judgement: str, *, user: str,
                app_sha: str | None, queue_file: str, queue_sha256: str,
                at: datetime | None = None) -> dict[str, Any]:
    """One shard row for ``row`` (a validated queue row) judged ``judgement``."""
    if judgement not in JUDGEMENTS:
        msg = f"judgement must be one of {sorted(JUDGEMENTS)}, got {judgement!r}"
        raise ValueError(msg)
    score = row.get("score")
    score_f = float(score) if score is not None and not pd.isna(score) else math.nan
    moment = (at or datetime.now(UTC)).astimezone(UTC)
    return {
        "recording": str(row["recording"]), "animal": str(row["animal"]),
        "cohort": str(row["cohort"]), "start_s": float(row["start_s"]),
        "stop_s": float(row["stop_s"]), "judgement": judgement, "source": "human",
        "basis": BASIS, "label_set": str(row["label_set"]), "label_source": "human",
        "draw": str(row["draw"]), "region_start_s": float(row["region_start_s"]),
        "region_stop_s": float(row["region_stop_s"]), "score": score_f,
        "by": user, "at": moment.isoformat(timespec="milliseconds"),
        "app_commit": app_sha, "queue_file": queue_file, "queue_sha256": queue_sha256,
        "judgement_id": uuid.uuid4().hex,
    }


# ---------------------------------------------------------------------------
# shards
# ---------------------------------------------------------------------------


def _frame(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(list(records), columns=list(SHARD_COLUMNS))
    for col in ("start_s", "stop_s", "region_start_s", "region_stop_s", "score"):
        df[col] = df[col].astype("float64")
    return df


def write_shards(store: GemsStore, records: Sequence[Mapping[str, Any]], user: str,
                 *, when: datetime | None = None) -> list[Path]:
    """Write ``records`` as NEW shards, one per ``(animal, cohort, label_set)``.

    Atomic (temp file in the same directory, then ``os.replace``) and write-once: a
    shard is never rewritten, and a name already taken is skipped to the next sequence
    number rather than replaced. Returns the paths written.
    """
    if not records:
        return []
    for r in records:
        if r["judgement"] not in JUDGEMENTS:
            msg = f"refusing to write judgement {r['judgement']!r} (unjudged is never written)"
            raise ValueError(msg)
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for r in records:
        groups.setdefault((r["animal"], r["cohort"], r["label_set"]), []).append(r)
    stamp = utc_stamp(when)
    written: list[Path] = []
    for (animal, cohort, label_set), rows in sorted(groups.items()):
        buf = io.BytesIO()
        _frame(rows).to_parquet(buf, index=False)
        data = buf.getvalue()
        for seq in range(1000):
            tag = validate_component(f"{stamp}-adj-{cohort}-{label_set}-{seq:03d}")
            path = store.labels_path(animal, user, stamp=tag)
            if not path.exists():
                break
        else:
            msg = f"no free shard name for {animal}/{user} at {stamp}"
            raise FileExistsError(msg)
        _write_once(path, data)
        written.append(path)
    return written


def _write_once(path: Path, data: bytes) -> None:
    """Atomic write that refuses to replace an existing file."""
    if path.exists():
        msg = f"label shards are write-once; {path.name} already exists"
        raise FileExistsError(msg)
    atomic_write_bytes(path, data)


def read_shards(store: GemsStore, user: str, animals: Iterable[str]) -> pd.DataFrame:
    """Every adjudication row in ``user``'s shards for ``animals``, with ``core_key``.

    Reads only files matching the store's own name pattern for this user, and only
    rows whose ``basis`` is ``adjudicated``. An unreadable shard raises naming it -
    silently skipping one would re-present cores already judged.
    """
    frames: list[pd.DataFrame] = []
    for animal in sorted(set(animals)):
        pattern = store.labels_path(animal, user, stamp="*")
        folder = pattern.parent
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob(pattern.name)):
            try:
                df = pd.read_parquet(path)
            except Exception as exc:
                msg = f"cannot read label shard {path}: {exc}"
                raise OSError(msg) from exc
            if "basis" not in df.columns:
                continue
            frames.append(df.loc[df["basis"] == BASIS])
    if not frames:
        out = _frame([])
    else:
        out = pd.concat(frames, ignore_index=True)
    out["core_key"] = [core_key(r, a, b) for r, a, b in
                       zip(out["recording"].astype(str), out["start_s"], out["stop_s"],
                           strict=True)]
    return out


# ---------------------------------------------------------------------------
# the local journal
# ---------------------------------------------------------------------------


def _canonical(obj: Mapping[str, Any]) -> str:
    clean = {k: v for k, v in obj.items()
             if v is not None and not (isinstance(v, float) and math.isnan(v))}
    return json.dumps(clean, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def journal_append(path: Path, op: str, payload: Mapping[str, Any]) -> None:
    """Append one ``judge`` or ``undo`` event to the local journal."""
    if op not in ("judge", "undo"):
        msg = f"unknown journal op {op!r}"
        raise ValueError(msg)
    append_line(path, _canonical({"op": op, **payload}))


def read_journal(path: Path) -> list[dict[str, Any]]:
    """Judgement records in the journal that were not later undone, in order."""
    if not path.is_file():
        return []
    judged: dict[str, dict[str, Any]] = {}
    for line in read_lines(path):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn last line from a crash; every earlier line is whole
        if ev.get("op") == "judge":
            rec = {k: v for k, v in ev.items() if k != "op"}
            rec.setdefault("score", math.nan)
            rec.setdefault("app_commit", None)
            judged[rec["judgement_id"]] = rec
        elif ev.get("op") == "undo":
            judged.pop(ev.get("judgement_id", ""), None)
    return list(judged.values())
