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
null if unknown), ``queue_file``, ``queue_sha256``, ``judgement_id`` (uuid4 hex) and
``alias_table_sha256`` (old-cohort rows: the hash of the animal alias table the letter
came from, which Andrea has not yet confirmed; null on new-cohort rows).

Reading back (:func:`read_shards`) keeps only rows whose ``by`` is the user - a shard
file pattern ``events_ann_*`` also matches ``events_ann_smith_*`` - and drops duplicate
``judgement_id``s, so a shard that was written twice in a retry counts once.

Crash safety
------------
Shards are written in batches, so between flushes each judgement and each undo is also
appended to a LOCAL per-user journal (JSONL, outside the store, never synced). On reopen,
journal judgements not undone and not found in any shard are restored as pending - only
their judgement, time, id, user and commit are taken from the journal; every other field
is rebuilt from the queue row. Journal lines are canonical ASCII JSON with absent keys
for missing values; a torn last line (a crash mid-write) is terminated before anything
new is appended, so it can never swallow the next event.
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
from gems_blanking_v2.model.labels import LABEL_COLUMNS, is_test_animal

from ui.adjudicate.queue import core_key

__all__ = [
    "BASIS",
    "EXTRA_COLUMNS",
    "JUDGEMENTS",
    "KEY_TO_JUDGEMENT",
    "SHARD_COLUMNS",
    "app_commit",
    "check_writable",
    "judgement_for_key",
    "make_record",
    "journal_append",
    "journal_repair",
    "read_journal",
    "read_shards",
    "write_shard",
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
    "queue_file", "queue_sha256", "judgement_id", "alias_table_sha256",
)
JOURNAL_KEPT: Final[tuple[str, ...]] = ("judgement", "at", "judgement_id", "by", "app_commit")
"""The fields a restored journal record contributes; the rest come from the queue row."""
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
                at: datetime | None = None, judgement_id: str | None = None,
                ) -> dict[str, Any]:
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
        "judgement_id": judgement_id or uuid.uuid4().hex,
        "alias_table_sha256": _text_or_none(row.get("alias_table_sha256")),
    }


def _text_or_none(v: Any) -> str | None:
    return v if isinstance(v, str) and v else None


# ---------------------------------------------------------------------------
# shards
# ---------------------------------------------------------------------------


def _frame(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(list(records), columns=list(SHARD_COLUMNS))
    for col in ("start_s", "stop_s", "region_start_s", "region_stop_s", "score"):
        df[col] = df[col].astype("float64")
    return df


def check_writable(records: Sequence[Mapping[str, Any]]) -> None:
    """Raise unless every record may be written: judged, and R1-consistent."""
    for r in records:
        if r["judgement"] not in JUDGEMENTS:
            msg = f"refusing to write judgement {r['judgement']!r} (unjudged is never written)"
            raise ValueError(msg)
        if is_test_animal(r["cohort"], r["animal"]) and r["label_set"] != "test":
            msg = (f"refusing to write {r['cohort']} {r['animal']} with label_set "
                   f"{r['label_set']!r}: a test animal's labels are 'test' (R1)")
            raise ValueError(msg)


def shard_groups(records: Sequence[Mapping[str, Any]]
                 ) -> list[tuple[tuple[str, str, str], list[Mapping[str, Any]]]]:
    """``records`` split by ``(animal, cohort, label_set)``, in a fixed order."""
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for r in records:
        groups.setdefault((r["animal"], r["cohort"], r["label_set"]), []).append(r)
    return sorted(groups.items())


def write_shard(store: GemsStore, rows: Sequence[Mapping[str, Any]], user: str,
                *, when: datetime | None = None) -> Path:
    """Write ONE group's rows (one animal, cohort and label_set) as a new shard.

    Atomic (temp file in the same directory, then ``os.replace``) and write-once: a
    shard is never rewritten, and a name already taken is skipped to the next sequence
    number rather than replaced. Refuses unjudged rows and test-animal rows not marked
    ``test``.
    """
    if not rows:
        msg = "write_shard needs at least one row"
        raise ValueError(msg)
    check_writable(rows)
    keys = {(r["animal"], r["cohort"], r["label_set"]) for r in rows}
    if len(keys) != 1:
        msg = f"one shard holds one (animal, cohort, label_set); got {sorted(keys)}"
        raise ValueError(msg)
    ((animal, cohort, label_set),) = keys
    buf = io.BytesIO()
    _frame(rows).to_parquet(buf, index=False)
    stamp = utc_stamp(when)
    for seq in range(1000):
        tag = validate_component(f"{stamp}-adj-{cohort}-{label_set}-{seq:03d}")
        path = store.labels_path(animal, user, stamp=tag)
        if not path.exists():
            break
    else:
        msg = f"no free shard name for {animal}/{user} at {stamp}"
        raise FileExistsError(msg)
    _write_once(path, buf.getvalue())
    return path


def write_shards(store: GemsStore, records: Sequence[Mapping[str, Any]], user: str,
                 *, when: datetime | None = None) -> list[Path]:
    """Write ``records`` as new shards, one per ``(animal, cohort, label_set)``.

    Everything is checked before the first file is written. Returns the paths written.
    """
    if not records:
        return []
    check_writable(records)
    return [write_shard(store, rows, user, when=when) for _k, rows in shard_groups(records)]


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
            if "basis" not in df.columns or "by" not in df.columns:
                continue
            frames.append(df.loc[(df["basis"] == BASIS) & (df["by"] == user)])
    if not frames:
        out = _frame([])
    else:
        out = pd.concat(frames, ignore_index=True)
        out = out.drop_duplicates(subset="judgement_id", keep="first", ignore_index=True)
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


def journal_repair(path: Path) -> bool:
    """Terminate a torn last line so the next append starts on its own line.

    Returns whether a newline was added. Called once when a session opens its journal.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb") as fh:
        fh.seek(-1, 2)
        last = fh.read(1)
    if last == b"\n":
        return False
    with path.open("ab") as fh:
        fh.write(b"\n")
    return True


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
        if ev.get("op") == "judge" and "judgement_id" in ev:
            rec = {k: v for k, v in ev.items() if k != "op"}
            rec.setdefault("app_commit", None)
            judged[rec["judgement_id"]] = rec
        elif ev.get("op") == "undo":
            judged.pop(ev.get("judgement_id", ""), None)
    return list(judged.values())
