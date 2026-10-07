"""The adjudication state machine, free of Qt so it can be driven and tested headless.

One queue, one user. The cursor walks the queue (in :func:`presentation_order`); a
judgement is recorded, journalled locally, and becomes *pending*; pending judgements are
written to the store as write-once shards in batches. The newest :data:`UNDO_DEPTH`
judgements are always held back from a batch flush so they can be undone - a shard on
the store is never edited, so an undo can only reach what has not been written. Closing
the session writes everything (``flush(force=True)``).

Progress is read back, not remembered: on open, the user's existing shards say which
cores are already judged, and the local journal restores anything judged but not yet
written when the app last stopped.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd
from gems_blanking_v2.io.store import GemsStore, safe_component

from ui.adjudicate import judgements as jd
from ui.adjudicate.queue import core_key

_log = logging.getLogger(__name__)

__all__ = ["FLUSH_BATCH", "UNDO_DEPTH", "AdjudicationSession", "AnimalProgress"]

UNDO_DEPTH: Final = 20
"""Judgements always kept unwritten so they can be undone (the app's 20-deep undo)."""

FLUSH_BATCH: Final = 20
"""A batch flush happens once this many judgements are pending beyond the undo depth."""


@dataclass(frozen=True, slots=True)
class AnimalProgress:
    """Judged and remaining cores for one animal (and cohort) in this queue."""

    animal: str
    cohort: str
    judged: int
    total: int

    @property
    def unjudged(self) -> int:
        """Cores in the queue nobody has judged yet."""
        return self.total - self.judged


def file_sha256(path: Path) -> str:
    """Hex sha256 of a file (the queue's identity in every record and the journal name)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _widened_of(rec: dict[str, Any]) -> tuple[float, float] | None:
    """A journal record's widened boundary, or ``None`` (absent keys).

    A line carrying only one of the two keys is malformed: it raises (the caller drops the
    line and logs it) rather than restoring a judgement with half a boundary.
    """
    a, b = rec.get("widened_start_s"), rec.get("widened_stop_s")
    if a is None and b is None:
        return None
    if a is None or b is None:
        msg = f"journal line {rec.get('judgement_id')} carries half a widened boundary"
        raise ValueError(msg)
    return float(a), float(b)


class AdjudicationSession:
    """Walks a validated, ordered queue and records one judgement per core.

    What survives a crash: every judgement (and its widened boundary) in the journal. A
    widen that is only PENDING - drawn but not yet recorded with key 1 - lives in the
    window and is lost with it. A journal line that cannot be restored (e.g. half a
    widened boundary) is dropped, logged, and listed in :attr:`dropped`.

    Parameters
    ----------
    queue
        Output of :func:`ui.adjudicate.queue.load_queue` (validated, ordered, keyed).
    store
        Where shards are written (``GemsStore.labels_path``).
    user
        Filename-safe user id recorded as ``by`` and in shard names.
    journal_dir
        LOCAL directory for the crash journal; never inside the store.
    queue_file, queue_sha256
        The queue's name and content hash, recorded on every row.
    app_sha
        This app's commit, recorded on every row (``None`` if unknown).
    clock
        Returns the judgement time; injectable for tests.
    rejudge
        Re-judge mode: only judgements made from THIS queue file (``queue_file``) count
        as done, so cores judged earlier from another queue are shown again. Nothing
        stored is changed or hidden; the new judgement is simply newer, and the newest
        judgement of a core is the one in force.
    """

    def __init__(self, queue: pd.DataFrame, store: GemsStore, user: str, *,
                 journal_dir: Path, queue_file: str, queue_sha256: str,
                 app_sha: str | None, undo_depth: int = UNDO_DEPTH,
                 flush_batch: int = FLUSH_BATCH,
                 clock: Callable[[], datetime] | None = None,
                 rejudge: bool = False) -> None:
        if "core_key" not in queue.columns:
            msg = "queue must come from load_queue/check_queue (no core_key column)"
            raise ValueError(msg)
        self.rows: list[dict[str, Any]] = [
            {str(k): v for k, v in r.items()} for r in queue.to_dict("records")]
        self.store = store
        self.user = safe_component(user)
        self.queue_file = queue_file
        self.queue_sha256 = queue_sha256
        self.app_sha = app_sha
        self.undo_depth = int(undo_depth)
        self.flush_batch = int(flush_batch)
        self._clock = clock
        self.journal_path = (Path(journal_dir)
                             / f"adjudication_{self.user}_{queue_sha256[:16]}.jsonl")
        self._keys = [r["core_key"] for r in self.rows]
        self._index = {k: i for i, k in enumerate(self._keys)}
        self.written: list[Path] = []
        self.dropped: list[str] = []

        self.rejudge = bool(rejudge)
        stored = jd.read_shards(store, self.user, {r["animal"] for r in self.rows})
        if self.rejudge:
            stored = stored.loc[stored["queue_file"].astype(str) == str(queue_file)]
        # The newest stored judgement of a core is the one in force.
        stored = stored.sort_values("at", kind="stable")
        self._judged: dict[str, str] = {
            k: j for k, j in zip(stored["core_key"], stored["judgement"], strict=True)
            if k in self._index}
        stored_ids = set(stored["judgement_id"].astype(str))
        self._pending: list[dict[str, Any]] = []
        jd.journal_repair(self.journal_path)
        for rec in jd.read_journal(self.journal_path):
            restored = self._restore(rec, stored_ids)
            if restored is not None:
                self._pending.append(restored)
                self._judged[restored["core_key"]] = restored["judgement"]
                del restored["core_key"]
        self.restored = len(self._pending)
        self.cursor = 0
        self._advance_to_unjudged(0)

    def _restore(self, rec: dict[str, Any], stored_ids: set[str]) -> dict[str, Any] | None:
        """A journal record rebuilt from its queue row, or ``None`` if it does not apply.

        Only :data:`judgements.JOURNAL_KEPT` comes from the journal; recording, times,
        cohort, label_set, draw and the rest are the queue's, so a journal line can never
        carry a field the queue does not say (e.g. a label_set the queue does not give).
        """
        try:
            if rec["judgement_id"] in stored_ids or rec.get("by") != self.user:
                return None
            key = core_key(rec["recording"], rec["start_s"], rec["stop_s"])
            if key not in self._index:
                return None
            out = jd.make_record(
                self.rows[self._index[key]], rec["judgement"], user=self.user,
                app_sha=rec.get("app_commit"), queue_file=self.queue_file,
                queue_sha256=self.queue_sha256, at=datetime.fromisoformat(rec["at"]),
                judgement_id=rec["judgement_id"], widened=_widened_of(rec))
        except (KeyError, TypeError, ValueError) as exc:
            # A malformed journal line is not a judgement; say so rather than drop it quietly.
            _log.warning("journal line %s dropped on restore: %s",
                         rec.get("judgement_id", "?"), exc)
            self.dropped.append(str(rec.get("judgement_id", "?")))
            return None
        out["core_key"] = key
        return out

    # -- navigation -------------------------------------------------------

    def _advance_to_unjudged(self, start: int) -> None:
        n = len(self.rows)
        for step in range(n):
            i = (start + step) % n
            if self._keys[i] not in self._judged:
                self.cursor = i
                return
        self.cursor = start % n if n else 0

    @property
    def done(self) -> bool:
        """Every core in the queue has a judgement."""
        return len(self._judged) >= len(self.rows)

    def current(self) -> dict[str, Any] | None:
        """The row to judge now, or ``None`` when the queue is finished."""
        if self.done:
            return None
        return self.rows[self.cursor]

    def judgement_of(self, row: dict[str, Any]) -> str | None:
        """The judgement in force for ``row``, or ``None`` if it is unjudged."""
        return self._judged.get(row["core_key"])

    def skip(self) -> dict[str, Any] | None:
        """Leave the current core unjudged and move to the next unjudged one."""
        if self.done:
            return None
        self._advance_to_unjudged(self.cursor + 1)
        return self.current()

    # -- judging ------------------------------------------------------------

    def judge_key(self, key: str, *, widened: tuple[float, float] | None = None
                  ) -> dict[str, Any] | None:
        """Judge the current core by keystroke; a non-judging key does nothing."""
        judgement = jd.judgement_for_key(key)
        if judgement is None:
            return None
        return self.judge(judgement, widened=widened)

    def judge(self, judgement: str, *, widened: tuple[float, float] | None = None
              ) -> dict[str, Any] | None:
        """Record ``judgement`` for the current core and advance. Returns the record.

        ``widened`` (recording-timeline seconds, containing the core) goes only with
        ``motion``; it is stored beside the core, never instead of it.
        """
        row = self.current()
        if row is None:
            return None
        rec = jd.make_record(row, judgement, user=self.user, app_sha=self.app_sha,
                            queue_file=self.queue_file, queue_sha256=self.queue_sha256,
                            at=self._clock() if self._clock else None, widened=widened)
        jd.journal_append(self.journal_path, "judge", rec)
        self._pending.append(rec)
        self._judged[row["core_key"]] = judgement
        self._advance_to_unjudged(self.cursor + 1)
        self.flush()
        return rec

    def can_undo(self) -> bool:
        """Whether a judgement is still unwritten and so can be undone."""
        return bool(self._pending)

    def undo(self) -> dict[str, Any] | None:
        """Withdraw the newest unwritten judgement and return to its core.

        Returns the withdrawn record, or ``None`` when nothing is undoable (everything
        has been written to the store, where shards are never edited).
        """
        if not self._pending:
            return None
        rec = self._pending.pop()
        jd.journal_append(self.journal_path, "undo", {"judgement_id": rec["judgement_id"]})
        key = core_key(rec["recording"], rec["start_s"], rec["stop_s"])
        self._judged.pop(key, None)
        self.cursor = self._index[key]
        return rec

    # -- writing --------------------------------------------------------------

    @property
    def pending(self) -> int:
        """Judgements recorded but not yet written to the store."""
        return len(self._pending)

    def flush(self, *, force: bool = False) -> list[Path]:
        """Write pending judgements as new shards.

        Without ``force``: only once at least ``undo_depth + flush_batch`` are pending,
        and then all but the newest ``undo_depth``. With ``force``: all of them.

        Shards are written one ``(animal, cohort, label_set)`` group at a time, and each
        group leaves the pending list (by ``judgement_id``) as soon as its shard lands -
        so a failure part-way through never writes a group twice on the retry.
        """
        if force:
            batch = list(self._pending)
        elif len(self._pending) >= self.undo_depth + self.flush_batch:
            batch = self._pending[: len(self._pending) - self.undo_depth]
        else:
            return []
        if not batch:
            return []
        jd.check_writable(batch)
        paths: list[Path] = []
        for _key, rows in jd.shard_groups(batch):
            path = jd.write_shard(self.store, rows, self.user)
            ids = {r["judgement_id"] for r in rows}
            self._pending = [r for r in self._pending if r["judgement_id"] not in ids]
            self.written.append(path)
            paths.append(path)
        return paths

    def close(self) -> list[Path]:
        """Write everything pending. Call when the window closes."""
        return self.flush(force=True)

    # -- progress -------------------------------------------------------------

    def progress(self) -> list[AnimalProgress]:
        """Judged / total per (animal, cohort), in first-appearance order."""
        totals: dict[tuple[str, str], list[int]] = {}
        for r in self.rows:
            t = totals.setdefault((str(r["animal"]), str(r["cohort"])), [0, 0])
            t[1] += 1
            if r["core_key"] in self._judged:
                t[0] += 1
        return [AnimalProgress(a, c, j, n) for (a, c), (j, n) in totals.items()]

    def counts(self) -> Counter[str]:
        """How many cores in this queue carry each judgement."""
        return Counter(self._judged.values())

    def position(self) -> tuple[int, int]:
        """``(judged, total)`` over the whole queue."""
        return len(self._judged), len(self.rows)
