"""Measure the candidate budget (task 09) on the eligible pool, as the audit reveals.

Task 09 has required since before any marks that the pinned generator produce
<= 3000 candidates per recording. Round 1 showed why (2026-09-28): in three of five
spans candidates covered 93-96% of the time, so recall was met by coverage alone.
A round may not be drawn until the CURRENT generator is measured within budget
(``gems_blanking_v2.detect.recall.budget_status``).

Each recording is measured exactly as the audit window reveals a span: loaded by
``bridge.load_new_cohort``, stim removed by the protocol window
(``stim_epoch_s`` / ``assessable_regions``), and candidates from
``bridge.reveal_for_region`` over each assessable region. "Candidates per
recording" is the sum over its regions; "time covered" is the union of candidate
intervals inside them. It reads whole recordings from the Drive.
"""

from __future__ import annotations

import random
from typing import Any

from gems_blanking_v2.detect.recall import budget_record, write_budget
from gems_blanking_v2.io.audit_pool import (
    EligibleRecording,
    assessable_regions,
    eligible_recordings,
    stim_epoch_s,
)
from gems_blanking_v2.io.stim_split import PROTOCOL_FILENAME, load_protocol_book
from gems_blanking_v2.io.store import GemsStore

from ui.audit import bridge

PLANNABLE = ("baseline", "stim_recovery")


def sample_pool(pool: list[EligibleRecording], per_cell: int, seed: int
                ) -> list[EligibleRecording]:
    """Up to ``per_cell`` recordings from every (animal, condition) cell, seeded."""
    rng = random.Random(seed)
    cells: dict[tuple[str, str], list[EligibleRecording]] = {}
    for r in sorted(pool, key=lambda r: r.session):
        if r.epoch in PLANNABLE and (r.duration_s or 0) > 0:
            cells.setdefault((r.animal, r.epoch), []).append(r)
    out: list[EligibleRecording] = []
    for _key, recs in sorted(cells.items()):
        rng.shuffle(recs)
        out += recs[:per_cell]
    return out


def _union_inside(intervals: Any, lo: float, hi: float) -> float:
    total, end = 0.0, float("-inf")
    for a, b in sorted((max(float(a), lo), min(float(b), hi)) for a, b in intervals):
        if b <= a:
            continue
        if a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def measure_recording(store: GemsStore, rec: EligibleRecording, book: Any) -> dict[str, Any]:
    """Candidates and time covered over one recording's assessable regions."""
    loaded = bridge.load_new_cohort(rec.source, rec.animal, store=store)
    recording = loaded.recording
    duration = recording.data.shape[0] / recording.fs
    stim, _method = stim_epoch_s(recording, rec.source, rec.epoch,
                                 book.for_path(store.relpath(rec.source)))
    regions = assessable_regions(duration, stim)
    candidates, covered, assessable = 0, 0.0, 0.0
    for lo, hi in regions:
        intervals, _traces = bridge.reveal_for_region(recording, (lo, hi))
        candidates += len(intervals)
        covered += _union_inside(intervals, lo, hi)
        assessable += hi - lo
    return {"session": rec.session, "animal": rec.animal, "condition": rec.epoch,
            "folder": rec.folder_name, "duration_s": duration, "assessable_s": assessable,
            "candidates": candidates, "covered_s": covered,
            "candidates_per_20min": candidates * 1200.0 / assessable if assessable else None}


def measure_budget(store: GemsStore, *, per_cell: int, seed: int,
                   progress: Any = print) -> dict[str, Any]:
    """Measure, record and return the budget for the current generator."""
    book = load_protocol_book(store.root / PROTOCOL_FILENAME)
    sample = sample_pool(eligible_recordings(store), per_cell, seed)
    rows = []
    for k, rec in enumerate(sample, 1):
        row = measure_recording(store, rec, book)
        rows.append(row)
        progress(f"[{k}/{len(sample)}] {row['animal']} {row['condition']:13} "
                 f"{row['folder']:34} {row['candidates']:6d} candidates, "
                 f"{row['covered_s'] / max(row['assessable_s'], 1e-9):.1%} of time")
    record = budget_record(rows, reveal_sha=bridge.reveal_sha(), sample_rule=(
        f"up to {per_cell} recordings per (animal, condition) cell of the eligible pool, "
        f"seed {seed}; candidates over each recording's assessable regions (stim removed "
        "by the protocol window), exactly as the audit window reveals"))
    write_budget(store, record)
    return record
