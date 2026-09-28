"""Score a committed blind-audit round (task 09), or check whether the next may be drawn.

Usage::

    python -m ui.audit.score <plan_id>          # gate score: write it, print the report
    python -m ui.audit.score --latest           # the most recent complete round
    python -m ui.audit.score --tuning <plan_id> # re-score with the CURRENT generator:
                                                #   "tuning check, not gate evidence"
    python -m ui.audit.score --pooled           # the cumulative gate over eligible rounds
    python -m ui.audit.score --check-next       # may another round be drawn now?
    python -m ui.audit.score --measure-budget [PER_CELL [SEED]]
                                                # candidates per recording on the pool

The scoring itself is ``gems_blanking_v2.detect.recall`` (pre-declared 2026-09-26).
This runner supplies the candidates and z-traces. The audit window keeps its reveal
in memory only, so they are RECOMPUTED here with the very function the window
revealed with - ``bridge.reveal_for_region`` on the recording loaded by
``bridge.load_new_cohort``, over the assessable region recorded for the span -
which is deterministic in the code and the data. Both repositories' commits are
recorded in the score, so a re-score under different code is visible as such.

It reads whole recordings from the Drive: run it when nobody is labelling.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import gems_blanking_v2
from gems_blanking_v2.detect.recall import (
    check_next_round,
    pooled_gate,
    score_stored_round,
)
from gems_blanking_v2.io.store import GemsStore, find_gems_root

from ui.audit import bridge

_REPO = Path(__file__).resolve().parents[2]


def _commit(repo: Path) -> str:
    """``<sha>`` or ``<sha>+dirty``; ``unknown`` outside a git checkout."""
    try:
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("+dirty" if dirty else "")


def _region(span: dict[str, Any]) -> tuple[float, float]:
    """The assessable region the reveal used: the recorded one containing the span."""
    start, stop = float(span["start_s"]), float(span["stop_s"])
    for lo, hi in span["record"].get("assessable_regions_s", []):
        if lo <= start and stop <= hi:
            return float(lo), float(hi)
    msg = (f"{span['span_id']}: no recorded assessable region contains "
           f"[{start}, {stop}) - the span record is missing or damaged")
    raise ValueError(msg)


def make_reveal(store: GemsStore) -> Any:
    """The reveal function for ``score_stored_round``: the window's own, recomputed."""
    from gems_blanking_v2.io.channel_map import meta_path

    def reveal(span: dict[str, Any]) -> tuple[Any, Any]:
        meta = json.loads(meta_path(store, span["animal"], span["recording_id"])
                          .read_text(encoding="utf-8"))
        loaded = bridge.load_new_cohort(store.root / meta["source_path"], span["animal"],
                                        store=store)
        return bridge.reveal_for_region(loaded.recording, _region(span))

    return reveal


def _latest_complete(store: GemsStore) -> str:
    from gems_blanking_v2.detect.recall import load_round

    folder = store.audit_plan_path("x").parent
    for path in sorted(folder.glob("plan_*.json"), reverse=True) if folder.is_dir() else []:
        plan_id = json.loads(path.read_text(encoding="utf-8"))["plan_id"]
        try:
            load_round(store, plan_id)
        except FileNotFoundError:
            continue
        return plan_id
    msg = "no complete audit round in the store"
    raise FileNotFoundError(msg)


def main(argv: list[str]) -> int:
    """Command-line entry point."""
    store = GemsStore(find_gems_root())
    if argv == ["--check-next"]:
        ok, why = check_next_round(store, reveal_sha=bridge.reveal_sha())
        print(("ALLOWED: " if ok else "REFUSED: ") + why)
        return 0 if ok else 1
    if argv[:1] == ["--measure-budget"]:
        from ui.audit.budget import measure_budget

        per_cell = int(argv[1]) if len(argv) > 1 else 5
        seed = int(argv[2]) if len(argv) > 2 else 20260928
        rec = measure_budget(store, per_cell=per_cell, seed=seed)
        print(json.dumps({k: rec[k] for k in ("key", "n_recordings", "candidates_median",
                                               "candidates_quantile", "fraction_over_budget",
                                               "time_covered_median", "within_budget")},
                         indent=1))
        return 0 if rec["within_budget"] else 1
    if argv == ["--pooled"]:
        print(json.dumps(pooled_gate(store), indent=1))
        return 0
    mode = "tuning" if argv[:1] == ["--tuning"] else "gate"
    args = argv[1:] if mode == "tuning" else argv
    if len(args) != 1:
        print(__doc__)
        return 2
    plan_id = _latest_complete(store) if args[0] == "--latest" else args[0]
    score = score_stored_round(store, plan_id, make_reveal(store), mode=mode, provenance={
        "reveal": "ui.audit.bridge.reveal_for_region (recomputed)",
        # The reveal's composition lives here, not in the package the generator hash
        # covers, so it is hashed too.
        "reveal_source_sha256": bridge.reveal_sha(),
        "detector_pyqt_commit": _commit(_REPO),
        # The installed package's own checkout (an editable install), never a
        # sibling folder found by name (invariant 21).
        "gems_blanking_v2_commit": _commit(Path(gems_blanking_v2.__file__).resolve().parents[1]),
    })
    print(score.report())
    if mode == "gate":
        print(f"\nwritten: {store.relpath(store.audit_score_path(plan_id))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
