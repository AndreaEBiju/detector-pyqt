"""The adjudication queue: one row per candidate core to be judged by a human.

Schema (a parquet file; one row per core)
-----------------------------------------
Required columns - a missing one raises :class:`QueueSchemaError` naming it:

``recording``       str. New cohort: the store session key (the TDT block key that
                    ``gems_blanking_v2.io.channel_map.meta_path`` takes). Old cohort: the
                    recording id ``rid`` whose signal is ``<rid>_notched.mat`` (or
                    ``<rid>.mat`` when ``rid`` already ends in ``_notched``).
``animal``          str. The animal letter the store files the recording under: exactly
                    one upper-case letter (``^[A-Z]$``); ``"j"`` or ``"J "`` is refused.
``cohort``          ``"new"`` or ``"old"``.
``start_s``         float64 seconds, the core's start on the RECORDING's timeline
                    (0 = first sample), 0-based half-open ``[start_s, stop_s)``.
``stop_s``          float64 seconds, the core's end (exclusive).
``region_start_s``  float64 seconds, the assessable region the core was detected in
``region_stop_s``   (the region z was referenced over). Must contain the core.
``draw``            ``"random"`` or ``"uncertainty"``: how the core entered the queue.
``label_set``       ``"train"`` or ``"test"``. Copied verbatim into every judgement.
                    Rows for which ``gems_blanking_v2.model.labels.is_test_animal(cohort,
                    animal)`` holds (new-cohort I, J, K; ruling 2026-10-07 (b) R1) MUST be
                    ``"test"``; a queue that says otherwise is refused. Old-cohort letters
                    are different rats (old JEL is also "J") and are not test animals.

Optional columns:

``folder``          str, POSIX path RELATIVE to the old cohort's Survivals root, of the
                    folder holding ``<rid>_notched.mat``. **Required on every old row**
                    (null on new rows). Never absolute (cross-platform rule 2).
``alias_table_sha256``
                    str, 64 lower-case hex: sha256 of the bytes of the old-cohort animal
                    alias table the ``animal`` letter came from. **Required on every old
                    row** (the alias table is not yet confirmed by Andrea, so every
                    old-cohort label records which version named its animal); null on
                    new rows. Recorded on every judgement.
``score``           float64, the model's P(motion) for the core; shown, and recorded.
``peak_z``, ``peak_band``, ``peak_signal``
                    Night 1's per-core fields; shown beside the plot when present.

Rules enforced by :func:`check_queue`: no nulls in required columns; enumerations as
above; every time finite with ``region_start_s <= start_s < stop_s <= region_stop_s``;
and the core key (:func:`core_key`) unique across the file - a key assembled from parts
is proved unique at load, not assumed (invariant 27).

Order: cores are presented grouped by recording, recordings in order of first
appearance and cores in file order within each (:func:`presentation_order`), because a
new-cohort recording is ~2 GB in memory and is loaded once per group.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Final

import numpy as np
import pandas as pd
from gems_blanking_v2.model.labels import is_test_animal

__all__ = [
    "COHORTS",
    "DRAWS",
    "LABEL_SETS",
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "QueueSchemaError",
    "check_queue",
    "core_key",
    "example_queue",
    "load_queue",
    "presentation_order",
]

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "recording", "animal", "cohort", "start_s", "stop_s",
    "region_start_s", "region_stop_s", "draw", "label_set",
)
OPTIONAL_COLUMNS: Final[tuple[str, ...]] = (
    "folder", "alias_table_sha256", "score", "peak_z", "peak_band", "peak_signal",
)
COHORTS: Final[frozenset[str]] = frozenset({"new", "old"})
DRAWS: Final[frozenset[str]] = frozenset({"random", "uncertainty"})
LABEL_SETS: Final[frozenset[str]] = frozenset({"train", "test"})

_TIME_COLUMNS: Final = ("start_s", "stop_s", "region_start_s", "region_stop_s")
_KEY_RESOLUTION_S: Final = 1e-6
"""Core keys round times to whole microseconds: two distinct cores are never closer."""


class QueueSchemaError(ValueError):
    """The queue file does not satisfy the schema in this module's docstring."""


def core_key(recording: str, start_s: float, stop_s: float) -> str:
    """The one construction site of a core's identity: ``recording|start_us|stop_us``.

    Integer microseconds, so the key never depends on how a float was formatted
    (invariant 22) and matches between the queue and the label shards exactly.
    """
    lo = round(float(start_s) / _KEY_RESOLUTION_S)
    hi = round(float(stop_s) / _KEY_RESOLUTION_S)
    return f"{recording}|{lo}|{hi}"


_ANIMAL_RE: Final = re.compile(r"[A-Z]")
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")


def check_queue(df: pd.DataFrame) -> pd.DataFrame:
    """Validate ``df`` against the schema; return it with a ``core_key`` column added.

    Raises :class:`QueueSchemaError` naming the column (or the offending rows).
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        msg = f"adjudication queue is missing required column(s): {missing}"
        raise QueueSchemaError(msg)
    if df.empty:
        msg = "adjudication queue has no rows"
        raise QueueSchemaError(msg)
    for col in REQUIRED_COLUMNS:
        n = int(df[col].isna().sum())
        if n:
            msg = f"column {col!r} is null on {n} row(s); every required column needs a value"
            raise QueueSchemaError(msg)
    for col, allowed in (("cohort", COHORTS), ("draw", DRAWS), ("label_set", LABEL_SETS)):
        bad = sorted(set(df[col].astype(str)) - allowed)
        if bad:
            msg = f"column {col!r} has value(s) {bad}; allowed: {sorted(allowed)}"
            raise QueueSchemaError(msg)
    bad_animal = sorted({a for a in df["animal"] if not (isinstance(a, str)
                                                        and _ANIMAL_RE.fullmatch(a))}, key=repr)
    if bad_animal:
        msg = (f"column 'animal' must be one upper-case letter, got {bad_animal[:5]!r} "
               "(no lower case, no whitespace)")
        raise QueueSchemaError(msg)
    out = df.copy()
    for col in _TIME_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="raise").astype(np.float64)
        if not np.isfinite(out[col].to_numpy()).all():
            msg = f"column {col!r} has non-finite values"
            raise QueueSchemaError(msg)
    bad_order = ~((out["region_start_s"] <= out["start_s"]) & (out["start_s"] < out["stop_s"])
                  & (out["stop_s"] <= out["region_stop_s"]))
    if bad_order.any():
        rows = list(out.index[bad_order][:5])
        msg = ("need region_start_s <= start_s < stop_s <= region_stop_s; violated on "
               f"{int(bad_order.sum())} row(s), first {rows}")
        raise QueueSchemaError(msg)
    old = out["cohort"] == "old"
    if old.any():
        if "folder" not in out.columns:
            msg = "column 'folder' is required when the queue has old-cohort rows"
            raise QueueSchemaError(msg)
        if out.loc[old, "folder"].isna().any():
            n_null = int(out.loc[old, "folder"].isna().sum())
            msg = f"column 'folder' is null on {n_null} old row(s)"
            raise QueueSchemaError(msg)
        for f in out.loc[old, "folder"].astype(str).unique():
            pure = PurePosixPath(f)
            if pure.is_absolute() or ".." in pure.parts or "\\" in f or ":" in f:
                msg = (f"column 'folder' must be a POSIX path relative to the Survivals "
                       f"root, got {f!r}")
                raise QueueSchemaError(msg)
        if "alias_table_sha256" not in out.columns:
            msg = "column 'alias_table_sha256' is required when the queue has old-cohort rows"
            raise QueueSchemaError(msg)
        bad_sha = [v for v in out.loc[old, "alias_table_sha256"]
                   if not (isinstance(v, str) and _SHA256_RE.fullmatch(v))]
        if bad_sha:
            msg = (f"column 'alias_table_sha256' must be 64 lower-case hex on every old row; "
                   f"{len(bad_sha)} row(s) are not, first {bad_sha[0]!r}")
            raise QueueSchemaError(msg)
    # R1: no test-animal label may enter training.
    leak = np.array([is_test_animal(c, a) and ls != "test" for c, a, ls in
                     zip(out["cohort"], out["animal"], out["label_set"], strict=True)],
                    dtype=bool)
    if leak.any():
        msg = (f"{int(leak.sum())} row(s) for which is_test_animal(cohort, animal) holds "
               "(new-cohort I/J/K) have label_set != 'test' (ruling 2026-10-07 (b) R1: "
               "their labels never enter training)")
        raise QueueSchemaError(msg)
    out["core_key"] = [core_key(r, a, b) for r, a, b in
                       zip(out["recording"].astype(str), out["start_s"], out["stop_s"],
                           strict=True)]
    dup = out["core_key"].duplicated(keep=False)
    if dup.any():
        msg = (f"{int(dup.sum())} rows share a core key (recording, start_s, stop_s); "
               f"first: {sorted(set(out.loc[dup, 'core_key']))[:3]}")
        raise QueueSchemaError(msg)
    return out


def presentation_order(df: pd.DataFrame) -> pd.DataFrame:
    """Group rows by recording (first-appearance order), file order within each."""
    first = {r: i for i, r in reversed(list(enumerate(df["recording"])))}
    order = sorted(range(len(df)), key=lambda i: (first[df["recording"].iloc[i]], i))
    return df.iloc[order].reset_index(drop=True)


def load_queue(path: Path) -> pd.DataFrame:
    """Read, validate and order a queue parquet file."""
    df = pd.read_parquet(Path(path))
    return presentation_order(check_queue(df))


EXAMPLE_ALIAS_SHA256: Final = "0" * 63 + "1"
"""A placeholder alias-table hash for the synthetic old row (not a real table)."""


def example_queue() -> pd.DataFrame:
    """A tiny synthetic queue covering every enumeration: for tests and documentation.

    Two new-cohort recordings (one train animal, one test animal) and one old-cohort
    recording, with both draws. Nothing here names a real recording.
    """
    rows = [
        # recording, animal, cohort, start, stop, region, draw, label_set, folder, score
        ("synth_a_rec1", "A", "new", 20.00, 20.12, (10.0, 190.0), "random", "train", None, 0.91),
        ("synth_a_rec1", "A", "new", 55.50, 55.60, (10.0, 190.0), "uncertainty", "train",
         None, 0.52),
        ("synth_j_rec1", "J", "new", 30.00, 30.30, (10.0, 190.0), "random", "test", None, 0.12),
        ("synth_a_rec1", "A", "new", 90.00, 90.05, (10.0, 190.0), "random", "train", None, 0.33),
        ("synth_old_1", "F", "old", 12.00, 12.20, (0.0, 60.0), "uncertainty", "train",
         "synth_folder/synth_old_1", 0.48),
    ]
    return pd.DataFrame({
        "recording": [r[0] for r in rows], "animal": [r[1] for r in rows],
        "cohort": [r[2] for r in rows],
        "start_s": np.array([r[3] for r in rows], dtype=np.float64),
        "stop_s": np.array([r[4] for r in rows], dtype=np.float64),
        "region_start_s": np.array([r[5][0] for r in rows], dtype=np.float64),
        "region_stop_s": np.array([r[5][1] for r in rows], dtype=np.float64),
        "draw": [r[6] for r in rows], "label_set": [r[7] for r in rows],
        "folder": [r[8] for r in rows],
        "alias_table_sha256": [EXAMPLE_ALIAS_SHA256 if r[2] == "old" else None for r in rows],
        "score": np.array([r[9] for r in rows], dtype=np.float64),
    })
