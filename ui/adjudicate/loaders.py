"""Reading the recording a queue row points at.

**New cohort** goes through task 03's loader via ``ui.audit.bridge.load_new_cohort``, with
the source path taken from the session's ``meta.json`` (store geometry, declared units,
file chanlabels) - the same route the blind audit uses, so both screens show the same
samples.

**Old cohort** (five channels: R, L hardware tripoles, then ANT1-3) is read from
``<rid>_notched.mat`` variable ``y``: the signal the labels were drawn on, unblanked.
Adapted from the Night 1 build's ``old_cohort.py`` (``load_old``; scratchpad, 2026-10-07),
keeping only the signal half - adjudication needs no label file. Units are DECLARED
volts for the whole cohort (:data:`OLD_COHORT_UNITS`, invariant 14) and converted with
``gems_blanking_v2.io.channel_map.scale_to_uv``; the store's plausibility check then
refuses a declaration the data cannot satisfy, rather than inferring anything.

The Survivals root (where old-cohort folders live) is a LOCAL path, resolved here from
an explicit argument, then ``GEMS_SURVIVALS_ROOT``, then :data:`DEFAULT_SURVIVALS_ROOT`
if it exists; the queue stores only paths relative to it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Final

import h5py
import numpy as np
from gems_blanking_v2.io.chanlabels import assert_plausible_units
from gems_blanking_v2.io.channel_map import meta_path, scale_to_uv
from gems_blanking_v2.io.store import GemsStore
from gems_blanking_v2.types import ChannelInfo, Recording
from scipy.io import loadmat

from ui.audit import bridge

__all__ = [
    "DEFAULT_SURVIVALS_ROOT",
    "OLD_COHORT_CHANNELS",
    "OLD_COHORT_UNITS",
    "load_new_row",
    "load_old_signal",
    "old_signal_path",
    "resolve_survivals_root",
]

OLD_COHORT_UNITS: Final = "V"
"""Old-cohort ``_notched.mat`` samples are volts (cohort-level, declared once)."""

OLD_COHORT_CHANNELS: Final[tuple[str, ...]] = ("RVN", "LVN", "ANT1", "ANT2", "ANT3")
"""Column order of the old cohort's ``y`` (adapter check 2026-09-28): R, L, ANT1-3."""

DEFAULT_SURVIVALS_ROOT: Final = Path(
    "G:/Shared drives/BIONICs Lab Workspace/Project Folders/GEMS/Survivals")
"""This build machine's mount of the Survivals folder. A local default, never stored."""

SURVIVALS_ENV: Final = "GEMS_SURVIVALS_ROOT"


def resolve_survivals_root(explicit: Path | None = None) -> Path | None:
    """The local Survivals root, or ``None`` when none is configured or present."""
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get(SURVIVALS_ENV)
    if env:
        return Path(env)
    return DEFAULT_SURVIVALS_ROOT if DEFAULT_SURVIVALS_ROOT.is_dir() else None


def load_new_row(store: GemsStore, animal: str, recording: str) -> Recording:
    """Load one new-cohort recording through task 03's loader. Refuses an excluded one."""
    meta_file = meta_path(store, animal, recording)
    if not meta_file.is_file():
        msg = f"no meta.json for {animal}/{recording} at {store.relpath(meta_file)}"
        raise FileNotFoundError(msg)
    meta: dict[str, Any] = json.loads(meta_file.read_text(encoding="utf-8"))
    if meta.get("excluded"):
        msg = f"{recording} is excluded ({meta['excluded']}); it should not be in the queue"
        raise RuntimeError(msg)
    if "source_path" not in meta:
        msg = f"{store.relpath(meta_file)} has no source_path"
        raise ValueError(msg)
    loaded = bridge.load_new_cohort(store.abspath(meta["source_path"]), animal, store=store)
    if loaded.provenance.get("excluded"):
        msg = f"{recording} is excluded: {loaded.provenance['excluded']}"
        raise RuntimeError(msg)
    rec: Recording = loaded.recording
    return rec


def old_signal_path(survivals_root: Path, folder: str, rid: str) -> Path:
    """``<root>/<folder>/<rid>_notched.mat``, or ``<rid>.mat`` when rid ends in _notched."""
    pure = PurePosixPath(folder)
    if pure.is_absolute() or ".." in pure.parts:
        msg = f"old-cohort folder must be relative to the Survivals root, got {folder!r}"
        raise ValueError(msg)
    name = f"{rid}.mat" if rid.endswith("_notched") else f"{rid}_notched.mat"
    return Path(survivals_root).joinpath(*pure.parts) / name


def _channels() -> list[ChannelInfo]:
    return [
        ChannelInfo(0, "RVN", "nerve", "R", None, None, "hw_tripole"),
        ChannelInfo(1, "LVN", "nerve", "L", None, None, "hw_tripole"),
        ChannelInfo(2, "ANT1", "stomach", None, None, None, "hw_tripole"),
        ChannelInfo(3, "ANT2", "stomach", None, None, None, "hw_tripole"),
        ChannelInfo(4, "ANT3", "stomach", None, None, None, "hw_tripole"),
    ]


def _read_y(path: Path) -> tuple[np.ndarray, float]:
    """``y`` as (n, 5) in FILE units, and ``fs``. v5 via scipy, v7.3 via h5py."""
    try:
        m = loadmat(path, variable_names=["y", "fs"])
        y = np.asarray(m["y"], dtype=np.float64)
        fs = float(np.asarray(m["fs"]).ravel()[0])
    except (NotImplementedError, ValueError):
        with h5py.File(path, "r") as f:
            # v7.3 stores MATLAB's column-major (n, 5) as (5, n): transpose, never reshape.
            y = np.asarray(f["y"], dtype=np.float64).T
            fs = float(np.asarray(f["fs"]).ravel()[0])
    n_ch = len(OLD_COHORT_CHANNELS)
    if y.ndim != 2:  # noqa: PLR2004
        msg = f"{path.name}: y has shape {y.shape}, expected (n, {n_ch})"
        raise ValueError(msg)
    if y.shape[1] != n_ch and y.shape[0] == n_ch:
        y = y.T
    if y.shape[1] != n_ch:
        msg = f"{path.name}: expected {n_ch} channels, got {y.shape}"
        raise ValueError(msg)
    return y, fs


def load_old_signal(path: Path, rid: str, animal: str, *,
                    units: str = OLD_COHORT_UNITS) -> Recording:
    """Load an old-cohort ``_notched.mat`` as a :class:`Recording` in microvolts."""
    y, fs = _read_y(Path(path))
    scale = scale_to_uv(units)
    assert_plausible_units(y, units, scale, where=Path(path).name)
    return Recording(fs=fs, data=y * scale, channels=_channels(), animal=animal,
                     session=rid, path=Path(path))
