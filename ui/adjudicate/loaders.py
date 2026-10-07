"""Reading the recording a queue row points at.

**New cohort** goes through task 03's loader via ``ui.audit.bridge.load_new_cohort``, with
the source path taken from the session's ``meta.json`` (store geometry, declared units,
file chanlabels) - the same route the blind audit uses, so both screens show the same
samples.

**Old cohort** (five channels: R, L hardware tripoles, then ANT1-3) is read from
``<rid>_notched.mat`` variable ``y``: the signal the labels were drawn on, unblanked.
Adapted from the Night 1 build's ``old_cohort.py`` (``load_old``; scratchpad, 2026-10-07),
keeping only the signal half - adjudication needs no label file. Units are DECLARED
(invariant 14): :func:`load_old_signal` takes ``units`` keyword-only with no default, and
the one call site passes :data:`OLD_COHORT_UNITS`, the cohort's single declaration
(volts). It lives here rather than in a ``protocol.yaml`` because the old cohort has no
``protocol.yaml`` in our store; if one is added, the declaration moves there. Conversion
is ``gems_blanking_v2.io.channel_map.scale_to_uv``, and the store's plausibility check
refuses a declaration the data cannot satisfy, rather than inferring anything.

The Survivals root (where old-cohort folders live) is a LOCAL path and is never
assumed: it comes from an explicit argument (``--survivals-root``), then the
``GEMS_SURVIVALS_ROOT`` environment variable, then a ``survivals_root = "..."`` line in
the per-user gems config file (``gems_blanking_v2.io.store.config_path()``, a
``platformdirs`` location). With none of these, old-cohort rows refuse to load. The
queue stores only paths relative to it.

**Beat trains** (for the spectrum panel's k×HR lines; :func:`old_beats`). Old cohort:
Andrea's ``HR_BR_HRVAnalysis`` output beside the signal, ``<base>*_HRBR.mat`` in the
row's folder (``<base>`` is the recording id without ``_notched``; matched
case-insensitively), whose ``heartlocs`` are 1-BASED sample indices into the analysed
signal. A file is used only if that signal is the whole recording - its ``t`` has
exactly the recording's sample count - because an epoch file (``..._recovery_HRBR.mat``)
indexes from the epoch start, which the file does not record. The exact name
``<base>_HRBR.mat`` is tried first, then the others in sorted order. New cohort: none -
the count-gated trains (ruling 2026-10-08 (b) 5) are not in the store, and nothing
here estimates a heart rate from the signal.
"""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

import h5py
import numpy as np
from gems_blanking_v2.io.chanlabels import assert_plausible_units
from gems_blanking_v2.io.channel_map import Units, meta_path, scale_to_uv
from gems_blanking_v2.io.store import GemsStore, config_path
from gems_blanking_v2.types import ChannelInfo, Recording
from scipy.io import loadmat, whosmat

from ui.audit import bridge

__all__ = [
    "OLD_COHORT_CHANNELS",
    "OLD_COHORT_UNITS",
    "BeatSource",
    "load_new_row",
    "load_old_signal",
    "old_beats",
    "old_signal_path",
    "resolve_survivals_root",
]

OLD_COHORT_UNITS: Final[Units] = "V"
"""Old-cohort ``_notched.mat`` samples are volts: the cohort's one declaration (there is
no old-cohort ``protocol.yaml`` in our store)."""

OLD_COHORT_CHANNELS: Final[tuple[str, ...]] = ("RVN", "LVN", "ANT1", "ANT2", "ANT3")
"""Column order of the old cohort's ``y`` (adapter check 2026-09-28): R, L, ANT1-3."""

SURVIVALS_ENV: Final = "GEMS_SURVIVALS_ROOT"
SURVIVALS_CONFIG_KEY: Final = "survivals_root"


def resolve_survivals_root(explicit: Path | None = None) -> Path | None:
    """The local Survivals root: argument, then environment, then per-user config.

    ``None`` when none of them names one. Nothing is guessed.
    """
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get(SURVIVALS_ENV)
    if env:
        return Path(env)
    cfg = config_path()
    if cfg.is_file():
        try:
            doc = tomllib.loads(cfg.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            msg = f"cannot parse {cfg}: {exc}"
            raise ValueError(msg) from exc
        value = doc.get(SURVIVALS_CONFIG_KEY)
        if isinstance(value, str) and value:
            return Path(value)
    return None


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
    if y.ndim != 2:
        msg = f"{path.name}: y has shape {y.shape}, expected (n, {n_ch})"
        raise ValueError(msg)
    if y.shape[1] != n_ch:
        msg = f"{path.name}: expected {n_ch} channels, got {y.shape}"
        raise ValueError(msg)
    return y, fs


def load_old_signal(path: Path, rid: str, animal: str, *, units: Units) -> Recording:
    """Load an old-cohort ``_notched.mat`` as a :class:`Recording` in microvolts.

    ``units`` is the cohort's declaration (:data:`OLD_COHORT_UNITS`), passed by the
    caller: required, never defaulted, never inferred.
    """
    y, fs = _read_y(Path(path))
    scale = scale_to_uv(units)
    assert_plausible_units(y, units, scale, where=Path(path).name)
    return Recording(fs=fs, data=y * scale, channels=_channels(), animal=animal,
                     session=rid, path=Path(path))


NEW_COHORT_NO_BEATS: Final = ("no stored beat train for the new cohort (the count-gated "
                              "trains are not in the store)")


@dataclass(frozen=True)
class BeatSource:
    """A recording's beat times (seconds on its timeline), or why there are none."""

    beats_s: np.ndarray | None
    source: str | None
    """The file the beats came from (its name only)."""
    reason: str | None


def _hrbr_shape(path: Path) -> tuple[np.ndarray, int]:
    """``(heartlocs, len(t))`` from an ``_HRBR.mat``: v7.3 via h5py, v5 via scipy."""
    try:
        with h5py.File(path, "r") as f:
            return (np.asarray(f["heartlocs"], dtype=np.float64).ravel(),
                    int(np.prod(f["t"].shape)))
    except OSError:  # not HDF5: a v5 MAT-file
        sizes = {name: shape for name, shape, _cls in whosmat(path)}
        if "t" not in sizes:
            msg = f"{path.name} has no t"
            raise KeyError(msg) from None
        m = loadmat(path, variable_names=["heartlocs"])
        return np.asarray(m["heartlocs"], dtype=np.float64).ravel(), int(np.prod(sizes["t"]))


def old_beats(survivals_root: Path, folder: str, rid: str, *, n_samples: int,
              fs: float) -> BeatSource:
    """The old-cohort beat train beside ``rid``'s signal (module docstring), or why not.

    Beat ``k`` (1-based ``heartlocs``) is at ``(k - 1) / fs`` seconds (invariant 15).
    """
    folder_path = old_signal_path(survivals_root, folder, rid).parent
    base = rid.removesuffix("_notched")
    exact = f"{base}_HRBR.mat".lower()
    try:
        names = sorted(p.name for p in folder_path.iterdir()
                       if p.name.lower().startswith(base.lower())
                       and p.name.lower().endswith("_hrbr.mat"))
    except OSError as exc:
        return BeatSource(None, None, f"cannot list {folder}: {exc}")
    if not names:
        return BeatSource(None, None, f"no {base}*_HRBR.mat beside the signal")
    names.sort(key=lambda n: n.lower() != exact)  # the exact name first, else sorted
    refused = []
    for name in names:
        try:
            heartlocs, n_t = _hrbr_shape(folder_path / name)
        except (OSError, KeyError, ValueError) as exc:
            refused.append(f"{name}: unreadable ({exc})")
            continue
        if n_t != n_samples:
            refused.append(f"{name}: covers {n_t} samples, the recording {n_samples} "
                           "(an epoch file)")
            continue
        return BeatSource((heartlocs - 1.0) / float(fs), name, None)
    return BeatSource(None, None, "no whole-recording _HRBR.mat: " + "; ".join(refused))
