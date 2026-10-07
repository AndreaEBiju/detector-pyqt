"""Shared synthetic generators for the adjudication-screen tests.

Every generator takes an explicit seed. They are served as factory FIXTURES (not
imported by name): the repository root has its own ``conftest.py``, and a second
module importable as ``conftest`` would resolve to whichever pytest loaded last.
Nothing here touches a real store, Drive path or per-user config.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from gems_blanking_v2.types import ChannelInfo, Recording

NEW_NAMES = ("RVN1", "RVN2", "RVN3", "LVN1", "LVN2", "LVN3", "ANT1", "ANT2", "ANT3")
OLD_NAMES = ("RVN", "LVN", "ANT1", "ANT2", "ANT3")


def _new_channels() -> list[ChannelInfo]:
    chans = []
    for i, name in enumerate(NEW_NAMES):
        if name.startswith(("RVN", "LVN")):
            chans.append(ChannelInfo(i, name, "nerve", name[0], int(name[-1]), None,
                                     "independent"))
        else:
            chans.append(ChannelInfo(i, name, "stomach", None, None, None, "independent"))
    return chans


def _old_channels() -> list[ChannelInfo]:
    return [ChannelInfo(0, "RVN", "nerve", "R", None, None, "hw_tripole"),
            ChannelInfo(1, "LVN", "nerve", "L", None, None, "hw_tripole"),
            *(ChannelInfo(i, n, "stomach", None, None, None, "hw_tripole")
              for i, n in ((2, "ANT1"), (3, "ANT2"), (4, "ANT3")))]


@pytest.fixture
def make_recording() -> Callable[..., Recording]:
    """``make(seed=, cohort="new"|"old", fs=2000, dur_s=60, noise_uv=10)``: white noise, µV."""

    def make(*, seed: int, cohort: str = "new", fs: float = 2000.0, dur_s: float = 60.0,
             noise_uv: float = 10.0, session: str = "synth") -> Recording:
        chans = _new_channels() if cohort == "new" else _old_channels()
        rng = np.random.default_rng(seed)
        data = rng.standard_normal((round(dur_s * fs), len(chans))) * noise_uv
        return Recording(fs=fs, data=data, channels=chans, animal="A", session=session,
                         path=Path(session))

    return make


@pytest.fixture
def add_burst() -> Callable[..., None]:
    """``add(rec, t_s, dur_s, amp_uv, channels=None, freq_hz=37)``: a sine burst, in place."""

    def add(rec: Recording, t_s: float, dur_s: float, amp_uv: float,
            channels: Sequence[int] | None = None, *, freq_hz: float = 37.0) -> None:
        i0 = round(t_s * rec.fs)
        i1 = round((t_s + dur_s) * rec.fs)
        tt = np.arange(i1 - i0) / rec.fs
        cols = range(rec.data.shape[1]) if channels is None else channels
        for ch in cols:
            rec.data[i0:i1, ch] += amp_uv * np.sin(2 * np.pi * freq_hz * tt)

    return add


@pytest.fixture
def add_tone() -> Callable[..., None]:
    """``add(rec, freq_hz, amp_uv, channels=None, seed=)``: a stationary sine, in place."""

    def add(rec: Recording, freq_hz: float, amp_uv: float,
            channels: Sequence[int] | None = None, *, seed: int = 0) -> None:
        phase = np.random.default_rng(seed).uniform(0, 2 * np.pi)
        tt = np.arange(rec.data.shape[0]) / rec.fs
        cols = range(rec.data.shape[1]) if channels is None else channels
        for ch in cols:
            rec.data[:, ch] += amp_uv * np.sin(2 * np.pi * freq_hz * tt + phase)

    return add


def _row(recording: str, start_s: float, stop_s: float, *, cohort: str = "new",
         region: tuple[float, float] = (0.0, 60.0), **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "recording": recording, "animal": "A" if cohort == "new" else "F", "cohort": cohort,
        "start_s": float(start_s), "stop_s": float(stop_s),
        "region_start_s": float(region[0]), "region_stop_s": float(region[1]),
        "draw": "random", "label_set": "train",
        "folder": None if cohort == "new" else f"synth_folder/{recording}",
        "alias_table_sha256": None if cohort == "new" else "0" * 63 + "1",
    }
    row.update(extra)
    return row


@pytest.fixture
def queue_row() -> Callable[..., dict[str, Any]]:
    """``row(recording, start_s, stop_s, cohort=, region=, **optional columns)``."""
    return _row


_APP: list[Any] = []


@pytest.fixture
def make_hum_window(tmp_path: Path) -> Iterator[Callable[..., Any]]:
    """``make(rows, recordings, beats_fn=None, async_panels=False)``: an offscreen window.

    ``recordings`` maps a recording id to its :class:`Recording`; the queue is written
    to ``tmp_path`` and judgements go to a store under ``tmp_path``.
    """
    from gems_blanking_v2.io.store import GemsStore
    from PySide6.QtWidgets import QApplication

    from ui.adjudicate.queue import load_queue
    from ui.adjudicate.session import AdjudicationSession

    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    made: list[Any] = []

    def make(rows: Sequence[dict[str, Any]], recordings: dict[str, Recording],
             beats_fn: Callable[..., Any] | None = None, *, async_panels: bool = False) -> Any:
        from ui.windows.adjudication_window import AdjudicationWindow

        root = tmp_path / f"w{len(made)}"
        store = GemsStore.initialise(root / "gems")
        path = root / "queue.parquet"
        pd.DataFrame(list(rows)).to_parquet(path, index=False)
        session = AdjudicationSession(load_queue(path), store, "tester",
                                      journal_dir=root / "journal", queue_file=path.name,
                                      queue_sha256="0" * 64, app_sha=None)
        w = AdjudicationWindow(session, load_fn=lambda r: recordings[str(r["recording"])],
                               async_traces=False, async_panels=async_panels,
                               beats_fn=beats_fn)
        made.append(w)
        return w

    yield make
    for w in made:
        w.close()
