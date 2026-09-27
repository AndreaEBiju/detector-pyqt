"""An in-memory stand-in for ``LazyRecording``, for data the audit already loaded.

``MultiChannelViewer`` reads through ``LazyRecording``, which opens HDF5 with
``h5py``. New-cohort ``_sig.mat`` files are MATLAB v7, not HDF5, and the audit
must read them through ``gems_blanking_v2``'s loader anyway (store geometry,
declared units, file chanlabels). This wraps that loader's array in the five
members the viewer actually uses - nothing more, so the viewer cannot be handed
anything the loader did not produce.
"""

from __future__ import annotations

import numpy as np


class ArrayRecording:
    """``(n_samples, n_channels)`` samples at ``fs``, served like ``LazyRecording``."""

    def __init__(self, data: np.ndarray, fs: float) -> None:
        arr = np.asarray(data)
        if arr.ndim != 2:  # (samples, channels)
            msg = f"expected (n_samples, n_channels), got shape {arr.shape}"
            raise ValueError(msg)
        if not fs > 0:
            msg = f"fs must be positive, got {fs}"
            raise ValueError(msg)
        self._y = arr.astype(np.float32, copy=False)
        self.fs = float(fs)
        self.n_samples, self.n_channels = self._y.shape
        self.duration_sec = self.n_samples / self.fs

    @property
    def has_overview(self) -> bool:
        """No pre-downsampled overview: the viewer falls back to ``get_range``."""
        return False

    def get_range(self, t_start: float, t_end: float) -> np.ndarray:
        """Samples in ``[t_start, t_end)`` as ``(n, n_ch)`` float32, clipped to bounds."""
        i0 = max(0, int(np.floor(t_start * self.fs)))
        i1 = min(self.n_samples, int(np.ceil(t_end * self.fs)))
        if i1 <= i0:
            return np.zeros((0, self.n_channels), dtype=np.float32)
        return self._y[i0:i1]
