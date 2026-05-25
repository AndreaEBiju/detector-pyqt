"""Baseline worker — runs `detector.baseline.compute_baseline` on a Qt
worker thread so the main UI stays responsive after Save while the
~5-15s baseline scan (medians / MADs / band-power stats over every
clean chunk) runs in the background.

Mirrors the InferenceWorker / HeldoutEvalWorker shape: a `QObject`
that lives in a `QThread` via `moveToThread` and emits `finished`
or `error` back to the main thread. `compute_baseline` doesn't take
a progress callback, so we don't expose a `progress` signal — the
status-bar message is the only UI feedback during the run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot

# Self-contained sys.path setup (same shape as inference_worker /
# heldout_eval_worker — detector-pyqt's `ui/` ahead of the Streamlit
# submodule `ui/`, detector-core's `detector/` on the path).
_repo_root = Path(__file__).resolve().parent.parent.parent
_detector_core = _repo_root / "detector-core"
if _detector_core.exists():
    if str(_detector_core) not in sys.path:
        sys.path.insert(0, str(_detector_core))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from detector.baseline import compute_baseline                  # noqa: E402


class BaselineWorker(QObject):
    """One-shot worker: call `compute_baseline` and emit the result
    dict (or an error string) back to the main thread.

    `finished` carries the baseline-path string so the slot can build
    a "baseline.h5 written" status message without re-deriving the
    path.
    """

    # baseline_path written to disk (str — Qt signals dislike Path)
    finished = Signal(str)
    # human-readable error message
    error = Signal(str)

    def __init__(
        self,
        recording_id: str,
        clean_path: Path,
        baseline_path: Path,
        fs: float,
        *,
        force: bool = True,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._recording_id = str(recording_id)
        self._clean_path = Path(clean_path)
        self._baseline_path = Path(baseline_path)
        self._fs = float(fs)
        self._force = bool(force)

    @Slot()
    def run(self) -> None:
        try:
            compute_baseline(
                self._recording_id,
                paths={
                    "clean_path": str(self._clean_path),
                    "baseline_path": str(self._baseline_path),
                    "fs": self._fs,
                },
                force=self._force,
            )
            self.finished.emit(str(self._baseline_path))
        except Exception as exc:                               # pragma: no cover — UI error path
            self.error.emit(f"{type(exc).__name__}: {exc}")
