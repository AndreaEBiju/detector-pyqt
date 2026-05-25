"""Blankmotion-migration worker.

Wraps `detector.migrate_blankmotion.migrate_files` in a Qt thread so
the UI stays responsive while the underlying multiprocessing.Pool
chews through possibly-many GB-sized recordings. Per-file results
stream back via the `progress` signal so the dialog's table can
update live.

Same pattern as the other Pool-driven workers in this folder
(`bulk_inference_worker.py`, `heldout_eval_worker.py`).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot

_repo_root = Path(__file__).resolve().parent.parent.parent
_detector_core = _repo_root / "detector-core"
if _detector_core.exists() and str(_detector_core) not in sys.path:
    sys.path.insert(0, str(_detector_core))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


class BlankmotionMigrationWorker(QObject):
    """One-shot worker over a list of `_blankmotion.mat` files.

    Signals:
      - progress(n_done, n_total, result_dict): fires once per file
        as workers finish (out-of-input-order). result_dict matches
        `migrate_one_blankmotion`'s return shape.
      - finished(summary_dict): fires once at the end with the full
        migrate_files summary.
      - error(msg): fires if the worker itself crashes outside the
        per-file try/except.
    """

    progress = Signal(int, int, dict)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        files: list[Path],
        *,
        force: bool = False,
        n_workers: Optional[int] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._files = [str(p) for p in files]
        self._force = bool(force)
        self._n_workers = n_workers
        self._cancelled = False

    def request_cancel(self) -> None:
        """Best-effort cancel. Pool workers already in-flight finish
        (they can't be interrupted cleanly mid-recording); no
        further files are dispatched."""
        self._cancelled = True

    @Slot()
    def run(self) -> None:
        try:
            from detector.migrate_blankmotion import migrate_files
        except Exception as e:
            self.error.emit(f"import failure: {type(e).__name__}: {e}")
            return

        def _cb(n_done, n_total, res):
            self.progress.emit(int(n_done), int(n_total), dict(res))

        try:
            summary = migrate_files(
                self._files,
                force=self._force,
                n_workers=self._n_workers,
                progress_callback=_cb,
            )
        except Exception as e:
            self.error.emit(
                f"migrate_files crashed: {type(e).__name__}: {e}"
            )
            return

        self.finished.emit(dict(summary))
