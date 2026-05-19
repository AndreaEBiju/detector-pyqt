"""Preprocessing worker — runs `detector.preprocessing.batch.execute_batch`
on a Qt worker thread so the multi-minute pipeline (TDT read +
notch filter + .mat write × N folders) doesn't freeze the UI.

Design
------
`PreprocessWorker` is a `QObject` (lives in a `QThread` via
`moveToThread`) that calls `execute_batch()` and re-emits the
progress callback as Qt signals — one signal per item-completed,
plus a single `finished` at the end with the full result list.

Cooperative cancellation: `request_cancel()` sets a flag the
progress callback checks. Because `batch.execute_batch` doesn't
itself support mid-stream cancel, the cancel takes effect at the
boundary between items (after the current TDT folder finishes its
notch + .mat write). For batches of 5–10 folders × 1–2 min each
that's acceptable — the worst case is finishing the in-flight item
before stopping.

Signals
-------
- file_started(idx, total, source_name)            # 0-based idx
- file_completed(idx, total, result_dict)          # 0-based idx
- progress(done, total, fraction)                  # cumulative
- finished(results: list[dict])                    # all done, success
- error(message: str)                              # batch-level abort
- cancelled(partial_results: list[dict])           # cooperative stop
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot

# Self-contained sys.path setup — same pattern as inference_worker.py.
# detector-pyqt must come before detector-core or our `ui/` package gets
# shadowed by detector-core's Streamlit `ui/`.
_repo_root = Path(__file__).resolve().parent.parent.parent
_detector_core = _repo_root / "detector-core"
if _detector_core.exists() and str(_detector_core) not in sys.path:
    sys.path.insert(0, str(_detector_core))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from detector.preprocessing.batch import (                       # noqa: E402
    BatchPlanItem, execute_batch,
)


class PreprocessWorker(QObject):
    """Background worker for the preprocess batch.

    Usage (typical wiring from PreprocessWindow):

        thread = QThread()
        worker = PreprocessWorker(plan)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.file_started.connect(on_file_started)
        worker.file_completed.connect(on_file_completed)
        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        worker.cancelled.connect(on_cancelled)
        # Auto-cleanup
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.start()
    """

    # Emitted when about to process item idx (before TDT load).
    # We don't actually have a pre-callback in execute_batch, so we
    # synthesize this from the previous item's completion + the next
    # item's index. The very first one fires from a deferred queue in
    # run().
    file_started = Signal(int, int, str)
    file_completed = Signal(int, int, dict)
    progress = Signal(int, int, float)
    finished = Signal(list)
    error = Signal(str)
    cancelled = Signal(list)

    def __init__(
        self,
        plan: list[BatchPlanItem],
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._plan = list(plan)
        self._cancel_requested = False
        self._partial: list[dict] = []

    @Slot()
    def request_cancel(self) -> None:
        """Ask the worker to stop after the current item finishes.

        Thread-safe via Qt's `QueuedConnection` (this slot is called
        from the GUI thread; the worker reads `_cancel_requested`
        from its own thread). The Python GIL guarantees the bool
        read is atomic; no extra mutex needed.
        """
        self._cancel_requested = True

    @Slot()
    def run(self) -> None:
        """Worker-thread entry point. Catches all exceptions so a
        bug inside batch processing surfaces as `error` rather than
        crashing the UI thread.

        We don't call `execute_batch` directly — we re-implement its
        loop here so we can:
          (a) emit `file_started` before each item
          (b) check the cancel flag between items
        Per-item processing still delegates to `process_batch_item`.
        """
        try:
            from detector.preprocessing.batch import process_batch_item
            total = len(self._plan)
            for i, item in enumerate(self._plan):
                if self._cancel_requested:
                    self.cancelled.emit(self._partial)
                    return
                # Announce start so the UI can update its "current
                # file" label before the (slow) TDT read.
                try:
                    src_name = Path(item.tdt_folder).name
                except Exception:
                    src_name = str(item.tdt_folder)
                self.file_started.emit(i, total, src_name)

                # Process this item synchronously on the worker thread.
                res = process_batch_item(item)
                self._partial.append(res)

                self.file_completed.emit(i, total, res)
                frac = (i + 1) / max(1, total)
                self.progress.emit(i + 1, total, float(frac))

            # All done with no cancel — emit final result.
            self.finished.emit(self._partial)
        except Exception as exc:                                # pragma: no cover — defensive
            self.error.emit(f"{type(exc).__name__}: {exc}")
