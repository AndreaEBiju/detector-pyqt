"""Bulk-inference worker for the queue's 'Run inference on all
pending' button.

Wraps `detector.bulk_inference.run_inference_and_save` into a Qt
worker that runs a multiprocessing.Pool over N recordings in
parallel. Worker count comes from the same machine-aware heuristic
the rest of the pipeline uses (psutil + cpu_count, capped at 12).

Used by main_window.MainWindow._on_bulk_inference (the new
parallel implementation, replacing the old sequential one that
opened each recording in the UI viewer one at a time).

Per-recording result dicts stream back via the `progress` signal so
the dialog can update a table as work completes. The `finished`
signal carries the full list of results at the end.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot

# Self-contained sys.path setup (matches the other workers).
_repo_root = Path(__file__).resolve().parent.parent.parent
_detector_core = _repo_root / "detector-core"
if _detector_core.exists() and str(_detector_core) not in sys.path:
    sys.path.insert(0, str(_detector_core))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


class BulkInferenceWorker(QObject):
    """Single-shot bulk-inference worker.

    Each item is processed in its own Pool worker process: load
    recording, run detect_bad, save versioned outputs. Workers may
    run in parallel; the same recording is NOT loaded by multiple
    workers concurrently (each Pool job is one recording).

    Memory budget: per worker peak ~3-5 GB (signal + filter buffers
    + model artifact). Default worker count via
    `_default_phase1_workers()` -- psutil-aware, conservative.
    """

    # (n_completed, n_total, result_dict_for_just_finished)
    progress = Signal(int, int, dict)
    # All result dicts at the end.
    finished = Signal(list)
    # Human-readable error message.
    error = Signal(str)

    def __init__(
        self,
        recording_paths: list[Path],
        model_artifact_path: Path,
        *,
        n_workers: Optional[int] = None,
        skip_existing: bool = True,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._recordings = [str(p) for p in recording_paths]
        self._model_path = str(model_artifact_path)
        self._n_workers = n_workers
        self._skip_existing = bool(skip_existing)
        self._cancelled = False

    def request_cancel(self) -> None:
        """Best-effort cancel. The current in-flight workers run to
        completion (their detect_bad calls can't be interrupted
        cleanly), but no further work is dispatched once this flag is
        set. Same UX as inference_worker."""
        self._cancelled = True

    @Slot()
    def run(self) -> None:
        try:
            from detector.bulk_inference import (
                run_inference_and_save_unpack,
            )
        except Exception as e:
            self.error.emit(f"import failure: {type(e).__name__}: {e}")
            return

        # Worker count: defer to the existing Phase-1 heuristic, then
        # clamp to len(jobs).
        n_workers = self._n_workers
        if n_workers is None:
            try:
                from detector.dataset import _default_phase1_workers
                n_workers = _default_phase1_workers()
            except Exception:
                import os
                n_workers = max(1, (os.cpu_count() or 4) - 1)
        n_workers = max(1, min(n_workers, len(self._recordings)))

        jobs = [
            (rec, self._model_path, self._skip_existing)
            for rec in self._recordings
        ]
        total = len(jobs)
        results: list[dict] = []

        try:
            if n_workers > 1 and total > 1:
                from multiprocessing import get_context
                ctx = get_context("spawn")
                with ctx.Pool(processes=n_workers) as pool:
                    it = pool.imap_unordered(
                        run_inference_and_save_unpack, jobs,
                    )
                    for i, res in enumerate(it):
                        if self._cancelled:
                            # Pool terminates when we leave the `with`.
                            break
                        results.append(res)
                        self.progress.emit(i + 1, total, res)
            else:
                # Serial fallback (single recording or worker cap of 1).
                from detector.bulk_inference import run_inference_and_save
                for i, (rec, art, skip) in enumerate(jobs):
                    if self._cancelled:
                        break
                    res = run_inference_and_save(
                        rec, art, skip_existing=skip,
                    )
                    results.append(res)
                    self.progress.emit(i + 1, total, res)
        except Exception as e:
            self.error.emit(
                f"bulk inference pool failed: {type(e).__name__}: {e}"
            )
            return

        self.finished.emit(results)
