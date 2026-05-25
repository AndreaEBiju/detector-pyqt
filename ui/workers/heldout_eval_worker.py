"""Held-out evaluation worker — runs `detector.heldout_eval` on a Qt
worker thread so the main UI stays responsive while the model is
chewing through possibly-many-minutes of recordings.

Mirrors the InferenceWorker shape: live in a QThread via moveToThread,
emit progress/finished/error signals back to the main thread.

Each held-out recording takes roughly the same time as one inference
run (~30s for a 1-channel ~30s file, longer for larger), so a manifest
with 5 held-out recordings can comfortably need several minutes.
Progress is emitted once per recording (start), not in finer-grained
sub-steps -- that matches what the underlying CLI prints and is enough
for the user to see "we're 3 of 7 done, keep waiting".
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot

# Self-contained sys.path setup (same shape as inference_worker).
_repo_root = Path(__file__).resolve().parent.parent.parent
_detector_core = _repo_root / "detector-core"
if _detector_core.exists():
    if str(_detector_core) not in sys.path:
        sys.path.insert(0, str(_detector_core))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from detector import heldout_eval as HE                          # noqa: E402
from detector.manifest import Manifest                            # noqa: E402
from detector.model_artifact import ModelArtifact                # noqa: E402


class HeldoutEvalWorker(QObject):
    """One-shot worker: load manifest, run evaluate_heldout(), emit
    the report dict + per-recording interval cache.

    Progress signal fires once per recording: (idx_done, total,
    next_recording_id). The dialog displays this as "Evaluating
    recording 3/7: rec_XYZ".

    The cache is `{recording_id: {"model": ndarray, "human": ndarray,
    "fs": float, "n_samples": int}}` -- the Phase-3 overlay window
    reads from it to redraw model vs. human bands without re-running
    `detect_bad` (which is the slow part of the eval). Empty when no
    recording was successfully evaluated.
    """

    # (idx_starting_now, total, recording_id)
    progress = Signal(int, int, str)
    # (report dict, interval_cache dict) from evaluate_heldout
    finished = Signal(dict, dict)
    # human-readable error message
    error = Signal(str)

    def __init__(
        self,
        manifest_path: Path,
        artifact: ModelArtifact,
        *,
        rec_ids: Optional[list[str]] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._manifest_path = Path(manifest_path)
        self._artifact = artifact
        self._rec_ids = list(rec_ids) if rec_ids else None

    @Slot()
    def run(self) -> None:
        try:
            m = Manifest.load(self._manifest_path)

            def _cb(i: int, total: int, rid: str) -> None:
                self.progress.emit(int(i), int(total), str(rid))

            interval_cache: dict = {}
            report = HE.evaluate_heldout(
                m, self._artifact,
                rec_ids=self._rec_ids,
                progress_callback=_cb,
                interval_cache=interval_cache,
            )
            self.finished.emit(report, interval_cache)
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")


class HeldoutEvalMultiModelWorker(QObject):
    """Multi-model variant of HeldoutEvalWorker. Runs
    `detector.heldout_eval.evaluate_heldout_multi_model` in a
    background QThread, emits a multi-model report dict + the
    multi-model interval cache.

    The cache shape (for the overlay viewer's model dropdown) is:
        cache[recording_id] = {
            "human":     (k, 2) int64,
            "fs":        float,
            "n_samples": int,
            "models":    {version: (k, 2) int64, ...},
        }

    Inputs:
      - manifest_path: same as single-model worker.
      - artifact_paths: list of Path to model_v*/ directories.
        These are PATHS not loaded artifacts, because the underlying
        multi-model function uses multiprocessing.Pool with spawn
        semantics -- workers reload each artifact themselves to
        avoid pickling LightGBM boosters across process boundaries.
      - n_workers: optional override; default from
        `detector.dataset._default_phase1_workers` (machine-aware).
    """

    # (idx_completed, total, recording_id_just_done)
    progress = Signal(int, int, str)
    # (multi-model report dict, interval_cache dict)
    finished = Signal(dict, dict)
    # human-readable error message
    error = Signal(str)

    def __init__(
        self,
        manifest_path: Path,
        artifact_paths: list[Path],
        *,
        rec_ids: Optional[list[str]] = None,
        n_workers: Optional[int] = None,
        parallelize: str = "recordings",
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._manifest_path = Path(manifest_path)
        self._artifact_paths = [Path(p) for p in artifact_paths]
        self._rec_ids = list(rec_ids) if rec_ids else None
        self._n_workers = n_workers
        # "recordings" | "flat" -- forwarded straight to the backend.
        # The backend may still fall back from "flat" to "recordings"
        # if the RAM cap drops below 2 workers; the resulting report
        # carries `_parallelize_used` so the dialog can surface it.
        self._parallelize = parallelize

    @Slot()
    def run(self) -> None:
        try:
            m = Manifest.load(self._manifest_path)

            def _cb(i: int, total: int, rid: str) -> None:
                self.progress.emit(int(i), int(total), str(rid))

            interval_cache: dict = {}
            report = HE.evaluate_heldout_multi_model(
                m, self._artifact_paths,
                rec_ids=self._rec_ids,
                progress_callback=_cb,
                interval_cache=interval_cache,
                n_workers=self._n_workers,
                parallelize=self._parallelize,
            )
            self.finished.emit(report, interval_cache)
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")
