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
    the report dict.

    Progress signal fires once per recording: (idx_done, total,
    next_recording_id). The dialog displays this as "Evaluating
    recording 3/7: rec_XYZ".
    """

    # (idx_starting_now, total, recording_id)
    progress = Signal(int, int, str)
    # report dict from evaluate_heldout
    finished = Signal(dict)
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

            report = HE.evaluate_heldout(
                m, self._artifact,
                rec_ids=self._rec_ids,
                progress_callback=_cb,
            )
            self.finished.emit(report)
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")
