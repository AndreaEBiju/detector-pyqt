"""Per-animal training worker -- wraps
`detector.retrain_per_animal.retrain_per_animal` on a Qt worker thread
so the main UI stays responsive across what can be hours of training.

Mirrors HeldoutEvalWorker's shape: a QObject moved to a QThread, with
progress/finished/error signals. The underlying orchestrator is
sequential (one animal at a time) and reports progress via a
`progress_callback(idx, total, animal, status)` hook, which we
forward as Qt signals.

The full result envelope (dict keyed by animal letter) is delivered
via the `finished` signal exactly as `retrain_per_animal` returns it,
including the 'skipped' entries -- the dialog renders them all.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional

from PySide6.QtCore import QObject, Signal, Slot

# Self-contained sys.path setup (same shape as inference_worker etc).
_repo_root = Path(__file__).resolve().parent.parent.parent
_detector_core = _repo_root / "detector-core"
if _detector_core.exists():
    if str(_detector_core) not in sys.path:
        sys.path.insert(0, str(_detector_core))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from detector import retrain_per_animal as RPA                  # noqa: E402


class PerAnimalTrainWorker(QObject):
    """One-shot per-animal training run. Signals:

      - progress(idx, total, animal_letter, status_str): fires twice
        per animal -- once before training starts ("starting") and once
        after ("ok" / "error" / "skipped").
      - finished(results: dict): the full {animal: result_dict}
        envelope from retrain_per_animal.
      - error(message: str): only fires when the orchestrator itself
        crashes (manifest load failure, etc.). Per-animal errors are
        captured INSIDE the results dict and don't trigger this signal.
    """

    progress = Signal(int, int, str, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        manifest_path: Path,
        *,
        artifacts_dir: Path,
        only_animals: Optional[Iterable[str]] = None,
        min_recordings_per_animal: int = 3,
        w_neg: float = 0.7,
        seed: int = 42,
        rebuild_dataset: bool = True,
        rebuild_phase2: bool = True,
        rebuild_loro: bool = True,
        skip_phase2_check: bool = True,
        skip_review: bool = False,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._manifest_path = Path(manifest_path)
        self._artifacts_dir = Path(artifacts_dir)
        self._only_animals = (
            list(only_animals) if only_animals is not None else None
        )
        self._min_recordings_per_animal = int(min_recordings_per_animal)
        self._w_neg = float(w_neg)
        self._seed = int(seed)
        self._rebuild_dataset = bool(rebuild_dataset)
        self._rebuild_phase2 = bool(rebuild_phase2)
        self._rebuild_loro = bool(rebuild_loro)
        self._skip_phase2_check = bool(skip_phase2_check)
        self._skip_review = bool(skip_review)

    @Slot()
    def run(self) -> None:
        try:
            def _cb(idx: int, total: int, animal: str, status: str) -> None:
                # Qt signals are thread-safe; the slot fires on the main
                # thread via the event loop.
                self.progress.emit(int(idx), int(total),
                                     str(animal), str(status))

            results = RPA.retrain_per_animal(
                self._manifest_path,
                artifacts_dir=self._artifacts_dir,
                only_animals=self._only_animals,
                min_recordings_per_animal=self._min_recordings_per_animal,
                progress_callback=_cb,
                w_neg=self._w_neg,
                seed=self._seed,
                rebuild_dataset=self._rebuild_dataset,
                rebuild_phase2=self._rebuild_phase2,
                rebuild_loro=self._rebuild_loro,
                skip_phase2_check=self._skip_phase2_check,
                skip_review=self._skip_review,
            )
            self.finished.emit(results)
        except Exception as e:                       # pragma: no cover -- UI error path
            self.error.emit(f"{type(e).__name__}: {e}")
