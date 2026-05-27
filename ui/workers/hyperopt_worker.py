"""Hyperopt worker -- Qt wrapper around
`detector.hyperopt.optimize_combined` / `optimize_per_animal`.

The backend calls block for 1-6 hours on a real corpus, so they run
in a background QThread and we surface progress to the UI via Qt
signals. The backend itself has no progress-callback hook, so we
poll the on-disk Optuna SQLite study files (which Optuna updates
after every trial commits) from a daemon thread inside `run()` --
that's the only way to get live per-trial updates without modifying
detector-core.

Layout the backend writes (and we read):

    <artifacts>/hyperopt_combined/hyperopt/study.db
                                 /best_params.json
                                 /trial_log.json
                                 /plots/*.png

    <artifacts>/hyperopt_per_animal/<animal>/hyperopt/study.db
                                            /best_params.json
                                            /trial_log.json
                                            /plots/*.png

For per-animal we also use the polling loop to detect when a new
animal's subdirectory appears (-> `animal_started`) and when its
trial count reaches n_trials (-> `animal_finished`).
"""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

from PySide6.QtCore import QObject, Signal, Slot

_repo_root = Path(__file__).resolve().parent.parent.parent
_detector_core = _repo_root / "detector-core"
if _detector_core.exists() and str(_detector_core) not in sys.path:
    sys.path.insert(0, str(_detector_core))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


# How often the polling loop checks the study.db files. 2 s is far
# below the per-trial cost (~1-3 min) so we never miss a trial; far
# above any sqlite lock contention so we don't fight the writer.
POLL_INTERVAL_S = 2.0


def _count_completed_trials(study_db: Path) -> int:
    """Return how many COMPLETE trials are in an Optuna SQLite study.
    Returns 0 if the file doesn't exist yet or if the schema isn't
    initialized -- both happen in the first second of a study."""
    if not study_db.exists():
        return 0
    try:
        # read-only open, short timeout: Optuna holds an exclusive
        # lock only while committing a trial, which is brief.
        con = sqlite3.connect(
            f"file:{study_db}?mode=ro", uri=True, timeout=1.0,
        )
        try:
            cur = con.execute(
                "SELECT COUNT(*) FROM trials WHERE state = 'COMPLETE'"
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0
        finally:
            con.close()
    except sqlite3.Error:
        return 0


def _last_complete_trial_objective(study_db: Path) -> Optional[float]:
    """Read the most-recently-completed trial's objective value, or
    None if the schema isn't there yet."""
    if not study_db.exists():
        return None
    try:
        con = sqlite3.connect(
            f"file:{study_db}?mode=ro", uri=True, timeout=1.0,
        )
        try:
            cur = con.execute(
                "SELECT tv.value FROM trial_values tv "
                "JOIN trials t ON t.trial_id = tv.trial_id "
                "WHERE t.state = 'COMPLETE' "
                "ORDER BY t.trial_id DESC LIMIT 1"
            )
            row = cur.fetchone()
            return float(row[0]) if row and row[0] is not None else None
        finally:
            con.close()
    except sqlite3.Error:
        return None


def _read_best_params(workdir: Path) -> Optional[dict]:
    """Read <workdir>/hyperopt/best_params.json if it exists."""
    bp = workdir / "hyperopt" / "best_params.json"
    if not bp.exists():
        return None
    try:
        return json.loads(bp.read_text())
    except Exception:
        return None


class HyperoptWorker(QObject):
    """One-shot hyperopt run for the combined model OR every eligible
    animal sequentially. Signals:

      - progress(scope, trial_num, total, objective):
          fires once per trial as the SQLite study commits. `scope` is
          "combined" for combined runs, or the animal letter for
          per-animal runs. `objective` is the just-completed trial's
          F-beta value (may be NaN if read failed).
      - animal_started(animal):
          per-animal scope only -- fires when a new animal's study.db
          first appears.
      - animal_finished(animal, best_params):
          per-animal scope only -- fires when an animal's study reaches
          n_trials and its best_params.json lands on disk.
      - finished(results: dict):
          full result envelope from the backend at the end of the run.
      - error(message: str):
          orchestrator-level crash (manifest load failure, optuna
          import failure, etc.). Per-animal errors are inside the
          results dict and don't trigger this signal.
    """

    progress = Signal(str, int, int, float)
    animal_started = Signal(str)
    animal_finished = Signal(str, dict)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        *,
        scope: str,
        manifest_path: Path,
        artifacts_dir: Path,
        n_trials: int,
        beta: float = 2.0,
        seed: int = 42,
        review_dir: Optional[Path] = None,
        only_animals: Optional[Iterable[str]] = None,
        min_recordings_per_animal: int = 4,
        holdout_rids_by_animal: Optional[dict] = None,
        w_neg_range: Optional[tuple] = None,
        fp_weight_range: Optional[tuple] = None,
        fn_weight_range: Optional[tuple] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        if scope not in ("combined", "per_animal"):
            raise ValueError(
                f"scope must be 'combined' or 'per_animal', got {scope!r}"
            )
        self._scope = scope
        self._manifest_path = Path(manifest_path)
        self._artifacts_dir = Path(artifacts_dir)
        self._n_trials = int(n_trials)
        self._beta = float(beta)
        self._seed = int(seed)
        self._review_dir = Path(review_dir) if review_dir else None
        self._only_animals = (
            [a.upper() for a in only_animals]
            if only_animals is not None else None
        )
        self._min_recordings_per_animal = int(min_recordings_per_animal)
        # Per-animal holdout override: maps animal letter -> list of
        # recording_ids to use as the validation holdout. None / empty
        # falls through to the orchestrator's default (alphabetically-
        # last recording for that animal).
        self._holdout_rids_by_animal: dict[str, list[str]] = {
            k.upper(): list(v)
            for k, v in (holdout_rids_by_animal or {}).items()
            if v
        }
        # Parameter-search ranges. None falls through to optimize_scope's
        # defaults. Each tuple is (low, high) -- low must be < high
        # (the UI validates this; the orchestrator does too).
        self._w_neg_range = (tuple(w_neg_range)
                             if w_neg_range else None)
        self._fp_weight_range = (tuple(fp_weight_range)
                                 if fp_weight_range else None)
        self._fn_weight_range = (tuple(fn_weight_range)
                                 if fn_weight_range else None)

    # ------------------------------------------------------------------
    # Polling helpers
    # ------------------------------------------------------------------

    def _combined_workdir(self) -> Path:
        return self._artifacts_dir / "hyperopt_combined"

    def _per_animal_root(self) -> Path:
        return self._artifacts_dir / "hyperopt_per_animal"

    def _poll_combined(self, last_count: int) -> int:
        """Check the combined-study db, emit a progress signal for any
        newly-completed trials, return the updated count."""
        db = self._combined_workdir() / "hyperopt" / "study.db"
        n = _count_completed_trials(db)
        if n > last_count:
            obj = _last_complete_trial_objective(db)
            self.progress.emit(
                "combined", int(n), int(self._n_trials),
                float(obj) if obj is not None else float("nan"),
            )
        return max(last_count, n)

    def _poll_per_animal(
        self,
        seen_animals: set[str],
        finished_animals: set[str],
        per_animal_counts: dict[str, int],
    ) -> None:
        """Walk the per-animal artifacts tree once. Mutates the given
        state dicts in place: emits animal_started for new ones, emits
        progress for any newly completed trials, emits animal_finished
        for animals that reached n_trials and have best_params.json."""
        root = self._per_animal_root()
        if not root.exists():
            return
        # Each subdir name == animal letter.
        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue
            animal = sub.name
            db = sub / "hyperopt" / "study.db"
            if animal not in seen_animals and db.exists():
                seen_animals.add(animal)
                self.animal_started.emit(animal)
            n = _count_completed_trials(db)
            prev = per_animal_counts.get(animal, 0)
            if n > prev:
                obj = _last_complete_trial_objective(db)
                self.progress.emit(
                    animal, int(n), int(self._n_trials),
                    float(obj) if obj is not None else float("nan"),
                )
                per_animal_counts[animal] = n
            # An animal counts as "finished" once best_params.json is on
            # disk -- the backend writes it right after the optimize
            # loop, before moving to the next animal.
            if (animal not in finished_animals
                    and (sub / "hyperopt" / "best_params.json").exists()):
                bp = _read_best_params(sub) or {}
                finished_animals.add(animal)
                self.animal_finished.emit(animal, bp)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    @Slot()
    def run(self) -> None:
        try:
            from detector.hyperopt import (
                optimize_combined, optimize_per_animal,
            )
        except Exception as e:
            self.error.emit(f"import failure: {type(e).__name__}: {e}")
            return

        result_holder: dict = {}
        error_holder: dict = {}

        # Build the range kwargs once -- the orchestrator signatures
        # accept them on both combined and per_animal entry points,
        # but skip any that the UI didn't set (so the orchestrator
        # default fires). All-None means no override.
        range_kwargs: dict = {}
        if self._w_neg_range is not None:
            range_kwargs["w_neg_range"] = self._w_neg_range
        if self._fp_weight_range is not None:
            range_kwargs["fp_weight_range"] = self._fp_weight_range
        if self._fn_weight_range is not None:
            range_kwargs["fn_weight_range"] = self._fn_weight_range

        def _do_work():
            try:
                if self._scope == "combined":
                    result_holder["result"] = optimize_combined(
                        self._manifest_path,
                        artifacts_dir=self._artifacts_dir,
                        n_trials=self._n_trials,
                        review_dir=self._review_dir,
                        beta=self._beta,
                        seed=self._seed,
                        **range_kwargs,
                    )
                else:
                    result_holder["result"] = optimize_per_animal(
                        self._manifest_path,
                        artifacts_dir=self._artifacts_dir,
                        n_trials=self._n_trials,
                        review_dir=self._review_dir,
                        beta=self._beta,
                        seed=self._seed,
                        only_animals=self._only_animals,
                        min_recordings_per_animal=
                            self._min_recordings_per_animal,
                        holdout_rids_by_animal=
                            self._holdout_rids_by_animal,
                        **range_kwargs,
                    )
            except Exception as e:                  # pragma: no cover
                error_holder["error"] = e

        t = threading.Thread(target=_do_work, daemon=True)
        t.start()

        # Poll until the backend thread terminates. While polling we
        # emit progress / animal-level signals.
        combined_count = 0
        seen_animals: set[str] = set()
        finished_animals: set[str] = set()
        per_animal_counts: dict[str, int] = {}
        while t.is_alive():
            time.sleep(POLL_INTERVAL_S)
            try:
                if self._scope == "combined":
                    combined_count = self._poll_combined(combined_count)
                else:
                    self._poll_per_animal(
                        seen_animals, finished_animals,
                        per_animal_counts,
                    )
            except Exception:                       # pragma: no cover
                # Polling is best-effort; never let it kill the run.
                pass
        # One final poll so the table reflects the last trial even if
        # the work finished between ticks.
        try:
            if self._scope == "combined":
                self._poll_combined(combined_count)
            else:
                self._poll_per_animal(
                    seen_animals, finished_animals, per_animal_counts,
                )
        except Exception:
            pass

        t.join()
        if "error" in error_holder:
            e = error_holder["error"]
            self.error.emit(f"{type(e).__name__}: {e}")
            return
        self.finished.emit(result_holder.get("result") or {})


# ----------------------------------------------------------------------
# Filesystem-only helpers (for re-attaching to a finished CLI run)
# ----------------------------------------------------------------------

def scan_completed_studies(artifacts_dir: Path) -> dict:
    """Walk artifacts_dir/hyperopt_combined and hyperopt_per_animal/*
    and return a dict of any studies that have best_params.json on
    disk:

        {
            "combined": {best_params, best_objective, ...} | None,
            "per_animal": {animal: {best_params, ...}, ...},
        }

    Used by the Hyperopt tab to populate the results table from a
    previous CLI run when the user opens the tab without launching a
    fresh study."""
    artifacts_dir = Path(artifacts_dir)
    out = {"combined": None, "per_animal": {}}

    combined_bp = (
        artifacts_dir / "hyperopt_combined" / "hyperopt" / "best_params.json"
    )
    if combined_bp.exists():
        try:
            out["combined"] = json.loads(combined_bp.read_text())
        except Exception:
            pass

    pa_root = artifacts_dir / "hyperopt_per_animal"
    if pa_root.exists():
        for sub in sorted(pa_root.iterdir()):
            bp = sub / "hyperopt" / "best_params.json"
            if not bp.exists():
                continue
            try:
                out["per_animal"][sub.name] = json.loads(bp.read_text())
            except Exception:
                pass
    return out
