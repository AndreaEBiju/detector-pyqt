"""Tests for the Hyperopt worker and the filesystem-only helpers.

The substantive coverage of the hyperopt backend lives in
detector-core's own tests. Here we exercise the Qt-side wrapper:

  - `scan_completed_studies` walks the artifacts tree and pulls out
    any studies that have written `best_params.json`. Robust to a
    partially-populated tree (e.g. some animals finished, others
    haven't yet).
  - `_count_completed_trials` and `_last_complete_trial_objective`
    correctly read an Optuna-shaped SQLite schema, and degrade
    gracefully when the file is missing or malformed.
  - `HyperoptWorker.run` polls the on-disk state during a stubbed
    backend call and emits progress / animal-level signals at the
    right moments.

The end-to-end signal test mocks `detector.hyperopt.optimize_*` with
a function that writes the same on-disk shape Optuna would. That
lets us run the full polling loop without LightGBM, Optuna, or the
~hour-per-trial cost.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

_DETECTOR_CORE = ROOT / "detector-core"
if _DETECTOR_CORE.exists() and str(_DETECTOR_CORE) not in sys.path:
    sys.path.insert(0, str(_DETECTOR_CORE))

from ui.workers.hyperopt_worker import (
    HyperoptWorker,
    _count_completed_trials,
    _last_complete_trial_objective,
    _read_best_params,
    scan_completed_studies,
)


# ----------------------------------------------------------------------
# SQLite-reader helpers
# ----------------------------------------------------------------------

def _make_optuna_like_db(
    path: Path,
    *,
    completed_objectives: list[float],
    running_trials: int = 0,
) -> None:
    """Minimal Optuna-shaped schema, enough that the worker's reader
    finds the rows it counts. Only the `trials` and `trial_values`
    tables are touched -- the rest of the Optuna schema isn't needed
    for our queries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS trials (
                trial_id INTEGER PRIMARY KEY,
                number INTEGER,
                state TEXT
            );
            CREATE TABLE IF NOT EXISTS trial_values (
                trial_value_id INTEGER PRIMARY KEY,
                trial_id INTEGER,
                value REAL
            );
            """
        )
        for i, obj in enumerate(completed_objectives):
            con.execute(
                "INSERT INTO trials (number, state) VALUES (?, 'COMPLETE')",
                (i,),
            )
            tid = con.execute("SELECT last_insert_rowid()").fetchone()[0]
            con.execute(
                "INSERT INTO trial_values (trial_id, value) VALUES (?, ?)",
                (tid, float(obj)),
            )
        for j in range(running_trials):
            con.execute(
                "INSERT INTO trials (number, state) VALUES (?, 'RUNNING')",
                (len(completed_objectives) + j,),
            )
        con.commit()
    finally:
        con.close()


def test_count_completed_trials_missing_file_returns_zero(tmp_path):
    assert _count_completed_trials(tmp_path / "nope.db") == 0


def test_count_completed_trials_counts_only_complete(tmp_path):
    db = tmp_path / "study.db"
    _make_optuna_like_db(
        db, completed_objectives=[0.1, 0.2, 0.3], running_trials=2,
    )
    assert _count_completed_trials(db) == 3


def test_count_completed_trials_handles_garbage_file(tmp_path):
    """A non-SQLite file at the path shouldn't crash the polling
    loop -- the worker swallows sqlite3.Error and returns 0."""
    bad = tmp_path / "study.db"
    bad.write_bytes(b"this is not sqlite")
    assert _count_completed_trials(bad) == 0


def test_last_complete_trial_objective_returns_most_recent(tmp_path):
    db = tmp_path / "study.db"
    _make_optuna_like_db(
        db, completed_objectives=[0.11, 0.22, 0.78], running_trials=1,
    )
    obj = _last_complete_trial_objective(db)
    assert obj == pytest.approx(0.78)


def test_last_complete_trial_objective_missing_file_returns_none(tmp_path):
    assert _last_complete_trial_objective(tmp_path / "nope.db") is None


# ----------------------------------------------------------------------
# scan_completed_studies
# ----------------------------------------------------------------------

def _write_best_params(workdir: Path, payload: dict) -> None:
    """Mirror what detector.hyperopt writes at the end of optimize_scope."""
    out = workdir / "hyperopt"
    out.mkdir(parents=True, exist_ok=True)
    (out / "best_params.json").write_text(json.dumps(payload))


def test_scan_completed_studies_empty_dir(tmp_path):
    out = scan_completed_studies(tmp_path)
    assert out == {"combined": None, "per_animal": {}}


def test_scan_completed_studies_combined_only(tmp_path):
    combined = tmp_path / "hyperopt_combined"
    _write_best_params(combined, {
        "best_params": {"w_neg": 0.4, "fp_weight": 3.2, "fn_weight": 5.0},
        "best_objective": 0.812,
    })
    out = scan_completed_studies(tmp_path)
    assert out["combined"]["best_objective"] == 0.812
    assert out["per_animal"] == {}


def test_scan_completed_studies_per_animal_partial(tmp_path):
    """Some animals finished, some haven't yet -- the scan should
    only surface the finished ones rather than crashing on the half-
    populated tree."""
    pa_root = tmp_path / "hyperopt_per_animal"
    _write_best_params(pa_root / "A", {
        "best_params": {"w_neg": 0.5, "fp_weight": 2.0, "fn_weight": 4.0},
        "best_objective": 0.7,
    })
    _write_best_params(pa_root / "B", {
        "best_params": {"w_neg": 0.3, "fp_weight": 5.0, "fn_weight": 3.5},
        "best_objective": 0.81,
    })
    # Animal C has its workdir but no best_params yet (mid-run).
    (pa_root / "C" / "hyperopt").mkdir(parents=True)
    out = scan_completed_studies(tmp_path)
    assert out["combined"] is None
    assert set(out["per_animal"].keys()) == {"A", "B"}
    assert out["per_animal"]["B"]["best_objective"] == 0.81


def test_read_best_params_missing_returns_none(tmp_path):
    assert _read_best_params(tmp_path) is None


def test_read_best_params_malformed_returns_none(tmp_path):
    (tmp_path / "hyperopt").mkdir()
    (tmp_path / "hyperopt" / "best_params.json").write_text("not json {{{")
    assert _read_best_params(tmp_path) is None


# ----------------------------------------------------------------------
# Worker construction
# ----------------------------------------------------------------------

def test_worker_rejects_unknown_scope(tmp_path):
    with pytest.raises(ValueError):
        HyperoptWorker(
            scope="something",
            manifest_path=tmp_path / "m.json",
            artifacts_dir=tmp_path,
            n_trials=10,
        )


def test_worker_only_animals_normalized_to_uppercase(tmp_path):
    """The backend keys group_recordings_by_animal results by
    uppercase letter, so the worker must normalize whatever the user
    passes in."""
    w = HyperoptWorker(
        scope="per_animal",
        manifest_path=tmp_path / "m.json",
        artifacts_dir=tmp_path,
        n_trials=10,
        only_animals=["a", "b", "c"],
    )
    assert w._only_animals == ["A", "B", "C"]


# ----------------------------------------------------------------------
# End-to-end signal pumping (no Qt, no LightGBM, no Optuna)
# ----------------------------------------------------------------------

@pytest.fixture
def qt_app():
    """Bare QApplication. Skipped if the platform plugin won't load
    (CI without display). Mirrors the qt_display gating used by
    test_inference_worker."""
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        return app
    except Exception as exc:                       # pragma: no cover
        pytest.skip(f"Qt platform unavailable: {exc}")


@pytest.mark.qt_display
def test_worker_emits_progress_for_combined_study(qt_app, tmp_path, monkeypatch):
    """End-to-end: stub `optimize_combined` with a function that
    incrementally builds an Optuna-shaped study.db and writes the
    final best_params.json. The worker's polling loop should emit a
    progress signal for each completed trial and a single finished
    signal at the end."""
    import ui.workers.hyperopt_worker as hw_mod

    artifacts = tmp_path
    workdir = artifacts / "hyperopt_combined" / "hyperopt"
    db = workdir / "study.db"
    final_objectives = [0.5, 0.6, 0.75]

    def fake_optimize_combined(*_args, **_kwargs):
        # Mimic Optuna: one row at a time, with a small sleep so the
        # worker's 2s poll catches each one. We use shorter sleeps
        # here to keep the test fast and rely on the worker's final-
        # poll-after-join to backfill any trials it missed.
        for i, obj in enumerate(final_objectives):
            _make_optuna_like_db(db, completed_objectives=final_objectives[:i + 1])
            time.sleep(0.1)
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "best_params.json").write_text(json.dumps({
            "best_params": {"w_neg": 0.4, "fp_weight": 3.0, "fn_weight": 5.0},
            "best_objective": 0.75,
        }))
        return {
            "study_name": "combined",
            "n_trials": 3,
            "best_params": {"w_neg": 0.4, "fp_weight": 3.0, "fn_weight": 5.0},
            "best_objective": 0.75,
        }

    # Patch the symbol where the worker imports it FROM.
    from detector import hyperopt as real_hyperopt
    monkeypatch.setattr(
        real_hyperopt, "optimize_combined", fake_optimize_combined,
    )
    # Speed up polling so the test completes in seconds rather than
    # 10+ seconds. POLL_INTERVAL_S is module-level so we override it.
    monkeypatch.setattr(hw_mod, "POLL_INTERVAL_S", 0.05)

    progress_calls: list[tuple] = []
    finished_calls: list[dict] = []
    error_calls: list[str] = []

    worker = HyperoptWorker(
        scope="combined",
        manifest_path=tmp_path / "manifest.json",
        artifacts_dir=artifacts,
        n_trials=3,
    )
    worker.progress.connect(lambda s, n, t, o: progress_calls.append(
        (s, n, t, o),
    ))
    worker.finished.connect(lambda d: finished_calls.append(d))
    worker.error.connect(lambda m: error_calls.append(m))

    # Run the worker in this thread (no QThread plumbing) -- the
    # backend dispatch is its own threading.Thread, so run() blocks
    # until everything completes.
    worker.run()
    # Let any queued signals drain.
    qt_app.processEvents()

    assert error_calls == [], f"unexpected errors: {error_calls}"
    assert len(finished_calls) == 1
    # At least one progress signal must have made it through. Exact
    # count depends on polling timing, but the final poll-after-join
    # guarantees we see the last trial.
    assert progress_calls, "no progress signals fired"
    scopes = {c[0] for c in progress_calls}
    assert scopes == {"combined"}
    # Last signal should be at trial 3 of 3.
    last = progress_calls[-1]
    assert last[1] == 3
    assert last[2] == 3
    assert last[3] == pytest.approx(0.75)


@pytest.mark.qt_display
def test_worker_emits_animal_signals_for_per_animal_study(
    qt_app, tmp_path, monkeypatch,
):
    """Per-animal scope -- stub `optimize_per_animal` with a function
    that walks two animals and writes each one's best_params.json
    in turn. The worker should emit `animal_started` when the dir
    appears and `animal_finished` when best_params.json lands."""
    import ui.workers.hyperopt_worker as hw_mod

    pa_root = tmp_path / "hyperopt_per_animal"

    def fake_optimize_per_animal(*_args, **_kwargs):
        # Animal A: 2 trials, then best_params.
        a_wd = pa_root / "A" / "hyperopt"
        for i in range(1, 3):
            _make_optuna_like_db(
                a_wd / "study.db",
                completed_objectives=[0.5] * i,
            )
            time.sleep(0.08)
        a_wd.mkdir(parents=True, exist_ok=True)
        (a_wd / "best_params.json").write_text(json.dumps({
            "best_params": {"w_neg": 0.4, "fp_weight": 3.0, "fn_weight": 5.0},
            "best_objective": 0.55,
        }))
        # Animal B: 2 trials, then best_params.
        b_wd = pa_root / "B" / "hyperopt"
        for i in range(1, 3):
            _make_optuna_like_db(
                b_wd / "study.db",
                completed_objectives=[0.6] * i,
            )
            time.sleep(0.08)
        b_wd.mkdir(parents=True, exist_ok=True)
        (b_wd / "best_params.json").write_text(json.dumps({
            "best_params": {"w_neg": 0.5, "fp_weight": 2.5, "fn_weight": 4.0},
            "best_objective": 0.64,
        }))
        return {
            "A": {"status": "ok", "best_objective": 0.55},
            "B": {"status": "ok", "best_objective": 0.64},
        }

    from detector import hyperopt as real_hyperopt
    monkeypatch.setattr(
        real_hyperopt, "optimize_per_animal", fake_optimize_per_animal,
    )
    monkeypatch.setattr(hw_mod, "POLL_INTERVAL_S", 0.04)

    started: list[str] = []
    finished_animals: list[tuple[str, dict]] = []
    overall_finished: list[dict] = []
    errors: list[str] = []

    worker = HyperoptWorker(
        scope="per_animal",
        manifest_path=tmp_path / "manifest.json",
        artifacts_dir=tmp_path,
        n_trials=2,
        min_recordings_per_animal=2,
    )
    worker.animal_started.connect(lambda a: started.append(a))
    worker.animal_finished.connect(
        lambda a, p: finished_animals.append((a, p))
    )
    worker.finished.connect(lambda d: overall_finished.append(d))
    worker.error.connect(lambda m: errors.append(m))

    worker.run()
    qt_app.processEvents()

    assert errors == []
    # Both animals' dirs were created, so both animal_started signals
    # should have fired. Order is alphabetical (sorted iterdir).
    assert started == ["A", "B"]
    assert {a for a, _ in finished_animals} == {"A", "B"}
    assert len(overall_finished) == 1
    assert set(overall_finished[0].keys()) == {"A", "B"}
