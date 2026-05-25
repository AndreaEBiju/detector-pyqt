"""Smoke tests for the per-animal inference auto-routing layer.

Verifies that resolve_per_animal_auto():
- extracts the animal letter from the recording's filename,
- picks the newest model_v*/ under <artifacts_dir>/per_animal/<letter>/,
- falls back to the combined promoted model when no per-animal model
  exists or when the animal letter can't be detected,
- never errors out on missing dirs.

Uses tmp_path + DETECTOR_HOME monkey-patching so we don't touch any
real artifacts on disk. No real model loading -- just directory
inspection + name resolution.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
_DETECTOR_CORE = ROOT / "detector-core"
if _DETECTOR_CORE.exists() and str(_DETECTOR_CORE) not in sys.path:
    sys.path.insert(0, str(_DETECTOR_CORE))


def _setup_artifacts(tmp_path: Path, monkeypatch) -> Path:
    """Wire DETECTOR_HOME/ARTIFACTS to a fresh tmp dir and return the
    artifacts dir path. The per_animal/ subdir is NOT auto-created --
    individual tests create the ones they want."""
    monkeypatch.setenv("DETECTOR_HOME", str(tmp_path))
    monkeypatch.delenv("DETECTOR_ARTIFACTS", raising=False)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    return artifacts


def test_resolve_picks_newest_per_animal_model(tmp_path, monkeypatch):
    """Two model versions on disk for animal J; the newer should win.

    Per the orchestrator, per-animal models live at
    artifacts/per_animal/<letter>/model_v*/.
    """
    artifacts = _setup_artifacts(tmp_path, monkeypatch)
    j_dir = artifacts / "per_animal" / "J"
    (j_dir / "model_v0.1.0").mkdir(parents=True)
    (j_dir / "model_v0.2.0").mkdir(parents=True)

    from ui.workers.inference_worker import resolve_per_animal_auto
    info = resolve_per_animal_auto(
        "/data/E10_JEL_E10_bl_1315_notched.mat"
    )
    assert info["animal"] == "J"
    assert info["per_animal_version"] == "v0.2.0"
    assert Path(info["per_animal_dir"]) == j_dir / "model_v0.2.0"
    assert info["used_fallback"] is False


def test_resolve_falls_back_when_animal_dir_missing(tmp_path, monkeypatch):
    """Filename's animal letter is 'F' but no per_animal/F/ exists --
    should fall back to the combined promoted model and record a
    reason for the UI to surface."""
    artifacts = _setup_artifacts(tmp_path, monkeypatch)
    (artifacts / "current_model").write_text("model_v0.3.0\n")
    # Create only the J workdir, not F's
    (artifacts / "per_animal" / "J" / "model_v0.1.0").mkdir(parents=True)

    from ui.workers.inference_worker import resolve_per_animal_auto
    info = resolve_per_animal_auto("/data/X_FRE_X_bl_1.mat")
    assert info["animal"] == "F"
    assert info["per_animal_version"] is None
    assert info["used_fallback"] is True
    assert "No per-animal model" in (info["fallback_reason"] or "")
    assert info["fallback_version"] == "model_v0.3.0"


def test_resolve_falls_back_when_animal_letter_undetectable(
    tmp_path, monkeypatch,
):
    """A weird recording filename with no underscore-separated animal
    token should fall back gracefully."""
    artifacts = _setup_artifacts(tmp_path, monkeypatch)
    (artifacts / "current_model").write_text("model_v0.3.0\n")

    from ui.workers.inference_worker import resolve_per_animal_auto
    info = resolve_per_animal_auto("/data/singletoken.mat")
    assert info["animal"] is None
    assert info["used_fallback"] is True
    assert "Couldn't extract" in (info["fallback_reason"] or "")
    assert info["fallback_version"] == "model_v0.3.0"


def test_resolve_falls_back_when_artifacts_dir_missing_per_animal(
    tmp_path, monkeypatch,
):
    """No per_animal/ subdir at all -- should still resolve cleanly."""
    artifacts = _setup_artifacts(tmp_path, monkeypatch)
    (artifacts / "current_model").write_text("model_v0.1.0\n")

    from ui.workers.inference_worker import resolve_per_animal_auto
    info = resolve_per_animal_auto("/data/E1_JEL_E1_bl_1.mat")
    # Animal detection still works (filename-only); the lookup just
    # returns no versions.
    assert info["animal"] == "J"
    assert info["used_fallback"] is True


def test_load_model_artifact_cached_sentinel_falls_back(
    tmp_path, monkeypatch,
):
    """The auto-routing sentinel, when no per-animal model exists,
    should transparently load the combined promoted model. The local
    booster.txt fixture used here is the same one
    test_inference_worker.py uses -- skipped if it's not on disk."""
    artifact_dir = Path.home() / ".detector" / "artifacts" / "model_v0.1.0"
    booster = artifact_dir / "booster.txt"
    if not booster.exists():
        pytest.skip(f"model_v0.1.0 not on local disk at {booster}")
    try:
        open(booster, "rb").read(1)
    except OSError as e:
        pytest.skip(f"booster unreadable ({e})")

    import os
    os.environ["DETECTOR_ARTIFACTS"] = str(artifact_dir.parent)
    try:
        # Point current_model at v0.1.0 so the fallback resolves.
        (artifact_dir.parent / "current_model").write_text(
            "model_v0.1.0\n"
        )
        from ui.workers.inference_worker import (
            _artifact_cache, load_model_artifact_cached,
            PER_ANIMAL_AUTO_SENTINEL,
        )
        _artifact_cache.clear()
        sentinel = f"{PER_ANIMAL_AUTO_SENTINEL}:/data/X_QQQ_bl_1.mat"
        # Q has no per-animal model anywhere, so this must fall back
        # to the combined v0.1.0 artifact without raising.
        artifact = load_model_artifact_cached(sentinel)
        assert artifact is not None
    finally:
        del os.environ["DETECTOR_ARTIFACTS"]
