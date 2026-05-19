"""Inference worker — runs `detect_bad_with_progress` on a Qt worker
thread so the main UI stays responsive during the ~30s–3min that
feature extraction takes (M1.4 timing).

Design
------
`InferenceWorker` is a `QObject` (lives in a `QThread` via
`moveToThread`) that calls `detector.predict.detect_bad_with_progress`
and re-emits its progress callback as a Qt signal. On success the
result dict is forwarded to the main window via `finished`; on
exception the message goes via `error`.

The detector-pyqt repo's PyQt UI runs only one inference job at a
time. We don't queue concurrent runs — the user clicks "Run
inference", a worker spawns, the toolbar action disables until
finished/error. This matches the Streamlit Phase 8 UX.

Model-artifact caching
----------------------
`load_model_artifact_cached(version)` is a module-level memoize so
re-running inference doesn't re-read the booster (~50MB) every time.
Cache key is the artifact version string; clearing happens
automatically when a different version is loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

# Self-contained sys.path setup (same pattern as the rest of the repo —
# detector-pyqt before detector-core or our `ui/` gets shadowed by the
# Streamlit `ui/` in the submodule).
_repo_root = Path(__file__).resolve().parent.parent.parent
_detector_core = _repo_root / "detector-core"
if _detector_core.exists():
    if str(_detector_core) not in sys.path:
        sys.path.insert(0, str(_detector_core))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from detector import paths as detector_paths                    # noqa: E402
from detector.model_artifact import ModelArtifact                # noqa: E402
from detector.predict import detect_bad_with_progress            # noqa: E402


# ----------------------------------------------------------------------
# Model-artifact cache
# ----------------------------------------------------------------------

_artifact_cache: dict[str, ModelArtifact] = {}


def load_model_artifact_cached(version: Optional[str]) -> ModelArtifact:
    """Load (and cache by version string) a `ModelArtifact`.

    `version=None` → the current promoted version (whatever
    `paths.get_current_model_version()` returns from the shared Drive
    folder). `version="model_v0.1.0"` → load that explicit version.
    The "model_" prefix is accepted both with and without (for
    backward compat with the Streamlit short form "v0.1.0").
    """
    if version is None:
        version = detector_paths.get_current_model_version()
        if version is None:
            raise RuntimeError(
                "No `current_model` pointer found in the artifacts "
                "directory. Run `detector init` to point at the shared "
                "model folder, or pass an explicit version string."
            )
    # Accept both "model_v0.1.0" and "v0.1.0" — the on-disk dir name
    # has the prefix; some callers will hand us the short form.
    if not version.startswith("model_"):
        version_dir_name = f"model_{version}"
    else:
        version_dir_name = version
    if version_dir_name in _artifact_cache:
        return _artifact_cache[version_dir_name]
    artifacts_dir = detector_paths.get_artifacts_dir()
    path = artifacts_dir / version_dir_name
    if not (path / "booster.txt").exists():
        raise FileNotFoundError(
            f"booster.txt not found at {path}. Either the model isn't "
            "synced from Drive yet, or the version name is wrong."
        )
    artifact = ModelArtifact.load(path)
    _artifact_cache[version_dir_name] = artifact
    return artifact


def clear_artifact_cache() -> None:
    """Drop all cached artifacts. Call when artifacts on disk change
    (rare in PyQt-side workflow; included for completeness)."""
    _artifact_cache.clear()


# ----------------------------------------------------------------------
# QThread worker
# ----------------------------------------------------------------------

class InferenceWorker(QObject):
    """Single-shot inference worker. Live in a QThread; emit a Qt signal
    on completion or error.

    Usage (typical pattern from main_window):

        thread = QThread()
        worker = InferenceWorker(y, fs, artifact, ...)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(on_progress)         # (frac, msg)
        worker.finished.connect(on_finished)         # (result_dict,)
        worker.error.connect(on_error)               # (message,)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.start()
    """

    # 0.0 .. 1.0 fraction, stage-name string
    progress = Signal(float, str)
    # detect_bad_with_progress result dict (with optional `features`
    # DataFrame when stim_skip_offset is non-zero — the offset has
    # already been applied before this fires).
    finished = Signal(dict)
    # Human-readable error message
    error = Signal(str)

    def __init__(
        self,
        y: np.ndarray,
        fs: float,
        artifact: ModelArtifact,
        *,
        stim_skip_offset: int = 0,
        return_features: bool = True,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._y = y
        self._fs = fs
        self._artifact = artifact
        self._stim_skip_offset = int(stim_skip_offset)
        self._return_features = bool(return_features)

    @Slot()
    def run(self) -> None:
        """Worker-thread entry point. Catches all exceptions so a
        Python error inside `detect_bad_with_progress` doesn't crash
        the main UI."""
        try:
            def _cb(frac: float, msg: str) -> None:
                # Qt signals are thread-safe — the connected slot on
                # the main thread fires via the event loop.
                self.progress.emit(float(frac), str(msg))
            res = detect_bad_with_progress(
                self._y, self._fs, self._artifact,
                progress_callback=_cb,
                return_features=self._return_features,
            )
            # Apply stim-skip offset bookkeeping. The caller sliced y
            # before passing it in; we know the offset and adjust the
            # 1-based output indices so the rest of the UI works in
            # full-recording coordinates.
            offset = self._stim_skip_offset
            if offset > 0:
                if res.get("intervals") is not None and res["intervals"].size:
                    res["intervals"] = (
                        res["intervals"] + np.int64(offset)
                    )
                if res.get("position_samples") is not None:
                    res["position_samples"] = (
                        res["position_samples"] + np.int64(offset)
                    )
                if res.get("first_position_sample") is not None:
                    res["first_position_sample"] = int(
                        res["first_position_sample"] + offset
                    )
                feats = res.get("features")
                if feats is not None and "position_sample" in feats:
                    feats = feats.copy()
                    feats["position_sample"] = (
                        feats["position_sample"] + np.int64(offset)
                    )
                    res["features"] = feats
            res["scope_offset_sample"] = int(offset)
            res["scope"] = "recovery_only" if offset else "full"
            self.finished.emit(res)
        except Exception as exc:                               # pragma: no cover — UI error path
            self.error.emit(f"{type(exc).__name__}: {exc}")


def current_promoted_version_short() -> Optional[str]:
    """Strip the `model_` prefix from `paths.get_current_model_version()`
    so callers that want a short label (e.g. dropdown text) don't have
    to handle the prefix themselves.

    Returns `"v0.1.0"` from `"model_v0.1.0"`, or `None` if no
    `current_model` pointer exists.
    """
    v = detector_paths.get_current_model_version()
    if v is None:
        return None
    return v[len("model_"):] if v.startswith("model_") else v
