"""Entry point for the PyQt detector UI.

M0.3 skeleton — opens a blank window so we can verify the Qt
environment + PySide6 + path resolution all work before building the
actual UI in M1+.

Run with:
    python ui/app.py
or, after `pip install -e .`:
    detector-pyqt
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `from detector import …` work whether we're running from a
# checked-out repo (submodule under ./detector-core/) or an installed
# wheel (where pyproject.toml's pythonpath has already done the job).
_repo_root = Path(__file__).resolve().parent.parent
_detector_core = _repo_root / "detector-core"
if _detector_core.exists() and str(_detector_core) not in sys.path:
    sys.path.insert(0, str(_detector_core))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget


def main() -> int:
    """QApplication boot. The body intentionally stays minimal until M1."""
    app = QApplication(sys.argv)
    app.setApplicationName("Detector — PyQt")
    app.setOrganizationName("GEMSBlanking")

    # Pull the active model version + paths so we can confirm the
    # detector backend is wired up correctly. Catch import errors so a
    # broken submodule doesn't crash the skeleton.
    try:
        from detector import paths as detector_paths  # type: ignore[import-not-found]
        info = (
            f"detector_home : {detector_paths.get_home()}\n"
            f"artifacts_dir : {detector_paths.get_artifacts_dir()}\n"
            f"active_model  : {detector_paths.get_current_model_version() or '(none)'}\n"
            f"manifest      : {detector_paths.get_manifest_path()}"
        )
    except Exception as exc:  # pragma: no cover — surfaced visibly in UI
        info = f"detector import failed: {exc}"

    window = QMainWindow()
    window.setWindowTitle("Detector — PyQt (M0 skeleton)")
    central = QWidget()
    layout = QVBoxLayout(central)
    hello = QLabel("Hello, PyQt detector")
    hello.setAlignment(Qt.AlignmentFlag.AlignCenter)
    hello.setStyleSheet("font-size: 24px; padding: 16px;")
    info_label = QLabel(info)
    info_label.setStyleSheet(
        "font-family: monospace; font-size: 12px; "
        "padding: 8px; color: #888;"
    )
    layout.addWidget(hello)
    layout.addWidget(info_label)
    layout.addStretch(1)
    window.setCentralWidget(central)
    window.resize(800, 400)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
