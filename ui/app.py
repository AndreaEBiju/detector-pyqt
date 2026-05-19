"""Entry point for the PyQt detector UI.

Phase M0 was a "Hello, PyQt detector" stub. M1 extends it with the
signal-viewer spike — pass a recording path and you get the multi-
channel viewer described in `ui/widgets/signal_viewer.py`.

Run with:
    python ui/app.py                       # M0 skeleton (no recording)
    python ui/app.py path/to/recording.mat  # M1 spike viewer

After `pip install -e .`:
    detector-pyqt
    detector-pyqt path/to/recording.mat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `from detector import …` and `from ui.widgets import …` work
# whether we're running from a checked-out repo (submodule under
# ./detector-core/) or an installed wheel. Insertion order matters:
# detector-core/ is the GEMSBlanking submodule and contains its own
# Streamlit `ui/` package (no `widgets/` subdir). If it sits BEFORE
# detector-pyqt in sys.path, `from ui.widgets …` resolves to the
# Streamlit `ui` and fails. Insert detector-core first, then prepend
# the detector-pyqt root on top of it. Final order:
#     [detector-pyqt, detector-core, …rest]
_repo_root = Path(__file__).resolve().parent.parent
_detector_core = _repo_root / "detector-core"
if _detector_core.exists():
    sys.path.insert(0, str(_detector_core))
sys.path.insert(0, str(_repo_root))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QLabel, QMainWindow, QToolBar, QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QAction


def _info_block() -> str:
    """Status-line summary of the path-resolution layer. Surfaced in
    the M0 skeleton view so M0's DoD is visible inside the app."""
    try:
        from detector import paths as detector_paths
        lines = [
            f"detector_home : {detector_paths.get_home()}",
            f"artifacts_dir : {detector_paths.get_artifacts_dir()}",
            f"active_model  : "
            f"{detector_paths.get_current_model_version() or '(none)'}",
            f"manifest      : {detector_paths.get_manifest_path()}",
        ]
        return "\n".join(lines)
    except Exception as exc:
        return f"detector import failed: {exc}"


def _build_window_skeleton() -> QMainWindow:
    """M0 skeleton — shown when no recording is supplied."""
    window = QMainWindow()
    window.setWindowTitle("Detector — PyQt (M0 skeleton)")
    central = QWidget()
    layout = QVBoxLayout(central)
    hello = QLabel("Hello, PyQt detector")
    hello.setAlignment(Qt.AlignmentFlag.AlignCenter)
    hello.setStyleSheet("font-size: 24px; padding: 16px;")
    info = QLabel(_info_block())
    info.setStyleSheet(
        "font-family: monospace; font-size: 12px; "
        "padding: 8px; color: #888;"
    )
    hint = QLabel(
        "Run `python ui/app.py <recording.mat>` to open the M1 signal "
        "viewer spike."
    )
    hint.setStyleSheet("color: #aaa; padding: 8px;")
    layout.addWidget(hello)
    layout.addWidget(info)
    layout.addWidget(hint)
    layout.addStretch(1)
    window.setCentralWidget(central)
    window.resize(800, 400)
    return window


def _build_window_viewer(recording_path: Path) -> QMainWindow:
    """M1 spike — opens the LazyRecording + MultiChannelViewer for the
    supplied path. The window's status bar shows the recording's basic
    stats and the current viewport position."""
    from ui.widgets.signal_viewer import LazyRecording, MultiChannelViewer

    recording = LazyRecording(recording_path)
    viewer = MultiChannelViewer(recording)
    window = QMainWindow()
    window.setWindowTitle(
        f"Detector — PyQt   |   {recording_path.name}"
    )
    window.setCentralWidget(viewer)
    window.resize(1300, 700)

    # Status bar with the recording stats — useful at a glance and
    # confirms the path-resolution layer surfaced the file we expected.
    status = window.statusBar()
    status.showMessage(
        f"fs={recording.fs:.1f} Hz  "
        f"channels={recording.n_channels}  "
        f"samples={recording.n_samples:,}  "
        f"duration={recording.duration_sec:.1f}s"
    )

    # A "Reset viewport" toolbar action is enough for the spike. M2
    # will replace this with a full menu/toolbar.
    toolbar = QToolBar()
    window.addToolBar(toolbar)
    reset_action = QAction("Reset viewport (0–60s)", window)
    reset_action.triggered.connect(
        lambda: viewer.set_viewport(0.0, min(60.0, recording.duration_sec))
    )
    toolbar.addAction(reset_action)
    fit_action = QAction("Fit all", window)
    fit_action.triggered.connect(
        lambda: viewer.set_viewport(0.0, recording.duration_sec)
    )
    toolbar.addAction(fit_action)
    return window


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="detector-pyqt",
        description=(
            "PyQt UI for the motion-artifact detector. Without an "
            "argument, opens the M0 skeleton. With a recording path, "
            "opens the M1 signal-viewer spike."
        ),
    )
    parser.add_argument(
        "recording", nargs="?", default=None,
        help="Path to a `.mat` or `.h5` recording (optional).",
    )
    args = parser.parse_args()

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Detector — PyQt")
    app.setOrganizationName("GEMSBlanking")

    if args.recording is None:
        window = _build_window_skeleton()
    else:
        recording_path = Path(args.recording).expanduser()
        if not recording_path.exists():
            print(
                f"error: recording not found at {recording_path}",
                file=sys.stderr,
            )
            return 2
        window = _build_window_viewer(recording_path)

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
