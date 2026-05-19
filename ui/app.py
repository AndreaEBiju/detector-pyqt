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

from PySide6.QtWidgets import QApplication

from ui.windows.main_window import MainWindow


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="detector-pyqt",
        description=(
            "PyQt UI for the motion-artifact detector. Opens the M2 "
            "Browser & Label window. Pass a recording path to open "
            "it on launch, or use File → Open from inside the window."
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

    recording_path = (
        Path(args.recording).expanduser() if args.recording else None
    )
    if recording_path is not None and not recording_path.exists():
        print(
            f"error: recording not found at {recording_path}",
            file=sys.stderr,
        )
        return 2
    window = MainWindow(recording_path=recording_path)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
