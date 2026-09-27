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

# `detector` comes from the detector-core distribution - a declared dependency,
# installed editable from the submodule - and is never found by inserting its
# path (invariant 21). What remains is this app's OWN package: `python ui/app.py`
# puts ui/ on sys.path, not the repo root, so `import ui...` needs the root.
#
# It must go FIRST, and that is load-bearing: detector-core's editable install
# exposes its whole checkout, which contains a Streamlit `ui/` package (no
# `widgets/`). Anywhere later than that entry, `from ui.widgets ...` resolves to
# the Streamlit `ui` and fails. tests/test_audit_dependency.py checks both.
_repo_root = Path(__file__).resolve().parent.parent
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
