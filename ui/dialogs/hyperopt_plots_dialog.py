"""Hyperopt diagnostic plots viewer.

The backend writes four Optuna PNGs per study under
`<study_workdir>/hyperopt/plots/`:

    - optimization_history.png
    - param_importances.png
    - parallel_coordinate.png
    - contour.png

This dialog tabs them in a QTabWidget so the user can flip through
without leaving the app. Each plot lives in a scroll area so large
images don't force the window oversized.

Missing PNGs (e.g. param_importances often skipped with too few
trials) show a placeholder message in their tab rather than being
hidden -- helps the user understand WHY a plot is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QTabWidget, QVBoxLayout, QWidget,
)


# (filename, human-readable tab label, one-line caption)
_PLOTS = [
    (
        "optimization_history.png",
        "History",
        "Objective value over the course of optimization. The dashed "
        "line is the best-so-far running maximum.",
    ),
    (
        "param_importances.png",
        "Importances",
        "Optuna's estimate of how much each hyperparameter influenced "
        "the objective. Needs > 2 trials with variance; if the bar "
        "shows nothing, the study is too short.",
    ),
    (
        "parallel_coordinate.png",
        "Parallel coords",
        "One line per trial across the three params; line colour is "
        "the objective. Lets you eyeball which combinations cluster "
        "near the best.",
    ),
    (
        "contour.png",
        "Contour",
        "Pairwise interaction surfaces. Useful for spotting "
        "ridge-shaped optima where two params trade off against "
        "each other.",
    ),
]


class HyperoptPlotsDialog(QDialog):
    """Tabbed PNG viewer for a single study's diagnostic plots."""

    def __init__(
        self,
        plots_dir: Path,
        *,
        study_label: str = "study",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._plots_dir = Path(plots_dir)
        self.setWindowTitle(f"Hyperopt plots — {study_label}")
        self.resize(1000, 720)

        layout = QVBoxLayout(self)

        header = QLabel(
            f"<b>Study:</b> {study_label}<br>"
            f"<span style='color:#888; font-family:monospace; "
            f"font-size:11px;'>{self._plots_dir}</span>"
        )
        header.setTextInteractionFlags(Qt.TextSelectableByMouse)
        header.setWordWrap(True)
        layout.addWidget(header)

        self._tabs = QTabWidget()
        for fname, tab_label, caption in _PLOTS:
            self._tabs.addTab(
                self._build_plot_tab(self._plots_dir / fname, caption),
                tab_label,
            )
        layout.addWidget(self._tabs, stretch=1)

        # Footer: open-in-Finder + close.
        footer = QHBoxLayout()
        btn_open = QPushButton("Open plots folder")
        btn_open.clicked.connect(self._open_plots_folder)
        footer.addWidget(btn_open)
        footer.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        footer.addWidget(bb)
        layout.addLayout(footer)

    def _build_plot_tab(self, png_path: Path, caption: str) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        cap = QLabel(caption)
        cap.setStyleSheet("color: #888; padding: 2px;")
        cap.setWordWrap(True)
        v.addWidget(cap)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QLabel()
        inner.setAlignment(Qt.AlignCenter)
        if png_path.exists():
            pix = QPixmap(str(png_path))
            if pix.isNull():
                inner.setText(
                    f"Could not load plot from {png_path}.\n"
                    "The file is on disk but Qt failed to decode it -- "
                    "try opening it manually in an image viewer."
                )
            else:
                inner.setPixmap(pix)
        else:
            inner.setText(
                f"No plot at {png_path.name} yet.\n\n"
                "This usually means the study had too few trials for "
                "Optuna to produce this diagnostic. Run more trials "
                "and reopen this dialog."
            )
            inner.setStyleSheet("color: #888;")
        scroll.setWidget(inner)
        v.addWidget(scroll, stretch=1)
        return w

    def _open_plots_folder(self) -> None:
        """Reveal the plots dir in the OS file manager. Best-effort."""
        import subprocess
        import sys
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(self._plots_dir)], check=False)
            elif sys.platform.startswith("win"):
                subprocess.run(["explorer", str(self._plots_dir)],
                               check=False)
            else:
                subprocess.run(["xdg-open", str(self._plots_dir)],
                               check=False)
        except Exception:                       # pragma: no cover
            pass
