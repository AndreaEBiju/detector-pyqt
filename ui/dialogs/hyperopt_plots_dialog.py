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

For per-animal hyperopt, multiple studies (one per animal) exist
and the user wants to switch between them without closing the
dialog. The constructor accepts EITHER a single `plots_dir`
(combined-scope) or a `studies` list of (label, plots_dir) tuples
(per-animal) with a dropdown at the top to switch between them.

Captions for each tab now describe what to look for AND explain
common failure modes (e.g. "empty importances plot" = fANOVA
couldn't separate signal from noise; "too few trials for contour"
= need ~25+ trials for pairwise interaction surfaces).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QTabWidget, QVBoxLayout, QWidget,
)


# (filename, human-readable tab label, caption with troubleshooting)
_PLOTS = [
    (
        "optimization_history.png",
        "History",
        "Objective value over the course of optimization. The dashed "
        "line is the best-so-far running maximum. A flat line late "
        "in the run = TPE has converged; further trials are exploit "
        "rather than explore.",
    ),
    (
        "param_importances.png",
        "Importances",
        "Optuna's fANOVA estimate of how much each hyperparameter "
        "influenced the objective. <b>Empty plot</b> means fANOVA "
        "couldn't separate signal from noise -- happens when (a) "
        "all trials produced near-identical objectives, (b) the "
        "study had too few trials (need ≥10 typically), or (c) the "
        "objective landscape is flat in the searched region. Widen "
        "the parameter ranges or run more trials to fix.",
    ),
    (
        "parallel_coordinate.png",
        "Parallel coords",
        "One line per trial across the params; line colour = "
        "objective (darker = better). <b>Too many crisscrossing "
        "lines</b> means no single param dominates -- the optimum "
        "is a multi-param interaction. Look for clusters of dark "
        "lines passing through the same ranges; those are the "
        "high-performing regions.",
    ),
    (
        "contour.png",
        "Contour",
        "Pairwise interaction surfaces for each (param-A, param-B) "
        "combination. <b>'Too few trials'</b> means Optuna needs "
        "more pairwise samples to render a smooth surface -- the "
        "minimum for 3 params is ~25-30 trials (it interpolates "
        "between observed points). Run a longer study or accept "
        "that contour isn't meaningful at low trial counts.",
    ),
]


class HyperoptPlotsDialog(QDialog):
    """Tabbed PNG viewer for one or many study plot folders.

    Backwards-compatible: callers passing a single `plots_dir` get
    the original single-study behavior. Callers passing `studies=
    [(label, plots_dir), ...]` get a dropdown at the top to switch.
    """

    def __init__(
        self,
        plots_dir: Optional[Path] = None,
        *,
        study_label: str = "study",
        studies: Optional[list] = None,
        initial_study_label: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        # Normalize into a list of (label, plots_dir) tuples.
        if studies is not None:
            self._studies: list[tuple[str, Path]] = [
                (str(lbl), Path(p)) for lbl, p in studies
            ]
        elif plots_dir is not None:
            self._studies = [(study_label, Path(plots_dir))]
        else:
            raise ValueError(
                "HyperoptPlotsDialog needs either `plots_dir` or "
                "`studies`"
            )

        # Pick initial study by label if requested, else first.
        self._current_idx = 0
        if initial_study_label is not None:
            for i, (lbl, _) in enumerate(self._studies):
                if lbl == initial_study_label:
                    self._current_idx = i
                    break

        self.setWindowTitle("Hyperopt plots")
        self.resize(1000, 760)

        layout = QVBoxLayout(self)

        # Top: study switcher (only meaningful when there are ≥2
        # studies; for a single study we hide the switcher to avoid
        # noise).
        if len(self._studies) > 1:
            switcher_row = QHBoxLayout()
            switcher_row.addWidget(QLabel("<b>Study:</b>"))
            self._study_combo = QComboBox()
            for label, plots_dir in self._studies:
                # Show "(no plots yet)" suffix for studies whose
                # plots folder hasn't been written -- a useful
                # signal that this animal's run errored or skipped.
                has_plots = any(
                    (plots_dir / fname).exists()
                    for fname, _, _ in _PLOTS
                )
                display = label if has_plots else f"{label} (no plots)"
                self._study_combo.addItem(display, label)
            self._study_combo.setCurrentIndex(self._current_idx)
            self._study_combo.currentIndexChanged.connect(
                self._on_study_changed
            )
            switcher_row.addWidget(self._study_combo, stretch=1)
            switcher_row.addStretch(0)
            layout.addLayout(switcher_row)

        # Header (label + path of the currently-selected study).
        self._header = QLabel("")
        self._header.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._header.setWordWrap(True)
        layout.addWidget(self._header)

        # Tabs. We rebuild them every time the user switches studies
        # rather than reusing the QLabel objects -- simpler than
        # tracking which QLabel needs which pixmap, and Qt handles
        # the deferred deletion cleanly.
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs, stretch=1)

        # Footer: open-in-Finder + close.
        footer = QHBoxLayout()
        self._btn_open = QPushButton("Open plots folder")
        self._btn_open.clicked.connect(self._open_plots_folder)
        footer.addWidget(self._btn_open)
        footer.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        footer.addWidget(bb)
        layout.addLayout(footer)

        # Initial population.
        self._reload_for_current_study()

    # ------------------------------------------------------------------
    # Switching studies
    # ------------------------------------------------------------------

    def _on_study_changed(self, idx: int) -> None:
        if 0 <= idx < len(self._studies):
            self._current_idx = idx
            self._reload_for_current_study()

    def _current_plots_dir(self) -> Path:
        return self._studies[self._current_idx][1]

    def _current_label(self) -> str:
        return self._studies[self._current_idx][0]

    def _reload_for_current_study(self) -> None:
        """Rebuild the header + all tab contents for whichever study
        is currently selected."""
        plots_dir = self._current_plots_dir()
        label = self._current_label()
        self._header.setText(
            f"<b>Study:</b> {label}<br>"
            f"<span style='color:#888; font-family:monospace; "
            f"font-size:11px;'>{plots_dir}</span>"
        )
        # Tear down existing tabs.
        while self._tabs.count() > 0:
            w = self._tabs.widget(0)
            self._tabs.removeTab(0)
            if w is not None:
                w.deleteLater()
        # Rebuild for this study.
        for fname, tab_label, caption in _PLOTS:
            self._tabs.addTab(
                self._build_plot_tab(plots_dir / fname, caption),
                tab_label,
            )

    # ------------------------------------------------------------------
    # Per-tab UI
    # ------------------------------------------------------------------

    def _build_plot_tab(self, png_path: Path, caption: str) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        cap = QLabel(caption)
        cap.setStyleSheet("color: #888; padding: 2px;")
        cap.setWordWrap(True)
        cap.setTextFormat(Qt.RichText)
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
                    "The file is on disk but Qt failed to decode "
                    "it -- try opening it manually in an image "
                    "viewer."
                )
            else:
                inner.setPixmap(pix)
        else:
            inner.setText(
                f"No plot at {png_path.name} yet.\n\n"
                "Either the study had too few trials for Optuna to "
                "produce this diagnostic, or the run errored before "
                "the plot step. See the per-plot caption above for "
                "what this plot needs."
            )
            inner.setStyleSheet("color: #888;")
        scroll.setWidget(inner)
        v.addWidget(scroll, stretch=1)
        return w

    # ------------------------------------------------------------------
    # File-manager open
    # ------------------------------------------------------------------

    def _open_plots_folder(self) -> None:
        """Reveal the current study's plots dir in the OS file
        manager. Best-effort -- silently skips on errors."""
        import subprocess
        import sys
        plots_dir = self._current_plots_dir()
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(plots_dir)], check=False)
            elif sys.platform.startswith("win"):
                subprocess.run(["explorer", str(plots_dir)],
                               check=False)
            else:
                subprocess.run(["xdg-open", str(plots_dir)],
                               check=False)
        except Exception:                       # pragma: no cover
            pass
