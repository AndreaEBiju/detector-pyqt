"""Disagreement review panel — modeless dockable widget that walks
the user through disagreements (model bad + human clean) one at a
time, with a ±2s context plot, top-3 SHAP features, score buttons,
notes, and prev/next nav.

Streamlit Phase 8 parity:
- Lazy single-card rendering: SHAP is computed for the visible
  disagreement only (~50ms) rather than precomputing for all 20.
- Score buttons map to keyboard 1/2/3 (true_artifact / borderline /
  false_positive).
- Notes field per disagreement, persisted across navigation.
- Save → writes `<recording_id>_review.json` in the Phase-5 schema
  (`detector.review`) so the standalone aggregator picks it up.
- Drag-to-mark-wider on the context plot — Shift+drag on the
  per-disagreement plot creates a user interval that REPLACES the
  model's narrower detection.

The panel is dockable (Qt's QDockWidget) so the user can keep the
main viewer visible for cross-reference, unlike the plan's modal
QDialog suggestion.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QProgressBar, QPushButton, QSizePolicy, QSplitter,
    QTextEdit, QVBoxLayout, QWidget, QFrame,
)

_repo_root = Path(__file__).resolve().parent.parent.parent
_detector_core = _repo_root / "detector-core"
if _detector_core.exists() and str(_detector_core) not in sys.path:
    sys.path.insert(0, str(_detector_core))

from detector import review as R                                  # noqa: E402
from detector.model_artifact import ModelArtifact                  # noqa: E402

from .signal_viewer import LazyRecording, MultiChannelViewer       # noqa: E402


SCORE_TRUE = "true_artifact"
SCORE_BORDER = "borderline"
SCORE_FALSE = "false_positive"


class ReviewPanel(QWidget):
    """The dockable review widget."""

    # Emitted when the user marks a wider region in the review-plot
    # (Shift+drag). Main window handles by appending to bad_intervals
    # with source="user" AND removing the corresponding model prediction.
    mark_wider_requested = Signal(float, float, int)
    # Emitted when the user has scored every disagreement — main window
    # can highlight or auto-save.
    all_reviewed = Signal()
    # Emitted on Save button.
    save_requested = Signal()
    # Emitted when the panel wants to exit review (close itself).
    exit_requested = Signal()

    def __init__(
        self,
        recording: LazyRecording,
        artifact: ModelArtifact,
        disagreements: list[R.Disagreement],
        feature_df,  # pandas DataFrame — full feature table from the inference run
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._recording = recording
        self._artifact = artifact
        self._disagreements = list(disagreements)
        self._feature_df = feature_df
        # Per-disagreement state. Keyed by position_sample so it
        # survives sort-by-prob and disagreement re-extraction (e.g.
        # after the user marks a wider region and we recompute).
        # Value: {"score": str, "notes": str}
        self._scored: dict[int, dict] = {}
        self._current: int = 0
        # The context plot's transient (start, end) when the user
        # Shift+drags. Cleared when the disagreement changes or the
        # user clicks Mark.
        self._pending_user_sel: Optional[tuple[float, float]] = None

        self._build_ui()
        self._render_current()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # Header row: "Disagreement N / K — prob = 0.92 — reviewed M / K"
        self._header_label = QLabel("")
        self._header_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        outer.addWidget(self._header_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setTextVisible(False)
        outer.addWidget(self._progress_bar)

        # Context plot — a small MultiChannelViewer-like display
        # showing ±2s around the disagreement.
        self._context_plot = pg.GraphicsLayoutWidget()
        self._context_plot.setBackground("#0e1117")
        self._context_plot.setMinimumHeight(280)
        self._context_curves: list[pg.PlotDataItem] = []
        self._context_plots: list[pg.PlotItem] = []
        for ch in range(self._recording.n_channels):
            p = self._context_plot.addPlot(row=ch, col=0)
            p.setMouseEnabled(x=True, y=False)
            p.showGrid(x=True, y=False, alpha=0.15)
            if ch < self._recording.n_channels - 1:
                p.getAxis("bottom").setStyle(showValues=False)
            if self._context_plots:
                p.setXLink(self._context_plots[0])
            curve = p.plot(pen=pg.mkPen(color="#aaa", width=1))
            self._context_plots.append(p)
            self._context_curves.append(curve)
        # Pending-region item per plot (for Shift+drag)
        self._pending_items: list[Optional[pg.LinearRegionItem]] = [
            None for _ in self._context_plots
        ]
        # Model-prediction band (orange) at the disagreement position.
        self._model_band_items: list[Optional[pg.LinearRegionItem]] = [
            None for _ in self._context_plots
        ]
        # Shift+drag event filter — installed on each ViewBox.
        from .signal_viewer import _ShiftDragFilter as _SDF
        for i, p in enumerate(self._context_plots):
            vb = p.getViewBox()
            vb.installEventFilter(_ReviewShiftDragFilter(self, i, parent=vb))
        outer.addWidget(self._context_plot, stretch=1)

        # SHAP features
        self._shap_label = QLabel("")
        self._shap_label.setStyleSheet(
            "font-family: monospace; font-size: 12px;"
        )
        outer.addWidget(self._shap_label)

        # Score buttons
        score_row = QHBoxLayout()
        self._btn_true = QPushButton("✓ true artifact (1)")
        self._btn_border = QPushButton("? borderline (2)")
        self._btn_false = QPushButton("✗ false positive (3)")
        self._btn_true.clicked.connect(lambda: self._score(SCORE_TRUE))
        self._btn_border.clicked.connect(lambda: self._score(SCORE_BORDER))
        self._btn_false.clicked.connect(lambda: self._score(SCORE_FALSE))
        score_row.addWidget(self._btn_true)
        score_row.addWidget(self._btn_border)
        score_row.addWidget(self._btn_false)
        outer.addLayout(score_row)

        # Notes field
        outer.addWidget(QLabel("Notes:"))
        self._notes_edit = QTextEdit()
        self._notes_edit.setMaximumHeight(60)
        self._notes_edit.textChanged.connect(self._on_notes_changed)
        outer.addWidget(self._notes_edit)

        # Nav row
        nav_row = QHBoxLayout()
        self._btn_prev = QPushButton("← prev")
        self._btn_next = QPushButton("next →")
        self._btn_prev.clicked.connect(lambda: self._goto(self._current - 1))
        self._btn_next.clicked.connect(lambda: self._goto(self._current + 1))
        nav_row.addWidget(self._btn_prev)
        nav_row.addWidget(self._btn_next)
        outer.addLayout(nav_row)

        # Action row
        action_row = QHBoxLayout()
        self._btn_save = QPushButton("💾 Save review JSON")
        self._btn_save.clicked.connect(self.save_requested.emit)
        self._btn_exit = QPushButton("Close review")
        self._btn_exit.clicked.connect(self.exit_requested.emit)
        action_row.addWidget(self._btn_save)
        action_row.addWidget(self._btn_exit)
        outer.addLayout(action_row)

        # Keyboard shortcuts (1/2/3, ←/→) — scoped to this widget so they
        # only fire when the panel has focus.
        for key, score in [("1", SCORE_TRUE), ("2", SCORE_BORDER),
                            ("3", SCORE_FALSE)]:
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(lambda s=score: self._score(s))
        sc_left = QShortcut(QKeySequence(Qt.Key_Left), self)
        sc_left.activated.connect(lambda: self._goto(self._current - 1))
        sc_right = QShortcut(QKeySequence(Qt.Key_Right), self)
        sc_right.activated.connect(lambda: self._goto(self._current + 1))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current(self) -> None:
        if not self._disagreements:
            self._header_label.setText("No disagreements — model agrees with human labels.")
            for curve in self._context_curves:
                curve.clear()
            self._shap_label.setText("")
            return
        n = len(self._disagreements)
        self._current = max(0, min(self._current, n - 1))
        d = self._disagreements[self._current]
        n_done = sum(
            1 for s in self._scored.values()
            if s.get("score") in (SCORE_TRUE, SCORE_BORDER, SCORE_FALSE)
        )
        self._header_label.setText(
            f"Disagreement {self._current + 1} / {n}  ·  "
            f"prob = {d.model_prob:.3f}  ·  reviewed {n_done} / {n}"
        )
        self._progress_bar.setMaximum(max(n, 1))
        self._progress_bar.setValue(n_done)

        # Plot ±2s context around the disagreement.
        fs = self._recording.fs
        s_sec = max(0.0, (d.context_start - 1) / fs)
        e_sec = min(self._recording.duration_sec, d.context_end / fs)
        data = self._recording.get_range(s_sec, e_sec)
        if data.shape[0] > 0:
            t = np.linspace(s_sec, s_sec + data.shape[0] / fs,
                             data.shape[0], endpoint=False)
            for ch, curve in enumerate(self._context_curves):
                color = _CHANNEL_COLORS[ch % len(_CHANNEL_COLORS)]
                curve.setData(t, data[:, ch],
                               pen=pg.mkPen(color=color, width=1))
            self._context_plots[0].setXRange(s_sec, e_sec, padding=0)

        # Render the model's prediction band (orange) at the
        # disagreement's center — width = 100 ms (the short window).
        n_short = int(round(0.100 * fs))
        center_sec = (d.position_sample - 1) / fs
        model_s = center_sec - n_short / 2 / fs
        model_e = center_sec + n_short / 2 / fs
        for ch, plot in enumerate(self._context_plots):
            old = self._model_band_items[ch]
            if old is not None:
                plot.removeItem(old)
            band = pg.LinearRegionItem(
                values=(model_s, model_e),
                orientation="vertical", movable=False,
                brush=pg.mkBrush(255, 127, 14, 60),
                pen=pg.mkPen(None),
            )
            band.setZValue(-5)
            plot.addItem(band)
            self._model_band_items[ch] = band

        # Clear pending region (it doesn't survive disagreement change).
        self._clear_pending_regions()
        self._pending_user_sel = None

        # SHAP for current row only.
        try:
            row_idx = d.feature_row_index
            X_one = self._feature_df.iloc[[row_idx]][
                self._artifact.feature_columns
            ]
            top, _, _ = R.compute_shap_for_windows(
                self._artifact.booster, X_one, top_n=3,
            )
            shap_pairs = top[0]
            lines = ["<b>Top SHAP features:</b>"]
            for name, val in shap_pairs:
                color = "#ff6b6b" if val >= 0 else "#5b9aff"
                sign = "+" if val >= 0 else ""
                lines.append(
                    f'<span style="color:{color}">{sign}{val:+.3f}</span>'
                    f'  <span style="color:#bbb">{name}</span>'
                )
            self._shap_label.setText("<br>".join(lines))
        except Exception as exc:
            self._shap_label.setText(f"<i>(SHAP unavailable: {exc})</i>")

        # Update notes
        notes_val = self._scored.get(d.position_sample, {}).get("notes", "")
        # Block the textChanged signal while we set
        self._notes_edit.blockSignals(True)
        self._notes_edit.setPlainText(notes_val)
        self._notes_edit.blockSignals(False)

        # Highlight the active score button
        current_score = self._scored.get(d.position_sample, {}).get("score")
        for btn, val in [
            (self._btn_true, SCORE_TRUE),
            (self._btn_border, SCORE_BORDER),
            (self._btn_false, SCORE_FALSE),
        ]:
            btn.setStyleSheet(
                "font-weight: bold; background-color: #4ea3ff;"
                if current_score == val
                else ""
            )

        # Enable/disable prev/next
        self._btn_prev.setEnabled(self._current > 0)
        self._btn_next.setEnabled(self._current < n - 1)

    # ------------------------------------------------------------------
    # Scoring / navigation
    # ------------------------------------------------------------------

    def _score(self, score_value: str) -> None:
        if not self._disagreements:
            return
        d = self._disagreements[self._current]
        existing = self._scored.get(d.position_sample, {})
        self._scored[d.position_sample] = {
            "score": score_value,
            "notes": existing.get("notes", ""),
        }
        n_done = sum(
            1 for s in self._scored.values()
            if s.get("score") in (SCORE_TRUE, SCORE_BORDER, SCORE_FALSE)
        )
        # Auto-advance if more to review
        if n_done == len(self._disagreements):
            self.all_reviewed.emit()
        if self._current < len(self._disagreements) - 1:
            self._goto(self._current + 1)
        else:
            self._render_current()

    def _on_notes_changed(self) -> None:
        if not self._disagreements:
            return
        d = self._disagreements[self._current]
        existing = self._scored.get(d.position_sample, {})
        existing["notes"] = self._notes_edit.toPlainText()
        # Don't override score if already set
        self._scored[d.position_sample] = existing

    def _goto(self, new_idx: int) -> None:
        if not self._disagreements:
            return
        new_idx = max(0, min(new_idx, len(self._disagreements) - 1))
        if new_idx != self._current:
            self._current = new_idx
            self._render_current()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scored(self) -> dict[int, dict]:
        """Returns {position_sample: {"score": ..., "notes": ...}}."""
        return dict(self._scored)

    def current_disagreement(self):
        if not self._disagreements:
            return None
        return self._disagreements[self._current]

    def write_review_json(
        self,
        output_path: Path,
        model_version: str,
        threshold_used: float,
        reviewer: Optional[str] = None,
    ) -> Path:
        """Persist scores in detector.review's JSON schema (same as
        the Streamlit save flow's `<rid>_review.json`)."""
        payload = {
            "fold_id": self._recording.path.stem,
            "model_version": model_version,
            "threshold_used": float(threshold_used),
            "reviewed_at": _dt.datetime.now(_dt.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "reviewer": reviewer,
            "scores": [
                {
                    "position_sample": d.position_sample,
                    "model_prob": d.model_prob,
                    "score": self._scored.get(
                        d.position_sample, {}
                    ).get("score"),
                    "notes": self._scored.get(
                        d.position_sample, {}
                    ).get("notes", ""),
                }
                for d in self._disagreements
            ],
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(output_path)
        return output_path

    # ------------------------------------------------------------------
    # Drag-to-mark-wider plumbing
    # ------------------------------------------------------------------

    def _start_pending_region(self, plot_idx: int, x_sec: float) -> None:
        self._clear_pending_regions()
        for ch, plot in enumerate(self._context_plots):
            region = pg.LinearRegionItem(
                values=(x_sec, x_sec), orientation="vertical",
                movable=False,
                brush=pg.mkBrush(44, 160, 44, 80),
                pen=pg.mkPen(None),
            )
            region.setZValue(-3)
            plot.addItem(region)
            self._pending_items[ch] = region
        self._pending_user_sel = (x_sec, x_sec)

    def _update_pending_region(self, x_sec: float) -> None:
        if self._pending_user_sel is None:
            return
        start, _ = self._pending_user_sel
        lo, hi = min(start, x_sec), max(start, x_sec)
        for region in self._pending_items:
            if region is not None:
                region.setRegion((lo, hi))
        self._pending_user_sel = (start, x_sec)

    def _commit_pending_region(self, x_sec: float) -> None:
        if self._pending_user_sel is None:
            return
        start, _ = self._pending_user_sel
        lo, hi = min(start, x_sec), max(start, x_sec)
        self._clear_pending_regions()
        # Drop micro-misclicks
        if hi - lo < 1.0 / self._recording.fs:
            self._pending_user_sel = None
            return
        d = self._disagreements[self._current]
        self.mark_wider_requested.emit(
            float(lo), float(hi), int(d.position_sample),
        )
        self._pending_user_sel = None

    def _clear_pending_regions(self) -> None:
        for ch, region in enumerate(self._pending_items):
            if region is not None:
                self._context_plots[ch].removeItem(region)
                self._pending_items[ch] = None

    def remove_current_after_mark(self) -> None:
        """Main window calls this after applying mark_wider_requested
        — removes the disagreement that was just handled and advances
        to the next one."""
        if not self._disagreements:
            return
        d = self._disagreements[self._current]
        # Auto-score as true_artifact since the user replaced the
        # model's narrow detection with their wider one.
        existing = self._scored.get(d.position_sample, {})
        self._scored[d.position_sample] = {
            "score": SCORE_TRUE,
            "notes": existing.get("notes", ""),
        }
        # Remove from displayed list
        self._disagreements.pop(self._current)
        if self._current >= len(self._disagreements):
            self._current = max(0, len(self._disagreements) - 1)
        self._render_current()


# Channel colors mirrored from MultiChannelViewer
_CHANNEL_COLORS = ("#4ea3ff", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd")


class _ReviewShiftDragFilter(pg.QtCore.QObject):
    """Like signal_viewer._ShiftDragFilter but routes to the review
    panel's drag-to-mark-wider handlers."""

    def __init__(self, panel: ReviewPanel, plot_idx: int, parent=None):
        super().__init__(parent)
        self._panel = panel
        self._plot_idx = plot_idx
        self._owning = False

    def eventFilter(self, obj, event) -> bool:
        from PySide6.QtCore import QEvent
        et = event.type()
        if et == QEvent.GraphicsSceneMousePress:
            mods = event.modifiers()
            if (mods & Qt.ShiftModifier) and event.button() == Qt.LeftButton:
                x_sec = self._scene_x_to_data(event.scenePos())
                if x_sec is None:
                    return False
                self._panel._start_pending_region(self._plot_idx, x_sec)
                self._owning = True
                event.accept()
                return True
            return False
        if et == QEvent.GraphicsSceneMouseMove and self._owning:
            x_sec = self._scene_x_to_data(event.scenePos())
            if x_sec is not None:
                self._panel._update_pending_region(x_sec)
            event.accept()
            return True
        if et == QEvent.GraphicsSceneMouseRelease and self._owning:
            x_sec = self._scene_x_to_data(event.scenePos())
            if x_sec is not None:
                self._panel._commit_pending_region(x_sec)
            else:
                self._panel._clear_pending_regions()
                self._panel._pending_user_sel = None
            self._owning = False
            event.accept()
            return True
        return False

    def _scene_x_to_data(self, scene_pos) -> Optional[float]:
        plot_item = self._panel._context_plots[self._plot_idx]
        vb = plot_item.getViewBox()
        try:
            data_pt = vb.mapSceneToView(scene_pos)
            return float(data_pt.x())
        except Exception:
            return None
