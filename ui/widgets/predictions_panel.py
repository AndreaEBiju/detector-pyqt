"""Predictions panel — table of model-predicted bad intervals with
Accept / Dismiss / Accept-all buttons.

Sits in the right dock alongside the user-marked region table. Lives
side-by-side via a QTabWidget; user toggles between "Marked" (the
authoritative bad-intervals list) and "Model" (the model's
predictions that haven't been accepted or dismissed yet).

Streamlit Phase 8 parity:
- One row per model interval with start/end/duration and probability.
- "Accept" → appends the interval to the bad-intervals list (with
  source="model_accepted") and removes it from the predictions list.
- "Dismiss" → removes it from predictions WITHOUT marking. Anything
  still in predictions at save time is the "model_unsure" set the
  save flow writes to `removedSegmentIdx_model_unsure`.
- "Accept ALL" → bulk-add every prediction, then clear the panel.
- Double-click row → jump viewport.

The panel owns NO authoritative state. Main window holds the
predictions array; this widget renders it via `set_predictions()`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView, QHeaderView, QMenu, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from .region_table import format_time


class PredictionsPanel(QWidget):
    """Composite widget: QPushButton row + QTableWidget."""

    # Emitted when user clicks Accept on row `idx`. Main window does
    # the actual move (append to bad_intervals with source="model_accepted"
    # + remove from predictions + re-render).
    interval_accepted = Signal(int)
    interval_dismissed = Signal(int)
    interval_jumped = Signal(int)
    accept_all_requested = Signal()

    COLUMNS = ("#", "start", "end", "Δ", "prob")

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._accept_all_btn = QPushButton("➕ Accept ALL into marked")
        self._accept_all_btn.clicked.connect(self.accept_all_requested.emit)
        self._accept_all_btn.setEnabled(False)
        layout.addWidget(self._accept_all_btn)

        self._table = QTableWidget(0, len(self.COLUMNS), self)
        self._table.setHorizontalHeaderLabels(self.COLUMNS)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.itemDoubleClicked.connect(self._on_double_clicked)
        self._table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self._table)

        self._probs: np.ndarray = np.zeros(0, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_predictions(
        self,
        intervals: np.ndarray,
        probs: Optional[np.ndarray],
        fs: float,
    ) -> None:
        """Re-render the table with `intervals` (k, 2) 1-based and
        per-interval `probs` (k,) in [0, 1].

        `probs` may be `None` (early in inference flow); column shows
        blank if so. The per-interval probability is computed by
        the main window from per-window probs at integration time.
        """
        intervals = (np.asarray(intervals, dtype=np.int64)
                      if intervals is not None
                      else np.zeros((0, 2), dtype=np.int64))
        self._probs = (np.asarray(probs, dtype=np.float32)
                        if probs is not None
                        else np.zeros(intervals.shape[0], dtype=np.float32))
        self._table.setRowCount(intervals.shape[0])
        self._accept_all_btn.setEnabled(intervals.shape[0] > 0)
        for i, (s, e) in enumerate(intervals):
            s, e = int(s), int(e)
            s_sec = (s - 1) / fs
            e_sec = (e - 1) / fs
            dur_sec = (e - s + 1) / fs
            prob_str = (f"{float(self._probs[i]):.3f}"
                         if i < len(self._probs) else "")
            row = [
                str(i), format_time(s_sec), format_time(e_sec),
                format_time(dur_sec), prob_str,
            ]
            for col, text in enumerate(row):
                item = QTableWidgetItem(text)
                if col == 0:
                    item.setData(Qt.UserRole, i)
                self._table.setItem(i, col, item)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_double_clicked(self, item: QTableWidgetItem) -> None:
        idx = self._table.row(item)
        if idx >= 0:
            self.interval_jumped.emit(idx)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            rows = self._table.selectedItems()
            if rows:
                self.interval_accepted.emit(self._table.row(rows[0]))
                event.accept()
                return
        if event.matches(QKeySequence.Delete) or event.key() == Qt.Key_Backspace:
            rows = self._table.selectedItems()
            if rows:
                self.interval_dismissed.emit(self._table.row(rows[0]))
                event.accept()
                return
        super().keyPressEvent(event)

    def _on_context_menu(self, pos) -> None:
        item = self._table.itemAt(pos)
        if item is None:
            return
        idx = self._table.row(item)
        menu = QMenu(self)
        accept_action = QAction("Accept (add to Marked)", self)
        accept_action.triggered.connect(
            lambda: self.interval_accepted.emit(idx)
        )
        dismiss_action = QAction("Dismiss (model-unsure)", self)
        dismiss_action.triggered.connect(
            lambda: self.interval_dismissed.emit(idx)
        )
        jump_action = QAction("Jump to this prediction", self)
        jump_action.triggered.connect(
            lambda: self.interval_jumped.emit(idx)
        )
        menu.addAction(accept_action)
        menu.addAction(jump_action)
        menu.addSeparator()
        menu.addAction(dismiss_action)
        menu.exec(self._table.viewport().mapToGlobal(pos))
