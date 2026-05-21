"""Predictions panel — table of model-predicted bad intervals with
Accept / Dismiss / Accept-all buttons.

Sits in the right dock alongside the user-marked region table. Lives
side-by-side via a QTabWidget; user toggles between "Marked" (the
authoritative bad-intervals list) and "Model" (the model's
predictions that haven't been accepted or dismissed yet).

Behavior:
- One row per model interval with start/end/duration and probability.
- Multi-select supported: Ctrl+click / Shift+click select multiple
  rows. Buttons "Accept N selected" / "Dismiss N selected" act on
  the whole selection at once. Single-row interactions (context-
  menu, Enter, Delete) emit a 1-element list under the same signal
  so the wiring is uniform.
- "Accept" → appends each selected interval to the bad-intervals
  list (with source="model_accepted") and removes them from the
  predictions list.
- "Dismiss" → removes selected rows from predictions WITHOUT
  marking. Anything still in predictions at save time is the
  "model_unsure" set the save flow writes to
  `removedSegmentIdx_model_unsure`.
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
    QAbstractItemView, QHBoxLayout, QHeaderView, QMenu, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .region_table import format_time


class PredictionsPanel(QWidget):
    """Composite widget: QPushButton row + QTableWidget."""

    # Emitted when user accepts/dismisses one or more rows. Payload
    # is the list of row indices (always sorted ascending; a single-
    # row interaction is just a 1-element list). Main window does the
    # actual move (append to bad_intervals with source="model_accepted"
    # + remove from predictions + re-render).
    interval_accepted = Signal(list)
    interval_dismissed = Signal(list)
    interval_jumped = Signal(int)
    accept_all_requested = Signal()

    COLUMNS = ("#", "start", "end", "Δ", "prob")

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Bulk action row — Accept selected / Dismiss selected /
        # Accept ALL. The first two operate on whichever rows are
        # selected in the table below. ALL ignores the selection.
        bulk_row = QHBoxLayout()
        self._accept_sel_btn = QPushButton("✓ Accept selected")
        self._accept_sel_btn.setToolTip(
            "Append every selected prediction to the Marked list. "
            "Use Ctrl+click / Shift+click in the table to pick more "
            "than one row."
        )
        self._accept_sel_btn.clicked.connect(
            lambda: self._emit_for_selection(self.interval_accepted)
        )
        self._accept_sel_btn.setEnabled(False)
        bulk_row.addWidget(self._accept_sel_btn)

        self._dismiss_sel_btn = QPushButton("✗ Dismiss selected")
        self._dismiss_sel_btn.setToolTip(
            "Remove every selected prediction from the Model list "
            "without marking. Dismissed-and-not-accepted intervals "
            "land in the model_unsure set at save time."
        )
        self._dismiss_sel_btn.clicked.connect(
            lambda: self._emit_for_selection(self.interval_dismissed)
        )
        self._dismiss_sel_btn.setEnabled(False)
        bulk_row.addWidget(self._dismiss_sel_btn)
        bulk_row.addStretch(1)
        layout.addLayout(bulk_row)

        self._accept_all_btn = QPushButton("➕ Accept ALL into marked")
        self._accept_all_btn.clicked.connect(self.accept_all_requested.emit)
        self._accept_all_btn.setEnabled(False)
        layout.addWidget(self._accept_all_btn)

        self._table = QTableWidget(0, len(self.COLUMNS), self)
        self._table.setHorizontalHeaderLabels(self.COLUMNS)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        # ExtendedSelection: Ctrl+click toggles, Shift+click extends
        # range — the standard Qt multi-row select. Replaces the
        # previous SingleSelection.
        self._table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
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

    def _selected_row_indices(self) -> list[int]:
        """Sorted-ascending unique list of currently-selected row
        indices. Centralised so every action (button, keyboard,
        context menu) reads the selection the same way."""
        rows = {idx.row() for idx in self._table.selectionModel().selectedRows()}
        return sorted(rows)

    def _emit_for_selection(self, signal) -> None:
        """Emit `signal` (interval_accepted or interval_dismissed)
        with the current selection. No-op if nothing is selected
        (the buttons are disabled in that case, but defensive)."""
        rows = self._selected_row_indices()
        if rows:
            signal.emit(rows)

    def _on_selection_changed(self) -> None:
        """Enable/disable bulk-action buttons based on selection
        count. The label updates so the user can see how many rows
        the next click will affect."""
        n = len(self._selected_row_indices())
        self._accept_sel_btn.setEnabled(n > 0)
        self._dismiss_sel_btn.setEnabled(n > 0)
        if n == 0:
            self._accept_sel_btn.setText("✓ Accept selected")
            self._dismiss_sel_btn.setText("✗ Dismiss selected")
        else:
            self._accept_sel_btn.setText(f"✓ Accept {n} selected")
            self._dismiss_sel_btn.setText(f"✗ Dismiss {n} selected")

    def _on_double_clicked(self, item: QTableWidgetItem) -> None:
        idx = self._table.row(item)
        if idx >= 0:
            self.interval_jumped.emit(idx)

    def keyPressEvent(self, event) -> None:
        # Enter / Delete operate on the entire selection — single or
        # multi. Lists of one are still wrapped as lists so the
        # main-window handler signature is uniform.
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            rows = self._selected_row_indices()
            if rows:
                self.interval_accepted.emit(rows)
                event.accept()
                return
        if event.matches(QKeySequence.Delete) or event.key() == Qt.Key_Backspace:
            rows = self._selected_row_indices()
            if rows:
                self.interval_dismissed.emit(rows)
                event.accept()
                return
        super().keyPressEvent(event)

    def _on_context_menu(self, pos) -> None:
        item = self._table.itemAt(pos)
        if item is None:
            return
        clicked_idx = self._table.row(item)
        # If the user right-clicked OUTSIDE their existing selection,
        # treat it as a fresh single-row action (don't accidentally
        # apply to whatever else was selected). If they right-clicked
        # INSIDE the selection, the action applies to the whole set.
        selected = self._selected_row_indices()
        if clicked_idx in selected:
            target = selected
        else:
            target = [clicked_idx]
        n = len(target)
        menu = QMenu(self)
        accept_label = (
            "Accept (add to Marked)" if n == 1
            else f"Accept {n} selected (add to Marked)"
        )
        dismiss_label = (
            "Dismiss (model-unsure)" if n == 1
            else f"Dismiss {n} selected (model-unsure)"
        )
        accept_action = QAction(accept_label, self)
        accept_action.triggered.connect(
            lambda: self.interval_accepted.emit(target)
        )
        dismiss_action = QAction(dismiss_label, self)
        dismiss_action.triggered.connect(
            lambda: self.interval_dismissed.emit(target)
        )
        # Jump always applies to the right-clicked row only — the
        # viewport can't show >1 location at once.
        jump_action = QAction("Jump to this prediction", self)
        jump_action.triggered.connect(
            lambda: self.interval_jumped.emit(clicked_idx)
        )
        menu.addAction(accept_action)
        menu.addAction(jump_action)
        menu.addSeparator()
        menu.addAction(dismiss_action)
        menu.exec(self._table.viewport().mapToGlobal(pos))
