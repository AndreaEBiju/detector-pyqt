"""Region table — a QTableWidget showing the currently marked bad
intervals.

UI semantics:
- Columns: `#`, `start`, `end`, `Δ`, `source`.
- Times rendered as `mm:ss.fff` (matches the Streamlit UI).
- `source` shows the per-interval provenance string (user / existing /
  model_accepted etc.) so the user can see what came from where.
- Double-click row → emits `interval_jumped(idx)`. Main window centres
  the viewer on that interval.
- Delete key (or right-click → Delete) → emits `interval_deleted(idx)`.
  Main window removes the interval and re-renders.

The widget holds NO authoritative state — it's a view over the
intervals array owned by the main window. Call `set_intervals()` after
the main window's data changes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView, QHeaderView, QMenu, QTableWidget, QTableWidgetItem,
)


def format_time(t_sec: float) -> str:
    """`12.345` → `00:12.345`. Matches the Streamlit region table."""
    if t_sec < 0:
        return "-" + format_time(-t_sec)
    minutes = int(t_sec // 60)
    seconds = t_sec - minutes * 60
    return f"{minutes:02d}:{seconds:06.3f}"


class RegionTable(QTableWidget):
    """Sortable, single-select table of bad intervals."""

    # Emitted when the user wants to jump the viewer to the interval
    # at row `idx`. Main window decides what "jump" means.
    interval_jumped = Signal(int)
    # Emitted on Delete-key press or right-click → Delete.
    interval_deleted = Signal(int)

    COLUMNS = ("#", "start", "end", "Δ", "source")

    def __init__(self, parent: Optional[object] = None):
        super().__init__(0, len(self.COLUMNS), parent)
        self.setHorizontalHeaderLabels(self.COLUMNS)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(False)
        # Stretch the start/end/Δ columns; fix # and source.
        hdr = self.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.verticalHeader().setVisible(False)
        self.itemDoubleClicked.connect(self._on_double_clicked)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_intervals(
        self,
        intervals: np.ndarray,
        sources: Optional[list[str]] = None,
        fs: float = 1.0,
    ) -> None:
        """Replace the table contents with `intervals` × `sources`.

        `intervals` is `(k, 2)` 1-based inclusive sample indices.
        `sources` is a parallel list of provenance strings; if None or
        shorter than `intervals`, missing entries display as 'unknown'.
        """
        intervals = (np.asarray(intervals, dtype=np.int64)
                      if intervals is not None
                      else np.zeros((0, 2), dtype=np.int64))
        sources = list(sources or [])
        self.setRowCount(intervals.shape[0])
        for i, (s, e) in enumerate(intervals):
            s, e = int(s), int(e)
            s_sec = (s - 1) / fs
            e_sec = (e - 1) / fs
            dur_sec = (e - s + 1) / fs
            src = sources[i] if i < len(sources) else "unknown"
            row = [
                str(i),
                format_time(s_sec),
                format_time(e_sec),
                format_time(dur_sec),
                src,
            ]
            for col, text in enumerate(row):
                item = QTableWidgetItem(text)
                if col == 0:
                    item.setData(Qt.UserRole, i)  # stable row index
                self.setItem(i, col, item)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_double_clicked(self, item: QTableWidgetItem) -> None:
        idx = self.row(item)
        if idx >= 0:
            self.interval_jumped.emit(idx)

    def keyPressEvent(self, event) -> None:
        if event.matches(QKeySequence.Delete) or event.key() == Qt.Key_Backspace:
            rows = self.selectedItems()
            if rows:
                idx = self.row(rows[0])
                self.interval_deleted.emit(idx)
                event.accept()
                return
        super().keyPressEvent(event)

    def _on_context_menu(self, pos) -> None:
        item = self.itemAt(pos)
        if item is None:
            return
        idx = self.row(item)
        menu = QMenu(self)
        jump_action = QAction("Jump to this interval", self)
        jump_action.triggered.connect(lambda: self.interval_jumped.emit(idx))
        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(lambda: self.interval_deleted.emit(idx))
        menu.addAction(jump_action)
        menu.addSeparator()
        menu.addAction(delete_action)
        menu.exec(self.viewport().mapToGlobal(pos))
