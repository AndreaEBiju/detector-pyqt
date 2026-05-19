"""Queue dock panel — lists the recordings in the active
RecordingQueue with status icons, click-to-open, and a toolbar of
queue actions (Add folder, Load queue, New queue, Open next pending,
Run inference on all pending).

Lifecycle:
- Created by the main window's "Queue → Show queue" menu.
- Holds an in-memory RecordingQueue; main window has a reference too
  so save/load round-trips through the same instance.
- Emits signals when the user wants to open or batch-process items.
  Main window handles by loading the recording and (optionally)
  scheduling background inference.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QMenu,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget, QInputDialog,
)

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from ui.data.queue import RecordingQueue, QueueItem            # noqa: E402


STATUS_ICONS = {
    "pending": "⏳",
    "in_progress": "▶",
    "done": "✓",
    "skipped": "—",
}


class QueuePanel(QWidget):
    """Dockable queue table + toolbar."""

    # The user double-clicked a row → open that recording in the main viewer.
    open_recording = Signal(str)
    # The user clicked "Run inference on all pending" → main window
    # orchestrates a sequential bulk-inference job.
    bulk_inference_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._queue: RecordingQueue = RecordingQueue.new()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Top row: queue identity + counts summary
        self._summary_label = QLabel("")
        self._summary_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._summary_label)

        # Toolbar of queue-level actions
        btn_row = QHBoxLayout()
        self._btn_new = QPushButton("New queue")
        self._btn_new.clicked.connect(self._on_new_queue)
        self._btn_load = QPushButton("Load queue…")
        self._btn_load.clicked.connect(self._on_load_queue)
        self._btn_save = QPushButton("Save")
        self._btn_save.clicked.connect(self._on_save_queue)
        self._btn_add_folder = QPushButton("➕ Add folder…")
        self._btn_add_folder.clicked.connect(self._on_add_folder)
        self._btn_add_files = QPushButton("➕ Add files…")
        self._btn_add_files.clicked.connect(self._on_add_files)
        self._btn_open_next = QPushButton("⏭ Open next pending")
        self._btn_open_next.clicked.connect(self._on_open_next_pending)
        self._btn_bulk_inference = QPushButton("⏯ Run inference on all pending")
        self._btn_bulk_inference.clicked.connect(self._on_bulk_inference)
        btn_row.addWidget(self._btn_new)
        btn_row.addWidget(self._btn_load)
        btn_row.addWidget(self._btn_save)
        btn_row.addWidget(self._btn_add_folder)
        btn_row.addWidget(self._btn_add_files)
        layout.addLayout(btn_row)
        nav_row = QHBoxLayout()
        nav_row.addWidget(self._btn_open_next)
        nav_row.addWidget(self._btn_bulk_inference)
        nav_row.addStretch(1)
        layout.addLayout(nav_row)

        # Table
        self._table = QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels([
            "status", "filename", "last opened", "notes",
        ])
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.itemDoubleClicked.connect(self._on_row_activated)
        self._table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self._table)
        self._render()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def queue(self) -> RecordingQueue:
        return self._queue

    def set_queue(self, queue: RecordingQueue) -> None:
        self._queue = queue
        self._render()

    def mark_status(self, path: str, status: str) -> None:
        """Main window calls this after the user finishes a recording
        (e.g. Save → mark "done") or skips it."""
        if self._queue.mark(Path(path), status):  # type: ignore[arg-type]
            self._queue.save()
            self._render()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        counts = self._queue.counts()
        total = sum(counts.values())
        self._summary_label.setText(
            f"queue {self._queue.queue_id}  ·  {total} items  "
            f"·  {counts['done']} done, {counts['pending']} pending, "
            f"{counts['skipped']} skipped"
        )
        self._table.setRowCount(len(self._queue.items))
        for i, it in enumerate(self._queue.items):
            icon = STATUS_ICONS.get(it.status, "?")
            status_cell = QTableWidgetItem(f"{icon} {it.status}")
            self._table.setItem(i, 0, status_cell)
            name_cell = QTableWidgetItem(Path(it.path).name)
            name_cell.setToolTip(it.path)
            self._table.setItem(i, 1, name_cell)
            opened = it.last_opened or "—"
            self._table.setItem(i, 2, QTableWidgetItem(opened))
            self._table.setItem(i, 3, QTableWidgetItem(it.notes))

    # ------------------------------------------------------------------
    # Toolbar actions
    # ------------------------------------------------------------------

    def _on_new_queue(self) -> None:
        if self._queue.items:
            resp = QMessageBox.question(
                self, "New queue?",
                "Replace the current queue? Make sure to save first if you "
                "want to keep it.",
            )
            if resp != QMessageBox.Yes:
                return
        self._queue = RecordingQueue.new()
        self._render()

    def _on_load_queue(self) -> None:
        candidates = RecordingQueue.list_all()
        if not candidates:
            QMessageBox.information(self, "No saved queues",
                                      "No queue files in ~/.detector/queues/.")
            return
        ids = [qid for qid, _ in candidates]
        chosen, ok = QInputDialog.getItem(
            self, "Load queue", "Pick a saved queue:", ids, 0, False,
        )
        if not ok:
            return
        try:
            self._queue = RecordingQueue.load_by_id(chosen)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        self._render()

    def _on_save_queue(self) -> None:
        try:
            p = self._queue.save()
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        QMessageBox.information(self, "Saved", f"Queue saved to:\n{p}")

    def _on_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Pick a folder")
        if not folder:
            return
        # Default to *.h5 — the splitter / ingest format. The user can
        # override via add_files for .mat sources.
        added = self._queue.add_folder(Path(folder), glob="*.h5")
        if added == 0:
            QMessageBox.information(
                self, "Nothing added",
                f"No new .h5 files in {folder} (or all already in queue).",
            )
        self._queue.save()
        self._render()

    def _on_add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Pick recording files", "",
            "Recordings (*.mat *.h5);;All files (*)",
        )
        if not paths:
            return
        added = self._queue.add_paths(Path(p) for p in paths)
        self._queue.save()
        self._render()

    def _on_open_next_pending(self) -> None:
        item = self._queue.next_pending()
        if item is None:
            QMessageBox.information(self, "Queue complete",
                                      "No pending items left.")
            return
        # Mark in_progress eagerly — the main window flips to done on
        # save, skipped on user request, or leaves in_progress if the
        # user opens but doesn't finish.
        self._queue.mark(Path(item.path), "in_progress")
        self._queue.save()
        self._render()
        self.open_recording.emit(item.path)

    def _on_bulk_inference(self) -> None:
        pending = [it for it in self._queue.items if it.status == "pending"]
        if not pending:
            QMessageBox.information(
                self, "Nothing to do",
                "No pending recordings in the queue.",
            )
            return
        resp = QMessageBox.question(
            self, "Run bulk inference?",
            f"This will run inference sequentially on {len(pending)} "
            "pending recording(s). It may take a while. Results are "
            "saved alongside each recording. Continue?",
        )
        if resp != QMessageBox.Yes:
            return
        self.bulk_inference_requested.emit()

    # ------------------------------------------------------------------
    # Per-row context menu
    # ------------------------------------------------------------------

    def _on_row_activated(self, item: QTableWidgetItem) -> None:
        idx = self._table.row(item)
        if idx < 0 or idx >= len(self._queue.items):
            return
        q_item = self._queue.items[idx]
        self._queue.mark(Path(q_item.path), "in_progress")
        self._queue.save()
        self._render()
        self.open_recording.emit(q_item.path)

    def _on_context_menu(self, pos) -> None:
        item = self._table.itemAt(pos)
        if item is None:
            return
        idx = self._table.row(item)
        q_item = self._queue.items[idx]
        menu = QMenu(self)
        open_action = QAction("Open recording", self)
        open_action.triggered.connect(
            lambda: self._on_row_activated(item)
        )
        mark_done = QAction("Mark done", self)
        mark_done.triggered.connect(
            lambda: (self._queue.mark(Path(q_item.path), "done"),
                     self._queue.save(), self._render())
        )
        mark_skipped = QAction("Mark skipped…", self)
        mark_skipped.triggered.connect(
            lambda: self._prompt_skip(idx)
        )
        mark_pending = QAction("Reset to pending", self)
        mark_pending.triggered.connect(
            lambda: (self._queue.mark(Path(q_item.path), "pending"),
                     self._queue.save(), self._render())
        )
        remove_action = QAction("Remove from queue", self)
        remove_action.triggered.connect(
            lambda: (self._queue.remove_path(Path(q_item.path)),
                     self._queue.save(), self._render())
        )
        menu.addAction(open_action)
        menu.addSeparator()
        menu.addAction(mark_done)
        menu.addAction(mark_skipped)
        menu.addAction(mark_pending)
        menu.addSeparator()
        menu.addAction(remove_action)
        menu.exec(self._table.viewport().mapToGlobal(pos))

    def _prompt_skip(self, idx: int) -> None:
        q_item = self._queue.items[idx]
        reason, ok = QInputDialog.getText(
            self, "Skip reason",
            f"Why are you skipping {Path(q_item.path).name}?",
        )
        if not ok:
            return
        self._queue.mark(Path(q_item.path), "skipped", notes=reason)
        self._queue.save()
        self._render()
