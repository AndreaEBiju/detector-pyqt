"""Multi-folder picker — TDT preprocessing batches need a list of
folders, but Qt's native macOS folder dialog only allows single
selection.

This dialog uses an iterative add-then-confirm pattern: the user
clicks "➕ Add folder…" to open a single-folder dialog (Qt's native
one, so it feels right on each OS), then sees the running list of
picked folders. Per-row Remove buttons let them drop misclicks.
"Continue" returns the accumulated list to the caller; "Cancel"
returns None.

Why not a real multi-select dialog: Qt's `QFileDialog` with the
`DontUseNativeDialog` flag does support multi-select, but the
fallback dialog looks out-of-place on macOS (doesn't match Finder
styling, doesn't show synced Drive folders cleanly). The list-
accumulator pattern is platform-consistent and only one extra click
per folder, which is the right trade for batches up to ~25
recordings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QFileDialog,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
    QVBoxLayout, QWidget,
)


class MultiFolderPicker(QDialog):
    """Modal dialog: returns a `list[Path]` (or None if cancelled)."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        title: str = "Select folders",
        start_dir: Optional[str] = None,
        instructions: Optional[str] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(720, 460)
        self._start_dir = start_dir or ""
        self._folders: list[Path] = []

        outer = QVBoxLayout(self)

        if instructions:
            lbl = QLabel(instructions)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("padding: 4px; color: #ccc;")
            outer.addWidget(lbl)

        # Toolbar row: Add / Remove
        btn_row = QHBoxLayout()
        self._add_btn = QPushButton("➕ Add folder…")
        self._add_btn.clicked.connect(self._on_add)
        self._remove_btn = QPushButton("— Remove selected")
        self._remove_btn.clicked.connect(self._on_remove)
        self._remove_btn.setEnabled(False)
        btn_row.addWidget(self._add_btn)
        btn_row.addWidget(self._remove_btn)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        # Accumulator list
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        outer.addWidget(self._list, stretch=1)

        # OK / Cancel
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self._buttons.button(QDialogButtonBox.Ok).setText("Continue")
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        outer.addWidget(self._buttons)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def folders(self) -> list[Path]:
        """The accumulated folder list. Only meaningful after the
        dialog returns Accepted."""
        return list(self._folders)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_add(self) -> None:
        path_str = QFileDialog.getExistingDirectory(
            self, "Pick a folder", self._start_dir,
        )
        if not path_str:
            return
        path = Path(path_str)
        if path in self._folders:
            return  # dedup
        self._folders.append(path)
        item = QListWidgetItem(f"{path.name}    ·    {path.parent}")
        item.setData(Qt.UserRole, str(path))
        item.setToolTip(str(path))
        self._list.addItem(item)
        # Remember the parent dir as the starting point for the next
        # Add — usually picking sibling folders next.
        self._start_dir = str(path.parent)
        self._update_ok_state()

    def _on_remove(self) -> None:
        rows = [self._list.row(it) for it in self._list.selectedItems()]
        # Remove from end so indices stay valid.
        for idx in sorted(rows, reverse=True):
            self._list.takeItem(idx)
            del self._folders[idx]
        self._update_ok_state()

    def _on_selection_changed(self) -> None:
        self._remove_btn.setEnabled(
            len(self._list.selectedItems()) > 0
        )

    def _update_ok_state(self) -> None:
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(
            len(self._folders) > 0
        )


def pick_folders(
    parent: Optional[QWidget] = None,
    title: str = "Select folders",
    start_dir: Optional[str] = None,
    instructions: Optional[str] = None,
) -> Optional[list[Path]]:
    """Convenience: pop the dialog modally and return the list or None."""
    dialog = MultiFolderPicker(
        parent, title=title, start_dir=start_dir, instructions=instructions,
    )
    if dialog.exec() == QDialog.Accepted:
        return dialog.folders
    return None
