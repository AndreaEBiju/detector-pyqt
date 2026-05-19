"""Batch setup table — row per TDT folder with editable condition /
output name / animal ID. Owns the per-row state model that the
preprocess window reads when transitioning to the per-animal review
loop.

Inferred-condition heuristics from
`detector.preprocessing.tdt_io.detect_naming_convention` pre-fill
the dropdowns; the user overrides per row.

The "Has profile?" column reads `~/.detector/preprocessing_profiles/
<animal_id>.json` reactively as the user types the animal ID — a
checkmark means the next step's channel-assignment + notch-review
will be pre-populated from that profile.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QWidget,
)

_repo_root = Path(__file__).resolve().parent.parent.parent
_detector_core = _repo_root / "detector-core"
if _detector_core.exists() and str(_detector_core) not in sys.path:
    sys.path.insert(0, str(_detector_core))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from detector.preprocessing.tdt_io import detect_naming_convention   # noqa: E402
from detector.preprocessing.profiles import Profile                   # noqa: E402


CONDITION_OPTIONS = ["baseline", "stim", "recovery", "stim_rec"]
PROFILE_PRESENT = "✓"
PROFILE_ABSENT = "—"


@dataclass
class BatchRowState:
    """Per-row mutable state — what the preprocess window reads when
    it moves to the per-animal review."""
    folder: Path
    inferred_condition: Optional[str]
    condition: Optional[str]
    output_condition_name: str
    animal_id: str = ""

    def is_valid(self) -> bool:
        return (
            self.condition is not None
            and bool(self.output_condition_name.strip())
            and bool(self.animal_id.strip())
        )


class TdtBatchTable(QTableWidget):
    """Editable table backing the preprocess window's Step 2."""

    # Emitted whenever a row changes — the parent window uses this to
    # enable/disable the Next button based on overall validity.
    row_changed = Signal()

    COLS = ("Folder", "Inferred", "Condition", "Output name", "Animal ID", "Profile?")

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(0, len(self.COLS), parent)
        self.setHorizontalHeaderLabels(self.COLS)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        # Cells are individually editable via custom widgets, not via
        # the QTableWidget's default in-place editor.
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setAlternatingRowColors(True)
        hdr = self.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.Stretch)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.verticalHeader().setVisible(False)
        self._rows: list[BatchRowState] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def rows(self) -> list[BatchRowState]:
        return list(self._rows)

    def set_folders(self, folders: list[Path]) -> None:
        """Replace the table contents with `folders`. Each folder gets
        a row with inferred-condition pre-filled."""
        self.setRowCount(0)
        self._rows = []
        for folder in folders:
            inferred = detect_naming_convention(folder)
            state = BatchRowState(
                folder=folder,
                inferred_condition=inferred["inferred_condition"],
                condition=inferred["inferred_condition"],
                output_condition_name=inferred["inferred_label"] or folder.name,
                animal_id="",
            )
            self._rows.append(state)
            self._render_row(len(self._rows) - 1)
        self.row_changed.emit()

    def all_valid(self) -> bool:
        """Every row has a condition + output name + animal_id."""
        if not self._rows:
            return False
        if not all(r.is_valid() for r in self._rows):
            return False
        # Output condition names must be unique within the batch.
        names = [r.output_condition_name.strip() for r in self._rows]
        return len(set(names)) == len(names)

    def validation_message(self) -> Optional[str]:
        """Human-readable description of what's blocking the Next
        button, or None if all good."""
        if not self._rows:
            return "Add at least one folder."
        missing_condition = [i for i, r in enumerate(self._rows) if not r.condition]
        if missing_condition:
            return f"Pick a condition for row(s): {missing_condition}"
        missing_animal = [i for i, r in enumerate(self._rows) if not r.animal_id.strip()]
        if missing_animal:
            return f"Fill animal ID for row(s): {missing_animal}"
        missing_name = [i for i, r in enumerate(self._rows) if not r.output_condition_name.strip()]
        if missing_name:
            return f"Fill output name for row(s): {missing_name}"
        names = [r.output_condition_name.strip() for r in self._rows]
        seen, dupes = set(), []
        for i, n in enumerate(names):
            if n in seen:
                dupes.append(n)
            seen.add(n)
        if dupes:
            return f"Output names must be unique. Duplicates: {sorted(set(dupes))}"
        return None

    def animal_ids_in_order(self) -> list[str]:
        """The list of unique animal IDs in the order they first
        appear in the table. Used by the per-animal review loop."""
        seen: list[str] = []
        for r in self._rows:
            if r.animal_id and r.animal_id not in seen:
                seen.append(r.animal_id)
        return seen

    # ------------------------------------------------------------------
    # Row rendering
    # ------------------------------------------------------------------

    def _render_row(self, idx: int) -> None:
        if self.rowCount() <= idx:
            self.insertRow(idx)
        state = self._rows[idx]

        # Folder (read-only)
        folder_item = QTableWidgetItem(state.folder.name)
        folder_item.setToolTip(str(state.folder))
        self.setItem(idx, 0, folder_item)

        # Inferred (read-only)
        inferred_text = state.inferred_condition or "—"
        inferred_item = QTableWidgetItem(inferred_text)
        inferred_item.setForeground(Qt.gray)
        self.setItem(idx, 1, inferred_item)

        # Condition (combo)
        condition_combo = QComboBox()
        condition_combo.addItem("(unset)", None)
        for opt in CONDITION_OPTIONS:
            condition_combo.addItem(opt, opt)
        if state.condition is not None:
            i = condition_combo.findData(state.condition)
            if i >= 0:
                condition_combo.setCurrentIndex(i)
        condition_combo.currentIndexChanged.connect(
            lambda _i, row=idx: self._on_condition_changed(row)
        )
        self.setCellWidget(idx, 2, condition_combo)

        # Output name (line edit)
        name_edit = QLineEdit(state.output_condition_name)
        name_edit.textChanged.connect(
            lambda text, row=idx: self._on_name_changed(row, text)
        )
        self.setCellWidget(idx, 3, name_edit)

        # Animal ID (line edit) — reactive: typing updates the
        # profile-present indicator.
        animal_edit = QLineEdit(state.animal_id)
        animal_edit.setPlaceholderText("required")
        animal_edit.textChanged.connect(
            lambda text, row=idx: self._on_animal_changed(row, text)
        )
        self.setCellWidget(idx, 4, animal_edit)

        # Profile? (read-only icon)
        profile_item = QTableWidgetItem(PROFILE_ABSENT)
        profile_item.setTextAlignment(Qt.AlignCenter)
        self.setItem(idx, 5, profile_item)
        self._refresh_profile_indicator(idx)

    def _refresh_profile_indicator(self, idx: int) -> None:
        animal_id = self._rows[idx].animal_id.strip()
        item = self.item(idx, 5)
        if item is None:
            return
        if not animal_id:
            item.setText(PROFILE_ABSENT)
            item.setToolTip("Animal ID empty")
            return
        prof = Profile.load(animal_id)
        if prof is not None:
            item.setText(PROFILE_PRESENT)
            item.setToolTip(
                f"Profile exists at {Profile.path_for(animal_id)}"
            )
        else:
            item.setText(PROFILE_ABSENT)
            item.setToolTip(
                f"No profile for {animal_id!r} yet — you'll set "
                "channel + notch in the review step"
            )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_condition_changed(self, idx: int) -> None:
        combo = self.cellWidget(idx, 2)
        if combo is None:
            return
        self._rows[idx].condition = combo.currentData()
        self.row_changed.emit()

    def _on_name_changed(self, idx: int, text: str) -> None:
        self._rows[idx].output_condition_name = text
        self.row_changed.emit()

    def _on_animal_changed(self, idx: int, text: str) -> None:
        self._rows[idx].animal_id = text
        self._refresh_profile_indicator(idx)
        self.row_changed.emit()
