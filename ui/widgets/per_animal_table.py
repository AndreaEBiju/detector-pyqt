"""Per-animal grouping table — view + edit the animal letter assigned
to each recording in the training manifest.

Rows are training-manifest recordings (plus optionally a few "extra"
files added via the per-animal tab's file-picker). Columns:

    Animal | Recording ID | rec_type | held_out | extra

Only the Animal column is editable. Edits emit
`animal_edited(recording_id, new_letter)` so the parent
(training_window) can persist them back to the manifest.

The widget holds NO authoritative state -- it's a view over the
recordings list owned by the training window. Call `set_recordings()`
after the parent's data changes (e.g. after the user re-loads the
manifest or after an animal edit triggers a re-save).

Summary stats (eligible vs skipped animal count) are computed live
from the current table contents, NOT from the original manifest,
so the user sees the impact of their edits before saving. That's
exposed via `summary()` so the parent can put it in a QLabel.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem,
)


COLUMNS = ("Animal", "Recording ID", "rec_type", "held_out", "extra")
ANIMAL_COL = 0
RID_COL = 1
RECTYPE_COL = 2
HELDOUT_COL = 3
EXTRA_COL = 4


def normalize_animal_letter(text: str) -> Optional[str]:
    """Strip whitespace + uppercase first char. Returns None if the
    input has no letter."""
    text = (text or "").strip()
    if not text:
        return None
    first = text[0]
    if not first.isalpha():
        return None
    return first.upper()


def compute_summary(
    recordings: list[dict],
    *,
    min_recordings_per_animal: int = 3,
) -> dict:
    """Return {n_eligible, n_skipped, groups, unknown} for a list of
    rec dicts. Recordings WITHOUT an animal letter land in
    `unknown` (not counted in groups). Held-out recordings are still
    shown in the table but never count toward eligibility -- they
    don't enter training.

    `groups` is {letter: count_of_non_held_out_recordings}.
    """
    groups: dict[str, int] = {}
    unknown = 0
    for r in recordings:
        if r.get("held_out", False):
            continue
        letter = normalize_animal_letter(str(r.get("animal") or ""))
        if not letter:
            unknown += 1
            continue
        groups[letter] = groups.get(letter, 0) + 1
    n_eligible = sum(1 for n in groups.values()
                     if n >= min_recordings_per_animal)
    n_skipped = sum(1 for n in groups.values()
                    if n < min_recordings_per_animal)
    return {
        "n_eligible": n_eligible,
        "n_skipped": n_skipped,
        "unknown": unknown,
        "groups": groups,
    }


class PerAnimalTable(QTableWidget):
    """Editable grouping table for per-animal training.

    Only the Animal column accepts edits; the rest are read-only.
    A row whose `recording_id` was passed in via the `extra` list
    gets the 'extra' column ticked, which the training window uses
    to keep those files out of the combined manifest.
    """

    # (recording_id, new_letter_or_None). Fires when the user finishes
    # editing the Animal column for a row.
    animal_edited = Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(0, len(COLUMNS), parent)
        self.setHorizontalHeaderLabels(COLUMNS)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setAlternatingRowColors(True)
        # Only the Animal column is editable. The default trigger is
        # NoEditTriggers (set globally), but we re-enable double-click +
        # Enter editing per-item by clearing the non-editable flag on
        # just that cell. See `_set_animal_cell`.
        self.setEditTriggers(
            QAbstractItemView.DoubleClicked
            | QAbstractItemView.EditKeyPressed
            | QAbstractItemView.SelectedClicked
        )
        hdr = self.horizontalHeader()
        hdr.setSectionResizeMode(ANIMAL_COL, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(RID_COL, QHeaderView.Stretch)
        hdr.setSectionResizeMode(RECTYPE_COL, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(HELDOUT_COL, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(EXTRA_COL, QHeaderView.ResizeToContents)

        # Cached state. `_recordings` is the list of dicts mirroring
        # what was passed in via set_recordings; we keep our own copy
        # so summary() can stay in sync with edits without re-asking
        # the parent.
        self._recordings: list[dict] = []
        self._min_recordings_per_animal: int = 3
        # Suppress itemChanged while we populate rows programmatically.
        self._populating: bool = False

        self.itemChanged.connect(self._on_item_changed)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def set_min_recordings_per_animal(self, n: int) -> None:
        self._min_recordings_per_animal = int(n)

    def set_recordings(self, recordings: list[dict]) -> None:
        """Replace the table contents. Each dict must have
        recording_id; `animal`, `rec_type`, `held_out`, `extra` are
        optional and default to sensible values."""
        self._populating = True
        try:
            # Sort: known animals alphabetically first, then unknown,
            # then by recording_id within each group. Held-out recordings
            # sort to the bottom of their animal group so they don't
            # clutter the eligible view.
            def _key(r: dict):
                letter = normalize_animal_letter(str(r.get("animal") or ""))
                return (
                    letter is None,        # unknown last
                    letter or "",
                    1 if r.get("held_out") else 0,
                    str(r.get("recording_id", "")),
                )
            self._recordings = sorted(list(recordings), key=_key)

            self.setRowCount(len(self._recordings))
            for i, r in enumerate(self._recordings):
                self._set_animal_cell(i, str(r.get("animal") or ""))
                self._set_readonly(i, RID_COL, str(r.get("recording_id", "")))
                self._set_readonly(i, RECTYPE_COL, str(r.get("rec_type", "")))
                held = bool(r.get("held_out", False))
                self._set_readonly(i, HELDOUT_COL, "✓" if held else "")
                extra = bool(r.get("extra", False))
                self._set_readonly(i, EXTRA_COL, "✓" if extra else "")
                if held or extra:
                    # Slight visual distinction for non-training rows.
                    for col in range(self.columnCount()):
                        it = self.item(i, col)
                        if it is not None:
                            it.setForeground(QBrush(QColor("#888")))
        finally:
            self._populating = False

    def recordings(self) -> list[dict]:
        """Return the current recordings list (sorted by display order)."""
        return list(self._recordings)

    def summary(self) -> dict:
        """Stats computed from current in-table state -- not the
        original passed-in recordings, so edits show up immediately."""
        return compute_summary(
            self._recordings,
            min_recordings_per_animal=self._min_recordings_per_animal,
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _set_animal_cell(self, row: int, text: str) -> None:
        item = QTableWidgetItem(text or "")
        item.setTextAlignment(Qt.AlignCenter)
        # editable -- the default for QTableWidgetItem
        self.setItem(row, ANIMAL_COL, item)

    def _set_readonly(self, row: int, col: int, text: str) -> None:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        if col != RID_COL:
            item.setTextAlignment(Qt.AlignCenter)
        self.setItem(row, col, item)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._populating:
            return
        if item.column() != ANIMAL_COL:
            return
        row = item.row()
        if row < 0 or row >= len(self._recordings):
            return
        new_letter = normalize_animal_letter(item.text())
        # Snap displayed value to the normalized form so the user sees
        # what got saved (e.g. "jel" -> "J", "  L " -> "L").
        self._populating = True
        try:
            item.setText(new_letter or "")
        finally:
            self._populating = False
        # Update the cached recording so subsequent summary() calls and
        # set_recordings round-trips reflect the edit.
        rid = str(self._recordings[row].get("recording_id", ""))
        self._recordings[row]["animal"] = new_letter
        self.animal_edited.emit(rid, new_letter)
