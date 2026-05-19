"""Channel-assignment dialog — for one TDT folder (the first one
belonging to the animal under review), let the user pick:

- Which stream is the raw signal (radio choice from streams with
  >1 channel typically).
- Which channels in that stream to include (defaults to all).
- The role (nerve / stomach / other) of each included channel.
- The human-readable label (VN1, VN2, Ant1, …).
- Auxiliary streams: stim, vibration, stim envelope.

The plan's defaults pre-fill from heuristics (Raww / BiPl / adc1 /
ADC2). The 2-nerve + 3-stomach role validation warns but doesn't
block — the user clicks through with a confirmation if they have a
different rig setup.

Returns a `channel_assignment` dict ready to drop into a
`detector.preprocessing.profiles.Profile`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

_repo_root = Path(__file__).resolve().parent.parent.parent
_detector_core = _repo_root / "detector-core"
if _detector_core.exists() and str(_detector_core) not in sys.path:
    sys.path.insert(0, str(_detector_core))

from detector.preprocessing.tdt_io import StreamInfo                # noqa: E402


# Plan-default stream names — guessed if the actual stream list
# contains them. The user can override via the dropdowns.
DEFAULT_RAW_PREFIXES = ("Raw",)                      # matches "Raww"
DEFAULT_STIM = ("BiPl", "stim", "Stim")
DEFAULT_VIBRATION = ("adc1", "ADC1", "vib")
DEFAULT_ENVELOPE = ("ADC2", "adc2", "env")

ROLE_OPTIONS = ("nerve", "stomach", "other", "exclude")
ROLE_NERVE = "nerve"
ROLE_STOMACH = "stomach"

# Match the labels the detector backend's CHANNEL_NAMES uses so
# what gets saved here is consistent with the main viewer's display.
DEFAULT_NERVE_LABELS = ("VN1", "VN2", "VN3", "VN4")
DEFAULT_STOMACH_LABELS = ("Ant1", "Ant2", "Ant3", "Ant4", "Ant5")


def _guess_stream(streams: dict[str, StreamInfo],
                   candidates: tuple[str, ...]) -> Optional[str]:
    """Pick the first stream in `streams` whose name matches any of
    `candidates`. Matches exactly first, then by prefix."""
    for c in candidates:
        if c in streams:
            return c
    for c in candidates:
        for name in streams:
            if name.startswith(c):
                return name
    return None


class ChannelAssignmentDialog(QDialog):
    """Modal channel-assignment dialog. Built once per animal in the
    per-animal review loop (or once total when the batch's electrode
    consistency is "Yes")."""

    def __init__(
        self,
        animal_id: str,
        streams: dict[str, StreamInfo],
        existing_assignment: Optional[dict] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Channel assignment — {animal_id}")
        self.resize(820, 620)
        self._streams = streams
        self._animal_id = animal_id

        outer = QVBoxLayout(self)

        # Heading
        outer.addWidget(QLabel(
            f"<b>{animal_id}</b> — pick the streams + channels to "
            "preprocess for this animal. Defaults are filled from the "
            "plan's stream-name heuristics; override anything that's wrong."
        ))

        # Stream selection group (raw + aux)
        stream_box = QGroupBox("Streams")
        form = QFormLayout(stream_box)

        # The raw-signal stream picker. Filter to streams with >1
        # channel (a single-channel "raw" stream wouldn't fit the
        # 5-channel detector layout).
        multi_ch = [name for name, info in streams.items() if info.n_channels > 1]
        self._raw_combo = QComboBox()
        for name in multi_ch:
            self._raw_combo.addItem(
                f"{name}  ({streams[name].n_channels} ch · "
                f"{streams[name].fs:.0f} Hz)", name,
            )
        if not multi_ch:
            # Worst case: no multi-channel stream. Fall back to all.
            for name in streams:
                self._raw_combo.addItem(name, name)
        form.addRow("Raw signal stream", self._raw_combo)

        # Aux streams — single picks each.
        def _aux_combo() -> QComboBox:
            c = QComboBox()
            c.addItem("(none)", None)
            for name, info in streams.items():
                c.addItem(
                    f"{name}  ({info.n_channels} ch · "
                    f"{info.fs:.0f} Hz)", name,
                )
            return c

        self._stim_combo = _aux_combo()
        self._vib_combo = _aux_combo()
        self._env_combo = _aux_combo()
        form.addRow("Stim stream", self._stim_combo)
        form.addRow("Vibration stream", self._vib_combo)
        form.addRow("Stim envelope stream", self._env_combo)
        outer.addWidget(stream_box)

        # Apply defaults or existing assignment.
        if existing_assignment:
            self._apply_existing(existing_assignment)
        else:
            self._apply_defaults()

        # Channel table (rebuilt when the raw-stream choice changes).
        self._channels_table = QTableWidget(0, 4)
        self._channels_table.setHorizontalHeaderLabels(
            ("Include", "TDT channel", "Role", "Label")
        )
        self._channels_table.verticalHeader().setVisible(False)
        hdr = self._channels_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.Stretch)
        outer.addWidget(self._channels_table, stretch=1)

        # Validation summary
        self._summary_label = QLabel("")
        self._summary_label.setStyleSheet("padding: 4px;")
        outer.addWidget(self._summary_label)

        # OK / Cancel
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self._buttons.button(QDialogButtonBox.Ok).setText("Accept channel assignment")
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        outer.addWidget(self._buttons)

        # Re-render the channel table whenever the raw stream changes
        # (and once now to populate the initial selection).
        self._raw_combo.currentIndexChanged.connect(self._rebuild_channels_table)
        self._rebuild_channels_table()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _apply_defaults(self) -> None:
        # Set raw to first matching name.
        raw_guess = _guess_stream(self._streams, DEFAULT_RAW_PREFIXES)
        if raw_guess:
            i = self._raw_combo.findData(raw_guess)
            if i >= 0:
                self._raw_combo.setCurrentIndex(i)
        for combo, candidates in [
            (self._stim_combo, DEFAULT_STIM),
            (self._vib_combo, DEFAULT_VIBRATION),
            (self._env_combo, DEFAULT_ENVELOPE),
        ]:
            guess = _guess_stream(self._streams, candidates)
            if guess:
                i = combo.findData(guess)
                if i >= 0:
                    combo.setCurrentIndex(i)

    def _apply_existing(self, existing: dict) -> None:
        for combo, key in [
            (self._raw_combo, "raw_stream"),
            (self._stim_combo, "stim_stream"),
            (self._vib_combo, "vibration_stream"),
            (self._env_combo, "stim_envelope_stream"),
        ]:
            name = existing.get(key)
            if name:
                i = combo.findData(name)
                if i >= 0:
                    combo.setCurrentIndex(i)

    def _rebuild_channels_table(self) -> None:
        raw_name = self._raw_combo.currentData()
        if not raw_name or raw_name not in self._streams:
            self._channels_table.setRowCount(0)
            self._update_summary()
            return
        n_ch = self._streams[raw_name].n_channels
        self._channels_table.setRowCount(n_ch)
        # Default role layout: first 2 channels = nerve, next 3 =
        # stomach, rest = other.
        for i in range(n_ch):
            include_item = QTableWidgetItem("✓")
            include_item.setFlags(
                include_item.flags() | Qt.ItemIsUserCheckable
            )
            include_item.setCheckState(Qt.Checked)
            include_item.setText("")
            self._channels_table.setItem(i, 0, include_item)
            self._channels_table.setItem(
                i, 1, QTableWidgetItem(f"ch{i} (TDT 0-based)"),
            )
            role_combo = QComboBox()
            for r in ROLE_OPTIONS:
                role_combo.addItem(r, r)
            role_combo.setCurrentText(
                ROLE_NERVE if i < 2 else (
                    ROLE_STOMACH if i < 5 else "other"
                )
            )
            self._channels_table.setCellWidget(i, 2, role_combo)
            role_combo.currentIndexChanged.connect(self._update_summary)
            # Default labels: VN1/VN2 for nerve, Ant1/Ant2/Ant3 for stomach
            label_edit = QLineEdit()
            if i < 2:
                label_edit.setText(DEFAULT_NERVE_LABELS[i])
            elif i < 5:
                label_edit.setText(DEFAULT_STOMACH_LABELS[i - 2])
            else:
                label_edit.setText(f"Ch{i + 1}")
            self._channels_table.setCellWidget(i, 3, label_edit)
        self._update_summary()
        # React to checkbox toggle to refresh the summary.
        self._channels_table.itemChanged.connect(self._update_summary)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _selected_channels(self) -> list[dict]:
        """Return the included rows as a list of channel dicts in the
        order the user wants them in the saved signal (top-to-bottom
        in the table)."""
        out: list[dict] = []
        signal_idx = 0
        for i in range(self._channels_table.rowCount()):
            include = self._channels_table.item(i, 0)
            if include is None or include.checkState() != Qt.Checked:
                continue
            role_combo = self._channels_table.cellWidget(i, 2)
            label_edit = self._channels_table.cellWidget(i, 3)
            role = role_combo.currentData() if role_combo else "unknown"
            if role == "exclude":
                continue
            label = label_edit.text() if label_edit else f"Ch{signal_idx + 1}"
            out.append({
                "signal_index": signal_idx,
                "tdt_index": i,         # 0-based; load_stream maps to 1-based
                "role": role,
                "label": label,
            })
            signal_idx += 1
        return out

    def _update_summary(self) -> None:
        selected = self._selected_channels()
        roles: dict[str, int] = {}
        for ch in selected:
            roles[ch["role"]] = roles.get(ch["role"], 0) + 1
        n_nerve = roles.get(ROLE_NERVE, 0)
        n_stomach = roles.get(ROLE_STOMACH, 0)
        # Color-code: green for the canonical 2+3 layout, yellow otherwise.
        if n_nerve == 2 and n_stomach == 3:
            color = "#2ca02c"
            note = "Matches detector's expected 2-nerve + 3-stomach layout."
        else:
            color = "#f0c000"
            note = (
                f"Detector expects exactly 2 nerve + 3 stomach. "
                "You can continue — predictions may not generalize."
            )
        self._summary_label.setText(
            f"<span style='color:{color}'>"
            f"Selected: {len(selected)} channel(s) "
            f"({n_nerve} nerve · {n_stomach} stomach)</span><br>"
            f"<span style='color:#aaa'>{note}</span>"
        )

    # ------------------------------------------------------------------
    # Accept
    # ------------------------------------------------------------------

    def _on_accept(self) -> None:
        selected = self._selected_channels()
        if not selected:
            QMessageBox.warning(
                self, "No channels selected",
                "Pick at least one channel to include.",
            )
            return
        roles = {ch["role"] for ch in selected}
        n_nerve = sum(1 for ch in selected if ch["role"] == ROLE_NERVE)
        n_stomach = sum(1 for ch in selected if ch["role"] == ROLE_STOMACH)
        if n_nerve != 2 or n_stomach != 3:
            resp = QMessageBox.question(
                self, "Non-standard role layout",
                f"You selected {n_nerve} nerve + {n_stomach} stomach. "
                "The detector's model_v0.1.0 was trained on 2 nerve + "
                "3 stomach — predictions on a different layout may not "
                "generalize.\n\n"
                "Continue anyway?",
            )
            if resp != QMessageBox.Yes:
                return
        self.accept()

    def channel_assignment(self) -> dict:
        """Return the channel_assignment dict suitable for
        `Profile.from_review_session`."""
        return {
            "raw_stream": self._raw_combo.currentData(),
            "stim_stream": self._stim_combo.currentData(),
            "vibration_stream": self._vib_combo.currentData(),
            "stim_envelope_stream": self._env_combo.currentData(),
            "channels": self._selected_channels(),
        }
