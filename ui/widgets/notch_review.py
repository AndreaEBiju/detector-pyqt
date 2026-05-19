"""Notch review dialog — per animal, on the first file. Shows the
signal before / after the candidate notch filter, lets the user
edit the harmonics list + Q factor, and re-runs detection on demand.

Layout (top to bottom):
- Toolbar: channel picker, window-start input, span input,
  Re-run-detection button.
- pyqtgraph plot: grey trace = raw, blue trace = notched (live).
- Form: harmonics field (comma-separated), Q factor, detrend toggle.
- Reduction summary: per-harmonic reductions on the current channel.
- Accept / Skip buttons.

For batch flow: this dialog runs once per animal, on the first file
for that animal. The accepted settings flow into all subsequent
files for the same animal via the per-animal Profile.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QRadioButton, QSpinBox, QVBoxLayout, QWidget,
)

_repo_root = Path(__file__).resolve().parent.parent.parent
_detector_core = _repo_root / "detector-core"
if _detector_core.exists() and str(_detector_core) not in sys.path:
    sys.path.insert(0, str(_detector_core))

from detector.preprocessing import notch as notch_mod                # noqa: E402
from detector.preprocessing.profiles import (                          # noqa: E402
    CURRENT_NOTCH_METRIC_VERSION,
)
from detector.preprocessing.tdt_io import load_stream                 # noqa: E402

from ui.data import settings as ui_settings                           # noqa: E402


class NotchReviewDialog(QDialog):
    """Modal notch-review dialog. Returns a `notch` dict suitable for
    `Profile.from_review_session`."""

    def __init__(
        self,
        animal_id: str,
        tdt_folder: Path,
        raw_stream: str,
        channels: list[dict],
        existing_notch: Optional[dict] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Notch review — {animal_id}")
        self.resize(1000, 700)
        self._animal_id = animal_id
        self._tdt_folder = tdt_folder
        self._raw_stream = raw_stream
        self._channels = channels        # list of {signal_index, tdt_index, role, label}
        # Loaded data for the current channel — cached so repeated
        # re-runs don't re-pay the load.
        self._loaded_data: Optional[np.ndarray] = None
        self._loaded_fs: Optional[float] = None
        self._loaded_channel_signal_idx: Optional[int] = None
        self._reductions: dict[float, float] = {}

        outer = QVBoxLayout(self)

        # Top toolbar: channel + window
        top = QHBoxLayout()
        top.addWidget(QLabel("Channel:"))
        self._channel_combo = QComboBox()
        for ch in channels:
            self._channel_combo.addItem(
                f"{ch['label']} ({ch['role']})", ch,
            )
        self._channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        top.addWidget(self._channel_combo)
        top.addSpacing(20)
        top.addWidget(QLabel("Window start (s):"))
        self._window_start_spin = QDoubleSpinBox()
        self._window_start_spin.setRange(0.0, 7200.0)
        self._window_start_spin.setValue(0.0)
        self._window_start_spin.setDecimals(2)
        top.addWidget(self._window_start_spin)
        top.addWidget(QLabel("Span (s):"))
        self._span_spin = QDoubleSpinBox()
        self._span_spin.setRange(1.0, 60.0)
        self._span_spin.setValue(10.0)
        self._span_spin.setDecimals(1)
        top.addWidget(self._span_spin)
        self._refresh_plot_btn = QPushButton("↻ Refresh plot")
        self._refresh_plot_btn.clicked.connect(self._refresh_plot)
        top.addWidget(self._refresh_plot_btn)
        top.addStretch(1)
        outer.addLayout(top)

        # Plot
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("#0e1117")
        self._plot_widget.setLabel("bottom", "Time (s)")
        self._plot_widget.setLabel("left", "Amplitude")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self._raw_curve = self._plot_widget.plot(
            pen=pg.mkPen(color="#999", width=0.7), name="raw"
        )
        self._filtered_curve = self._plot_widget.plot(
            pen=pg.mkPen(color="#4ea3ff", width=1.0), name="notched"
        )
        outer.addWidget(self._plot_widget, stretch=1)

        # Notch params form. Defaults come from the Training window's
        # Preprocessing tab (ui_settings) unless an existing per-animal
        # profile passes its own values via `existing_notch`.
        settings = ui_settings.load_settings()
        default_q = float(settings.get("preprocessing_q_factor", 30.0))
        default_detrend = bool(settings.get("preprocessing_detrend", True))
        default_harmonics = settings.get(
            "preprocessing_candidate_harmonics", [60.0, 120.0]
        )

        form_box = QGroupBox("Notch parameters")
        form = QFormLayout(form_box)
        self._harmonics_edit = QLineEdit(
            ", ".join(f"{float(h):g}" for h in default_harmonics[:2])
            if default_harmonics else "60.0, 120.0"
        )
        self._harmonics_edit.editingFinished.connect(self._refresh_plot)
        form.addRow("Harmonics (Hz, comma-separated)", self._harmonics_edit)
        self._q_spin = QDoubleSpinBox()
        self._q_spin.setRange(1.0, 200.0)
        self._q_spin.setValue(default_q)
        self._q_spin.setDecimals(1)
        self._q_spin.valueChanged.connect(self._refresh_plot)
        form.addRow("Q factor", self._q_spin)
        self._detrend_check = QCheckBox("Detrend (subtract per-channel mean)")
        self._detrend_check.setChecked(default_detrend)
        self._detrend_check.toggled.connect(self._refresh_plot)
        form.addRow(self._detrend_check)
        self._redetect_btn = QPushButton(
            "Auto-detect harmonics (run on current channel)"
        )
        self._redetect_btn.clicked.connect(self._run_auto_detect)
        form.addRow(self._redetect_btn)

        # Apply-to scope (plan P10.7): the filter always applies to
        # every channel of the saved output; this radio just controls
        # whether the plot's right-side preview filters ALL selected
        # channels (so the user can scroll through the channel combo
        # and spot-check each) or only the currently-shown channel
        # (cheaper to recompute on big windows). Stored on the dialog
        # so `notch_settings()` reports the choice for the profile.
        apply_box = QGroupBox("Apply notch filter to")
        apply_v = QVBoxLayout(apply_box)
        self._apply_all_radio = QRadioButton(
            "All channels in this animal's recordings (recommended)"
        )
        self._apply_all_radio.setChecked(True)
        self._apply_selected_radio = QRadioButton(
            "Selected channel only (preview / spot-check before committing)"
        )
        apply_v.addWidget(self._apply_all_radio)
        apply_v.addWidget(self._apply_selected_radio)
        form.addRow(apply_box)
        outer.addWidget(form_box)

        # Reduction summary
        self._summary_label = QLabel("(detect harmonics to see reductions)")
        self._summary_label.setStyleSheet(
            "font-family: monospace; font-size: 11px; padding: 4px;"
        )
        outer.addWidget(self._summary_label)

        # Accept / Skip
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self._buttons.button(QDialogButtonBox.Ok).setText("Accept and continue")
        self._buttons.button(QDialogButtonBox.Cancel).setText("Skip notch (no filter)")
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self._on_skip)
        outer.addWidget(self._buttons)

        # Apply existing-profile notch params if given
        if existing_notch:
            freqs = existing_notch.get("frequencies_filtered", [60.0, 120.0])
            self._harmonics_edit.setText(
                ", ".join(f"{float(h):.1f}" for h in freqs)
            )
            self._q_spin.setValue(float(existing_notch.get("q_factor", 30.0)))
            self._detrend_check.setChecked(
                bool(existing_notch.get("detrend", True))
            )
            self._reductions = {
                float(k): float(v)
                for k, v in (existing_notch.get("reductions_per_harmonic") or {}).items()
            }

        # Initial render — load data + paint
        self._on_channel_changed()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _ensure_data_loaded(self) -> bool:
        """Lazy-load the current channel's signal. Returns True on
        success, False otherwise (in which case the plot is cleared
        and a warning shown).

        The full-channel load can be slow (~tens of seconds on a
        20-min recording). We cache by signal_index so panning the
        time window doesn't re-pay it.
        """
        ch = self._channel_combo.currentData()
        if ch is None:
            return False
        # If the folder doesn't actually exist (e.g. dialog
        # constructed for a smoke test, or the path moved between
        # runs), bail silently so we don't pop a blocking modal
        # during dialog construction.
        if not Path(self._tdt_folder).exists():
            return False
        signal_idx = ch["signal_index"]
        if (
            self._loaded_data is not None
            and self._loaded_channel_signal_idx == signal_idx
        ):
            return True
        try:
            data, fs = load_stream(
                self._tdt_folder, self._raw_stream,
                channel_indices=[ch["tdt_index"]],
            )
            self._loaded_data = data
            self._loaded_fs = fs
            self._loaded_channel_signal_idx = signal_idx
            # Clamp the window-start spinbox to the recording length.
            duration_s = data.shape[0] / fs
            self._window_start_spin.setMaximum(max(0.0, duration_s - 0.1))
            return True
        except Exception as exc:
            QMessageBox.warning(
                self, "Failed to load channel",
                f"Could not load channel {signal_idx} from "
                f"{self._raw_stream!r}:\n{exc}",
            )
            self._loaded_data = None
            self._loaded_fs = None
            self._loaded_channel_signal_idx = None
            return False

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def _refresh_plot(self) -> None:
        if not self._ensure_data_loaded():
            return
        fs = float(self._loaded_fs)
        data = self._loaded_data
        n = data.shape[0]
        # Extract the current window.
        t_start = float(self._window_start_spin.value())
        span = float(self._span_spin.value())
        i0 = max(0, int(t_start * fs))
        i1 = min(n, i0 + int(span * fs))
        if i1 <= i0:
            return
        window = data[i0:i1, :]
        t = np.arange(i0, i1) / fs
        # Raw trace
        self._raw_curve.setData(t, window[:, 0])
        # Filtered trace
        harmonics = self._parse_harmonics()
        if harmonics:
            try:
                filtered = notch_mod.apply_notch_filter(
                    window, fs, harmonics,
                    q_factor=float(self._q_spin.value()),
                    detrend=self._detrend_check.isChecked(),
                )
                self._filtered_curve.setData(t, filtered[:, 0])
            except Exception as exc:
                # Bad parameters (e.g. > Nyquist) — clear the filtered
                # trace and let the user fix the input.
                self._filtered_curve.clear()
                self._summary_label.setText(
                    f"<span style='color:#ff6b6b'>Filter error: {exc}</span>"
                )
                return
        else:
            # No harmonics → just clear the filtered overlay.
            self._filtered_curve.clear()
        # Update reduction summary if we have detection results.
        self._update_reduction_summary()

    def _parse_harmonics(self) -> list[float]:
        text = self._harmonics_edit.text().strip()
        if not text:
            return []
        try:
            return [float(t.strip()) for t in text.split(",") if t.strip()]
        except ValueError:
            return []

    # ------------------------------------------------------------------
    # Auto-detection
    # ------------------------------------------------------------------

    def _run_auto_detect(self) -> None:
        if not self._ensure_data_loaded():
            return
        # Pick candidate harmonics + threshold + cap from settings so
        # the Training window's Preprocessing tab is the single source
        # of truth for defaults.
        settings = ui_settings.load_settings()
        params = notch_mod.NotchParams(
            q_factor=float(self._q_spin.value()),
            detrend=self._detrend_check.isChecked(),
            candidate_harmonics=list(
                settings.get(
                    "preprocessing_candidate_harmonics",
                    [60.0, 120.0, 180.0, 240.0, 300.0],
                )
            ),
            reduction_threshold=float(
                settings.get("preprocessing_reduction_threshold", 0.5)
            ),
            max_harmonics_filtered=int(
                settings.get("preprocessing_max_harmonics", 4)
            ),
        )
        # Use a longer chunk for detection (60s) for better PSD
        # resolution. Take from the middle of the recording to avoid
        # any edge transients.
        fs = float(self._loaded_fs)
        data = self._loaded_data
        n = data.shape[0]
        chunk_n = min(n, int(60.0 * fs))
        start = max(0, (n - chunk_n) // 2)
        chunk = data[start:start + chunk_n, :]
        reds = notch_mod.detect_significant_harmonics(chunk, fs, params)
        self._reductions = reds
        selected = notch_mod.select_harmonics_to_filter(reds, params)
        self._harmonics_edit.setText(
            ", ".join(f"{float(h):.1f}" for h in selected)
        )
        self._refresh_plot()

    def _update_reduction_summary(self) -> None:
        if not self._reductions:
            self._summary_label.setText(
                "(click Auto-detect to populate per-harmonic reductions)"
            )
            return
        active = set(self._parse_harmonics())
        parts = []
        for h in sorted(self._reductions):
            r = self._reductions[h]
            tag = "*" if h in active else " "
            # 0.0% means "no real peak above the local PSD baseline";
            # 99.x% means "peak collapsed by ~all of its prominence".
            parts.append(f"{tag} {h:>6.1f} Hz: {r * 100:>5.1f}%")
        ch_label = (
            self._channel_combo.currentData()["label"]
            if self._channel_combo.currentData() else "?"
        )
        legend = (
            f"<span style='color:#aaa'>(* = currently in filter chain · "
            f"channel: {ch_label}) · 0% means no peak above "
            "background — no real hum to remove</span>"
        )
        self._summary_label.setText(
            "Peak-prominence reductions on detection chunk:<br>"
            f"<pre style='margin: 4px;'>" + "<br>".join(parts) + "</pre>"
            + legend
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_channel_changed(self) -> None:
        # Drop the cached load so the new channel gets pulled lazily.
        self._loaded_data = None
        self._loaded_fs = None
        self._loaded_channel_signal_idx = None
        self._refresh_plot()

    def _on_skip(self) -> None:
        # Skip = accept with empty notch chain.
        self._harmonics_edit.setText("")
        self.accept()

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def notch_settings(self) -> dict:
        """Return the dict to feed `Profile.from_review_session`.

        Reduction threshold + max harmonics use the user's
        Training-window defaults at the time of the review. The
        `apply_scope` field records whether the review previewed
        on all channels or just the selected one — informational;
        the actual batch save always applies the filter chain to
        every saved channel.
        """
        settings = ui_settings.load_settings()
        harmonics = self._parse_harmonics()
        return {
            "q_factor": float(self._q_spin.value()),
            "detrend": self._detrend_check.isChecked(),
            "frequencies_filtered": harmonics,
            "reductions_per_harmonic": {
                str(k): float(v) for k, v in self._reductions.items()
            },
            # Stamp the metric version so a future re-load can tell
            # this dict was produced under the peak-prominence metric
            # (semver "2.0"). The Profile.load migration drops stale
            # reductions from pre-2.0 saves.
            "notch_metric_version": CURRENT_NOTCH_METRIC_VERSION,
            "reduction_threshold": float(settings.get(
                "preprocessing_reduction_threshold", 0.5,
            )),
            "max_harmonics_filtered": int(settings.get(
                "preprocessing_max_harmonics", 4,
            )),
            "apply_scope": (
                "all_channels" if self._apply_all_radio.isChecked()
                else "selected_channel_preview"
            ),
        }
