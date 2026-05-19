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

        # Plot — two stacked subplots with linked X-axes so the user
        # can compare RAW vs NOTCHED at the same scale. A single
        # overlaid plot was confusing: when the filter knocks out the
        # mains line, the blue trace hides under the grey one in the
        # parts where the signals match (most of the recording), and
        # the user can't tell where the difference is. Stacking makes
        # the before/after explicit.
        plots_container = pg.GraphicsLayoutWidget()
        plots_container.setBackground("#0e1117")
        self._plot_raw = plots_container.addPlot(row=0, col=0)
        self._plot_raw.setTitle("Raw", color="#bbb", size="10pt")
        self._plot_raw.setLabel("left", "Amplitude")
        self._plot_raw.showGrid(x=True, y=True, alpha=0.15)
        self._raw_curve = self._plot_raw.plot(
            pen=pg.mkPen(color="#bbb", width=0.7), name="raw"
        )

        self._plot_filtered = plots_container.addPlot(row=1, col=0)
        self._plot_filtered.setTitle(
            "After notch", color="#4ea3ff", size="10pt",
        )
        self._plot_filtered.setLabel("bottom", "Time (s)")
        self._plot_filtered.setLabel("left", "Amplitude")
        self._plot_filtered.showGrid(x=True, y=True, alpha=0.15)
        self._filtered_curve = self._plot_filtered.plot(
            pen=pg.mkPen(color="#4ea3ff", width=0.7), name="notched"
        )
        # Link X-axes so panning/zooming syncs across both subplots.
        # Y-axes stay independent — the user can occasionally see the
        # filtered trace at a different Y scale if the raw has DC
        # offset, but for detrended signals they match.
        self._plot_filtered.setXLink(self._plot_raw)
        outer.addWidget(plots_container, stretch=1)

        # Notch params form. Defaults come from the Training window's
        # Preprocessing tab (ui_settings) unless an existing per-animal
        # profile passes its own values via `existing_notch`.
        settings = ui_settings.load_settings()
        default_q = float(settings.get("preprocessing_q_factor", 30.0))
        default_detrend = bool(settings.get("preprocessing_detrend", True))

        form_box = QGroupBox("Notch parameters")
        form = QFormLayout(form_box)
        # Default to filtering the user's configured mains harmonics
        # (Training window's Preprocessing tab → "Default mains
        # harmonics"). Out-of-the-box that's 60/120/180 — matching
        # gi-vagus-viewer's `freqs_hz=(60.0, 120.0, 180.0)`. Mains
        # hum is ubiquitous in lab recordings, so we filter
        # prophylactically rather than auto-detecting.
        default_freqs = settings.get(
            "preprocessing_default_freqs_hz", [60.0, 120.0, 180.0]
        )
        self._harmonics_edit = QLineEdit(
            ", ".join(f"{float(h):g}" for h in default_freqs)
            if default_freqs else "60.0, 120.0, 180.0"
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

        # Apply existing-profile notch params if given. Important:
        # if the saved `frequencies_filtered` is EMPTY (e.g. a stale
        # profile saved under the old Auto-detect-with-threshold
        # behavior that rejected every candidate), fall back to the
        # 60/120/180 default rather than leaving the field empty —
        # an empty field disables the filter chain entirely, which
        # is never what the user wants on dialog open.
        if existing_notch:
            freqs = existing_notch.get("frequencies_filtered") or []
            if not freqs:
                # Stale empty profile — fall back to user-configured
                # defaults (or hard-coded 60/120/180) so the field is
                # never blank on dialog open.
                freqs = list(default_freqs) or [60.0, 120.0, 180.0]
            self._harmonics_edit.setText(
                ", ".join(f"{float(h):.1f}" for h in freqs)
            )
            self._q_spin.setValue(float(existing_notch.get("q_factor", 30.0)))
            self._detrend_check.setChecked(
                bool(existing_notch.get("detrend", True))
            )
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
        # Raw trace (top subplot, always visible)
        self._raw_curve.setData(t, window[:, 0])
        # Filtered trace (bottom subplot). When no harmonics are set,
        # show the raw signal in the bottom plot too so the user sees
        # both panels populated (and notices: "they match, no filter
        # active") rather than a blank panel.
        harmonics = self._parse_harmonics()
        if harmonics:
            try:
                filtered = notch_mod.apply_notch_filter(
                    window, fs, harmonics,
                    q_factor=float(self._q_spin.value()),
                    detrend=self._detrend_check.isChecked(),
                )
                self._filtered_curve.setData(t, filtered[:, 0])
                self._plot_filtered.setTitle(
                    f"After notch  ·  {', '.join(f'{h:g}' for h in harmonics)} Hz",
                    color="#4ea3ff", size="10pt",
                )
            except Exception as exc:
                # Bad parameters (e.g. > Nyquist) — show the raw in
                # the bottom panel and surface the error in the
                # summary so the user sees what went wrong.
                self._filtered_curve.setData(t, window[:, 0])
                self._plot_filtered.setTitle(
                    "After notch  ·  (filter error)",
                    color="#ff6b6b", size="10pt",
                )
                self._summary_label.setText(
                    f"<span style='color:#ff6b6b'>Filter error: {exc}</span>"
                )
                return
        else:
            # No harmonics → show the raw trace in the bottom panel
            # too, with a title indicating no filter is applied. This
            # is more discoverable than a blank panel.
            self._filtered_curve.setData(t, window[:, 0])
            self._plot_filtered.setTitle(
                "After notch  ·  (no harmonics set — same as raw)",
                color="#888", size="10pt",
            )
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
    # Summary panel
    # ------------------------------------------------------------------

    def _update_reduction_summary(self) -> None:
        """Show the single headline metric: noise floor σ before vs
        after applying the current notch chain on a 60-s reference
        chunk. Matches the gi-vagus-viewer approach — no per-harmonic
        breakdown (every prior version's per-candidate metric had a
        failure mode and confused more than it informed).
        """
        if (
            self._loaded_data is None or
            self._loaded_fs is None
        ):
            self._summary_label.setText(
                "(load a channel to see the noise-floor σ change)"
            )
            return

        fs = float(self._loaded_fs)
        data = self._loaded_data
        n = data.shape[0]
        chunk_n = min(n, int(60.0 * fs))
        start = max(0, (n - chunk_n) // 2)
        chunk = data[start:start + chunk_n, :]

        harmonics = self._parse_harmonics()
        col = chunk[:, 0].astype(np.float64, copy=False)
        if self._detrend_check.isChecked():
            col = col - col.mean()
        try:
            sigma_before = float(
                notch_mod.estimate_noise_floor(col)
            )
        except Exception as exc:
            self._summary_label.setText(
                f"<span style='color:#ff6b6b'>σ estimate failed: {exc}</span>"
            )
            return

        ch_label = (
            self._channel_combo.currentData()["label"]
            if self._channel_combo.currentData() else "?"
        )
        if harmonics and sigma_before > 0:
            try:
                filtered = notch_mod.apply_notch_filter(
                    chunk, fs, harmonics,
                    q_factor=float(self._q_spin.value()),
                    detrend=self._detrend_check.isChecked(),
                )
                fcol = filtered[:, 0].astype(np.float64, copy=False)
                sigma_after = float(notch_mod.estimate_noise_floor(fcol))
                drop_pct = (sigma_before - sigma_after) / sigma_before * 100
                headline = (
                    f"<b>Noise floor σ (Quiroga MAD on 60-s chunk · "
                    f"channel: {ch_label}):</b>"
                    f"<pre style='margin: 4px;'>"
                    f"  before: {sigma_before:>9.3f}<br>"
                    f"  after:  {sigma_after:>9.3f}   "
                    f"({drop_pct:+.1f}%)</pre>"
                )
            except Exception as exc:
                headline = (
                    f"<span style='color:#ff6b6b'>"
                    f"Couldn't apply chain: {exc}</span>"
                )
        else:
            headline = (
                f"<b>Noise floor σ (Quiroga MAD · channel: {ch_label}):</b> "
                f"{sigma_before:.3f}  "
                "(no harmonics in chain — set some to see the reduction)"
            )

        legend = (
            "<span style='color:#aaa'>σ uses the robust Quiroga MAD "
            "estimator (median(|x|)/0.6745) — the same noise floor "
            "the gi-vagus-viewer downstream pipeline uses for "
            "spike-detection thresholds.</span>"
        )
        self._summary_label.setText(headline + "<br>" + legend)

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
        harmonics = self._parse_harmonics()
        return {
            "q_factor": float(self._q_spin.value()),
            "detrend": self._detrend_check.isChecked(),
            "frequencies_filtered": harmonics,
            # Stamp the metric version so a future re-load can tell
            # this dict was produced under the fixed-harmonics
            # gi-vagus-viewer-style algorithm (semver "5.0"). The
            # Profile.load migration drops stale detection numbers
            # from older saves.
            "notch_metric_version": CURRENT_NOTCH_METRIC_VERSION,
            "apply_scope": (
                "all_channels" if self._apply_all_radio.isChecked()
                else "selected_channel_preview"
            ),
        }
