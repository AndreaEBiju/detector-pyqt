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


# Engineering prefixes used when auto-scaling small signal amplitudes
# (TDT raw recordings are typically in the µV–mV range; raw σ values
# come out as 1e-5 to 1e-3, which read as "0.000" if formatted with
# fixed precision). Each tuple is (multiplier, suffix). Picked by
# bisection on the value's magnitude.
_ENG_PREFIXES: tuple[tuple[float, str], ...] = (
    (1e-12, "pV"),
    (1e-9,  "nV"),
    (1e-6,  "µV"),
    (1e-3,  "mV"),
    (1.0,   "V"),
)


def _fmt_amplitude(value: float) -> str:
    """Format a Volts-scale amplitude with the right engineering
    prefix (pV / nV / µV / mV / V). Picks the prefix so the displayed
    number lands in [1, 1000)."""
    av = abs(value)
    if av == 0:
        return "0.000 V"
    # Walk from largest prefix down to find the one where the scaled
    # value lands in [1, 1000).
    for mult, suffix in reversed(_ENG_PREFIXES):
        scaled = value / mult
        if abs(scaled) >= 1.0:
            return f"{scaled:.3f} {suffix}"
    # Smaller than 1 pV — show in pV anyway with extra precision.
    return f"{value / 1e-12:.4f} pV"


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

        # Plot — single overlaid panel with both traces on the same
        # Y-axis so you can see exactly where they diverge. Raw
        # underneath (light grey), filtered on top (blue). The legend
        # in the top-right makes the channel role explicit.
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("#0e1117")
        self._plot_widget.setLabel("bottom", "Time (s)")
        self._plot_widget.setLabel("left", "Amplitude")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self._plot_widget.addLegend(offset=(-10, 10))
        self._raw_curve = self._plot_widget.plot(
            pen=pg.mkPen(color="#cccccc", width=0.8), name="Raw",
        )
        self._filtered_curve = self._plot_widget.plot(
            pen=pg.mkPen(color="#4ea3ff", width=1.0), name="After notch",
        )
        # "Removed" trace = raw − filtered. The energy the notch
        # chain took out. A flat line at 0 means the filter did
        # nothing (no hum at the selected frequencies); a visible
        # 60 Hz oscillation means real mains pickup was removed.
        # Lighter / dashed pen so it doesn't crowd the main traces.
        self._diff_curve = self._plot_widget.plot(
            pen=pg.mkPen(color="#ff6b6b", width=0.7,
                          style=Qt.DashLine),
            name="Removed (raw − notch)",
        )
        outer.addWidget(self._plot_widget, stretch=1)

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
        # Detrend toggle. When on, the per-channel mean is subtracted
        # from BOTH the raw and notched display traces (and the
        # batch save's filter input), so both sit at the same
        # baseline and the user can see the actual filter effect
        # rather than the DC shift. Default ON because most lab
        # recordings have meaningful DC offset.
        self._detrend_check = QCheckBox(
            "Detrend (subtract per-channel mean before display + filter)"
        )
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
        # Detrend handling. When the checkbox is on, subtract the
        # mean from the SAME source (the visible window) for both
        # raw display and the filter input. This ensures the grey
        # raw and blue notched traces sit at the same baseline —
        # the previous "they look shifted" bug came from detrending
        # only the filter input.
        do_detrend = self._detrend_check.isChecked()
        window_for_display = window.copy()
        if do_detrend:
            window_for_display -= window_for_display.mean(axis=0, keepdims=True)
        # Raw trace — always visible underneath in light grey.
        self._raw_curve.setData(t, window_for_display[:, 0])
        # Filtered trace overlaid on top in blue. When no harmonics,
        # the curve clears so the legend stays accurate.
        harmonics = self._parse_harmonics()
        if harmonics:
            try:
                # Filter the detrended (or raw, per checkbox) window
                # so the filtered trace has the same baseline as the
                # raw trace shown above.
                filtered = notch_mod.apply_notch_filter(
                    window_for_display, fs, harmonics,
                    q_factor=float(self._q_spin.value()),
                )
                self._filtered_curve.setData(t, filtered[:, 0])
                # Difference trace: what the notch chain REMOVED.
                # A flat red line at y=0 → filter did nothing (no
                # hum at the selected frequencies). A sinusoidal
                # oscillation → mains hum was present. This is the
                # unambiguous visual confirmation of what the
                # filter is doing.
                diff = window_for_display[:, 0] - filtered[:, 0]
                self._diff_curve.setData(t, diff)
            except Exception as exc:
                self._filtered_curve.setData([], [])
                self._diff_curve.setData([], [])
                self._summary_label.setText(
                    f"<span style='color:#ff6b6b'>Filter error: {exc}</span>"
                )
                return
        else:
            # No harmonics → hide the filtered + diff overlays;
            # legend stays so the user knows what each color means.
            self._filtered_curve.setData([], [])
            self._diff_curve.setData([], [])
        # Update the σ before/after summary.
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
        # Detrend ONLY for the σ measurement (estimate_noise_floor
        # assumes ~zero-mean input). Doesn't affect the filter or
        # the plot — gi-vagus-viewer applies the same detrend
        # internally to its noise estimator.
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
                )
                fcol = filtered[:, 0].astype(np.float64, copy=False)
                fcol = fcol - fcol.mean()  # detrend for σ measurement
                sigma_after = float(notch_mod.estimate_noise_floor(fcol))
                drop_pct = (sigma_before - sigma_after) / sigma_before * 100
                headline = (
                    f"<b>Noise floor σ (Quiroga MAD on 60-s chunk · "
                    f"channel: {ch_label}):</b>"
                    f"<pre style='margin: 4px;'>"
                    f"  before:  {_fmt_amplitude(sigma_before):>12}<br>"
                    f"  after:   {_fmt_amplitude(sigma_after):>12}"
                    f"   ({drop_pct:+.2f}%)</pre>"
                )
            except Exception as exc:
                headline = (
                    f"<span style='color:#ff6b6b'>"
                    f"Couldn't apply chain: {exc}</span>"
                )
        else:
            headline = (
                f"<b>Noise floor σ (Quiroga MAD · channel: {ch_label}):</b> "
                f"{_fmt_amplitude(sigma_before)}  "
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
            "detrend": bool(self._detrend_check.isChecked()),
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
