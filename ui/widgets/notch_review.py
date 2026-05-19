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
        # Per-harmonic fractional reduction of the recording's
        # Quiroga MAD-based noise floor (σ) when an iirnotch is
        # applied at that frequency in isolation. Populated by
        # `_run_auto_detect`. Empty dict means the user hasn't run
        # Auto-detect yet.
        self._noise_reductions: dict[float, float] = {}

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

        form_box = QGroupBox("Notch parameters")
        form = QFormLayout(form_box)
        # Default to filtering 60 / 120 / 180 Hz — the standard mains
        # harmonics. Matches the gi-vagus-viewer pipeline's default
        # `freqs_hz=(60.0, 120.0, 180.0)`. Mains hum is ubiquitous in
        # lab recordings, so we filter prophylactically rather than
        # waiting for a detection metric to confirm.
        self._harmonics_edit = QLineEdit("60.0, 120.0, 180.0")
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
        # The button MEASURES per-candidate σ reductions and displays
        # them — it does NOT overwrite the harmonics field. Mains
        # hum is filtered prophylactically (defaults are pre-filled);
        # this button is a sanity check, not a replacement for the
        # user's manual selection.
        self._redetect_btn = QPushButton(
            "↻ Measure noise-floor reductions (current channel)"
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
            # Noise-floor reductions (fraction) from a profile saved
            # under the current metric. Profile.load on older profiles
            # strips this field so the dict stays empty until the user
            # runs Auto-detect with the new metric.
            self._noise_reductions = {
                float(k): float(v)
                for k, v in (existing_notch.get(
                    "noise_reductions_per_harmonic"
                ) or {}).items()
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
                settings.get("preprocessing_reduction_threshold", 0.05)
            ),
            max_harmonics_filtered=int(
                settings.get("preprocessing_max_harmonics", 4)
            ),
        )
        # Use a longer chunk for detection (60s) so the MAD estimator
        # has enough samples to be stable. Take from the middle of
        # the recording to avoid edge transients.
        fs = float(self._loaded_fs)
        data = self._loaded_data
        n = data.shape[0]
        chunk_n = min(n, int(60.0 * fs))
        start = max(0, (n - chunk_n) // 2)
        chunk = data[start:start + chunk_n, :]
        reds = notch_mod.detect_significant_harmonics(chunk, fs, params)
        self._noise_reductions = {float(k): float(v) for k, v in reds.items()}
        # Intentionally DO NOT modify the harmonics field. The
        # measurement is informational — the user's selection stays
        # whatever they put in (defaulting to 60/120/180). The display
        # in `_update_reduction_summary` shows the per-candidate
        # reductions so the user can see which would contribute most.
        self._refresh_plot()

    def _update_reduction_summary(self) -> None:
        """Build the summary panel with two pieces:

        1. **Headline**: overall σ before vs after applying the
           current notch chain (from the field) on the 60-s detection
           chunk. This is the most useful single number — "did your
           chosen notch chain reduce my noise floor?"
        2. **Per-candidate breakdown**: σ reduction if each candidate
           harmonic were applied IN ISOLATION. Populated by the
           "Measure" button. Informational — helps the user decide
           whether to add/remove specific frequencies.
        """
        # Need the loaded channel data to compute the overall metric.
        if (
            self._loaded_data is None or
            self._loaded_fs is None
        ):
            if self._noise_reductions:
                lines = ["(load a channel to see the overall σ change)"]
            else:
                lines = [
                    "(load a channel and click Measure to see "
                    "noise-floor reductions)"
                ]
            self._summary_label.setText("<br>".join(lines))
            return

        fs = float(self._loaded_fs)
        data = self._loaded_data
        n = data.shape[0]
        chunk_n = min(n, int(60.0 * fs))
        start = max(0, (n - chunk_n) // 2)
        chunk = data[start:start + chunk_n, :]

        # Overall σ before vs after applying the CURRENT chain.
        harmonics = self._parse_harmonics()
        col = chunk[:, 0].astype(np.float64, copy=False)
        if self._detrend_check.isChecked():
            col = col - col.mean()
        sigma_before = float(np.median(np.abs(col)) / 0.6745)
        if harmonics and sigma_before > 0:
            try:
                filtered = notch_mod.apply_notch_filter(
                    chunk, fs, harmonics,
                    q_factor=float(self._q_spin.value()),
                    detrend=self._detrend_check.isChecked(),
                )
                fcol = filtered[:, 0].astype(np.float64, copy=False)
                sigma_after = float(np.median(np.abs(fcol)) / 0.6745)
                drop_pct = (sigma_before - sigma_after) / sigma_before * 100
                overall = (
                    f"<b>Noise floor σ (Quiroga MAD on 60-s chunk):</b><br>"
                    f"<pre style='margin: 4px;'>"
                    f"  before: {sigma_before:>8.2f}<br>"
                    f"  after:  {sigma_after:>8.2f}"
                    f"   ({drop_pct:+.1f}%)</pre>"
                )
            except Exception as exc:
                overall = (
                    f"<span style='color:#ff6b6b'>"
                    f"Couldn't apply chain: {exc}</span>"
                )
        else:
            overall = (
                f"<b>Noise floor σ (Quiroga MAD):</b> {sigma_before:.2f}"
                "  (no harmonics in chain — set some to see reduction)"
            )

        # Per-candidate measurements, if the user clicked "Measure".
        per_cand = ""
        if self._noise_reductions:
            active = set(harmonics)
            parts = []
            for h in sorted(self._noise_reductions):
                r = self._noise_reductions[h]
                tag = "*" if h in active else " "
                parts.append(f"{tag} {h:>6.1f} Hz: {r * 100:>5.1f}%")
            per_cand = (
                "<br><b>Per-candidate σ reduction (if applied alone):</b>"
                f"<pre style='margin: 4px;'>" + "<br>".join(parts) + "</pre>"
            )

        ch_label = (
            self._channel_combo.currentData()["label"]
            if self._channel_combo.currentData() else "?"
        )
        legend = (
            f"<span style='color:#aaa'>(channel: {ch_label}) · σ uses "
            "the robust Quiroga MAD estimator (median(|x|)/0.6745), "
            "matching the gi-vagus-viewer downstream pipeline. * = "
            "currently in the user's notch chain.</span>"
        )
        self._summary_label.setText(overall + per_cand + "<br>" + legend)

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
            # Per-harmonic fractional drop in the Quiroga MAD-based
            # noise floor when an iirnotch is applied at that
            # frequency in isolation. Same noise estimator the
            # gi-vagus-viewer downstream pipeline uses.
            "noise_reductions_per_harmonic": {
                str(k): float(v)
                for k, v in self._noise_reductions.items()
            },
            # Stamp the metric version so a future re-load can tell
            # this dict was produced under the σ-reduction metric
            # (semver "4.0"). The Profile.load migration drops stale
            # detection numbers from older saves.
            "notch_metric_version": CURRENT_NOTCH_METRIC_VERSION,
            "reduction_threshold": float(settings.get(
                "preprocessing_reduction_threshold", 0.05,
            )),
            "max_harmonics_filtered": int(settings.get(
                "preprocessing_max_harmonics", 4,
            )),
            "apply_scope": (
                "all_channels" if self._apply_all_radio.isChecked()
                else "selected_channel_preview"
            ),
        }
