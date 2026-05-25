"""Held-out evaluation results dialog.

Shows the report from `detector.heldout_eval.evaluate_heldout`:
  - Header with model version + recording count.
  - Per-recording table (recording_id, agreement, precision, recall,
    F1, FP%, FN%, human bad-fraction, model bad-fraction, elapsed,
    and a "View overlay" button that opens a per-recording
    HeldoutOverlayWindow).
  - Aggregate block with both micro (sample-weighted) and macro
    (recording-weighted) averages.
  - "Save report JSON…" button.

The per-row "View" button opens `HeldoutOverlayWindow` (Phase 3),
which shows the recording's signal with human ground-truth + model
prediction overlays. Intervals come from the cache the worker
captured during the eval run, so opening an overlay does NOT
re-run detect_bad.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)


def _fmt_pct(v) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.2f}%"


def _fmt_ratio(v) -> str:
    if v is None:
        return "—"
    return f"{v:.4f}"


class HeldoutEvalDialog(QDialog):
    def __init__(
        self,
        report: dict,
        *,
        interval_cache: Optional[dict] = None,
        manifest_path: Optional[Path] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        version = report.get("model_version", "unknown")
        n_eval = report.get("n_recordings_evaluated", 0)
        n_skip = report.get("n_recordings_skipped", 0)
        self.setWindowTitle(f"Held-out evaluation — {version}")
        self.resize(1200, 660)
        self._report = report
        self._interval_cache = interval_cache or {}
        self._manifest_path = (
            Path(manifest_path) if manifest_path is not None else None
        )
        # Resolve manifest -> recording-id lookup once so the View
        # button click is a constant-time dict access. Done lazily to
        # tolerate a missing manifest path (e.g. tests that exercise
        # the dialog with a synthetic report).
        self._rec_by_id: dict[str, dict] = self._load_recordings_by_id()
        # Hold references to overlay windows so they aren't garbage-
        # collected the moment the click handler returns.
        self._overlay_windows: list[QWidget] = []

        root = QVBoxLayout(self)

        # ---- Header ------------------------------------------------
        header = QLabel(
            f"<b>Model:</b> {version}     "
            f"<b>Evaluated:</b> {n_eval} recording(s)     "
            f"<b>Skipped:</b> {n_skip}"
        )
        header.setTextFormat(Qt.RichText)
        root.addWidget(header)

        if n_eval == 0:
            empty = QLabel(
                "<i>No held-out recordings were evaluated. Mark a "
                "recording's <b>held_out</b> field True in Training "
                "Management → Add recording to populate this report.</i>"
            )
            empty.setTextFormat(Qt.RichText)
            empty.setWordWrap(True)
            root.addWidget(empty)
        else:
            root.addWidget(self._build_per_recording_table(report))
            root.addWidget(self._build_aggregate_box(report))

        if n_skip > 0:
            skipped_box = self._build_skipped_box(report)
            root.addWidget(skipped_box)

        # ---- Buttons ------------------------------------------------
        btn_row = QHBoxLayout()
        btn_save = QPushButton("Save report JSON…")
        btn_save.clicked.connect(self._save_json)
        btn_row.addWidget(btn_save)
        btn_row.addStretch(1)
        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.rejected.connect(self.reject)
        btn_box.accepted.connect(self.accept)
        btn_row.addWidget(btn_box)
        root.addLayout(btn_row)

    # ----------------------------------------------------------------
    # Per-recording table
    # ----------------------------------------------------------------

    def _build_per_recording_table(self, report: dict) -> QWidget:
        per_rec = report.get("per_recording", []) or []
        columns = [
            ("recording_id", "Recording"),
            ("agreement", "Agreement"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1"),
            ("fp_rate", "FP rate"),
            ("fn_rate", "FN rate"),
            ("bad_fraction_human", "Human bad%"),
            ("bad_fraction_model", "Model bad%"),
            ("n_samples", "N samples"),
            ("elapsed_s", "Elapsed (s)"),
            ("__overlay", "Overlay"),
        ]
        table = QTableWidget(len(per_rec), len(columns))
        table.setHorizontalHeaderLabels([c[1] for c in columns])
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)

        for row_i, rec in enumerate(per_rec):
            for col_i, (key, _label) in enumerate(columns):
                if key == "__overlay":
                    rid = str(rec.get("recording_id", ""))
                    btn = QPushButton("View")
                    btn.setToolTip(
                        "Open the signal with human + model overlays "
                        "for this held-out recording."
                    )
                    # Disabled when we can't resolve the manifest entry
                    # (e.g. dialog opened from a saved JSON report
                    # without a corresponding manifest path).
                    enabled = rid in self._rec_by_id
                    btn.setEnabled(enabled)
                    if not enabled:
                        btn.setToolTip(
                            "Manifest entry not found for this "
                            "recording -- overlay unavailable."
                        )
                    btn.clicked.connect(
                        lambda _checked=False, _rid=rid:
                        self._open_overlay(_rid)
                    )
                    table.setCellWidget(row_i, col_i, btn)
                    continue
                v = rec.get(key)
                if key == "recording_id":
                    txt = str(v)
                elif key == "n_samples":
                    txt = f"{int(v):,}" if v is not None else "—"
                elif key == "elapsed_s":
                    txt = f"{v:.1f}" if v is not None else "—"
                elif key in ("agreement", "fp_rate", "fn_rate",
                              "bad_fraction_human", "bad_fraction_model"):
                    txt = _fmt_pct(v)
                else:   # precision / recall / f1
                    txt = _fmt_ratio(v)
                item = QTableWidgetItem(txt)
                if key != "recording_id":
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(row_i, col_i, item)

        table.resizeColumnsToContents()
        table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        return table

    # ----------------------------------------------------------------
    # Overlay window plumbing
    # ----------------------------------------------------------------

    def _load_recordings_by_id(self) -> dict[str, dict]:
        """Build {recording_id: manifest_entry} so the View button can
        look up source_path + splitter_bad_path in O(1). Returns an
        empty dict (no error) if the manifest can't be loaded -- the
        View buttons just stay disabled in that case."""
        if self._manifest_path is None or not self._manifest_path.exists():
            return {}
        try:
            from detector.manifest import Manifest
            m = Manifest.load(self._manifest_path)
            return {
                r["recording_id"]: r
                for r in m.list_recordings(include_held_out=True)
            }
        except Exception:
            return {}

    def _open_overlay(self, recording_id: str) -> None:
        rec = self._rec_by_id.get(recording_id)
        if rec is None:
            QMessageBox.warning(
                self, "Overlay unavailable",
                f"No manifest entry found for recording {recording_id}.",
            )
            return
        cache_entry = self._interval_cache.get(recording_id)
        # Lazy import to keep the dialog import light (signal_viewer
        # transitively imports pyqtgraph + h5py).
        from ui.windows.heldout_overlay_window import HeldoutOverlayWindow
        win = HeldoutOverlayWindow(rec, cache_entry, parent=self)
        # Use top-level Window flag so the user can park it next to
        # the results dialog without it being modal-trapped.
        win.setWindowFlag(Qt.Window, True)
        win.show()
        self._overlay_windows.append(win)

    # ----------------------------------------------------------------
    # Aggregate box
    # ----------------------------------------------------------------

    def _build_aggregate_box(self, report: dict) -> QGroupBox:
        agg = report.get("aggregate") or {}
        n = agg.get("n_recordings", 0)
        box = QGroupBox(f"Aggregate (n = {n} recordings)")
        layout = QHBoxLayout(box)
        for col_label, key in [("Sample-weighted (micro)", "micro"),
                                ("Recording-weighted (macro)", "macro")]:
            block = agg.get(key) or {}
            sub = QGroupBox(col_label)
            sub_l = QVBoxLayout(sub)
            rows = [
                ("Agreement",        _fmt_pct(block.get("agreement"))),
                ("Precision",        _fmt_ratio(block.get("precision"))),
                ("Recall",           _fmt_ratio(block.get("recall"))),
                ("F1",               _fmt_ratio(block.get("f1"))),
                ("FP rate",          _fmt_pct(block.get("fp_rate"))),
                ("FN rate",          _fmt_pct(block.get("fn_rate"))),
                ("Human bad%",       _fmt_pct(block.get("bad_fraction_human"))),
                ("Model bad%",       _fmt_pct(block.get("bad_fraction_model"))),
            ]
            for k, v in rows:
                sub_l.addWidget(QLabel(f"<b>{k}:</b> {v}"))
            sub_l.addStretch(1)
            layout.addWidget(sub)
        return box

    # ----------------------------------------------------------------
    # Skipped recordings
    # ----------------------------------------------------------------

    def _build_skipped_box(self, report: dict) -> QGroupBox:
        skipped = report.get("skipped", []) or []
        box = QGroupBox(f"Skipped — {len(skipped)} recording(s)")
        layout = QVBoxLayout(box)
        for s in skipped:
            layout.addWidget(QLabel(
                f"<b>{s.get('recording_id', '?')}:</b> "
                f"{s.get('error', '?')}"
            ))
        return box

    # ----------------------------------------------------------------
    # Save report
    # ----------------------------------------------------------------

    def _save_json(self) -> None:
        version = self._report.get("model_version", "model")
        default_name = f"heldout_eval_{version}.json"
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save held-out evaluation report",
            default_name, "JSON (*.json)",
        )
        if not path_str:
            return
        Path(path_str).write_text(
            json.dumps(self._report, indent=2, default=float)
        )
