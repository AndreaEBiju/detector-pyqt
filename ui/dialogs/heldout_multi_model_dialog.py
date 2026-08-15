"""Multi-model held-out evaluation: picker + results dialogs.

Two dialogs:

  - HeldoutMultiModelPicker: lets the user choose 2-N model versions
    to compare against the same held-out recordings. Opened by Tools
    -> "Compare models on held-out…" before any work starts. Returns
    the list of chosen version short strings (e.g. ["v0.2.1",
    "v0.2.2"]). Cancel aborts.

  - HeldoutMultiModelDialog: results display. Side-by-side aggregate
    metrics columns (one per model) and a per-recording table where
    each recording is one row with model-grouped columns. Per-row
    "View overlay" button opens the multi-model HeldoutOverlayWindow
    (which itself has a Recording + Model dropdown for interactive
    browsing).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QDialog, QDialogButtonBox, QFileDialog,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
    QPushButton, QRadioButton, QScrollArea, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)


# ----------------------------------------------------------------------
# Model picker
# ----------------------------------------------------------------------

class HeldoutMultiModelPicker(QDialog):
    """Checkbox-list picker for comparing multiple models. Caps at 3
    by default so the side-by-side metrics columns stay readable in
    the results dialog; pass `max_models` to relax."""

    def __init__(
        self,
        versions: list[str],
        *,
        promoted: Optional[str] = None,
        default_selected: Optional[list[str]] = None,
        max_models: int = 3,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Compare models on held-out")
        self.resize(420, 360)
        self._max_models = int(max_models)
        self._promoted = promoted

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"<b>Select {self._max_models} model(s) to compare.</b><br>"
            "Each will run inference against every held-out recording;"
            " models are parallelized across recordings.<br>"
            "<i>★ = currently promoted</i>"
        ))

        default_set = set(default_selected or ([promoted] if promoted else []))
        self._checks: list[tuple[QCheckBox, str]] = []
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_l = QVBoxLayout(inner)
        for v in versions:
            label = f"{v} ★" if v == promoted else v
            cb = QCheckBox(label)
            cb.setChecked(v in default_set)
            cb.stateChanged.connect(self._enforce_cap)
            inner_l.addWidget(cb)
            self._checks.append((cb, v))
        inner_l.addStretch(1)
        scroll.setWidget(inner)
        layout.addWidget(scroll)

        self._count_label = QLabel("")
        layout.addWidget(self._count_label)
        self._update_count()

        # ---- Parallelization mode ----------------------------------
        # Default recommendation depends on detected RAM. We surface
        # the hint via the option's subtitle so the user knows which
        # to pick without reading docs.
        para_box = QGroupBox("Parallelization")
        para_l = QVBoxLayout(para_box)
        para_l.addWidget(QLabel(
            "<i>How to parallelize when running models against "
            "recordings.</i>"
        ))
        self._parallelize_group = QButtonGroup(self)
        self._radio_recordings = QRadioButton(
            "Across recordings (recommended, safest)"
        )
        self._radio_recordings.setToolTip(
            "One worker per recording; models run sequentially inside "
            "each worker. Each recording's signal is loaded once. "
            "Bounded by len(held_recs), so no benefit beyond that many "
            "cores -- but it's the safest on memory-constrained "
            "machines and never reloads the same recording."
        )
        self._radio_flat = QRadioButton(
            "Fully parallel (across recordings AND models)"
        )
        self._radio_flat.setToolTip(
            "Flattens (recording, model) pairs into one job each. "
            "Higher parallelism on high-core boxes with few recordings; "
            "the same recording may be loaded by multiple workers "
            "concurrently (more peak RAM). The backend auto-caps the "
            "worker count to fit available RAM (with a 20% safety "
            "buffer) and falls back to 'across recordings' if the cap "
            "drops below 2 workers."
        )
        self._parallelize_group.addButton(self._radio_recordings, 0)
        self._parallelize_group.addButton(self._radio_flat, 1)
        para_l.addWidget(self._radio_recordings)
        para_l.addWidget(self._radio_flat)
        # Auto-recommend "flat" only on machines with both spare cores
        # AND ample RAM. Otherwise default to the safer mode.
        recommend_flat = self._machine_can_handle_flat()
        if recommend_flat:
            self._radio_flat.setChecked(True)
            self._radio_flat.setText(
                "Fully parallel (across recordings AND models) — "
                "recommended for your machine"
            )
        else:
            self._radio_recordings.setChecked(True)
        layout.addWidget(para_box)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self._on_ok)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)
        self._btn_box = btn_box

    @staticmethod
    def _machine_can_handle_flat() -> bool:
        """Recommend 'flat' mode only when the machine has BOTH:
          - many cores (>= 8)  -- otherwise per-recording parallelism
            already saturates the CPU
          - ample RAM (>= 24 GB)  -- so a few extra concurrent signal
            loads don't push us toward OOM
        Conservative threshold: errs on the side of recommending the
        safer mode. Users with edge-case hardware can flip the radio
        regardless of recommendation."""
        try:
            import os as _os
            cpu = _os.cpu_count() or 4
        except Exception:
            cpu = 4
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            ram_gb = 8.0
        return cpu >= 8 and ram_gb >= 24.0

    def _enforce_cap(self, _state: int) -> None:
        """When user already has `max_models` checked, the other
        checkboxes auto-disable so they can't sneak past the cap."""
        chosen = self._chosen_versions()
        at_cap = len(chosen) >= self._max_models
        for cb, v in self._checks:
            if v in chosen:
                cb.setEnabled(True)
            else:
                cb.setEnabled(not at_cap)
        self._update_count()

    def _update_count(self) -> None:
        n = len(self._chosen_versions())
        cap = self._max_models
        self._count_label.setText(
            f"Selected: <b>{n}/{cap}</b>"
        )

    def _chosen_versions(self) -> list[str]:
        return [v for cb, v in self._checks if cb.isChecked()]

    def _on_ok(self) -> None:
        if len(self._chosen_versions()) < 2:
            QMessageBox.information(
                self, "Need at least 2 models",
                "Multi-model comparison needs at least 2 models. "
                "Use the single-model 'Held-out evaluation…' for one.",
            )
            return
        self.accept()

    def chosen_versions(self) -> list[str]:
        return self._chosen_versions()

    def chosen_parallelize(self) -> str:
        """Returns the backend's parallelize= argument: 'flat' or
        'recordings'. The picker exposes it as a radio above OK."""
        if self._radio_flat.isChecked():
            return "flat"
        return "recordings"


# ----------------------------------------------------------------------
# Results dialog
# ----------------------------------------------------------------------

def _fmt_pct(v) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.2f}%"


def _fmt_ratio(v) -> str:
    if v is None:
        return "—"
    return f"{v:.4f}"


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


_METRIC_ROWS = [
    ("agreement",          "Agreement",          True),   # pct
    ("precision",          "Precision",          False),
    ("recall",             "Recall",             False),
    ("f1",                 "F1",                 False),
    ("fp_rate",            "FP rate",            True),
    ("fn_rate",            "FN rate",            True),
    ("bad_fraction_human", "Human bad%",         True),
    ("bad_fraction_model", "Model bad%",         True),
]

# Weighted business score used to rank models after held-out comparison.
# Positive metrics are better as-is; error-rate metrics are inverted
# (1 - rate) before aggregation so larger is always better.
_GLOBAL_SCORE_WEIGHTS = {
    "recall": 0.30,
    "precision": 0.20,
    "f1": 0.20,
    "agreement": 0.15,
    "fp_rate": 0.075,
    "fn_rate": 0.075,
}


def _compute_global_score(
    metrics: dict, weights: Optional[dict] = None,
) -> Optional[float]:
    """Compute a normalized weighted score in [0, 1].

    Missing metrics are skipped and remaining weights are renormalized.
    Returns None only if no weighted metric is available.
    """
    ws = dict(weights or _GLOBAL_SCORE_WEIGHTS)
    total_weight = 0.0
    score_sum = 0.0
    for key, weight in ws.items():
        raw = metrics.get(key)
        if raw is None:
            continue
        try:
            v = _clamp01(float(raw))
        except Exception:
            continue
        contrib = v if key not in {"fp_rate", "fn_rate"} else (1.0 - v)
        score_sum += float(weight) * contrib
        total_weight += float(weight)
    if total_weight <= 0:
        return None
    return _clamp01(score_sum / total_weight)


def _rank_models_from_report(
    report: dict, weights: Optional[dict] = None,
) -> list[dict]:
    """Build score/ranking rows from a multi-model held-out report."""
    rows: list[dict] = []
    for m in (report.get("models", []) or []):
        version = m.get("model_version", "?")
        micro = ((m.get("report") or {}).get("aggregate") or {}).get("micro") or {}
        score = _compute_global_score(micro, weights=weights)
        rows.append({
            "model_version": str(version),
            "score": score,
            "metrics": {
                "agreement": micro.get("agreement"),
                "precision": micro.get("precision"),
                "recall": micro.get("recall"),
                "f1": micro.get("f1"),
                "fp_rate": micro.get("fp_rate"),
                "fn_rate": micro.get("fn_rate"),
            },
        })

    rows.sort(
        key=lambda r: (
            r["score"] is None,
            -float(r["score"]) if r["score"] is not None else 0.0,
            r["model_version"],
        )
    )
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


class HeldoutMultiModelDialog(QDialog):
    def __init__(
        self,
        report: dict,
        *,
        interval_cache: Optional[dict] = None,
        manifest_path: Optional[Path] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Multi-model held-out evaluation")
        self.resize(1280, 720)
        self._report = report
        self._score_weights = dict(_GLOBAL_SCORE_WEIGHTS)
        self._ranking_rows = _rank_models_from_report(
            report, weights=self._score_weights
        )
        self._score_by_version = {
            row["model_version"]: row["score"]
            for row in self._ranking_rows
        }
        self._interval_cache = interval_cache or {}
        self._manifest_path = (
            Path(manifest_path) if manifest_path is not None else None
        )
        self._rec_by_id: dict[str, dict] = self._load_recordings_by_id()
        self._overlay_windows: list[QWidget] = []

        models = report.get("models", []) or []
        versions = [m["model_version"] for m in models]
        n_targeted = report.get("n_recordings_targeted", 0)

        root = QVBoxLayout(self)

        # ---- Header --------------------------------------------------
        # Surface the parallelize-used audit fields so the user can see
        # whether a "flat" request was honored or fell back to
        # "recordings" mode under RAM pressure (the backend records
        # this in _parallelize_used / _workers_cap_reason).
        para_req = report.get("_parallelize_requested", "?")
        para_used = report.get("_parallelize_used", "?")
        n_workers_used = report.get("_n_workers_used", "?")
        para_note = ""
        if para_req != "?" and para_used != "?":
            if para_req != para_used:
                para_note = (
                    f"     <span style='color:#d62728;'>"
                    f"<b>Parallelization:</b> requested <i>{para_req}</i> → "
                    f"fell back to <i>{para_used}</i> "
                    f"({n_workers_used} workers)</span>"
                )
            else:
                para_note = (
                    f"     <b>Parallelization:</b> {para_used} "
                    f"({n_workers_used} workers)"
                )
        header = QLabel(
            f"<b>Models:</b> {', '.join(versions) or '—'}     "
            f"<b>Held-out recordings:</b> {n_targeted}"
            f"{para_note}"
        )
        header.setTextFormat(Qt.RichText)
        header.setToolTip(
            str(report.get("_workers_cap_reason", ""))
        )
        root.addWidget(header)

        if not models or n_targeted == 0:
            empty = QLabel(
                "<i>No comparison data. Mark at least one recording's "
                "<b>held_out</b> flag True in Training Management → Add "
                "recording, then re-run.</i>"
            )
            empty.setTextFormat(Qt.RichText)
            empty.setWordWrap(True)
            root.addWidget(empty)
        else:
            root.addWidget(self._build_recommendation_box())
            root.addWidget(self._build_aggregate_box(report))
            root.addWidget(self._build_ranking_table())
            root.addWidget(self._build_per_recording_table(report))

        if report.get("skipped"):
            root.addWidget(self._build_skipped_box(report))

        # ---- Buttons -------------------------------------------------
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
    # Aggregate: side-by-side columns, one per model
    # ----------------------------------------------------------------

    def _build_aggregate_box(self, report: dict) -> QGroupBox:
        models = report["models"]
        box = QGroupBox(
            f"Aggregate metrics (n = {report.get('n_recordings_targeted', 0)} "
            "recordings, micro = sample-weighted)"
        )
        layout = QHBoxLayout(box)
        for m in models:
            version = m["model_version"]
            agg = m["report"].get("aggregate") or {}
            block = agg.get("micro") or {}
            sub = QGroupBox(version)
            sub_l = QVBoxLayout(sub)
            for key, label, is_pct in _METRIC_ROWS:
                v = block.get(key)
                s = _fmt_pct(v) if is_pct else _fmt_ratio(v)
                sub_l.addWidget(QLabel(f"<b>{label}:</b> {s}"))
            score = self._score_by_version.get(version)
            sub_l.addWidget(QLabel(
                f"<b>Global score:</b> {_fmt_pct(score)}"
            ))
            sub_l.addStretch(1)
            layout.addWidget(sub)
        return box

    def _build_recommendation_box(self) -> QGroupBox:
        box = QGroupBox("Recommended model")
        layout = QVBoxLayout(box)
        if not self._ranking_rows:
            layout.addWidget(QLabel("No model available for ranking."))
            return box
        best = self._ranking_rows[0]
        score = best.get("score")
        if score is None:
            layout.addWidget(QLabel(
                "Not enough aggregate metrics to compute a global score."
            ))
            return box

        runner_up = self._ranking_rows[1] if len(self._ranking_rows) > 1 else None
        gap = None
        if runner_up is not None and runner_up.get("score") is not None:
            gap = float(score) - float(runner_up["score"])

        metrics = best.get("metrics", {})
        expl = (
            f"<b>{best.get('model_version', '?')}</b> is currently the top "
            f"candidate with global score <b>{_fmt_pct(score)}</b> "
            f"(Recall {_fmt_pct(metrics.get('recall'))}, "
            f"Precision {_fmt_ratio(metrics.get('precision'))}, "
            f"F1 {_fmt_ratio(metrics.get('f1'))}, "
            f"FP {_fmt_pct(metrics.get('fp_rate'))}, "
            f"FN {_fmt_pct(metrics.get('fn_rate'))})."
        )
        if gap is not None:
            expl += f" Margin vs #2: <b>{_fmt_pct(gap)}</b>."
        lbl = QLabel(expl)
        lbl.setTextFormat(Qt.RichText)
        lbl.setWordWrap(True)
        layout.addWidget(lbl)
        return box

    def _build_ranking_table(self) -> QGroupBox:
        box = QGroupBox("Automatic ranking")
        layout = QVBoxLayout(box)
        rows = self._ranking_rows
        table = QTableWidget(len(rows), 8)
        table.setHorizontalHeaderLabels([
            "Rank", "Model", "Global score",
            "Recall", "Precision", "F1", "FP%", "FN%",
        ])
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)

        for i, row in enumerate(rows):
            m = row.get("metrics", {})
            vals = [
                str(row.get("rank", i + 1)),
                str(row.get("model_version", "?")),
                _fmt_pct(row.get("score")),
                _fmt_pct(m.get("recall")),
                _fmt_ratio(m.get("precision")),
                _fmt_ratio(m.get("f1")),
                _fmt_pct(m.get("fp_rate")),
                _fmt_pct(m.get("fn_rate")),
            ]
            for col, txt in enumerate(vals):
                item = QTableWidgetItem(txt)
                if col not in (1,):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(i, col, item)

        table.resizeColumnsToContents()
        table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        layout.addWidget(table)

        weights_txt = ", ".join(
            f"{k}={v:.3f}" for k, v in self._score_weights.items()
        )
        note = QLabel(
            "Global score uses weighted aggregate held-out metrics: "
            f"{weights_txt}. For FP/FN rates, lower is better "
            "(converted internally as 1 - rate)."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        return box

    # ----------------------------------------------------------------
    # Per-recording table: each row = recording, columns grouped per
    # model. Last column = a "View overlay" button.
    # ----------------------------------------------------------------

    def _build_per_recording_table(self, report: dict) -> QWidget:
        models = report["models"]
        versions = [m["model_version"] for m in models]
        # Pivot the per-recording rows: for each rid we want one row
        # with each model's metrics. Each model report has a
        # `per_recording` list -- they share rec_id order in normal
        # cases but defensively index by rid.
        per_model_by_rid: dict[str, dict[str, dict]] = {}
        for m in models:
            v = m["model_version"]
            for rec in m["report"].get("per_recording", []):
                rid = rec.get("recording_id", "")
                per_model_by_rid.setdefault(rid, {})[v] = rec
        rids = sorted(per_model_by_rid.keys())

        # Column layout:
        #   col 0:  Recording
        #   cols 1..K:  for each model -> (Agreement, Prec, Recall, FP%, FN%)
        #   last col: View overlay button
        per_model_cols = [
            ("agreement",          "Agree",    True),
            ("precision",          "Prec",     False),
            ("recall",             "Recall",   False),
            ("f1",                 "F1",       False),
            ("fp_rate",            "FP%",      True),
            ("fn_rate",            "FN%",      True),
            ("bad_fraction_model", "Bad%",     True),
        ]
        n_cols = 1 + len(versions) * len(per_model_cols) + 1
        headers: list[str] = ["Recording"]
        for v in versions:
            for _key, label, _pct in per_model_cols:
                headers.append(f"{v}\n{label}")
        headers.append("Overlay")

        table = QTableWidget(len(rids), n_cols)
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)

        for row_i, rid in enumerate(rids):
            table.setItem(row_i, 0, QTableWidgetItem(rid))
            col_idx = 1
            for v in versions:
                rec = per_model_by_rid.get(rid, {}).get(v) or {}
                for key, _label, is_pct in per_model_cols:
                    raw = rec.get(key)
                    if "__error__" in rec:
                        cell_txt = "err"
                    elif raw is None:
                        cell_txt = "—"
                    elif is_pct:
                        cell_txt = _fmt_pct(raw)
                    else:
                        cell_txt = _fmt_ratio(raw)
                    item = QTableWidgetItem(cell_txt)
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    if "__error__" in rec:
                        item.setToolTip(str(rec.get("__error__", "")))
                    table.setItem(row_i, col_idx, item)
                    col_idx += 1
            # View overlay button
            btn = QPushButton("View")
            enabled = rid in self._rec_by_id and rid in self._interval_cache
            btn.setEnabled(enabled)
            if not enabled:
                btn.setToolTip(
                    "Overlay unavailable -- either the manifest entry "
                    "for this recording isn't loadable, or no interval "
                    "cache was captured for it."
                )
            else:
                btn.setToolTip(
                    "Open the signal with human ground truth + per-"
                    "model prediction overlays. Switch which model "
                    "to display via the dropdown in the overlay window."
                )
            btn.clicked.connect(
                lambda _checked=False, _rid=rid, _vs=list(versions):
                self._open_overlay(_rid, _vs)
            )
            table.setCellWidget(row_i, col_idx, btn)

        table.resizeColumnsToContents()
        table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        return table

    # ----------------------------------------------------------------
    # Manifest -> rid lookup (for overlay button availability)
    # ----------------------------------------------------------------

    def _load_recordings_by_id(self) -> dict[str, dict]:
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

    def _open_overlay(self, recording_id: str, versions: list[str]) -> None:
        rec = self._rec_by_id.get(recording_id)
        cache_entry = self._interval_cache.get(recording_id)
        if rec is None or cache_entry is None:
            QMessageBox.warning(
                self, "Overlay unavailable",
                f"No manifest entry or interval cache for "
                f"{recording_id}.",
            )
            return
        # Lazy import so opening the dialog doesn't drag in pyqtgraph.
        from ui.windows.heldout_overlay_window import HeldoutOverlayWindow
        # The single-model overlay window already accepts a multi-model
        # cache shape via the optional `available_versions` argument
        # (added in the same Phase-3 work that built the dropdown).
        # Pass the cache straight through.
        win = HeldoutOverlayWindow(
            rec, cache_entry,
            available_versions=versions,
            parent=self,
        )
        win.setWindowFlag(Qt.Window, True)
        win.show()
        self._overlay_windows.append(win)

    # ----------------------------------------------------------------
    # Skipped + save
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

    def _save_json(self) -> None:
        default_name = "heldout_eval_multi.json"
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save multi-model held-out report",
            default_name, "JSON (*.json)",
        )
        if not path_str:
            return
        payload = dict(self._report)
        payload["ui_global_score"] = {
            "weights": dict(self._score_weights),
            "ranking": list(self._ranking_rows),
            "recommended_model": (
                self._ranking_rows[0]["model_version"]
                if self._ranking_rows and self._ranking_rows[0]["score"] is not None
                else None
            ),
        }
        Path(path_str).write_text(
            json.dumps(payload, indent=2, default=float)
        )
