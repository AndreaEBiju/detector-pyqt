"""Training management window — manifest table, model versions,
retrain controls, settings. Mirrors the Streamlit
`2_Training_Management.py` page.

Four tabs in a QTabWidget:
- **Manifest** — recordings currently in `training_manifest.json`.
  Add/remove with the usual safeguards.
- **Versions** — model_v*/ directories with provenance + rollback.
- **Retrain** — w_neg / seed / no-loro / rebuild / force-promote
  controls; spawns a subprocess via `ui.workers.retrain_worker` and
  surfaces live progress + log tail. Past-jobs section at the bottom.
- **Settings** — UI-side preferences that don't fit on the main
  window's Edit menu.

The window is non-modal so the user can keep the main viewer open
for cross-reference while editing the manifest or watching a retrain.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QInputDialog, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSpinBox, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QTextEdit, QToolBar, QVBoxLayout, QWidget,
    QDoubleSpinBox,
)

_repo_root = Path(__file__).resolve().parent.parent.parent
_detector_core = _repo_root / "detector-core"
if _detector_core.exists() and str(_detector_core) not in sys.path:
    sys.path.insert(0, str(_detector_core))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from detector import paths as detector_paths
from detector import retrain as RT
from detector.manifest import Manifest, ManifestError
from detector.preprocessing import profiles as detector_profiles

from ui.data import settings as ui_settings
from ui.workers.retrain_worker import (
    RetrainMonitor, list_recent_jobs,
)


class TrainingWindow(QMainWindow):
    """4-tab training management window."""

    # Emitted whenever the manifest or current_model changes so the
    # caller (main window) can refresh its toolbar dropdowns etc.
    manifest_or_versions_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Training management")
        self.resize(1100, 720)

        self._monitor = RetrainMonitor(self)
        self._monitor.status_changed.connect(self._on_retrain_status)
        self._monitor.finished.connect(self._on_retrain_finished)
        self._monitor.failed.connect(self._on_retrain_failed)

        self._tabs = QTabWidget()
        self._tab_manifest = self._build_manifest_tab()
        self._tab_versions = self._build_versions_tab()
        self._tab_retrain = self._build_retrain_tab()
        self._tab_convergence = self._build_convergence_tab()
        self._tab_preprocessing = self._build_preprocessing_tab()
        self._tab_settings = self._build_settings_tab()
        self._tabs.addTab(self._tab_manifest, "Manifest")
        self._tabs.addTab(self._tab_versions, "Versions")
        self._tabs.addTab(self._tab_retrain, "Retrain")
        self._tabs.addTab(self._tab_convergence, "Convergence")
        self._tabs.addTab(self._tab_preprocessing, "Preprocessing")
        self._tabs.addTab(self._tab_settings, "Settings")
        self.setCentralWidget(self._tabs)

        self._refresh_all()
        # If a retrain is already running from a previous session, pick
        # up its monitoring on first open.
        self._maybe_attach_running_job()

    # ==================================================================
    # Manifest tab
    # ==================================================================

    def _build_manifest_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        # Summary header
        self._manifest_summary = QLabel("")
        self._manifest_summary.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(self._manifest_summary)

        # Table
        self._manifest_table = QTableWidget(0, 7)
        self._manifest_table.setHorizontalHeaderLabels([
            "ID", "type", "fs", "n_samples", "added", "last trained on",
            "notes",
        ])
        self._manifest_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._manifest_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._manifest_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._manifest_table.setAlternatingRowColors(True)
        hdr = self._manifest_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(6, QHeaderView.Stretch)
        layout.addWidget(self._manifest_table)

        # Action buttons
        btn_row = QHBoxLayout()
        self._btn_add_rec = QPushButton("➕ Add recording…")
        self._btn_add_rec.clicked.connect(self._on_add_recording)
        self._btn_remove_rec = QPushButton("— Remove selected (preserves files)")
        self._btn_remove_rec.clicked.connect(self._on_remove_recording)
        self._btn_refresh_manifest = QPushButton("↻ Refresh")
        self._btn_refresh_manifest.clicked.connect(self._refresh_manifest_tab)
        btn_row.addWidget(self._btn_add_rec)
        btn_row.addWidget(self._btn_remove_rec)
        btn_row.addStretch(1)
        btn_row.addWidget(self._btn_refresh_manifest)
        layout.addLayout(btn_row)
        return w

    def _refresh_manifest_tab(self) -> None:
        manifest_path = detector_paths.get_manifest_path()
        if not manifest_path.exists():
            self._manifest_summary.setText(
                f"No manifest at {manifest_path}. Run `detector init` to create one."
            )
            self._manifest_table.setRowCount(0)
            return
        try:
            m = Manifest.load(manifest_path)
        except Exception as exc:
            self._manifest_summary.setText(f"Failed to load manifest: {exc}")
            self._manifest_table.setRowCount(0)
            return
        recs = m.list_recordings()
        cmv = m.current_model_version or "—"
        n_untrained = (
            len(m.diff_since(cmv)["added"]) if m.current_model_version else 0
        )
        self._manifest_summary.setText(
            f"{len(recs)} recordings  ·  current model: {cmv}  ·  "
            f"untrained since current: {n_untrained}"
        )
        self._manifest_table.setRowCount(len(recs))
        for i, r in enumerate(recs):
            cells = [
                str(r.get("recording_id", "")),
                str(r.get("rec_type", "")),
                str(r.get("fs", "")),
                f"{int(r.get('n_samples', 0)):,}",
                str(r.get("added_at", "")),
                str(r.get("model_version_last_trained_on") or "—"),
                str(r.get("notes", "")),
            ]
            for col, text in enumerate(cells):
                self._manifest_table.setItem(i, col, QTableWidgetItem(text))

    def _on_add_recording(self) -> None:
        dialog = AddRecordingDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        rec = dialog.recording_dict()
        try:
            manifest_path = detector_paths.get_manifest_path()
            m = Manifest.load(manifest_path)
            m.add_recording(rec, by="pyqt")
            m.save(manifest_path)
        except ManifestError as exc:
            QMessageBox.critical(self, "Add failed", str(exc))
            return
        self._refresh_all()
        self.manifest_or_versions_changed.emit()
        # Auto-retrain trigger (M4.4)
        self._maybe_prompt_auto_retrain()

    def _on_remove_recording(self) -> None:
        rows = self._manifest_table.selectedItems()
        if not rows:
            QMessageBox.information(self, "No selection", "Select a row first.")
            return
        idx = self._manifest_table.row(rows[0])
        rid_item = self._manifest_table.item(idx, 0)
        if rid_item is None:
            return
        rid = rid_item.text()
        confirm, ok = QInputDialog.getText(
            self, "Confirm remove",
            f"Type the recording_id `{rid}` again to confirm.\n"
            "(Files on disk are NOT deleted; only the manifest entry.)",
        )
        if not ok or confirm.strip() != rid:
            QMessageBox.information(
                self, "Cancelled", "Removal not confirmed; nothing changed.",
            )
            return
        manifest_path = detector_paths.get_manifest_path()
        try:
            m = Manifest.load(manifest_path)
            m.remove_recording(rid)
            m.save(manifest_path)
        except ManifestError as exc:
            QMessageBox.critical(self, "Remove failed", str(exc))
            return
        self._refresh_all()
        self.manifest_or_versions_changed.emit()

    # ==================================================================
    # Versions tab
    # ==================================================================

    def _build_versions_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        self._versions_summary = QLabel("")
        self._versions_summary.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(self._versions_summary)

        self._versions_table = QTableWidget(0, 9)
        self._versions_table.setHorizontalHeaderLabels([
            "version", "current?", "created", "recall_real", "recall_syn",
            "mean_bad_frac", "G1", "G2", "G3",
        ])
        self._versions_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._versions_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._versions_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._versions_table.setAlternatingRowColors(True)
        layout.addWidget(self._versions_table)

        btn_row = QHBoxLayout()
        self._btn_view_provenance = QPushButton("📄 View provenance JSON")
        self._btn_view_provenance.clicked.connect(self._on_view_provenance)
        self._btn_rollback = QPushButton("⟲ Roll back to selected version…")
        self._btn_rollback.clicked.connect(self._on_rollback)
        btn_row.addWidget(self._btn_view_provenance)
        btn_row.addWidget(self._btn_rollback)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)
        return w

    def _refresh_versions_tab(self) -> None:
        versions = RT.list_versions()
        cmv = None
        try:
            m = Manifest.load(detector_paths.get_manifest_path())
            cmv = m.current_model_version
        except Exception:
            pass
        self._versions_summary.setText(
            f"{len(versions)} version(s) on disk  ·  current: {cmv or '—'}"
        )
        self._versions_table.setRowCount(len(versions))
        for i, v in enumerate(versions):
            prov = RT.load_provenance(v) or {}
            ls = (prov or {}).get("loro_summary") or {}
            cells = [
                v,
                "★" if v == cmv else "",
                (prov or {}).get("created_at", "—"),
                _fmt(ls.get("recall_real"), 3),
                _fmt(ls.get("recall_syn"), 3),
                _fmt(ls.get("mean_bad_fraction"), 3),
                f"{ls.get('gate_1_pass_count', '—')}/{ls.get('n_folds', '—')}",
                f"{ls.get('gate_2_pass_count', '—')}/{ls.get('n_folds', '—')}",
                f"{ls.get('gate_3_pass_count', '—')}/{ls.get('n_folds', '—')}",
            ]
            for col, text in enumerate(cells):
                self._versions_table.setItem(i, col, QTableWidgetItem(str(text)))

    def _on_view_provenance(self) -> None:
        rows = self._versions_table.selectedItems()
        if not rows:
            QMessageBox.information(self, "No selection", "Pick a version row.")
            return
        idx = self._versions_table.row(rows[0])
        version_item = self._versions_table.item(idx, 0)
        if version_item is None:
            return
        version = version_item.text()
        prov = RT.load_provenance(version)
        if not prov:
            QMessageBox.information(
                self, "No provenance", f"No provenance.json for {version}.",
            )
            return
        ProvenanceDialog(version, prov, self).exec()

    def _on_rollback(self) -> None:
        rows = self._versions_table.selectedItems()
        if not rows:
            QMessageBox.information(self, "No selection", "Pick a version row.")
            return
        idx = self._versions_table.row(rows[0])
        version_item = self._versions_table.item(idx, 0)
        if version_item is None:
            return
        target_version = version_item.text()
        manifest_path = detector_paths.get_manifest_path()
        try:
            m = Manifest.load(manifest_path)
        except Exception as exc:
            QMessageBox.critical(self, "Rollback failed",
                                   f"Manifest load: {exc}")
            return
        if m.current_model_version == target_version:
            QMessageBox.information(
                self, "Already current",
                f"{target_version} is already current_model_version.",
            )
            return
        reason, ok = QInputDialog.getText(
            self, "Confirm rollback",
            f"current_model_version: {m.current_model_version or '—'} → "
            f"{target_version}\n\nReason (audited in manifest history):",
        )
        if not ok or not reason.strip():
            return
        try:
            res = RT.rollback_to_version(
                manifest_path, target_version, reason=reason.strip(),
                enforce_feature_schema=True,
            )
        except SystemExit as exc:
            # Feature-schema mismatch raises SystemExit per detector.retrain.
            override = QMessageBox.question(
                self, "Schema mismatch",
                f"Rollback failed: {exc}\n\n"
                "Override the feature-schema check? Only do this if you "
                "have verified the schemas are equivalent.",
            )
            if override != QMessageBox.Yes:
                return
            try:
                res = RT.rollback_to_version(
                    manifest_path, target_version, reason=reason.strip(),
                    enforce_feature_schema=False,
                )
            except Exception as exc2:
                QMessageBox.critical(self, "Rollback failed", str(exc2))
                return
        except Exception as exc:
            QMessageBox.critical(self, "Rollback failed", str(exc))
            return
        QMessageBox.information(
            self, "Rolled back",
            f"current_model_version: {res['from_version']} → {res['to_version']}",
        )
        self._refresh_all()
        self.manifest_or_versions_changed.emit()

    # ==================================================================
    # Retrain tab
    # ==================================================================

    def _build_retrain_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        # Controls group
        ctrl_box = QGroupBox("Retrain settings")
        form = QFormLayout(ctrl_box)
        self._w_neg_spin = QDoubleSpinBox()
        self._w_neg_spin.setRange(0.05, 1.0)
        self._w_neg_spin.setSingleStep(0.05)
        self._w_neg_spin.setValue(0.3)
        form.addRow("w_neg", self._w_neg_spin)
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 100_000)
        self._seed_spin.setValue(42)
        form.addRow("seed", self._seed_spin)
        self._no_loro_cb = QCheckBox(
            "Reuse cached LORO (only safe if manifest unchanged)"
        )
        form.addRow(self._no_loro_cb)
        self._rebuild_cb = QCheckBox(
            "Rebuild dataset_phase1 (~5-10 min per recording; "
            "needed after adding recordings)"
        )
        self._rebuild_cb.setToolTip(
            "Rebuilds raw features (Phase 1) from the manifest. "
            "Single-threaded feature extraction takes ~5-10 minutes per "
            "recording; with 4 parallel workers (default), the total "
            "depends on recording count and CPU core count. "
            "Override worker count with the DETECTOR_PHASE1_WORKERS "
            "env var.\n\n"
            "Does NOT regenerate synthetic positives (Phase 2) -- if you "
            "added new recordings, ALSO check the 'rebuild dataset_phase2' "
            "box below, or the model will silently train on stale Phase 2 "
            "data missing your new recordings."
        )
        form.addRow(self._rebuild_cb)
        self._rebuild_phase2_cb = QCheckBox(
            "Rebuild dataset_phase2 synthetic augmentation (~2-4 hours; "
            "required after adding recordings)"
        )
        self._rebuild_phase2_cb.setToolTip(
            "Regenerates dataset_phase2.parquet with synthetic positives "
            "for every recording in your current manifest. This is the "
            "step that ACTUALLY makes new recordings reachable for "
            "training -- without it, retrain trains on whatever recordings "
            "were in Phase 2 last time, regardless of what's in the manifest. "
            "Slow (~hours) but only needed when you've added recordings."
        )
        form.addRow(self._rebuild_phase2_cb)
        self._skip_review_cb = QCheckBox(
            "Skip Phase 5 review HTML regeneration"
        )
        self._skip_review_cb.setChecked(True)
        form.addRow(self._skip_review_cb)
        self._force_promote_cb = QCheckBox(
            "Force promote even if regressions detected (requires reason)"
        )
        form.addRow(self._force_promote_cb)
        # Active-learning loop: optionally fold the previous model's
        # review judgments back in as elevated-weight negatives. Empty
        # = no review feedback (default; legacy behavior).
        self._review_dir_edit = QLineEdit()
        self._review_dir_edit.setPlaceholderText(
            "(optional) path to previous model's review/ folder -- e.g. "
            "<artifacts>/model_v0.2.0/review"
        )
        review_row = QHBoxLayout()
        review_row.addWidget(self._review_dir_edit)
        btn_pick_review = QPushButton("...")
        btn_pick_review.clicked.connect(self._pick_review_dir)
        review_row.addWidget(btn_pick_review)
        form.addRow("review feedback dir", review_row)
        self._review_dir_edit.setToolTip(
            "If you reviewed the previous version's disagreement HTMLs "
            "and saved per-recording <rid>_review.json files, point at "
            "the folder containing them. Each 'false_positive' judgment "
            "becomes a high-weight (default 3.0) negative training "
            "example -- the canonical active-learning loop that turns "
            "your reviews into actual model improvements."
        )
        layout.addWidget(ctrl_box)

        # Action row
        action_row = QHBoxLayout()
        self._btn_start_retrain = QPushButton("▶ Retrain now")
        self._btn_start_retrain.clicked.connect(self._on_start_retrain)
        self._btn_cancel_retrain = QPushButton("✖ Cancel")
        self._btn_cancel_retrain.clicked.connect(self._on_cancel_retrain)
        self._btn_cancel_retrain.setEnabled(False)
        action_row.addWidget(self._btn_start_retrain)
        action_row.addWidget(self._btn_cancel_retrain)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        # Progress + log
        status_box = QGroupBox("Status")
        status_layout = QVBoxLayout(status_box)
        self._retrain_phase_label = QLabel("idle")
        self._retrain_phase_label.setStyleSheet(
            "font-family: monospace; font-size: 12px;"
        )
        status_layout.addWidget(self._retrain_phase_label)
        self._retrain_progress = QProgressBar()
        self._retrain_progress.setRange(0, 100)
        status_layout.addWidget(self._retrain_progress)
        self._retrain_log = QPlainTextEdit()
        self._retrain_log.setReadOnly(True)
        self._retrain_log.setMaximumBlockCount(2000)
        self._retrain_log.setStyleSheet(
            "font-family: monospace; font-size: 11px; background: #1a1a1a; color: #ddd;"
        )
        status_layout.addWidget(self._retrain_log, stretch=1)
        layout.addWidget(status_box, stretch=1)

        # Past jobs
        past_box = QGroupBox("Past retrain jobs (newest first)")
        past_layout = QVBoxLayout(past_box)
        self._past_jobs_label = QLabel("")
        self._past_jobs_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        self._past_jobs_label.setWordWrap(True)
        past_layout.addWidget(self._past_jobs_label)
        layout.addWidget(past_box)
        return w

    def _refresh_retrain_tab(self) -> None:
        jobs = list_recent_jobs(n=10)
        if not jobs:
            self._past_jobs_label.setText("(none yet)")
            return
        lines = []
        for j in jobs:
            try:
                from detector.retrain_subprocess import status as _status
                st = _status(j.job_id)
                phase = st.get("phase", "?")
                state = st.get("state", "?")
            except Exception:
                phase = "?"
                state = "?"
            lines.append(f"{j.job_id}  {state:<10s}  {phase}")
        self._past_jobs_label.setText("\n".join(lines))

    def _maybe_attach_running_job(self) -> None:
        """On window open, see if a retrain is still running from a
        previous session and attach to it."""
        for job in list_recent_jobs(n=5):
            try:
                from detector.retrain_subprocess import status as _status
                st = _status(job.job_id)
                if st.get("state") == "running":
                    self._monitor.attach(job.job_id)
                    self._btn_start_retrain.setEnabled(False)
                    self._btn_cancel_retrain.setEnabled(True)
                    self._retrain_phase_label.setText(
                        f"resuming job {job.job_id} …"
                    )
                    return
            except Exception:
                continue

    def _pick_review_dir(self) -> None:
        """File picker for the active-learning review feedback dir."""
        start_dir = self._review_dir_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(
            self, "Pick previous model's review/ folder", start_dir,
        )
        if path:
            self._review_dir_edit.setText(path)

    def _on_start_retrain(self) -> None:
        # Pre-flight readiness check BEFORE we spawn anything.
        # Catches missing files + missing Python packages + a real
        # h5py-with-compression smoke test so the user gets a clear
        # "fix these N things first" dialog instead of a multi-hour
        # subprocess that dies cryptically.
        rebuild_phase2 = bool(self._rebuild_phase2_cb.isChecked())
        rebuild_dataset = bool(self._rebuild_cb.isChecked())
        try:
            from detector import retrain as _RT
            report = _RT.check_retrain_readiness(
                detector_paths.get_manifest_path(),
                rebuild_dataset=rebuild_dataset,
                rebuild_phase2=rebuild_phase2,
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Pre-flight check crashed",
                f"check_retrain_readiness raised: {exc}\n\n"
                "This is unusual. Falling back to the unchecked spawn "
                "path -- if the retrain dies, run "
                "`python -m detector.cli check-retrain` from a "
                "terminal for diagnostics."
            )
            report = None
        if report and report.get("blockers"):
            # Two paths depending on what's blocking:
            #
            # (1) If the only blockers are missing data files (baselines,
            #     splitter clean/bad, etc.), offer an interactive "pick
            #     a folder, we'll auto-locate and copy" dialog so the
            #     user doesn't have to copy files by hand.
            # (2) If there are non-file blockers (missing Python deps,
            #     low disk, etc.), they can't be auto-fixed -- show the
            #     classic blocking dialog.
            missing = report.get("missing_files") or []
            non_file_blockers = [
                b for b in report["blockers"]
                if not (" missing: " in b
                        and ("baseline" in b
                             or "splitter_clean_path" in b
                             or "splitter_bad_path" in b))
            ]
            if missing and not non_file_blockers:
                if not self._offer_auto_locate(missing,
                                                rebuild_dataset,
                                                rebuild_phase2):
                    return  # user cancelled
                # _offer_auto_locate already re-ran the pre-flight if
                # it succeeded; fall through to spawn.
            else:
                blockers = report["blockers"]
                head = "\n".join(f"  - {b}" for b in blockers[:15])
                more = (f"\n  ... and {len(blockers) - 15} more"
                        if len(blockers) > 15 else "")
                QMessageBox.critical(
                    self, "Retrain pre-flight FAILED",
                    f"{len(blockers)} blocker(s) must be fixed before "
                    "the retrain can start:\n\n"
                    f"{head}{more}\n\n"
                    "(See the full report on the CLI: "
                    "`python -m detector.cli check-retrain "
                    f"{'--rebuild-dataset ' if rebuild_dataset else ''}"
                    f"{'--rebuild-phase2' if rebuild_phase2 else ''}`)"
                )
                return

        if self._force_promote_cb.isChecked():
            reason, ok = QInputDialog.getText(
                self, "Force promote reason",
                "Force-promote requires a reason (audited in manifest history):",
            )
            if not ok or not reason.strip():
                return
            force_reason = reason.strip()
        else:
            force_reason = None
        try:
            review_dir_text = self._review_dir_edit.text().strip()
            review_dir = Path(review_dir_text) if review_dir_text else None
            job = self._monitor.start_new(
                w_neg=float(self._w_neg_spin.value()),
                seed=int(self._seed_spin.value()),
                rebuild_dataset=rebuild_dataset,
                rebuild_phase2=rebuild_phase2,
                no_loro=bool(self._no_loro_cb.isChecked()),
                skip_review=bool(self._skip_review_cb.isChecked()),
                force_promote=bool(self._force_promote_cb.isChecked()),
                force_promote_reason=force_reason,
                manifest_path=detector_paths.get_manifest_path(),
                review_dir=review_dir,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Start retrain failed", str(exc))
            return
        self._btn_start_retrain.setEnabled(False)
        self._btn_cancel_retrain.setEnabled(True)
        self._retrain_log.clear()
        self._retrain_phase_label.setText(f"started job {job.job_id}")
        self._retrain_progress.setValue(0)

    def _offer_auto_locate(
        self, missing: list[dict], rebuild_dataset: bool,
        rebuild_phase2: bool,
    ) -> bool:
        """Interactive flow for resolving missing data files.

        Shows a dialog listing the N missing files and a 'Pick folder
        to search...' button. If the user picks a folder, walks it
        recursively for matching basenames and copies them to the
        expected paths. Re-runs the pre-flight after. Loops until
        either all files are found OR the user cancels.

        Returns True if the pre-flight passes after the helping
        (caller should proceed with spawning the retrain), False if
        the user cancelled (caller should abort).
        """
        from detector import retrain as _RT
        while missing:
            # Build a compact preview of what's missing.
            head_lines = []
            for entry in missing[:8]:
                rid = entry.get("recording_id") or "(no recording_id)"
                kind = entry.get("kind", "file")
                head_lines.append(
                    f"  - [{rid}] {kind}: "
                    f"{Path(entry['expected_path']).name}"
                )
            more = (f"\n  ... and {len(missing) - 8} more"
                    if len(missing) > 8 else "")
            head = "\n".join(head_lines)

            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Missing data files")
            msg.setText(
                f"{len(missing)} file(s) the retrain needs are not "
                "where the manifest expects them.\n\n"
                f"{head}{more}\n\n"
                "Pick a folder where these files might live (e.g. your "
                "data drive, a collaborator's baselines folder synced "
                "via Drive, etc.) and the UI will scan it for matching "
                "filenames and copy whatever it finds into the right "
                "place."
            )
            btn_pick = msg.addButton(
                "Pick folder to search...", QMessageBox.AcceptRole
            )
            msg.addButton("Cancel retrain", QMessageBox.RejectRole)
            msg.exec()

            if msg.clickedButton() is not btn_pick:
                return False  # user cancelled

            folder = QFileDialog.getExistingDirectory(
                self, "Pick a folder to search for missing files",
                str(Path.home()),
            )
            if not folder:
                continue  # user backed out of file dialog; loop again

            result = _RT.locate_and_copy_missing_files(missing, Path(folder))
            copied = result.get("copied", [])
            skipped = result.get("skipped", [])

            if not copied:
                QMessageBox.warning(
                    self, "No matching files found",
                    f"Scanned {folder} but found none of the "
                    f"{len(missing)} expected filenames. Try a "
                    "different folder, or cancel."
                )
                # Loop -- user can pick another folder.
                continue

            QMessageBox.information(
                self, "Files copied",
                f"Copied {len(copied)} of {len(missing)} files from "
                f"{folder}.\n\nRe-running the pre-flight check..."
            )

            # Re-run pre-flight to refresh the missing list.
            try:
                report = _RT.check_retrain_readiness(
                    detector_paths.get_manifest_path(),
                    rebuild_dataset=rebuild_dataset,
                    rebuild_phase2=rebuild_phase2,
                )
            except Exception as exc:
                QMessageBox.critical(
                    self, "Pre-flight crashed after locate",
                    f"check_retrain_readiness raised: {exc}"
                )
                return False

            # If there are NON-file blockers now (e.g. a stale dataset
            # we couldn't help with), bail to the classic dialog path.
            file_blockers = [
                b for b in report["blockers"]
                if " missing: " in b
                and ("baseline" in b or "splitter_clean" in b
                     or "splitter_bad" in b)
            ]
            non_file = [b for b in report["blockers"]
                        if b not in file_blockers]
            if non_file:
                # New issues we can't auto-fix -- show the standard
                # blocker dialog and abort.
                QMessageBox.critical(
                    self, "Other blockers remain",
                    "Files were copied but other blockers remain:\n\n"
                    + "\n".join(f"  - {b}" for b in non_file[:10])
                )
                return False
            if not report.get("missing_files"):
                return True  # all clean; proceed with retrain
            missing = report["missing_files"]  # loop with reduced set

        return True

    def _on_cancel_retrain(self) -> None:
        ok = self._monitor.cancel()
        if not ok:
            QMessageBox.warning(self, "Cancel failed",
                                  "Could not signal the retrain process.")

    def _on_retrain_status(self, st: dict) -> None:
        self._retrain_phase_label.setText(
            f"phase: {st.get('phase', '?')}  ·  "
            f"fold {st.get('fold_no', 0)}/12  ·  "
            f"gates {st.get('gate_no', 0)}/12  ·  "
            f"pid={st.get('pid')}"
        )
        self._retrain_progress.setValue(int(float(st.get("progress", 0.0)) * 100))
        tail = st.get("tail", "")
        if tail:
            # PlainText replace so we don't accumulate unbounded
            self._retrain_log.setPlainText(tail)
            cursor = self._retrain_log.textCursor()
            # PySide6's strict scoped enums require accessing the
            # enum on the class (or its MoveOperation subscope), not
            # the instance. `cursor.End` worked on PyQt5 / older
            # PySide via attribute fallback, but PySide 6.x raises
            # AttributeError — and this fires every 500ms via the
            # status timer, so the regression flooded the console.
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self._retrain_log.setTextCursor(cursor)

    def _on_retrain_finished(self, st: dict) -> None:
        self._btn_start_retrain.setEnabled(True)
        self._btn_cancel_retrain.setEnabled(False)
        self._retrain_phase_label.setText(f"succeeded · {st.get('phase', 'done')}")
        self._retrain_progress.setValue(100)
        self._refresh_versions_tab()
        self._refresh_manifest_tab()
        self._refresh_retrain_tab()
        self.manifest_or_versions_changed.emit()
        # Show regression report dialog: compare new model with previous
        self._show_regression_report_after_finish()

    def _on_retrain_failed(self, st: dict) -> None:
        self._btn_start_retrain.setEnabled(True)
        self._btn_cancel_retrain.setEnabled(False)
        self._retrain_phase_label.setText(f"FAILED · {st.get('phase', '?')}")
        QMessageBox.critical(
            self, "Retrain failed",
            f"Exit code: {st.get('exit_code')}\nphase: {st.get('phase')}\n\n"
            f"Last log:\n{st.get('tail', '')[-500:]}",
        )
        self._refresh_retrain_tab()

    def _show_regression_report_after_finish(self) -> None:
        """Surface the latest version's regression_report.json (if
        present) in a modal dialog."""
        versions = RT.list_versions()
        if not versions:
            return
        latest = versions[-1]
        try:
            artifacts_dir = detector_paths.get_artifacts_dir()
            rep_path = artifacts_dir / f"model_{latest}" / "regression_report.json"
            if rep_path.exists():
                report = json.loads(rep_path.read_text())
                RegressionReportDialog(latest, report, self).exec()
        except Exception:
            pass

    # ==================================================================
    # Convergence tab
    # ==================================================================

    def _build_convergence_tab(self) -> QWidget:
        """Plots LightGBM's per-iteration validation + training metrics
        for any version that has a `training_curve.json` sidecar (any
        version retrained after the curve-capture feature shipped).

        The plot uses pyqtgraph (already a dependency of the signal
        viewer) — one subplot per metric (logloss, AUC, …), with the
        train + val curves overlaid and a dashed vertical line at
        the iteration LightGBM picked via early stopping.
        """
        import pyqtgraph as pg

        w = QWidget()
        layout = QVBoxLayout(w)

        # Version selector at the top — most recent populated by default.
        top = QHBoxLayout()
        top.addWidget(QLabel("Model version:"))
        self._cv_version_combo = QComboBox()
        self._cv_version_combo.setMinimumWidth(200)
        self._cv_version_combo.currentIndexChanged.connect(
            lambda _i: self._refresh_convergence_plot()
        )
        top.addWidget(self._cv_version_combo)
        top.addStretch(1)
        self._cv_reload_btn = QPushButton("↻ Reload list")
        self._cv_reload_btn.clicked.connect(self._refresh_convergence_versions)
        top.addWidget(self._cv_reload_btn)
        layout.addLayout(top)

        # Status / metadata line shown above the plot.
        self._cv_status_label = QLabel("")
        self._cv_status_label.setStyleSheet("color: #aaa; padding: 4px;")
        self._cv_status_label.setWordWrap(True)
        layout.addWidget(self._cv_status_label)

        # Plot container — pyqtgraph GraphicsLayoutWidget supports
        # stacked subplots with shared X-axis. We populate it
        # dynamically in `_refresh_convergence_plot` because the
        # number of metrics isn't fixed at construction time.
        self._cv_plot_container = pg.GraphicsLayoutWidget()
        self._cv_plot_container.setBackground("#0e1117")
        layout.addWidget(self._cv_plot_container, stretch=1)

        # Legend / caption.
        legend = QLabel(
            "<span style='color:#aaa'>"
            "Grey = train metric · Blue = validation · "
            "Red dashed = best iteration picked by early stopping. "
            "A widening train↔val gap indicates overfitting; the "
            "dashed line is where LightGBM stopped to avoid it."
            "</span>"
        )
        legend.setWordWrap(True)
        layout.addWidget(legend)

        # Initial population.
        self._refresh_convergence_versions()
        return w

    def _refresh_convergence_versions(self) -> None:
        """Repopulate the version dropdown with everything that has
        a training_curve.json sidecar."""
        try:
            all_versions = RT.list_versions()
        except Exception:
            all_versions = []
        artifacts_dir = detector_paths.get_artifacts_dir()

        with_curve: list[str] = []
        without_curve: list[str] = []
        for v in all_versions:
            curve_path = artifacts_dir / f"model_{v}" / "training_curve.json"
            (with_curve if curve_path.exists() else without_curve).append(v)

        self._cv_version_combo.blockSignals(True)
        self._cv_version_combo.clear()
        for v in with_curve:
            self._cv_version_combo.addItem(v, v)
        # Disabled-looking entries for versions that predate the
        # curve-capture feature — surfaced for clarity, not pickable.
        for v in without_curve:
            self._cv_version_combo.addItem(f"{v}  (no curve)", None)
        # Default: prefer the current promoted model if it has a curve;
        # otherwise the most recently trained version.
        try:
            cmv = detector_paths.get_current_model_version() or ""
            cmv_short = cmv.removeprefix("model_") if cmv else ""
        except Exception:
            cmv_short = ""
        if cmv_short and cmv_short in with_curve:
            self._cv_version_combo.setCurrentIndex(with_curve.index(cmv_short))
        elif with_curve:
            self._cv_version_combo.setCurrentIndex(len(with_curve) - 1)
        self._cv_version_combo.blockSignals(False)
        self._refresh_convergence_plot()

    def _refresh_convergence_plot(self) -> None:
        """Repaint the plot for the currently-selected version."""
        import pyqtgraph as pg
        from detector.model_artifact import load_training_curve

        self._cv_plot_container.clear()
        version = self._cv_version_combo.currentData()
        if not version:
            self._cv_status_label.setText(
                "<i>No version selected — pick one from the dropdown. "
                "Versions trained before the curve-capture feature "
                "shipped have no data to plot.</i>"
            )
            return
        artifacts_dir = detector_paths.get_artifacts_dir()
        curve = load_training_curve(artifacts_dir / f"model_{version}")
        if curve is None:
            self._cv_status_label.setText(
                f"<i>No training_curve.json for <b>{version}</b>.</i>"
            )
            return
        metrics = curve.get("metrics", {})
        metric_names = sorted({
            m for split in metrics.values() for m in split.keys()
        })
        if not metric_names:
            self._cv_status_label.setText(
                "Curve file present but no metrics recorded."
            )
            return

        best_iter = int(curve.get("best_iteration") or 0)
        num_round = int(curve.get("num_boost_round") or 0)
        es = int(curve.get("early_stopping_rounds") or 0)
        final_val = metrics.get("val", {}).get(
            metric_names[0], [0.0],
        )[-1]
        self._cv_status_label.setText(
            f"<b>{version}</b> · best iteration: <b>{best_iter}</b> of "
            f"{num_round} max · early stopping = {es} rounds · "
            f"final {metric_names[0]} (val) = <b>{final_val:.4f}</b>"
        )

        # Build one stacked subplot per metric.
        prev_plot = None
        for row_i, mname in enumerate(metric_names):
            plot = self._cv_plot_container.addPlot(row=row_i, col=0)
            plot.setTitle(mname, color="#bbb", size="10pt")
            plot.setLabel("left", mname)
            plot.showGrid(x=True, y=True, alpha=0.15)
            if prev_plot is not None:
                plot.setXLink(prev_plot)
            prev_plot = plot
            for split, color, width in (
                ("train", "#888888", 1.0),
                ("val", "#4ea3ff", 1.5),
            ):
                series = metrics.get(split, {}).get(mname)
                if not series:
                    continue
                x = np.arange(1, len(series) + 1)
                y = np.asarray(series, dtype=np.float64)
                plot.plot(
                    x, y, pen=pg.mkPen(color=color, width=width),
                    name=f"{split} {mname}",
                )
            if best_iter > 0:
                line = pg.InfiniteLine(
                    pos=best_iter, angle=90,
                    pen=pg.mkPen(color="#d62728", width=1,
                                  style=Qt.DashLine),
                )
                plot.addItem(line)
            if row_i == len(metric_names) - 1:
                plot.setLabel("bottom", "Iteration")

    # ==================================================================
    # Preprocessing tab
    # ==================================================================

    def _build_preprocessing_tab(self) -> QWidget:
        """Defaults that pre-fill the Preprocess window's per-batch
        review UI. Editing here writes to `pyqt_settings.json`; the
        next NEW animal review picks them up (existing per-animal
        profiles are untouched — they're authoritative for their
        own animals).
        """
        w = QWidget()
        layout = QVBoxLayout(w)

        layout.addWidget(QLabel(
            "<b>Defaults for new animal reviews.</b>  "
            "These pre-fill the notch-review + channel-assignment "
            "dialogs the first time an animal is processed. They "
            "do <b>not</b> change saved profiles — each animal's "
            "stored profile is authoritative for that animal."
        ))

        settings = ui_settings.load_settings()

        # --- Notch defaults ------------------------------------------------
        notch_group = QGroupBox("Notch filter defaults")
        notch_form = QFormLayout(notch_group)

        self._pp_q_factor = QDoubleSpinBox()
        self._pp_q_factor.setRange(1.0, 200.0)
        self._pp_q_factor.setDecimals(1)
        self._pp_q_factor.setValue(
            float(settings.get("preprocessing_q_factor", 30.0))
        )
        self._pp_q_factor.valueChanged.connect(
            lambda v: ui_settings.update_setting(
                "preprocessing_q_factor", float(v),
            )
        )
        notch_form.addRow("Q factor", self._pp_q_factor)

        self._pp_detrend = QCheckBox(
            "Detrend (subtract per-channel mean before filter + display)"
        )
        self._pp_detrend.setChecked(
            bool(settings.get("preprocessing_detrend", True))
        )
        self._pp_detrend.setToolTip(
            "When on, the per-channel mean is subtracted from both "
            "the raw trace shown alongside the filtered trace AND "
            "the signal fed into the filter for the batch save. "
            "Keeps both traces at the same baseline. Off → both "
            "show their natural DC offset."
        )
        self._pp_detrend.toggled.connect(
            lambda v: ui_settings.update_setting(
                "preprocessing_detrend", bool(v),
            )
        )
        notch_form.addRow(self._pp_detrend)

        # Default notch frequencies — pre-fill the notch-review
        # dialog's Harmonics field on dialog open for new animals.
        # 60/120/180 matches the gi-vagus-viewer pipeline default;
        # European users should switch to 50/100/150.
        default_freqs = ", ".join(
            f"{float(h):g}" for h in
            settings.get("preprocessing_default_freqs_hz",
                          [60.0, 120.0, 180.0])
        )
        self._pp_default_freqs = QLineEdit(default_freqs)
        self._pp_default_freqs.editingFinished.connect(
            self._on_default_freqs_edited
        )
        self._pp_default_freqs.setToolTip(
            "Comma-separated mains harmonics applied by default. The "
            "notch review dialog can be edited per animal; this is "
            "just the starting point for new reviews."
        )
        notch_form.addRow(
            "Default mains harmonics (Hz)", self._pp_default_freqs,
        )
        layout.addWidget(notch_group)

        # --- Animal profiles management -----------------------------------
        profiles_group = QGroupBox("Animal profiles")
        profiles_layout = QVBoxLayout(profiles_group)
        self._pp_profiles_label = QLabel("")
        self._pp_profiles_label.setStyleSheet("font-family: monospace;")
        profiles_layout.addWidget(self._pp_profiles_label)
        btn_row = QHBoxLayout()
        self._pp_refresh_profiles_btn = QPushButton("↻ Refresh list")
        self._pp_refresh_profiles_btn.clicked.connect(
            self._refresh_profiles_summary
        )
        btn_row.addWidget(self._pp_refresh_profiles_btn)
        self._pp_reset_profiles_btn = QPushButton(
            "Reset ALL animal profiles…"
        )
        self._pp_reset_profiles_btn.clicked.connect(
            self._on_reset_all_profiles
        )
        self._pp_reset_profiles_btn.setStyleSheet("color: #c44;")
        btn_row.addWidget(self._pp_reset_profiles_btn)
        btn_row.addStretch(1)
        profiles_layout.addLayout(btn_row)
        layout.addWidget(profiles_group)

        layout.addStretch(1)

        self._refresh_profiles_summary()
        return w

    def _on_default_freqs_edited(self) -> None:
        text = self._pp_default_freqs.text().strip()
        try:
            vals = [
                float(t.strip())
                for t in text.split(",")
                if t.strip()
            ]
        except ValueError:
            QMessageBox.warning(
                self, "Bad default frequencies",
                "Couldn't parse — keep it as a comma-separated list "
                "of numbers, e.g. \"60, 120, 180\".",
            )
            # Revert to the persisted value
            current = ui_settings.load_settings().get(
                "preprocessing_default_freqs_hz", [60.0, 120.0, 180.0]
            )
            self._pp_default_freqs.setText(
                ", ".join(f"{float(h):g}" for h in current)
            )
            return
        if not vals:
            return
        ui_settings.update_setting(
            "preprocessing_default_freqs_hz", vals,
        )

    def _refresh_profiles_summary(self) -> None:
        try:
            ids = detector_profiles.list_profiles()
        except Exception:
            ids = []
        if not ids:
            self._pp_profiles_label.setText(
                "(no saved animal profiles)"
            )
            return
        lines = []
        for animal_id in sorted(ids):
            p = detector_profiles.Profile.load(animal_id)
            if p is None:
                continue
            n_ch = len(p.channel_assignment.get("channels", []))
            harms = p.notch.get("frequencies_filtered", [])
            harm_txt = (
                ", ".join(f"{float(h):g}" for h in harms) or "(none)"
            )
            lines.append(
                f"  • {animal_id}  —  {n_ch} ch · "
                f"notches: {harm_txt}  ({p.updated_at[:10]})"
            )
        self._pp_profiles_label.setText("\n".join(lines))

    def _on_reset_all_profiles(self) -> None:
        try:
            ids = detector_profiles.list_profiles()
        except Exception:
            ids = []
        if not ids:
            QMessageBox.information(
                self, "Nothing to reset",
                "There are no saved animal profiles.",
            )
            return
        resp = QMessageBox.question(
            self, "Reset all animal profiles?",
            f"This permanently deletes {len(ids)} profile(s):\n\n"
            + "\n".join(f"  • {a}" for a in sorted(ids))
            + "\n\nThe next batch for any of these animals will need "
            "a fresh review. Continue?",
        )
        if resp != QMessageBox.Yes:
            return
        deleted = 0
        for animal_id in ids:
            p = detector_profiles.Profile.path_for(animal_id)
            try:
                if p.exists():
                    p.unlink()
                    deleted += 1
            except Exception:
                pass
        QMessageBox.information(
            self, "Profiles cleared",
            f"Deleted {deleted} profile file(s).",
        )
        self._refresh_profiles_summary()

    # ==================================================================
    # Settings tab
    # ==================================================================

    def _build_settings_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        # We persist PyQt settings via ui.data.settings.
        # M4 settings tab focuses on retrain-affecting + auto-retrain
        # prefs. Other PyQt-UI prefs live in main_window's Edit menu.
        settings = ui_settings.load_settings()

        group = QGroupBox("Auto-retrain")
        form = QFormLayout(group)
        self._auto_retrain_cb = QCheckBox(
            "Prompt to retrain when N recordings have been added since current model"
        )
        self._auto_retrain_cb.setChecked(
            bool(settings.get("auto_retrain_enabled", False))
        )
        self._auto_retrain_cb.toggled.connect(
            lambda v: ui_settings.update_setting("auto_retrain_enabled", v)
        )
        form.addRow(self._auto_retrain_cb)
        self._auto_retrain_n = QSpinBox()
        self._auto_retrain_n.setRange(1, 100)
        self._auto_retrain_n.setValue(
            int(settings.get("auto_retrain_threshold_n", 5))
        )
        self._auto_retrain_n.valueChanged.connect(
            lambda v: ui_settings.update_setting("auto_retrain_threshold_n", int(v))
        )
        form.addRow("N (untrained recordings before prompt)", self._auto_retrain_n)
        layout.addWidget(group)

        # Paths section (read-only info)
        paths_group = QGroupBox("Resolved paths")
        paths_layout = QFormLayout(paths_group)
        paths_layout.addRow("Detector home:",
                              QLabel(str(detector_paths.get_home())))
        paths_layout.addRow("Artifacts dir:",
                              QLabel(str(detector_paths.get_artifacts_dir())))
        paths_layout.addRow("Manifest:",
                              QLabel(str(detector_paths.get_manifest_path())))
        layout.addWidget(paths_group)
        layout.addStretch(1)
        return w

    # ==================================================================
    # Auto-retrain trigger (M4.4)
    # ==================================================================

    def _maybe_prompt_auto_retrain(self) -> None:
        """Called after a manifest add. If enabled and the untrained
        count crosses the threshold, prompt the user to retrain."""
        settings = ui_settings.load_settings()
        if not settings.get("auto_retrain_enabled", False):
            return
        try:
            m = Manifest.load(detector_paths.get_manifest_path())
        except Exception:
            return
        cmv = m.current_model_version
        if not cmv:
            return
        n_untrained = len(m.diff_since(cmv)["added"])
        threshold = int(settings.get("auto_retrain_threshold_n", 5))
        if n_untrained < threshold:
            return
        resp = QMessageBox.question(
            self, "Retrain suggested",
            f"You have {n_untrained} recording(s) added since the current "
            f"model ({cmv}). Retrain now?",
        )
        if resp == QMessageBox.Yes:
            self._tabs.setCurrentWidget(self._tab_retrain)

    # ==================================================================
    # House-keeping
    # ==================================================================

    def _refresh_all(self) -> None:
        self._refresh_manifest_tab()
        self._refresh_versions_tab()
        self._refresh_retrain_tab()
        # The convergence-tab dropdown is keyed off list_versions(),
        # which changes after a retrain. Repopulating here picks up
        # the newly trained model automatically.
        try:
            self._refresh_convergence_versions()
        except Exception:
            pass


def _fmt(v, decimals: int) -> str:
    """Format a number safely for table display."""
    if v is None:
        return "—"
    try:
        return f"{float(v):.{decimals}f}"
    except Exception:
        return str(v)


# ----------------------------------------------------------------------
# Sub-dialogs
# ----------------------------------------------------------------------

class AddRecordingDialog(QDialog):
    """Form for adding a new recording to the manifest. Pre-fills
    fields from the picked clean.h5 + bad.h5 paths where possible."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Add recording")
        self.resize(640, 360)

        form = QFormLayout(self)
        self._rid_edit = QLineEdit()
        form.addRow("recording_id", self._rid_edit)
        self._source_edit = QLineEdit()
        source_row = QHBoxLayout()
        source_row.addWidget(self._source_edit)
        btn_source = QPushButton("…")
        btn_source.clicked.connect(self._pick_source)
        source_row.addWidget(btn_source)
        form.addRow("source .mat", source_row)
        self._clean_edit = QLineEdit()
        clean_row = QHBoxLayout()
        clean_row.addWidget(self._clean_edit)
        btn_clean = QPushButton("…")
        btn_clean.clicked.connect(self._pick_clean)
        clean_row.addWidget(btn_clean)
        form.addRow("clean.h5", clean_row)
        self._bad_edit = QLineEdit()
        bad_row = QHBoxLayout()
        bad_row.addWidget(self._bad_edit)
        btn_bad = QPushButton("…")
        btn_bad.clicked.connect(self._pick_bad)
        bad_row.addWidget(btn_bad)
        form.addRow("bad.h5", bad_row)
        # Baseline file path. Optional in the manifest schema -- if
        # left blank, the retrain pipeline falls back to its three-tier
        # resolution (legacy <data_root>/baselines/<rid>_baseline.h5 if
        # present, otherwise next-to-clean.h5). Picking explicitly here
        # lets the user point at a baseline that lives anywhere on disk,
        # so the manifest fully describes where every file is.
        self._baseline_edit = QLineEdit()
        self._baseline_edit.setPlaceholderText(
            "(optional -- defaults to next to clean.h5 if blank)"
        )
        baseline_row = QHBoxLayout()
        baseline_row.addWidget(self._baseline_edit)
        btn_baseline = QPushButton("…")
        btn_baseline.clicked.connect(self._pick_baseline)
        baseline_row.addWidget(btn_baseline)
        form.addRow("baseline.h5", baseline_row)
        self._rec_type_combo = QComboBox()
        self._rec_type_combo.addItems(["stim_rec", "baseline"])
        form.addRow("rec_type", self._rec_type_combo)
        self._fs_spin = QDoubleSpinBox()
        self._fs_spin.setRange(1.0, 1_000_000.0)
        self._fs_spin.setDecimals(4)
        self._fs_spin.setValue(24414.0625)
        form.addRow("fs (Hz)", self._fs_spin)
        self._n_samples_spin = QSpinBox()
        self._n_samples_spin.setRange(1, 2_000_000_000)
        self._n_samples_spin.setValue(1_000_000)
        form.addRow("n_samples", self._n_samples_spin)
        self._n_ch_spin = QSpinBox()
        self._n_ch_spin.setRange(1, 64)
        self._n_ch_spin.setValue(5)
        form.addRow("n_channels", self._n_ch_spin)
        self._notes_edit = QLineEdit()
        form.addRow("notes", self._notes_edit)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self._on_ok)
        button_box.rejected.connect(self.reject)
        form.addRow(button_box)

    def _pick_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick source .mat", "", "MATLAB (*.mat);;All files (*)",
        )
        if path:
            self._source_edit.setText(path)
            self._maybe_autofill_from_clean_h5()

    def _pick_clean(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick clean.h5", "",
            "Splitter clean.h5 (*_clean.h5);;HDF5 (*.h5);;All files (*)",
        )
        if path:
            self._clean_edit.setText(path)
            # Try to auto-fill bad.h5 sibling, fs, n_samples
            self._maybe_autofill_from_clean_h5()

    def _pick_bad(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick bad.h5", "",
            "Splitter bad.h5 (*_bad.h5);;HDF5 (*.h5);;All files (*)",
        )
        if path:
            self._bad_edit.setText(path)

    def _pick_baseline(self) -> None:
        # Default the dialog to the clean.h5's parent folder if set --
        # baselines most commonly live alongside (Option A) or in a
        # sibling `baselines/` folder (legacy convention).
        start_dir = ""
        clean = self._clean_edit.text().strip()
        if clean:
            start_dir = str(Path(clean).parent)
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick baseline.h5", start_dir,
            "Baseline (*_baseline.h5);;HDF5 (*.h5);;All files (*)",
        )
        if path:
            self._baseline_edit.setText(path)

    def _maybe_autofill_from_clean_h5(self) -> None:
        """Inspect the clean.h5 file (if set) to suggest sibling bad.h5,
        the rec_type from filename, fs, n_samples, recording_id."""
        clean = self._clean_edit.text().strip()
        if not clean:
            return
        clean_path = Path(clean)
        if not clean_path.exists():
            return
        stem = clean_path.stem
        if stem.endswith("_clean"):
            base = stem[:-len("_clean")]
            # Auto-fill recording_id if blank
            if not self._rid_edit.text().strip():
                self._rid_edit.setText(base)
            # Auto-fill bad.h5 if blank
            if not self._bad_edit.text().strip():
                bad_sib = clean_path.with_name(f"{base}_bad.h5")
                if bad_sib.exists():
                    self._bad_edit.setText(str(bad_sib))
            # Auto-fill baseline.h5 if blank -- check the two most
            # common locations: next-to-clean.h5 (Option A) and the
            # legacy <repo>/baselines/<rid>_baseline.h5.
            if not self._baseline_edit.text().strip():
                next_to = clean_path.with_name(f"{base}_baseline.h5")
                if next_to.exists():
                    self._baseline_edit.setText(str(next_to))
                else:
                    # Legacy sibling: <parent>/../baselines/<rid>_baseline.h5
                    legacy = (clean_path.parent.parent
                              / "baselines" / f"{base}_baseline.h5")
                    if legacy.exists():
                        self._baseline_edit.setText(str(legacy))
            # Auto-detect rec_type from filename
            if "_stim_rec" in base:
                self._rec_type_combo.setCurrentText("stim_rec")
            elif "_bl_" in base or base.endswith("_bl"):
                self._rec_type_combo.setCurrentText("baseline")
        # Try to read fs + n_samples from the clean.h5 itself.
        try:
            import h5py
            with h5py.File(str(clean_path), "r") as f:
                fs = float(f.attrs.get("fs", 0.0))
                if fs > 0:
                    self._fs_spin.setValue(fs)
                # n_samples is the max end_idx across chunks
                max_end = 0
                first_data_n_ch = None
                for k in f:
                    if k.startswith("chunk_"):
                        max_end = max(max_end, int(f[k]["end_idx"][()]))
                        if first_data_n_ch is None:
                            first_data_n_ch = int(f[k]["data"].shape[1])
                if max_end > 0:
                    self._n_samples_spin.setValue(max_end)
                if first_data_n_ch:
                    self._n_ch_spin.setValue(first_data_n_ch)
        except Exception:
            # Best-effort; user can still fill manually.
            pass

    def _on_ok(self) -> None:
        if not self._rid_edit.text().strip():
            QMessageBox.warning(self, "Missing", "recording_id is required.")
            return
        if not self._source_edit.text().strip():
            QMessageBox.warning(self, "Missing", "source .mat is required.")
            return
        if not self._clean_edit.text().strip():
            QMessageBox.warning(self, "Missing", "clean.h5 is required.")
            return
        if not self._bad_edit.text().strip():
            QMessageBox.warning(self, "Missing", "bad.h5 is required.")
            return
        self.accept()

    def recording_dict(self) -> dict:
        d = {
            "recording_id": self._rid_edit.text().strip(),
            "source_path": str(Path(self._source_edit.text().strip()).resolve()),
            "splitter_clean_path": str(Path(self._clean_edit.text().strip()).resolve()),
            "splitter_bad_path": str(Path(self._bad_edit.text().strip()).resolve()),
            "rec_type": self._rec_type_combo.currentText(),
            "fs": float(self._fs_spin.value()),
            "n_samples": int(self._n_samples_spin.value()),
            "n_channels": int(self._n_ch_spin.value()),
            "label_source": "human",
            "added_by": "pyqt_ui",
            "notes": self._notes_edit.text(),
            "model_version_last_trained_on": None,
        }
        # Optional: only set splitter_baseline_path if user picked one.
        # An empty field means "let the resolution fallback in
        # dataset.py decide" (legacy compat).
        baseline = self._baseline_edit.text().strip()
        if baseline:
            d["splitter_baseline_path"] = str(Path(baseline).resolve())
        return d


class ProvenanceDialog(QDialog):
    """Read-only JSON viewer for a model version's provenance."""

    def __init__(self, version: str, provenance: dict,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(f"Provenance — {version}")
        self.resize(720, 520)
        layout = QVBoxLayout(self)
        view = QPlainTextEdit()
        view.setReadOnly(True)
        view.setStyleSheet(
            "font-family: monospace; font-size: 11px;"
        )
        view.setPlainText(json.dumps(provenance, indent=2, default=str))
        layout.addWidget(view)
        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.rejected.connect(self.reject)
        btn_box.accepted.connect(self.accept)
        layout.addWidget(btn_box)


class RegressionReportDialog(QDialog):
    """Summary of the just-finished retrain's regression check vs the
    previous version. Separates real metric/gate regressions from
    scope changes (recordings added/removed) which are informational
    not regressive."""

    def __init__(self, version: str, report: dict,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(f"Regression report — {version}")
        self.resize(820, 600)
        layout = QVBoxLayout(self)
        header = QLabel(
            f"<b>Model {version}</b> — regression report\n"
            f"vs previous version: {report.get('previous_version', '—')}"
        )
        header.setStyleSheet("padding: 6px;")
        layout.addWidget(header)

        body = self._format_human_readable(report)
        view = QPlainTextEdit()
        view.setReadOnly(True)
        view.setStyleSheet("font-family: monospace; font-size: 11px;")
        view.setPlainText(body)
        layout.addWidget(view, stretch=1)

        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.rejected.connect(self.reject)
        btn_box.accepted.connect(self.accept)
        layout.addWidget(btn_box)

    @staticmethod
    def _format_human_readable(report: dict) -> str:
        """Render report sections in priority order:
          1. Aggregate metric regressions (loro_summary.*)
          2. Per-recording gate flips (recs in both old + new)
          3. Improvements (good news)
          4. Scope changes (informational -- not regressions)
          5. Raw JSON dump for full audit trail
        """
        lines: list[str] = []
        regressions = report.get("regressions", []) or []
        improvements = report.get("improvements", []) or []
        scope = report.get("scope_changes") or {}

        # Split regressions into aggregate vs per-recording so the
        # user can tell at a glance which kind they have.
        agg_reg = [r for r in regressions
                   if not r.get("metric", "").startswith("per_recording.")]
        per_rec_reg = [r for r in regressions
                       if r.get("metric", "").startswith("per_recording.")]
        agg_imp = [r for r in improvements
                   if not r.get("metric", "").startswith("per_recording.")]
        per_rec_imp = [r for r in improvements
                       if r.get("metric", "").startswith("per_recording.")]

        # 1. Aggregate regressions
        if agg_reg:
            lines.append("=" * 60)
            lines.append(f"[REGRESSION] {len(agg_reg)} aggregate metric(s) "
                          "got worse:")
            for r in agg_reg:
                old_v = r.get("old"); new_v = r.get("new")
                rel = r.get("relative_change")
                rel_s = f" ({rel * 100:+.1f}%)" if rel is not None else ""
                lines.append(
                    f"  - {r['metric']}: {old_v} -> {new_v}{rel_s}"
                )
            lines.append("")

        # 2. Per-recording gate regressions (on recordings in BOTH
        # corpora -- these are the ACTUAL gate flips, not scope-change
        # artifacts).
        if per_rec_reg:
            lines.append("=" * 60)
            lines.append(
                f"[REGRESSION] {len(per_rec_reg)} per-recording gate flip(s) "
                "(passed before, fails now):"
            )
            for r in per_rec_reg:
                lines.append(f"  - {r['metric']}")
            lines.append("")

        # 3. Improvements
        if agg_imp or per_rec_imp:
            lines.append("=" * 60)
            lines.append(
                f"[IMPROVEMENT] {len(agg_imp) + len(per_rec_imp)} metric(s) "
                "got better:"
            )
            for r in agg_imp:
                old_v = r.get("old"); new_v = r.get("new")
                rel = r.get("relative_change")
                rel_s = f" ({rel * 100:+.1f}%)" if rel is not None else ""
                lines.append(
                    f"  - {r['metric']}: {old_v} -> {new_v}{rel_s}"
                )
            for r in per_rec_imp:
                lines.append(f"  - {r['metric']}")
            lines.append("")

        # 4. Scope changes -- NOT regressions, just FYI.
        only_old = scope.get("removed_from_manifest") or []
        only_new = scope.get("added_to_manifest") or []
        if only_old or only_new:
            lines.append("=" * 60)
            lines.append("[SCOPE CHANGE] The training corpus differs between "
                         "versions; the recordings below were tested in only "
                         "one of them and don't constitute regressions or "
                         "improvements:")
            if only_old:
                lines.append(f"  Removed from manifest since "
                              f"{report.get('previous_version', 'previous')}:")
                for rid in only_old[:20]:
                    lines.append(f"    - {rid}")
                if len(only_old) > 20:
                    lines.append(f"    ... ({len(only_old) - 20} more)")
            if only_new:
                lines.append("  Added to manifest in this version:")
                for rid in only_new[:20]:
                    lines.append(f"    - {rid}")
                if len(only_new) > 20:
                    lines.append(f"    ... ({len(only_new) - 20} more)")
            lines.append("")

        # 5. Verdict
        lines.append("=" * 60)
        if report.get("regressed"):
            lines.append("Verdict: REGRESSION detected -- model NOT promoted.")
            lines.append("Force-promote via Tools -> Training Management -> "
                          "Retrain -> Force promote (with a reason) if you "
                          "want this version active anyway.")
        else:
            lines.append("Verdict: No regressions -- model auto-promoted to "
                          "current.")
        lines.append("")
        lines.append("=" * 60)
        lines.append("Raw report JSON (for audit / scripting):")
        lines.append("")
        lines.append(json.dumps(report, indent=2, default=str))
        return "\n".join(lines)
