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
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QKeySequence, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDialog,
    QDialogButtonBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QProgressDialog, QPushButton,
    QRadioButton, QSizePolicy, QSpinBox, QSplitter, QTabWidget, QTableWidget,
    QTableWidgetItem, QTextEdit, QToolBar, QVBoxLayout, QWidget,
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
from ui.widgets.per_animal_table import (
    PerAnimalTable, normalize_animal_letter,
)
from ui.dialogs.hyperopt_plots_dialog import HyperoptPlotsDialog
from ui.workers.hyperopt_worker import (
    HyperoptWorker, scan_completed_studies,
)
from ui.workers.per_animal_train_worker import PerAnimalTrainWorker
from ui.workers.retrain_worker import (
    RetrainMonitor, list_recent_jobs,
)


# ----------------------------------------------------------------------
# Shared record resolver
# ----------------------------------------------------------------------

def _strip_known_suffix(stem: str) -> tuple[str, str]:
    """Strip known training-pipeline suffixes from a filename stem.
    Returns (core_stem, which_suffix_was_stripped). Suffix order matters:
    longer matches first so `_blankmotion_stim` isn't mis-stripped to
    just `_blankmotion`."""
    for sfx in ("_blankmotion_recovery", "_blankmotion_stim",
                "_blankmotion", "_clean", "_notched", "_notchblanked"):
        if stem.endswith(sfx):
            return stem[: -len(sfx)], sfx
    return stem, ""


def resolve_record_from_picked_file(
    picked: Path,
    *,
    parent_widget=None,
    auto_migrate_blankmotion: bool = True,
) -> Optional[dict]:
    """Given any of these file types, resolve a complete manifest
    record (source_path, splitter_clean_path, splitter_bad_path,
    splitter_baseline_path, fs, n_samples, n_channels, rec_type,
    animal):

      - `<base>_clean.h5` (splitter clean output -- preferred)
      - `<base>.mat` / `<base>_notched.mat` (source recording)
      - `<base>_blankmotion.mat` (legacy MATLAB labels)

    Behavior:
      1. Determine the core stem (strip known suffixes).
      2. Look for sibling clean.h5 / bad.h5 / baseline.h5.
      3. If all splitter files exist: use them. Read fs/n_samples
         from the clean.h5.
      4. If clean.h5 is missing BUT a `_blankmotion.mat` sibling
         exists and `auto_migrate_blankmotion=True`: prompt the user
         to migrate it inline, then use the freshly-created files.
      5. If no source .mat sibling exists either: return None and
         tell the user (we need the source signal for training).

    Returns a fully-populated dict suitable for
    Manifest.add_recording, or None if resolution failed / user
    cancelled the migration prompt.
    """
    from detector.animal_id import extract_animal_letter
    parent_dir = picked.parent
    stem = picked.stem
    core, _suffix = _strip_known_suffix(stem)
    # When the user picks a path like "<base>.mat" with NO suffix to
    # strip, core == stem. That's fine -- siblings live at <core>_clean.h5 etc.

    # Candidate paths (priority order)
    cand_source = [
        parent_dir / f"{core}_notched.mat",
        parent_dir / f"{core}.mat",
    ]
    cand_clean = parent_dir / f"{core}_clean.h5"
    cand_bad = parent_dir / f"{core}_bad.h5"
    cand_baseline = parent_dir / f"{core}_baseline.h5"
    cand_blankmotion = parent_dir / f"{core}_blankmotion.mat"

    source_path = next((c for c in cand_source if c.exists()), None)

    # Splitter files missing -> offer to migrate from blankmotion.
    if (not cand_clean.exists() or not cand_bad.exists()):
        if (auto_migrate_blankmotion and cand_blankmotion.exists()
                and source_path is not None):
            do_migrate = True
            if parent_widget is not None:
                resp = QMessageBox.question(
                    parent_widget,
                    "Migrate legacy blankmotion?",
                    f"{cand_blankmotion.name} found, but no splitter "
                    f"`_clean.h5` / `_bad.h5` siblings.\n\n"
                    f"Migrate the legacy MATLAB labels now? This will:\n"
                    f"  1. Read the bad intervals from "
                    f"{cand_blankmotion.name}\n"
                    f"  2. Apply them to {source_path.name}\n"
                    f"  3. Write {cand_clean.name} + "
                    f"{cand_bad.name} + {cand_baseline.name}\n\n"
                    f"Takes ~10-30 s per file. Same operation as the "
                    f"Tools -> Migrate blankmotion bulk window, just "
                    f"for this one recording.",
                )
                do_migrate = (resp == QMessageBox.Yes)
            if not do_migrate:
                return None
            # Run the migration inline (single-file path).
            from detector.migrate_blankmotion import migrate_one_blankmotion
            res = migrate_one_blankmotion(
                str(cand_blankmotion), force=False,
            )
            if res["status"] != "ok":
                if parent_widget is not None:
                    QMessageBox.critical(
                        parent_widget, "Migration failed",
                        f"Could not migrate "
                        f"{cand_blankmotion.name}:\n\n"
                        f"{res.get('error', 'unknown error')}",
                    )
                return None
            # Migration succeeded; clean.h5 + bad.h5 + baseline.h5
            # now exist on disk.

    # Final check: do we have what we need?
    if not cand_clean.exists() or not cand_bad.exists():
        if parent_widget is not None:
            QMessageBox.warning(
                parent_widget, "Missing splitter files",
                f"No `{core}_clean.h5` / `{core}_bad.h5` siblings "
                f"found next to {picked.name}.\n\n"
                f"To label this recording, use File -> Open in the "
                f"main window, mark bad regions, then Save. That "
                f"produces the splitter files.",
            )
        return None
    if source_path is None:
        if parent_widget is not None:
            QMessageBox.warning(
                parent_widget, "Missing source recording",
                f"No `{core}.mat` or `{core}_notched.mat` sibling "
                f"found. The source recording is required to "
                f"re-extract features at training time.",
            )
        return None

    # Read fs / n_samples / n_channels from the clean.h5.
    fs = 24414.0625
    n_samples = 0
    n_channels = 5
    try:
        import h5py
        with h5py.File(str(cand_clean), "r") as f:
            fs = float(f.attrs.get("fs", fs))
            max_end = 0
            first_n_ch = None
            for k in f:
                if k.startswith("chunk_"):
                    max_end = max(max_end, int(f[k]["end_idx"][()]))
                    if first_n_ch is None:
                        first_n_ch = int(f[k]["data"].shape[1])
            if max_end > 0:
                n_samples = max_end
            if first_n_ch:
                n_channels = first_n_ch
    except Exception:
        pass

    # rec_type from filename convention
    if "_stim_rec" in core:
        rec_type = "stim_rec"
    elif "_bl_" in core or core.endswith("_bl"):
        rec_type = "baseline"
    else:
        rec_type = "baseline"

    return {
        "recording_id": core,
        "source_path": str(source_path.resolve()),
        "splitter_clean_path": str(cand_clean.resolve()),
        "splitter_bad_path": str(cand_bad.resolve()),
        "splitter_baseline_path": (
            str(cand_baseline.resolve()) if cand_baseline.exists() else None
        ),
        "rec_type": rec_type,
        "fs": float(fs),
        "n_samples": int(n_samples),
        "n_channels": int(n_channels),
        "label_source": "human",
        "added_by": "pyqt_ui",
        "notes": "",
        "model_version_last_trained_on": None,
        "held_out": False,
        "animal": extract_animal_letter(core),
        # Filename the user actually picked. Surfaced in the per-animal
        # table's "Source file" column so Andrea can verify whether a
        # folder-scan grabbed the `_blankmotion.mat` or something else.
        "source_file_basename": picked.name,
        "source_file_path": str(picked.resolve()),
    }


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

        # Per-animal training state. Owned here so the worker outlives
        # any single tab refresh. None when no job is running.
        self._per_animal_thread: Optional[QThread] = None
        self._per_animal_worker: Optional[PerAnimalTrainWorker] = None
        self._per_animal_progress: Optional[QProgressDialog] = None
        # Pre-train migration state. Set up only when the user clicks
        # Train with pending blankmotion entries in the extras list.
        # Reset to None after the migration finishes + control flows
        # back into _on_per_animal_train.
        self._pre_train_thread = None
        self._pre_train_worker = None
        self._pre_train_progress = None
        # Extra files added via the per-animal "Add additional files..."
        # button. These get trained alongside the manifest-driven recs
        # but never enter the combined-training manifest. Each entry is
        # a recording-dict-shaped subset with extra=True. Reset on
        # window open; not persisted.
        self._per_animal_extras: list[dict] = []

        # Hyperopt state. Same shape as per-animal: a QThread + worker
        # that live until the optimization finishes (1-6 hours), with
        # a "running, don't close" banner during the run.
        self._hyperopt_thread: Optional[QThread] = None
        self._hyperopt_worker: Optional[HyperoptWorker] = None
        # Per-animal hyperopt weight overrides. Populated by the
        # "Use all best params for per-animal training" button on the
        # Hyperopt tab. When the user clicks "Train per-animal
        # models", this dict (if non-empty) is passed to the
        # PerAnimalTrainWorker so each animal trains with its OWN
        # hyperopt-found optimum instead of a global scalar.
        # Format: {"F": {"w_neg": 0.21, "fp_weight": 8.0,
        # "fn_weight": 1.2}, "J": {...}, ...}
        # Reset on window open; not persisted.
        self._per_animal_hyperopt_weights: dict = {}
        # Per-scope tracking populated as the worker streams results.
        # Combined: single entry under key "combined". Per-animal:
        # one entry per animal letter as soon as its study spins up.
        self._hyperopt_results: dict[str, dict] = {}

        self._tabs = QTabWidget()
        self._tab_manifest = self._build_manifest_tab()
        self._tab_versions = self._build_versions_tab()
        self._tab_retrain = self._build_retrain_tab()
        self._tab_per_animal = self._build_per_animal_tab()
        self._tab_hyperopt = self._build_hyperopt_tab()
        self._tab_convergence = self._build_convergence_tab()
        self._tab_preprocessing = self._build_preprocessing_tab()
        self._tab_settings = self._build_settings_tab()
        self._tabs.addTab(self._tab_manifest, "Manifest")
        self._tabs.addTab(self._tab_versions, "Versions")
        self._tabs.addTab(self._tab_retrain, "Retrain")
        self._tabs.addTab(self._tab_per_animal, "Per-animal training")
        self._tabs.addTab(self._tab_hyperopt, "Hyperopt")
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
        # "Make current" handles both directions: promoting a newly-
        # trained version that the regression-gate refused to auto-
        # promote, AND rolling back to an older version when a recent
        # one misbehaves. Under the hood it's the same operation
        # (set current_model_version), so one button covers both cases.
        self._btn_rollback = QPushButton("★ Make selected version current…")
        self._btn_rollback.setToolTip(
            "Set current_model_version to whichever version is "
            "selected above. Works in both directions: promoting "
            "a newer version that didn't auto-promote (e.g. blocked "
            "by a scope-change regression) AND rolling back to an "
            "older version when a recent one misbehaves."
        )
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
            QMessageBox.critical(self, "Make current failed",
                                   f"Manifest load: {exc}")
            return
        if m.current_model_version == target_version:
            QMessageBox.information(
                self, "Already current",
                f"{target_version} is already current_model_version.",
            )
            return
        reason, ok = QInputDialog.getText(
            self, f"Confirm: make {target_version} current",
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
                f"Set current_model_version failed: {exc}\n\n"
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
                QMessageBox.critical(
                    self, "Make current failed", str(exc2),
                )
                return
        except Exception as exc:
            QMessageBox.critical(self, "Make current failed", str(exc))
            return
        QMessageBox.information(
            self, "Done",
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
            "becomes a high-weight negative training example (weight "
            "set by 'review FP weight' below) -- the canonical active-"
            "learning loop that turns your reviews into actual model "
            "improvements."
        )
        # Companion spinbox for the FP-correction strength. Default
        # 3.0 (10x the default w_neg=0.3, "strong"). Bump higher to
        # accelerate convergence when iteration is plateauing -- the
        # model gets a much steeper gradient on those corrections.
        self._review_fp_weight_spin = QDoubleSpinBox()
        self._review_fp_weight_spin.setRange(0.5, 20.0)
        self._review_fp_weight_spin.setSingleStep(0.5)
        self._review_fp_weight_spin.setDecimals(1)
        self._review_fp_weight_spin.setValue(3.0)
        self._review_fp_weight_spin.setToolTip(
            "Sample-weight assigned to windows the user marked as "
            "false_positive in 'review feedback dir'. Default 3.0 (10x "
            "the default w_neg=0.3). Bump to 6.0-10.0 to accelerate "
            "convergence when iteration is slow and you have high "
            "recall_real headroom -- pushes harder on 'these are "
            "definitely clean' at the cost of some recall."
        )
        form.addRow("review FP weight", self._review_fp_weight_spin)
        # FN-correction strength (manifest false-negatives). A
        # `true_artifact` verdict means the manifest had this row as
        # label=0 but the human caught it as bad. The retrain flips
        # label 0->1 and bumps weight to fn_weight. Default 3.0
        # mirrors fp_weight; bump higher to push recall up faster.
        self._review_fn_weight_spin = QDoubleSpinBox()
        self._review_fn_weight_spin.setRange(0.5, 20.0)
        self._review_fn_weight_spin.setSingleStep(0.5)
        self._review_fn_weight_spin.setDecimals(1)
        self._review_fn_weight_spin.setValue(3.0)
        self._review_fn_weight_spin.setToolTip(
            "Sample-weight assigned to windows the user marked as "
            "true_artifact in 'review feedback dir' -- manifest "
            "false-negatives the human caught. Those rows are flipped "
            "from label=0 to label=1 AND given this weight, so the "
            "model gets high-confidence positive training examples "
            "from regions the manifest mislabeled. Default 3.0; raise "
            "to push recall up when the model is missing real artifacts; "
            "lower (toward 1.0) to keep the manifest authoritative."
        )
        form.addRow("review FN weight", self._review_fn_weight_spin)

        # ----- Auto-FN (model-driven FN identification) ----------------
        # Run the picked model on the training corpus before training,
        # find label=1 rows where prev_model_proba < threshold (these
        # are misses the previous model makes on its own training data),
        # bump their weight to auto_fn_weight. Direct recall recovery
        # signal -- targets what the current model is actually missing
        # instead of what the manifest mislabeled.
        self._auto_fn_check = QCheckBox(
            "Enable auto-FN: bump rows the previous model misses"
        )
        self._auto_fn_check.setToolTip(
            "Pre-train step. Run the model picked below on the FULL "
            "training corpus, find every label=1 row whose predicted "
            "probability is BELOW threshold (= rows the previous model "
            "is currently missing), and bump those rows' sample_weight "
            "to 'auto-FN weight'. Sidecar JSON auto_fn_corrections.json "
            "in the new artifact dir records exactly which rows were "
            "flagged + their prev-model probabilities."
        )
        self._auto_fn_check.toggled.connect(self._on_auto_fn_toggled)
        form.addRow(self._auto_fn_check)
        # Model picker (populated lazily when the user opens this dialog
        # so freshly-trained models are visible without restart).
        self._auto_fn_model_combo = QComboBox()
        self._auto_fn_model_combo.setToolTip(
            "Which model's misses to identify. Default: the currently "
            "promoted model (★)."
        )
        self._auto_fn_model_combo.setEnabled(False)
        self._populate_auto_fn_model_combo()
        form.addRow("auto-FN model", self._auto_fn_model_combo)
        self._auto_fn_weight_spin = QDoubleSpinBox()
        self._auto_fn_weight_spin.setRange(0.5, 20.0)
        self._auto_fn_weight_spin.setSingleStep(0.5)
        self._auto_fn_weight_spin.setDecimals(1)
        self._auto_fn_weight_spin.setValue(5.0)
        self._auto_fn_weight_spin.setEnabled(False)
        self._auto_fn_weight_spin.setToolTip(
            "Sample-weight applied to rows the auto-FN scan flags as "
            "currently-missed positives. Default 5.0 -- 5x stronger "
            "than the default review FN weight because auto-FN is "
            "targeting actual current misses (high signal) vs review-"
            "judgment manifest corrections (lower-signal label fixes). "
            "Raise to 6-10 for aggressive recall recovery; lower toward "
            "3.0 to be gentler."
        )
        form.addRow("auto-FN weight", self._auto_fn_weight_spin)
        self._auto_fn_max_spin = QSpinBox()
        self._auto_fn_max_spin.setRange(0, 1_000_000)
        self._auto_fn_max_spin.setSingleStep(500)
        self._auto_fn_max_spin.setValue(3000)
        self._auto_fn_max_spin.setEnabled(False)
        self._auto_fn_max_spin.setToolTip(
            "Cap on how many rows to bump. The auto-FN scan can flag "
            "thousands of label=1 rows as currently missed; applying "
            "all at once may dominate the next training pass. The cap "
            "picks the LOWEST-probability rows first (= most "
            "confidently missed). 0 = no cap. Default 3000 is "
            "conservative; raise to 10000+ if the previous model's "
            "recall is very low and you want strong recovery."
        )
        form.addRow("auto-FN max corrections (0 = no cap)",
                     self._auto_fn_max_spin)

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

    # ------------------------------------------------------------------
    # Auto-FN UI helpers
    # ------------------------------------------------------------------

    def _on_auto_fn_toggled(self, checked: bool) -> None:
        """Enable/disable the auto-FN sub-controls in lockstep with
        the master checkbox so the user can't accidentally configure
        a value that won't get used."""
        self._auto_fn_model_combo.setEnabled(checked)
        self._auto_fn_weight_spin.setEnabled(checked)
        self._auto_fn_max_spin.setEnabled(checked)
        if checked and self._auto_fn_model_combo.count() == 0:
            # Re-populate -- maybe a fresh model landed in artifacts/.
            self._populate_auto_fn_model_combo()

    def _populate_auto_fn_model_combo(self) -> None:
        """Fill the picker with every model_v*/ on disk. Highlight the
        currently-promoted one (★) and pre-select it."""
        from detector.predict import list_available_versions
        try:
            from ui.workers.inference_worker import (
                current_promoted_version_short,
            )
            promoted = current_promoted_version_short()
        except Exception:
            promoted = None
        self._auto_fn_model_combo.clear()
        versions = list_available_versions()
        if not versions:
            self._auto_fn_model_combo.addItem("(no models on disk)", None)
            self._auto_fn_model_combo.setEnabled(False)
            return
        for v in versions:
            label = f"{v} ★" if v == promoted else v
            self._auto_fn_model_combo.addItem(label, v)
        # Default to the promoted version when available.
        if promoted:
            for i in range(self._auto_fn_model_combo.count()):
                if self._auto_fn_model_combo.itemData(i) == promoted:
                    self._auto_fn_model_combo.setCurrentIndex(i)
                    break

    def _resolved_auto_fn_model_path(self):
        """Returns the path to the auto-FN model's artifact dir, or
        None when auto-FN is disabled / no models exist."""
        if not self._auto_fn_check.isChecked():
            return None
        version = self._auto_fn_model_combo.currentData()
        if not version:
            return None
        from detector import paths as detector_paths
        artifacts_dir = detector_paths.get_artifacts_dir()
        dir_name = (version if version.startswith("model_")
                    else f"model_{version}")
        return artifacts_dir / dir_name

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
                review_fp_weight=float(self._review_fp_weight_spin.value()),
                review_fn_weight=float(self._review_fn_weight_spin.value()),
                auto_fn_model_path=self._resolved_auto_fn_model_path(),
                auto_fn_weight=float(self._auto_fn_weight_spin.value()),
                auto_fn_max_corrections=(
                    int(self._auto_fn_max_spin.value())
                    if self._auto_fn_max_spin.value() > 0 else None
                ),
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
    # Per-animal training tab
    # ==================================================================

    def _build_per_animal_tab(self) -> QWidget:
        """Per-animal training UI: edit the Animal column inline, see
        eligible/skipped counts live, optionally add 'extra' recordings
        that train alongside the manifest but don't enter the combined
        corpus, kick off a sequential per-animal training round."""
        w = QWidget()
        layout = QVBoxLayout(w)

        # Top summary -- updated live as the user edits the animal col.
        self._pa_summary = QLabel("")
        self._pa_summary.setStyleSheet("font-weight: bold; padding: 4px;")
        self._pa_summary.setWordWrap(True)
        layout.addWidget(self._pa_summary)

        # Per-animal count badges (one chip per letter).
        self._pa_counts = QLabel("")
        self._pa_counts.setStyleSheet("font-family: monospace; padding: 2px;")
        self._pa_counts.setWordWrap(True)
        layout.addWidget(self._pa_counts)

        # Hyperopt-override status: shown when the user has loaded
        # per-animal hyperopt weights via the "Use all best params"
        # button on the Hyperopt tab. Hidden when no override is
        # active so the row doesn't take up space.
        override_row = QHBoxLayout()
        self._pa_hp_override_label = QLabel("")
        self._pa_hp_override_label.setWordWrap(True)
        self._pa_hp_override_label.setStyleSheet(
            "color: #4c72b0; font-style: italic; padding: 2px;"
        )
        self._btn_pa_clear_hp_override = QPushButton(
            "Clear hyperopt override"
        )
        self._btn_pa_clear_hp_override.setToolTip(
            "Discard the per-animal hyperopt weights loaded from "
            "the Hyperopt tab. The next training run will use the "
            "scalar w_neg / FP weight / FN weight from the Retrain "
            "tab instead."
        )
        self._btn_pa_clear_hp_override.clicked.connect(
            self._on_pa_clear_hp_override,
        )
        self._btn_pa_clear_hp_override.setVisible(False)
        override_row.addWidget(self._pa_hp_override_label, stretch=1)
        override_row.addWidget(self._btn_pa_clear_hp_override)
        layout.addLayout(override_row)

        # The grouping table.
        self._pa_table = PerAnimalTable()
        self._pa_table.set_min_recordings_per_animal(3)
        self._pa_table.animal_edited.connect(self._on_per_animal_animal_edited)
        self._pa_table.held_out_changed.connect(
            self._on_per_animal_held_out_edited
        )
        layout.addWidget(self._pa_table, stretch=1)

        # Action row
        action_row = QHBoxLayout()
        self._btn_pa_add_files = QPushButton("➕ Add additional files…")
        self._btn_pa_add_files.setToolTip(
            "Pick individual recording files (clean.h5 / blankmotion.mat "
            "/ source .mat). Each file's siblings are resolved at add "
            "time; if a blankmotion has no splitter siblings, you're "
            "prompted to migrate inline (one at a time)."
        )
        self._btn_pa_add_files.clicked.connect(self._on_per_animal_add_files)
        # NEW: bulk folder scan. Defers migration to train time so the
        # user doesn't have to wait per file. Recursively finds
        # `_blankmotion.mat` files; pre-migrated ones land as full
        # records, un-migrated ones land as pending records that get
        # batch-migrated in parallel right before training starts.
        self._btn_pa_add_folder = QPushButton("📁 Add folder…")
        self._btn_pa_add_folder.setToolTip(
            "Recursively scan a folder for `_blankmotion.mat` files. "
            "Pre-migrated files (with `_clean.h5` siblings) are added "
            "immediately. Un-migrated files become pending entries -- "
            "they batch-migrate in parallel just before the training "
            "run starts, so you don't wait per file at add time."
        )
        self._btn_pa_add_folder.clicked.connect(
            self._on_per_animal_add_folder
        )
        # Remove the rows the user has selected in the table. Extras
        # (added via Add files / Add folder) drop out of the per-animal
        # pool. Manifest rows are NOT removable from this UI -- those
        # are core training corpus and have to be removed via the
        # Manifest tab (which is wired to the audited Manifest.remove
        # path). We warn rather than silently no-op.
        self._btn_pa_remove_selected = QPushButton("➖ Remove selected")
        self._btn_pa_remove_selected.setToolTip(
            "Drop the selected row(s) from the per-animal pool. "
            "Only EXTRA rows (those added here via 'Add files' / "
            "'Add folder') can be removed -- recordings from the "
            "main training manifest stay put; manage those via the "
            "Manifest tab.\n\n"
            "Shift- or Ctrl-click to select multiple rows."
        )
        self._btn_pa_remove_selected.clicked.connect(
            self._on_per_animal_remove_selected
        )
        # Convert every extra to a real manifest entry in one click.
        # Migrates any pending blankmotion entries first, then writes
        # the result to the manifest. After this, the extras flow
        # into hyperopt and per-animal training automatically -- those
        # orchestrators only consume Manifest rows, never the in-memory
        # extras list.
        self._btn_pa_promote = QPushButton("⤴ Promote extras to manifest")
        self._btn_pa_promote.setToolTip(
            "Add every EXTRA row to the training manifest. Pending "
            "blankmotion entries get batch-migrated to splitter "
            "format first. After promotion, the rows appear under "
            "Manifest and become visible to hyperopt + per-animal "
            "training -- both orchestrators only see manifest "
            "recordings, never the in-memory extras list."
        )
        self._btn_pa_promote.clicked.connect(
            self._on_per_animal_promote_extras
        )
        self._btn_pa_promote.setEnabled(False)
        self._btn_pa_clear_extras = QPushButton("Clear extras")
        self._btn_pa_clear_extras.clicked.connect(
            self._on_per_animal_clear_extras
        )
        self._btn_pa_clear_extras.setEnabled(False)
        self._btn_pa_train = QPushButton("▶ Train per-animal models")
        self._btn_pa_train.clicked.connect(self._on_per_animal_train)
        self._btn_pa_train.setEnabled(False)
        self._btn_pa_refresh = QPushButton("↻ Refresh")
        self._btn_pa_refresh.clicked.connect(self._refresh_per_animal_tab)
        action_row.addWidget(self._btn_pa_add_files)
        action_row.addWidget(self._btn_pa_add_folder)
        action_row.addWidget(self._btn_pa_remove_selected)
        action_row.addWidget(self._btn_pa_promote)
        action_row.addWidget(self._btn_pa_clear_extras)
        action_row.addStretch(1)
        action_row.addWidget(self._btn_pa_refresh)
        action_row.addWidget(self._btn_pa_train)
        layout.addLayout(action_row)

        # Results table (populated after a training round completes).
        results_box = QGroupBox("Last training results")
        results_layout = QVBoxLayout(results_box)
        self._pa_results_label = QLabel("(no per-animal training run yet)")
        self._pa_results_label.setStyleSheet("color: #aaa; padding: 2px;")
        results_layout.addWidget(self._pa_results_label)
        self._pa_results_table = QTableWidget(0, 5)
        self._pa_results_table.setHorizontalHeaderLabels([
            "Animal", "Status", "Version", "Elapsed (s)", "Notes",
        ])
        self._pa_results_table.setSelectionBehavior(
            QAbstractItemView.SelectRows
        )
        self._pa_results_table.setSelectionMode(
            QAbstractItemView.SingleSelection
        )
        self._pa_results_table.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )
        self._pa_results_table.setAlternatingRowColors(True)
        hdr = self._pa_results_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.Stretch)
        results_layout.addWidget(self._pa_results_table)
        layout.addWidget(results_box)
        return w

    def _refresh_per_animal_tab(self) -> None:
        """Reload manifest rows + merge in current extras + recompute
        the summary."""
        manifest_path = detector_paths.get_manifest_path()
        if not manifest_path.exists():
            self._pa_summary.setText(
                f"No manifest at {manifest_path}. Run `detector init` "
                "to create one."
            )
            self._pa_counts.setText("")
            self._pa_table.set_recordings([])
            self._btn_pa_train.setEnabled(False)
            return
        try:
            m = Manifest.load(manifest_path)
        except Exception as exc:
            self._pa_summary.setText(f"Failed to load manifest: {exc}")
            self._pa_counts.setText("")
            self._pa_table.set_recordings([])
            self._btn_pa_train.setEnabled(False)
            return
        # Include held-out rows so the user can see them in the table,
        # even though they don't contribute to eligibility counts.
        recs = m.list_recordings(include_held_out=True)
        combined = list(recs) + list(self._per_animal_extras)
        self._pa_table.set_recordings(combined)
        self._btn_pa_clear_extras.setEnabled(
            len(self._per_animal_extras) > 0
        )
        self._btn_pa_promote.setEnabled(
            len(self._per_animal_extras) > 0
        )
        self._refresh_per_animal_hp_override_label()
        self._update_per_animal_summary()

    def _update_per_animal_summary(self) -> None:
        """Repaint the live eligible/skipped/badges header from the
        table's current state."""
        s = self._pa_table.summary()
        n_eligible = s["n_eligible"]
        n_skipped = s["n_skipped"]
        unknown = s["unknown"]
        unknown_txt = (
            f"  ·  {unknown} recording(s) missing an animal letter "
            "(edit the Animal column to set one)" if unknown else ""
        )
        self._pa_summary.setText(
            f"{n_eligible} eligible animal(s) (>= 3 recordings each), "
            f"{n_skipped} skipped (too few recordings){unknown_txt}"
        )
        # Per-animal count chips. Highlight letters that are eligible
        # (n>=3) in green-ish, skipped (n<3) in dim grey.
        chips = []
        for letter in sorted(s["groups"].keys()):
            n = s["groups"][letter]
            colour = "#4caf50" if n >= 3 else "#888"
            chips.append(
                f"<span style='color:{colour}; padding-right:10px;'>"
                f"<b>{letter}</b>:{n}</span>"
            )
        self._pa_counts.setText("  ".join(chips) or
                                  "<i>(no animals identified)</i>")
        # Enable training only if at least one eligible animal exists,
        # and no training is currently running.
        running = (self._per_animal_thread is not None
                   and self._per_animal_thread.isRunning())
        self._btn_pa_train.setEnabled(n_eligible > 0 and not running)

    def _on_per_animal_animal_edited(self, recording_id: str,
                                       new_letter: object) -> None:
        """User edited the Animal column. Save back to the manifest
        (skip if this is an 'extra' row -- those aren't in the
        manifest) and refresh the summary."""
        # Track whether the edit belongs to an extra (not in the
        # manifest) so we update self._per_animal_extras instead.
        for extra in self._per_animal_extras:
            if extra.get("recording_id") == recording_id:
                extra["animal"] = new_letter
                self._update_per_animal_summary()
                return
        # Otherwise it's a manifest row -- persist it.
        manifest_path = detector_paths.get_manifest_path()
        try:
            m = Manifest.load(manifest_path)
            hit = False
            for r in m.recordings:
                if r.get("recording_id") == recording_id:
                    r["animal"] = new_letter
                    hit = True
                    break
            if not hit:
                # Stale row (manifest changed under us); refresh to
                # avoid a silent no-op.
                self._refresh_per_animal_tab()
                return
            m.save(manifest_path)
        except Exception as exc:
            QMessageBox.critical(
                self, "Save failed",
                f"Could not save animal edit to manifest:\n{exc}",
            )
            self._refresh_per_animal_tab()
            return
        self._update_per_animal_summary()
        # Notify other tabs / the main window.
        self.manifest_or_versions_changed.emit()

    def _on_per_animal_held_out_edited(
        self, recording_id: str, new_held: bool,
    ) -> None:
        """User toggled the held_out checkbox in the table. For extras,
        update the in-memory dict; for manifest rows, persist to the
        manifest JSON. Then refresh the tab so the dimming + summary
        counts reflect the change."""
        # Extras path: just update the in-memory dict.
        for extra in self._per_animal_extras:
            if extra.get("recording_id") == recording_id:
                extra["held_out"] = bool(new_held)
                self._refresh_per_animal_tab()
                return
        # Manifest path: load, mutate, save -- same pattern as the
        # animal edit handler above.
        manifest_path = detector_paths.get_manifest_path()
        try:
            m = Manifest.load(manifest_path)
            hit = False
            for r in m.recordings:
                if r.get("recording_id") == recording_id:
                    r["held_out"] = bool(new_held)
                    hit = True
                    break
            if not hit:
                # Stale row -- refresh to resync.
                self._refresh_per_animal_tab()
                return
            m.save(manifest_path)
        except Exception as exc:
            QMessageBox.critical(
                self, "Save failed",
                f"Could not save held_out toggle to manifest:\n{exc}",
            )
            self._refresh_per_animal_tab()
            return
        self._refresh_per_animal_tab()
        # Other tabs (Manifest, Versions) depend on held_out state for
        # their counts/eligibility logic, so notify.
        self.manifest_or_versions_changed.emit()

    def _on_per_animal_add_files(self) -> None:
        """File-picker for additional recordings to fold into the
        per-animal pool. Accepts ANY of:
          - `<base>_clean.h5` (splitter clean output, preferred)
          - `<base>_blankmotion.mat` (legacy MATLAB labels)
          - `<base>.mat` / `<base>_notched.mat` (source recording)

        For each picked file, the shared `resolve_record_from_picked_file`
        helper:
          - Finds the splitter siblings (clean.h5 / bad.h5 / baseline.h5)
          - Reads fs / n_samples / n_channels from the clean.h5
          - Auto-detects rec_type + animal letter from the recording_id
          - For legacy blankmotion files with no splitter siblings,
            offers to migrate inline (uses the same logic as Tools ->
            Migrate legacy blankmotion files).

        The resulting record has every field the per-animal training
        pipeline needs to load the recording during Phase 1. Without
        these, training would crash trying to read `source_path`.
        """
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Pick recordings to add to per-animal training "
            "(clean.h5 / blankmotion.mat / source.mat)",
            "",
            "Recordings + labels (*_clean.h5 *_blankmotion.mat *.mat);;"
            "Splitter clean.h5 (*_clean.h5);;"
            "Legacy blankmotion (*_blankmotion.mat);;"
            "Source MAT (*.mat);;"
            "All files (*)",
        )
        if not paths:
            return

        existing_ids = {
            r.get("recording_id") for r in self._pa_table.recordings()
        }
        added = 0
        skipped_duplicate = 0
        failed: list[str] = []
        for p in paths:
            picked = Path(p)
            record = resolve_record_from_picked_file(
                picked, parent_widget=self,
            )
            if record is None:
                # resolve_record_from_picked_file already showed any
                # user-facing message (missing siblings, migration
                # cancelled, etc). Just note the path for the summary.
                failed.append(picked.name)
                continue
            if record["recording_id"] in existing_ids:
                skipped_duplicate += 1
                continue
            existing_ids.add(record["recording_id"])
            # Mark as `extra` so the table can show a 📎 indicator
            # and the per-animal pipeline knows not to fold it into
            # the main training manifest.
            record["extra"] = True
            self._per_animal_extras.append(record)
            added += 1

        if added == 0 and skipped_duplicate == 0 and failed:
            # All-failure case: no need for the "nothing to add" toast.
            return
        if added == 0 and skipped_duplicate > 0:
            QMessageBox.information(
                self, "Nothing to add",
                f"All {skipped_duplicate} picked file(s) were "
                "already in the table.",
            )
            return
        self._refresh_per_animal_tab()
        # Surface skipped/failed counts so the user knows what
        # happened. Avoid a popup for the all-success path.
        if skipped_duplicate or failed:
            details = []
            if skipped_duplicate:
                details.append(
                    f"{skipped_duplicate} already in the table"
                )
            if failed:
                details.append(
                    f"{len(failed)} couldn't be resolved "
                    f"(missing siblings or migration cancelled)"
                )
            QMessageBox.information(
                self, f"Added {added} recording(s)",
                "Added " + str(added) + " new recording(s) "
                "to the per-animal pool.\n\n"
                "Also: " + "; ".join(details) + ".",
            )

    def _on_per_animal_add_folder(self) -> None:
        """Recursively scan a folder for `_blankmotion.mat` files and
        queue them as extras. Pre-migrated files (with `_clean.h5`
        siblings) land as complete records right away. Un-migrated
        files become PENDING records -- they batch-migrate in
        parallel just before per-animal training starts, so the user
        doesn't sit through one prompt per file at add-time.

        The pending record carries:
          - recording_id, source_path, blankmotion_path (real)
          - splitter_* paths set to where they WILL be after migration
          - pending_migration=True flag
          - placeholder fs/n_samples/n_channels (filled in post-migration)
        """
        from detector.animal_id import extract_animal_letter
        from detector.migrate_blankmotion import (
            find_blankmotion_files, _find_source_for_blankmotion,
        )

        folder = QFileDialog.getExistingDirectory(
            self, "Pick folder to scan for `_blankmotion.mat` files "
                  "(recursive)", "",
        )
        if not folder:
            return

        found = find_blankmotion_files(Path(folder), recursive=True)
        if not found:
            QMessageBox.information(
                self, "No files found",
                f"No `_blankmotion.mat` files under:\n{folder}",
            )
            return

        existing_ids = {
            r.get("recording_id") for r in self._pa_table.recordings()
        }
        added_now = 0       # fully-resolved (already migrated)
        added_pending = 0   # waiting for batch migration at train time
        skipped_duplicate = 0
        skipped_no_source = 0

        for bp in found:
            # Try to resolve with siblings -- if all splitter files are
            # already on disk, take that path (no migration needed).
            record = resolve_record_from_picked_file(
                bp, parent_widget=None,    # silent: no prompts here
                auto_migrate_blankmotion=False,
            )
            if record is not None:
                if record["recording_id"] in existing_ids:
                    skipped_duplicate += 1
                    continue
                existing_ids.add(record["recording_id"])
                record["extra"] = True
                self._per_animal_extras.append(record)
                added_now += 1
                continue
            # Resolver returned None -- splitter siblings missing.
            # Build a pending record if a source exists.
            source = _find_source_for_blankmotion(bp)
            if source is None:
                skipped_no_source += 1
                continue
            # Derive the recording's CORE stem from the SOURCE filename
            # (not the blankmotion's), so it matches the migration's
            # output naming exactly. `_find_source_for_blankmotion`
            # strips period suffixes -- so for both
            # `<base>_blankmotion.mat` and `<base>_recovery_blankmotion.mat`
            # the source is `<base>_notched.mat` and the core is
            # `<base>`. This is what the migration writes outputs
            # against (via `_output_paths_for_source`):
            #     <base>_clean.h5 / <base>_bad.h5 / <base>_baseline.h5
            # Previously we used `bp.stem[:-len("_blankmotion")]` which
            # kept the period suffix and produced expected output
            # paths like `<base>_recovery_clean.h5` that never get
            # written. Every recovery extra would then drop with
            # "no clean.h5 produced" even though migration succeeded.
            source_stem = source.stem
            if source_stem.endswith("_notched"):
                core = source_stem[: -len("_notched")]
            elif source_stem.endswith("_notchblanked"):
                core = source_stem[: -len("_notchblanked")]
            else:
                core = source_stem
            if core in existing_ids:
                # Multiple blankmotion variants for the same recording
                # (full + period, or stim + recovery) collapse to one
                # extra. The migration's conflict-detection picks
                # which variant actually runs; we just need one entry
                # in the per-animal table.
                skipped_duplicate += 1
                continue
            existing_ids.add(core)
            # Expected output paths post-migration -- live next to the
            # source, named by the full-recording core (NOT the period-
            # specific base). Matches `_output_paths_for_source`.
            parent_dir = source.parent
            clean_out = parent_dir / f"{core}_clean.h5"
            bad_out = parent_dir / f"{core}_bad.h5"
            baseline_out = parent_dir / f"{core}_baseline.h5"
            rec_type = ("stim_rec" if "_stim_rec" in core
                        else "baseline" if "_bl_" in core
                        else "baseline")
            self._per_animal_extras.append({
                "recording_id": core,
                "source_path": str(source.resolve()),
                "blankmotion_path": str(bp.resolve()),
                "splitter_clean_path": str(clean_out.resolve()),
                "splitter_bad_path": str(bad_out.resolve()),
                "splitter_baseline_path": str(baseline_out.resolve()),
                "rec_type": rec_type,
                # Placeholders -- re-read from clean.h5 after migration.
                "fs": 24414.0625,
                "n_samples": 0,
                "n_channels": 5,
                "label_source": "human",
                "added_by": "pyqt_ui_folder_add",
                "notes": "pending migration from blankmotion",
                "model_version_last_trained_on": None,
                "held_out": False,
                "animal": extract_animal_letter(core),
                "extra": True,
                "pending_migration": True,
                # Show the blankmotion filename in the Source file
                # column so the user can confirm folder detection
                # actually picked up `_blankmotion.mat` (vs e.g.
                # silently grabbing a `_clean.h5` or `.mat`).
                "source_file_basename": bp.name,
                "source_file_path": str(bp.resolve()),
            })
            added_pending += 1

        if (added_now or added_pending or
                skipped_duplicate or skipped_no_source):
            self._refresh_per_animal_tab()
            # Summary popup so the user knows what happened.
            lines = [f"Scanned folder: {folder}"]
            lines.append(f"  blankmotion files found: {len(found)}")
            if added_now:
                lines.append(
                    f"  added (already migrated): {added_now}"
                )
            if added_pending:
                lines.append(
                    f"  added (pending migration): {added_pending}"
                )
                lines.append(
                    "    -> these batch-migrate in parallel right "
                    "before training starts"
                )
            if skipped_duplicate:
                lines.append(
                    f"  skipped (already in table): {skipped_duplicate}"
                )
            if skipped_no_source:
                lines.append(
                    f"  skipped (no source .mat sibling): "
                    f"{skipped_no_source}"
                )
            QMessageBox.information(
                self, "Folder scan complete", "\n".join(lines),
            )

    def _on_per_animal_remove_selected(self) -> None:
        """Drop the user-selected rows from the per-animal pool.

        Only `extra` rows (those added via Add files / Add folder)
        can be removed. Manifest-driven rows stay -- removing those
        belongs to the Manifest tab where the removal is audited
        through Manifest.remove + history. Surfaces a clear count
        of what got removed vs what was protected."""
        selected_ids = set(self._pa_table.selected_recording_ids())
        if not selected_ids:
            QMessageBox.information(
                self, "Nothing selected",
                "Pick one or more rows in the table first.\n\n"
                "Tip: Shift- or Ctrl-click to select multiple rows.",
            )
            return

        # Partition: which selected ids are extras vs manifest-driven.
        extras_to_drop = []
        manifest_protected = []
        extras_ids = {
            r.get("recording_id") for r in self._per_animal_extras
        }
        for rid in selected_ids:
            if rid in extras_ids:
                extras_to_drop.append(rid)
            else:
                manifest_protected.append(rid)

        if not extras_to_drop:
            # All selected rows are manifest-driven -- nothing to do here.
            QMessageBox.information(
                self, "Nothing removable",
                f"All {len(manifest_protected)} selected row(s) are "
                "in the training manifest (not extras added via this "
                "tab).\n\n"
                "To remove a recording from the training manifest, "
                "switch to the Manifest tab and use Remove recording "
                "there -- that path is audited in the manifest "
                "history.",
            )
            return

        # Confirm
        plural = "s" if len(extras_to_drop) != 1 else ""
        protected_note = (
            f"\n\n{len(manifest_protected)} other selected row(s) are "
            f"in the main manifest and will NOT be removed -- use the "
            f"Manifest tab for those."
            if manifest_protected else ""
        )
        resp = QMessageBox.question(
            self, f"Remove {len(extras_to_drop)} extra recording{plural}?",
            f"Drop {len(extras_to_drop)} extra recording{plural} from "
            f"the per-animal pool:\n\n  " +
            "\n  ".join(extras_to_drop[:10]) +
            (f"\n  ... ({len(extras_to_drop) - 10} more)"
             if len(extras_to_drop) > 10 else "") +
            "\n\nThis only removes them from the per-animal table -- "
            "files on disk and the training manifest are untouched." +
            protected_note,
        )
        if resp != QMessageBox.Yes:
            return

        # Apply
        drop_set = set(extras_to_drop)
        self._per_animal_extras = [
            r for r in self._per_animal_extras
            if r.get("recording_id") not in drop_set
        ]
        self._refresh_per_animal_tab()

    def _on_per_animal_clear_extras(self) -> None:
        if not self._per_animal_extras:
            return
        self._per_animal_extras = []
        self._refresh_per_animal_tab()

    def _refresh_per_animal_hp_override_label(self) -> None:
        """Show the loaded per-animal hyperopt weights in a status
        label above the table so the user can see that the next
        Train click will use overridden weights (not the Retrain
        tab's scalars)."""
        weights = self._per_animal_hyperopt_weights
        if not weights:
            self._pa_hp_override_label.setText("")
            self._pa_hp_override_label.setVisible(False)
            self._btn_pa_clear_hp_override.setVisible(False)
            return
        lines = [
            f"<b>Hyperopt override active</b> "
            f"({len(weights)} animal(s)): next training run will "
            f"use the loaded per-animal weights."
        ]
        for animal in sorted(weights.keys()):
            wts = weights[animal]
            lines.append(
                f"&nbsp;&nbsp;{animal}: "
                f"w_neg={wts.get('w_neg', '—'):.4g}, "
                f"fp={wts.get('fp_weight', '—'):.4g}, "
                f"fn={wts.get('fn_weight', '—'):.4g}"
            )
        self._pa_hp_override_label.setText("<br>".join(lines))
        self._pa_hp_override_label.setVisible(True)
        self._btn_pa_clear_hp_override.setVisible(True)

    def _on_pa_clear_hp_override(self) -> None:
        """Discard the loaded per-animal hyperopt weights."""
        if not self._per_animal_hyperopt_weights:
            return
        self._per_animal_hyperopt_weights = {}
        self._refresh_per_animal_hp_override_label()

    # ------------------------------------------------------------------
    # Promote extras to manifest
    # ------------------------------------------------------------------
    #
    # The per-animal training + hyperopt orchestrators both consume
    # Manifest entries only. "Extras" (added via Add files / Add folder
    # on this tab) are an in-memory preview list, not a real corpus.
    # This action moves them into the manifest so they actually count.
    #
    # Two-stage chain mirroring the pre-train migration:
    #   click Promote -> migrate pending blankmotions -> on finish:
    #   write each (now-resolved) extra to the manifest.

    def _on_per_animal_promote_extras(self) -> None:
        if not self._per_animal_extras:
            QMessageBox.information(
                self, "Nothing to promote",
                "There are no extra recordings to promote. Add files "
                "via 'Add additional files...' or scan a folder first.",
            )
            return
        n_total = len(self._per_animal_extras)
        n_pending = sum(
            1 for x in self._per_animal_extras
            if x.get("pending_migration")
        )
        msg_lines = [
            f"Promote {n_total} extra recording(s) to the training "
            "manifest?",
            "",
            "After promotion they appear as normal manifest rows and "
            "become visible to hyperopt + per-animal training.",
        ]
        if n_pending:
            msg_lines += [
                "",
                f"{n_pending} of them still need to be migrated from "
                "legacy `_blankmotion.mat` to splitter format. That "
                "happens automatically (parallel, ~10-30 s per file) "
                "before the manifest write.",
            ]
        resp = QMessageBox.question(
            self, "Promote extras to manifest?", "\n".join(msg_lines),
        )
        if resp != QMessageBox.Yes:
            return

        pending = [
            x for x in self._per_animal_extras
            if x.get("pending_migration")
        ]
        if pending:
            # Re-use the pre-train migration worker, but redirect the
            # completion callback to our promotion flow instead of
            # restarting training. We flag the intent so the shared
            # `_on_pre_train_migration_done` can route correctly.
            self._post_migration_promote = True
            self._run_pre_train_migration(pending)
            return
        # No migrations needed -- write straight to manifest.
        self._finalize_promote_extras_to_manifest()

    def _finalize_promote_extras_to_manifest(self) -> None:
        """Write every extra in `self._per_animal_extras` to the
        training manifest, then clear the extras list and refresh.

        Skips duplicates (by recording_id) so re-clicking is safe.
        Reports a per-file summary at the end so the user knows what
        landed."""
        if not self._per_animal_extras:
            return
        manifest_path = detector_paths.get_manifest_path()
        try:
            m = Manifest.load(manifest_path)
        except Exception as exc:
            QMessageBox.critical(
                self, "Promote failed",
                f"Could not load manifest:\n{exc}",
            )
            return
        existing_ids = {
            r.get("recording_id") for r in m.list_recordings(
                include_held_out=True,
            )
        }
        added: list[str] = []
        skipped_duplicate: list[str] = []
        failed: list[tuple[str, str]] = []
        # Fields that are per-animal-tab bookkeeping / UI display
        # metadata and aren't part of the manifest schema. Strip
        # before handing to Manifest.add_recording, which rejects
        # unknown keys with "recording has unknown fields: [...]".
        STRIP = {
            "extra",
            "pending_migration",
            # Per-animal table's Source file column metadata --
            # surface-only, no training-pipeline meaning.
            "source_file_basename",
            "source_file_path",
            # Set on folder-added records so the pre-train migration
            # worker knows which blankmotion file to migrate. Once
            # promoted to the manifest the migration is already done
            # and the field is no longer needed.
            "blankmotion_path",
        }
        for extra in list(self._per_animal_extras):
            rid = extra.get("recording_id") or ""
            if not rid:
                failed.append(("(no recording_id)", "missing recording_id"))
                continue
            if rid in existing_ids:
                skipped_duplicate.append(rid)
                continue
            rec = {k: v for k, v in extra.items() if k not in STRIP}
            try:
                m.add_recording(rec, by="pyqt_promote_extras")
                existing_ids.add(rid)
                added.append(rid)
            except ManifestError as exc:
                failed.append((rid, str(exc)))
            except Exception as exc:
                failed.append((rid, repr(exc)))
        # Save once at the end, only if we actually added anything.
        if added:
            try:
                m.save(manifest_path)
            except Exception as exc:
                QMessageBox.critical(
                    self, "Promote failed",
                    f"Wrote {len(added)} record(s) to the manifest "
                    f"object but the save failed:\n{exc}\n\n"
                    "Extras list NOT cleared so you can retry.",
                )
                return
        # Clear the promoted entries from extras (preserve ones that
        # failed so the user can fix + retry).
        added_set = set(added)
        self._per_animal_extras = [
            x for x in self._per_animal_extras
            if x.get("recording_id") not in added_set
        ]
        self._refresh_all()
        self.manifest_or_versions_changed.emit()
        # Summary popup so the user sees what happened.
        lines = [
            f"Promoted {len(added)} recording(s) to the manifest.",
        ]
        if skipped_duplicate:
            lines.append(
                f"  Skipped {len(skipped_duplicate)} already in the "
                "manifest."
            )
        if failed:
            lines.append(
                f"  {len(failed)} failed to add (kept in extras):"
            )
            for rid, err in failed[:5]:
                lines.append(f"    - {rid}: {err}")
            if len(failed) > 5:
                lines.append(f"    ... and {len(failed) - 5} more")
        if added:
            lines.append("")
            lines.append(
                "They are now visible in the Manifest, Hyperopt, "
                "and per-animal training flows."
            )
        QMessageBox.information(
            self, "Promote complete", "\n".join(lines),
        )

    def _on_per_animal_train(self) -> None:
        """Kick off retrain_per_animal in a background thread. Disables
        the train button + opens a progress dialog while it runs."""
        if (self._per_animal_thread is not None
                and self._per_animal_thread.isRunning()):
            QMessageBox.information(
                self, "Already running",
                "Per-animal training is already in progress.",
            )
            return
        s = self._pa_table.summary()
        if s["n_eligible"] == 0:
            QMessageBox.information(
                self, "Nothing to train",
                "No animals have >= 3 recordings. Edit the Animal "
                "column to group recordings, or add more recordings "
                "for the existing animals first.",
            )
            return
        # If there are pending-migration extras, batch-migrate them
        # FIRST before kicking off training. Defers the per-file
        # blankmotion -> splitter step to here so the user doesn't
        # have to wait at add-time.
        pending = [
            x for x in self._per_animal_extras
            if x.get("pending_migration")
        ]
        if pending:
            self._run_pre_train_migration(pending)
            # Migration runs async; when it finishes,
            # `_on_pre_train_migration_done` re-enters this method
            # via `_continue_per_animal_train_after_migration`.
            return
        # If the user added 'extra' files, those aren't in the manifest
        # we're going to pass to retrain_per_animal -- the orchestrator
        # only sees Manifest entries. Warn the user before kicking off.
        if self._per_animal_extras:
            resp = QMessageBox.question(
                self, "Extras not yet wired",
                "You added 'extra' recordings via the Add additional "
                "files... button, but the orchestrator currently only "
                "consumes recordings from the training manifest. The "
                "extras will be IGNORED in this round.\n\n"
                "Continue anyway?",
            )
            if resp != QMessageBox.Yes:
                return
        manifest_path = detector_paths.get_manifest_path()
        artifacts_dir = detector_paths.get_artifacts_dir()
        # Re-use the existing retrain tab's settings for w_neg/seed/
        # rebuild flags so the user doesn't have to set them twice.
        # rebuild_phase2 defaults to True for per-animal because each
        # sub-corpus needs its own fresh Phase 2 generation.
        try:
            self._per_animal_worker = PerAnimalTrainWorker(
                manifest_path,
                artifacts_dir=artifacts_dir,
                only_animals=None,
                min_recordings_per_animal=3,
                # Scalar fallbacks -- used when the per-animal
                # hyperopt override doesn't supply a value for a
                # given animal, OR when no override is loaded.
                w_neg=float(self._w_neg_spin.value()),
                seed=int(self._seed_spin.value()),
                review_fp_weight=float(
                    self._review_fp_weight_spin.value()
                ),
                review_fn_weight=float(
                    self._review_fn_weight_spin.value()
                ),
                rebuild_dataset=True,
                rebuild_phase2=True,
                rebuild_loro=True,
                skip_phase2_check=True,
                skip_review=bool(self._skip_review_cb.isChecked()),
                # Per-animal hyperopt override: when populated via
                # the Hyperopt tab's "Use all best params for per-
                # animal training" button, each animal trains with
                # its OWN tuned (w_neg, fp_weight, fn_weight). When
                # empty (default), every animal uses the scalar
                # fallbacks above. Mixed case is also supported --
                # an animal not in the dict uses the scalar.
                weights_by_animal=(
                    dict(self._per_animal_hyperopt_weights)
                    if self._per_animal_hyperopt_weights else None
                ),
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Could not start per-animal training", str(exc),
            )
            return
        self._per_animal_thread = QThread()
        self._per_animal_worker.moveToThread(self._per_animal_thread)
        self._per_animal_thread.started.connect(
            self._per_animal_worker.run
        )
        self._per_animal_worker.progress.connect(
            self._on_per_animal_progress
        )
        self._per_animal_worker.finished.connect(
            self._on_per_animal_finished
        )
        self._per_animal_worker.error.connect(self._on_per_animal_error)
        self._per_animal_worker.finished.connect(
            self._per_animal_thread.quit
        )
        self._per_animal_worker.error.connect(
            self._per_animal_thread.quit
        )
        self._per_animal_thread.finished.connect(
            self._per_animal_worker.deleteLater
        )
        self._per_animal_thread.finished.connect(
            self._per_animal_thread.deleteLater
        )

        # Range 0..100 because the dialog converts (idx,total) into a
        # percentage of total animals trained.
        self._per_animal_progress = QProgressDialog(
            "Preparing per-animal training…", "Cancel", 0, 100, self,
        )
        self._per_animal_progress.setWindowTitle("Per-animal training")
        self._per_animal_progress.setWindowModality(Qt.WindowModal)
        self._per_animal_progress.setMinimumDuration(0)
        # The orchestrator can't be safely cancelled mid-animal -- the
        # retrain subprocess has its own gates and produces partial
        # artifacts. So Cancel is best-effort: it just hides the dialog
        # and stops listening; the background thread keeps running
        # until the CURRENT animal finishes.
        self._per_animal_progress.canceled.connect(
            lambda: self._per_animal_progress.hide()
        )
        self._per_animal_progress.setValue(0)
        self._btn_pa_train.setEnabled(False)
        self._per_animal_thread.start()

    # ------------------------------------------------------------------
    # Pre-train migration of pending-blankmotion extras
    # ------------------------------------------------------------------
    #
    # When the user adds a folder of `_blankmotion.mat` files via
    # "📁 Add folder...", any file without splitter siblings becomes
    # a "pending" extra. We batch-migrate them in parallel right
    # before training so the user doesn't sit through one prompt per
    # file at add-time. This is a two-stage chain:
    #
    #   click Train -> migration worker runs -> on finish: re-resolve
    #   extras, then start the actual per-animal training worker.

    def _run_pre_train_migration(
        self, pending: list[dict],
    ) -> None:
        """Kick off BlankmotionMigrationWorker on the pending extras'
        blankmotion_path entries. On completion, calls
        _on_pre_train_migration_done which re-resolves the pending
        records and then re-enters _on_per_animal_train."""
        from ui.workers.blankmotion_migration_worker import (
            BlankmotionMigrationWorker,
        )

        blankmotion_paths = [
            Path(p["blankmotion_path"]) for p in pending
            if p.get("blankmotion_path")
        ]
        if not blankmotion_paths:
            # Nothing actually migrate-able; treat as done.
            self._on_pre_train_migration_done({})
            return

        self._pre_train_pending_extras = pending
        self._pre_train_worker = BlankmotionMigrationWorker(
            blankmotion_paths,
            force=False,    # respect skip-existing
            n_workers=None,  # auto
        )
        self._pre_train_thread = QThread()
        self._pre_train_worker.moveToThread(self._pre_train_thread)
        self._pre_train_thread.started.connect(
            self._pre_train_worker.run
        )
        self._pre_train_worker.progress.connect(
            self._on_pre_train_migration_progress
        )
        self._pre_train_worker.finished.connect(
            self._on_pre_train_migration_done
        )
        self._pre_train_worker.error.connect(
            self._on_pre_train_migration_error
        )
        self._pre_train_worker.finished.connect(
            self._pre_train_thread.quit
        )
        self._pre_train_worker.error.connect(
            self._pre_train_thread.quit
        )
        self._pre_train_thread.finished.connect(
            self._pre_train_worker.deleteLater
        )
        self._pre_train_thread.finished.connect(
            self._pre_train_thread.deleteLater
        )
        # Tailor the dialog copy to what the user actually clicked.
        # The same migration worker serves both "click Train" and
        # "click Promote to manifest" -- the next-step messaging
        # differs, so branch on the routing flag.
        promote_mode = getattr(self, "_post_migration_promote", False)
        next_step = ("manifest write" if promote_mode
                     else "training")
        self._pre_train_progress = QProgressDialog(
            f"Migrating {len(blankmotion_paths)} blankmotion file(s) "
            f"to splitter format (this happens once, then "
            f"{next_step} starts)…",
            "Cancel", 0, len(blankmotion_paths), self,
        )
        self._pre_train_progress.setWindowTitle(
            "Promote: migrating blankmotion files" if promote_mode
            else "Pre-train: migrating blankmotion files"
        )
        self._pre_train_progress.setWindowModality(Qt.WindowModal)
        self._pre_train_progress.setMinimumDuration(0)
        self._pre_train_progress.canceled.connect(
            self._on_pre_train_migration_cancel
        )
        self._btn_pa_train.setEnabled(False)
        self._pre_train_thread.start()

    def _on_pre_train_migration_progress(
        self, n_done: int, n_total: int, res: dict,
    ) -> None:
        if self._pre_train_progress is None:
            return
        self._pre_train_progress.setValue(n_done)
        bp_name = Path(res.get("blankmotion_path", "?")).name
        st = res.get("status", "?")
        promote_mode = getattr(self, "_post_migration_promote", False)
        follow_up = (
            "Manifest write will happen automatically when all "
            "migrations finish."
            if promote_mode else
            "Training will start automatically when all migrations "
            "finish."
        )
        self._pre_train_progress.setLabelText(
            f"[{n_done}/{n_total}] {bp_name}  -> {st}\n\n{follow_up}"
        )

    def _on_pre_train_migration_done(self, summary: dict) -> None:
        """Migration finished. Re-resolve each pending extra to fill
        in real fs/n_samples/etc from the freshly-created clean.h5,
        then proceed to either the manifest promotion flow or the
        per-animal training launcher.

        Hardened in two ways after Andrea hit a partial-failure case:
          1. Messaging is promote-aware (not stuck on "Training will
             still start" copy).
          2. If no successfully-migrated extras remain, stop here
             instead of falling through to launch training/promote
             on an empty corpus -- that previously left the user
             clicking through a stale 'Continue anyway?' prompt and
             starting a no-op training run.
          3. Routing to the next step is wrapped so an exception
             surfaces as a dialog instead of silently terminating
             the app.
        """
        promote_mode = bool(getattr(self, "_post_migration_promote", False))
        # Clear the routing flag as soon as we've captured it, so any
        # downstream exception can't cause the flag to bleed into a
        # later re-entry.
        self._post_migration_promote = False

        if self._pre_train_progress is not None:
            self._pre_train_progress.close()
            self._pre_train_progress = None

        # Re-resolve each pending extra. If the migration succeeded,
        # the splitter siblings now exist and the resolver returns a
        # full record. If migration failed for some, drop those.
        n_resolved = 0
        n_dropped = 0
        dropped_examples: list[str] = []
        new_extras: list[dict] = []
        for extra in self._per_animal_extras:
            if not extra.get("pending_migration"):
                new_extras.append(extra)
                continue
            # Re-resolve from the (now-existing) clean.h5 sibling.
            clean_path = extra.get("splitter_clean_path")
            rid = extra.get("recording_id", "?")
            if not clean_path or not Path(clean_path).exists():
                n_dropped += 1
                if len(dropped_examples) < 5:
                    dropped_examples.append(
                        f"{rid}: no clean.h5 produced"
                    )
                continue
            resolved = resolve_record_from_picked_file(
                Path(clean_path),
                parent_widget=None,
                auto_migrate_blankmotion=False,
            )
            if resolved is None:
                n_dropped += 1
                if len(dropped_examples) < 5:
                    dropped_examples.append(
                        f"{rid}: resolver couldn't read splitter siblings"
                    )
                continue
            # Preserve the original `animal` if the user had edited it,
            # otherwise use the auto-detect from the re-resolve.
            if extra.get("animal"):
                resolved["animal"] = extra["animal"]
            resolved["extra"] = True
            # Preserve the original picked-file label (the blankmotion
            # filename the folder-scan found) so the Source file column
            # keeps showing what the user actually added, not the
            # post-migration clean.h5.
            if extra.get("source_file_basename"):
                resolved["source_file_basename"] = (
                    extra["source_file_basename"]
                )
            if extra.get("source_file_path"):
                resolved["source_file_path"] = extra["source_file_path"]
            if extra.get("blankmotion_path"):
                resolved["blankmotion_path"] = extra["blankmotion_path"]
            # Preserve any held_out toggle the user made while the
            # row was pending.
            if extra.get("held_out"):
                resolved["held_out"] = True
            new_extras.append(resolved)
            n_resolved += 1
        self._per_animal_extras = new_extras
        self._refresh_per_animal_tab()

        # Surface failures with actual reasons drawn from the worker's
        # summary, not just the count. The worker returns a per-file
        # results list with status + error fields.
        n_summary_errors = int(summary.get("n_error", 0)) if summary else 0
        n_summary_conflicts = (
            int(summary.get("n_skipped_conflict", 0)) if summary else 0
        )
        n_summary_skipped_existing = (
            int(summary.get("n_skipped_existing", 0)) if summary else 0
        )
        n_summary_ok = int(summary.get("n_ok", 0)) if summary else 0
        per_file_errors: list[tuple[str, str]] = []
        per_file_conflicts: list[tuple[str, str]] = []
        for r in (summary or {}).get("results", []):
            st = r.get("status")
            if st == "error":
                bp = Path(str(r.get("blankmotion_path", "?"))).name
                err = str(r.get("error", "unknown error"))
                per_file_errors.append((bp, err))
            elif st == "skipped_conflict":
                bp = Path(str(r.get("blankmotion_path", "?"))).name
                err = str(r.get("error", "skipped"))
                per_file_conflicts.append((bp, err))
        conflict_warnings: list[str] = list(
            (summary or {}).get("warnings", [])
        )
        # Promote-aware partial-failure dialog. Uses an OK button only --
        # the user has already consented to the operation; this is just
        # an FYI before the next step kicks off.
        if (n_summary_errors > 0 or n_dropped > 0
                or n_summary_conflicts > 0 or conflict_warnings):
            next_step = "Manifest write" if promote_mode else "Training"
            lines = [
                f"Migration summary: "
                f"{n_summary_ok} newly migrated, "
                f"{n_summary_skipped_existing} skipped (outputs "
                f"already on disk), "
                f"{n_summary_conflicts} skipped (file collision), "
                f"{n_summary_errors} errored.",
                "",
                f"Re-resolved {n_resolved} extra(s) post-migration; "
                f"{n_dropped} dropped (couldn't find the expected "
                f"clean.h5).",
            ]
            if conflict_warnings:
                # The collision warnings are the most important thing
                # to surface -- they explain WHY some files were
                # skipped and what the implication is for the outputs.
                lines.append("")
                lines.append("File-collision warnings:")
                for w in conflict_warnings[:5]:
                    lines.append(f"  - {w}")
                if len(conflict_warnings) > 5:
                    lines.append(
                        f"  ... and {len(conflict_warnings) - 5} more"
                    )
            if per_file_conflicts:
                lines.append("")
                lines.append("Sample skipped-by-conflict files:")
                for bp, err in per_file_conflicts[:5]:
                    lines.append(f"  - {bp}: {err}")
                if len(per_file_conflicts) > 5:
                    lines.append(
                        f"  ... and {len(per_file_conflicts) - 5} more"
                    )
            if per_file_errors:
                lines.append("")
                lines.append("Sample migration errors:")
                for bp, err in per_file_errors[:5]:
                    lines.append(f"  - {bp}: {err}")
                if len(per_file_errors) > 5:
                    lines.append(
                        f"  ... and {len(per_file_errors) - 5} more"
                    )
            if dropped_examples:
                lines.append("")
                lines.append("Sample dropped-extras reasons:")
                for d in dropped_examples:
                    lines.append(f"  - {d}")
                if n_dropped > len(dropped_examples):
                    lines.append(
                        f"  ... and {n_dropped - len(dropped_examples)} more"
                    )
            lines.append("")
            if n_resolved > 0:
                lines.append(
                    f"{next_step} will proceed with the "
                    f"{n_resolved} successfully-migrated file(s)."
                )
            else:
                lines.append(
                    f"{next_step} will NOT run -- nothing migrated "
                    "successfully."
                )
            QMessageBox.warning(
                self, "Migration complete with notes", "\n".join(lines),
            )

        # Bail out cleanly if nothing migrated. We don't want to
        # launch promote/train on a corpus that lost everything --
        # that's the silent crash path Andrea hit.
        if n_resolved == 0:
            self._btn_pa_train.setEnabled(True)
            return

        # Route to next step inside a try/except so a crash there
        # becomes a dialog, not a silent process exit.
        try:
            if promote_mode:
                self._finalize_promote_extras_to_manifest()
            else:
                self._on_per_animal_train()
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            QMessageBox.critical(
                self,
                ("Promote failed after migration" if promote_mode
                 else "Training launch failed after migration"),
                f"Migration finished but the next step raised:\n\n"
                f"{type(exc).__name__}: {exc}\n\n"
                f"Traceback:\n{tb[-1500:]}",
            )
            self._btn_pa_train.setEnabled(True)

    def _on_pre_train_migration_error(self, msg: str) -> None:
        if self._pre_train_progress is not None:
            self._pre_train_progress.close()
            self._pre_train_progress = None
        self._btn_pa_train.setEnabled(True)
        # Clear the promote-routing flag so a subsequent Train click
        # doesn't get redirected to manifest promotion.
        was_promote = getattr(self, "_post_migration_promote", False)
        self._post_migration_promote = False
        title = ("Promote migration failed" if was_promote
                 else "Pre-train migration failed")
        tail = ("\n\nNo extras were promoted." if was_promote
                else "\n\nNo training will run.")
        QMessageBox.critical(self, title, f"{msg}{tail}")

    def _on_pre_train_migration_cancel(self) -> None:
        if hasattr(self, "_pre_train_worker") and self._pre_train_worker:
            try:
                self._pre_train_worker.request_cancel()
            except Exception:
                pass
        self._btn_pa_train.setEnabled(True)
        self._post_migration_promote = False

    def _on_per_animal_progress(self, idx: int, total: int,
                                  animal: str, status: str) -> None:
        if self._per_animal_progress is None:
            return
        # The orchestrator calls the callback twice per animal: once
        # before training ("starting") and once after (ok/error/skip).
        # We bucket by completed-animal count for the bar percentage.
        completed = idx if status != "starting" else max(idx, 0)
        if total > 0:
            pct = int(min(100, max(0, 100 * completed / total)))
        else:
            pct = 0
        self._per_animal_progress.setValue(pct)
        label = (f"Animal {animal} -- {status}"
                 if status == "starting"
                 else f"Animal {animal}: {status} "
                      f"({completed}/{total} done)")
        self._per_animal_progress.setLabelText(label)

    def _on_per_animal_finished(self, results: dict) -> None:
        if self._per_animal_progress is not None:
            self._per_animal_progress.setValue(100)
            self._per_animal_progress.close()
            self._per_animal_progress = None
        self._per_animal_worker = None
        self._per_animal_thread = None
        self._render_per_animal_results(results)
        # Per-animal training writes new model_v*/ subdirs under
        # per_animal/<animal>/, so the inference auto-routing layer
        # will pick them up next time. Refresh tabs that depend on
        # version listings (combined-model versions tab is unaffected
        # but the main window's per-animal-aware dropdown is).
        self.manifest_or_versions_changed.emit()
        self._update_per_animal_summary()

    def _on_per_animal_error(self, message: str) -> None:
        if self._per_animal_progress is not None:
            self._per_animal_progress.close()
            self._per_animal_progress = None
        self._per_animal_worker = None
        self._per_animal_thread = None
        QMessageBox.critical(
            self, "Per-animal training failed",
            f"The per-animal orchestrator crashed:\n\n{message}",
        )
        self._update_per_animal_summary()

    def _render_per_animal_results(self, results: dict) -> None:
        """Populate the results table from a retrain_per_animal()
        return dict."""
        ok = sum(1 for r in results.values() if r.get("status") == "ok")
        err = sum(1 for r in results.values() if r.get("status") == "error")
        skip = sum(1 for r in results.values() if r.get("status") == "skipped")
        self._pa_results_label.setText(
            f"<b>{ok}</b> trained · <b>{err}</b> failed · "
            f"<b>{skip}</b> skipped"
        )
        # Sort: ok first (alphabetically), then errors, then skips.
        order = {"ok": 0, "error": 1, "skipped": 2}
        rows = sorted(
            results.items(),
            key=lambda kv: (order.get(kv[1].get("status"), 3), kv[0]),
        )
        self._pa_results_table.setRowCount(len(rows))
        for i, (animal, r) in enumerate(rows):
            status = r.get("status", "?")
            version = r.get("new_version") or r.get("model_dir") or "—"
            elapsed = r.get("elapsed_s")
            elapsed_s = f"{elapsed:.1f}" if isinstance(elapsed, (int, float)) else "—"
            notes = r.get("reason") or r.get("error") or ""
            if status == "ok" and r.get("promoted"):
                notes = "promoted" + (f" · {notes}" if notes else "")
            cells = [animal, status, str(version), elapsed_s, str(notes)]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if status == "error":
                    item.setForeground(QBrush(QColor("#c44")))
                elif status == "skipped":
                    item.setForeground(QBrush(QColor("#888")))
                self._pa_results_table.setItem(i, col, item)

    # ==================================================================
    # Hyperopt tab
    # ==================================================================

    def _build_hyperopt_tab(self) -> QWidget:
        """Hyperparameter optimization for the active-learning weights.

        Wraps `detector.hyperopt.optimize_combined` (one study over the
        full manifest) or `optimize_per_animal` (one study per animal
        letter). Each trial trains one LightGBM booster -- 1-3 min --
        so a 30-trial combined study lands at ~1-2 hours and a per-
        animal sweep across N animals at ~N x 20 x 2 min.

        UI shape: scope radio at the top, a config form, run/cancel
        row, live progress section, then a results table with buttons
        to view plots and apply best params to the next retrain.
        """
        w = QWidget()
        layout = QVBoxLayout(w)

        # ----- Scope picker --------------------------------------------
        scope_box = QGroupBox("Scope")
        scope_row = QHBoxLayout(scope_box)
        self._hp_scope_combined = QRadioButton("Combined model")
        self._hp_scope_combined.setChecked(True)
        self._hp_scope_per_animal = QRadioButton("Per-animal models")
        self._hp_scope_group = QButtonGroup(self)
        self._hp_scope_group.addButton(self._hp_scope_combined)
        self._hp_scope_group.addButton(self._hp_scope_per_animal)
        self._hp_scope_combined.toggled.connect(self._on_hp_scope_toggled)
        scope_row.addWidget(self._hp_scope_combined)
        scope_row.addWidget(self._hp_scope_per_animal)
        scope_row.addStretch(1)
        layout.addWidget(scope_box)

        # ----- Settings form -------------------------------------------
        cfg_box = QGroupBox("Hyperopt settings")
        form = QFormLayout(cfg_box)

        self._hp_n_trials_spin = QSpinBox()
        self._hp_n_trials_spin.setRange(2, 500)
        self._hp_n_trials_spin.setValue(30)
        self._hp_n_trials_spin.setToolTip(
            "Number of Optuna trials. Each trial trains one LightGBM "
            "booster (~1-3 min). 30 is a reasonable default for the "
            "combined model; per-animal studies use 20 by default to "
            "stay within Andrea's overnight window."
        )
        form.addRow("n_trials", self._hp_n_trials_spin)

        self._hp_beta_spin = QDoubleSpinBox()
        self._hp_beta_spin.setRange(0.5, 4.0)
        self._hp_beta_spin.setSingleStep(0.1)
        self._hp_beta_spin.setDecimals(2)
        self._hp_beta_spin.setValue(2.0)
        self._hp_beta_spin.setToolTip(
            "F-beta objective weight. beta=1 is balanced F1; beta=2 "
            "(default) penalizes false negatives 4x more than false "
            "positives -- matches the 'missed artifacts hurt "
            "downstream' bias."
        )
        form.addRow("beta (F-beta)", self._hp_beta_spin)

        self._hp_seed_spin = QSpinBox()
        self._hp_seed_spin.setRange(0, 100_000)
        self._hp_seed_spin.setValue(42)
        form.addRow("seed", self._hp_seed_spin)

        self._hp_review_dir_edit = QLineEdit()
        self._hp_review_dir_edit.setPlaceholderText(
            "(optional) path to a review/ folder -- e.g. "
            "<artifacts>/model_v0.2.0/review"
        )
        self._hp_review_dir_edit.setToolTip(
            "Same active-learning loop as the Retrain tab: if you "
            "have a previous model's review/ folder, point at it so "
            "every trial trains with the FP/FN judgments folded in. "
            "Blank = no review feedback."
        )
        review_row = QHBoxLayout()
        review_row.addWidget(self._hp_review_dir_edit)
        btn_pick_review = QPushButton("...")
        btn_pick_review.clicked.connect(self._on_hp_pick_review_dir)
        review_row.addWidget(btn_pick_review)
        form.addRow("review feedback dir", review_row)

        # Per-animal-only widgets. Gated by `_on_hp_scope_toggled`.
        self._hp_min_recs_spin = QSpinBox()
        self._hp_min_recs_spin.setRange(2, 50)
        self._hp_min_recs_spin.setValue(4)
        self._hp_min_recs_spin.setEnabled(False)
        self._hp_min_recs_spin.setToolTip(
            "Animals with fewer than this many training recordings "
            "are skipped -- not enough rows for a meaningful "
            "train/val split."
        )
        form.addRow("min recordings per animal", self._hp_min_recs_spin)

        # "Only animals" picker -- a grid of checkboxes inside a group
        # box. Populated lazily from the manifest in
        # `_refresh_hyperopt_animal_checks` each time the tab is
        # refreshed or scope flips to per-animal.
        self._hp_only_box = QGroupBox(
            "Only animals (leave all unchecked = all eligible)"
        )
        self._hp_only_grid = QGridLayout(self._hp_only_box)
        self._hp_only_grid.setHorizontalSpacing(12)
        self._hp_animal_checks: dict[str, QCheckBox] = {}
        self._hp_only_hint = QLabel(
            "<i>(per-animal scope only)</i>"
        )
        self._hp_only_hint.setStyleSheet("color: #888;")
        self._hp_only_grid.addWidget(self._hp_only_hint, 0, 0)
        self._hp_only_box.setEnabled(False)
        form.addRow(self._hp_only_box)

        # Per-animal holdout picker -- one dropdown per animal letting
        # the user choose which recording to use as the validation
        # set for hyperopt trial scoring. Defaults to "(alphabetical
        # default)" which lets the orchestrator pick. The default can
        # produce F-β=0 for every trial when the picked recording
        # has no positives in its windowed slice -- this UI is the
        # fix for that.
        self._hp_holdout_box = QGroupBox(
            "Hyperopt validation holdout (per animal)"
        )
        self._hp_holdout_grid = QGridLayout(self._hp_holdout_box)
        self._hp_holdout_grid.setHorizontalSpacing(8)
        self._hp_holdout_grid.setVerticalSpacing(4)
        # Maps animal letter -> QComboBox. Each combo has
        # "(alphabetical default)" + one entry per recording_id of
        # that animal. The currentData() is either None (default) or
        # the picked recording_id string.
        self._hp_holdout_combos: dict[str, "QComboBox"] = {}
        self._hp_holdout_hint = QLabel(
            "<i>(per-animal scope only -- pick the recording to use "
            "as the validation set for each animal's hyperopt trials. "
            "'(alphabetical default)' lets the orchestrator pick, but "
            "that can pick a recording with no positives in the eval "
            "slice, producing F-β=0 for every trial.)</i>"
        )
        self._hp_holdout_hint.setWordWrap(True)
        self._hp_holdout_hint.setStyleSheet("color: #888; padding: 2px;")
        self._hp_holdout_grid.addWidget(self._hp_holdout_hint, 0, 0, 1, 2)
        self._hp_holdout_box.setEnabled(False)
        form.addRow(self._hp_holdout_box)

        # Parameter ranges -- the search space Optuna's TPE sampler
        # explores. Defaults match detector/hyperopt.py's defaults.
        # Widening these is the right move when a previous run's best
        # parameter sat right at the edge of the range (e.g. F's best
        # w_neg=0.207 at the lower bound of [0.2, 1.5] -- the true
        # optimum might be at 0.05). Narrowing is useful for
        # exploitation runs after you've identified a promising
        # neighborhood.
        ranges_box = QGroupBox("Parameter ranges")
        ranges_grid = QGridLayout(ranges_box)
        ranges_grid.setHorizontalSpacing(12)
        ranges_grid.setVerticalSpacing(4)
        ranges_hint = QLabel(
            "<i>Search bounds for each Optuna parameter. Widen a "
            "bound when the previous best landed at it -- TPE can't "
            "sample outside the range. Phase 1 + Phase 2 + baselines "
            "are independent of these; changing ranges only costs "
            "the trial loop (~20-60 min/animal).</i>"
        )
        ranges_hint.setWordWrap(True)
        ranges_hint.setStyleSheet("color: #888; padding: 2px;")
        ranges_grid.addWidget(ranges_hint, 0, 0, 1, 3)
        # Header row.
        ranges_grid.addWidget(QLabel("<b>Parameter</b>"), 1, 0)
        ranges_grid.addWidget(QLabel("<b>Min</b>"), 1, 1)
        ranges_grid.addWidget(QLabel("<b>Max</b>"), 1, 2)

        # w_neg: log-sampled in [0.001, 100.0] gives ample headroom
        # for both extremely-permissive and aggressive-negative-down-
        # weighting configurations. Defaults match optimize_scope.
        ranges_grid.addWidget(QLabel("w_neg"), 2, 0)
        self._hp_w_neg_lo = QDoubleSpinBox()
        self._hp_w_neg_lo.setRange(0.001, 100.0)
        self._hp_w_neg_lo.setDecimals(3)
        self._hp_w_neg_lo.setSingleStep(0.05)
        self._hp_w_neg_lo.setValue(0.2)
        self._hp_w_neg_lo.setToolTip(
            "Lower bound for the negative-class sample weight. "
            "Lower = model is less penalized for missing label=0 "
            "samples. F's best at 0.207 hit this bound -- try "
            "lowering to 0.05 or 0.01 to see if the true optimum "
            "is below 0.2."
        )
        ranges_grid.addWidget(self._hp_w_neg_lo, 2, 1)
        self._hp_w_neg_hi = QDoubleSpinBox()
        self._hp_w_neg_hi.setRange(0.001, 100.0)
        self._hp_w_neg_hi.setDecimals(3)
        self._hp_w_neg_hi.setSingleStep(0.05)
        self._hp_w_neg_hi.setValue(1.5)
        self._hp_w_neg_hi.setToolTip(
            "Upper bound for w_neg. Higher = stronger penalty for "
            "FP."
        )
        ranges_grid.addWidget(self._hp_w_neg_hi, 2, 2)

        # fp_weight: linear in [0.1, 50.0]
        ranges_grid.addWidget(QLabel("fp_weight"), 3, 0)
        self._hp_fp_weight_lo = QDoubleSpinBox()
        self._hp_fp_weight_lo.setRange(0.1, 50.0)
        self._hp_fp_weight_lo.setDecimals(2)
        self._hp_fp_weight_lo.setSingleStep(0.5)
        self._hp_fp_weight_lo.setValue(1.0)
        self._hp_fp_weight_lo.setToolTip(
            "Lower bound for the false-positive weight applied to "
            "review-judged FPs (when a review_dir is given) or to "
            "model-predicted FPs during retrain. 1.0 = standard "
            "weight."
        )
        ranges_grid.addWidget(self._hp_fp_weight_lo, 3, 1)
        self._hp_fp_weight_hi = QDoubleSpinBox()
        self._hp_fp_weight_hi.setRange(0.1, 50.0)
        self._hp_fp_weight_hi.setDecimals(2)
        self._hp_fp_weight_hi.setSingleStep(0.5)
        self._hp_fp_weight_hi.setValue(10.0)
        self._hp_fp_weight_hi.setToolTip(
            "Upper bound for fp_weight. F's optimum at ~8.0 sat "
            "well inside this range -- raise only if you see best "
            "values approaching 10."
        )
        ranges_grid.addWidget(self._hp_fp_weight_hi, 3, 2)

        # fn_weight: linear in [0.1, 50.0]
        ranges_grid.addWidget(QLabel("fn_weight"), 4, 0)
        self._hp_fn_weight_lo = QDoubleSpinBox()
        self._hp_fn_weight_lo.setRange(0.1, 50.0)
        self._hp_fn_weight_lo.setDecimals(2)
        self._hp_fn_weight_lo.setSingleStep(0.5)
        self._hp_fn_weight_lo.setValue(1.0)
        self._hp_fn_weight_lo.setToolTip(
            "Lower bound for false-negative weight (true_artifact "
            "judgments). F's best at 1.23 sat near this bound -- "
            "try 0.5 to see if going lower helps."
        )
        ranges_grid.addWidget(self._hp_fn_weight_lo, 4, 1)
        self._hp_fn_weight_hi = QDoubleSpinBox()
        self._hp_fn_weight_hi.setRange(0.1, 50.0)
        self._hp_fn_weight_hi.setDecimals(2)
        self._hp_fn_weight_hi.setSingleStep(0.5)
        self._hp_fn_weight_hi.setValue(10.0)
        self._hp_fn_weight_hi.setToolTip(
            "Upper bound for fn_weight."
        )
        ranges_grid.addWidget(self._hp_fn_weight_hi, 4, 2)

        # Reset-to-defaults button so the user has an easy way back.
        self._btn_hp_reset_ranges = QPushButton("Reset to defaults")
        self._btn_hp_reset_ranges.setToolTip(
            "Restore the original parameter ranges: w_neg ∈ "
            "[0.2, 1.5], fp_weight ∈ [1.0, 10.0], fn_weight ∈ "
            "[1.0, 10.0]."
        )
        self._btn_hp_reset_ranges.clicked.connect(
            self._on_hp_reset_ranges
        )
        ranges_grid.addWidget(self._btn_hp_reset_ranges, 5, 0, 1, 3)

        form.addRow(ranges_box)

        layout.addWidget(cfg_box)

        # ----- Action row ----------------------------------------------
        action_row = QHBoxLayout()
        self._btn_hp_run = QPushButton("▶ Run hyperopt")
        self._btn_hp_run.clicked.connect(self._on_run_hyperopt)
        action_row.addWidget(self._btn_hp_run)
        self._hp_running_banner = QLabel("")
        self._hp_running_banner.setStyleSheet(
            "color: #c84; font-weight: bold; padding: 4px;"
        )
        action_row.addWidget(self._hp_running_banner)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        # ----- Progress display ----------------------------------------
        prog_box = QGroupBox("Progress")
        prog_layout = QVBoxLayout(prog_box)
        self._hp_phase_label = QLabel("idle")
        self._hp_phase_label.setStyleSheet(
            "font-family: monospace; font-size: 12px;"
        )
        prog_layout.addWidget(self._hp_phase_label)
        self._hp_progress = QProgressBar()
        self._hp_progress.setRange(0, 100)
        prog_layout.addWidget(self._hp_progress)
        self._hp_log = QPlainTextEdit()
        self._hp_log.setReadOnly(True)
        self._hp_log.setMaximumBlockCount(2000)
        self._hp_log.setStyleSheet(
            "font-family: monospace; font-size: 11px; "
            "background: #1a1a1a; color: #ddd;"
        )
        self._hp_log.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding,
        )
        prog_layout.addWidget(self._hp_log, stretch=1)
        layout.addWidget(prog_box, stretch=1)

        # ----- Results table -------------------------------------------
        results_box = QGroupBox("Results (per study)")
        results_layout = QVBoxLayout(results_box)
        self._hp_results_summary = QLabel("(no hyperopt run yet)")
        self._hp_results_summary.setStyleSheet("color: #aaa; padding: 2px;")
        results_layout.addWidget(self._hp_results_summary)
        self._hp_results_table = QTableWidget(0, 6)
        self._hp_results_table.setHorizontalHeaderLabels([
            "Study", "Trials", "w_neg", "fp_weight", "fn_weight",
            "Best objective",
        ])
        self._hp_results_table.setSelectionBehavior(
            QAbstractItemView.SelectRows,
        )
        self._hp_results_table.setSelectionMode(
            QAbstractItemView.SingleSelection,
        )
        self._hp_results_table.setEditTriggers(
            QAbstractItemView.NoEditTriggers,
        )
        self._hp_results_table.setAlternatingRowColors(True)
        rhdr = self._hp_results_table.horizontalHeader()
        rhdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for c in range(1, 6):
            rhdr.setSectionResizeMode(c, QHeaderView.Stretch)
        results_layout.addWidget(self._hp_results_table)

        # Action buttons for the selected row.
        res_actions = QHBoxLayout()
        self._btn_hp_open_plots = QPushButton("Open plots folder")
        self._btn_hp_open_plots.setToolTip(
            "Reveal the selected study's plots/ directory and open "
            "the four optimization PNGs in a tabbed viewer."
        )
        self._btn_hp_open_plots.clicked.connect(self._on_hp_open_plots)
        self._btn_hp_apply = QPushButton(
            "Apply to combined retrain"
        )
        self._btn_hp_apply.setToolTip(
            "Copy the selected row's best w_neg / fp_weight / fn_weight "
            "into the Retrain tab's spinboxes so the next COMBINED "
            "retrain uses the tuned weights. Use the button next to "
            "this one if you want per-animal training with each "
            "animal's own optimum."
        )
        self._btn_hp_apply.clicked.connect(self._on_hp_apply_to_retrain)
        # New: separate button for per-animal training. Reads ALL
        # per-animal studies' best_params.json and stores them in
        # self._per_animal_hyperopt_weights so each animal's
        # subsequent retrain uses its OWN optimum (not a global
        # scalar). Switches to the Per-animal training tab.
        self._btn_hp_apply_per_animal = QPushButton(
            "Use all best params for per-animal training"
        )
        self._btn_hp_apply_per_animal.setToolTip(
            "Load EVERY animal's best params from disk and store "
            "them as overrides for the next per-animal training run. "
            "Each animal gets its OWN w_neg / fp_weight / fn_weight "
            "based on its own hyperopt study, instead of all animals "
            "sharing a single set. Switches to the Per-animal tab."
        )
        self._btn_hp_apply_per_animal.clicked.connect(
            self._on_hp_apply_per_animal,
        )
        self._btn_hp_refresh_results = QPushButton(
            "↻ Reload from disk"
        )
        self._btn_hp_refresh_results.setToolTip(
            "Re-scan <artifacts>/hyperopt_*/ for completed studies and "
            "repopulate the table. Useful when a CLI run finished "
            "while the UI wasn't open."
        )
        self._btn_hp_refresh_results.clicked.connect(
            self._refresh_hyperopt_results_table,
        )
        res_actions.addWidget(self._btn_hp_open_plots)
        res_actions.addWidget(self._btn_hp_apply)
        res_actions.addWidget(self._btn_hp_apply_per_animal)
        res_actions.addStretch(1)
        res_actions.addWidget(self._btn_hp_refresh_results)
        results_layout.addLayout(res_actions)
        layout.addWidget(results_box, stretch=1)

        return w

    # ------------------------------------------------------------------
    # Hyperopt-tab helpers
    # ------------------------------------------------------------------

    def _on_hp_scope_toggled(self, _checked: bool) -> None:
        """Per-animal scope toggles the only-animals / min-recordings
        controls and bumps the default n_trials per the CLI defaults
        (30 combined / 20 per-animal). Only nudges the value if the
        user hasn't changed it."""
        per_animal = self._hp_scope_per_animal.isChecked()
        self._hp_min_recs_spin.setEnabled(per_animal)
        self._hp_only_box.setEnabled(per_animal)
        self._hp_holdout_box.setEnabled(per_animal)
        # Default n_trials adjustment -- only if the field still holds
        # the previous scope's default, so we don't overwrite a value
        # the user already tuned.
        cur = self._hp_n_trials_spin.value()
        if per_animal and cur == 30:
            self._hp_n_trials_spin.setValue(20)
        elif not per_animal and cur == 20:
            self._hp_n_trials_spin.setValue(30)
        if per_animal:
            self._refresh_hyperopt_animal_checks()

    def _refresh_hyperopt_animal_checks(self) -> None:
        """Repopulate the per-animal checkbox grid from the current
        manifest. Called when the tab is refreshed and when the user
        toggles to the per-animal scope."""
        # Wipe existing checkboxes (keep the hint label managed
        # separately).
        for cb in self._hp_animal_checks.values():
            self._hp_only_grid.removeWidget(cb)
            cb.deleteLater()
        self._hp_animal_checks.clear()
        # Remove the placeholder hint if present.
        if self._hp_only_hint is not None:
            self._hp_only_grid.removeWidget(self._hp_only_hint)
            self._hp_only_hint.setParent(None)
            self._hp_only_hint = None

        try:
            from detector.animal_id import group_recordings_by_animal
            manifest_path = detector_paths.get_manifest_path()
            if not manifest_path.exists():
                self._hp_only_hint = QLabel(
                    "<i>(no manifest yet -- create one to enable "
                    "per-animal scope)</i>"
                )
                self._hp_only_hint.setStyleSheet("color: #888;")
                self._hp_only_grid.addWidget(self._hp_only_hint, 0, 0)
                return
            m = Manifest.load(manifest_path)
            groups = group_recordings_by_animal(m.list_recordings())
        except Exception as exc:
            self._hp_only_hint = QLabel(f"<i>(error: {exc})</i>")
            self._hp_only_hint.setStyleSheet("color: #c44;")
            self._hp_only_grid.addWidget(self._hp_only_hint, 0, 0)
            return

        # Drop the None bucket (animals without an auto-detected letter).
        letters = sorted(a for a in groups.keys() if a is not None)
        if not letters:
            self._hp_only_hint = QLabel(
                "<i>(no animal letters detected -- edit the Animal "
                "column on the Per-animal tab to assign letters)</i>"
            )
            self._hp_only_hint.setStyleSheet("color: #888;")
            self._hp_only_grid.addWidget(self._hp_only_hint, 0, 0)
            return
        # Layout: max 6 checkboxes per row, label includes the count.
        cols = 6
        for i, letter in enumerate(letters):
            n = len(groups[letter])
            cb = QCheckBox(f"{letter} ({n})")
            cb.setToolTip(
                f"Include animal {letter} (n={n} recordings) in the "
                "per-animal hyperopt run."
            )
            self._hp_animal_checks[letter] = cb
            self._hp_only_grid.addWidget(cb, i // cols, i % cols)
        # Repopulate the per-animal HOLDOUT combos alongside the
        # checkbox grid. One row per animal: "F:  [dropdown]"
        self._refresh_hyperopt_holdout_combos(groups, letters)

    def _refresh_hyperopt_holdout_combos(
        self, groups: dict, letters: list[str],
    ) -> None:
        """Build / repopulate the per-animal validation-holdout
        combos. Each animal gets a QComboBox listing
        '(alphabetical default)' + every recording_id in that
        animal's set. currentData() == None means use the
        orchestrator's default; otherwise it's the recording_id
        string to use as the holdout."""
        # Wipe existing combo widgets (and their labels).
        for w in list(self._hp_holdout_combos.values()):
            self._hp_holdout_grid.removeWidget(w)
            w.deleteLater()
        # Wipe the row-label widgets too. We tagged them by storing
        # them under "label::<letter>" so they're easy to find.
        for letter in list(self._hp_holdout_combos.keys()):
            for child in self._hp_holdout_box.findChildren(QLabel):
                if child.objectName() == f"hp_holdout_label_{letter}":
                    self._hp_holdout_grid.removeWidget(child)
                    child.deleteLater()
        self._hp_holdout_combos.clear()
        # Re-add the placeholder hint at row 0 spanning both cols.
        # (Already added at build time -- recreate it if it's gone.)
        if self._hp_holdout_hint is None:
            self._hp_holdout_hint = QLabel(
                "<i>(per-animal scope only -- pick the recording to "
                "use as the validation set for each animal's "
                "hyperopt trials.)</i>"
            )
            self._hp_holdout_hint.setWordWrap(True)
            self._hp_holdout_hint.setStyleSheet(
                "color: #888; padding: 2px;"
            )
            self._hp_holdout_grid.addWidget(
                self._hp_holdout_hint, 0, 0, 1, 2,
            )
        # Add one row per animal: "F: [combo]".
        for i, letter in enumerate(letters, start=1):
            recs = sorted(
                r["recording_id"] for r in groups.get(letter, [])
            )
            row_label = QLabel(f"<b>{letter}</b>:")
            row_label.setObjectName(f"hp_holdout_label_{letter}")
            row_label.setMinimumWidth(40)
            self._hp_holdout_grid.addWidget(row_label, i, 0)
            combo = QComboBox()
            combo.setToolTip(
                f"Validation holdout for animal {letter}'s hyperopt "
                "trials. '(alphabetical default)' lets the "
                "orchestrator pick the last-sorted recording, which "
                "can yield F-β=0 trials if the picked recording has "
                "no positives in its eval slice. Pick a specific "
                "recording when you know the default is failing."
            )
            combo.addItem("(alphabetical default)", None)
            for rid in recs:
                combo.addItem(rid, rid)
            self._hp_holdout_combos[letter] = combo
            self._hp_holdout_grid.addWidget(combo, i, 1)

    def _on_hp_reset_ranges(self) -> None:
        """Restore the default parameter ranges. The defaults match
        detector/hyperopt.py's `optimize_scope` defaults so a reset
        produces the same search space as a fresh install."""
        self._hp_w_neg_lo.setValue(0.2)
        self._hp_w_neg_hi.setValue(1.5)
        self._hp_fp_weight_lo.setValue(1.0)
        self._hp_fp_weight_hi.setValue(10.0)
        self._hp_fn_weight_lo.setValue(1.0)
        self._hp_fn_weight_hi.setValue(10.0)

    def _on_hp_pick_review_dir(self) -> None:
        start_dir = (
            self._hp_review_dir_edit.text().strip() or str(Path.home())
        )
        path = QFileDialog.getExistingDirectory(
            self, "Pick previous model's review/ folder", start_dir,
        )
        if path:
            self._hp_review_dir_edit.setText(path)

    def _on_run_hyperopt(self) -> None:
        """Kick off the worker. Disables the run button + shows a
        running banner; the user can keep using other tabs."""
        if (self._hyperopt_thread is not None
                and self._hyperopt_thread.isRunning()):
            QMessageBox.information(
                self, "Already running",
                "A hyperopt run is already in progress.",
            )
            return
        manifest_path = detector_paths.get_manifest_path()
        if not manifest_path.exists():
            QMessageBox.warning(
                self, "No manifest",
                f"No training manifest at {manifest_path}. Create one "
                "via `detector init` or the Manifest tab.",
            )
            return
        artifacts_dir = detector_paths.get_artifacts_dir()
        scope = ("per_animal" if self._hp_scope_per_animal.isChecked()
                 else "combined")
        review_text = self._hp_review_dir_edit.text().strip()
        review_dir = Path(review_text) if review_text else None
        if review_dir is not None and not review_dir.exists():
            resp = QMessageBox.question(
                self, "Review dir not found",
                f"The review feedback dir {review_dir} doesn't exist. "
                "Run anyway without review feedback?",
            )
            if resp != QMessageBox.Yes:
                return
            review_dir = None

        # Validate parameter ranges -- each low must be < its high.
        # Optuna's TPESampler would raise mid-trial otherwise; catch
        # it early with a useful message.
        ranges_to_check = [
            ("w_neg", self._hp_w_neg_lo.value(), self._hp_w_neg_hi.value()),
            ("fp_weight",
             self._hp_fp_weight_lo.value(), self._hp_fp_weight_hi.value()),
            ("fn_weight",
             self._hp_fn_weight_lo.value(), self._hp_fn_weight_hi.value()),
        ]
        for name, lo, hi in ranges_to_check:
            if lo >= hi:
                QMessageBox.critical(
                    self, "Invalid parameter range",
                    f"{name}: min ({lo}) must be strictly less than "
                    f"max ({hi}). Fix the range in the Parameter "
                    "ranges group before running.",
                )
                return
        w_neg_range = (float(self._hp_w_neg_lo.value()),
                       float(self._hp_w_neg_hi.value()))
        fp_weight_range = (float(self._hp_fp_weight_lo.value()),
                           float(self._hp_fp_weight_hi.value()))
        fn_weight_range = (float(self._hp_fn_weight_lo.value()),
                           float(self._hp_fn_weight_hi.value()))

        only_animals: Optional[list[str]] = None
        holdout_rids_by_animal: dict[str, list[str]] = {}
        if scope == "per_animal":
            picked = [
                letter for letter, cb in self._hp_animal_checks.items()
                if cb.isChecked()
            ]
            only_animals = picked if picked else None
            # Collect per-animal holdout choices. If the combo's
            # current data is None ("(alphabetical default)"), we
            # omit that animal -- the orchestrator falls back to its
            # default selection. Otherwise we pass the picked
            # recording_id.
            for letter, combo in self._hp_holdout_combos.items():
                picked_rid = combo.currentData()
                if picked_rid:
                    holdout_rids_by_animal[letter] = [str(picked_rid)]

        # Warn about the time cost so the user doesn't kick this off
        # accidentally.
        n_trials = int(self._hp_n_trials_spin.value())
        rough_hours = (n_trials * 2) / 60.0
        scope_label = ("per-animal" if scope == "per_animal" else "combined")
        resp = QMessageBox.question(
            self, "Run hyperopt?",
            f"This will run a {scope_label} hyperopt with "
            f"{n_trials} trials per study.\n\n"
            f"Each trial trains one LightGBM booster (~1-3 min), so a "
            f"single study takes roughly {rough_hours:.1f} hours. "
            f"Per-animal scope multiplies that by the number of "
            f"eligible animals.\n\n"
            f"The UI stays responsive -- you can keep using other "
            f"tabs. Don't close the application until the run "
            f"finishes.\n\nContinue?",
        )
        if resp != QMessageBox.Yes:
            return

        try:
            self._hyperopt_worker = HyperoptWorker(
                scope=scope,
                manifest_path=manifest_path,
                artifacts_dir=artifacts_dir,
                n_trials=n_trials,
                beta=float(self._hp_beta_spin.value()),
                seed=int(self._hp_seed_spin.value()),
                review_dir=review_dir,
                only_animals=only_animals,
                min_recordings_per_animal=int(self._hp_min_recs_spin.value()),
                holdout_rids_by_animal=(
                    holdout_rids_by_animal or None
                ),
                w_neg_range=w_neg_range,
                fp_weight_range=fp_weight_range,
                fn_weight_range=fn_weight_range,
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Could not start hyperopt", str(exc),
            )
            return

        self._hyperopt_thread = QThread()
        self._hyperopt_worker.moveToThread(self._hyperopt_thread)
        self._hyperopt_thread.started.connect(self._hyperopt_worker.run)
        self._hyperopt_worker.progress.connect(self._on_hp_progress)
        self._hyperopt_worker.animal_started.connect(self._on_hp_animal_started)
        self._hyperopt_worker.animal_finished.connect(
            self._on_hp_animal_finished
        )
        self._hyperopt_worker.finished.connect(self._on_hp_finished)
        self._hyperopt_worker.error.connect(self._on_hp_error)
        self._hyperopt_worker.finished.connect(self._hyperopt_thread.quit)
        self._hyperopt_worker.error.connect(self._hyperopt_thread.quit)
        self._hyperopt_thread.finished.connect(
            self._hyperopt_worker.deleteLater
        )
        self._hyperopt_thread.finished.connect(
            self._hyperopt_thread.deleteLater
        )

        self._btn_hp_run.setEnabled(False)
        self._hp_running_banner.setText(
            "● Running -- do not close the application"
        )
        self._hp_phase_label.setText(
            f"starting {scope_label} hyperopt ({n_trials} trials)…"
        )
        self._hp_progress.setRange(0, 100)
        self._hp_progress.setValue(0)
        self._hp_log.clear()
        self._hp_log.appendPlainText(
            f"[hyperopt] starting {scope_label} run, n_trials={n_trials}, "
            f"beta={self._hp_beta_spin.value()}, "
            f"seed={self._hp_seed_spin.value()}"
        )
        if review_dir is not None:
            self._hp_log.appendPlainText(
                f"[hyperopt] using review feedback at {review_dir}"
            )
        # Reset per-scope tracking.
        self._hyperopt_results = {}
        self._hyperopt_thread.start()

    def _on_hp_progress(
        self, scope: str, trial_num: int, total: int, objective: float,
    ) -> None:
        """One trial just landed. Update the progress bar (per-current-
        study) + append a log line."""
        if total > 0:
            pct = int(min(100, max(0, 100 * trial_num / total)))
        else:
            pct = 0
        self._hp_progress.setValue(pct)
        self._hp_phase_label.setText(
            f"{scope} -- trial {trial_num}/{total}"
        )
        obj_txt = (f"{objective:.4f}"
                   if objective == objective         # NaN check
                   else "n/a")
        self._hp_log.appendPlainText(
            f"[{scope}] trial {trial_num}/{total} -- objective={obj_txt}"
        )

    def _on_hp_animal_started(self, animal: str) -> None:
        self._hp_log.appendPlainText(
            f"[hyperopt] -- animal {animal} started --"
        )
        self._hp_phase_label.setText(f"animal {animal} -- starting…")
        self._hp_progress.setValue(0)

    def _on_hp_animal_finished(
        self, animal: str, best_params: dict,
    ) -> None:
        bp = best_params.get("best_params", {}) if best_params else {}
        bo = best_params.get("best_objective") if best_params else None
        # Cache the result so the results table picks it up even before
        # `finished` fires (e.g. mid-run when N-1 animals are done).
        self._hyperopt_results[animal] = {
            "best_params": bp,
            "best_objective": bo,
        }
        self._hp_log.appendPlainText(
            f"[hyperopt] -- animal {animal} done: "
            f"best_objective={bo}, params={bp} --"
        )
        # Live-update the results table.
        self._refresh_hyperopt_results_table()

    def _drain_hyperopt_thread(self) -> None:
        """Explicitly quit + wait on the hyperopt QThread before we
        drop our Python references. Without this, Qt's runtime races
        between the worker.finished signal triggering thread.quit
        (async) and Python's GC reclaiming the QThread object --
        when the C++ destructor runs while the OS thread hasn't
        terminated yet, Qt prints "QThread: Destroyed while thread
        '' is still running" and (on Windows) sometimes kills the
        process.

        Calling quit() + wait() here from the main thread ensures
        the worker thread's event loop has exited and the OS thread
        has joined before any deleteLater can fire. Safe to call
        even if the thread has already finished (quit + wait become
        no-ops).
        """
        thread = self._hyperopt_thread
        if thread is None:
            return
        try:
            thread.quit()
            # 10s should be plenty -- worker.run() already returned
            # by the time we're here; we're just waiting for the
            # event loop's exec_() to unwind. Anything longer than a
            # few ms means something is genuinely stuck.
            if not thread.wait(10_000):
                self._hp_log.appendPlainText(
                    "[hyperopt] WARNING: QThread didn't exit within "
                    "10s of quit() -- continuing anyway."
                )
        except RuntimeError:
            # Thread already destroyed (e.g. deleteLater fired
            # before this slot). Harmless.
            pass

    def _on_hp_finished(self, results: dict) -> None:
        # Drain the QThread BEFORE dropping refs. See
        # `_drain_hyperopt_thread` for the rationale -- this is the
        # fix for the "auto-close at end of every hyperopt run" bug.
        self._drain_hyperopt_thread()
        self._hyperopt_worker = None
        self._hyperopt_thread = None
        self._btn_hp_run.setEnabled(True)
        self._hp_running_banner.setText("")
        self._hp_phase_label.setText("done")
        self._hp_progress.setValue(100)
        # The full results envelope shape differs between scopes:
        #   combined  -> single scope_result_dict
        #   per_animal-> {animal: scope_result_dict, ...}
        # The results table reads best_params.json from disk via
        # scan_completed_studies(), so we just trigger a reload.
        self._hp_log.appendPlainText(
            "[hyperopt] run finished -- reloading results table"
        )
        self._refresh_hyperopt_results_table()

    def _on_hp_error(self, message: str) -> None:
        # Same drain pattern as _on_hp_finished -- the error path
        # has the same QThread race.
        self._drain_hyperopt_thread()
        self._hyperopt_worker = None
        self._hyperopt_thread = None
        self._btn_hp_run.setEnabled(True)
        self._hp_running_banner.setText("")
        self._hp_phase_label.setText("FAILED")
        self._hp_log.appendPlainText(
            f"[hyperopt] FAILED: {message}"
        )
        QMessageBox.critical(
            self, "Hyperopt failed",
            f"The hyperopt orchestrator crashed:\n\n{message}",
        )

    def _refresh_hyperopt_results_table(self) -> None:
        """Re-scan the artifacts dir for completed studies and rebuild
        the results table. Called on tab refresh, after each animal
        finishes, and on demand via the Reload button."""
        try:
            artifacts_dir = detector_paths.get_artifacts_dir()
            studies = scan_completed_studies(artifacts_dir)
        except Exception as exc:
            self._hp_results_summary.setText(
                f"<span style='color:#c44'>Error scanning artifacts: "
                f"{exc}</span>"
            )
            return

        # Build rows: ("combined", payload) first if present, then
        # one row per animal in alphabetical order.
        rows: list[tuple[str, dict]] = []
        if studies.get("combined"):
            rows.append(("combined", studies["combined"]))
        for animal in sorted(studies.get("per_animal", {}).keys()):
            rows.append((animal, studies["per_animal"][animal]))

        if not rows:
            self._hp_results_summary.setText(
                "(no hyperopt run yet -- launch one above, or run "
                "`detector hyperopt` from the CLI)"
            )
            self._hp_results_table.setRowCount(0)
            return

        self._hp_results_summary.setText(
            f"<b>{len(rows)}</b> completed study(ies)"
        )
        self._hp_results_table.setRowCount(len(rows))
        for i, (label, payload) in enumerate(rows):
            bp = payload.get("best_params", {}) or {}
            bo = payload.get("best_objective")
            # holdout_rids isn't shown but it's load-bearing for the
            # apply-button to remember which study a row came from.
            cells = [
                label,
                # We don't have n_trials per row on disk in
                # best_params.json -- read from trial_log.json if
                # cheaply available, else "—".
                _hp_n_trials_for(label),
                _fmt(bp.get("w_neg"), 4),
                _fmt(bp.get("fp_weight"), 3),
                _fmt(bp.get("fn_weight"), 3),
                _fmt(bo, 4),
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(str(text))
                self._hp_results_table.setItem(i, col, item)
        # Auto-select the first row so the apply/plots buttons act on
        # something meaningful by default.
        self._hp_results_table.selectRow(0)

    def _selected_hyperopt_study(self) -> Optional[tuple[str, Path]]:
        """Return (study_label, study_workdir) for the currently-
        selected row, or None if nothing is selected."""
        row = self._hp_results_table.currentRow()
        if row < 0:
            return None
        label_item = self._hp_results_table.item(row, 0)
        if label_item is None:
            return None
        label = label_item.text()
        artifacts_dir = detector_paths.get_artifacts_dir()
        if label == "combined":
            return label, artifacts_dir / "hyperopt_combined"
        return label, artifacts_dir / "hyperopt_per_animal" / label

    def _on_hp_open_plots(self) -> None:
        """Open the plots dialog. For per-animal hyperopt, includes
        a dropdown at the top to switch between all animals' studies
        without closing the dialog. The currently-selected row in
        the results table determines which animal opens first.
        """
        sel = self._selected_hyperopt_study()
        if sel is None:
            QMessageBox.information(
                self, "Pick a study",
                "Select a row in the results table first.",
            )
            return
        sel_label, sel_workdir = sel
        artifacts_dir = detector_paths.get_artifacts_dir()

        # Build the full list of studies available on disk so the
        # dialog's switcher covers every animal -- not just the row
        # the user clicked. Order: combined first, then per-animal
        # alphabetically, matching the results-table order.
        studies: list[tuple[str, Path]] = []
        combined_plots = (
            artifacts_dir / "hyperopt_combined" / "hyperopt" / "plots"
        )
        if combined_plots.parent.parent.exists():
            studies.append(("combined", combined_plots))
        per_animal_root = artifacts_dir / "hyperopt_per_animal"
        if per_animal_root.exists():
            for entry in sorted(per_animal_root.iterdir()):
                if entry.is_dir():
                    studies.append(
                        (entry.name, entry / "hyperopt" / "plots"),
                    )

        if not studies:
            QMessageBox.warning(
                self, "No studies found",
                f"No hyperopt study folders under {artifacts_dir}. "
                "Run hyperopt first.",
            )
            return

        # If the currently-selected row's plots dir doesn't exist,
        # warn the user rather than silently opening with whatever's
        # available -- they specifically asked to view THAT study's
        # plots.
        if not sel_workdir.joinpath("hyperopt", "plots").exists():
            resp = QMessageBox.warning(
                self,
                "Selected study has no plots",
                f"The plots folder for `{sel_label}` doesn't exist "
                f"({sel_workdir / 'hyperopt' / 'plots'}). The "
                "study's plot step may have failed -- check the run "
                "log.\n\n"
                "Open the dialog anyway to view other studies' "
                "plots?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if resp != QMessageBox.Yes:
                return
        dlg = HyperoptPlotsDialog(
            studies=studies,
            initial_study_label=sel_label,
            parent=self,
        )
        dlg.exec()

    def _on_hp_apply_to_retrain(self) -> None:
        """Copy the selected study's best params into the Retrain tab's
        spinboxes (w_neg, review FP weight, review FN weight) and
        switch to that tab so the user sees the values landed."""
        sel = self._selected_hyperopt_study()
        if sel is None:
            QMessageBox.information(
                self, "Pick a study",
                "Select a row in the results table first.",
            )
            return
        label, workdir = sel
        bp_path = workdir / "hyperopt" / "best_params.json"
        if not bp_path.exists():
            QMessageBox.warning(
                self, "No best_params.json",
                f"Expected {bp_path} but the file is missing. The "
                "study may not have finished.",
            )
            return
        try:
            data = json.loads(bp_path.read_text())
            params = data.get("best_params", {})
        except Exception as exc:
            QMessageBox.critical(
                self, "Could not read best_params.json",
                f"{exc}",
            )
            return

        w_neg = params.get("w_neg")
        fp_weight = params.get("fp_weight")
        fn_weight = params.get("fn_weight")
        # Spinboxes clamp to their configured ranges, so out-of-range
        # values just get pinned (and we tell the user).
        clamped_notes: list[str] = []
        if w_neg is not None:
            target = float(w_neg)
            lo, hi = self._w_neg_spin.minimum(), self._w_neg_spin.maximum()
            if target < lo or target > hi:
                clamped_notes.append(
                    f"w_neg {target:.4f} clamped to [{lo}, {hi}]"
                )
            self._w_neg_spin.setValue(max(lo, min(hi, target)))
        if fp_weight is not None:
            target = float(fp_weight)
            lo = self._review_fp_weight_spin.minimum()
            hi = self._review_fp_weight_spin.maximum()
            if target < lo or target > hi:
                clamped_notes.append(
                    f"fp_weight {target:.3f} clamped to [{lo}, {hi}]"
                )
            self._review_fp_weight_spin.setValue(max(lo, min(hi, target)))
        if fn_weight is not None:
            target = float(fn_weight)
            lo = self._review_fn_weight_spin.minimum()
            hi = self._review_fn_weight_spin.maximum()
            if target < lo or target > hi:
                clamped_notes.append(
                    f"fn_weight {target:.3f} clamped to [{lo}, {hi}]"
                )
            self._review_fn_weight_spin.setValue(max(lo, min(hi, target)))

        clamp_msg = (
            "\n\nNote: " + "; ".join(clamped_notes)
            if clamped_notes else ""
        )
        QMessageBox.information(
            self, "Applied",
            f"Best params from '{label}' applied to the Retrain tab:\n"
            f"  w_neg = {w_neg}\n"
            f"  review FP weight = {fp_weight}\n"
            f"  review FN weight = {fn_weight}"
            f"{clamp_msg}",
        )
        self._tabs.setCurrentWidget(self._tab_retrain)

    def _on_hp_apply_per_animal(self) -> None:
        """Load every per-animal study's best_params.json on disk
        and store them as overrides for the next per-animal training
        run. Each animal gets its OWN tuned w_neg / fp_weight /
        fn_weight from its own hyperopt study, instead of all
        animals sharing one global set."""
        artifacts_dir = detector_paths.get_artifacts_dir()
        per_animal_root = artifacts_dir / "hyperopt_per_animal"
        if not per_animal_root.exists():
            QMessageBox.warning(
                self, "No per-animal studies",
                f"No hyperopt_per_animal/ folder under "
                f"{artifacts_dir}. Run per-animal hyperopt first.",
            )
            return

        loaded: dict[str, dict] = {}
        missing: list[str] = []
        for animal_dir in sorted(per_animal_root.iterdir()):
            if not animal_dir.is_dir():
                continue
            animal = animal_dir.name
            bp_path = animal_dir / "hyperopt" / "best_params.json"
            if not bp_path.exists():
                missing.append(animal)
                continue
            try:
                data = json.loads(bp_path.read_text())
                params = data.get("best_params") or {}
                # Keep only the keys that meaningfully tune training.
                tuned = {
                    k: float(v) for k, v in params.items()
                    if k in ("w_neg", "fp_weight", "fn_weight")
                    and isinstance(v, (int, float))
                }
                if tuned:
                    loaded[animal] = tuned
                else:
                    missing.append(f"{animal} (best_params empty)")
            except Exception as e:
                missing.append(f"{animal} ({type(e).__name__}: {e})")

        if not loaded:
            QMessageBox.warning(
                self, "Nothing to apply",
                f"No per-animal best_params.json files found under "
                f"{per_animal_root}.\n\n"
                + (
                    "Issues: " + "; ".join(missing)
                    if missing else ""
                )
            )
            return

        self._per_animal_hyperopt_weights = loaded
        # Update the per-animal tab's status label so the user can
        # see the override is loaded.
        if hasattr(self, "_pa_hp_override_label"):
            self._refresh_per_animal_hp_override_label()

        summary_lines = [
            f"Loaded per-animal best params for {len(loaded)} animal(s):",
            "",
        ]
        for animal in sorted(loaded.keys()):
            wts = loaded[animal]
            summary_lines.append(
                f"  {animal}: "
                f"w_neg={wts.get('w_neg', '—'):.4g}, "
                f"fp_weight={wts.get('fp_weight', '—'):.4g}, "
                f"fn_weight={wts.get('fn_weight', '—'):.4g}"
            )
        if missing:
            summary_lines.append("")
            summary_lines.append("Skipped (no usable best_params):")
            for m in missing:
                summary_lines.append(f"  {m}")
        summary_lines.append("")
        summary_lines.append(
            "Switching to the Per-animal training tab. The next "
            "click of '▶ Train per-animal models' will use these "
            "per-animal weights."
        )
        QMessageBox.information(
            self,
            "Per-animal hyperopt params loaded",
            "\n".join(summary_lines),
        )
        self._tabs.setCurrentWidget(self._tab_per_animal)

    def _refresh_hyperopt_tab(self) -> None:
        """Tab-level refresh: rebuilds the animal-letter checkboxes (in
        case the manifest changed) and reloads the results table from
        disk."""
        # Only rebuild the animal checks if we're showing the per-
        # animal scope; otherwise it's wasted work.
        if self._hp_scope_per_animal.isChecked():
            self._refresh_hyperopt_animal_checks()
        self._refresh_hyperopt_results_table()

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

        # Cross-machine path remap (Drive mount points differ between
        # computers; manifest paths get translated transparently on
        # load). Auto-detect handles the common case but the user can
        # add explicit prefix rules here.
        remap_group = QGroupBox("Cross-machine path remap")
        remap_layout = QVBoxLayout(remap_group)
        remap_intro = QLabel(
            "<i>The training manifest stores absolute paths to "
            "recordings on Google Drive. Mount points differ between "
            "computers (e.g. <code>G:\\Shared drives\\...</code> on "
            "Windows vs <code>~/Library/CloudStorage/GoogleDrive-.../"
            "Shared drives/...</code> on macOS). Auto-detection "
            "searches common Drive roots for the same subpath; for "
            "edge cases, add explicit prefix rules in the JSON file "
            "below.</i>"
        )
        remap_intro.setWordWrap(True)
        remap_intro.setStyleSheet("color: #aaa; padding-bottom: 4px;")
        remap_layout.addWidget(remap_intro)

        try:
            from detector.path_remap import (
                CONFIG_PATH as _remap_cfg_path,
                is_auto_detect_enabled,
                set_auto_detect_enabled,
                get_remap_rules,
            )
            self._remap_auto_cb = QCheckBox(
                "Auto-detect Drive mounts (try candidate roots when "
                "the stored path doesn't exist)"
            )
            self._remap_auto_cb.setChecked(is_auto_detect_enabled())
            self._remap_auto_cb.toggled.connect(
                lambda v: set_auto_detect_enabled(v)
            )
            remap_layout.addWidget(self._remap_auto_cb)
            rules = get_remap_rules()
            self._remap_status_lbl = QLabel(
                f"<b>{len(rules)}</b> explicit rule(s) configured. "
                f"Config file: <code>{_remap_cfg_path}</code>"
            )
            self._remap_status_lbl.setStyleSheet(
                "font-family: monospace; font-size: 11px;"
            )
            self._remap_status_lbl.setTextInteractionFlags(
                Qt.TextSelectableByMouse
            )
            remap_layout.addWidget(self._remap_status_lbl)
            btn_row = QHBoxLayout()
            self._btn_open_remap_cfg = QPushButton(
                "Open remap config file"
            )
            self._btn_open_remap_cfg.setToolTip(
                f"Open {_remap_cfg_path} in the system editor. Edit "
                "the JSON to add explicit prefix-replacement rules "
                "when auto-detect can't find the right mount. "
                "Schema: { \"auto_detect\": true, \"remap_rules\": "
                "[{\"match\": \"<old_prefix>\", \"replace\": "
                "\"<local_prefix>\"}] }"
            )
            self._btn_open_remap_cfg.clicked.connect(
                self._on_open_remap_config
            )
            self._btn_test_remap = QPushButton(
                "Test path resolution…"
            )
            self._btn_test_remap.setToolTip(
                "Paste a manifest path and see whether the remap "
                "layer can resolve it on this machine."
            )
            self._btn_test_remap.clicked.connect(
                self._on_test_remap_path
            )
            btn_row.addWidget(self._btn_open_remap_cfg)
            btn_row.addWidget(self._btn_test_remap)
            btn_row.addStretch(1)
            remap_layout.addLayout(btn_row)
        except Exception as e:
            err_lbl = QLabel(
                f"<i>(path remap module unavailable: {e})</i>"
            )
            err_lbl.setStyleSheet("color: #c44;")
            remap_layout.addWidget(err_lbl)
        layout.addWidget(remap_group)

        layout.addStretch(1)
        return w

    # ------------------------------------------------------------------
    # Path-remap settings handlers
    # ------------------------------------------------------------------

    def _on_open_remap_config(self) -> None:
        """Open the path-remap JSON config in the system's default
        editor. If the file doesn't exist yet, create a stub with the
        current auto-detect state so the user has something to edit."""
        from detector.path_remap import CONFIG_PATH
        if not CONFIG_PATH.exists():
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            CONFIG_PATH.write_text(json.dumps({
                "auto_detect": True,
                "remap_rules": [
                    {
                        "match": "<old_prefix e.g. G:\\Shared drives>",
                        "replace": "<local_prefix e.g. /Volumes/GoogleDrive/Shared drives>",
                    },
                ],
                "extra_roots": [],
            }, indent=2))
        # Open in the system editor.
        import subprocess
        import sys as _sys
        try:
            if _sys.platform == "darwin":
                subprocess.Popen(["open", str(CONFIG_PATH)])
            elif _sys.platform == "win32":
                os.startfile(str(CONFIG_PATH))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(CONFIG_PATH)])
        except Exception as e:
            QMessageBox.warning(
                self, "Couldn't open config",
                f"Edit it manually at:\n{CONFIG_PATH}\n\nError: {e}",
            )

    def _on_test_remap_path(self) -> None:
        """Pop a dialog to paste a path and see whether the remap
        layer can resolve it. Useful for debugging cross-machine
        manifests."""
        from detector.path_remap import (
            remap_path, format_remap_diagnostic,
            learn_rule_from_resolution, set_remap_rules,
            get_remap_rules,
        )
        path_text, ok = QInputDialog.getText(
            self, "Test path resolution",
            "Paste a manifest path to test:",
        )
        if not ok or not path_text.strip():
            return
        stored = path_text.strip()
        resolved = remap_path(stored)
        if resolved == stored and not Path(stored).exists():
            QMessageBox.warning(
                self, "Could not resolve",
                format_remap_diagnostic(stored),
            )
            return
        if resolved == stored:
            QMessageBox.information(
                self, "Resolved",
                f"Path exists as-is on this machine:\n\n{stored}",
            )
            return
        # Different path -- auto-detect resolved it. Offer to save
        # the inferred rule for next time.
        rule = learn_rule_from_resolution(stored, resolved)
        msg = (
            f"Stored path:\n  {stored}\n\n"
            f"Resolved to:\n  {resolved}\n\n"
        )
        if rule:
            msg += (
                f"Inferred remap rule:\n"
                f"  match:   {rule['match']}\n"
                f"  replace: {rule['replace']}\n\n"
                "Save this rule so future resolutions skip the auto-"
                "detect search?"
            )
            resp = QMessageBox.question(
                self, "Save remap rule?", msg,
                QMessageBox.Yes | QMessageBox.No,
            )
            if resp == QMessageBox.Yes:
                rules = get_remap_rules()
                # Avoid duplicate rules.
                if not any(r.get("match") == rule["match"]
                           and r.get("replace") == rule["replace"]
                           for r in rules):
                    rules.append(rule)
                    set_remap_rules(rules)
                # Refresh status label.
                if hasattr(self, "_remap_status_lbl"):
                    from detector.path_remap import (
                        CONFIG_PATH as _cfg,
                    )
                    self._remap_status_lbl.setText(
                        f"<b>{len(rules)}</b> explicit rule(s) "
                        f"configured. Config file: <code>{_cfg}</code>"
                    )
        else:
            QMessageBox.information(self, "Resolved", msg)

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
        self._refresh_per_animal_tab()
        try:
            self._refresh_hyperopt_tab()
        except Exception:
            pass
        # The convergence-tab dropdown is keyed off list_versions(),
        # which changes after a retrain. Repopulating here picks up
        # the newly trained model automatically.
        try:
            self._refresh_convergence_versions()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Window-close handling
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Shut down any running background threads cleanly before
        Qt destroys this window.

        Without this, closing the training window (or the entire app)
        while a long-running operation is active destroys the
        underlying QThread C++ object while its native worker is
        still executing. Qt prints `QThread: Destroyed while thread
        '' is still running` and kills the process.

        Two-stage handling:
          1. If a long-running op is active, prompt the user with
             explicit Yes/No -- close AND lose the in-progress trial
             / file / fold, OR cancel the close and keep the run
             alive. Completed work (trials in study.db, files in
             clean.h5, etc.) is always safe; only the
             currently-executing item is lost.
          2. If the user confirms close, drain each running thread
             via request_cancel + quit + wait(timeout) so the OS
             thread finishes before Qt tears down the QThread C++
             object.
        """
        # Detect running threads BEFORE prompting.
        running_ops: list[tuple[str, str]] = []
        if (self._hyperopt_thread is not None
                and self._hyperopt_thread.isRunning()):
            running_ops.append((
                "Hyperopt",
                "Completed trials are flushed to study.db after "
                "each one finishes -- you can resume by clicking "
                "Run hyperopt again. Only the currently-running "
                "trial is lost.",
            ))
        if (self._per_animal_thread is not None
                and self._per_animal_thread.isRunning()):
            running_ops.append((
                "Per-animal training",
                "Animals already finished have their model_v* "
                "directories written. Only the currently-training "
                "animal is lost; the others stay.",
            ))
        if (self._pre_train_thread is not None
                and self._pre_train_thread.isRunning()):
            running_ops.append((
                "Blankmotion migration",
                "Files already migrated have their _clean.h5 / "
                "_bad.h5 on disk -- re-running the migration will "
                "skip them. Only the currently-migrating file is "
                "interrupted.",
            ))

        if running_ops:
            op_names = ", ".join(op for op, _ in running_ops)
            details = "\n\n".join(
                f"• {op}: {note}" for op, note in running_ops
            )
            resp = QMessageBox.question(
                self,
                "Long-running operation in progress",
                f"<b>{op_names}</b> still running.\n\n"
                "If you close now:\n\n"
                f"{details}\n\n"
                "Close anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,    # default: cancel close
            )
            if resp != QMessageBox.Yes:
                event.ignore()
                return

        # ---- pre-train blankmotion migration ------------------------
        if (self._pre_train_thread is not None
                and self._pre_train_thread.isRunning()):
            try:
                if (self._pre_train_worker is not None
                        and hasattr(self._pre_train_worker,
                                     "request_cancel")):
                    self._pre_train_worker.request_cancel()
            except Exception:
                pass
            try:
                self._pre_train_thread.quit()
                # 30 s for a small per-file migration to finish or
                # cancel cleanly.
                self._pre_train_thread.wait(30_000)
            except Exception:
                pass

        # ---- per-animal training ------------------------------------
        if (self._per_animal_thread is not None
                and self._per_animal_thread.isRunning()):
            try:
                if (self._per_animal_worker is not None
                        and hasattr(self._per_animal_worker,
                                     "request_cancel")):
                    self._per_animal_worker.request_cancel()
            except Exception:
                pass
            try:
                self._per_animal_thread.quit()
                # 60 s -- the orchestrator may be mid-fold and can't
                # be interrupted cleanly, but waiting lets it write
                # its current partial artifact safely.
                self._per_animal_thread.wait(60_000)
            except Exception:
                pass

        # ---- hyperopt ------------------------------------------------
        # This is the long-running one (1-6 hours). It runs through
        # Pool workers internally that we can't easily signal, so the
        # best we can do is quit() + wait() with a generous timeout.
        # If wait() times out, we accept the crash risk and let Qt
        # tear down -- the user has explicitly asked to close.
        if (self._hyperopt_thread is not None
                and self._hyperopt_thread.isRunning()):
            try:
                if (self._hyperopt_worker is not None
                        and hasattr(self._hyperopt_worker,
                                     "request_cancel")):
                    self._hyperopt_worker.request_cancel()
            except Exception:
                pass
            try:
                self._hyperopt_thread.quit()
                # 90 s -- enough to flush whatever the worker is
                # writing (parquet checkpoints, trial logs) but not
                # so long that the user gives up and force-quits.
                # Internal Pool workers may still be running after
                # this; we accept that for now -- their subprocesses
                # are independent of this QThread and won't crash
                # the GUI.
                self._hyperopt_thread.wait(90_000)
            except Exception:
                pass

        # Always accept the close -- we've done our best to clean up.
        # Anything still running is the user's problem to monitor via
        # the subprocess status file.
        super().closeEvent(event)


def _fmt(v, decimals: int) -> str:
    """Format a number safely for table display."""
    if v is None:
        return "—"
    try:
        return f"{float(v):.{decimals}f}"
    except Exception:
        return str(v)


def _hp_n_trials_for(label: str) -> str:
    """Count completed trials for a hyperopt study label by reading
    its trial_log.json (cheap; ~10 KB for a 30-trial study). Falls
    back to '—' if the log is missing -- best_params.json can be
    present without a trial_log if the user did some manual surgery,
    but typically the two land together.
    """
    artifacts_dir = detector_paths.get_artifacts_dir()
    if label == "combined":
        log = (artifacts_dir / "hyperopt_combined" / "hyperopt"
               / "trial_log.json")
    else:
        log = (artifacts_dir / "hyperopt_per_animal" / label / "hyperopt"
               / "trial_log.json")
    if not log.exists():
        return "—"
    try:
        return str(len(json.loads(log.read_text())))
    except Exception:
        return "—"


# ----------------------------------------------------------------------
# Sub-dialogs
# ----------------------------------------------------------------------

class AddRecordingDialog(QDialog):
    """Form for adding a new recording to the manifest. Pre-fills
    fields from any of clean.h5 / blankmotion.mat / source.mat
    via the shared `resolve_record_from_picked_file` helper."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Add recording")
        self.resize(680, 420)

        outer = QVBoxLayout(self)

        # Banner explaining the wider input scope. Without this, users
        # who only ever see "clean.h5" labels won't know they can pick
        # a `_blankmotion.mat` and get everything auto-filled (with
        # an inline migration prompt if the splitter files don't
        # exist yet).
        banner = QLabel(
            "<b>Quick fill:</b> click the <code>…</code> next to "
            "<i>clean.h5</i> and pick ANY of "
            "<code>*_clean.h5</code>, <code>*_blankmotion.mat</code>, "
            "or source <code>*.mat</code> -- all other fields "
            "auto-fill from siblings. For legacy "
            "<code>_blankmotion.mat</code> files with no splitter "
            "siblings, you'll be prompted to migrate inline."
        )
        banner.setTextFormat(Qt.RichText)
        banner.setWordWrap(True)
        banner.setStyleSheet(
            "padding: 6px; background-color: #eef5ff; "
            "border-left: 3px solid #2a7fff;"
        )
        outer.addWidget(banner)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        outer.addWidget(form_widget)
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
        btn_clean.setToolTip(
            "Pick clean.h5, blankmotion.mat, OR source .mat. "
            "Every other field on the form auto-fills from the "
            "siblings discovered alongside the picked file."
        )
        btn_clean.clicked.connect(self._pick_clean)
        clean_row.addWidget(btn_clean)
        form.addRow("clean.h5 (or any input file)", clean_row)
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
        # Held-out flag. When checked, this recording is EXCLUDED from
        # the training corpus (manifest.list_recordings filters it out
        # by default) and shows up only in the held-out evaluator
        # (Tools -> Held-out evaluation). Use this for fresh-eye
        # ground-truth recordings from never-seen animals, so we can
        # measure honest generalization across retrains.
        self._held_out_check = QCheckBox(
            "Hold out from training (use only for evaluation)"
        )
        self._held_out_check.setToolTip(
            "Recording will be EXCLUDED from training corpus and used "
            "only by the held-out evaluator to measure model-vs-human "
            "accuracy on never-trained data."
        )
        form.addRow("held_out", self._held_out_check)

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
        """File-picker that accepts clean.h5 / blankmotion.mat / source.mat
        and uses the shared `resolve_record_from_picked_file` helper to
        populate every field in the dialog.

        For legacy `_blankmotion.mat` files with no splitter siblings,
        the resolver prompts to run migrate_one_blankmotion inline -- so
        you can add legacy MATLAB-labeled recordings to the manifest
        without manually running the bulk migration tool first."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick recording or labels file", "",
            "Recordings + labels (*_clean.h5 *_blankmotion.mat *.mat);;"
            "Splitter clean.h5 (*_clean.h5);;"
            "Legacy blankmotion (*_blankmotion.mat);;"
            "Source MAT (*.mat);;"
            "All files (*)",
        )
        if not path:
            return
        record = resolve_record_from_picked_file(
            Path(path), parent_widget=self,
        )
        if record is None:
            # Resolver already showed a message about what was missing.
            # Fall back to the legacy behavior so the user can still
            # fill fields manually.
            self._clean_edit.setText(path)
            self._maybe_autofill_from_clean_h5()
            return
        # Populate every field from the resolver's record.
        if not self._rid_edit.text().strip():
            self._rid_edit.setText(record["recording_id"])
        self._source_edit.setText(record["source_path"])
        self._clean_edit.setText(record["splitter_clean_path"])
        self._bad_edit.setText(record["splitter_bad_path"])
        if record.get("splitter_baseline_path"):
            self._baseline_edit.setText(record["splitter_baseline_path"])
        self._fs_spin.setValue(float(record["fs"]))
        self._n_samples_spin.setValue(int(record["n_samples"]))
        self._n_ch_spin.setValue(int(record["n_channels"]))
        self._rec_type_combo.setCurrentText(record["rec_type"])

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
            "held_out": bool(self._held_out_check.isChecked()),
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
