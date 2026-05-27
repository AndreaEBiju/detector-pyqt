"""Migrate-blankmotion bulk UI.

Opens from Tools -> "Migrate legacy blankmotion files…". Lets the
user build a queue of `_blankmotion.mat` files (via individual file
pickers OR a folder picker), pre-validates each one's source-sibling
status, and one button kicks off the bulk migration that produces
modern splitter `_clean.h5` + `_bad.h5` + `_baseline.h5` outputs.

Per-file progress streams back live into the table so the user can
watch which files complete and which fail. After the run a summary
dialog reports counts; errored files stay in the table with their
status flipped to "error" + the error message in the notes column,
so the user can fix + retry.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QFileDialog, QHBoxLayout, QHeaderView,
    QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QSpinBox, QTableWidget, QTableWidgetItem, QToolBar, QVBoxLayout,
    QWidget,
)

# Self-contained sys.path setup -- same as the other windows in this folder.
_repo_root = Path(__file__).resolve().parent.parent.parent
_detector_core = _repo_root / "detector-core"
if _detector_core.exists() and str(_detector_core) not in sys.path:
    sys.path.insert(0, str(_detector_core))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from detector.migrate_blankmotion import (                          # noqa: E402
    _find_source_for_blankmotion, find_blankmotion_files,
)
from ui.workers.blankmotion_migration_worker import (               # noqa: E402
    BlankmotionMigrationWorker,
)


# Table column indices
COL_FILENAME = 0
COL_SOURCE = 1
COL_STATUS = 2
COL_NOTES = 3


class BlankmotionMigrationWindow(QMainWindow):
    """Non-modal window for bulk-migrating legacy `_blankmotion.mat`
    labels into the modern splitter training-corpus format."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Migrate blankmotion -> splitter h5")
        self.resize(960, 560)

        self._files: list[Path] = []      # in-order; same order as table rows
        self._worker: Optional[BlankmotionMigrationWorker] = None
        self._thread: Optional[QThread] = None

        self._build_ui()
        self._refresh_summary()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ---- Top toolbar: Add files / Add folder / Remove / Clear ----
        tb = QToolBar("Queue actions")
        tb.setMovable(False)
        self.addToolBar(tb)

        act_add_files = QAction("➕ Add files…", self)
        act_add_files.triggered.connect(self._on_add_files)
        tb.addAction(act_add_files)

        act_add_folder = QAction("📁 Add folder…", self)
        act_add_folder.triggered.connect(self._on_add_folder)
        tb.addAction(act_add_folder)

        tb.addSeparator()

        act_remove = QAction("➖ Remove selected", self)
        act_remove.triggered.connect(self._on_remove_selected)
        tb.addAction(act_remove)

        act_clear = QAction("🗑 Clear queue", self)
        act_clear.triggered.connect(self._on_clear)
        tb.addAction(act_clear)

        # ---- Help banner ----
        help_lbl = QLabel(
            "<i>Pick legacy <code>_blankmotion.mat</code> files. Each "
            "needs a matching source <code>&lt;base&gt;.mat</code> or "
            "<code>&lt;base&gt;_notched.mat</code> as a sibling. The "
            "&quot;Source&quot; column shows ✓ when found.</i>"
        )
        help_lbl.setTextFormat(Qt.RichText)
        help_lbl.setWordWrap(True)
        help_lbl.setContentsMargins(8, 4, 8, 4)
        root.addWidget(help_lbl)

        # ---- Main table ----
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            ["Blankmotion file", "Source", "Status", "Notes / output"]
        )
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        h = self._table.horizontalHeader()
        h.setSectionResizeMode(COL_FILENAME, QHeaderView.Stretch)
        h.setSectionResizeMode(COL_SOURCE, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(COL_STATUS, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(COL_NOTES, QHeaderView.Stretch)
        root.addWidget(self._table)

        # ---- Options row ----
        opt_row = QHBoxLayout()
        self._force_cb = QCheckBox("Force (overwrite existing outputs)")
        self._force_cb.setToolTip(
            "By default, files whose 3-file output set (clean.h5 + "
            "bad.h5 + baseline.h5) already exists are skipped. Check "
            "this to overwrite them."
        )
        opt_row.addWidget(self._force_cb)
        # NEW: also add each successfully-migrated recording to the
        # training manifest. Default OFF -- manifest changes are
        # consequential (every retrain uses the manifest), so users
        # should explicitly opt in.
        self._add_to_manifest_cb = QCheckBox(
            "Also add to training manifest"
        )
        self._add_to_manifest_cb.setToolTip(
            "After each successful migration, register the recording "
            "in the training manifest (~/.detector/training_manifest.json) "
            "so it participates in the next retrain. Default OFF -- "
            "review the migration outputs first if you're unsure."
        )
        opt_row.addWidget(self._add_to_manifest_cb)
        opt_row.addStretch(1)
        opt_row.addWidget(QLabel("Workers:"))
        self._workers_spin = QSpinBox()
        self._workers_spin.setRange(0, 32)
        self._workers_spin.setValue(0)
        self._workers_spin.setToolTip(
            "0 = auto-detect from machine (recommended). Otherwise the "
            "number of parallel migration workers. Each loads one big "
            ".mat at a time; lower for memory-constrained machines."
        )
        opt_row.addWidget(self._workers_spin)
        root.addLayout(opt_row)

        # ---- Run / Cancel / Close + progress bar ----
        action_row = QHBoxLayout()
        self._btn_run = QPushButton("▶ Process all (0 files)")
        self._btn_run.clicked.connect(self._on_run)
        self._btn_run.setEnabled(False)
        action_row.addWidget(self._btn_run)

        self._btn_cancel = QPushButton("✖ Cancel")
        self._btn_cancel.clicked.connect(self._on_cancel)
        self._btn_cancel.setEnabled(False)
        action_row.addWidget(self._btn_cancel)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFormat("")
        action_row.addWidget(self._progress, 1)

        self._btn_close = QPushButton("Close")
        self._btn_close.clicked.connect(self.close)
        action_row.addWidget(self._btn_close)
        root.addLayout(action_row)

        # ---- Status bar at bottom ----
        self._summary_lbl = QLabel("")
        self._summary_lbl.setContentsMargins(8, 0, 8, 4)
        root.addWidget(self._summary_lbl)

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def _last_dir(self) -> str:
        """Default directory for the file/folder pickers. Remembers
        the most recent picked location so the user doesn't have to
        re-navigate from C:\\ every time. Falls back to the user's
        home dir. Passing a real path (not '') also dodges Qt's
        Windows-only 'Unhandled scheme: data' warning from
        QWindowsNativeFileDialogBase when the dialog's default is
        ambiguous."""
        return getattr(self, "_remembered_dir", None) or str(Path.home())

    def _remember_dir(self, path: Path) -> None:
        self._remembered_dir = str(path.parent if path.is_file() else path)

    def _on_add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Pick legacy _blankmotion.mat files",
            self._last_dir(),
            "Blankmotion (*_blankmotion.mat);;All MAT files (*.mat);;All files (*)",
        )
        if paths:
            self._remember_dir(Path(paths[0]))
            self._add_files([Path(p) for p in paths])

    def _on_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Pick folder to scan for _blankmotion.mat (recursive)",
            self._last_dir(),
        )
        if folder:
            self._remember_dir(Path(folder))
            found = find_blankmotion_files(Path(folder), recursive=True)
            if not found:
                QMessageBox.information(
                    self, "No files found",
                    f"No `_blankmotion.mat` files under:\n{folder}",
                )
                return
            self._add_files(found)

    def _add_files(self, paths: list[Path]) -> None:
        """Append paths to the queue, deduplicating against what's
        already there. Auto-detect source-sibling status."""
        existing = {str(p) for p in self._files}
        added = 0
        for p in paths:
            if str(p) in existing:
                continue
            existing.add(str(p))
            self._files.append(p)
            self._append_table_row(p)
            added += 1
        self._refresh_summary()
        if added < len(paths):
            QMessageBox.information(
                self, "Some duplicates ignored",
                f"Added {added} new file(s); "
                f"{len(paths) - added} were already in the queue.",
            )

    def _append_table_row(self, p: Path) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        # Filename (with full path as tooltip)
        item_name = QTableWidgetItem(p.name)
        item_name.setToolTip(str(p))
        self._table.setItem(row, COL_FILENAME, item_name)

        # Source-found check
        src = _find_source_for_blankmotion(p)
        if src is None:
            src_item = QTableWidgetItem("✗ missing")
            src_item.setForeground(QColor("#d62728"))
            src_item.setToolTip(
                "No matching source `<base>.mat` or "
                "`<base>_notched.mat` found in the same folder. This "
                "file cannot be migrated until a source is present."
            )
        else:
            src_item = QTableWidgetItem(f"✓ {src.name}")
            src_item.setForeground(QColor("#2ca02c"))
            src_item.setToolTip(str(src))
        self._table.setItem(row, COL_SOURCE, src_item)

        # Status starts as "queued"
        self._table.setItem(row, COL_STATUS, QTableWidgetItem("queued"))
        # Notes empty initially
        self._table.setItem(row, COL_NOTES, QTableWidgetItem(""))

    def _on_remove_selected(self) -> None:
        rows = sorted(
            {idx.row() for idx in self._table.selectedIndexes()},
            reverse=True,
        )
        for row in rows:
            self._table.removeRow(row)
            del self._files[row]
        self._refresh_summary()

    def _on_clear(self) -> None:
        if not self._files:
            return
        resp = QMessageBox.question(
            self, "Clear queue",
            f"Remove all {len(self._files)} file(s) from the queue?",
        )
        if resp != QMessageBox.Yes:
            return
        self._table.setRowCount(0)
        self._files.clear()
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        n = len(self._files)
        n_with_source = 0
        for i, p in enumerate(self._files):
            src_item = self._table.item(i, COL_SOURCE)
            if src_item is not None and src_item.text().startswith("✓"):
                n_with_source += 1
        self._btn_run.setText(
            f"▶ Process all ({n_with_source} of {n} ready)"
        )
        self._btn_run.setEnabled(
            n_with_source > 0 and self._worker is None
        )
        self._summary_lbl.setText(
            f"{n} queued; {n_with_source} have a matching source"
            f"{f' ({n - n_with_source} missing source)' if n > n_with_source else ''}"
        )

    # ------------------------------------------------------------------
    # Run / Cancel
    # ------------------------------------------------------------------

    def _on_run(self) -> None:
        # Only process files with a source -- the orphans would just
        # produce "no source found" errors. Still keep them in the
        # table so the user knows they were skipped at this stage.
        eligible: list[Path] = []
        eligible_rows: list[int] = []
        for i, p in enumerate(self._files):
            src_item = self._table.item(i, COL_SOURCE)
            if src_item is not None and src_item.text().startswith("✓"):
                eligible.append(p)
                eligible_rows.append(i)
                self._table.setItem(i, COL_STATUS,
                                      QTableWidgetItem("pending"))
            else:
                self._table.setItem(i, COL_STATUS,
                                      QTableWidgetItem("(no source)"))
        if not eligible:
            QMessageBox.warning(
                self, "Nothing to process",
                "All queued files are missing their source sibling. "
                "Add the source recordings (`<base>.mat` or "
                "`<base>_notched.mat`) into the same folders.",
            )
            return

        # Spin up the worker.
        n_workers = self._workers_spin.value() or None
        self._worker = BlankmotionMigrationWorker(
            eligible,
            force=self._force_cb.isChecked(),
            n_workers=n_workers,
        )
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._progress.setRange(0, len(eligible))
        self._progress.setValue(0)
        self._progress.setFormat(f"0 / {len(eligible)}")
        self._btn_run.setEnabled(False)
        self._btn_cancel.setEnabled(True)

        self._thread.start()

    def _on_progress(self, n_done: int, n_total: int, res: dict) -> None:
        """One file just finished -- update its row + the progress bar."""
        bp = res.get("blankmotion_path")
        row = self._row_for_path(bp)
        # Tally how many got added to manifest this run so we can
        # report it in the final summary.
        manifest_status = None
        if (res["status"] in ("ok", "skipped_existing")
                and self._add_to_manifest_cb.isChecked()
                and res.get("source_path")):
            manifest_status = self._add_to_manifest(res)
            # Track counts on `self` for the finished-summary dialog.
            self._n_manifest_added = getattr(
                self, "_n_manifest_added", 0,
            ) + (1 if manifest_status == "added" else 0)
            self._n_manifest_dup = getattr(
                self, "_n_manifest_dup", 0,
            ) + (1 if manifest_status == "duplicate" else 0)
            self._n_manifest_fail = getattr(
                self, "_n_manifest_fail", 0,
            ) + (1 if manifest_status not in ("added", "duplicate")
                 else 0)
        if row is not None:
            status_item = QTableWidgetItem(res.get("status", "?"))
            notes_text = ""
            if res["status"] == "ok":
                status_item.setForeground(QColor("#2ca02c"))
                n_int = res.get("n_intervals", "?")
                notes_text = (f"{n_int} intervals migrated; "
                              f"baseline {res.get('baseline_status', '?')}")
            elif res["status"] == "skipped_existing":
                status_item.setForeground(QColor("#9467bd"))
                notes_text = "outputs already existed (use Force to overwrite)"
            elif res["status"] == "error":
                status_item.setForeground(QColor("#d62728"))
                notes_text = res.get("error", "?")
            # Append manifest-add status to the notes column so the
            # user sees per-row what happened in the manifest step.
            if manifest_status == "added":
                notes_text += "  · +manifest"
            elif manifest_status == "duplicate":
                notes_text += "  · manifest already has it"
            elif manifest_status not in (None, "ok"):
                notes_text += f"  · manifest add FAILED: {manifest_status}"
            self._table.setItem(row, COL_STATUS, status_item)
            self._table.setItem(row, COL_NOTES,
                                  QTableWidgetItem(notes_text))
        self._progress.setValue(n_done)
        self._progress.setFormat(f"{n_done} / {n_total}")

    def _add_to_manifest(self, res: dict) -> str:
        """Register the just-migrated recording in the training
        manifest. Returns one of: 'added', 'duplicate', or an error
        string. Reuses `resolve_record_from_picked_file` so the field
        population matches what the Training Management add-recording
        flow uses."""
        try:
            from ui.windows.training_window import (
                resolve_record_from_picked_file,
            )
            from detector.manifest import Manifest
            from detector import paths as detector_paths

            source_path = res.get("source_path")
            if not source_path:
                return "no source_path in result"
            record = resolve_record_from_picked_file(
                Path(source_path),
                parent_widget=None,           # no prompts in bulk mode
                auto_migrate_blankmotion=False,
            )
            if record is None:
                return "could not resolve record fields"

            manifest_path = detector_paths.get_manifest_path()
            if not manifest_path.exists():
                return f"manifest not found at {manifest_path}"
            m = Manifest.load(manifest_path)
            # Duplicate check
            if any(r["recording_id"] == record["recording_id"]
                   for r in m.recordings):
                return "duplicate"
            m.add_recording(record, by="blankmotion_migration_ui")
            m.save(manifest_path)
            return "added"
        except Exception as e:
            return f"{type(e).__name__}: {e}"

    def _row_for_path(self, bp: Optional[str]) -> Optional[int]:
        if not bp:
            return None
        target = Path(bp)
        for i, p in enumerate(self._files):
            if Path(p) == target:
                return i
        return None

    def _on_finished(self, summary: dict) -> None:
        self._worker = None
        self._thread = None
        self._btn_cancel.setEnabled(False)
        self._refresh_summary()
        n_conflicts = int(summary.get("n_skipped_conflict", 0))
        warnings = list(summary.get("warnings", []))
        msg = (
            f"Done.\n\n"
            f"  ok:                 {summary['n_ok']}\n"
            f"  skipped (existing): {summary['n_skipped_existing']}\n"
            f"  skipped (conflict): {n_conflicts}\n"
            f"  errors:             {summary['n_error']}\n"
            f"  elapsed:            {summary['elapsed_s']:.1f}s"
        )
        # File-collision warnings happen when the user's folder has
        # the full `_blankmotion.mat` AND its period-specific variants
        # (`_stim_blankmotion.mat` / `_recovery_blankmotion.mat`), OR
        # has multiple period variants without a full file. All
        # variants target the same `_clean.h5` / `_bad.h5` output
        # pair, so without the conflict pass the last writer would
        # silently overwrite earlier outputs. Surface the warnings
        # prominently -- this is the user's only signal that some
        # intervals didn't make it into the migrated outputs.
        if warnings:
            msg += "\n\nFile-collision warnings:"
            for w in warnings[:8]:
                msg += f"\n  - {w}"
            if len(warnings) > 8:
                msg += f"\n  ... and {len(warnings) - 8} more"
        # If manifest-add was enabled, append those counts to the
        # summary so the user sees what landed in the manifest.
        if self._add_to_manifest_cb.isChecked():
            n_added = getattr(self, "_n_manifest_added", 0)
            n_dup = getattr(self, "_n_manifest_dup", 0)
            n_fail = getattr(self, "_n_manifest_fail", 0)
            msg += (
                f"\n\nManifest:\n"
                f"  added:              {n_added}\n"
                f"  already present:    {n_dup}\n"
                f"  failed to add:      {n_fail}"
            )
            # Reset counters for the next run.
            self._n_manifest_added = 0
            self._n_manifest_dup = 0
            self._n_manifest_fail = 0
        if summary["n_error"] > 0 or warnings:
            title = ("Migration complete (with errors)"
                     if summary["n_error"] > 0 else
                     "Migration complete (with conflict warnings)")
            QMessageBox.warning(self, title, msg)
        else:
            QMessageBox.information(self, "Migration complete", msg)

    def _on_error(self, msg: str) -> None:
        self._worker = None
        self._thread = None
        self._btn_cancel.setEnabled(False)
        self._refresh_summary()
        QMessageBox.critical(self, "Migration crashed", msg)

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.request_cancel()
        self._btn_cancel.setEnabled(False)

    def closeEvent(self, event) -> None:
        # If a migration is in flight, ask the user, then WAIT for the
        # thread to actually exit before letting the window close.
        # Without the wait, the Python QThread reference drops while
        # the underlying C++ thread is still running -- Qt prints:
        #   QThread: Destroyed while thread '' is still running
        # The warning is benign in our case (Pool workers self-clean),
        # but it's confusing and looks alarming in the terminal.
        if self._worker is not None and self._thread is not None:
            resp = QMessageBox.question(
                self, "Migration in progress",
                "A migration is still running. Cancel it and close?",
            )
            if resp != QMessageBox.Yes:
                event.ignore()
                return
            self._worker.request_cancel()
            self._thread.quit()
            # Wait up to 30s for the thread loop to exit cleanly.
            # imap_unordered will keep returning results from already-
            # in-flight Pool workers; once they finish, the loop ends.
            self._thread.wait(30_000)
        super().closeEvent(event)
