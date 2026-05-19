"""Preprocess window — TDT batch preprocessing workflow.

A QStackedWidget that steps the user through:

    Step 1: Select folders          (MultiFolderPicker → list of TDTs)
    Step 2: Configure batch         (TdtBatchTable: condition / output
                                     name / animal id per row)
    Step 3: Channel consistency?    (one yes/no question — same
                                     electrodes for all? if Yes, ask
                                     once; if No, ask per-animal)
    Step 4: Per-animal review       (loop: channel assignment + notch
                                     review, once per animal_id; skips
                                     animals with an existing profile
                                     unless the user opts to re-review)
    Step 5: Confirm / Already-done  (P11: detect existing output files
                                     in the destination, ask per-file
                                     skip / re-process / suffix)
    Step 6: Progress                (live: file_started, file_completed,
                                     cancel button, progress bar)
    Step 7: Report                  (per-file ok/skipped/error rows,
                                     validation warnings, Open output
                                     folder)

The window is launched modally from MainWindow's File menu. On
close, it stays detached — there's no state to bring back into the
main window (the produced .mat files are loaded via the normal
Open recording… flow).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFrame, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QRadioButton, QStackedWidget,
    QStatusBar, QVBoxLayout, QWidget,
)

_repo_root = Path(__file__).resolve().parent.parent.parent
_detector_core = _repo_root / "detector-core"
if _detector_core.exists() and str(_detector_core) not in sys.path:
    sys.path.insert(0, str(_detector_core))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from detector.preprocessing.batch import BatchPlanItem               # noqa: E402
from detector.preprocessing.profiles import Profile                   # noqa: E402
from detector.preprocessing.tdt_io import (                           # noqa: E402
    StreamInfo, list_streams_cached,
)

from ui.dialogs.folder_picker import pick_folders                     # noqa: E402
from ui.widgets.tdt_batch_table import TdtBatchTable, BatchRowState   # noqa: E402
from ui.widgets.channel_assignment import ChannelAssignmentDialog     # noqa: E402
from ui.widgets.notch_review import NotchReviewDialog                 # noqa: E402
from ui.workers.preprocess_worker import PreprocessWorker             # noqa: E402


STEP_SELECT_FOLDERS = 0
STEP_CONFIGURE_BATCH = 1
STEP_CONSISTENCY = 2
STEP_REVIEW = 3
STEP_CONFIRM = 4
STEP_PROGRESS = 5
STEP_REPORT = 6


class PreprocessWindow(QMainWindow):
    """Main preprocessing workflow window."""

    # Emitted at the end so MainWindow can refresh anything that
    # depends on the produced files (currently no-op).
    batch_completed = Signal(list)         # list[dict] result list

    def __init__(self, parent: Optional[QWidget] = None,
                 start_dir: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Preprocess TDT data")
        self.resize(1100, 760)
        # Modal-like: stay on top of the main window.
        self.setWindowModality(Qt.ApplicationModal)
        self._start_dir = start_dir or ""

        # Per-step state.
        self._folders: list[Path] = []
        # Cached per-folder streams so we don't re-pay 2 min on each
        # step. Built when we first enter step 3/4.
        self._streams_cache: dict[Path, dict[str, StreamInfo]] = {}
        # animal_id → channel_assignment dict, decided in step 4
        self._channel_assignments: dict[str, dict] = {}
        # animal_id → notch dict, decided in step 4
        self._notch_settings: dict[str, dict] = {}
        # Channel-consistency answer for THIS batch (step 3)
        self._channels_consistent: bool = False
        # The final plan that step 6 executes
        self._plan: list[BatchPlanItem] = []
        # Already-processed prompt state — populated in step 5.
        # Mirrors per-row decisions: "process" | "skip" | "suffix:_v2".
        self._already_processed_decisions: dict[int, str] = {}

        # Background worker state
        self._worker_thread: Optional[QThread] = None
        self._worker: Optional[PreprocessWorker] = None

        # UI scaffolding
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(12, 12, 12, 6)

        # Title strip — shows current step name.
        self._title_label = QLabel()
        self._title_label.setStyleSheet(
            "font-size: 14pt; font-weight: bold; padding: 4px;"
        )
        outer.addWidget(self._title_label)

        # Stacked content
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_step_select())            # 0
        self._stack.addWidget(self._build_step_configure())         # 1
        self._stack.addWidget(self._build_step_consistency())       # 2
        self._stack.addWidget(self._build_step_review())            # 3
        self._stack.addWidget(self._build_step_confirm())           # 4
        self._stack.addWidget(self._build_step_progress())          # 5
        self._stack.addWidget(self._build_step_report())            # 6
        outer.addWidget(self._stack, stretch=1)

        # Bottom navigation strip
        nav = QHBoxLayout()
        self._back_btn = QPushButton("◀ Back")
        self._back_btn.clicked.connect(self._on_back)
        nav.addWidget(self._back_btn)
        nav.addStretch(1)
        self._validation_label = QLabel("")
        self._validation_label.setStyleSheet("color: #f0c000; padding: 4px;")
        nav.addWidget(self._validation_label)
        self._next_btn = QPushButton("Next ▶")
        self._next_btn.setDefault(True)
        self._next_btn.clicked.connect(self._on_next)
        nav.addWidget(self._next_btn)
        outer.addLayout(nav)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

        # Initial step
        self._go_to_step(STEP_SELECT_FOLDERS)

    # ==================================================================
    # Step builders
    # ==================================================================

    def _build_step_select(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(QLabel(
            "Pick the TDT block folders to preprocess. You can add "
            "as many as you like — they'll all run as one batch."
        ))
        self._folders_list = QListWidget()
        v.addWidget(self._folders_list, stretch=1)
        btn_row = QHBoxLayout()
        self._pick_btn = QPushButton("➕ Pick folders…")
        self._pick_btn.clicked.connect(self._on_pick_folders)
        btn_row.addWidget(self._pick_btn)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._on_clear_folders)
        btn_row.addWidget(self._clear_btn)
        btn_row.addStretch(1)
        v.addLayout(btn_row)
        return w

    def _build_step_configure(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(QLabel(
            "Configure each folder: pick the condition, set the "
            "output name (used as the .mat filename prefix), and "
            "enter the animal ID. A ✓ in the Profile? column means "
            "we'll reuse the existing channel + notch settings for "
            "that animal."
        ))
        self._batch_table = TdtBatchTable()
        self._batch_table.row_changed.connect(self._refresh_nav_state)
        v.addWidget(self._batch_table, stretch=1)
        return w

    def _build_step_consistency(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(QLabel(
            "Were the same electrodes used across all the folders "
            "in this batch?"
        ))
        v.addSpacing(8)
        self._consistent_yes = QRadioButton(
            "Yes — same electrode placement. Ask once and apply to all."
        )
        self._consistent_no = QRadioButton(
            "No — placement varies. Ask per animal."
        )
        # Default to "Yes" — most batches in the lab share electrodes.
        self._consistent_yes.setChecked(True)
        v.addWidget(self._consistent_yes)
        v.addWidget(self._consistent_no)
        v.addSpacing(8)
        v.addWidget(QLabel(
            "If you're not sure, pick No — it just means a few more "
            "clicks (one channel-assignment dialog per animal ID)."
        ))
        v.addStretch(1)
        return w

    def _build_step_review(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(QLabel(
            "For each animal in the batch, review the channel "
            "assignment + notch parameters. Existing profiles are "
            "loaded as defaults — you can accept them or override."
        ))
        # Live list of which animals have been reviewed.
        self._review_status_list = QListWidget()
        v.addWidget(self._review_status_list, stretch=1)
        # Action buttons
        btn_row = QHBoxLayout()
        self._review_next_btn = QPushButton(
            "Review the next animal…"
        )
        self._review_next_btn.clicked.connect(self._on_review_next_animal)
        btn_row.addWidget(self._review_next_btn)
        btn_row.addStretch(1)
        v.addLayout(btn_row)
        return w

    def _build_step_confirm(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(QLabel(
            "Final check before running. Files that already exist "
            "in the output folder are flagged — for each, choose "
            "Skip, Re-process (overwrite), or Re-process with a "
            "suffix (saves to <name>_v2.mat etc.)."
        ))
        self._confirm_list = QListWidget()
        v.addWidget(self._confirm_list, stretch=1)
        # Summary
        self._confirm_summary = QLabel("")
        v.addWidget(self._confirm_summary)
        return w

    def _build_step_progress(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self._progress_label = QLabel("Starting…")
        self._progress_label.setStyleSheet("font-size: 11pt;")
        v.addWidget(self._progress_label)
        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setValue(0)
        v.addWidget(self._progress_bar)
        # Live per-file status list
        v.addWidget(QLabel("Per-file status:"))
        self._progress_list = QListWidget()
        v.addWidget(self._progress_list, stretch=1)
        # Cancel
        cancel_row = QHBoxLayout()
        cancel_row.addStretch(1)
        self._cancel_btn = QPushButton("Cancel after current file")
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        cancel_row.addWidget(self._cancel_btn)
        v.addLayout(cancel_row)
        return w

    def _build_step_report(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self._report_summary_label = QLabel("")
        self._report_summary_label.setStyleSheet("font-size: 11pt;")
        v.addWidget(self._report_summary_label)
        self._report_list = QListWidget()
        v.addWidget(self._report_list, stretch=1)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._open_output_btn = QPushButton("Open output folder…")
        self._open_output_btn.clicked.connect(self._on_open_output_folder)
        btn_row.addWidget(self._open_output_btn)
        self._done_btn = QPushButton("Done")
        self._done_btn.clicked.connect(self.close)
        btn_row.addWidget(self._done_btn)
        v.addLayout(btn_row)
        return w

    # ==================================================================
    # Navigation
    # ==================================================================

    _STEP_TITLES = {
        STEP_SELECT_FOLDERS: "Step 1 of 7 — Select folders",
        STEP_CONFIGURE_BATCH: "Step 2 of 7 — Configure batch",
        STEP_CONSISTENCY: "Step 3 of 7 — Channel consistency",
        STEP_REVIEW: "Step 4 of 7 — Per-animal review",
        STEP_CONFIRM: "Step 5 of 7 — Confirm",
        STEP_PROGRESS: "Step 6 of 7 — Processing",
        STEP_REPORT: "Step 7 of 7 — Report",
    }

    def _go_to_step(self, step: int) -> None:
        self._stack.setCurrentIndex(step)
        self._title_label.setText(self._STEP_TITLES.get(step, ""))
        # Step-entry hooks
        if step == STEP_REVIEW:
            self._enter_review_step()
        elif step == STEP_CONFIRM:
            self._enter_confirm_step()
        elif step == STEP_PROGRESS:
            self._enter_progress_step()
        # Update navigation button enablement / labels.
        self._refresh_nav_state()

    def _on_back(self) -> None:
        cur = self._stack.currentIndex()
        if cur == STEP_PROGRESS:
            # No back during processing.
            return
        if cur == 0:
            self.close()
            return
        self._go_to_step(cur - 1)

    def _on_next(self) -> None:
        cur = self._stack.currentIndex()
        if cur == STEP_SELECT_FOLDERS:
            if not self._folders:
                return
            # Hand the folder list to the table.
            self._batch_table.set_folders(self._folders)
            self._go_to_step(STEP_CONFIGURE_BATCH)
        elif cur == STEP_CONFIGURE_BATCH:
            msg = self._batch_table.validation_message()
            if msg:
                QMessageBox.warning(self, "Batch not ready", msg)
                return
            self._go_to_step(STEP_CONSISTENCY)
        elif cur == STEP_CONSISTENCY:
            self._channels_consistent = self._consistent_yes.isChecked()
            self._go_to_step(STEP_REVIEW)
        elif cur == STEP_REVIEW:
            if not self._all_animals_reviewed():
                resp = QMessageBox.question(
                    self, "Not all animals reviewed",
                    "Some animals still need a channel-assignment "
                    "review. Continue anyway? (Those rows will be "
                    "skipped in the batch.)",
                )
                if resp != QMessageBox.Yes:
                    return
            self._go_to_step(STEP_CONFIRM)
        elif cur == STEP_CONFIRM:
            self._go_to_step(STEP_PROGRESS)
        elif cur == STEP_PROGRESS:
            # During processing, "Next" is disabled; this branch
            # is reached only after worker finishes.
            self._go_to_step(STEP_REPORT)
        elif cur == STEP_REPORT:
            self.close()

    def _refresh_nav_state(self) -> None:
        cur = self._stack.currentIndex()
        # Disable Next when the current step is incomplete.
        ok = False
        warning = ""
        if cur == STEP_SELECT_FOLDERS:
            ok = bool(self._folders)
            warning = "" if ok else "Pick at least one TDT folder."
        elif cur == STEP_CONFIGURE_BATCH:
            msg = self._batch_table.validation_message()
            ok = msg is None
            warning = msg or ""
        elif cur == STEP_CONSISTENCY:
            ok = True
        elif cur == STEP_REVIEW:
            ok = self._has_any_review()  # need at least one animal reviewed
            warning = ""
        elif cur == STEP_CONFIRM:
            ok = self._any_planned()
            warning = "" if ok else "All items marked Skip — nothing to do."
        elif cur == STEP_PROGRESS:
            ok = False
            warning = ""
        elif cur == STEP_REPORT:
            ok = True
        self._next_btn.setEnabled(ok)
        self._validation_label.setText(warning)
        # Customize the Next-button label per step.
        if cur == STEP_REPORT:
            self._next_btn.setText("Close")
        elif cur == STEP_PROGRESS:
            self._next_btn.setText("Working…")
        elif cur == STEP_CONFIRM:
            self._next_btn.setText("Start processing ▶")
        else:
            self._next_btn.setText("Next ▶")
        # Back button: hidden on first + during processing.
        self._back_btn.setVisible(
            cur not in (STEP_SELECT_FOLDERS, STEP_PROGRESS, STEP_REPORT)
        )

    # ==================================================================
    # Step 1 — Select folders
    # ==================================================================

    def _on_pick_folders(self) -> None:
        new_folders = pick_folders(
            self,
            title="Pick TDT block folders to preprocess",
            start_dir=self._start_dir,
            instructions=(
                "Add one TDT block folder at a time. Each picked "
                "folder will become one row in the next step."
            ),
        )
        if not new_folders:
            return
        for f in new_folders:
            if f in self._folders:
                continue
            self._folders.append(f)
            item = QListWidgetItem(f"{f.name}    ·    {f.parent}")
            item.setToolTip(str(f))
            self._folders_list.addItem(item)
        self._refresh_nav_state()

    def _on_clear_folders(self) -> None:
        self._folders = []
        self._folders_list.clear()
        self._refresh_nav_state()

    # ==================================================================
    # Step 4 — Per-animal review
    # ==================================================================

    def _animals_in_order(self) -> list[str]:
        return self._batch_table.animal_ids_in_order()

    def _has_any_review(self) -> bool:
        return any(
            a in self._channel_assignments and a in self._notch_settings
            for a in self._animals_in_order()
        )

    def _all_animals_reviewed(self) -> bool:
        animals = self._animals_in_order()
        if not animals:
            return False
        return all(
            a in self._channel_assignments and a in self._notch_settings
            for a in animals
        )

    def _enter_review_step(self) -> None:
        """Pre-populate from existing profiles, render the review list."""
        self._review_status_list.clear()
        for animal_id in self._animals_in_order():
            prof = Profile.load(animal_id)
            if prof is not None and prof.channel_assignment and prof.notch:
                # Pre-fill from existing profile so users can skip
                # the review for this animal.
                self._channel_assignments[animal_id] = dict(prof.channel_assignment)
                self._notch_settings[animal_id] = dict(prof.notch)
            row_text = self._review_row_text(animal_id)
            item = QListWidgetItem(row_text)
            item.setData(Qt.UserRole, animal_id)
            self._review_status_list.addItem(item)
        # If the batch picked "Channels consistent? Yes" and there's
        # only one animal needing review, the wording is a bit
        # different but mechanics are the same — one click per animal.

    def _review_row_text(self, animal_id: str) -> str:
        has_ca = animal_id in self._channel_assignments
        has_nch = animal_id in self._notch_settings
        if has_ca and has_nch:
            ca = self._channel_assignments[animal_id]
            n_ch = len(ca.get("channels", []))
            n_harm = len(self._notch_settings[animal_id].get(
                "frequencies_filtered", []
            ))
            return (f"  ✓  {animal_id}  —  {n_ch} channels · "
                    f"{n_harm} notch harmonics  (click below to re-review)")
        return f"  ☐  {animal_id}  —  needs review"

    def _on_review_next_animal(self) -> None:
        """Walk through animals in batch order, reviewing each.

        If channel-consistency is "Yes", the channel-assignment
        dialog only opens for the very first animal — subsequent
        animals reuse that assignment and only the notch dialog is
        shown. If "No", both dialogs run per animal.
        """
        # Find the next animal that doesn't have BOTH dicts populated
        # (i.e. the user hasn't accepted them in this session — pre-
        # filled profile data counts as accepted).
        animals = self._animals_in_order()
        target = None
        for a in animals:
            if a not in self._channel_assignments or a not in self._notch_settings:
                target = a
                break
        if target is None:
            # Allow re-review of the currently-selected row.
            current = self._review_status_list.currentItem()
            if current is None:
                QMessageBox.information(
                    self, "All animals reviewed",
                    "Every animal has a channel + notch decision. "
                    "Click an entry above and press this button again "
                    "to re-review.",
                )
                return
            target = current.data(Qt.UserRole)
        # Pick the first batch row whose animal_id matches `target` —
        # that's the TDT folder we'll preview from.
        sample_folder = self._first_folder_for_animal(target)
        if sample_folder is None:
            QMessageBox.warning(
                self, "No folder found",
                f"Couldn't find a folder for animal {target!r}.",
            )
            return
        streams = self._streams_for(sample_folder)
        if streams is None:
            return  # error already shown

        # Channel-assignment dialog
        if self._channels_consistent and target != animals[0] and (
            animals[0] in self._channel_assignments
        ):
            # Reuse the first animal's assignment.
            ca = dict(self._channel_assignments[animals[0]])
        else:
            existing = self._channel_assignments.get(target)
            if not existing:
                prof = Profile.load(target)
                existing = (prof.channel_assignment if prof else None)
            ca_dlg = ChannelAssignmentDialog(
                animal_id=target,
                streams=streams,
                existing_assignment=existing,
                parent=self,
            )
            if ca_dlg.exec() != QDialog.Accepted:
                return
            ca = ca_dlg.channel_assignment()
        self._channel_assignments[target] = ca

        # Notch review dialog. Uses the channels just accepted.
        existing_notch = self._notch_settings.get(target)
        if existing_notch is None:
            prof = Profile.load(target)
            if prof is not None and prof.notch:
                existing_notch = dict(prof.notch)
        notch_dlg = NotchReviewDialog(
            animal_id=target,
            tdt_folder=sample_folder,
            raw_stream=ca["raw_stream"],
            channels=ca["channels"],
            existing_notch=existing_notch,
            parent=self,
        )
        if notch_dlg.exec() != QDialog.Accepted:
            # User cancelled the notch step — don't claim a partial
            # review for this animal.
            return
        self._notch_settings[target] = notch_dlg.notch_settings()

        # Persist the profile right now, so a later batch can reuse
        # it even if processing is cancelled mid-run.
        try:
            prof = Profile.from_review_session(
                animal_id=target,
                channel_assignment=self._channel_assignments[target],
                notch=self._notch_settings[target],
            )
            prof.save()
        except Exception as exc:                                  # pragma: no cover
            QMessageBox.warning(
                self, "Profile save failed",
                f"Couldn't save profile for {target}:\n{exc}",
            )

        self._refresh_review_list()
        self._refresh_nav_state()

    def _refresh_review_list(self) -> None:
        for i in range(self._review_status_list.count()):
            item = self._review_status_list.item(i)
            animal_id = item.data(Qt.UserRole)
            item.setText(self._review_row_text(animal_id))

    def _first_folder_for_animal(self, animal_id: str) -> Optional[Path]:
        for row in self._batch_table.rows:
            if row.animal_id == animal_id:
                return row.folder
        return None

    def _streams_for(self, folder: Path) -> Optional[dict[str, StreamInfo]]:
        """Cached `list_streams_cached` wrapper with a busy cursor +
        error message on failure."""
        if folder in self._streams_cache:
            return self._streams_cache[folder]
        self.statusBar().showMessage(
            f"Reading streams from {folder.name} (first call may "
            "take ~2 min for an uncached block)…",
        )
        try:
            streams = list_streams_cached(folder)
        except Exception as exc:
            self.statusBar().clearMessage()
            QMessageBox.critical(
                self, "Failed to read TDT block",
                f"Couldn't list streams in {folder}:\n{exc}",
            )
            return None
        self.statusBar().clearMessage()
        self._streams_cache[folder] = streams
        return streams

    # ==================================================================
    # Step 5 — Confirm + already-processed detection
    # ==================================================================

    def _existing_outputs_for(self, row: BatchRowState) -> list[Path]:
        """Return any existing output files (sig/notched/meta) for
        this row in its target folder."""
        out_dir = row.folder
        prefix = row.output_condition_name.strip()
        suffixes = ("_sig.mat", "_notched.mat", "_meta.json")
        out = []
        for s in suffixes:
            p = out_dir / f"{prefix}{s}"
            if p.exists():
                out.append(p)
        return out

    def _enter_confirm_step(self) -> None:
        self._confirm_list.clear()
        self._already_processed_decisions.clear()
        planned = 0
        skipped = 0
        suffixed = 0
        for idx, row in enumerate(self._batch_table.rows):
            if (row.animal_id not in self._channel_assignments
                    or row.animal_id not in self._notch_settings):
                txt = (f"[skip — no review] {row.output_condition_name}"
                       f"  ({row.folder.name})")
                item = QListWidgetItem(txt)
                item.setForeground(Qt.gray)
                self._confirm_list.addItem(item)
                self._already_processed_decisions[idx] = "skip"
                skipped += 1
                continue
            existing = self._existing_outputs_for(row)
            if existing:
                # Offer skip / overwrite / suffix.
                dlg = AlreadyProcessedDialog(
                    row, existing, parent=self,
                )
                if dlg.exec() == QDialog.Accepted:
                    decision = dlg.decision()
                else:
                    decision = "skip"
                self._already_processed_decisions[idx] = decision
                if decision == "skip":
                    txt = (f"[skip — already processed] "
                           f"{row.output_condition_name}  ({row.folder.name})")
                    self._confirm_list.addItem(
                        self._confirm_item(txt, Qt.gray)
                    )
                    skipped += 1
                elif decision.startswith("suffix:"):
                    suffix = decision.split(":", 1)[1]
                    new_name = row.output_condition_name + suffix
                    txt = (f"[re-process as '{new_name}'] "
                           f"{row.folder.name}")
                    self._confirm_list.addItem(self._confirm_item(txt))
                    suffixed += 1
                    planned += 1
                else:
                    txt = (f"[overwrite existing] "
                           f"{row.output_condition_name}  ({row.folder.name})")
                    self._confirm_list.addItem(self._confirm_item(txt))
                    planned += 1
            else:
                self._already_processed_decisions[idx] = "process"
                txt = (f"[process] {row.output_condition_name}  "
                       f"({row.folder.name})")
                self._confirm_list.addItem(self._confirm_item(txt))
                planned += 1
        self._confirm_summary.setText(
            f"<b>{planned}</b> to process · "
            f"<span style='color:#aaa'>{suffixed} with suffix</span> · "
            f"<span style='color:#666'>{skipped} skipped</span>"
        )
        self._refresh_nav_state()

    def _confirm_item(self, text: str,
                       color=None) -> QListWidgetItem:
        item = QListWidgetItem(text)
        if color is not None:
            item.setForeground(color)
        return item

    def _any_planned(self) -> bool:
        return any(
            d != "skip" and not d.startswith("nodone")
            for d in self._already_processed_decisions.values()
        )

    def _build_final_plan(self) -> list[BatchPlanItem]:
        """Translate the (table + per-animal review + already-processed
        decisions) state into a flat list of BatchPlanItem for the
        worker.

        Items the user chose to Skip are still emitted as plan items
        with `skip_reason` set so the worker reports them in the
        result list — easier than filtering twice.
        """
        plan: list[BatchPlanItem] = []
        for idx, row in enumerate(self._batch_table.rows):
            decision = self._already_processed_decisions.get(idx, "process")
            output_name = row.output_condition_name.strip()
            skip_reason: Optional[str] = None
            if decision == "skip":
                skip_reason = "user-skip"
            elif decision.startswith("suffix:"):
                output_name = output_name + decision.split(":", 1)[1]
            if row.animal_id not in self._channel_assignments:
                skip_reason = "no channel assignment"
            elif row.animal_id not in self._notch_settings:
                skip_reason = "no notch review"
            profile = Profile.from_review_session(
                animal_id=row.animal_id,
                channel_assignment=self._channel_assignments.get(
                    row.animal_id, {}
                ),
                notch=self._notch_settings.get(row.animal_id, {}),
            )
            plan.append(BatchPlanItem(
                tdt_folder=row.folder,
                output_folder=row.folder,
                animal_id=row.animal_id,
                condition=row.condition or "unknown",
                output_condition_name=output_name,
                profile=profile,
                skip_reason=skip_reason,
            ))
        return plan

    # ==================================================================
    # Step 6 — Progress
    # ==================================================================

    def _enter_progress_step(self) -> None:
        self._plan = self._build_final_plan()
        self._progress_list.clear()
        total = sum(1 for it in self._plan if it.skip_reason is None)
        if total == 0:
            QMessageBox.information(
                self, "Nothing to process",
                "All items are marked Skip. Returning to Confirm.",
            )
            self._go_to_step(STEP_CONFIRM)
            return
        self._progress_bar.setMaximum(len(self._plan))
        self._progress_bar.setValue(0)
        self._progress_label.setText(
            f"Processing {total} of {len(self._plan)} item(s)…"
        )

        # Spin up worker
        self._worker_thread = QThread(self)
        self._worker = PreprocessWorker(self._plan)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.file_started.connect(self._on_worker_file_started)
        self._worker.file_completed.connect(self._on_worker_file_completed)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.cancelled.connect(self._on_worker_cancelled)
        # Auto-cleanup
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.error.connect(self._worker_thread.quit)
        self._worker.cancelled.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.start()

    def _on_worker_file_started(self, idx: int, total: int,
                                  src_name: str) -> None:
        self._progress_label.setText(
            f"({idx + 1}/{total}) {src_name}…"
        )

    def _on_worker_file_completed(self, idx: int, total: int,
                                    result: dict) -> None:
        item = result.get("item")
        status = result.get("status", "?")
        reason = result.get("reason") or ""
        prefix = ""
        color = None
        if status == "ok":
            prefix = "✓"
        elif status == "skipped":
            prefix = "⊘"
            color = Qt.gray
        else:
            prefix = "✗"
            color = Qt.red
        out_name = (item.output_condition_name if item is not None else "?")
        text = f"{prefix} {out_name} — {status}"
        if reason:
            text += f"  ({reason})"
        li = QListWidgetItem(text)
        if color is not None:
            li.setForeground(color)
        self._progress_list.addItem(li)
        self._progress_list.scrollToBottom()

    def _on_worker_progress(self, done: int, total: int,
                              fraction: float) -> None:
        self._progress_bar.setValue(int(done))

    def _on_worker_finished(self, results: list) -> None:
        self._finalize_results(results, cancelled=False)

    def _on_worker_cancelled(self, partial: list) -> None:
        self._finalize_results(partial, cancelled=True)

    def _on_worker_error(self, message: str) -> None:
        QMessageBox.critical(
            self, "Preprocessing failed",
            f"The batch worker errored:\n{message}",
        )
        self._finalize_results([], cancelled=True)

    def _on_cancel_clicked(self) -> None:
        if self._worker is None:
            return
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setText("Cancelling after current file…")
        self._worker.request_cancel()

    def _finalize_results(self, results: list, *, cancelled: bool) -> None:
        # Move to the report step.
        self._results = results
        self._report_cancelled = cancelled
        self._enter_report_step()
        self._go_to_step(STEP_REPORT)
        self.batch_completed.emit(results)

    # ==================================================================
    # Step 7 — Report
    # ==================================================================

    def _enter_report_step(self) -> None:
        results = getattr(self, "_results", [])
        cancelled = getattr(self, "_report_cancelled", False)
        n_ok = sum(1 for r in results if r.get("status") == "ok")
        n_skip = sum(1 for r in results if r.get("status") == "skipped")
        n_err = sum(1 for r in results if r.get("status") == "error")
        header = "Cancelled. " if cancelled else "Completed. "
        self._report_summary_label.setText(
            header +
            f"<b>{n_ok}</b> ok · "
            f"<span style='color:#aaa'>{n_skip} skipped</span> · "
            f"<span style='color:#ff6b6b'>{n_err} error</span>"
        )
        self._report_list.clear()
        for r in results:
            item = r.get("item")
            status = r.get("status", "?")
            reason = r.get("reason") or ""
            warnings_list = r.get("validation_warnings") or []
            outputs = r.get("output_paths") or {}
            out_name = (item.output_condition_name
                         if item is not None else "?")
            head = f"{status.upper():>8}  {out_name}"
            if reason:
                head += f"  ({reason})"
            if outputs:
                head += "  →  " + ", ".join(p.name for p in outputs.values())
            for w in warnings_list:
                head += f"\n          ⚠ {w}"
            li = QListWidgetItem(head)
            if status == "error":
                li.setForeground(Qt.red)
            elif status == "skipped":
                li.setForeground(Qt.gray)
            self._report_list.addItem(li)

    def _on_open_output_folder(self) -> None:
        # Pick any output folder from the results — they may differ
        # per row if the user customized output_folder.
        results = getattr(self, "_results", [])
        for r in results:
            outputs = r.get("output_paths") or {}
            for p in outputs.values():
                self._open_in_finder(p.parent)
                return
        QMessageBox.information(
            self, "No outputs",
            "No outputs were produced — nothing to open.",
        )

    def _open_in_finder(self, path: Path) -> None:
        """Reveal `path` in the OS file manager."""
        import subprocess
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            elif sys.platform == "win32":
                subprocess.Popen(["explorer", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception:                                         # pragma: no cover
            pass

    # ==================================================================
    # Cleanup
    # ==================================================================

    def closeEvent(self, event) -> None:
        # If the worker is still running, ask before closing.
        if self._worker_thread is not None and self._worker_thread.isRunning():
            resp = QMessageBox.question(
                self, "Cancel preprocessing?",
                "A batch is still running. Cancel it and close?",
            )
            if resp != QMessageBox.Yes:
                event.ignore()
                return
            if self._worker is not None:
                self._worker.request_cancel()
            # Wait briefly for graceful exit
            if not self._worker_thread.wait(8000):
                self._worker_thread.terminate()
                self._worker_thread.wait(2000)
        super().closeEvent(event)


class AlreadyProcessedDialog(QDialog):
    """Tiny modal asking what to do with already-processed outputs.

    Pops up once per row in step 5 if the row's target output folder
    already contains `<prefix>_sig.mat` / `_notched.mat` / `_meta.json`.
    The choice is recorded into PreprocessWindow's
    `_already_processed_decisions` map, which `_build_final_plan`
    consults.
    """

    def __init__(
        self,
        row: BatchRowState,
        existing_outputs: list[Path],
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Already processed")
        self.resize(560, 280)
        outer = QVBoxLayout(self)
        outer.addWidget(QLabel(
            f"<b>{row.output_condition_name}</b> already has outputs "
            f"in {row.folder}:"
        ))
        files_label = QLabel(
            "<br>".join(f"&nbsp;&nbsp;• {p.name}" for p in existing_outputs)
        )
        files_label.setStyleSheet(
            "font-family: monospace; padding: 4px; color: #ccc;"
        )
        outer.addWidget(files_label)
        outer.addSpacing(8)
        self._skip_btn = QRadioButton("Skip (leave existing files alone)")
        self._skip_btn.setChecked(True)
        outer.addWidget(self._skip_btn)
        self._overwrite_btn = QRadioButton(
            "Re-process (overwrite the existing files)"
        )
        outer.addWidget(self._overwrite_btn)
        self._suffix_btn = QRadioButton(
            "Re-process with suffix — saves as <name>_v2_sig.mat, etc."
        )
        outer.addWidget(self._suffix_btn)
        outer.addStretch(1)

        bb = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        outer.addWidget(bb)

    def decision(self) -> str:
        if self._overwrite_btn.isChecked():
            return "process"
        if self._suffix_btn.isChecked():
            return "suffix:_v2"
        return "skip"
