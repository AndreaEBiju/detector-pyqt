"""Candidate adjudication (task 16 Change 1): a queue of cores, one keystroke each.

The primary labelling mode from here on: instead of marking intervals freely, the
labeller is shown one candidate CORE at a time - the raw channels around it (detection
reads every raw contact, invariant 6), the core highlighted, the queue's other cores in
the same recording shaded for context - and judges it with one key:

    1  motion        2  physiology (not motion)        3  unsure        4  line noise

``4`` is ruling 2026-10-07 (c) item 3: line noise is not motion and counts as a
negative. ``Space`` skips (the core stays unjudged and comes round again);
``Ctrl+Z`` / ``Backspace`` undoes the last judgement (up to the 20 still unwritten);
``Home`` re-centres; ``+`` / ``-`` widen or narrow the context.

**Widening a boundary** ("where the extent is visibly wrong", task 16 Change 1):
``Shift+drag`` on the plot widens the current core's boundary to cover the drag (the
union with the core and any earlier drag, clipped to the core's region), drawn in red.
The judged unit stays the core (R3): the widened span is stored as
``widened_start_s`` / ``widened_stop_s`` beside it, never instead of it. A widened
boundary goes only with **motion**: ``1`` records it; ``2`` / ``3`` / ``4`` are refused
while one is pending (clear it first). ``Esc`` clears it; ``Ctrl+Z`` clears a pending
widen first, then undoes judgements as before - and undoing a motion judgement withdraws
its widened boundary with it.

Band z-traces (the maximum over every signal, per band, with ``z_enter``) are
available per recording but OFF by default: they mean running the detection chain over
the core's whole assessable region (Night 1 measured about three minutes and several GB
per new-cohort recording), so they are computed in the background only when ticked.

Judgements go to the store as per-user write-once shards (``ui.adjudicate.judgements``);
the queue schema is in ``ui.adjudicate.queue``.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from gems_blanking_v2.io.store import GemsStore
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.adjudicate import loaders
from ui.adjudicate.judgements import KEY_TO_JUDGEMENT, judgement_for_key
from ui.adjudicate.session import AdjudicationSession
from ui.audit import bridge
from ui.audit.array_recording import ArrayRecording
from ui.audit.controller import seconds_to_samples
from ui.widgets.signal_viewer import MultiChannelViewer
from ui.widgets.ztrace_dock import ZTraceDock

LoadFn = Callable[[dict[str, Any]], Any]
"""Queue row -> a ``Recording`` (``fs``, ``data`` in µV (n, ch), ``channels``)."""
TracesFn = Callable[[Any, tuple[float, float]], tuple[np.ndarray, list[Any]]]

CONTEXT_S = 2.0
"""Seconds shown either side of the core by default (the review panel's ±2 s)."""
CONTEXT_STEPS = (0.5, 1.0, 2.0, 5.0, 10.0, 30.0)

_LABELS = {"motion": "1  Motion", "physiology": "2  Physiology (not motion)",
           "unsure": "3  Unsure", "line_noise": "4  Line noise (not motion)"}
_COLOURS = ("#4c9be8", "#6bb3f0", "#9ccbf5", "#e8834c", "#f0a06b", "#f5bf9c",
            "#5cc98a", "#7fd6a3", "#a6e3bf")


def _breakable(name: str) -> str:
    """Let a long underscore-joined id wrap in a label (zero-width space after ``_``)."""
    return name.replace("_", "_" + chr(0x200B))


class _TraceSignals(QObject):
    done = Signal(str, object, object)  # recording key, intervals, traces
    failed = Signal(str, str)


class _TraceJob(QRunnable):
    def __init__(self, key: str, fn: TracesFn, recording: Any,
                 region: tuple[float, float]) -> None:
        super().__init__()
        self.key, self.fn, self.recording, self.region = key, fn, recording, region
        self.signals = _TraceSignals()
        self.setAutoDelete(False)  # the window keeps the reference; Qt must not delete it

    def run(self) -> None:
        try:
            intervals, traces = self.fn(self.recording, self.region)
        except Exception as exc:  # noqa: BLE001 - reported in the window
            self.signals.failed.emit(self.key, str(exc))
            return
        self.signals.done.emit(self.key, intervals, traces)


class AdjudicationWindow(QMainWindow):
    """Shows one queue core at a time and records one judgement per keystroke."""

    def __init__(self, session: AdjudicationSession, *, load_fn: LoadFn,
                 traces_fn: TracesFn | None = None, async_traces: bool = True,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Candidate adjudication - {session.queue_file}")
        self.session = session
        self._load_fn = load_fn
        self._traces_fn: TracesFn = traces_fn or bridge.reveal_for_region
        self._async_traces = async_traces
        self._rec_id: str | None = None
        self._recording: Any = None
        self._traces_cache: dict[str, list[Any]] = {}
        self._traces_running: set[str] = set()
        self._jobs: list[_TraceJob] = []
        self._context = CONTEXT_S
        self._core_items: list[pg.LinearRegionItem] = []
        self._widen: tuple[float, float] | None = None
        self._widen_items: list[pg.LinearRegionItem] = []
        # The core_key actually on screen. Set only once its recording is loaded, its
        # details written and its band drawn; a judging key is refused unless it equals
        # the current core's key, so nothing can label a core the labeller never saw.
        self._shown_key: str | None = None
        self.viewer: MultiChannelViewer | None = None
        self.last_error: str | None = None

        # -- left: the core, the keys, progress
        self.core_info = QLabel("")
        self.core_info.setWordWrap(True)
        self.core_info.setTextFormat(Qt.RichText)
        self.progress = QLabel("")
        self.progress.setWordWrap(True)
        self.progress.setTextFormat(Qt.RichText)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self._buttons: dict[str, QPushButton] = {}
        keys = QWidget()
        klay = QVBoxLayout(keys)
        for key, judgement in KEY_TO_JUDGEMENT.items():
            b = QPushButton(_LABELS[judgement])
            b.clicked.connect(self._slot(lambda k=key: self.press(k)))
            klay.addWidget(b)
            self._buttons[key] = b
        self._skip_btn = QPushButton("Space  Skip (stays unjudged)")
        self._undo_btn = QPushButton("Ctrl+Z  Undo last judgement")
        self._save_btn = QPushButton("Write judgements to the store now")
        self._skip_btn.clicked.connect(self._slot(self.skip))
        self._undo_btn.clicked.connect(self._slot(self.undo))
        self._save_btn.clicked.connect(self._slot(self.save_now))
        for b in (self._skip_btn, self._undo_btn, self._save_btn):
            klay.addWidget(b)
        # No button ever takes focus: a focused button also fires on Space, which is
        # the skip key, and a click followed by Space would judge the next core.
        for b in (*self._buttons.values(), self._skip_btn, self._undo_btn, self._save_btn):
            b.setFocusPolicy(Qt.NoFocus)
        self.traces_box = QCheckBox("Band z-traces (slow)")
        self.traces_box.setToolTip("Runs the detection chain over the core's whole "
                                   "assessable region, in the background, once per region.")
        self.traces_box.toggled.connect(self._slot(self._on_traces_toggled))
        self.traces_box.setFocusPolicy(Qt.NoFocus)  # Space toggles a focused checkbox
        left = QWidget()
        # Fixed-width side panel: the plot is what is being judged and gets the rest.
        left.setFixedWidth(360)
        lay = QVBoxLayout(left)
        lay.addWidget(self.core_info)
        lay.addWidget(keys)
        lay.addWidget(self.traces_box)
        lay.addWidget(self.progress)
        lay.addWidget(self.status)
        lay.addStretch(1)

        self._centre = QWidget()
        self._centre_lay = QVBoxLayout(self._centre)
        self._placeholder = QLabel("")
        self._centre_lay.addWidget(self._placeholder)
        root = QWidget()
        rlay = QHBoxLayout(root)
        rlay.addWidget(left, 0)
        rlay.addWidget(self._centre, 1)
        self.setCentralWidget(root)

        self.ztrace = ZTraceDock()
        dock = QDockWidget("Band z-traces (max over all signals)", self)
        dock.setWidget(self.ztrace)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self._trace_dock = dock
        dock.hide()  # shown only while z-traces are ticked; the plot gets the room

        for key in KEY_TO_JUDGEMENT:
            QShortcut(QKeySequence(key), self, activated=self._slot(lambda k=key: self.press(k)))
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._slot(self.skip))
        QShortcut(QKeySequence.Undo, self, activated=self._slot(self.undo))
        QShortcut(QKeySequence(Qt.Key_Backspace), self, activated=self._slot(self.undo))
        QShortcut(QKeySequence(Qt.Key_Home), self, activated=self._slot(self.recentre))
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self._slot(self.clear_widen))
        QShortcut(QKeySequence(Qt.Key_Plus), self, activated=self._slot(lambda: self.zoom(+1)))
        QShortcut(QKeySequence(Qt.Key_Equal), self, activated=self._slot(lambda: self.zoom(+1)))
        QShortcut(QKeySequence(Qt.Key_Minus), self, activated=self._slot(lambda: self.zoom(-1)))

        if session.restored:
            self.status.setText(f"Restored {session.restored} judgement(s) from the local "
                                "journal that had not reached the store yet.")
        self.show_current()  # a recording that cannot load is reported, not raised

    # -- plumbing -------------------------------------------------------------

    def _slot(self, action: Callable[[], Any]) -> Callable[..., None]:
        """Call ``action()`` with no arguments; show any failure on screen."""
        def slot(*_ignored: object) -> None:
            try:
                action()
            except Exception as exc:  # noqa: BLE001 - shown, not swallowed
                self._report(exc)
        return slot

    def _report(self, exc: Exception) -> None:
        """Put a failure on screen (status line and a non-modal box) and remember it."""
        self.last_error = str(exc)
        self.status.setText(f"Could not do that: {exc}")
        box = QMessageBox(QMessageBox.Warning, "Candidate adjudication", str(exc),
                          parent=self)
        box.setModal(False)
        box.show()

    # -- actions --------------------------------------------------------------

    def press(self, key: str) -> dict[str, Any] | None:
        """A judging keystroke. Unknown keys do nothing.

        Refused unless the current core is the one on screen (:attr:`_shown_key`): after
        a recording fails to load, the keys must not label a core nobody has seen.
        """
        before = self.session.current()
        if before is None or before["core_key"] != self._shown_key:
            if before is not None:
                self.status.setText("This core is not on screen (its recording did not "
                                    "load), so it cannot be judged. Space skips it.")
            return None
        judgement = judgement_for_key(key)
        if judgement is None:
            return None
        if self._widen is not None and judgement != "motion":
            self.status.setText(f"A widened boundary goes only with motion: press 1 to record "
                                f"motion with it, or Esc to clear it before judging {judgement}.")
            return None
        rec = self.session.judge_key(key, widened=self._widen)
        if rec is None:
            return None
        where = f"{before['recording']} {float(before['start_s']):.3f} s" if before else ""
        self.status.setText(f"{rec['judgement']}: {where}")
        self.show_current()
        return rec

    def skip(self) -> None:
        """Move on, leaving the current core unjudged."""
        self.session.skip()
        self.show_current()

    def undo(self) -> None:
        """Clear a pending widen; otherwise withdraw the newest unwritten judgement.

        Undoing a motion judgement withdraws its widened boundary with it (one record).
        """
        if self._widen is not None:
            self.clear_widen()
            return
        rec = self.session.undo()
        if rec is None:
            self.status.setText("Nothing to undo: every judgement so far is already "
                                "written to the store (shards are never edited).")
            return
        self.status.setText(f"Undid {rec['judgement']} on {rec['recording']} "
                            f"{rec['start_s']:.3f}-{rec['stop_s']:.3f} s.")
        self.show_current()

    def save_now(self) -> list[Path]:
        """Write every pending judgement (undo cannot reach them afterwards)."""
        paths = self.session.flush(force=True)
        self.status.setText(f"Wrote {len(paths)} shard(s); nothing pending.")
        self._refresh_progress()
        return paths

    def zoom(self, direction: int) -> None:
        """Step the context half-width through :data:`CONTEXT_STEPS`."""
        i = min(range(len(CONTEXT_STEPS)), key=lambda k: abs(CONTEXT_STEPS[k] - self._context))
        i = max(0, min(len(CONTEXT_STEPS) - 1, i + direction))
        self._context = CONTEXT_STEPS[i]
        self.recentre()

    def recentre(self) -> None:
        """Put the current core back in the middle of the view."""
        row = self.session.current()
        if row is None or self.viewer is None:
            return
        dur = self.viewer.recording.duration_sec
        lo = max(0.0, float(row["start_s"]) - self._context)
        hi = min(dur, float(row["stop_s"]) + self._context)
        self.viewer.set_viewport(lo, hi)
        self._set_trace_range(lo, hi)

    # -- widening ------------------------------------------------------------

    def _on_widen_drag(self, lo: float, hi: float) -> None:
        try:
            self.widen(lo, hi)
        except Exception as exc:  # noqa: BLE001 - shown, not swallowed
            self._report(exc)

    def widen(self, lo: float, hi: float) -> tuple[float, float] | None:
        """Widen the on-screen core's boundary to cover ``[lo, hi)`` (recording seconds).

        The pending widen is the union of the core, any earlier drag and this one,
        clipped to the core's region. Returns it, or ``None`` if no core is on screen.
        """
        row = self.session.current()
        if row is None or row["core_key"] != self._shown_key:
            self.status.setText("No core is on screen to widen.")
            return None
        r0, r1 = float(row["region_start_s"]), float(row["region_stop_s"])
        a = min(float(row["start_s"]), max(r0, min(lo, hi)))
        b = max(float(row["stop_s"]), min(r1, max(lo, hi)))
        if self._widen is not None:
            a, b = min(a, self._widen[0]), max(b, self._widen[1])
        self._widen = (a, b)
        self._draw_widen()
        self.status.setText(f"Widened to {a:.3f}-{b:.3f} s. Press 1 to record motion with "
                            "it; Esc clears it.")
        return self._widen

    def clear_widen(self) -> None:
        """Drop the pending widened boundary (nothing was recorded)."""
        had = self._widen is not None
        self._widen = None
        self._draw_widen()
        if had:
            self.status.setText("Cleared the widened boundary.")

    @property
    def pending_widen(self) -> tuple[float, float] | None:
        """The widened boundary waiting for its motion judgement, if any."""
        return self._widen

    def _draw_widen(self) -> None:
        if self.viewer is not None:
            for (plot, _c), item in zip(self.viewer.plots, self._widen_items, strict=False):
                plot.removeItem(item)
        self._widen_items = []
        if self._widen is None or self.viewer is None:
            return
        for plot, _curve in self.viewer.plots:
            item = pg.LinearRegionItem(values=self._widen, orientation="vertical",
                                       movable=False, brush=pg.mkBrush(214, 39, 40, 40),
                                       pen=pg.mkPen("#d62728", width=1))
            item.setZValue(-6)
            plot.addItem(item)
            self._widen_items.append(item)

    # -- display --------------------------------------------------------------

    def _set_judging_enabled(self, enabled: bool) -> None:
        for b in self._buttons.values():
            b.setEnabled(enabled)

    def show_current(self) -> None:
        """Load (if needed) and display the current core; or say the queue is done.

        The judging keys and buttons are live only once the core is fully on screen; a
        load that fails leaves them disabled (Space still skips past it).
        """
        self._shown_key = None
        self._widen = None  # a widen belongs to the core it was drawn on
        self._draw_widen()
        self._set_judging_enabled(False)
        self._refresh_progress()
        row = self.session.current()
        if row is None:
            # Written on close (or "Write judgements now"), not here, so the last
            # judgements stay undoable until the labeller is done.
            self.core_info.setText("<b>Queue finished.</b> Every core has a judgement. "
                                   "Close the window to write the rest to the store.")
            self._skip_btn.setEnabled(False)
            return
        self._skip_btn.setEnabled(True)
        self._undo_btn.setEnabled(self.session.can_undo())
        self.core_info.setText(f"<b>Loading</b> {_breakable(str(row['recording']))} ...")
        try:
            self._ensure_recording(row)
            self._describe(row)
            self._highlight(row)
        except Exception as exc:  # noqa: BLE001 - shown; the keys stay disabled
            self.core_info.setText(f"<b>Could not show this core</b> "
                                   f"({_breakable(str(row['recording']))}). Space skips it.")
            self._report(exc)
            return
        self._shown_key = row["core_key"]
        self._set_judging_enabled(True)
        self.recentre()
        self._show_traces(row)

    def _ensure_recording(self, row: dict[str, Any]) -> None:
        rid = str(row["recording"])
        if rid == self._rec_id and self.viewer is not None:
            return
        # One recording in memory at a time: a new-cohort file is ~2 GB as float64.
        if self.viewer is not None:
            self.viewer.setParent(None)
            self.viewer.deleteLater()
            self.viewer = None
        self._recording = None
        self._rec_id = None
        gc.collect()
        self.status.setText(f"Loading {rid} ...")
        recording = self._load_fn(row)
        names = tuple(c.name for c in recording.channels)
        self.viewer = MultiChannelViewer(ArrayRecording(recording.data, recording.fs),
                                         channel_names=names,
                                         channel_colors=_COLOURS[: len(names)])
        self.viewer.bad_interval_added.connect(self._on_widen_drag)
        self.viewer.follow_mark_drag = True  # a widen may run past the viewport
        self._placeholder.hide()
        self._centre_lay.insertWidget(0, self.viewer, 1)
        self._widen_items = []
        self._recording, self._rec_id = recording, rid
        self._core_items = []
        others = np.array([[r["start_s"], r["stop_s"]] for r in self.session.rows
                           if str(r["recording"]) == rid], dtype=np.float64).reshape(-1, 2)
        self.viewer.set_model_intervals(seconds_to_samples(others, float(recording.fs)))
        self.status.setText(f"Loaded {rid}.")

    def _highlight(self, row: dict[str, Any]) -> None:
        """The current core, drawn above everything on every channel."""
        assert self.viewer is not None
        for (plot, _c), item in zip(self.viewer.plots, self._core_items, strict=False):
            plot.removeItem(item)
        self._core_items = []
        for plot, _curve in self.viewer.plots:
            item = pg.LinearRegionItem(
                values=(float(row["start_s"]), float(row["stop_s"])), orientation="vertical",
                movable=False, brush=pg.mkBrush(255, 230, 0, 70),
                pen=pg.mkPen("#ffe600", width=1))
            item.setZValue(-5)
            plot.addItem(item)
            self._core_items.append(item)

    def _describe(self, row: dict[str, Any]) -> None:
        dur_ms = 1000.0 * (float(row["stop_s"]) - float(row["start_s"]))
        test = (" &nbsp;<span style='color:#d62728'><b>TEST SET</b> (never trains)</span>"
                if row["label_set"] == "test" else "")
        extra = []
        for col, fmt in (("score", "P(motion) {:.2f}"), ("peak_z", "peak z {:.1f}")):
            v = row.get(col)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                extra.append(fmt.format(float(v)))
        for col in ("peak_band", "peak_signal"):
            v = row.get(col)
            if isinstance(v, str) and v:
                extra.append(f"{col.split('_')[1]} {v}")
        judged, total = self.session.position()
        self.core_info.setText(
            f"<b>Core {judged + 1} of {total}</b>{test}<br>"
            f"animal <b>{row['animal']}</b> ({row['cohort']} cohort) - draw {row['draw']}<br>"
            f"{_breakable(str(row['recording']))}<br>"
            f"{float(row['start_s']):.3f} - {float(row['stop_s']):.3f} s ({dur_ms:.0f} ms)"
            + (f"<br>{', '.join(extra)}" if extra else ""))

    def _refresh_progress(self) -> None:
        lines = ["<b>Progress by animal</b> (judged / total - unjudged)"]
        for p in self.session.progress():
            lines.append(f"{p.animal} ({p.cohort}): {p.judged} / {p.total} - {p.unjudged}")
        c = self.session.counts()
        lines.append("motion {} - physiology {} - unsure {} - line noise {}".format(
            c.get("motion", 0), c.get("physiology", 0), c.get("unsure", 0),
            c.get("line_noise", 0)))
        lines.append(f"{self.session.pending} judgement(s) not yet written to the store "
                     f"({len(self.session.written)} shard(s) written this session)")
        self.progress.setText("<br>".join(lines))
        self._undo_btn.setEnabled(self.session.can_undo())

    # -- z-traces -------------------------------------------------------------

    def _on_traces_toggled(self, *_a: object) -> None:
        self._trace_dock.setVisible(self.traces_box.isChecked())
        row = self.session.current()
        if row is not None:
            self._show_traces(row)

    def _trace_key(self, row: dict[str, Any]) -> str:
        return f"{row['recording']}|{float(row['region_start_s'])}|{float(row['region_stop_s'])}"

    def _show_traces(self, row: dict[str, Any]) -> None:
        if not self.traces_box.isChecked():
            self.ztrace.set_traces([])
            return
        key = self._trace_key(row)
        if key in self._traces_cache:
            self.ztrace.set_traces(self._traces_cache[key])
            self.ztrace.show_span(float(row["start_s"]), float(row["stop_s"]))
            self.recentre_traces(row)
            return
        self.ztrace.set_traces([])
        if key in self._traces_running:
            return
        region = (float(row["region_start_s"]), float(row["region_stop_s"]))
        job = _TraceJob(key, self._traces_fn, self._recording, region)
        job.signals.done.connect(self._on_traces_done)
        job.signals.failed.connect(self._on_traces_failed)
        self._traces_running.add(key)
        self.status.setText("Computing band z-traces for this region in the background...")
        if self._async_traces:
            self._jobs.append(job)
            QThreadPool.globalInstance().start(job)
        else:
            job.run()

    def recentre_traces(self, row: dict[str, Any]) -> None:
        lo = max(0.0, float(row["start_s"]) - self._context)
        self._set_trace_range(lo, float(row["stop_s"]) + self._context)

    def _set_trace_range(self, lo: float, hi: float) -> None:
        self.ztrace.set_x_range(lo, hi)

    def _on_traces_done(self, key: str, _intervals: object, traces: object) -> None:
        self._traces_running.discard(key)
        self._traces_cache[key] = list(traces)  # type: ignore[call-overload]
        self.status.setText("Band z-traces ready.")
        row = self.session.current()
        if row is not None and self._trace_key(row) == key:
            self._show_traces(row)

    def _on_traces_failed(self, key: str, why: str) -> None:
        self._traces_running.discard(key)
        self.last_error = why
        self.status.setText(f"z-traces unavailable for this region: {why}")

    # -- closing ----------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override
        """Write everything pending, then close.

        If the write fails the window still closes, after saying so: the judgements are
        in the local journal and are restored the next time this queue is opened.
        """
        try:
            self.session.close()
        except Exception as exc:  # noqa: BLE001 - shown; the journal still holds them
            self.last_error = str(exc)
            QMessageBox.warning(
                self, "Candidate adjudication",
                f"Could not write judgements to the store:\n{exc}\n\nThey are kept in the "
                f"local journal ({self.session.journal_path}) and will be restored next "
                "time this queue is opened.")
        event.accept()


def open_queue(queue_path: Path, store: GemsStore, *, user: str | None = None,
               journal_dir: Path | None = None, survivals_root: Path | None = None,
               ) -> AdjudicationWindow:
    """Build the session and window for ``queue_path`` with the production loaders."""
    from gems_blanking_v2.io.store import cache_dir, resolve_user_id

    from ui.adjudicate.judgements import app_commit
    from ui.adjudicate.queue import load_queue
    from ui.adjudicate.session import file_sha256

    queue_path = Path(queue_path)
    queue = load_queue(queue_path)
    who = resolve_user_id(user)
    session = AdjudicationSession(
        queue, store, who.user_id,
        journal_dir=journal_dir or (cache_dir() / "adjudication"),
        queue_file=queue_path.name, queue_sha256=file_sha256(queue_path),
        app_sha=app_commit())
    root = loaders.resolve_survivals_root(survivals_root)

    def load_fn(row: dict[str, Any]) -> Any:
        if row["cohort"] == "new":
            return loaders.load_new_row(store, str(row["animal"]), str(row["recording"]))
        if root is None:
            msg = ("old-cohort row but no Survivals root: pass --survivals-root or set "
                   f"{loaders.SURVIVALS_ENV}")
            raise FileNotFoundError(msg)
        path = loaders.old_signal_path(root, str(row["folder"]), str(row["recording"]))
        return loaders.load_old_signal(path, str(row["recording"]), str(row["animal"]),
                                       units=loaders.OLD_COHORT_UNITS)

    win = AdjudicationWindow(session, load_fn=load_fn)
    if not who.is_confident:
        win.status.setText(f"Labelling as '{who.user_id}' (from the OS account - set git "
                           "user.email to record who judged).")
    return win
