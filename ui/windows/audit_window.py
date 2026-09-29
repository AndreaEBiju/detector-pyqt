"""The blind recall audit, launchable: plan five spans, mark each blind, commit, reveal.

Task 16 Change 2's pieces (plan, session, controller, dock) existed but nothing
opened them, so Andrea could not start labelling - the human critical path to
the 09 gate. This window is that entry point:

* **The formal audit is a five-span PLAN**, drawn once by ``plan_audit`` (five
  contiguous 2-min spans; animals rotated, conditions balanced across the gate's
  eligible pool by ``recall.condition_plan``) with its
  seed and pool written to the store BEFORE the first span is shown. Each open
  advances through it ("span k of 5"); progress is read back from which spans
  have committed marks, so closing the window and returning resumes the plan.
* **A trial span is not audit data.** "Trial span" draws one span on a chosen
  recording, for learning the UI; its record says ``"trial": true``.
* **Only eligible recordings are offered or planned** (``audit_pool``): excluded
  ones never appear, and a direct request for one is refused.
* **Stim epochs are removed before any span is drawn** (the protocol window at the
  start of every stim/recovery file), and each span's ``Assessable`` asserts it.
* **Marks are written to the store under the session key before anything is
  revealed** (``store.audit_dir``); candidates are not computed until after the
  commit, so there is nothing in memory to leak into the blind phase.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from gems_blanking_v2.detect import recall
from gems_blanking_v2.io.audit_pool import (
    EligibleRecording,
    assessable_regions,
    eligible_recordings,
    planned_regions,
    stim_epoch_s,
)
from gems_blanking_v2.io.stim_split import (
    PROTOCOL_FILENAME,
    ProtocolSpec,
    load_protocol_book,
)
from gems_blanking_v2.io.store import GemsStore, atomic_write_text
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.audit import bridge
from ui.audit.array_recording import ArrayRecording
from ui.audit.controller import sync_viewer
from ui.audit.plan import N_SPANS, SPAN_S, Assessable, Span, plan_audit, replace_spans
from ui.audit.session import Phase, SpanSession
from ui.widgets.signal_viewer import MultiChannelViewer
from ui.widgets.ztrace_dock import ZTraceDock

RevealFn = Callable[[Any, tuple[float, float]], tuple[np.ndarray, list[Any]]]

_COLOURS = ("#4c9be8", "#6bb3f0", "#9ccbf5", "#e8834c", "#f0a06b", "#f5bf9c",
            "#5cc98a", "#7fd6a3", "#a6e3bf")


class AuditWindow(QMainWindow):
    """The five-span blind audit, plus trial spans, on eligible new-cohort data."""

    def __init__(
        self, store: GemsStore, *, reveal_fn: RevealFn | None = None,
        confirm_commit: Callable[[str], bool] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Blind recall audit")
        self._store = store
        self._reveal_fn: RevealFn = reveal_fn or bridge.reveal_for_region
        self._confirm_commit: Callable[[str], bool] = confirm_commit or self._confirm_merged
        self.last_merge_summary: str | None = None
        self._pool = eligible_recordings(store)
        self.last_error: str | None = None
        self._book = load_protocol_book(store.root / PROTOCOL_FILENAME)
        self._rec: EligibleRecording | None = None
        self._recording: Any = None
        self._session: SpanSession | None = None
        self._region: tuple[float, float] | None = None
        self._record: dict[str, Any] = {}
        self.viewer: MultiChannelViewer | None = None
        self.plan: dict[str, Any] | None = self._latest_open_plan()

        self._list = QListWidget()
        for r in self._pool:
            item = QListWidgetItem(f"{r.animal}  {r.epoch:14}  {r.folder_name}")
            item.setData(Qt.UserRole, r.session)
            self._list.addItem(item)
        self._plan_btn = QPushButton("Create the five-span audit plan")
        self._next_btn = QPushButton("Open next audit span")
        self._trial_btn = QPushButton("Trial span on selected recording (not audit data)")
        self._undo_btn = QPushButton("Undo last mark")
        self._commit_btn = QPushButton("Commit marks (then reveal)")
        # Scrolling past the span is allowed - context helps - so getting back is
        # one click (or Home). Andrea scrolled to the recording's start and every
        # mark there was, correctly, refused as outside the span.
        self._span_btn = QPushButton("Back to span (Home)")
        self.progress = QLabel("")
        self._status = QLabel(f"{len(self._pool)} eligible recordings.")
        # Every button goes through _clicked: ``clicked`` emits ``checked`` and
        # PySide6 hands it to any slot with a free parameter, so connecting
        # create_plan directly made every plan's seed int(False) == 0.
        self._plan_btn.clicked.connect(self._clicked(self.create_plan))
        self._next_btn.clicked.connect(self._clicked(self.open_next_span))
        self._trial_btn.clicked.connect(self._clicked(self._on_trial_clicked))
        self._undo_btn.clicked.connect(self._clicked(self.undo_mark))
        self._commit_btn.clicked.connect(self._clicked(self.commit))
        self._span_btn.clicked.connect(self._clicked(self.back_to_span))
        QShortcut(QKeySequence(Qt.Key_Home), self, activated=self._clicked(self.back_to_span))

        left = QWidget()
        lay = QVBoxLayout(left)
        lay.addWidget(QLabel("Formal audit"))
        lay.addWidget(self._plan_btn)
        lay.addWidget(self._next_btn)
        lay.addWidget(self.progress)
        lay.addWidget(QLabel("Eligible recordings (excluded ones are never listed)"))
        lay.addWidget(self._list)
        lay.addWidget(self._trial_btn)
        bar = QWidget()
        blay = QHBoxLayout(bar)
        for w in (self._span_btn, self._undo_btn, self._commit_btn, self._status):
            blay.addWidget(w)
        self._centre = QWidget()
        self._centre_lay = QVBoxLayout(self._centre)
        self._centre_lay.addWidget(bar)
        root = QWidget()
        rlay = QHBoxLayout(root)
        rlay.addWidget(left, 1)
        rlay.addWidget(self._centre, 4)
        self.setCentralWidget(root)

        self.ztrace = ZTraceDock()
        dock = QDockWidget("z-traces (after commit)", self)
        dock.setWidget(self.ztrace)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self._refresh()

    def _clicked(self, action: Callable[[], Any]) -> Callable[..., None]:
        """A button slot: call ``action()`` with no arguments; show any failure.

        A refusal (no plan possible, a planned recording since excluded, the
        store unreachable) must reach the person at the screen, not a console
        they cannot see.
        """
        def slot(*_ignored: object) -> None:
            try:
                action()
            except Exception as exc:  # noqa: BLE001 - shown, not swallowed
                self._status.setText(f"Could not do that: {exc}")
                box = QMessageBox(QMessageBox.Warning, "Blind recall audit", str(exc),
                                  parent=self)
                box.setModal(False)
                box.show()
                self.last_error = str(exc)
        return slot

    # -- the pool ---------------------------------------------------------

    @property
    def offered(self) -> list[EligibleRecording]:
        """What the list shows - by construction, only eligible recordings."""
        return list(self._pool)

    def _protocol(self, rec: EligibleRecording) -> ProtocolSpec:
        return self._book.for_path(self._store.relpath(rec.source))

    # -- the five-span plan -------------------------------------------------

    def create_plan(self, seed: int | None = None) -> dict[str, Any]:
        """Draw the plan once, write it with its seed, and make it the active plan."""
        if self.plan is not None:
            msg = f"plan {self.plan['plan_id']} is still open; finish it first"
            raise RuntimeError(msg)
        # The sequential rule (task 09): a new round only once the last is scored and
        # every miss is diagnosed and closed - fixed, not_target, or an accepted
        # limitation (ruling 2026-09-29).
        allowed, why = recall.check_next_round(self._store, reveal_sha=bridge.reveal_sha())
        if not allowed:
            msg = f"a new audit round cannot be drawn yet: {why}"
            raise RuntimeError(msg)
        pool = self._assessable_pool()
        seed = secrets.randbits(32) if seed is None else int(seed)
        # Conditions are balanced across the gate's eligible pool (ruling 2026-09-29),
        # from the composition of earlier eligible plans only - never their scores.
        conditions = recall.condition_plan(self._store, N_SPANS)
        drawn = plan_audit(pool, seed, conditions=conditions)
        plan_id = f"plan_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{seed:08x}"
        record = {
            "plan_id": plan_id, "created_at": datetime.now(UTC).isoformat(),
            **drawn.to_json(), "pool_size": len(pool),
            # Declared before any labelling (Andrea, 2026-09-28): one artifact is one
            # connected run of committed marks. The scorer reads it from here.
            "scoring_unit": recall.CURRENT_SCORING_UNIT,
            "condition_rule": "each span takes the condition with fewer spans in the "
                              "gate's eligible pool (ties alternate, baseline first); "
                              "from earlier eligible plans' composition, never scores",
            "pool_rule": "eligible (non-excluded) baseline and stim_recovery recordings "
                         "with a recorded duration; stim_recovery loses 0 to "
                         "stim_duration + tolerance s; 20 s edge guards",
        }
        path = self._store.audit_plan_path(plan_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(record, indent=1, sort_keys=True) + "\n")
        self.plan = record
        self._refresh()
        return record

    def _assessable_pool(self) -> list[Assessable]:
        """Every eligible recording with room for a span, as the planner sees it."""
        pool: list[Assessable] = []
        for r in self._pool:
            planned = planned_regions(r, self._protocol(r))
            if planned is None or not planned[0]:
                continue
            pool.append(Assessable(recording_id=r.session, animal=r.animal,
                                   condition=r.epoch, regions=planned[0],
                                   excluded=planned[1]))
        return pool

    def ineligible_spans(self) -> list[int]:
        """Indices of the open plan's UNCOMMITTED spans whose recording is no longer
        eligible (excluded, or unreachable, since the plan was drawn)."""
        if self.plan is None:
            return []
        eligible = {r.session for r in self._pool}
        done = self._committed(self.plan)
        return [k for k, sp in enumerate(self.plan["spans"])
                if k not in done and sp["recording_id"] not in eligible]

    def replace_ineligible_spans(self, reason: str, seed: int | None = None) -> list[int]:
        """Replace the open plan's ineligible uncommitted spans, and record it in the plan.

        Committed spans are never touched. Each replacement keeps its span's
        condition and index (so its span id is unchanged); the plan keeps the
        original spans, the fresh seed and ``reason`` under ``amendments``.
        """
        indices = self.ineligible_spans()
        if not indices or self.plan is None:
            return []
        seed = secrets.randbits(32) if seed is None else int(seed)
        old = [Span(**sp) for sp in self.plan["spans"]]
        new = replace_spans(old, indices, self._assessable_pool(), seed)
        self.plan["spans"] = [asdict(sp) for sp in new]
        self.plan.setdefault("amendments", []).append({
            "at": datetime.now(UTC).isoformat(), "seed": seed, "reason": reason,
            "rule": "same condition, a recording not already in the plan chosen "
                    "uniformly, start uniform within its placeable range",
            "replaced": [{"span": k + 1, "old": asdict(old[k]), "new": asdict(new[k])}
                         for k in indices],
        })
        atomic_write_text(self._store.audit_plan_path(self.plan["plan_id"]),
                          json.dumps(self.plan, indent=1, sort_keys=True) + "\n")
        self._refresh()
        return indices

    def _latest_open_plan(self) -> dict[str, Any] | None:
        folder = self._store.audit_plan_path("x").parent
        if not folder.is_dir():
            return None
        for path in sorted(folder.glob("plan_*.json"), reverse=True):
            plan = json.loads(path.read_text(encoding="utf-8"))
            if len(self._committed(plan)) < len(plan["spans"]):
                return plan
        return None

    def _span_id(self, plan: dict[str, Any], k: int) -> str:
        # One construction site, shared with the scorer that reads these files back.
        return recall.span_id(plan, k)

    def _committed(self, plan: dict[str, Any]) -> set[int]:
        """Span indices whose marks are on disk - progress is read, not remembered."""
        done = set()
        for k, span in enumerate(plan["spans"]):
            marks = (self._store.audit_dir(span["animal"], span["recording_id"])
                     / f"{self._span_id(plan, k)}_blind_marks.json")
            if marks.is_file():
                done.add(k)
        return done

    def open_next_span(self) -> SpanSession | None:
        """Open the first uncommitted span of the active plan."""
        plan = self.plan
        if plan is None:
            self._status.setText("Create the audit plan first.")
            return None
        todo = [k for k in range(len(plan["spans"])) if k not in self._committed(plan)]
        if not todo:
            self._status.setText("All spans of this plan are committed.")
            return None
        k = todo[0]
        span = plan["spans"][k]
        rec = next((r for r in self._pool if r.session == span["recording_id"]), None)
        if rec is None:
            msg = (f"span {k + 1}'s recording {span['recording_id']} is no longer "
                   "eligible (excluded or unreachable since the plan was drawn). "
                   "Replace it with replace_ineligible_spans(); committed spans stay.")
            raise RuntimeError(msg)
        return self._open_span(
            rec, float(span["start_s"]), float(span["stop_s"]), self._span_id(plan, k),
            {"plan_id": plan["plan_id"], "span_index": k + 1,
             "n_spans": len(plan["spans"]), "seed": plan["seed"], "trial": False},
        )

    # -- a trial span -------------------------------------------------------

    def _on_trial_clicked(self) -> None:
        row = self._list.currentRow()
        if row < 0:
            self._status.setText("Pick a recording first.")
            return
        self.open_recording(self._pool[row])

    def open_recording(self, rec: EligibleRecording, seed: int | None = None) -> SpanSession:
        """A TRIAL span: one span drawn on ``rec``. Recorded as trial, not audit data."""
        duration = rec.duration_s
        if duration is None:
            duration = float("inf")  # resolved against the loaded length in _open_span
        seed = secrets.randbits(32) if seed is None else int(seed)
        stim, _ = stim_epoch_s(None, rec.source, rec.epoch, self._protocol(rec))
        regions = assessable_regions(duration, stim) if np.isfinite(duration) else None
        if regions is None:
            loaded = bridge.load_new_cohort(rec.source, rec.animal, store=self._store)
            regions = assessable_regions(
                loaded.recording.data.shape[0] / loaded.recording.fs, stim)
        placeable = Assessable(recording_id=rec.session, animal=rec.animal,
                               condition=rec.epoch, regions=regions,
                               excluded=(stim,) if stim is not None else ()
                               ).placeable(SPAN_S)
        if not placeable:
            msg = f"{rec.session}: no assessable stretch long enough for a {SPAN_S:g} s span"
            raise ValueError(msg)
        rng = np.random.default_rng(seed)
        widths = np.array([b - a for a, b in placeable])
        i = int(rng.choice(len(placeable), p=widths / widths.sum()))
        start = float(rng.uniform(*placeable[i]))
        return self._open_span(rec, start, start + SPAN_S,
                               f"trial_{round(start * 1000):09d}ms",
                               {"trial": True, "seed": seed})

    # -- one span (shared) ----------------------------------------------------

    def _open_span(
        self, rec: EligibleRecording, start: float, stop: float, span_id: str,
        extra: dict[str, Any],
    ) -> SpanSession:
        """Load ``rec`` and enter BLIND on ``[start, stop)``, asserting it is assessable."""
        loaded = bridge.load_new_cohort(rec.source, rec.animal, store=self._store)
        if loaded.provenance.get("excluded"):  # defence in depth: never offered
            msg = f"{rec.session} is excluded: {loaded.provenance['excluded']}"
            raise RuntimeError(msg)
        recording = loaded.recording
        duration = recording.data.shape[0] / recording.fs
        stim, method = stim_epoch_s(recording, rec.source, rec.epoch, self._protocol(rec))
        regions = assessable_regions(duration, stim)
        Assessable(recording_id=rec.session, animal=rec.animal, condition=rec.epoch,
                   regions=regions, excluded=(stim,) if stim is not None else ())
        region = next((r for r in regions if r[0] <= start and stop <= r[1]), None)
        if region is None:
            msg = (f"{span_id}: [{start:.1f}, {stop:.1f}) is not inside an assessable "
                   f"region of {rec.session} {regions}")
            raise ValueError(msg)
        self._region = region
        self._rec, self._recording = rec, recording
        self._session = SpanSession(span_id=span_id, recording_id=rec.session,
                                    start_s=start, stop_s=stop)
        self._record = {
            "session": rec.session, "animal": rec.animal, "epoch": rec.epoch,
            "source_path": self._store.relpath(rec.source), "span_id": span_id,
            "span_s": [start, stop], "assessable_regions_s": [list(r) for r in regions],
            "stim_epoch_s": list(stim) if stim is not None else None,
            "stim_epoch_method": method, "opened_at": datetime.now(UTC).isoformat(),
            **extra,
        }
        if self.viewer is not None:
            self.viewer.setParent(None)
        names = tuple(c.name for c in recording.channels)
        self.viewer = MultiChannelViewer(ArrayRecording(recording.data, recording.fs),
                                         channel_names=names,
                                         channel_colors=_COLOURS[: len(names)])
        self.viewer.bad_interval_added.connect(self._on_mark)
        self.viewer.follow_mark_drag = True  # a mark may run past the viewport
        self._centre_lay.insertWidget(0, self.viewer, 1)
        self.viewer.set_viewport(start, stop)
        self._shade_outside_span(start, stop, recording.data.shape[0] / recording.fs)
        self.ztrace.set_traces([])
        self._sync()
        label = ("TRIAL span (not audit data)" if extra.get("trial")
                 else f"Audit span {extra['span_index']} of {extra['n_spans']}")
        self._status.setText(f"BLIND - {label}: {rec.folder_name}, {start:.1f}-{stop:.1f} s. "
                             "Shift+drag to mark every artifact you see.")
        self._refresh()
        return self._session

    def back_to_span(self) -> None:
        """Snap the viewer back to exactly the open span."""
        if self.viewer is not None and self._session is not None:
            self.viewer.set_viewport(self._session.start_s, self._session.stop_s)

    def _shade_outside_span(self, start: float, stop: float, duration: float) -> None:
        """Grey out everything before and after the span, on every channel.

        Only the span can be marked; the shading makes it visible when scrolling
        has left it, instead of a refusal being the first sign.
        """
        self._outside_items = []
        for plot, _curve in self.viewer.plots:
            for lo, hi in ((0.0, start), (stop, duration)):
                if hi > lo:
                    item = pg.LinearRegionItem(values=(lo, hi), orientation="vertical",
                                               movable=False, brush=pg.mkBrush(128, 128, 128, 90),
                                               pen=pg.mkPen(None))
                    item.setZValue(-10)
                    plot.addItem(item)
                    self._outside_items.append(item)

    def _on_mark(self, lo: float, hi: float) -> None:
        """A Shift+drag in the viewer. Clipped to the span; outside it is refused."""
        s = self._session
        if s is None or s.phase is not Phase.BLIND:
            return
        a, b = max(lo, s.start_s), min(hi, s.stop_s)
        if not b > a:
            self._status.setText(
                f"That mark ({lo:.1f}-{hi:.1f} s) is outside this span "
                f"({s.start_s:.1f}-{s.stop_s:.1f} s, unshaded) and was not recorded. "
                "Click 'Back to span' or press Home.")
            return
        s.add_mark(a, b)
        self._sync()
        self._status.setText(f"BLIND - {len(s.marks)} mark(s).")

    def undo_mark(self) -> None:
        """Remove the most recent mark while blind."""
        s = self._session
        if s is not None and s.phase is Phase.BLIND and s.marks:
            s.remove_mark(len(s.marks) - 1)
            self._sync()

    def merge_summary(self) -> tuple[int, int, str]:
        """``(marks, artifacts, text)`` for the open span: overlapping or touching marks
        merged, as the scorer will count them, and close-but-separate pairs named."""
        s = self._session
        marks = [(m.start_s, m.stop_s) for m in (s.marks if s else [])]
        if not marks:
            return 0, 0, "No marks."
        merged, groups = recall.merge_marks(marks)
        order = sorted(range(len(marks)), key=lambda i: marks[i])
        num = {i: k + 1 for k, i in enumerate(order)}  # 1-based, in time order
        lines = [f"{len(marks)} marks -> {len(merged)} artifacts."]
        for (a, b), g in zip(merged, groups, strict=True):
            if len(g) > 1:
                lines.append(f"  marks {' + '.join(str(num[i]) for i in sorted(g, key=num.get))} "
                             f"overlap or touch: one artifact, {a:.2f}-{b:.2f} s")
        close = recall.close_pairs(marks)
        if close:
            lines.append("Kept separate (gap under 100 ms - two artifacts if you meant two):")
            lines += [f"  marks {num[i]} and {num[j]}, {1000 * gap:.0f} ms apart"
                      for i, j, gap in close]
        return len(marks), len(merged), "\n".join(lines)

    def _confirm_merged(self, text: str) -> bool:
        box = QMessageBox(QMessageBox.Question, "Commit - marks merged", text + "\n\nCommit "
                          "these marks? They are saved exactly as drawn; the merge is how "
                          "they are scored.", QMessageBox.Ok | QMessageBox.Cancel, self)
        return box.exec() == QMessageBox.Ok

    def commit(self) -> Path | None:
        """Write marks and the span record to the store, THEN compute and show the reveal.

        When marks overlap or touch, the merged view and its count are shown first and
        the commit waits for confirmation (Andrea, 2026-09-28); Cancel keeps the span
        blind. The marks file records the marks exactly as drawn.
        """
        s, rec = self._session, self._rec
        if s is None or rec is None or s.phase is not Phase.BLIND:
            return None
        n_marks, n_artifacts, text = self.merge_summary()
        self.last_merge_summary = text
        if n_artifacts < n_marks and not self._confirm_commit(text):
            self._status.setText(f"BLIND - not committed. {text.splitlines()[0]}")
            return None
        out_dir = self._store.audit_dir(rec.animal, rec.session)
        path = s.commit(out_dir)
        atomic_write_text(out_dir / f"{s.span_id}_plan.json",
                          json.dumps(self._record, indent=1, sort_keys=True) + "\n")
        self._status.setText("Committed. Computing candidates and z-traces...")
        assert self._region is not None
        intervals, traces = self._reveal_fn(self._recording, self._region)
        # What was revealed, digested into the span record (ratified 2026-09-27): the
        # scorer recomputes the candidates and refuses a span whose recompute does
        # not reproduce this. The marks file is untouched - it was written above.
        self._record["reveal"] = {
            "candidates_sha256": recall.candidate_digest(intervals),
            "n_candidates": int(np.asarray(intervals).reshape(-1, 2).shape[0]),
            "digest_rule": recall.DIGEST_RULE,
            "revealed_at": datetime.now(UTC).isoformat(),
        }
        atomic_write_text(out_dir / f"{s.span_id}_plan.json",
                          json.dumps(self._record, indent=1, sort_keys=True) + "\n")
        s.reveal(candidates=intervals, traces=traces)
        self._sync()
        self.ztrace.set_traces(traces)
        try:
            shown = self._store.relpath(path)
        except ValueError:
            shown = str(path)
        self._status.setText(f"REVEALED - {len(s.marks)} mark(s), {len(intervals)} "
                             f"candidate(s). Marks saved to {shown}")
        self._refresh()
        return path

    def _sync(self) -> None:
        if self.viewer is not None and self._session is not None:
            sync_viewer(self.viewer, self._session, float(self._recording.fs))

    def _refresh(self) -> None:
        blind = self._session is not None and self._session.phase is Phase.BLIND
        self._undo_btn.setEnabled(blind)
        self._commit_btn.setEnabled(blind)
        self._span_btn.setEnabled(self._session is not None and self.viewer is not None)
        self._plan_btn.setEnabled(self.plan is None and not blind)
        if self.plan is None:
            self.progress.setText(f"No open plan. Creating one draws {N_SPANS} spans.")
            self._next_btn.setEnabled(False)
            return
        done = len(self._committed(self.plan))
        n = len(self.plan["spans"])
        self.progress.setText(f"Plan {self.plan['plan_id']}: {done} of {n} spans committed")
        self._next_btn.setEnabled(not blind and done < n)
        if done == n:
            self.progress.setText(f"Plan {self.plan['plan_id']}: all {n} spans committed - done")
