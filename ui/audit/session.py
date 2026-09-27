"""Blind-then-reveal audit state. The ordering is enforced, not merely intended.

The human marks artifacts on the raw signal with candidates **hidden**, commits,
and only then are candidates and z-traces revealed. If candidates are visible
while marking, the labeller anchors on them and the measured recall is the
detector's own output reflected back - the gate would be grading itself.

That ordering is the entire value of the audit, so it is a state machine with
illegal transitions that raise, rather than a convention about the order in which
a window calls its own methods. A UI refactor can reorder method calls; it cannot
reorder these without the tests failing.

The committed marks are **serialised before** the reveal happens, so even a crash
between commit and reveal leaves a usable, uncontaminated record.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np


class Phase(StrEnum):
    """Where a span is in the blind-then-reveal cycle."""

    BLIND = "blind"
    """Marking. No candidate, prediction or z information is available."""
    COMMITTED = "committed"
    """Marks frozen and written. Nothing revealed yet."""
    REVEALED = "revealed"
    """Candidates and z-traces visible. Marks are read-only."""


class PhaseError(RuntimeError):
    """An operation was attempted in a phase that forbids it."""


@dataclass(frozen=True, slots=True)
class Mark:
    """One human-marked artifact, in seconds on the recording's timeline."""

    start_s: float
    stop_s: float

    def __post_init__(self) -> None:
        if not self.stop_s > self.start_s:
            msg = f"mark must have positive duration, got [{self.start_s}, {self.stop_s})"
            raise ValueError(msg)


@dataclass
class SpanSession:
    """One span's worth of blind marking, then reveal.

    Attributes
    ----------
    span_id
        Identifies the span within the plan.
    recording_id, start_s, stop_s
        Where this span sits.
    phase
        See :class:`Phase`. Transitions are one-way.
    marks
        The human's marks. Mutable only while :attr:`phase` is ``BLIND``.
    committed_at
        UTC ISO-8601 timestamp of the commit, or ``None`` before it. Recorded so
        the audit record can show marking preceded reveal rather than asserting it.
    """

    span_id: str
    recording_id: str
    start_s: float
    stop_s: float
    phase: Phase = Phase.BLIND
    marks: list[Mark] = field(default_factory=list)
    committed_at: str | None = None
    _revealed: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- blind phase ----------------------------------------------------
    def add_mark(self, start_s: float, stop_s: float) -> Mark:
        """Record one artifact. Only while blind."""
        if self.phase is not Phase.BLIND:
            msg = (
                f"cannot add a mark in phase {self.phase}: marks are frozen at "
                "commit so that what the human found cannot be edited after "
                "seeing the detector's answer"
            )
            raise PhaseError(msg)
        m = Mark(start_s, stop_s)
        self.marks.append(m)
        return m

    def remove_mark(self, index: int) -> None:
        """Undo a mark. Only while blind."""
        if self.phase is not Phase.BLIND:
            msg = f"cannot remove a mark in phase {self.phase}: marks are frozen"
            raise PhaseError(msg)
        del self.marks[index]

    # -- the gate -------------------------------------------------------
    def commit(self, out_dir: Path) -> Path:
        """Freeze the marks, write them, and allow the reveal.

        Writing happens **here**, before anything is revealed, so the record on
        disk can never have been influenced by the detector's output. Returns the
        path written.
        """
        if self.phase is not Phase.BLIND:
            msg = f"already committed (phase {self.phase})"
            raise PhaseError(msg)
        self.committed_at = datetime.now(UTC).isoformat()
        self.phase = Phase.COMMITTED

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.span_id}_blind_marks.json"
        body = {
            "span_id": self.span_id,
            "recording_id": self.recording_id,
            "start_s": self.start_s,
            "stop_s": self.stop_s,
            "committed_at": self.committed_at,
            "blind": True,
            "note": (
                "Written at commit, BEFORE any candidate or z information was "
                "shown. These marks are the audit's independent ground truth."
            ),
            "marks": [{"start_s": m.start_s, "stop_s": m.stop_s} for m in self.marks],
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(body, indent=1), encoding="utf-8", newline="\n"
        )
        tmp.replace(path)
        return path

    # -- reveal ---------------------------------------------------------
    def reveal(self, *, candidates: np.ndarray, traces: list[Any]) -> None:
        """Make candidates and z-traces available. Only after a commit."""
        if self.phase is Phase.BLIND:
            msg = (
                "cannot reveal before commit. Showing candidates while the human "
                "can still edit marks is what makes the gate circular - it would "
                "measure the detector against a label set the detector shaped."
            )
            raise PhaseError(msg)
        self.phase = Phase.REVEALED
        self._revealed = {"candidates": candidates, "traces": traces}

    @property
    def candidates(self) -> np.ndarray:
        """Candidate intervals, or an empty array while still hidden.

        Empty rather than raising: the viewer asks for this every repaint, and a
        phase check at the call site would be one more place to get it wrong.
        """
        if self.phase is not Phase.REVEALED:
            return np.zeros((0, 2), dtype=np.float64)
        return self._revealed["candidates"]

    @property
    def traces(self) -> list[Any]:
        """Band z-traces, or empty while still hidden."""
        if self.phase is not Phase.REVEALED:
            return []
        return list(self._revealed["traces"])

    def marks_array(self) -> np.ndarray:
        """``(n, 2)`` seconds, the shape the viewer's overlays take."""
        if not self.marks:
            return np.zeros((0, 2), dtype=np.float64)
        return np.asarray([[m.start_s, m.stop_s] for m in self.marks], dtype=np.float64)
