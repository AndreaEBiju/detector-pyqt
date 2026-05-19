"""Retrain worker — thin Qt wrapper around the backend's subprocess
orchestration (`detector.retrain_subprocess`).

The actual retrain runs in a child process spawned via
`start_retrain()`. This worker doesn't host the child — it polls the
job's status file via a QTimer and emits Qt signals so the training
window can update the progress bar and live-log view without
blocking the UI thread.

Crash isolation: a retrain that OOMs or segfaults kills only the
child process. The Qt UI keeps running and the next status() poll
returns state=`failed` with the exit code, so the user gets a clean
error report instead of a frozen window.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QTimer, Signal

_repo_root = Path(__file__).resolve().parent.parent.parent
_detector_core = _repo_root / "detector-core"
if _detector_core.exists() and str(_detector_core) not in sys.path:
    sys.path.insert(0, str(_detector_core))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from detector.retrain_subprocess import (        # noqa: E402
    RetrainJob, start_retrain, status, cancel, list_jobs,
)


POLL_INTERVAL_MS = 500


class RetrainMonitor(QObject):
    """Watches one retrain job via QTimer. Emits Qt signals on each
    status poll. Single-shot in spirit — once `finished` or `failed`
    fires the timer stops.

    Lifetime model:
      - Caller calls `start_new(...)` to spawn a new retrain and start
        monitoring.
      - Alternatively `attach(job_id)` reattaches to an in-progress
        job (e.g. after re-opening the training window).
      - The monitor lives until the job terminates OR the caller calls
        `stop()` to abandon monitoring (the child keeps running).
    """

    # `state` is one of "running" / "succeeded" / "failed" / "unknown".
    status_changed = Signal(dict)
    finished = Signal(dict)            # succeeded — full status dict
    failed = Signal(dict)              # failed — full status dict
    log_tail_changed = Signal(str)     # live log tail snapshot

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)
        self._job_id: Optional[str] = None
        self._last_state: Optional[str] = None
        self._last_tail: str = ""

    @property
    def job_id(self) -> Optional[str]:
        return self._job_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_new(self, **kwargs) -> RetrainJob:
        """Spawn a new retrain subprocess and begin polling. kwargs
        are forwarded to `detector.retrain_subprocess.start_retrain`."""
        if self._timer.isActive():
            raise RuntimeError(
                "RetrainMonitor is already watching a job. Call stop() first."
            )
        job = start_retrain(**kwargs)
        self._job_id = job.job_id
        self._last_state = None
        self._last_tail = ""
        self._timer.start()
        return job

    def attach(self, job_id: str) -> None:
        """Begin polling an in-progress job by id."""
        if self._timer.isActive():
            raise RuntimeError("monitor already running")
        self._job_id = job_id
        self._last_state = None
        self._last_tail = ""
        self._timer.start()

    def stop(self) -> None:
        """Abandon monitoring. The child process keeps running."""
        self._timer.stop()

    def cancel(self) -> bool:
        """SIGTERM the child. The next poll will see state=failed."""
        if self._job_id is None:
            return False
        return cancel(self._job_id)

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def _on_tick(self) -> None:
        if self._job_id is None:
            return
        try:
            st = status(self._job_id)
        except Exception as exc:                         # pragma: no cover
            self._timer.stop()
            self.failed.emit({
                "state": "failed", "phase": "unknown",
                "progress": 0.0, "tail": f"status() raised: {exc}",
            })
            return

        state = st.get("state", "unknown")
        tail = st.get("tail", "")
        if tail != self._last_tail:
            self._last_tail = tail
            self.log_tail_changed.emit(tail)
        if state != self._last_state:
            self._last_state = state
        self.status_changed.emit(st)
        if state == "succeeded":
            self._timer.stop()
            self.finished.emit(st)
        elif state == "failed":
            self._timer.stop()
            self.failed.emit(st)


def list_recent_jobs(n: int = 10) -> list[RetrainJob]:
    """All retrain jobs on disk, newest first."""
    jobs = list_jobs()
    return list(reversed(jobs))[:n]
