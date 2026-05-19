"""Recording queue — persistent ledger of "recordings to work through"
that lets you point the tool at a folder of N recordings and walk
through them with one click per recording.

Persistence: queue state is saved as JSON at
`~/.detector/queues/<queue_id>.json`. Survives app restarts so you
can pick up where you left off the next day.

Schema:
{
  "queue_id": "20260519T123456Z",
  "created_at": "2026-05-19T12:34:56Z",
  "items": [
    {
      "path": "/abs/path/recording.mat",
      "status": "pending" | "in_progress" | "done" | "skipped",
      "last_opened": "2026-05-19T13:00:00Z" | null,
      "notes": ""
    },
    ...
  ]
}
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Optional

_repo_root = Path(__file__).resolve().parent.parent.parent
_detector_core = _repo_root / "detector-core"
if _detector_core.exists() and str(_detector_core) not in sys.path:
    sys.path.insert(0, str(_detector_core))

from detector import paths as detector_paths       # noqa: E402


QueueStatus = Literal["pending", "in_progress", "done", "skipped"]


def _utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class QueueItem:
    path: str
    status: QueueStatus = "pending"
    last_opened: Optional[str] = None
    notes: str = ""


@dataclass
class RecordingQueue:
    queue_id: str
    created_at: str = field(default_factory=_utcnow_iso)
    items: list[QueueItem] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def queues_dir() -> Path:
        d = detector_paths.get_queues_dir()
        d.mkdir(parents=True, exist_ok=True)
        return d

    @classmethod
    def new(cls, queue_id: Optional[str] = None) -> "RecordingQueue":
        """Create a fresh queue with the given (or auto-generated) id."""
        if queue_id is None:
            queue_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return cls(queue_id=queue_id)

    @classmethod
    def load(cls, path: Path) -> "RecordingQueue":
        data = json.loads(Path(path).read_text())
        items = [QueueItem(**it) for it in data.get("items", [])]
        return cls(
            queue_id=data["queue_id"],
            created_at=data.get("created_at", _utcnow_iso()),
            items=items,
        )

    @classmethod
    def load_by_id(cls, queue_id: str) -> "RecordingQueue":
        return cls.load(cls.queues_dir() / f"{queue_id}.json")

    @classmethod
    def list_all(cls) -> list[tuple[str, Path]]:
        """All queue IDs on disk + their JSON path."""
        d = cls.queues_dir()
        out = []
        for p in sorted(d.glob("*.json")):
            out.append((p.stem, p))
        return out

    def save(self) -> Path:
        """Atomic write to the canonical location."""
        d = self.queues_dir()
        out = d / f"{self.queue_id}.json"
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "queue_id": self.queue_id,
            "created_at": self.created_at,
            "items": [asdict(it) for it in self.items],
        }, indent=2))
        tmp.replace(out)
        return out

    def delete_on_disk(self) -> None:
        p = self.queues_dir() / f"{self.queue_id}.json"
        if p.exists():
            p.unlink()

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------

    def add_paths(self, paths: Iterable[Path]) -> int:
        """Append paths that aren't already in the queue. Returns the
        number actually added (skips duplicates)."""
        existing = {it.path for it in self.items}
        added = 0
        for p in paths:
            abs_p = str(Path(p).resolve())
            if abs_p in existing:
                continue
            self.items.append(QueueItem(path=abs_p))
            existing.add(abs_p)
            added += 1
        return added

    def add_folder(self, folder: Path, glob: str = "*.h5") -> int:
        """Scan `folder` for files matching `glob` and add them."""
        return self.add_paths(sorted(Path(folder).glob(glob)))

    def remove_path(self, path: Path) -> bool:
        abs_p = str(Path(path).resolve())
        before = len(self.items)
        self.items = [it for it in self.items if it.path != abs_p]
        return len(self.items) != before

    # ------------------------------------------------------------------
    # Workflow helpers
    # ------------------------------------------------------------------

    def next_pending(self) -> Optional[QueueItem]:
        """First item with status='pending', or None if all done."""
        for it in self.items:
            if it.status == "pending":
                return it
        return None

    def mark(self, path: Path, status: QueueStatus,
              notes: Optional[str] = None) -> bool:
        """Update one item's status. Returns True if the item was found."""
        abs_p = str(Path(path).resolve())
        for it in self.items:
            if it.path == abs_p:
                it.status = status
                if status in ("in_progress", "done"):
                    it.last_opened = _utcnow_iso()
                if notes is not None:
                    it.notes = notes
                return True
        return False

    def counts(self) -> dict[str, int]:
        """Aggregate counts per status."""
        out = {"pending": 0, "in_progress": 0, "done": 0, "skipped": 0}
        for it in self.items:
            out[it.status] = out.get(it.status, 0) + 1
        return out

    def __len__(self) -> int:
        return len(self.items)
