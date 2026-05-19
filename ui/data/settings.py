"""Per-user UI settings — small JSON sidecar at
`~/.detector/pyqt_settings.json` (separate from Streamlit's
`~/.detector/settings.json` because the two UIs have different
preference surfaces).

Keys persisted:
- `model_version` — last-used `model_v*` directory name, or None
  → fall back to `paths.get_current_model_version()`.
- `auto_run_on_open` — bool, default True. Streamlit Phase 8 parity.
- `inference_skip_stim` — bool, default True (only honoured for
  stim_rec recordings with a boundary set).
- `last_recording_dir` — last path the Open dialog was pointed at.
  Reduces clicks when working through a folder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent.parent
_detector_core = _repo_root / "detector-core"
if _detector_core.exists() and str(_detector_core) not in sys.path:
    sys.path.insert(0, str(_detector_core))

from detector import paths as detector_paths  # noqa: E402


DEFAULTS: dict[str, Any] = {
    "model_version": None,           # None → resolve via current_model pointer
    "auto_run_on_open": True,
    "inference_skip_stim": True,
    "last_recording_dir": "",
}


def _settings_path() -> Path:
    return detector_paths.get_home() / "pyqt_settings.json"


def load_settings() -> dict[str, Any]:
    """Return defaults merged with whatever is in the on-disk JSON.
    Missing file → all defaults. Unreadable JSON → defaults (silent)."""
    p = _settings_path()
    if not p.exists():
        return dict(DEFAULTS)
    try:
        raw = json.loads(p.read_text())
    except Exception:
        return dict(DEFAULTS)
    out = dict(DEFAULTS)
    for k in DEFAULTS:
        if k in raw:
            out[k] = raw[k]
    return out


def save_settings(settings: dict[str, Any]) -> None:
    """Atomic write (tmp + rename) to the settings JSON."""
    p = _settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(settings, indent=2))
    tmp.replace(p)


def update_setting(key: str, value: Any) -> None:
    """Shortcut: load → set → save."""
    s = load_settings()
    s[key] = value
    save_settings(s)
