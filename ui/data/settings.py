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
- `preprocessing_*` — defaults that pre-fill the per-batch
  preprocessing review UI. They never override an existing animal
  profile (profiles store their own resolved values); they only
  apply when a brand-new animal is reviewed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# `detector` is the detector-core distribution, a declared dependency installed
# editable from the submodule - never found by inserting its path (invariant 21).
from detector import paths as detector_paths


DEFAULTS: dict[str, Any] = {
    "model_version": None,           # None → resolve via current_model pointer
    "auto_run_on_open": True,
    "inference_skip_stim": True,
    "last_recording_dir": "",
    # Preprocessing defaults — mirror NotchParams in
    # detector.preprocessing.notch. Editing these in the Training
    # window's Preprocessing tab updates the JSON; the next NEW
    # animal's review picks them up. Existing per-animal profiles
    # are unaffected (they store their own values).
    # Defaults for the notch review dialog: mains hum is filtered at
    # 60 Hz ONLY, Q=30. There is no per-harmonic auto-detection and it
    # must not be enabled — the user adjusts the harmonics field
    # manually if their setup needs different frequencies (e.g. 50 Hz
    # for European mains).
    #
    # 120 and 180 were removed, measured: a 120 Hz notch rings INSIDE
    # the 300-3000 Hz ENG band at 2.41 uV per mV of excursion, so a
    # 7.5 mV motion excursion puts its ringing at the 4.5-sigma spike
    # threshold and manufactures spikes out of the filter. The 60 Hz
    # notch contributes 0.221 uV/mV, an order of magnitude less,
    # because the bandpass attenuates 60 Hz by -36.4 dB through
    # filtfilt. Prophylactic filtering of a harmonic that may not be
    # present is not worth injecting artifact into the consumer band.
    "preprocessing_q_factor": 30.0,
    # Detrend toggle. When on, the per-channel mean is subtracted
    # from the signal before filtering AND from the raw trace shown
    # alongside the filtered trace — so both display at the same
    # baseline. Off → both traces show their natural DC offset.
    "preprocessing_detrend": True,
    "preprocessing_default_freqs_hz": [60.0],
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
