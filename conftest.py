"""Add the detector-core submodule to sys.path so `from detector import …`
works in both `python -c "..."` invocations and pytest.

The submodule layout is `detector-pyqt/detector-core/detector/`, so we
prepend `detector-core/` to sys.path — that puts the `detector` package
at the top of the search path."""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
_detector_core = _root / "detector-core"
if _detector_core.exists() and str(_detector_core) not in sys.path:
    sys.path.insert(0, str(_detector_core))
