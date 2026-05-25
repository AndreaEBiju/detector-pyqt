"""Add the detector-core submodule to sys.path so `from detector import …`
works in both `python -c "..."` invocations and pytest.

The submodule layout is `detector-pyqt/detector-core/detector/`, so we
prepend `detector-core/` to sys.path — that puts the `detector` package
at the top of the search path.

Also registers the `qt_display` marker (default-deselected) for widget
tests that need a real Qt display. Bare QApplication construction
segfaults on truly headless boxes, so those tests are opt-in: run with
`pytest -m qt_display` from a Mac terminal."""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
_detector_core = _root / "detector-core"
if _detector_core.exists() and str(_detector_core) not in sys.path:
    sys.path.insert(0, str(_detector_core))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "qt_display: requires a real Qt display (default deselected); "
        "run explicitly with `pytest -m qt_display`",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-deselect qt_display-marked tests unless the user explicitly
    asked for them via `-m qt_display` (or any explicit `-m` expression
    that mentions qt_display)."""
    keyword_expr = config.getoption("-m", default="")
    if "qt_display" in (keyword_expr or ""):
        return
    import pytest
    skip_qt = pytest.mark.skip(
        reason="qt_display marker -- needs a real display; "
               "run with `pytest -m qt_display` to enable"
    )
    for item in items:
        if "qt_display" in item.keywords:
            item.add_marker(skip_qt)
