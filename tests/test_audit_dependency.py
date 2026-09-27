"""gems_blanking_v2 is a declared, installed dependency - not a path hack (invariant 21)."""

from __future__ import annotations

import importlib.metadata as md
import inspect

from ui.audit import bridge


def test_the_bridge_does_not_touch_sys_path() -> None:
    """The old _ensure_path() inserted a sibling checkout; whichever copy it found won."""
    source = inspect.getsource(bridge)
    assert "sys.path" not in source.replace("``sys.path``", "")
    assert not hasattr(bridge, "_ensure_path")


def test_gems_blanking_v2_is_an_installed_distribution() -> None:
    dist = md.distribution("gems-blanking-v2")
    assert dist is not None
    requires = md.distribution("detector-pyqt").requires if _installed("detector-pyqt") else None
    if requires is not None:
        assert any(r.startswith("gems-blanking-v2") for r in requires)


def _installed(name: str) -> bool:
    try:
        md.distribution(name)
    except md.PackageNotFoundError:
        return False
    return True
