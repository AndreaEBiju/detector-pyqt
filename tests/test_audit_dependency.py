"""gems_blanking_v2 and detector-core are declared, installed dependencies - not path hacks.

Invariant 21: a library resolved by ``sys.path`` order is an accident, not a
dependency. Both used to be found by inserting a sibling or submodule path; both
are now installed editable from their one checkout, and these tests keep it so.
"""

from __future__ import annotations

import importlib.metadata as md
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from ui.audit import bridge

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "detector-core"


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


def test_no_source_puts_detector_core_on_sys_path() -> None:
    """detector-core is declared in pyproject and installed. Any module inserting its
    path re-creates resolution by path order the moment it is imported - which is
    why removing only pytest's ``pythonpath`` entry would have changed nothing."""
    pattern = re.compile(r"sys\.path\.(insert|append)\(.*(detector_core|DETECTOR_CORE|detector-core)")
    sources = [p for sub in ("ui", "tests", "scripts") for p in (ROOT / sub).rglob("*.py")]
    offenders = [p.relative_to(ROOT).as_posix() for p in [*sources, ROOT / "conftest.py"]
                 if pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert not re.search(r"^pythonpath\s*=", pyproject, re.MULTILINE)
    assert re.search(r'^\s*"detector-core",', pyproject, re.MULTILINE)


def test_detector_comes_from_the_one_installed_checkout() -> None:
    import detector

    assert md.distribution("detector-core") is not None
    assert Path(detector.__file__).resolve().is_relative_to(CORE.resolve())


def _fresh(probe: str) -> str:
    """Run ``probe`` in a fresh interpreter outside the repo; return its last line.

    Fresh, so this test process's own ``sys.path`` cannot make a check pass.
    """
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         cwd=ROOT.parent, env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
                         timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


def test_our_ui_wins_over_detector_cores_streamlit_ui_in_a_fresh_launch() -> None:
    """detector-core's editable install exposes its whole checkout, Streamlit ``ui/``
    included. Launched as Andrea launches it (``python ui/app.py``), ``ui`` must
    still be ours."""
    app = str(ROOT / "ui" / "app.py")
    line = _fresh(
        "import runpy\n"
        f"runpy.run_path({app!r}, run_name='not_main')\n"
        "import ui, ui.widgets\n"
        "print(ui.__file__)\n"
    )
    resolved = Path(line).resolve()
    assert resolved.is_relative_to((ROOT / "ui").resolve())
    assert "detector-core" not in resolved.parts


def test_importing_every_ui_module_leaves_detector_core_where_the_install_put_it() -> None:
    """Behavioural, so no spelling evades it: import every ui module in a fresh
    interpreter; detector-core's position on sys.path must not change."""
    line = _fresh(
        "import importlib, json, os, pkgutil, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        f"core = os.path.normcase({str(CORE)!r})\n"
        "where = lambda: [i for i, p in enumerate(sys.path) if os.path.normcase(p) == core]\n"
        "before = where()\n"
        "import ui\n"
        "for m in pkgutil.walk_packages(ui.__path__, 'ui.'):\n"
        "    if m.name != 'ui.app':\n"
        "        importlib.import_module(m.name)\n"
        "print(json.dumps([before, where()]))\n"
    )
    before, after = json.loads(line)
    assert after == before
