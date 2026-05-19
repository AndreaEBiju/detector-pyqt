"""Cross-platform PyInstaller build driver.

Run on each target machine — PyInstaller doesn't cross-build, so you
need a Mac to make the .app bundle, a Linux box for the Linux dist,
and a Windows machine for the Windows .exe.

Usage:
    python scripts/build_app.py                  # build for the host OS
    python scripts/build_app.py --clean          # also rm -rf build/ + dist/
    python scripts/build_app.py --skip-tests     # don't run pytest first

What it does:
  1. Confirm the submodule is present (detector-core/).
  2. Run the unit tests (skip on demand).
  3. Invoke `pyinstaller app.spec`.
  4. Print the path to the resulting bundle for your host platform.

Output:
  - macOS:   dist/"Detector — PyQt.app"
  - Linux:   dist/detector-pyqt/ (run `./detector-pyqt`)
  - Windows: dist\\detector-pyqt\\ (run `detector-pyqt.exe`)

If a build fails with `ModuleNotFoundError: No module named 'X'` when
launching the bundle, add `X` to `hiddenimports` in `app.spec`.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _platform_label() -> str:
    """A human-readable name for the host."""
    sys_p = platform.system()
    if sys_p == "Darwin":
        return f"macOS ({platform.mac_ver()[0]}, {platform.machine()})"
    if sys_p == "Linux":
        return f"Linux ({platform.machine()})"
    if sys_p == "Windows":
        return f"Windows ({platform.release()}, {platform.machine()})"
    return f"{sys_p} ({platform.machine()})"


def _expected_bundle_path() -> Path:
    sys_p = platform.system()
    if sys_p == "Darwin":
        return PROJECT_ROOT / "dist" / "Detector — PyQt.app"
    return PROJECT_ROOT / "dist" / "detector-pyqt"


def _check_prereqs() -> None:
    """Sanity-check the build environment."""
    # Submodule present?
    submod = PROJECT_ROOT / "detector-core" / "detector" / "__init__.py"
    if not submod.exists():
        print(
            "ERROR: detector-core submodule not initialised.\n"
            "       Run `git submodule update --init` from the repo root.",
            file=sys.stderr,
        )
        sys.exit(1)
    # PyInstaller installed?
    try:
        import PyInstaller        # noqa: F401
    except ImportError:
        print(
            "ERROR: PyInstaller not installed in the active Python env.\n"
            "       Run `pip install pyinstaller`.",
            file=sys.stderr,
        )
        sys.exit(1)


def _run_tests() -> None:
    print("running unit tests (skip with --skip-tests) ...")
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=PROJECT_ROOT,
    )
    if res.returncode != 0:
        print("ERROR: tests failed; aborting build", file=sys.stderr)
        sys.exit(2)


def _clean() -> None:
    for d in ("build", "dist"):
        p = PROJECT_ROOT / d
        if p.exists():
            print(f"  rm -rf {p}")
            shutil.rmtree(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true",
                    help="rm -rf build/ + dist/ before building")
    ap.add_argument("--skip-tests", action="store_true",
                    help="skip running the unit-test suite")
    args = ap.parse_args()

    print(f"Detector — PyQt build  ·  host: {_platform_label()}")
    print(f"project: {PROJECT_ROOT}")

    _check_prereqs()
    if args.clean:
        _clean()
    if not args.skip_tests:
        _run_tests()

    print("\ninvoking pyinstaller …\n")
    res = subprocess.run(
        [sys.executable, "-m", "PyInstaller",
         "--noconfirm",
         "app.spec"],
        cwd=PROJECT_ROOT,
    )
    if res.returncode != 0:
        print("ERROR: pyinstaller exited non-zero", file=sys.stderr)
        return res.returncode

    expected = _expected_bundle_path()
    print(f"\nbuild complete. Bundle at: {expected}")
    if platform.system() == "Darwin":
        print("Open with: open \"" + str(expected) + "\"")
    elif platform.system() == "Linux":
        print("Run with:  " + str(expected / "detector-pyqt"))
    elif platform.system() == "Windows":
        print(f"Run with:  {expected}\\detector-pyqt.exe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
