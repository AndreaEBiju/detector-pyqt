# PyInstaller spec — portable across macOS / Linux / Windows.
#
# Run via `scripts/build_app.py` (recommended) or directly:
#     pyinstaller app.spec
#
# The spec is written as a Python file so we can branch on
# `sys.platform` and produce native bundles per OS:
#   - macOS  : .app bundle via BUNDLE() with osx-bundle-identifier
#   - Linux  : single-folder dist in dist/detector-pyqt/
#   - Windows: same single-folder dist plus an .exe entry point
#
# Hidden-import / collect-all directives below cover the libraries
# PyInstaller's static analysis tends to miss. If a build fails with
# `ModuleNotFoundError: No module named 'X'` at runtime, add X to the
# `hiddenimports` list.

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

PROJECT_ROOT = Path(SPECPATH).resolve()
DETECTOR_CORE = PROJECT_ROOT / "detector-core"


# --- Data files -----------------------------------------------------------
# Bundle the detector-core submodule's `detector/` package. Use a top-level
# layout in the bundle so `from detector import …` works without sys.path
# tweaks in the frozen build.
datas = [
    (str(DETECTOR_CORE / "detector"), "detector"),
]
# Collect package data (e.g. lightgbm's lib_lightgbm.so, h5py's plugins)
for pkg in ("lightgbm", "h5py", "scipy", "pyqtgraph"):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass


# --- Hidden imports -------------------------------------------------------
# PyInstaller's static analysis can miss:
#   - pyqtgraph's plot-style plugins
#   - PySide6's plugin modules used at runtime
#   - lightgbm's compiled extensions
#   - h5py's HDF5 plugin loaders
hiddenimports = []
for pkg in ("pyqtgraph", "PySide6", "lightgbm", "h5py"):
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass
hiddenimports += [
    "detector",
    "detector.predict",
    "detector.retrain",
    "detector.retrain_subprocess",
    "detector.manifest",
    "detector.recording_io",
    "detector.labeled_save",
    "detector.review",
    "detector.paths",
]


# --- Analysis -------------------------------------------------------------
a = Analysis(
    [str(PROJECT_ROOT / "ui" / "app.py")],
    pathex=[
        str(PROJECT_ROOT),
        str(DETECTOR_CORE),
    ],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Big libraries we don't need at runtime — keeps the bundle small
        "matplotlib",       # Streamlit-side, not used in PyQt
        "streamlit",
        "altair",
        "plotly",
        "tornado",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)


# --- Executable -----------------------------------------------------------
# `console=False` on Windows produces a non-console .exe (no terminal window).
# On macOS this is handled by the BUNDLE step below.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="detector-pyqt",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)


# --- COLLECT --------------------------------------------------------------
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="detector-pyqt",
)


# --- macOS .app bundle ----------------------------------------------------
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Detector — PyQt.app",
        icon=None,                # add path/to/icon.icns later if desired
        bundle_identifier="com.gemsblanking.detector-pyqt",
        info_plist={
            "CFBundleShortVersionString": "0.1.0",
            "CFBundleVersion": "0.1.0",
            # Allow opening recordings via "Open With"
            "CFBundleDocumentTypes": [
                {
                    "CFBundleTypeName": "Recording",
                    "CFBundleTypeRole": "Editor",
                    "LSItemContentTypes": [
                        "public.data",
                        "public.item",
                    ],
                    "CFBundleTypeExtensions": ["mat", "h5"],
                },
            ],
            # Drive folders live in ~/Library/CloudStorage; require
            # macOS to grant the bundle access on first launch.
            "NSDocumentsFolderUsageDescription":
                "Required to read recordings from your Drive folder.",
        },
    )
