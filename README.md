# detector-pyqt

Desktop PyQt + pyqtgraph UI for the motion-artifact detector. Pairs
with the Streamlit-based [GEMSBlanking](https://github.com/AndreaEBiju/GEMSBlanking)
UI — both share the same detector backend, manifest, and model
artifacts.

**Phase status:** M0–M6 complete. Browser & labeling (M2), inference
+ disagreement review (M3), training management (M4), recording-queue
batch workflow (M5), and PyInstaller packaging + parity tests (M6)
are all in.

For end users, see [USER_GUIDE.md](USER_GUIDE.md).
For maintainers, see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).

## Quick start

```bash
git clone --recurse-submodules https://github.com/AndreaEBiju/detector-pyqt.git
cd detector-pyqt
python3 -m pip install -e ".[dev]"
python3 -m detector.cli init        # one-time: point at the shared model folder
python3 ui/app.py                   # launch
```

Pass a recording path to open it on launch:

```bash
python3 ui/app.py path/to/recording.mat
```

## Layout

```
detector-pyqt/
├── ui/                              PyQt application
│   ├── app.py                       QApplication entry point
│   ├── windows/                     main_window, training_window
│   ├── widgets/                     signal_viewer, overview_strip,
│   │                                region_table, predictions_panel,
│   │                                review_panel, queue_panel
│   ├── workers/                     inference_worker, retrain_worker
│   ├── data/                        settings, queue
│   └── dialogs/                     (empty — dialogs live inline)
├── detector-core/                   git submodule → GEMSBlanking
│   └── detector/                    `from detector import …` consumers
├── scripts/
│   ├── m1_benchmark.py              perf-regression test
│   ├── m1_ingest.py                 .mat → flat HDF5 with overview
│   └── build_app.py                 PyInstaller driver
├── tests/                           pytest unit + parity tests
├── app.spec                         PyInstaller spec (all platforms)
├── spike_results.md                 M1 perf decision log
├── USER_GUIDE.md
└── DEVELOPER_GUIDE.md
```

## Build a distributable bundle

PyInstaller doesn't cross-build — run the build script on each
target machine you need a bundle for:

```bash
python3 scripts/build_app.py            # for the host OS
python3 scripts/build_app.py --clean    # rm -rf build/ + dist/ first
```

Output:
- macOS:   `dist/Detector — PyQt.app` — double-click to launch
- Linux:   `dist/detector-pyqt/detector-pyqt`
- Windows: `dist\detector-pyqt\detector-pyqt.exe`

## Updating the detector backend

When the backend changes in the GEMSBlanking repo:

```bash
cd detector-core
git pull origin main
cd ..
git add detector-core
git commit -m "Bump detector-core to <sha>"
```

The PyQt UI sees the updated backend on its next launch.
Parity tests in `tests/test_parity.py` catch any silent drift.

## Testing

```bash
python3 -m pytest tests/                # full suite (~10s)
python3 -m pytest tests/test_parity.py  # cross-UI contract only
```
