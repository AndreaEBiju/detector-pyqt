# detector-pyqt

Desktop PyQt + pyqtgraph UI for the motion-artifact detector. Built
alongside (not replacing) the Streamlit UI in
[GEMSBlanking](https://github.com/AndreaEBiju/GEMSBlanking). Both UIs
import the same detector backend via a git submodule, and both read
model artifacts from the same `~/.detector/` user-data directory.

**Phase status:** M0 (skeleton) — the app opens a blank window and
confirms the detector backend imports. Real functionality lands in
M1 (signal viewer spike) onward. See `pyqt_migration_plan.md` in the
detector-core submodule for the full plan.

## Layout

```
detector-pyqt/
├── ui/                          # the PyQt application
│   ├── app.py                   #   QApplication entry point (M0 stub)
│   ├── windows/                 #   main / training / settings windows
│   ├── widgets/                 #   signal viewer, overview, region table, ...
│   ├── workers/                 #   QThread-based inference / retrain workers
│   ├── dialogs/                 #   file pickers, retrain progress, etc.
│   └── data/                    #   loaders, exporters, cache, settings
├── detector-core/               # git submodule → GEMSBlanking
│   └── detector/                #   imported as `from detector import …`
├── tests/                       # PyQt-specific + integration tests
├── conftest.py                  # adds detector-core/ to sys.path
└── pyproject.toml               # PySide6, pyqtgraph, detector deps
```

## Development setup

```bash
git clone --recurse-submodules <repo-url> detector-pyqt
cd detector-pyqt
pip install -e ".[dev]"
python ui/app.py
```

If you cloned without `--recurse-submodules`:

```bash
git submodule update --init
```

## First-time configuration

Same as the Streamlit repo — point at the shared model folder:

```bash
python -m detector.cli init
```

This writes `~/.detector/config.json` and creates an empty training
manifest. After that, both UIs read from the shared folder
automatically. See `detector-core/DEVELOPER_GUIDE.md` for the
publish-a-new-model workflow.

## Syncing detector improvements

When the detector backend changes in the `GEMSBlanking` repo:

```bash
cd detector-core
git pull origin main
cd ..
git add detector-core
git commit -m "Update detector-core to <sha>"
```

The PyQt UI now sees the updated backend. Output-parity tests (M6 in
the migration plan) catch silent drift between the two UIs.
