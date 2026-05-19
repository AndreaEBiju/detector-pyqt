# Detector — PyQt developer guide

Notes for whoever is maintaining this repo. Architecture, code map,
how to keep things in sync with the Streamlit UI, how to ship.

## Architecture in one paragraph

The detector backend (model training, inference, manifest, review
generation) lives in the **GEMSBlanking** repo and is consumed by
this repo as a git submodule at `detector-core/`. Both UIs (this PyQt
one and the Streamlit one in GEMSBlanking) import from
`detector-core/detector/`. UI-specific code lives in each UI's own
repo. The split exists so detector improvements made for one UI flow
into the other without code duplication, and so the two UIs can
evolve their UX independently.

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
│   └── dialogs/                     (empty — dialogs are inline)
├── detector-core/                   submodule → GEMSBlanking
│   ├── detector/                    `from detector import …`
│   │   ├── predict.py               detect_bad, detect_bad_with_progress,
│   │   │                            find_latest_model, list_available_versions
│   │   ├── recording_io.py          Recording dataclass, load_recording
│   │   ├── labeled_save.py          save_native, save_matlab_compatible,
│   │   │                            save_period_split, save_segment_table
│   │   ├── retrain_subprocess.py    start_retrain, status, cancel, …
│   │   ├── retrain.py               retrain orchestrator, rollback_to_version
│   │   ├── manifest.py              Manifest class
│   │   ├── review.py                extract_disagreements, compute_shap_for_windows
│   │   ├── paths.py                 path resolution (env > config > fallback)
│   │   └── ...
│   └── ui/                          Streamlit UI (we don't import this)
├── scripts/
│   ├── m1_benchmark.py              perf-regression test for the viewer
│   ├── m1_ingest.py                 .mat → flat HDF5 with overview
│   └── build_app.py                 PyInstaller driver (M6)
├── tests/
│   ├── test_signal_viewer.py        LazyRecording + MultiChannelViewer
│   ├── test_region_table.py
│   ├── test_settings.py
│   ├── test_inference_worker.py
│   ├── test_queue.py
│   └── test_parity.py               M6 parity contract
├── app.spec                         PyInstaller spec
├── spike_results.md                 M1 perf decision log
├── USER_GUIDE.md
├── DEVELOPER_GUIDE.md               (this file)
└── pyproject.toml
```

## Shared-backend boundaries

The following modules in `detector-core/detector/` are imported by
both UIs and MUST stay UI-agnostic (no Streamlit imports, no PySide6
imports):

| Module | What it owns |
|---|---|
| `paths.py` | Per-user paths, three-tier artifacts-dir lookup, `current_model` pointer reading |
| `recording_io.py` | Recording dataclass + load_recording for .mat / chunked HDF5 / flat HDF5 |
| `labeled_save.py` | All four save outputs (clean.h5, bad.h5, .mat, segments.json) + per-period split |
| `predict.py` | detect_bad + detect_bad_with_progress + ProgressCallback + list_available_versions |
| `retrain_subprocess.py` | subprocess-based retrain plumbing (start, status, cancel) |
| `retrain.py` | The orchestrator that retrain_subprocess wraps via the CLI |
| `manifest.py` | Manifest dataclass + load / save / add / remove / history |
| `review.py` | extract_disagreements + compute_shap_for_windows + Disagreement |

Anything UI-specific (Qt signals, Streamlit caching, dock layouts)
lives in the UI repos — not in `detector-core/detector/`.

### Lifting code to the backend

When you write UI-side code that turns out to be UI-agnostic, lift
it to `detector/`. The standard recipe (used four times already in
M2–M4):

1. Move the file into `detector-core/detector/<new_name>.py`.
2. If it's replacing code that already lived in
   `detector-core/ui/data/` or `detector-core/ui/workers/`, leave a
   thin re-export shim there so existing Streamlit imports keep
   working unchanged.
3. Run `pytest` in detector-core to confirm Streamlit-side imports
   are still intact.
4. Commit + push detector-core.
5. In detector-pyqt: `cd detector-core && git pull && cd ..` → 
   `git add detector-core && git commit -m "Bump …"`.

Examples in the repo history:
- M2: `ui/data/loaders.py` → `detector/recording_io.py`
- M2: `ui/data/exporters.py` → `detector/labeled_save.py`
- M3: `detect_bad_with_progress` + `list_available_versions` → `detector/predict.py`
- M4: `ui/workers/training_worker.py` → `detector/retrain_subprocess.py`

### Pinning the submodule

`detector-core` is pinned by SHA in `.gitmodules` (or rather in the
git index — the submodule entry stores a specific commit). When the
backend changes:

```bash
cd detector-core
git pull origin main
cd ..
git add detector-core
git commit -m "Bump detector-core to <sha-prefix> (<what changed>)"
git push
```

The PyQt UI now sees the updated backend on its next launch.

If a backend change breaks something in PyQt (caught by parity tests
in M6.1), pin the submodule back to the previous SHA while you fix
the UI side, then re-bump.

## sys.path quirks

The PyQt UI's `ui/` package and the Streamlit UI's `ui/` package
(inside the submodule) have the same name. If detector-core ends up
EARLIER on sys.path than detector-pyqt's root, `from ui.widgets …`
will find the Streamlit-side `ui` and fail because Streamlit has no
`widgets/` subdir.

Every entry point in this repo handles this with the same idiom:

```python
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.<as-many-parents-as-needed>
_detector_core = _repo_root / "detector-core"
if _detector_core.exists():
    if str(_detector_core) not in sys.path:
        sys.path.insert(0, str(_detector_core))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
```

Order matters: insert detector-core FIRST so it ends up at position 1
in sys.path, THEN insert the repo root which ends up at position 0.
That way `from ui.widgets …` resolves to our PyQt widgets, while
`from detector …` falls through to the submodule.

Affected files (search for "_detector_core"):
- `ui/app.py`
- `ui/windows/main_window.py`
- `ui/windows/training_window.py`
- `ui/widgets/review_panel.py`
- `ui/workers/inference_worker.py`
- `ui/workers/retrain_worker.py`
- `ui/data/settings.py`
- `ui/data/queue.py`
- `scripts/m1_benchmark.py`
- `scripts/m1_ingest.py`

If you add a new module that imports from `detector.*`, include this
boilerplate at the top.

## Threading model

| Concern | Pattern |
|---|---|
| Long-running inference (~30s–3min) | `QThread` + `QObject` (`ui/workers/inference_worker.py`'s `InferenceWorker`) |
| Long-running retrain (~10–60min) | **Subprocess** (`ui/workers/retrain_worker.py`'s `RetrainMonitor` polls a child process spawned by `detector.retrain_subprocess.start_retrain`) |
| File I/O during open | Synchronous on the main thread (acceptable because lazy h5py reads are <100ms for small windows) |
| Plot paints during pan/zoom | pyqtgraph's `autoDownsample=True` keeps these on the main thread but bounded by visible-pixel count |

The subprocess pattern for retrain exists for crash isolation: a
retrain that segfaults during LORO doesn't take down the UI, and
the user can re-open the training window and re-attach to the
in-progress job.

## Testing

```bash
python3 -m pytest tests/                # full suite
python3 -m pytest tests/test_parity.py  # just the cross-UI contract
```

Tests divide into:
- **Pure-Python** unit tests (loaders, exporters, settings, queue,
  region helpers) — run anywhere, fast.
- **Contract / parity** tests (`tests/test_parity.py`) — verify
  shared-backend imports + schema compatibility with the Streamlit
  UI. Pin the architecture.
- **Manual-only Qt** tests (`tests/test_inference_worker.py::
  test_inference_worker_signals_fire_and_intervals_offset`) — gated
  behind `@pytest.mark.skipif(True)` because headless CI segfaults
  on QApplication. Run manually on a Mac terminal with display.

The interactive UI work (drag-to-mark, scroll-wheel zoom, etc.) is
covered by the manual acceptance checklist in each phase's commit
message + `spike_results.md`'s subjective UX notes.

## Adding a new feature

Typical flow:

1. **Decide what's UI-agnostic vs UI-specific.** If the new logic
   could be useful to the Streamlit UI too, write it in
   `detector-core/detector/` (or extract during the next lift cycle).
2. **Add the widget / window / worker** in `ui/`. Follow the
   sys.path boilerplate at the top.
3. **Wire Qt signals** between widgets — never call across widgets
   directly. The main window is the orchestrator.
4. **Write tests** for any pure-Python piece. Skip pytest-qt unless
   the interaction is hard to verify any other way.
5. **Update USER_GUIDE.md** if the change is user-visible.
6. **Update parity tests** in `test_parity.py` if you've added a
   contract that should stay stable across both UIs.

## Packaging (M6)

```bash
python3 scripts/build_app.py           # build for the host OS
python3 scripts/build_app.py --clean   # rm -rf build/ + dist/ first
```

Outputs:
- macOS: `dist/Detector — PyQt.app` (double-click to launch)
- Linux: `dist/detector-pyqt/detector-pyqt` (executable)
- Windows: `dist\detector-pyqt\detector-pyqt.exe`

PyInstaller doesn't cross-build. You need a Mac to make the .app,
a Linux box (or container) for the Linux bundle, a Windows box
for the .exe. The same `app.spec` works on all three — it branches
on `sys.platform` for native packaging differences.

If the bundle fails to launch with `ModuleNotFoundError: No module
named 'X'`, add `X` to the `hiddenimports` list at the top of
`app.spec`. Common culprits: pyqtgraph plot styles, h5py plugin
loaders, lightgbm's compiled lib.

## Detector-core release process

When you train and promote a new model:

```bash
# 1. Train locally
python3 -m detector.cli retrain --rebuild-dataset

# 2. Copy artifacts to the shared Drive folder
SHARED=$(python3 -c "from detector.paths import get_config; print(get_config().get('shared_artifacts_path', ''))")
cp -r ~/.detector/artifacts/model_v0.X.Y "$SHARED/"

# 3. Update the pointer
echo "model_v0.X.Y" > "$SHARED/current_model"

# 4. Drive syncs to all collaborators automatically
```

Every collaborator's next inference run reads the new pointer. No
restart, no app update.

To roll back: edit `current_model` back to the older version. Same
shared-folder edit, same auto-propagation.

## Cross-repo CI

If a backend change ever breaks PyQt without breaking Streamlit
(or vice versa), the parity tests in `test_parity.py` catch it on
the next test run. Set up CI in both repos to run the parity tests
on every PR.

## What this repo deliberately doesn't do

- **Web deployment.** Desktop app only. If lab-wide multi-user access
  ever becomes a need, Streamlit-on-server is the answer (see the
  Streamlit repo's Phase 9 docs). The PyQt UI stays single-user-
  desktop.
- **Auto-update the bundle.** No Sparkle, no auto-updater. You
  re-run `scripts/build_app.py` on each platform when a new
  release is cut.
- **Bundle the model artifacts.** Models live in the shared Drive
  folder (configured per machine via `detector init`). Bundling
  would make distribution awkward when the lab promotes a new
  model — collaborators would need a new bundle just to get the
  new weights.
