# Detector — PyQt user guide

A desktop tool for motion-artifact labeling, inference, review, and
retraining. Pairs with the Streamlit-based GEMSBlanking UI — both
share the same detector backend, manifest, and model artifacts, so
you can switch between them at will.

This guide covers the PyQt UI specifically. For backend internals
(how detection works, training pipeline, manifest schema) see
`DEVELOPER_GUIDE.md` and the per-phase docs in the GEMSBlanking repo.

---

## 1. First-time setup

### Install

You need **Python 3.12**. From the repo root:

```bash
git clone --recurse-submodules https://github.com/AndreaEBiju/detector-pyqt.git
cd detector-pyqt
python3 -m pip install -e ".[dev]"
```

If you cloned without `--recurse-submodules`:

```bash
git submodule update --init
```

### Point the tool at the shared model folder

Run once per machine:

```bash
python3 -m detector.cli init
```

You'll be prompted for the path to the lab's shared Drive folder
(typically something like `~/Library/CloudStorage/GoogleDrive-…/
Shared drives/<workspace>/detector-models`). The init command:

- Writes `~/.detector/config.json` with the shared path.
- Creates an empty `~/.detector/training_manifest.json` if you don't
  have one yet.
- Reports which model version is currently active (read from the
  shared folder's `current_model` text file).

This is the same setup the Streamlit UI uses — share it.

### Launch

```bash
python3 ui/app.py
# or, after install, just:
detector-pyqt
```

Pass a recording path to open it directly:

```bash
python3 ui/app.py path/to/recording.mat
```

---

## 2. Main window tour

The window is laid out as:

```
┌─────────────────────────────────────────────────────────────┐
│ menu bar: File | Edit | View | Tools | Help                 │
├─────────────────────────────────────────────────────────────┤
│ toolbar: Open · Save | zooms | Model:[v0.1.0★] Run inf …    │
├──────────────────────────┬──────────────────────────────────┤
│                          │ Intervals (tabbed)               │
│                          │ ┌──────────────┬──────────────┐ │
│                          │ │ Marked (N)   │ Model (M)    │ │
│  Multi-channel viewer    │ ├──────────────┴──────────────┤ │
│  (5 channels, lazy)      │ │ #  start    end   Δ  src   │ │
│                          │ │ 0  00:01.23 …               │ │
│                          │ │ …                           │ │
│                          │ │                             │ │
│                          │ │                             │ │
│                          │ └─────────────────────────────┘ │
├──────────────────────────┴──────────────────────────────────┤
│ Overview strip (whole recording, viewport indicator)        │
├─────────────────────────────────────────────────────────────┤
│ status bar: recording info · marked counts · stim_end · ●   │
└─────────────────────────────────────────────────────────────┘
```

Optional dock panels you can toggle:

- **Queue** (`Tools → Recording queue`, `Ctrl+Q`) — left dock, lists
  recordings you're working through as a batch.
- **Disagreement review** (toolbar `🔍 Review` after inference) — right
  dock, tabbed with the intervals panel.

---

## 3. Daily workflow

### Open a recording

`File → Open` (`Ctrl+O`) — pick a `.mat` or `.h5` recording.

For best performance, ingest your `.mat` files to flat HDF5 once:

```bash
python3 scripts/m1_ingest.py path/to/recording.mat
```

Then open the resulting `recording.mat.flat.h5`. The ingested file
includes a pre-downsampled overview for fast wide-zoom navigation
(see the [M1 spike notes](spike_results.md)).

### Mark a bad region

**Hold Shift** and drag horizontally on any channel of the main
viewer. A green pending band appears as you drag; on release it
turns red and shows up in the **Marked** intervals tab on the right.

Plain drag (no Shift) pans the viewport. Scroll-wheel zooms.

### Navigate

- **Drag the viewport rectangle** on the bottom overview strip.
- **Step buttons** in the toolbar: Reset zoom (0-60s), Fit all,
  Zoom in (×0.5), Zoom out (×2).
- **Keyboard**:
  - `R` — reset zoom (0-60s)
  - `F` — fit all
  - `Z` / `X` — zoom in / out
  - `←` / `→` — jump to previous / next marked interval

### Edit marked intervals

Right-side **Marked** tab:

- **Double-click a row** to jump the viewer to that interval.
- **Delete** key (or right-click → Delete) removes the selected row.
- Source tags (user / model_accepted / existing) are shown — you can
  see at a glance which intervals you drew vs which came from the
  model vs which were loaded from disk.

`Edit → Undo` (`Ctrl+Z`) / `Edit → Redo` (`Ctrl+Shift+Z`) — 20-deep
stack.

### Stim/recovery boundary

For `stim_rec` recordings, a dashed line marks the boundary:

- **Drag it** on any channel to adjust (all channels stay in sync).
- **Edit → Set stim/recovery boundary…** opens a numeric entry.
- Current value is shown in the status bar at the bottom.

---

## 4. Inference

### Run

Toolbar **▶ Run inference** (`Ctrl+R`) — kicks off a background job.
A progress dialog tracks four phases: feature extraction → scoring →
post-processing → done. Predictions appear as orange bands.

**Auto-run on open** (Edit menu, default ON) — opens fire inference
automatically right after loading. Toggle off to inspect raw signal
first.

**Stim-skip toggle** (Edit menu, default ON) — for `stim_rec`
recordings with a boundary set, only the recovery portion is
scored. Faster, and avoids stim artifacts confusing the
motion-artifact model. The Run Inference toast tells you when this
fires (`scope = recovery-only`).

### Model version

The **Model:** dropdown in the toolbar lists every `model_v*`
directory in your artifacts dir. The ★ marker shows the version
named by the shared `current_model` pointer.

The dropdown auto-picks your last-used version on next launch
(persisted in `~/.detector/pyqt_settings.json`).

### Predictions panel

Right-side **Model** tab fills in after inference:

- One row per predicted interval with probability.
- **Accept** moves it to Marked (source=`model_accepted`) and removes
  from the Model tab.
- **Dismiss** removes without marking. Predictions still in this tab
  at save time become the `model_unsure` set (see Save section).
- **Accept ALL** bulk-adds everything.
- **Double-click** jumps the viewer.

---

## 5. Disagreement review

After inference, toolbar **🔍 Review** opens a dockable review panel
on the right. The panel walks you through the top-20 disagreements
(predictions outside your human-marked regions) one at a time:

- **Context plot**: 5 channels, ±2s around the disagreement. Orange
  band marks the model's 100ms window.
- **SHAP**: top-3 features driving the prediction, color-coded
  (red = pushing toward bad, blue = pulling away).
- **Score**: click a button or press `1` (true artifact) / `2`
  (borderline) / `3` (false positive).
- **Notes**: per-disagreement free-text field.
- **Navigate**: `←` / `→` arrows or the buttons.
- **Drag-to-mark-wider**: if the model's 100ms detection doesn't
  cover the full artifact, **Shift+drag** on the context plot to
  define a wider region. Click "🎯 Mark this wider region instead" —
  it adds your selection to Marked, drops the model prediction, and
  advances to the next disagreement.
- **💾 Save review JSON** writes
  `~/.detector/reviews/<recording_id>_review.json` in the same
  schema the Streamlit UI uses, so the standalone Phase-5
  aggregator pools both.

---

## 6. Saving

`File → Save` (`Ctrl+S`) writes four artifacts next to the source
recording:

| File | What's in it |
|---|---|
| `<id>_clean.h5` / `<id>_bad.h5` | splitter HDF5 pair — chunked clean + bad segments |
| `<id>_blankmotion.mat` | MATLAB v5 with `yOut` (signal with bad ranges zeroed), `t`, `fs`, `removedSegmentIdx`, and per-source arrays `removedSegmentIdx_user / _model_accepted / _existing / _model_unsure` |
| `<id>_stim_blankmotion.mat` + `<id>_recovery_blankmotion.mat` | per-period splits when a stim/recovery boundary is set — each file's `yOut` and indices are rebased to start at 1 inside the portion |
| `<id>_segments.json` | flat segment table with full provenance: `start_sample`, `end_sample`, `start_sec`, `end_sec`, `duration_sec`, `source`, `accepted` per row |

`File → Save As` (`Ctrl+Shift+S`) lets you pick the output directory.

Source values in `bad_sources`:
- `user` — manually drawn in the viewer
- `model_accepted` — model-predicted, you clicked Accept
- `existing` — loaded from the recording's sibling `_bad.h5` on open
- `model_unsure` — only in the `removedSegmentIdx_model_unsure` /
  `segments_unsure[]` slots. Predictions you neither accepted nor
  dismissed at save time.

---

## 7. Training management

`Tools → Training management…` (`Ctrl+T`) opens a separate window
with four tabs:

### Manifest tab

Lists all recordings in `~/.detector/training_manifest.json`.

- **➕ Add recording…** — form-based add. Auto-fills `recording_id`,
  sibling `bad.h5`, `fs`, `n_samples`, `n_channels` from the picked
  `clean.h5` so you only type what we couldn't infer.
- **— Remove selected** — requires typing the recording_id twice to
  confirm. Files on disk are NOT deleted.

### Versions tab

Lists every `model_v*` directory in the artifacts folder with
provenance metrics (recall on real positives + synthetic, mean bad
fraction, per-gate pass counts):

- **📄 View provenance JSON** — read-only viewer for the full
  `provenance.json`.
- **⟲ Roll back…** — point `current_model_version` at a different
  version. Requires a reason (audited in the manifest history).
  Feature-schema mismatches are caught — you can override if you've
  verified the schemas are equivalent.

### Retrain tab

Spawns `python -m detector.cli retrain …` as a subprocess (crash-
isolated from the UI — even if the retrain segfaults, the UI keeps
running). Progress bar + live log tail update every ~500 ms.

Controls:

- `w_neg` — per-row weight for unlabeled-clean rows.
- `seed` — RNG seed.
- `no_loro` — reuse cached LORO summary (only safe if the manifest
  hasn't changed since the last LORO).
- `rebuild_dataset` — rebuild `dataset_phase1.parquet` (~25 min;
  required after adding recordings).
- `skip_review` — don't regenerate Phase-5 review HTMLs.
- `force_promote` — promote even if regressions are detected
  (requires a reason).

After retrain finishes, a **regression report dialog** pops up
showing old vs new metrics if `regression_report.json` exists.

If you close the training window mid-retrain, the subprocess keeps
going. Re-open the window and it'll auto-attach.

### Settings tab

Auto-retrain prompt: when enabled, adding the Nth recording since
the current model triggers a "Retrain now?" prompt and switches the
tabs to Retrain.

---

## 8. Recording queue (batch workflow)

`Tools → Recording queue` (`Ctrl+Q`) toggles a queue dock on the
left. Use this when working through many recordings:

1. **New queue** then **➕ Add folder…** — scans a folder for `.h5`
   files. They appear with `⏳ pending`.
2. **Open next pending** opens the next file in the main viewer. On
   Save the item flips to `✓ done`.
3. **⏯ Run inference on all pending** — sequential bulk job: open
   each pending file, auto-run inference, auto-save, mark done,
   move to next. Cancellable from the progress dialog.
4. Queues persist at `~/.detector/queues/<queue_id>.json`. Restart
   the app and **Load queue…** to resume.

Per-row right-click menu: Open, Mark done, Mark skipped (with
reason), Reset to pending, Remove from queue.

---

## 9. Keyboard shortcuts

`Help → Keyboard shortcuts` (`F1`) shows the full table inside the
app. Highlights:

| Where | Shortcut | Action |
|---|---|---|
| File menu | `Ctrl+O` / `Ctrl+S` / `Ctrl+Shift+S` | Open / Save / Save as |
| Edit menu | `Ctrl+Z` / `Ctrl+Shift+Z` | Undo / redo |
| View menu | `R` / `F` / `Z` / `X` | Reset zoom / fit all / zoom in / out |
| View menu | `←` / `→` | Step to prev / next interval |
| Tools menu | `Ctrl+T` / `Ctrl+Q` | Training mgmt / Queue panel |
| Inference | `Ctrl+R` | Run inference |
| Review panel | `1` / `2` / `3` | Score true / borderline / FP |
| Viewer | `Shift+drag` | Mark a bad region |
| Help | `F1` | This list |

---

## 10. Troubleshooting

### Some menus look empty / missing

macOS Qt auto-moves actions named "About" / "Preferences" / "Quit"
into the Application Menu (the leftmost `python3` menu). You'll find
About there if it's not under Help.

### Drag doesn't seem to mark a region

Make sure you're holding **Shift** before clicking. Plain drag is
pan; **Shift+drag** marks. The cursor doesn't change but the green
band should appear under your mouse.

### Inference is slow

Use `scripts/m1_ingest.py` to convert `.mat` files to flat HDF5
once. The viewer reads from the flat file in chunks; the original
`.mat` layout requires 5× more disk seeks per pan. See
`spike_results.md` for the M1 benchmark data.

### Retrain hangs

Check the live log tail in `Tools → Training management → Retrain
tab`. Common stalls: feature extraction on a large dataset (~25
min), LORO running on 12+ folds. If genuinely stuck, click Cancel —
the subprocess receives SIGTERM and exits at the next fold
boundary.

### Drive folder unreachable

If `paths.get_current_model_version()` returns the shared Drive
path but inference can't actually open `booster.txt`, the file may
be a Drive Desktop placeholder (online-only). Force a local
download: in Finder, right-click the file → "Available offline" /
"Keep on this device". The runtime caches it permanently after that.

### `ModuleNotFoundError: No module named 'detector'`

You're not in a context where Python can see the
`detector-core/detector/` submodule. Either:

- Run from the repo root (`cd /path/to/detector-pyqt; python3 ui/app.py`)
- Or `pip install -e .` once (after which `detector-pyqt` runs anywhere)

---

## 11. Where the data lives

```
~/.detector/                           per-user state
├── config.json                        detector init writes this
├── pyqt_settings.json                 PyQt UI prefs
├── settings.json                      Streamlit UI prefs (separate)
├── training_manifest.json             your training corpus ledger
├── artifacts/                         local-fallback model artifacts
│   ├── current_model                  pointer to active version
│   └── model_v0.x.x/                  trained models
├── reviews/                           disagreement-review JSON files
├── queues/                            recording-queue JSON files
└── cache/                             overview-strip caches, etc.

<Drive shared folder>/detector-models/  shared model artifacts
├── current_model                       pointer the whole lab reads
├── model_v0.1.0/
├── model_v0.2.0/
└── …                                   one folder per promoted version
```

The path-resolution layer (`detector.paths`) prefers the shared
folder when configured, falls back to `~/.detector/artifacts/` if
Drive isn't reachable. Override either with the env vars
`$DETECTOR_HOME` (relocate per-user state) or `$DETECTOR_ARTIFACTS`
(pin a specific local artifacts directory, useful for testing).
