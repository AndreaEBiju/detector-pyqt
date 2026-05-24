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

Prerequisite (both platforms): **Python 3.12** installed system-wide.

> **Why a dedicated venv?** Sharing your Anaconda `base` env or the
> system Python with other projects means any unrelated
> `pip install` / `conda update` can silently break this project's
> pinned numpy/scipy versions — and the symptom is often a
> hours-into-a-retrain native crash, not a clean import error. The
> setup scripts below create an isolated `.venv/` inside the repo.
> Other projects on the same machine can update freely without
> touching this one.

### 1.1 macOS — first-time setup

1. **Install Python 3.12** if you don't have it:
   - From [python.org](https://www.python.org/downloads/macos/) (recommended), or
   - Via Homebrew: `brew install python@3.12`

2. **Clone the repo with the detector-core submodule:**
   ```bash
   git clone --recurse-submodules https://github.com/AndreaEBiju/detector-pyqt.git
   cd detector-pyqt
   ```
   If you forgot `--recurse-submodules`:
   ```bash
   git submodule update --init --recursive
   ```

3. **Create the venv and install all dependencies:**
   ```bash
   bash scripts/setup_env.sh
   ```
   The script creates `.venv/`, installs detector-pyqt + detector-core
   editable, pulls every transitive dep (numpy, scipy, h5py, lightgbm,
   matplotlib, hdf5plugin, shap, pyarrow, etc.), and runs a verification
   block at the end that imports `detector.review`, decompresses an
   HDF5 file with the registered filters, and confirms `numpy.linalg.solve`
   works. If any of those fail, setup aborts with a clear message.

4. **Point the tool at the shared model folder** (once per machine):
   ```bash
   source .venv/bin/activate
   python -m detector.cli init
   ```
   At the prompt, paste the path to your Google Drive shared folder,
   e.g.:
   ```
   /Users/<you>/Library/CloudStorage/GoogleDrive-<email>/Shared drives/<workspace>/detector-models
   ```

5. **Launch the UI:**
   ```bash
   bash scripts/run_pyqt.sh
   ```
   This wrapper activates the venv, sets `PYTHONFAULTHANDLER=1` so
   any native crash writes a real stack trace, and starts the app.

### 1.2 Windows — first-time setup

1. **Install Python 3.12** from [python.org](https://www.python.org/downloads/windows/).
   - Check **"Add Python 3.12 to PATH"** in the installer.
   - **IMPORTANT — disable the Microsoft Store Python aliases**: Settings → Apps → Advanced app settings → App execution aliases → toggle **OFF** `python.exe` and `python3.exe`. Otherwise PATH will resolve `python` to a stub that opens the Microsoft Store and breaks subprocess spawning. (See Troubleshooting 11.7.)

2. **Clone the repo with the detector-core submodule** (in PowerShell):
   ```powershell
   git clone --recurse-submodules https://github.com/AndreaEBiju/detector-pyqt.git
   cd detector-pyqt
   ```
   If you forgot `--recurse-submodules`:
   ```powershell
   git submodule update --init --recursive
   ```

3. **Allow PowerShell to run local scripts (one-time, per-user):**
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
   ```
   Type `Y` when prompted. Without this, step 4's setup script will
   fail with **"running scripts is disabled on this system"** —
   Windows ships with `Restricted` execution policy by default, which
   blocks every `.ps1` file. `RemoteSigned` still blocks unsigned
   scripts downloaded from the internet but allows local ones you
   cloned yourself, which is the right setting for development work.
   You only need to do this once per user account.

4. **Create the venv and install all dependencies:**
   ```powershell
   .\scripts\setup_env.ps1
   ```
   Same checks as the macOS script (verifies `detector.review` import,
   HDF5 filter decode, numpy/scipy linear-algebra). If for any reason
   you don't want to change the execution policy permanently, you can
   bypass for just this invocation:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
   ```

4. **Point the tool at the shared model folder** (once per machine):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   python -m detector.cli init
   ```
   At the prompt, paste the path to your Google Drive shared folder.
   On Windows it's usually under your Drive client's sync root, e.g.:
   ```
   G:\Shared drives\<workspace>\detector-models
   ```

5. **Launch the UI:**
   ```powershell
   .\scripts\run_pyqt.ps1
   ```
   Wrapper handles activation + faulthandler + launch.

### 1.3 Activate the env in future shells

After the one-time setup, every new terminal session needs the env activated before running anything (unless you use the launcher scripts which activate for you):

**macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```powershell
.\.venv\Scripts\Activate.ps1
```

Your prompt prefix changes to `(.venv)` to confirm. Run `deactivate` to leave.

### 1.4 Verify the install works end-to-end

The setup scripts already do this, but if you ever want to re-check (for instance after `git pull`), with the env active:

```bash
python -m detector.cli check-retrain --rebuild-dataset --rebuild-phase2
```

This runs the full pre-flight audit (see section 7.4 — Retrain). A `[PASS]` exit means every file, package, and HDF5 filter is in place. A `[FAIL]` lists the specific blockers with remediation commands.

### 1.5 Legacy install — only if you really know what you're doing

If you'd rather install into your existing Anaconda or system Python (skip the venv):

```bash
pip install -e ./detector-core
pip install -e ".[dev]"
```

Be aware that this puts the project at the mercy of any other tool on the machine that touches numpy/scipy/h5py. We've debugged multiple multi-hour outages caused by exactly this — a `conda update` that bumped numpy across a major version boundary, an OS Python shadow that hijacked `pip`, an Anaconda environment that had matplotlib pre-bundled but the venv didn't. **The venv is strongly recommended.**

### 1.6 Pass a recording path on launch

Both launcher scripts accept a recording path to open it directly:

**macOS:**
```bash
bash scripts/run_pyqt.sh path/to/recording.mat
```

**Windows:**
```powershell
.\scripts\run_pyqt.ps1 path\to\recording.mat
```

Or after activating the venv manually:
```bash
python ui/app.py path/to/recording.mat
# also installed as a console script:
detector-pyqt
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

`File → Open recording…` (`Ctrl+O`) — pick a `.mat` or `.h5`
recording file.

For a raw TDT block folder, use `File → Open TDT folder…`
(`Ctrl+Shift+O`) instead — it loads existing pipeline outputs if
present or opens the Preprocess window seeded with that folder.
See section 8 for the full preprocessing flow.

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
- `rebuild_dataset` — rebuild `dataset_phase1.parquet` (raw features
  only). Single-threaded feature extraction takes **~5-10 min per
  recording**; with the default 4-worker parallel pool this scales
  near-linearly with CPU cores. For 18 recordings on a typical 8-core
  laptop, expect **~30-45 minutes** total. Override the worker
  count via the `DETECTOR_PHASE1_WORKERS` env var (default 4, set
  higher if you have a beefy machine with abundant RAM ~1.5 GB per
  worker peak). **Does NOT regenerate Phase 2 — if you added new
  recordings, ALSO check `rebuild_phase2` below or the model will
  silently train on the stale Phase 2 file.**
- `rebuild_phase2` — regenerate `dataset_phase2.parquet` (synthetic
  augmentation; ~2-4 hours). This is the step that actually makes
  newly-added recordings reachable for training. Required whenever
  you've added recordings since the last training round.
- `skip_review` — when UNchecked (default), retrain generates
  per-fold **disagreement-review HTML reports** alongside the
  trained model (see "Phase 5 review HTMLs" subsection below for
  what they are and how to use them). Parallelized across folds:
  ~5-15 min total on a 12-worker machine. Check the box only if
  you're iterating on retrain settings and don't need the review
  HTMLs for every attempt.
- `force_promote` — promote even if regressions are detected
  (requires a reason).

After retrain finishes, a **regression report dialog** pops up
showing old vs new metrics if `regression_report.json` exists.

If you close the training window mid-retrain, the subprocess keeps
going. Re-open the window and it'll auto-attach.

#### Phase 5 review HTMLs (post-training diagnostics)

When `skip_review` is unchecked, retrain produces one HTML report
per recording showing every place where the new model's
predictions **disagreed** with your human-labeled regions during
LORO. Each entry on the page shows:

- A channel plot of the signal segment containing the disputed window
- The model's probability + the LORO held-out probability for that
  window
- SHAP top-20 features explaining the model's decision (red = pushing
  toward "bad", blue = pulling toward "clean")

Use these to spot-check the model's reasoning, find labels you may
want to flip, and audit systematic errors (e.g. "the model keeps
flagging clean stim onset as bad" or "the model misses a class of
real artifacts that look like X").

**Where they land:**

```
<artifacts_dir>/model_v0.X.Y/review/
├── <recording_id_1>_review.html
├── <recording_id_2>_review.html
├── ...
└── <recording_id_N>_review.html
```

For Andrea: `~/.detector/artifacts/model_v0.2.0/review/` (or your
shared Drive's `model_v0.2.0/review/` if `detector init` is pointing
there).

**To view them:** Open any of the HTMLs in a web browser. They're
self-contained — no internet, no Python, no extra files. Drag-and-drop
into Chrome / Safari / Firefox works.

**To re-generate (without retraining the model):** Sometimes you want
to re-run Phase 5 against an existing model — e.g. you've adjusted
labels and want to see how disagreements change. The standalone CLI
does this:

```bash
# In an active venv:
python -m detector.cli review --model <artifact_dir>
# Or pass --fold <recording_id> for just one recording
```

Parallelized the same way the in-retrain path is; auto-detects
worker count from your machine. Pass `--workers N` to override.

**To aggregate review judgments into a summary JSON** (after you've
clicked through HTMLs and saved per-recording judgment JSONs):

```bash
python -m detector.cli review --aggregate \
  --out-dir <artifact_dir>/review/
```

Produces `review_summary.json` with `true_artifact_count`,
`borderline_count`, `false_positive_count`, and rates -- useful for
deciding which judgments to feed back into manifest label updates.

#### Automatic pre-flight check

**Clicking ▶ Retrain now automatically runs the readiness check
first** — you don't have to do anything. The audit takes ~10
seconds and inspects:

- The manifest exists + loads.
- Every recording's `clean.h5`, `bad.h5`, and `baseline.h5` exist at
  the paths the manifest references.
- `dataset_phase1.parquet` and `dataset_phase2.parquet` are present
  (or flagged for rebuild based on your checkboxes).
- Phase 2 covers all the recordings in your manifest (if you
  skipped `rebuild_phase2`).
- Every Python package the pipeline needs imports cleanly —
  numpy, scipy, pandas, sklearn, h5py, lightgbm, matplotlib,
  hdf5plugin, shap, pyarrow.
- End-to-end import smoke test of every `detector.*` submodule the
  retrain touches.
- A real HDF5 decompression test: opens the first manifest
  recording's `clean.h5` and reads a sample. Catches the
  "filter returned failure during read" error class at click time.
- The artifacts directory is writable.
- At least 5 GB free disk.

If any of these fail, the retrain **does not start** — you get a
modal dialog listing the specific blockers with remediation hints
(e.g. "Python module not installed: hdf5plugin → `pip install
hdf5plugin`"). Fix them, click Retrain again. Only when the
pre-flight passes does the actual subprocess spawn.

#### Manual pre-flight check (CLI)

You can also run the same audit from the terminal, for example
on a collaborator's machine right after a data handoff before
they even open the UI:

```bash
python -m detector.cli check-retrain --rebuild-dataset --rebuild-phase2
```

Pass the same flags you intend to use for the actual retrain. A
`[PASS]` exit means everything's lined up; a `[FAIL]` lists every
blocker. Useful for scripting / CI / sanity-checking from a
PowerShell or Terminal session without opening the UI.

#### Safety properties of the retrain flow

A few invariants the pipeline enforces so you can't accidentally
destroy a working model:

- **Auto-bump on version collision.** If your local manifest's
  `current_model_version` is out of sync with the shared artifacts
  directory (e.g. you have `null` but the Drive already has
  `model_v0.1.0`), retrain bumps the patch digit to `v0.1.1` rather
  than overwriting. A loud `WARNING:` line in the log flags the
  auto-bump.
- **Phase 2 mismatch guard.** If `dataset_phase2.parquet` is missing
  recordings from your manifest and you didn't check `rebuild_phase2`,
  retrain refuses to start with a clear remediation message. Pass
  `--skip-phase2-check` only if you knowingly want to train on the
  partial Phase 2.
- **Native-crash visibility.** The subprocess runs with
  `PYTHONFAULTHANDLER=1` so any segfault in numpy / h5py / lightgbm
  writes a C stack trace to the log instead of silently dying.

### Preprocessing tab

Defaults that pre-fill new animal reviews in the Preprocess window
(see section 8). They do **not** override saved per-animal profiles
— each animal's stored profile is authoritative for that animal.

- **Q factor** (default 30) — sharpness of the notch filter. Higher
  Q = narrower notch.
- **Reduction threshold** (default 0.5) — a candidate harmonic is
  kept only if iirnotch + filtfilt actually reduces its band-power
  by at least this fraction. Avoids including useless notches.
- **Max harmonics filtered** (default 4) — cap on how many harmonics
  the cascade includes, ranked by their measured reduction.
- **Detrend** (default on) — subtract the per-channel mean before
  filtering. Recommended; harmless on already-detrended recordings.
- **Candidate harmonics** (default `60, 120, 180, 240, 300`) —
  comma-separated frequencies the auto-detection considers. Add
  50/100/150 for European mains.

The **animal profiles** section lists every saved profile (the
last-updated date, channel count, applied notches per animal). The
**Reset ALL animal profiles** button deletes every file in
`~/.detector/preprocessing_profiles/` after a confirmation. Use it
when your electrode setup or noise profile has changed enough that
you'd rather re-review than override one animal at a time.

### Settings tab

Auto-retrain prompt: when enabled, adding the Nth recording since
the current model triggers a "Retrain now?" prompt and switches the
tabs to Retrain.

---

## 8. Preprocessing TDT data

The Preprocess window converts raw TDT block folders into the
`_sig.mat` / `_vib.mat` / `_stim.mat` / `_notched.mat` files the
rest of the pipeline (labeler, splitter, model) reads. Replaces the
MATLAB preprocessing script some workflows still use.

### When to use it

You have one or more **TDT block folders** (each contains
`.tev / .tsq / .tin` files from Synapse / OpenEx) and want to land
in the labeler with everything denoised and channel-mapped.

### How to open it

Two routes:

- `File → Preprocess TDT data…` — the explicit entry. Opens the
  window empty; you pick folders in step 1.
- `File → Open TDT folder…` (`Ctrl+Shift+O`) — opens a folder
  picker. If the picked folder already contains a `_notched.mat`,
  the labeler opens it directly (skips preprocessing). Otherwise
  the Preprocess window pops up pre-populated with that folder
  so you can run it through the pipeline.

Note: `File → Open recording…` (`Ctrl+O`) is for `.mat`/`.h5`
files only. Qt's native macOS file dialog treats folder clicks as
"descend into," so picking a folder requires the separate `Open
TDT folder…` entry.

### The 7-step flow

The window walks through these steps with **Back / Next** buttons.
You can re-enter any prior step except while a batch is running.

1. **Select folders.** Click `➕ Pick folders…`. Qt's macOS dialog
   only allows single-folder selection, so you add them one at a
   time — for batches up to ~25 folders this is one extra click per
   folder, in exchange for a native picker. Use `Clear` to start
   over.

2. **Configure batch.** A row per folder: pick the condition
   (baseline / stim / recovery / stim_rec), edit the output
   filename prefix (defaults to the folder name), enter the
   **animal ID**. The **Profile?** column shows ✓ if a saved
   profile already exists for that animal — that means we'll reuse
   the channel + notch settings and the per-animal review is just
   a confirmation click.

3. **Channels consistent?** A yes/no question:
   - **Yes** — same electrodes used across every recording in the
     batch. The channel-assignment dialog opens **once** (for the
     first animal) and applies to all profiles built in this batch.
   - **No** — each animal gets its own channel-assignment review.

4. **Per-animal review.** For each animal in the batch order:
   - **Channel assignment** dialog opens (unless the animal has a
     saved profile + you're in "channels consistent" mode):
     - Pick which TDT stream is the raw signal (default: `Raww`).
     - Pick aux streams: stim (`BiPl`), vibration (`adc1`), stim
       envelope (`ADC2`).
     - Per-channel table: include / role (nerve / stomach / other
       / exclude) / label (`VN1`, `Ant1`, …).
     - **Load 5-s sparkline previews** button reads the first few
       seconds of every channel so you can spot dead ones at a
       glance. Constant-zero channels get a red background tint.
     - The summary line shows "X nerve · Y stomach" — yellow if it
       doesn't match the detector's expected 2 + 3, and a
       confirmation dialog warns before Accept.
   - **Notch review** dialog opens:
     - pyqtgraph plot: grey raw + blue notched on a 10-s window of
       the selected channel.
     - **Auto-detect harmonics** runs `detect_significant_harmonics`
       on the middle 60 s and populates the field with the
       candidates whose power reduction passed the threshold.
     - You can edit the harmonics field directly (comma-separated),
       change the Q factor / detrend flag, and the filtered trace
       updates live.
     - **Apply to** radio: by default the filter applies to all
       saved channels; "selected channel only" is a preview mode if
       you want to inspect one channel before committing.
     - Reduction summary shows the per-harmonic % reduction.
   - Profiles save immediately when you Accept the notch step, so
     a mid-batch cancel still leaves the next batch with usable
     defaults.

5. **Confirm.** Per-row preview of what's about to happen:
   `[process]` / `[overwrite existing]` / `[re-process as 'x_v2']`
   / `[skip — already processed]`. If a row's output filename
   prefix already exists in the destination folder, a small modal
   asks you: **Skip** / **Re-process (overwrite)** / **Re-process
   with suffix `_v2`**.

6. **Processing.** Live progress bar + per-file status list. Click
   **Cancel after current file** to stop after the current TDT
   folder finishes (there's no mid-file cancel — the worker can't
   safely interrupt iirnotch).

7. **Report.** Per-row OK / skipped / error summary with the
   produced output paths, validation warnings ("nerve count was 1,
   expected 2"), and an **Open output folder…** button that opens
   one of the produced folders in Finder.

When the batch finishes, the labeler auto-prompts to open the
newest produced `_notched.mat`, so you usually end up straight in
the labeler ready to mark intervals.

### Per-animal profile reuse

Profiles persist at `~/.detector/preprocessing_profiles/<animal_id>.json`.
Adding the same animal to a new batch later picks the saved
channel + notch settings as defaults — the review steps still run
but everything is pre-filled. Manage / inspect / wipe profiles
from the Training window's **Preprocessing** tab.

### Output files

For a row with `output_condition_name = "subj01_bl_1"`, you'll get
the following in the source TDT folder:

- `subj01_bl_1_sig.mat` — raw signal, channel-subsetted (not
  notched). MATLAB variable `signal` shape `(n_ch, N)`, with `fs`
  and `times`.
- `subj01_bl_1_vib.mat` — vibration stream (`vib`, `fs_vib`,
  `times_vib`).
- `subj01_bl_1_stim.mat` — stim stream (`stim`, `fs`, `times`).
- `subj01_bl_1_notched.mat` — notched signal (`y`, `fs`, `times`).
  This is what you open in the labeler.
- `subj01_bl_1_meta.json` — sidecar with channel roles, animal ID,
  condition, notch params + per-harmonic reductions, source TDT
  folder path. Validation warnings (role counts, missing fields)
  surface in the Report screen.

---

## 9. Recording queue (batch workflow)

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

## 10. Keyboard shortcuts

`Help → Keyboard shortcuts` (`F1`) shows the full table inside the
app. Highlights:

| Where | Shortcut | Action |
|---|---|---|
| File menu | `Ctrl+O` / `Ctrl+Shift+O` / `Ctrl+S` / `Ctrl+Shift+S` | Open recording / Open TDT folder / Save / Save as |
| Edit menu | `Ctrl+Z` / `Ctrl+Shift+Z` | Undo / redo |
| View menu | `R` / `F` / `Z` / `X` | Reset zoom / fit all / zoom in / out |
| View menu | `←` / `→` | Step to prev / next interval |
| Tools menu | `Ctrl+T` / `Ctrl+Q` | Training mgmt / Queue panel |
| Inference | `Ctrl+R` | Run inference |
| Review panel | `1` / `2` / `3` | Score true / borderline / FP |
| Viewer | `Shift+drag` | Mark a bad region |
| Help | `F1` | This list |

---

## 11. Troubleshooting

### 11.1 Some menus look empty / missing (macOS)

macOS Qt auto-moves actions named "About" / "Preferences" / "Quit"
into the Application Menu (the leftmost `python3` menu). You'll find
About there if it's not under Help.

### 11.2 Drag doesn't seem to mark a region (both platforms)

Make sure you're holding **Shift** before clicking. Plain drag is
pan; **Shift+drag** marks. The cursor doesn't change but the green
band should appear under your mouse.

### 11.3 Inference is slow (both platforms)

Use `scripts/m1_ingest.py` to convert `.mat` files to flat HDF5
once. The viewer reads from the flat file in chunks; the original
`.mat` layout requires 5× more disk seeks per pan. See
`spike_results.md` for the M1 benchmark data.

### 11.4 Retrain hangs (both platforms)

Check the live log tail in `Tools → Training management → Retrain
tab`. Common stalls: feature extraction on a large recording
(~5-15 min per recording), LORO running across 12+ folds. If
genuinely stuck, click Cancel — the subprocess receives a graceful
signal and exits at the next fold boundary.

### 11.5 Drive folder unreachable (both platforms)

If `paths.get_current_model_version()` returns the shared Drive
path but inference can't actually open `booster.txt`, the file may
be a Drive Desktop placeholder (online-only).

**macOS**: in Finder, right-click the file → "Available offline" /
"Keep on this device". The runtime caches it permanently after that.

**Windows**: in File Explorer, right-click → "Always keep on this
device". Same effect.

### 11.6 `ModuleNotFoundError: No module named 'detector'`

You're not running in the project's venv. Three ways to fix:

- Activate first: `source .venv/bin/activate` (Mac) or
  `.\.venv\Scripts\Activate.ps1` (Windows).
- Or use the launcher scripts (`scripts/run_pyqt.sh` /
  `scripts/run_pyqt.ps1`) — they activate for you.
- Or `cd` to the repo root and run with the explicit venv Python:
  `.venv/bin/python ui/app.py` (Mac) or
  `.\.venv\Scripts\python.exe ui\app.py` (Windows).

### 11.6.5 Windows-only: "running scripts is disabled on this system"

PowerShell refuses to run any `.ps1` file (including `setup_env.ps1`
and `run_pyqt.ps1`) because Windows defaults the execution policy to
`Restricted`. Fix once per user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Type `Y` when prompted. Then re-run whatever script you were trying.

If you'd rather not change the policy at all, you can bypass it for
a single command:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

### 11.7 Windows-only: subprocess errors with `[WinError 11]` or `[WinError 2]`

If you see one of these in the Retrain failed dialog:

- `[WinError 11] An attempt was made to load a program with an incorrect format`
- `[WinError 2] The system cannot find the file specified`

The Microsoft Store's App Execution Alias for `python3.exe` is
hijacking the subprocess spawner. **Disable it:** Settings → Apps →
Advanced app settings → App execution aliases → toggle off both
`python.exe` and `python3.exe`. Then close + reopen the PyQt app.

The aliases are 0-byte stubs that exist only to open the Microsoft
Store. They're a known footgun for any tool that shells out to
`python` via subprocess.

### 11.8 `OSError: Can't synchronously read data (filter returned failure during read)`

The HDF5 file uses a compression filter (LZ4 / zstd / blosc) that
your venv doesn't have the codec for. Install:

```bash
pip install hdf5plugin
```

(With the venv active.) The detector package imports `hdf5plugin`
automatically at startup, so once it's installed, all subsequent
file reads work. This error class is also detected by the
pre-flight check now — if you've pulled latest, you'll see it as a
clear "install hdf5plugin" dialog at Retrain-click time instead.

### 11.9 macOS-only: `Bus error` in scipy.signal / numpy.linalg

If the PyQt UI crashes with `Fatal Python error: Bus error` from
`scipy.signal.sosfiltfilt` or `numpy.linalg.solve`, your numpy and
scipy were built against mismatched BLAS/LAPACK libraries — a
common Anaconda failure mode after a partial `conda update`. Fix
inside the venv:

```bash
source .venv/bin/activate
python -m pip install --force-reinstall --no-cache-dir "numpy<2" "scipy<1.14"
```

Verify:

```bash
python -c "import numpy as np; A=np.array([[1.,2.],[3.,4.]]); print(np.linalg.solve(A, np.array([1.,2.])))"
```

Should print `[0. 0.5]`. If it crashes again, your Python install
itself is corrupted — reinstall Python from python.org.

### 11.10 `Retrain failed` dialog with no obvious error in the log

Three known causes:

1. **Process killed by Windows OOM / antivirus** — check Windows
   Event Viewer → Windows Logs → Application for `python.exe`
   crashes. Resource-Exhaustion-Detector entries indicate OOM.
2. **Process killed mid-Phase-1 by cumulative memory pressure on
   Windows** — fixed in detector-core 40bb541+ (streaming Phase 1
   to disk between recordings). Update with
   `git pull --recurse-submodules`.
3. **Native crash in numpy/scipy** — should print a faulthandler
   stack trace in `.ui_cache/retrain_jobs/<job_id>.log`. If not,
   pull the latest detector-core which enables PYTHONFAULTHANDLER
   in the subprocess env.

For any "no error in log" case, also try running the retrain in
the foreground from a terminal — it produces output in real time
that the log file might not capture:

**macOS:**
```bash
source .venv/bin/activate
python -X faulthandler -m detector.cli retrain --rebuild-dataset --rebuild-phase2 --skip-review
```

**Windows:**
```powershell
.\.venv\Scripts\Activate.ps1
python -X faulthandler -m detector.cli retrain --rebuild-dataset --rebuild-phase2 --skip-review
```

### 11.11 "Python 3.x not found" after a macOS crash dialog

macOS's "Reopen" button on a crash dialog tries to relaunch via a
hardcoded path. If your PATH has changed (e.g. you switched conda
envs or removed a Python install), Reopen fails to find the
original interpreter.

**Don't click Reopen.** Just relaunch from terminal:

```bash
cd /path/to/detector-pyqt
bash scripts/run_pyqt.sh
```

### 11.12 First-time install: the pre-flight check says I'm missing N modules

The `setup_env.sh` / `setup_env.ps1` script declares every required
package, but if you skipped the setup script and installed manually,
or if you're on a fresh venv that pre-dates this week's dep additions,
you might be missing `matplotlib`, `hdf5plugin`, or `pyarrow`. Quickest
fix:

```bash
python -m pip install -r requirements.txt
```

Or just re-run the setup script. Either way, the pre-flight will
keep flagging missing modules until they're all installed.

---

## 12. Where the data lives

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
