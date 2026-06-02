"""Generate publication-quality plots describing the detector model.

Reads from the detector artifacts directory and writes PNGs to
./plots_out/ (override via --out-dir). Each plot function is
independent; comment out the calls at the bottom to skip ones you
don't want, or pass --only to run a subset.

Plots produced (matching the codes in the spec list):
    A1 / F4 : feature_importance (one per model -- combined + per-animal)
    A3 / F3 : synth_type_distribution
    B1      : per_animal_vs_combined_recall
    B3      : per_animal_bad_fraction_human_vs_model
    B4      : per_animal_gate_pass_rate
    B5      : per_animal_recall_vs_bad_fraction
    C1      : hyperopt_optimization_history
    C2      : hyperopt_param_importance
    C3      : hyperopt_best_params_radar
    C3-alt  : hyperopt_3d_trial_scatter
    C4      : hyperopt_parallel_coordinate (copies existing PNGs)
    F1      : combined_corpus_composition
    F2      : phase1_vs_phase2_size
    G1      : combined_per_recording_recall
    G2      : combined_bad_fraction_human_vs_model
    G3      : combined_gates_heatmap
    G4      : calibration_plot   [requires inference; use --with-inference]
    G5      : pr_curve            [requires inference; use --with-inference]
    H1      : combined_recall_progression
    H2      : combined_gate_progression
    H3      : combined_threshold_history
    I1      : active_learning_progression
    I2      : auto_fn_effectiveness

Usage:
    python scripts/make_model_plots.py
    python scripts/make_model_plots.py --out-dir results/plots
    python scripts/make_model_plots.py --with-inference  # also runs G4 + G5
    python scripts/make_model_plots.py --only B1,C3,H1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# Wire detector-core onto the path so this script runs from a fresh
# Python without needing the package installed.
_repo_root = Path(__file__).resolve().parent.parent
_detector_core = _repo_root / "detector-core"
if _detector_core.exists() and str(_detector_core) not in sys.path:
    sys.path.insert(0, str(_detector_core))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from detector import paths as detector_paths  # noqa: E402

# ------------------------------------------------------------------
# Constants & style
# ------------------------------------------------------------------

ANIMALS = ["F", "J", "L", "O"]
ANIMAL_COLORS = {
    "F": "#1f77b4",   # blue
    "J": "#ff7f0e",   # orange
    "L": "#2ca02c",   # green
    "O": "#d62728",   # red
}
GATE_COLORS = {"pass": "#4caf50", "fail": "#e53935"}

# Synth-type column is _synth_type in dataset_phase2.parquet (with
# underscore prefix because Phase 2 also has a "type" column for
# something else).
SYNTH_TYPE_COL = "_synth_type"

# Hyperopt parameter search bounds (matches the UI defaults). Used
# for normalizing the radar plot. If you widened ranges in the UI,
# update here too.
HYPEROPT_BOUNDS = {
    "w_neg":      (0.001, 100.0),  # log-sampled
    "fp_weight":  (1.0,    10.0),
    "fn_weight":  (1.0,    10.0),
}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.framealpha": 0.9,
})


# ==================================================================
# Helpers
# ==================================================================

def _safe(fn):
    """Decorator: catch+log exceptions per plot so one failure
    doesn't abort the whole script."""
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(f"[{fn.__name__}] FAILED: {type(e).__name__}: {e}",
                  flush=True)
            traceback.print_exc()
            return None
    wrapped.__name__ = fn.__name__
    return wrapped


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_path = out_dir / f"{name}.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> wrote {out_path.name}")


def list_versions(parent_dir: Path) -> list[Path]:
    """Return all model_v*/ dirs sorted by version (oldest first).
    Skips _archived_* and any non-model_v* siblings."""
    if not parent_dir.exists():
        return []
    return sorted(
        [p for p in parent_dir.iterdir()
         if p.is_dir() and p.name.startswith("model_v")],
        key=lambda p: p.name,
    )


def latest_version(parent_dir: Path) -> Optional[Path]:
    versions = list_versions(parent_dir)
    return versions[-1] if versions else None


def load_provenance(model_dir: Path) -> Optional[dict]:
    p = model_dir / "provenance.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"  warn: couldn't read {p}: {e}")
        return None


def load_threshold(model_dir: Path) -> Optional[float]:
    p = model_dir / "threshold.json"
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text())["threshold"])
    except Exception:
        return None


def get_combined_versions(artroot: Path) -> list[Path]:
    """Combined model versions live directly under artifacts root."""
    return list_versions(artroot)


def get_per_animal_versions(artroot: Path, animal: str) -> list[Path]:
    return list_versions(artroot / "per_animal" / animal)


def get_hyperopt_dir(artroot: Path, animal: str) -> Path:
    return artroot / "hyperopt_per_animal" / animal / "hyperopt"


def short_rid(rid: str, max_len: int = 30) -> str:
    """Compact recording id for tight x-tick labels."""
    if len(rid) <= max_len:
        return rid
    return rid[: max_len - 3] + "..."


def animal_color_for_rid(rid: str) -> str:
    # Best-effort: identify animal from rid (uppercase letter, often _Jel_/_JEL_).
    for a in ANIMALS:
        if f"_{a}EL_" in rid.upper() or f"_{a}el_".lower() in rid.lower():
            return ANIMAL_COLORS[a]
        if rid.upper().startswith(f"{a}_") or rid.upper().startswith(f"{a}EL"):
            return ANIMAL_COLORS[a]
    return "#888"


# ==================================================================
# A1 / F4 - Feature importance
# ==================================================================

@_safe
def plot_feature_importance(model_dir: Path, out_dir: Path,
                              tag: str, top_n: int = 20) -> None:
    print(f"[plot_feature_importance:{tag}] {model_dir.name}")
    try:
        import lightgbm as lgb
    except ImportError:
        print("  skip: lightgbm not installed")
        return
    booster_path = model_dir / "booster.txt"
    if not booster_path.exists():
        print(f"  skip: no booster.txt at {booster_path}")
        return
    booster = lgb.Booster(model_file=str(booster_path))
    importances = booster.feature_importance(importance_type="gain")
    # Feature column names (in the same order)
    feat_path = model_dir / "feature_columns.json"
    if feat_path.exists():
        feat_names = json.loads(feat_path.read_text())
    else:
        feat_names = [f"f{i}" for i in range(len(importances))]
    order = np.argsort(importances)[::-1][:top_n]
    names = [feat_names[i] for i in order]
    vals = importances[order]
    fig, ax = plt.subplots(figsize=(7, 0.32 * top_n + 1))
    y = np.arange(len(names))[::-1]   # so largest is at top
    ax.barh(y, vals, color="#4c72b0")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Feature importance (gain)")
    ax.set_title(f"Top {top_n} features — {tag}")
    for i, v in zip(y, vals):
        ax.text(v, i, f"  {v:,.0f}", va="center", fontsize=7)
    _save(fig, out_dir, f"A1_F4_feature_importance_{tag}")


# ==================================================================
# A3 / F3 - Synthetic positive type distribution
# ==================================================================

@_safe
def plot_synth_type_distribution(phase2_path: Path, out_dir: Path,
                                    tag: str) -> None:
    print(f"[plot_synth_type_distribution:{tag}] {phase2_path.parent.name}")
    if not phase2_path.exists():
        print(f"  skip: no parquet at {phase2_path}")
        return
    try:
        import pandas as pd
        df = pd.read_parquet(
            phase2_path,
            columns=["trust_level", SYNTH_TYPE_COL] if _has_col(phase2_path) else None,
        )
    except Exception as e:
        # Fall back to reading the whole thing.
        import pandas as pd
        df = pd.read_parquet(phase2_path)
        if SYNTH_TYPE_COL not in df.columns:
            print(f"  skip: column {SYNTH_TYPE_COL} not in parquet")
            return
    synth = df[df["trust_level"] == "synthetic_positive"]
    if synth.empty:
        print("  skip: no synthetic_positive rows")
        return
    counts = synth[SYNTH_TYPE_COL].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]
    bars = ax.bar(counts.index, counts.values,
                  color=colors[: len(counts)])
    ax.set_ylabel("Count")
    ax.set_title(f"Synthetic positive type distribution — {tag}\n"
                 f"(total synth: {counts.sum():,})")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v,
                f"{v}\n({v / counts.sum() * 100:.1f}%)",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, counts.max() * 1.15)
    _save(fig, out_dir, f"A3_F3_synth_type_{tag}")


def _has_col(parquet_path: Path) -> bool:
    """Best-effort: peek at parquet schema."""
    try:
        import pyarrow.parquet as pq
        return SYNTH_TYPE_COL in pq.read_schema(parquet_path).names
    except Exception:
        return False


# ==================================================================
# B1 - Per-animal vs combined recall (the architecture money plot)
# ==================================================================

@_safe
def plot_per_animal_vs_combined_recall(artroot: Path,
                                          out_dir: Path) -> None:
    print("[plot_per_animal_vs_combined_recall]")
    combined_latest = latest_version(artroot)
    if combined_latest is None:
        print("  skip: no combined model")
        return
    combined_prov = load_provenance(combined_latest) or {}
    cls_combined = combined_prov.get("loro_summary", {}).get(
        "per_recording", {})
    if not cls_combined:
        print("  skip: combined model has no per_recording metrics")
        return
    # For each animal: average recall on that animal's recordings,
    # computed from the COMBINED model + the per-animal model.
    combined_avg = {}
    per_animal_avg = {}
    rid_animals = _load_rid_to_animal_map(artroot)
    for a in ANIMALS:
        rids = [rid for rid, an in rid_animals.items() if an == a]
        if not rids:
            continue
        # Combined model's recall on this animal's recordings
        vals_c = [cls_combined[rid].get("recall_real", 0.0)
                   for rid in rids if rid in cls_combined]
        if vals_c:
            combined_avg[a] = float(np.mean(vals_c))
        # Per-animal model's recall (its own LORO aggregate)
        pa_latest = latest_version(artroot / "per_animal" / a)
        if pa_latest is None:
            continue
        pa_prov = load_provenance(pa_latest) or {}
        r = pa_prov.get("loro_summary", {}).get("recall_real")
        if r is not None:
            per_animal_avg[a] = float(r)
    if not combined_avg and not per_animal_avg:
        print("  skip: no data")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    animals = sorted(set(combined_avg) | set(per_animal_avg))
    x = np.arange(len(animals))
    width = 0.38
    c_vals = [combined_avg.get(a, np.nan) for a in animals]
    pa_vals = [per_animal_avg.get(a, np.nan) for a in animals]
    ax.bar(x - width / 2, c_vals, width, label="Combined model",
           color="#888", edgecolor="black")
    ax.bar(x + width / 2, pa_vals, width, label="Per-animal model",
           color=[ANIMAL_COLORS[a] for a in animals],
           edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"animal {a}" for a in animals])
    ax.set_ylabel("LORO recall_real")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-animal models vs combined model\n"
                 "(LORO recall_real, per-animal model uses the animal's own data)")
    ax.legend(loc="lower right")
    for xi, (cv, pv) in enumerate(zip(c_vals, pa_vals)):
        if not np.isnan(cv):
            ax.text(xi - width / 2, cv + 0.01, f"{cv:.2f}",
                    ha="center", fontsize=8)
        if not np.isnan(pv):
            ax.text(xi + width / 2, pv + 0.01, f"{pv:.2f}",
                    ha="center", fontsize=8)
    _save(fig, out_dir, "B1_per_animal_vs_combined_recall")


def _load_rid_to_animal_map(artroot: Path) -> dict[str, str]:
    """Build {recording_id -> animal_letter} from the manifest."""
    try:
        from detector.manifest import Manifest
        m = Manifest.load(detector_paths.get_manifest_path())
        out = {}
        for r in m.list_recordings(include_held_out=True):
            rid = r.get("recording_id")
            animal = (r.get("animal") or "").upper()
            if rid and animal:
                out[rid] = animal
        return out
    except Exception as e:
        print(f"  warn: couldn't load manifest: {e}")
        return {}


# ==================================================================
# B3 - Per-animal bad-fraction comparison
# ==================================================================

@_safe
def plot_per_animal_bad_fraction(artroot: Path, out_dir: Path) -> None:
    print("[plot_per_animal_bad_fraction]")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    plotted = False
    for ax, a in zip(axes, ANIMALS):
        pa_latest = latest_version(artroot / "per_animal" / a)
        if pa_latest is None:
            ax.axis("off")
            continue
        prov = load_provenance(pa_latest) or {}
        per_rec = prov.get("loro_summary", {}).get("per_recording", {})
        if not per_rec:
            ax.axis("off")
            continue
        rids = list(per_rec.keys())
        # Human bad fraction (window-level) = n_real_pos / n_test
        human_bf = [per_rec[r].get("n_real_pos", 0)
                    / max(per_rec[r].get("n_test", 1), 1)
                    for r in rids]
        model_bf = [per_rec[r].get("bad_fraction", 0.0) for r in rids]
        x = np.arange(len(rids))
        width = 0.4
        ax.bar(x - width / 2, human_bf, width, label="Human",
               color="#888")
        ax.bar(x + width / 2, model_bf, width, label="Model",
               color=ANIMAL_COLORS[a])
        ax.set_xticks(x)
        ax.set_xticklabels([short_rid(r, 18) for r in rids],
                            rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Bad fraction (window-level)")
        ax.set_title(f"Animal {a} — bad fraction per recording")
        ax.legend(loc="upper right", fontsize=8)
        plotted = True
    if not plotted:
        print("  skip: no per-animal models")
        return
    fig.suptitle("Per-animal model: human vs model bad fraction "
                 "(per LORO-held-out recording)", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, "B3_per_animal_bad_fraction")


# ==================================================================
# B4 - Per-animal gate pass rate
# ==================================================================

@_safe
def plot_per_animal_gate_pass_rate(artroot: Path,
                                       out_dir: Path) -> None:
    print("[plot_per_animal_gate_pass_rate]")
    rows = []
    for a in ANIMALS:
        pa_latest = latest_version(artroot / "per_animal" / a)
        if pa_latest is None:
            continue
        prov = load_provenance(pa_latest) or {}
        ls = prov.get("loro_summary", {})
        n = ls.get("n_folds", 0)
        if not n:
            continue
        rows.append({
            "animal": a,
            "G1_pass": ls.get("gate_1_pass_count", 0),
            "G2_pass": ls.get("gate_2_pass_count", 0),
            "G3_pass": ls.get("gate_3_pass_count", 0),
            "n_folds": n,
        })
    if not rows:
        print("  skip: no per-animal models")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    animals = [r["animal"] for r in rows]
    n_folds = [r["n_folds"] for r in rows]
    x = np.arange(len(rows))
    width = 0.25
    for i, gate in enumerate(["G1", "G2", "G3"]):
        passes = [r[f"{gate}_pass"] for r in rows]
        rates = [p / n for p, n in zip(passes, n_folds)]
        bars = ax.bar(x + (i - 1) * width, rates, width,
                      label=f"{gate} (band ratios / phase / heart)"
                      if i == 0 else gate,
                      color=["#4c72b0", "#dd8452", "#55a868"][i])
        for b, p, n in zip(bars, passes, n_folds):
            ax.text(b.get_x() + b.get_width() / 2, p / n + 0.02,
                    f"{p}/{n}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"animal {a}" for a in animals])
    ax.set_ylabel("Gate pass rate")
    ax.set_ylim(0, 1.15)
    ax.set_title("Per-animal model: LORO gate pass rates\n"
                 "G1 = frequency-band ratios, G2 = phase histogram, G3 = heart-beat retention")
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, out_dir, "B4_per_animal_gate_pass_rate")


# ==================================================================
# B5 - Per-animal recall vs bad-fraction scatter (calibration drift)
# ==================================================================

@_safe
def plot_per_animal_recall_vs_bad_fraction(artroot: Path,
                                                out_dir: Path) -> None:
    print("[plot_per_animal_recall_vs_bad_fraction]")
    fig, ax = plt.subplots(figsize=(8, 6))
    plotted_any = False
    for a in ANIMALS:
        pa_latest = latest_version(artroot / "per_animal" / a)
        if pa_latest is None:
            continue
        prov = load_provenance(pa_latest) or {}
        per_rec = prov.get("loro_summary", {}).get("per_recording", {})
        if not per_rec:
            continue
        rids = list(per_rec.keys())
        recalls = [per_rec[r].get("recall_real", 0.0) for r in rids]
        bfs = [per_rec[r].get("bad_fraction", 0.0) for r in rids]
        ax.scatter(bfs, recalls, color=ANIMAL_COLORS[a],
                   label=f"animal {a}", s=70, edgecolor="black",
                   linewidth=0.8, alpha=0.85)
        # Annotate each point with the rid (compact)
        for rid, x, y in zip(rids, bfs, recalls):
            ax.annotate(short_rid(rid, 18), (x, y),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=6, color="#444")
        plotted_any = True
    if not plotted_any:
        print("  skip: no per-animal data")
        return
    ax.set_xlabel("Model bad fraction (per recording, window-level)")
    ax.set_ylabel("Recall_real (per recording)")
    ax.set_title("Recall vs bad-fraction across LORO held-out recordings\n"
                 "Diffuse cloud = calibration drift; tight band = consistent operating point")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    _save(fig, out_dir, "B5_recall_vs_bad_fraction")


# ==================================================================
# C1 - Hyperopt optimization history (4-up grid)
# ==================================================================

@_safe
def plot_hyperopt_optimization_history(artroot: Path,
                                          out_dir: Path) -> None:
    print("[plot_hyperopt_optimization_history]")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()
    plotted = False
    for ax, a in zip(axes, ANIMALS):
        log_path = get_hyperopt_dir(artroot, a) / "trial_log.json"
        if not log_path.exists():
            ax.axis("off")
            ax.set_title(f"animal {a}: no hyperopt")
            continue
        try:
            trials = json.loads(log_path.read_text())
        except Exception as e:
            ax.set_title(f"animal {a}: read error")
            continue
        values = [(t.get("number"), t.get("value"))
                   for t in trials if t.get("value") is not None]
        values.sort(key=lambda x: x[0] if x[0] is not None else 0)
        if not values:
            ax.axis("off")
            continue
        ns = np.array([n for n, _ in values])
        vs = np.array([v for _, v in values])
        best = np.maximum.accumulate(vs)
        ax.plot(ns, vs, "o", ms=4, color=ANIMAL_COLORS[a],
                alpha=0.55, label="trial value")
        ax.plot(ns, best, "-", color=ANIMAL_COLORS[a], lw=2,
                label="best so far")
        ax.set_title(f"animal {a} (n_trials={len(ns)})")
        ax.set_xlabel("Trial number")
        ax.set_ylabel("F-β objective")
        ax.legend(loc="lower right", fontsize=7)
        plotted = True
    if not plotted:
        print("  skip: no trial logs found")
        return
    fig.suptitle("Hyperopt optimization history "
                 "(per-animal Optuna TPE)", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, "C1_optimization_history")


# ==================================================================
# C2 - Hyperopt parameter importance grouped bar
# ==================================================================

@_safe
def plot_hyperopt_param_importance(artroot: Path,
                                       out_dir: Path) -> None:
    print("[plot_hyperopt_param_importance]")
    importances = {}
    for a in ANIMALS:
        txt_path = (get_hyperopt_dir(artroot, a) / "plots"
                    / "param_importances.txt")
        if not txt_path.exists():
            continue
        try:
            # Format: lines "param: value" or JSON-ish; be flexible.
            content = txt_path.read_text()
            d = {}
            for line in content.splitlines():
                line = line.strip()
                if ":" not in line:
                    continue
                k, _, v = line.partition(":")
                k = k.strip().strip(",").strip('"').strip("'")
                v = v.strip().strip(",").strip()
                try:
                    d[k] = float(v)
                except ValueError:
                    continue
            if d:
                importances[a] = d
        except Exception as e:
            print(f"  warn: {a}: {e}")
    if not importances:
        # Fallback: try best_params.json (which doesn't have
        # importances) -> skip with note.
        print("  skip: no param_importances.txt files (run hyperopt first)")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    params = ["w_neg", "fp_weight", "fn_weight"]
    animals = sorted(importances.keys())
    x = np.arange(len(params))
    width = 0.2
    for i, a in enumerate(animals):
        vals = [importances[a].get(p, 0.0) for p in params]
        ax.bar(x + (i - (len(animals) - 1) / 2) * width, vals, width,
               label=f"animal {a}", color=ANIMAL_COLORS[a])
        for xi, v in zip(x, vals):
            ax.text(xi + (i - (len(animals) - 1) / 2) * width, v + 0.01,
                    f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.set_ylabel("fANOVA importance")
    ax.set_title("Hyperopt parameter importance per animal "
                 "(fANOVA; values sum to ~1 per animal)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    _save(fig, out_dir, "C2_param_importance")


# ==================================================================
# C3 - Best-params radar plot
# ==================================================================

def _normalize_param(p: str, v: float) -> float:
    """Normalize a hyperopt param to [0, 1] over its search bounds.
    Uses log-scale for w_neg since it's log-sampled."""
    lo, hi = HYPEROPT_BOUNDS[p]
    if p == "w_neg":
        v = max(v, lo)  # guard
        return (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return (v - lo) / (hi - lo)


@_safe
def plot_hyperopt_best_params_radar(artroot: Path,
                                       out_dir: Path) -> None:
    print("[plot_hyperopt_best_params_radar]")
    best = {}
    for a in ANIMALS:
        bp_path = get_hyperopt_dir(artroot, a) / "best_params.json"
        if not bp_path.exists():
            continue
        try:
            data = json.loads(bp_path.read_text())
            best[a] = data.get("best_params", {})
        except Exception:
            continue
    if not best:
        print("  skip: no best_params.json files")
        return
    params = ["w_neg", "fp_weight", "fn_weight"]
    angles = np.linspace(0, 2 * np.pi, len(params),
                          endpoint=False).tolist()
    angles += angles[:1]  # close polygon
    fig, ax = plt.subplots(figsize=(7, 7),
                            subplot_kw={"projection": "polar"})
    for a in sorted(best.keys()):
        vals = [_normalize_param(p, best[a].get(p, 0)) for p in params]
        vals += vals[:1]
        ax.plot(angles, vals, "-o", color=ANIMAL_COLORS[a],
                label=f"animal {a}", linewidth=2, ms=6)
        ax.fill(angles, vals, color=ANIMAL_COLORS[a], alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"{p}\n(log)" if p == "w_neg" else p
                        for p in params])
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["25%", "50%", "75%"], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Hyperopt best parameters per animal\n"
                 "(normalized to [0,1] over search bounds; w_neg log-scaled)",
                 pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0))
    # Annotate raw values inside the plot for each animal
    txt = []
    for a in sorted(best.keys()):
        bp = best[a]
        txt.append(f"{a}: w_neg={bp.get('w_neg', 0):.3f}, "
                   f"fp={bp.get('fp_weight', 0):.2f}, "
                   f"fn={bp.get('fn_weight', 0):.2f}")
    fig.text(0.5, -0.02, "\n".join(txt), ha="center", fontsize=8,
             family="monospace")
    _save(fig, out_dir, "C3_best_params_radar")


# ==================================================================
# C3-alt - 3D scatter of trials (one per animal)
# ==================================================================

@_safe
def plot_hyperopt_3d_trial_scatter(artroot: Path,
                                      out_dir: Path) -> None:
    print("[plot_hyperopt_3d_trial_scatter]")
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    for a in ANIMALS:
        log_path = get_hyperopt_dir(artroot, a) / "trial_log.json"
        if not log_path.exists():
            continue
        try:
            trials = json.loads(log_path.read_text())
        except Exception as e:
            print(f"  warn: {a}: {e}")
            continue
        rows = []
        for t in trials:
            p = t.get("params") or {}
            v = t.get("value")
            if v is None:
                continue
            try:
                rows.append((float(p["w_neg"]), float(p["fp_weight"]),
                             float(p["fn_weight"]), float(v)))
            except (KeyError, ValueError, TypeError):
                continue
        if not rows:
            continue
        rows = np.array(rows)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(rows[:, 0], rows[:, 1], rows[:, 2],
                        c=rows[:, 3], cmap="viridis", s=40,
                        edgecolor="black", linewidth=0.3)
        ax.set_xlabel("w_neg")
        ax.set_ylabel("fp_weight")
        ax.set_zlabel("fn_weight")
        # log scale on w_neg for readability
        try:
            ax.set_xscale("log")
        except Exception:
            pass
        ax.set_title(f"animal {a} — hyperopt trials\n"
                     f"(color = F-β objective, n={len(rows)} trials)")
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.1)
        cbar.set_label("F-β objective")
        # Mark the best trial with a red star
        best_i = int(np.argmax(rows[:, 3]))
        bp = rows[best_i]
        ax.scatter([bp[0]], [bp[1]], [bp[2]], color="red",
                    marker="*", s=240, edgecolor="black",
                    linewidth=1.5, label=f"best={bp[3]:.3f}")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        _save(fig, out_dir, f"C3alt_3d_trials_{a}")


# ==================================================================
# C4 - Parallel coordinate plot (copy existing PNG)
# ==================================================================

@_safe
def copy_parallel_coordinate_plots(artroot: Path, out_dir: Path) -> None:
    print("[copy_parallel_coordinate_plots]")
    for a in ANIMALS:
        src = get_hyperopt_dir(artroot, a) / "plots" / "parallel_coordinate.png"
        if not src.exists():
            continue
        dst = out_dir / f"C4_parallel_coordinate_{a}.png"
        shutil.copy(src, dst)
        print(f"  copied {a}: {dst.name}")


# ==================================================================
# F1 - Combined corpus composition
# ==================================================================

@_safe
def plot_combined_corpus_composition(artroot: Path,
                                         out_dir: Path) -> None:
    print("[plot_combined_corpus_composition]")
    latest = latest_version(artroot)
    if latest is None:
        print("  skip: no combined model")
        return
    prov = load_provenance(latest) or {}
    real_pos = prov.get("n_real_positives", 0)
    syn_pos = prov.get("n_synthetic_positives", 0)
    clean = prov.get("n_unlabeled_clean", 0)
    total = real_pos + syn_pos + clean
    if total == 0:
        print("  skip: zero counts")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    cats = ["Real positives", "Synthetic positives",
            "Negatives (unlabeled clean)"]
    vals = [real_pos, syn_pos, clean]
    colors = ["#c44e52", "#dd8452", "#888"]
    bars = ax.bar(cats, vals, color=colors, edgecolor="black")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v,
                f"{v:,}\n({v/total*100:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Row count (Phase 2 windows)")
    ax.set_title(f"Combined-model training corpus — {latest.name}\n"
                 f"total: {total:,} windows  ·  "
                 f"imbalance neg:pos = {clean / max(real_pos + syn_pos, 1):.0f}:1")
    ax.set_ylim(0, max(vals) * 1.18)
    _save(fig, out_dir, "F1_combined_corpus_composition")


# ==================================================================
# F2 - Phase 1 vs Phase 2 size
# ==================================================================

@_safe
def plot_phase1_vs_phase2_size(artroot: Path, out_dir: Path) -> None:
    print("[plot_phase1_vs_phase2_size]")
    p1 = artroot / "dataset_phase1.parquet"
    p2 = artroot / "dataset_phase2.parquet"
    counts = {}
    try:
        import pyarrow.parquet as pq
        if p1.exists():
            counts["Phase 1\n(raw features)"] = pq.read_metadata(p1).num_rows
        if p2.exists():
            counts["Phase 2\n(with synth augmentation)"] = pq.read_metadata(p2).num_rows
    except ImportError:
        print("  skip: pyarrow not available")
        return
    if not counts:
        print("  skip: no parquet files at artifacts root")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    cats = list(counts.keys())
    vals = list(counts.values())
    bars = ax.bar(cats, vals, color=["#888", "#4c72b0"],
                  edgecolor="black")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}",
                ha="center", va="bottom", fontsize=10)
    if len(vals) == 2:
        delta = vals[1] - vals[0]
        ax.set_title(f"Combined dataset: Phase 1 → Phase 2\n"
                     f"+{delta:,} synthetic positives "
                     f"({delta / vals[0] * 100:.1f}% augmentation)")
    else:
        ax.set_title("Combined dataset sizes")
    ax.set_ylabel("Row count")
    _save(fig, out_dir, "F2_phase1_vs_phase2")


# ==================================================================
# G1 - Combined: per-recording recall (colored by animal)
# ==================================================================

@_safe
def plot_combined_per_recording_recall(artroot: Path,
                                          out_dir: Path) -> None:
    print("[plot_combined_per_recording_recall]")
    latest = latest_version(artroot)
    if latest is None:
        print("  skip: no combined model")
        return
    prov = load_provenance(latest) or {}
    per_rec = prov.get("loro_summary", {}).get("per_recording", {})
    if not per_rec:
        print("  skip: no per_recording data")
        return
    rid_to_animal = _load_rid_to_animal_map(artroot)
    # Sort by animal then recall ascending
    rows = []
    for rid, m in per_rec.items():
        rows.append((
            rid,
            rid_to_animal.get(rid, "?"),
            m.get("recall_real", 0.0),
            m.get("n_real_pos", 0),
            m.get("bad_fraction", 0.0),
        ))
    rows.sort(key=lambda r: (r[1], r[2]))
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(rows)), 5))
    colors = [ANIMAL_COLORS.get(r[1], "#999") for r in rows]
    x = np.arange(len(rows))
    bars = ax.bar(x, [r[2] for r in rows], color=colors,
                  edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([short_rid(r[0], 20) for r in rows],
                        rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Recall_real")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Combined model — per-recording recall  "
                 f"({latest.name}, threshold={prov.get('threshold')})")
    # Animal legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=ANIMAL_COLORS[a])
               for a in ANIMALS if a in {r[1] for r in rows}]
    labels = [f"animal {a}" for a in ANIMALS
              if a in {r[1] for r in rows}]
    ax.legend(handles, labels, loc="upper right")
    for b, r in zip(bars, rows):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{r[2]:.2f}", ha="center", fontsize=7)
    _save(fig, out_dir, "G1_combined_per_recording_recall")


# ==================================================================
# G2 - Combined: bad fraction human vs model per recording
# ==================================================================

@_safe
def plot_combined_bad_fraction(artroot: Path, out_dir: Path) -> None:
    print("[plot_combined_bad_fraction]")
    latest = latest_version(artroot)
    if latest is None:
        print("  skip: no combined model")
        return
    prov = load_provenance(latest) or {}
    per_rec = prov.get("loro_summary", {}).get("per_recording", {})
    if not per_rec:
        print("  skip: no per_recording data")
        return
    rid_to_animal = _load_rid_to_animal_map(artroot)
    rows = []
    for rid, m in per_rec.items():
        h = m.get("n_real_pos", 0) / max(m.get("n_test", 1), 1)
        rows.append((rid, rid_to_animal.get(rid, "?"), h,
                     m.get("bad_fraction", 0.0)))
    fig, ax = plt.subplots(figsize=(8, 6))
    for a in ANIMALS:
        a_rows = [r for r in rows if r[1] == a]
        if not a_rows:
            continue
        ax.scatter([r[2] for r in a_rows], [r[3] for r in a_rows],
                    color=ANIMAL_COLORS[a], label=f"animal {a}",
                    s=80, edgecolor="black", linewidth=0.7,
                    alpha=0.85)
        for rid, _, h, m_bf in a_rows:
            ax.annotate(short_rid(rid, 18), (h, m_bf),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=6, color="#444")
    # Diagonal
    mx = max(max(r[2] for r in rows), max(r[3] for r in rows)) * 1.05
    ax.plot([0, mx], [0, mx], "--", color="#888", alpha=0.6,
            label="y = x (perfect calibration)")
    ax.set_xlabel("Human bad fraction (per recording)")
    ax.set_ylabel("Model bad fraction (per recording)")
    ax.set_title(f"Combined model — human vs model bad fraction  "
                 f"({latest.name})\n"
                 "Points above diagonal = over-blanking")
    ax.legend(loc="upper left")
    ax.set_xlim(0, mx)
    ax.set_ylim(0, mx)
    _save(fig, out_dir, "G2_combined_bad_fraction")


# ==================================================================
# G3 - Combined: gates pass/fail per fold heatmap
# ==================================================================

@_safe
def plot_combined_gates_heatmap(artroot: Path,
                                    out_dir: Path) -> None:
    print("[plot_combined_gates_heatmap]")
    latest = latest_version(artroot)
    if latest is None:
        print("  skip: no combined model")
        return
    prov = load_provenance(latest) or {}
    per_rec = prov.get("loro_summary", {}).get("per_recording", {})
    if not per_rec:
        print("  skip: no per_recording")
        return
    rid_to_animal = _load_rid_to_animal_map(artroot)
    rows = sorted(per_rec.items(),
                   key=lambda kv: (rid_to_animal.get(kv[0], "?"),
                                    kv[0]))
    rids = [r[0] for r in rows]
    n = len(rids)
    M = np.zeros((n, 3))  # 1 = pass, 0 = fail
    for i, (rid, m) in enumerate(rows):
        M[i, 0] = 1.0 if m.get("gate_1_pass") else 0.0
        M[i, 1] = 1.0 if m.get("gate_2_pass") else 0.0
        M[i, 2] = 1.0 if m.get("gate_3_pass") else 0.0
    fig, ax = plt.subplots(figsize=(5, max(4, 0.32 * n)))
    cmap = plt.cm.colors.ListedColormap([GATE_COLORS["fail"],
                                            GATE_COLORS["pass"]])
    ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    # Annotate cells
    for i in range(n):
        for j in range(3):
            txt = "✓" if M[i, j] > 0.5 else "✗"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=12, color="white", fontweight="bold")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["G1\n(band ratios)", "G2\n(phase)",
                         "G3\n(heart)"])
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([
        f"[{rid_to_animal.get(r, '?')}] {short_rid(r, 26)}"
        for r in rids
    ], fontsize=7)
    ax.set_title(f"Combined gates — per-recording pass/fail  "
                 f"({latest.name})")
    _save(fig, out_dir, "G3_combined_gates_heatmap")


# ==================================================================
# G4 / G5 - Calibration & PR (require inference)
# ==================================================================

def _collect_probs_for_held_out(model_dir: Path, animal_filter=None
                                  ) -> list[tuple[float, int, str]]:
    """Run the model on every held_out=True recording in the
    manifest and collect (prob, label, rid) tuples at the window
    level. Returns [] on failure.

    NOTE: This is the slow path. ~30 s - 2 min per held-out recording
    depending on size.

    Uses `build_recording_frame` from dataset.py to produce a labeled
    feature frame, then applies the model artifact's predict_proba.
    Resolves baseline_dir relative to the model artifact (combined
    model uses <artroot>/baselines; per-animal uses
    <artroot>/per_animal/<a>/baselines).
    """
    try:
        from detector.model_artifact import ModelArtifact
        from detector.manifest import Manifest
        from detector import paths as DP
        from detector.dataset import build_recording_frame
    except Exception as e:
        print(f"  warn: imports failed: {e}")
        return []
    try:
        m = Manifest.load(DP.get_manifest_path())
    except Exception as e:
        print(f"  warn: manifest load failed: {e}")
        return []
    artifact = ModelArtifact.load(model_dir)
    # Where do baselines live? Combined: artroot/baselines.
    # Per-animal: model_dir is artroot/per_animal/<a>/model_v*/,
    # baselines live at artroot/per_animal/<a>/baselines.
    if "per_animal" in model_dir.parts:
        baseline_dir = model_dir.parent / "baselines"
    else:
        baseline_dir = model_dir.parent / "baselines"
    if not baseline_dir.exists():
        # Fallback: use the global one
        baseline_dir = DP.get_artifacts_dir() / "baselines"
    print(f"    using baseline_dir: {baseline_dir}")
    out = []
    for r in m.recordings:
        if not r.get("held_out"):
            continue
        if animal_filter and (r.get("animal") or "").upper() != animal_filter:
            continue
        rid = r.get("recording_id")
        print(f"    running inference on {rid} ...", flush=True)
        try:
            df = build_recording_frame(r, baseline_dir=baseline_dir)
        except Exception as e:
            print(f"      skip ({type(e).__name__}: {e})")
            continue
        if df is None or len(df) == 0:
            continue
        if "label" not in df.columns:
            continue
        labels = df["label"].values
        try:
            feats = df[list(artifact.feature_columns)]
        except KeyError as e:
            print(f"      skip: feature schema mismatch ({e})")
            continue
        probs = artifact.predict_proba(feats)
        for p_v, lbl in zip(probs, labels):
            out.append((float(p_v), int(lbl), rid))
    print(f"    collected {len(out):,} window predictions")
    return out


@_safe
def plot_calibration(artroot: Path, out_dir: Path,
                      target: str = "combined") -> None:
    print(f"[plot_calibration:{target}]")
    if target == "combined":
        model_dir = latest_version(artroot)
        animal_filter = None
        tag = "combined"
    else:
        model_dir = latest_version(artroot / "per_animal" / target)
        animal_filter = target
        tag = f"per_animal_{target}"
    if model_dir is None:
        print(f"  skip: no model for {target}")
        return
    pairs = _collect_probs_for_held_out(model_dir, animal_filter)
    if not pairs:
        print("  skip: no held-out recordings produced probabilities")
        return
    probs = np.array([p for p, _, _ in pairs])
    labels = np.array([l for _, l, _ in pairs])
    # Bin by predicted probability decile
    bins = np.linspace(0, 1, 11)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    empirical = []
    n_in_bin = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if hi == 1.0:
            mask = (probs >= lo) & (probs <= hi)
        n = mask.sum()
        n_in_bin.append(n)
        empirical.append(labels[mask].mean() if n else np.nan)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={"width_ratios": [3, 1]})
    ax1.plot([0, 1], [0, 1], "--", color="#888",
              label="perfectly calibrated")
    ax1.plot(bin_centers, empirical, "o-", color="#4c72b0", lw=2,
              ms=8, label="model")
    ax1.set_xlabel("Predicted probability (bin center)")
    ax1.set_ylabel("Empirical positive rate")
    ax1.set_title(f"Calibration plot — {tag}\n"
                  f"(N={len(pairs):,} window predictions across held-out recordings)")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)
    ax2.bar(bin_centers, n_in_bin, width=0.08, color="#888")
    ax2.set_yscale("log")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count per bin (log)")
    ax2.set_title("Bin support")
    _save(fig, out_dir, f"G4_calibration_{tag}")


@_safe
def plot_pr_curve(artroot: Path, out_dir: Path,
                   target: str = "combined") -> None:
    print(f"[plot_pr_curve:{target}]")
    if target == "combined":
        model_dir = latest_version(artroot)
        animal_filter = None
        tag = "combined"
    else:
        model_dir = latest_version(artroot / "per_animal" / target)
        animal_filter = target
        tag = f"per_animal_{target}"
    if model_dir is None:
        print(f"  skip: no model for {target}")
        return
    threshold = load_threshold(model_dir)
    pairs = _collect_probs_for_held_out(model_dir, animal_filter)
    if not pairs:
        print("  skip: no held-out recordings produced probabilities")
        return
    probs = np.array([p for p, _, _ in pairs])
    labels = np.array([l for _, l, _ in pairs])
    # Compute PR over a thresholds sweep
    thr_sweep = np.unique(np.concatenate([
        np.linspace(0.0001, 0.05, 40),
        np.linspace(0.05, 0.5, 30),
        np.linspace(0.5, 1.0, 10),
    ]))
    precision = []
    recall = []
    for t in thr_sweep:
        pred = probs >= t
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision.append(p)
        recall.append(r)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, "-", color="#4c72b0", lw=2)
    if threshold is not None:
        # Highlight the operating point
        op_pred = probs >= threshold
        tp = int(((op_pred == 1) & (labels == 1)).sum())
        fp = int(((op_pred == 1) & (labels == 0)).sum())
        fn = int(((op_pred == 0) & (labels == 1)).sum())
        op_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        op_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ax.plot([op_r], [op_p], "*", color="red", ms=20,
                  markeredgecolor="black",
                  label=f"chosen threshold = {threshold:.4f}\n"
                        f"(P={op_p:.3f}, R={op_r:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall — {tag}\n"
                 f"(N={len(pairs):,} window predictions)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend()
    _save(fig, out_dir, f"G5_pr_curve_{tag}")


# ==================================================================
# H1 - Combined recall progression across versions
# ==================================================================

@_safe
def plot_combined_recall_progression(artroot: Path,
                                         out_dir: Path) -> None:
    print("[plot_combined_recall_progression]")
    versions = get_combined_versions(artroot)
    if len(versions) < 2:
        print("  skip: fewer than 2 combined versions")
        return
    xs, ys, syns, badf = [], [], [], []
    for v in versions:
        prov = load_provenance(v) or {}
        ls = prov.get("loro_summary", {})
        r = ls.get("recall_real")
        if r is None:
            continue
        xs.append(v.name.replace("model_", ""))
        ys.append(float(r))
        syns.append(float(ls.get("recall_syn", 0)))
        badf.append(float(ls.get("mean_bad_fraction", 0)))
    if len(xs) < 2:
        print("  skip: not enough provenance data")
        return
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(xs, ys, "o-", color="#c44e52", lw=2, ms=8,
              label="recall_real")
    ax1.plot(xs, syns, "s--", color="#4c72b0", lw=1.5, ms=6,
              alpha=0.7, label="recall_syn")
    ax1.set_ylabel("Recall")
    ax1.set_xlabel("Combined model version")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower left")
    ax2 = ax1.twinx()
    ax2.plot(xs, badf, "^:", color="#888", lw=1.5, ms=6,
              label="mean_bad_fraction")
    ax2.set_ylabel("Mean model bad fraction", color="#666")
    ax2.legend(loc="lower right")
    plt.title("Combined model: recall + bad-fraction across versions")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    _save(fig, out_dir, "H1_combined_recall_progression")


# ==================================================================
# H2 - Combined gate progression
# ==================================================================

@_safe
def plot_combined_gate_progression(artroot: Path,
                                       out_dir: Path) -> None:
    print("[plot_combined_gate_progression]")
    versions = get_combined_versions(artroot)
    if len(versions) < 2:
        print("  skip: fewer than 2 combined versions")
        return
    xs, g1, g2, g3, n_folds = [], [], [], [], []
    for v in versions:
        prov = load_provenance(v) or {}
        ls = prov.get("loro_summary", {})
        if not ls.get("n_folds"):
            continue
        xs.append(v.name.replace("model_", ""))
        g1.append(ls.get("gate_1_pass_count", 0))
        g2.append(ls.get("gate_2_pass_count", 0))
        g3.append(ls.get("gate_3_pass_count", 0))
        n_folds.append(ls.get("n_folds", 0))
    if len(xs) < 2:
        print("  skip: not enough data")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, [g / n for g, n in zip(g1, n_folds)], "o-",
              color="#4c72b0", lw=2, ms=7, label="G1 (band ratios)")
    ax.plot(xs, [g / n for g, n in zip(g2, n_folds)], "s-",
              color="#dd8452", lw=2, ms=7, label="G2 (phase)")
    ax.plot(xs, [g / n for g, n in zip(g3, n_folds)], "^-",
              color="#55a868", lw=2, ms=7, label="G3 (heart)")
    for i, (a, b, c, n) in enumerate(zip(g1, g2, g3, n_folds)):
        ax.text(i, a / n + 0.025, f"{a}/{n}", ha="center", fontsize=7,
                  color="#4c72b0")
        ax.text(i, b / n + 0.025, f"{b}/{n}", ha="center", fontsize=7,
                  color="#dd8452")
        ax.text(i, c / n + 0.025, f"{c}/{n}", ha="center", fontsize=7,
                  color="#55a868")
    ax.set_ylabel("Gate pass rate")
    ax.set_xlabel("Combined model version")
    ax.set_ylim(0, 1.15)
    ax.set_title("Combined model: gate pass rates across versions")
    ax.legend(loc="upper left")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    _save(fig, out_dir, "H2_combined_gate_progression")


# ==================================================================
# H3 - Combined threshold history
# ==================================================================

@_safe
def plot_combined_threshold_history(artroot: Path,
                                        out_dir: Path) -> None:
    print("[plot_combined_threshold_history]")
    versions = get_combined_versions(artroot)
    if not versions:
        print("  skip: no combined models")
        return
    xs, ts = [], []
    for v in versions:
        t = load_threshold(v)
        if t is None:
            continue
        xs.append(v.name.replace("model_", ""))
        ts.append(t)
    if not xs:
        print("  skip: no thresholds found")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ts, "o-", color="#4c72b0", lw=2, ms=8)
    for i, t in enumerate(ts):
        ax.text(i, t + 0.0005, f"{t:.4f}", ha="center", fontsize=8)
    ax.set_ylabel("Auto-selected threshold")
    ax.set_xlabel("Combined model version")
    ax.set_title("Combined model: threshold-selection history\n"
                 "(picked by retrain to meet gate-target weighted distance)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    _save(fig, out_dir, "H3_combined_threshold_history")


# ==================================================================
# I1 - Active-learning effect
# ==================================================================

@_safe
def plot_active_learning_progression(artroot: Path,
                                          out_dir: Path) -> None:
    print("[plot_active_learning_progression]")
    versions = get_combined_versions(artroot)
    rows = []
    for v in versions:
        prov = load_provenance(v) or {}
        rs = prov.get("review_summary") or {}
        if not rs:
            continue
        rows.append({
            "version": v.name.replace("model_", ""),
            "n_fp_corrections": rs.get("n_false_positive", 0),
            "n_fn_corrections": rs.get("n_true_artifact", 0),
            "recall_real": prov.get("loro_summary", {}).get(
                "recall_real", np.nan),
        })
    if len(rows) < 1:
        print("  skip: no review_summary in any version")
        return
    fig, ax1 = plt.subplots(figsize=(8, 5))
    xs = [r["version"] for r in rows]
    fp = [r["n_fp_corrections"] for r in rows]
    fn = [r["n_fn_corrections"] for r in rows]
    rc = [r["recall_real"] for r in rows]
    x = np.arange(len(xs))
    w = 0.35
    ax1.bar(x - w/2, fp, w, color="#dd8452",
            label="Review FP corrections")
    ax1.bar(x + w/2, fn, w, color="#4c72b0",
            label="Review FN corrections")
    ax1.set_xticks(x)
    ax1.set_xticklabels(xs, rotation=45, ha="right")
    ax1.set_ylabel("# corrections applied")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(x, rc, "*-", color="#c44e52", ms=14, lw=2,
              label="recall_real")
    ax2.set_ylabel("recall_real", color="#c44e52")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="upper right")
    plt.title("Active-learning loop: review corrections vs recall")
    fig.tight_layout()
    _save(fig, out_dir, "I1_active_learning_progression")


# ==================================================================
# I2 - Auto-FN scan effectiveness
# ==================================================================

@_safe
def plot_auto_fn_effectiveness(artroot: Path,
                                  out_dir: Path) -> None:
    print("[plot_auto_fn_effectiveness]")
    versions = get_combined_versions(artroot)
    found_any = False
    for v in versions:
        afn_path = v / "auto_fn_corrections.json"
        if not afn_path.exists():
            continue
        try:
            data = json.loads(afn_path.read_text())
        except Exception:
            continue
        rows = data.get("corrections") or data.get("flagged") or []
        if not rows:
            continue
        probs = [r.get("prev_model_proba") or r.get("proba") or 0.0
                 for r in rows]
        if not probs:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(probs, bins=40, color="#4c72b0", edgecolor="black")
        ax.set_xlabel("Previous model's probability on label=1 rows")
        ax.set_ylabel("Count")
        ax.set_title(f"{v.name}: auto-FN scan — "
                     f"{len(probs):,} label=1 rows the previous model "
                     "marked confidently CLEAN\n"
                     "(Each was bumped to high sample_weight for the "
                     "next training pass)")
        # Mark the threshold the auto-FN scan used (if known)
        thr = data.get("threshold")
        if thr is not None:
            ax.axvline(thr, color="red", linestyle="--",
                        label=f"prev threshold={thr:.4f}")
            ax.legend()
        _save(fig, out_dir, f"I2_auto_fn_{v.name}")
        found_any = True
    if not found_any:
        print("  skip: no auto_fn_corrections.json found in any version")


# ==================================================================
# Main
# ==================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="plots_out",
                     help="output dir (default: ./plots_out)")
    ap.add_argument("--with-inference", action="store_true",
                     help="also run G4 (calibration) and G5 (PR curve), "
                          "which need to run inference on held-out "
                          "recordings (slow, ~30s-2min per recording)")
    ap.add_argument("--only", default=None,
                     help="comma-separated subset of plot codes to run "
                          "(e.g. B1,C3,H1)")
    args = ap.parse_args()

    artroot = detector_paths.get_artifacts_dir()
    print(f"artifacts root: {artroot}")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir:     {out_dir}")
    print()

    # Each item: (code, callable). The callable takes no arguments
    # except artroot + out_dir (closed over below).
    combined_latest = latest_version(artroot)
    plot_jobs = []

    # A1 / F4 - feature importance for the combined model + each per-animal
    if combined_latest is not None:
        plot_jobs.append(("A1_F4_combined",
                          lambda: plot_feature_importance(
                              combined_latest, out_dir, "combined")))
    for a in ANIMALS:
        pa_latest = latest_version(artroot / "per_animal" / a)
        if pa_latest is not None:
            plot_jobs.append((f"A1_F4_{a}",
                              lambda d=pa_latest, t=a:
                              plot_feature_importance(d, out_dir,
                                                      f"per_animal_{t}")))

    # A3 / F3 - synth type distribution
    p2_combined = artroot / "dataset_phase2.parquet"
    if p2_combined.exists():
        plot_jobs.append(("A3_F3_combined",
                          lambda: plot_synth_type_distribution(
                              p2_combined, out_dir, "combined")))
    for a in ANIMALS:
        p2 = artroot / "hyperopt_per_animal" / a / "dataset_phase2.parquet"
        # Per-animal hyperopt + per-animal training use the same
        # workdir under hyperopt_per_animal/<a>/, so the phase2 lives
        # there. Fall back to per_animal/<a>/ if the hyperopt copy
        # isn't present.
        alt = artroot / "per_animal" / a / "dataset_phase2.parquet"
        path = p2 if p2.exists() else alt
        if path.exists():
            plot_jobs.append((f"A3_F3_{a}",
                              lambda p=path, t=a:
                              plot_synth_type_distribution(
                                  p, out_dir, f"per_animal_{t}")))

    # Per-animal performance
    plot_jobs += [
        ("B1", lambda: plot_per_animal_vs_combined_recall(artroot, out_dir)),
        ("B3", lambda: plot_per_animal_bad_fraction(artroot, out_dir)),
        ("B4", lambda: plot_per_animal_gate_pass_rate(artroot, out_dir)),
        ("B5", lambda: plot_per_animal_recall_vs_bad_fraction(artroot, out_dir)),
    ]

    # Hyperopt
    plot_jobs += [
        ("C1", lambda: plot_hyperopt_optimization_history(artroot, out_dir)),
        ("C2", lambda: plot_hyperopt_param_importance(artroot, out_dir)),
        ("C3", lambda: plot_hyperopt_best_params_radar(artroot, out_dir)),
        ("C3alt", lambda: plot_hyperopt_3d_trial_scatter(artroot, out_dir)),
        ("C4", lambda: copy_parallel_coordinate_plots(artroot, out_dir)),
    ]

    # Combined
    plot_jobs += [
        ("F1", lambda: plot_combined_corpus_composition(artroot, out_dir)),
        ("F2", lambda: plot_phase1_vs_phase2_size(artroot, out_dir)),
        ("G1", lambda: plot_combined_per_recording_recall(artroot, out_dir)),
        ("G2", lambda: plot_combined_bad_fraction(artroot, out_dir)),
        ("G3", lambda: plot_combined_gates_heatmap(artroot, out_dir)),
    ]

    # Iteration history
    plot_jobs += [
        ("H1", lambda: plot_combined_recall_progression(artroot, out_dir)),
        ("H2", lambda: plot_combined_gate_progression(artroot, out_dir)),
        ("H3", lambda: plot_combined_threshold_history(artroot, out_dir)),
        ("I1", lambda: plot_active_learning_progression(artroot, out_dir)),
        ("I2", lambda: plot_auto_fn_effectiveness(artroot, out_dir)),
    ]

    # Inference-required plots
    if args.with_inference:
        plot_jobs.append(("G4_combined",
                          lambda: plot_calibration(artroot, out_dir,
                                                    "combined")))
        plot_jobs.append(("G5_combined",
                          lambda: plot_pr_curve(artroot, out_dir,
                                                  "combined")))
        for a in ANIMALS:
            plot_jobs.append((f"G4_{a}",
                              lambda t=a: plot_calibration(artroot,
                                                            out_dir, t)))
            plot_jobs.append((f"G5_{a}",
                              lambda t=a: plot_pr_curve(artroot,
                                                          out_dir, t)))

    # Filter by --only
    if args.only:
        wanted = {s.strip().upper() for s in args.only.split(",")}
        plot_jobs = [(code, fn) for code, fn in plot_jobs
                      if any(code.upper().startswith(w) for w in wanted)]
        if not plot_jobs:
            print(f"no jobs matched --only={args.only}")
            return 1

    # Run them
    print(f"running {len(plot_jobs)} plot job(s):")
    for code, fn in plot_jobs:
        print(f"\n----- {code} -----")
        fn()

    print(f"\ndone. {len(list(out_dir.glob('*.png')))} PNGs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
