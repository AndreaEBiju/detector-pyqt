"""M1.3 pan/zoom benchmark for the pyqtgraph signal viewer.

Times `MultiChannelViewer.set_viewport()` at three viewport sizes — 60s,
600s, and 7200s — over 20 random pan positions each. Reports mean,
median, and p95 wall time per pan, plus peak process RSS. Writes a
markdown table to `--out` (default `spike_results.md`) for inclusion
in the M1 spike report.

Usage:
    python scripts/m1_benchmark.py --recording /path/to/2h.mat

Pre-conditions:
- A working PySide6 environment (the GUI is shown — needs a display).
- The recording file must be a `.mat` (MATLAB v7.3) or `.h5` matching
  the LazyRecording schema. See `ui/widgets/signal_viewer.py`.

Output:
- Console: per-size timing summary, peak memory.
- File: `spike_results.md` is updated in-place between the
  `<!-- AUTO:results-start -->` and `<!-- AUTO:results-end -->`
  markers (the rest of the file is left untouched, so any prose you
  add survives a re-run).
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Ensure imports resolve whether this is run as `python scripts/...`
# or `python -m scripts.m1_benchmark`. Insertion order matters:
#   - `detector-core/` is the GEMSBlanking submodule and contains its
#     own Streamlit `ui/` package (no `widgets/` subdir). If it sits
#     BEFORE detector-pyqt in sys.path, Python finds the Streamlit
#     `ui` first and `from ui.widgets …` fails.
#   - So insert detector-core first, then prepend detector-pyqt
#     root on top of it. Final order is
#         [detector-pyqt, detector-core, …rest]
#     which lets `from ui.widgets …` resolve to our PyQt widgets and
#     `from detector …` fall through to the submodule.
_repo_root = Path(__file__).resolve().parent.parent
_detector_core = _repo_root / "detector-core"
if _detector_core.exists():
    sys.path.insert(0, str(_detector_core))
sys.path.insert(0, str(_repo_root))


def _peak_rss_mb() -> float:
    """Process peak resident set size in MB. POSIX-only (macOS and
    Linux); Windows returns 0.0."""
    try:
        import resource
    except ImportError:
        return 0.0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KB. Heuristic: if it looks
    # too big to be a KB count for a normal Python process, assume bytes.
    if platform.system() == "Darwin":
        return rss / (1024 * 1024)
    return rss / 1024  # Linux: KB → MB


def bench_viewport_size(viewer, viewport_sec: float, recording_dur: float,
                         n_iters: int = 20, seed: int = 0) -> dict:
    """Run `n_iters` random pan operations at the given viewport size."""
    from PySide6.QtWidgets import QApplication

    rng = np.random.default_rng(seed)
    times_ms: list[float] = []
    for _ in range(n_iters):
        max_start = max(0.0, recording_dur - viewport_sec)
        t_start = float(rng.uniform(0.0, max_start)) if max_start > 0 else 0.0
        t_end = min(t_start + viewport_sec, recording_dur)
        t0 = time.perf_counter()
        viewer.set_viewport(t_start, t_end)
        # processEvents() forces Qt to flush the paint + downsample
        # work so we time the full round-trip, not just the Python-side
        # set_viewport call.
        QApplication.processEvents()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)
    arr = np.asarray(times_ms)
    return {
        "n": int(n_iters),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
        "min_ms": float(arr.min()),
    }


def _format_results_md(
    recording_path: Path,
    duration_sec: float,
    fs: float,
    n_channels: int,
    results: dict,
    peak_mb: float,
) -> str:
    """Render the auto-replaced section of spike_results.md."""
    bars = {"60s": 100.0, "600s": 300.0, "7200s": 1000.0}
    rows = []
    for size, target in bars.items():
        r = results.get(size)
        if r is None or r.get("skipped"):
            rows.append(
                f"| {size:>5s} | n/a | n/a | n/a | {target:.0f} | "
                f"⚠ skipped ({r.get('reason', 'unknown') if r else 'no data'}) |"
            )
        else:
            verdict = "✅ pass" if r["mean_ms"] < target else "❌ fail"
            rows.append(
                f"| {size:>5s} | {r['mean_ms']:.1f} | {r['median_ms']:.1f} "
                f"| {r['p95_ms']:.1f} | {target:.0f} | {verdict} |"
            )

    md = []
    md.append(f"_Generated_: {datetime.now(timezone.utc).isoformat()}")
    md.append("")
    md.append("### Test conditions")
    md.append("")
    md.append(f"- **Recording**: `{recording_path}`")
    md.append(f"- **Duration**: {duration_sec:.1f} s "
              f"({duration_sec / 60:.1f} min, {duration_sec / 3600:.2f} h)")
    md.append(f"- **fs**: {fs:.2f} Hz")
    md.append(f"- **Channels**: {n_channels}")
    md.append(f"- **Platform**: {platform.platform()}")
    md.append(f"- **Python**: {sys.version.split()[0]}")
    try:
        import pyqtgraph as pg
        import PySide6
        md.append(f"- **PySide6**: {PySide6.__version__}")
        md.append(f"- **pyqtgraph**: {pg.__version__}")
    except Exception:
        pass
    md.append("")
    md.append("### Pan/zoom timings (20 random viewport positions each)")
    md.append("")
    md.append("| Viewport | mean (ms) | median (ms) | p95 (ms) | target (ms) | verdict |")
    md.append("|---------:|----------:|------------:|---------:|------------:|:--------|")
    md.extend(rows)
    md.append("")
    md.append(f"### Peak memory: {peak_mb:.0f} MB (target: < 2048 MB)")
    md.append("")
    return "\n".join(md)


def update_spike_results(path: Path, body: str) -> None:
    """Replace the AUTO section in spike_results.md, leaving prose
    above/below intact."""
    start = "<!-- AUTO:results-start -->"
    end = "<!-- AUTO:results-end -->"
    if path.exists():
        text = path.read_text()
        if start in text and end in text:
            pre, _, rest = text.partition(start)
            _, _, post = rest.partition(end)
            new = f"{pre}{start}\n{body}\n{end}{post}"
            path.write_text(new)
            return
    # No template yet — write a minimal one
    path.write_text(
        f"# M1 signal-viewer spike — results\n\n"
        f"{start}\n{body}\n{end}\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", required=True,
                    help="path to .mat / .h5 recording")
    ap.add_argument("--out", default=str(_repo_root / "spike_results.md"),
                    help="path to spike_results.md (default at repo root)")
    ap.add_argument("--iters", type=int, default=20,
                    help="iterations per viewport size (default 20)")
    args = ap.parse_args()

    # Imports here, not at top: argparse should fail cleanly before we
    # spin up Qt.
    from PySide6.QtWidgets import QApplication
    from ui.widgets.signal_viewer import LazyRecording, MultiChannelViewer

    recording_path = Path(args.recording).expanduser()
    if not recording_path.exists():
        print(f"error: recording not found at {recording_path}", file=sys.stderr)
        return 2

    print(f"loading {recording_path} ...")
    recording = LazyRecording(recording_path)
    print(f"  fs={recording.fs:.1f} Hz, channels={recording.n_channels}, "
          f"samples={recording.n_samples:,}, duration={recording.duration_sec:.1f}s")

    app = QApplication.instance() or QApplication(sys.argv)
    viewer = MultiChannelViewer(recording)
    viewer.resize(1300, 700)
    viewer.show()
    app.processEvents()  # paint once before we start timing

    results: dict[str, dict] = {}
    for size_s, label in [(60.0, "60s"), (600.0, "600s"), (7200.0, "7200s")]:
        if size_s > recording.duration_sec:
            print(f"skip {label}: viewport > recording duration "
                  f"({recording.duration_sec:.1f}s)")
            results[label] = {
                "skipped": True,
                "reason": f"recording is {recording.duration_sec:.0f}s "
                           f"< viewport {size_s:.0f}s",
            }
            continue
        print(f"benchmarking {label} viewport ({args.iters} iters)...")
        r = bench_viewport_size(viewer, size_s, recording.duration_sec,
                                  n_iters=args.iters)
        print(f"  mean={r['mean_ms']:.1f}ms  median={r['median_ms']:.1f}ms  "
              f"p95={r['p95_ms']:.1f}ms  max={r['max_ms']:.1f}ms")
        results[label] = r

    peak_mb = _peak_rss_mb()
    print(f"\npeak RSS: {peak_mb:.0f} MB")

    out_path = Path(args.out).expanduser()
    body = _format_results_md(
        recording_path, recording.duration_sec, recording.fs,
        recording.n_channels, results, peak_mb,
    )
    update_spike_results(out_path, body)
    print(f"wrote results → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
