"""Ingest a `_notched.mat` (or similar) into a fast-read flat HDF5.

The existing MATLAB files store data as `/y` shape `(n_ch, N)`. h5py's
lazy slice `y[:, i0:i1]` then reads 5 non-contiguous rows per viewport,
which on the M1 spike measured ~350ms for a 60s window on a 22-min
file — well over the 100ms target.

Re-saving the data as `(N, n_ch)` with chunked storage along the time
axis gives single contiguous-chunk reads per viewport: ~12ms for the
same 60s window. A 28x speedup, taking us under the M1 bar.

The cost is a one-time conversion per recording (~10s for a 22-min
file; ~1min for a 2h file). The output goes next to the source:

    <recording>.flat.h5

Pass that to LazyRecording instead of the original `.mat` and reads
become contiguous.

Usage:
    python scripts/m1_ingest.py /path/to/recording.mat
    python scripts/m1_ingest.py /path/to/recording.mat --out /tmp/cached.h5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np


def flat_h5_path_for(source: Path) -> Path:
    """Conventional location for the flat-HDF5 cache next to `source`."""
    return source.with_suffix(source.suffix + ".flat.h5")


def ingest(source: Path, out: Path,
           dtype: str = "float32",
           chunk_seconds: float = 10.0,
           batch_seconds: float = 60.0) -> dict:
    """Convert a (n_ch, N) MATLAB-style HDF5 → (N, n_ch) flat HDF5.

    - `dtype="float32"` halves disk size vs the source float64 and is
      indistinguishable visually at scope range.
    - `chunk_seconds` controls the HDF5 chunk size along the time
      axis. 10s chunks at fs=24414Hz = 244,140 rows per chunk, which
      is in pyqtgraph's sweet spot for downsampling.
    - `batch_seconds` controls how much we read from the source per
      transpose-and-write cycle — bigger is faster but uses more RAM
      during conversion (one batch lives in memory at a time).
    """
    source = Path(source)
    out = Path(out)
    t0 = time.perf_counter()

    with h5py.File(str(source), "r") as src, h5py.File(str(out), "w") as dst:
        # Identify the data dataset (same logic as LazyRecording)
        if "yOut" in src:
            key = "yOut"
        elif "y" in src:
            key = "y"
        else:
            raise ValueError(
                f"{source}: no /y or /yOut dataset; keys = {list(src.keys())}"
            )
        y_src = src[key]
        a, b = y_src.shape
        if a <= 32 and b > 32:
            n_ch, n_samples = a, b
            transposed = True
        else:
            n_samples, n_ch = a, b
            transposed = False
        fs = float(np.asarray(src["fs"][()]).squeeze())

        chunk_rows = max(1, int(round(chunk_seconds * fs)))
        batch_rows = max(chunk_rows, int(round(batch_seconds * fs)))

        # gzip-compressed chunked dataset in (N, n_ch) order.
        dst.create_dataset(
            "y",
            shape=(n_samples, n_ch),
            dtype=dtype,
            chunks=(chunk_rows, n_ch),
            compression="gzip",
            compression_opts=2,  # level 2: cheap CPU, ~30% size reduction
            shuffle=True,
        )
        dst.create_dataset("fs", data=np.float64(fs))
        dst.attrs["source_path"] = str(source)
        dst.attrs["source_key"] = key
        dst.attrs["chunk_seconds"] = float(chunk_seconds)
        dst.attrs["ingest_dtype"] = dtype

        # Stream the conversion in batches to keep peak RAM bounded
        # at ~ batch_seconds × n_ch × 8 bytes (= ~24 MB for default
        # 60s × 5 channels float64).
        for i0 in range(0, n_samples, batch_rows):
            i1 = min(i0 + batch_rows, n_samples)
            if transposed:
                batch = y_src[:, i0:i1].T  # (rows, n_ch) view
            else:
                batch = y_src[i0:i1, :]
            dst["y"][i0:i1, :] = batch.astype(dtype, copy=False)

    elapsed = time.perf_counter() - t0
    size_mb = out.stat().st_size / (1024 * 1024)
    return {
        "source": str(source),
        "out": str(out),
        "elapsed_sec": elapsed,
        "size_mb": size_mb,
        "fs": fs,
        "n_samples": n_samples,
        "n_channels": n_ch,
        "transposed_source": transposed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="path to source .mat / .h5")
    ap.add_argument("--out", default=None,
                    help="output path (default: <source>.flat.h5 next to source)")
    ap.add_argument("--dtype", default="float32",
                    choices=["float16", "float32", "float64"])
    ap.add_argument("--chunk-seconds", type=float, default=10.0)
    ap.add_argument("--batch-seconds", type=float, default=60.0)
    args = ap.parse_args()

    source = Path(args.source).expanduser()
    if not source.exists():
        print(f"error: source not found at {source}", file=sys.stderr)
        return 2
    out = (Path(args.out).expanduser() if args.out
           else flat_h5_path_for(source))
    print(f"ingesting {source}")
    print(f"  → {out}")
    print(f"  dtype={args.dtype}, chunk_seconds={args.chunk_seconds}, "
          f"batch_seconds={args.batch_seconds}")

    result = ingest(source, out,
                     dtype=args.dtype,
                     chunk_seconds=args.chunk_seconds,
                     batch_seconds=args.batch_seconds)
    print(f"  fs                 : {result['fs']:.2f} Hz")
    print(f"  channels           : {result['n_channels']}")
    print(f"  samples            : {result['n_samples']:,}")
    print(f"  source layout      : {'(n_ch, N)' if result['transposed_source'] else '(N, n_ch)'}")
    print(f"  output size        : {result['size_mb']:.1f} MB")
    print(f"  conversion elapsed : {result['elapsed_sec']:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
