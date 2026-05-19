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
           batch_seconds: float = 60.0,
           overview_points: int = 8000) -> dict:
    """Convert a (n_ch, N) MATLAB-style HDF5 → (N, n_ch) flat HDF5
    plus a pre-downsampled `/y_overview` for fast wide-viewport reads.

    The flat dataset `y` is what the signal viewer reads at narrow
    viewports (single contiguous chunk per pan). At wide viewports —
    where the user wants to see the whole recording or a multi-minute
    slice — reading the full data is bandwidth-bound (M1.4 finding).
    `y_overview` is a coarse min-max-pairs summary of the same signal
    at ~`overview_points` resolution per channel, regardless of
    recording length. Reads from it are sub-millisecond.

    The overview uses min-max downsampling (each output row alternates
    min/max within a bucket of source samples), so wide-zoom paint
    preserves the extremes — important for catching motion-artifact
    spikes that a naive subsample would skip.

    Parameters
    ----------
    dtype             : float32 (default) — halves on-disk size vs
                        the source float64. Visually equivalent at
                        scope range.
    chunk_seconds     : HDF5 chunk size along the time axis (10s ≈
                        244k rows at fs=24414Hz; pyqtgraph's sweet
                        spot for its peak downsampler).
    batch_seconds     : how much to read from the source per
                        transpose-write cycle. Caps peak RAM at
                        ~batch_seconds × n_ch × 8 bytes.
    overview_points   : approximate per-channel rows in /y_overview.
                        8000 is plenty for a wide-zoom paint at any
                        reasonable display width.
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

        # M2.0 overview accumulator. We compute it on the fly during
        # the main flat-write pass — no second read pass over the
        # source file required.
        #
        # Bucketing: divide n_samples into `overview_buckets`
        # contiguous bins. For each bin we emit one (min, max) pair,
        # so /y_overview ends up shape (2 * overview_buckets, n_ch).
        # We size buckets to land near `overview_points / 2` so the
        # final overview has ~overview_points rows.
        overview_buckets = max(2, overview_points // 2)
        bucket_size = max(1, n_samples // overview_buckets)
        # n_samples may not divide evenly — recompute the actual
        # bucket count from bucket_size so we don't lose the tail.
        n_buckets = (n_samples + bucket_size - 1) // bucket_size
        overview_rows = 2 * n_buckets
        ov_dst = dst.create_dataset(
            "y_overview",
            shape=(overview_rows, n_ch),
            dtype=dtype,
        )
        dst.attrs["overview_bucket_size"] = int(bucket_size)
        dst.attrs["overview_n_buckets"] = int(n_buckets)

        # Stream the conversion in batches. We align batches to bucket
        # boundaries so each batch can contribute whole min/max pairs
        # to /y_overview without inter-batch fixup.
        batch_rows = max(batch_rows, bucket_size)

        bucket_cursor = 0  # how many buckets we've written so far
        for i0 in range(0, n_samples, batch_rows):
            i1 = min(i0 + batch_rows, n_samples)
            if transposed:
                batch = y_src[:, i0:i1].T  # (rows, n_ch) view
            else:
                batch = y_src[i0:i1, :]
            batch_f = batch.astype(dtype, copy=False)
            dst["y"][i0:i1, :] = batch_f

            # Compute (min, max) per bucket within this batch.
            b_rows = batch_f.shape[0]
            batch_buckets = (b_rows + bucket_size - 1) // bucket_size
            for bi in range(batch_buckets):
                bs = bi * bucket_size
                be = min(bs + bucket_size, b_rows)
                window = batch_f[bs:be, :]
                ov_dst[2 * (bucket_cursor + bi), :] = window.min(axis=0)
                ov_dst[2 * (bucket_cursor + bi) + 1, :] = window.max(axis=0)
            bucket_cursor += batch_buckets

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
        "overview_rows": int(overview_rows),
        "overview_bucket_size": int(bucket_size),
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
    print(f"  overview rows      : {result['overview_rows']:,} "
          f"(bucket size {result['overview_bucket_size']:,} samples)")
    print(f"  conversion elapsed : {result['elapsed_sec']:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
