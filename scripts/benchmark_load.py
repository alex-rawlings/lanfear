"""Benchmark loading an OrbitResults ``.npz``: where does the load time go?

The saved format is a DEFLATE-compressed zip (``np.savez_compressed``), and
:meth:`OrbitResults.load` eagerly reads every member, so each load decompresses
the whole archive. This script breaks a load into

  (a) opening the zip container (reading its directory), and
  (b) decompressing each array on access,

so we can see whether the cost is disk or decompression CPU. It then re-encodes
the same data as uncompressed and/or float32 and times a full read of each, to
quantify the speed-up achievable by dropping compression / narrowing the dtype
before committing to any format change.

    load_py313
    python scripts/benchmark_load.py            # uses the default file below
    python scripts/benchmark_load.py --file /path/to/orbits.npz
    python scripts/benchmark_load.py --no-reencode   # diagnostic breakdown only

Re-encode variants are written one at a time (to --workdir, default alongside
the source file) and deleted immediately after timing, so peak extra disk is one
variant. Timings are best-of-``--repeat`` with a warm page cache, so they
reflect decompression CPU rather than cold disk.
"""

import argparse
import gc
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lanfear as lf  # noqa: E402

DEFAULT_FILE = (
    "/orion/ptmp/arawling/tests/core-kick/kick-vel-0000/lanfear_orbits/"
    "orbits_snap_002.npz"
)


def human_bytes(n):
    """Format a byte count as a human-readable string."""
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024 or unit == "TiB":
            return f"{x:.1f} {unit}"
        x /= 1024


def best_time(fn, repeat):
    """Return (best wall time over ``repeat`` runs, last result)."""
    best = float("inf")
    result = None
    for _ in range(repeat):
        gc.collect()
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return best, result


def full_read(path):
    """Read every array in an ``.npz`` into memory (mimics OrbitResults.load I/O)."""
    with np.load(path, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


def breakdown(path, repeat):
    """Time opening the archive and decompressing each member separately."""

    # (a) Opening the zip: reads the central directory only, no array data.
    def _open():
        with np.load(path, allow_pickle=False) as npz:
            return list(npz.files)

    t_open, keys = best_time(_open, repeat)

    # (b) Per-array decompression. NpzFile does not cache members, so repeated
    # npz[key] re-decompresses; opening once outside the timed call isolates the
    # decompression cost from the (already-measured) open cost.
    npz = np.load(path, allow_pickle=False)
    per = []
    try:
        for key in npz.files:
            t, arr = best_time(lambda k=key: npz[k], repeat)
            per.append((key, t, int(arr.nbytes), arr.dtype, arr.shape))
    finally:
        npz.close()
    return t_open, per


def to_float32(arrays):
    """Return a copy with multi-element float64 arrays narrowed to float32."""
    out = {}
    for k, v in arrays.items():
        if v.dtype == np.float64 and v.ndim >= 1:
            out[k] = v.astype(np.float32)
        else:
            out[k] = v
    return out


def time_variant(name, arrays, compressed, workdir, repeat):
    """Write ``arrays`` as an .npz variant, time a full read, then delete it."""
    path = os.path.join(workdir, f"_bench_{name}.npz")
    saver = np.savez_compressed if compressed else np.savez
    saver(path, **arrays)
    size = os.path.getsize(path)
    try:
        t, _ = best_time(lambda: full_read(path), repeat)
    finally:
        os.remove(path)
    return t, size


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", type=str, default=DEFAULT_FILE, help="OrbitResults .npz")
    ap.add_argument("--repeat", type=int, default=3, help="best-of-N timing runs")
    ap.add_argument(
        "--workdir",
        type=str,
        default=None,
        help="where to write re-encode variants (default: source directory)",
    )
    ap.add_argument(
        "--no-reencode",
        action="store_true",
        help="skip the uncompressed/float32 re-encode comparison",
    )
    args = ap.parse_args()

    path = args.file
    if not os.path.exists(path):
        raise SystemExit(f"file not found: {path}")
    workdir = args.workdir or os.path.dirname(os.path.abspath(path))
    disk_size = os.path.getsize(path)

    print(f"file        : {path}")
    print(f"on disk     : {human_bytes(disk_size)}")
    print(f"repeat      : best of {args.repeat} (warm page cache)")
    print()

    # The real thing: OrbitResults.load end to end.
    t_load, res = best_time(lambda: lf.OrbitResults.load(path), args.repeat)
    n = len(res.ids)
    print(f"OrbitResults.load(): {t_load:.3f} s  ({n:,} orbits)")
    print()

    # Where does it go: open vs per-array decompression.
    t_open, per = breakdown(path, args.repeat)
    total_nbytes = sum(nb for _, _, nb, _, _ in per)
    t_decomp = sum(t for _, t, _, _, _ in per)

    print("per-array read (decompression):")
    print(f"  {'array':<15}{'time':>9}{'%':>7}{'uncompressed':>15}  dtype/shape")
    print("  " + "-" * 64)
    for key, t, nb, dt, shape in sorted(per, key=lambda r: -r[1]):
        pct = 100.0 * t / t_decomp if t_decomp else 0.0
        print(
            f"  {key:<15}{t:>8.3f}s{pct:>6.1f}%{human_bytes(nb):>15}  "
            f"{np.dtype(dt).name}{tuple(shape)}"
        )
    print("  " + "-" * 64)
    print(
        f"  {'open archive':<15}{t_open:>8.3f}s"
        f"{'':>7}{'':>15}  (zip directory only)"
    )
    print(
        f"  {'decompress sum':<15}{t_decomp:>8.3f}s{'':>7}"
        f"{human_bytes(total_nbytes):>15}  in RAM once decoded"
    )
    ratio = disk_size / total_nbytes if total_nbytes else float("nan")
    print(
        f"\n  compression ratio {ratio:.2f}x "
        f"({human_bytes(disk_size)} on disk vs {human_bytes(total_nbytes)} raw); "
        f"open is {100.0 * t_open / t_load:.1f}% of load, "
        f"decompress is {100.0 * t_decomp / t_load:.1f}%"
    )

    if args.no_reencode:
        return

    # Re-encode comparison. Load every array once, then write/time each variant
    # sequentially (deleting between) to bound extra disk to a single variant.
    print("\nre-encode comparison (full read of each variant):")
    arrays = full_read(path)
    variants = [
        ("compressed_f64", arrays, True),
        ("uncompressed_f64", arrays, False),
        ("compressed_f32", to_float32(arrays), True),
        ("uncompressed_f32", to_float32(arrays), False),
    ]
    print(f"  {'variant':<20}{'load':>9}{'size':>13}{'speed-up':>11}")
    print("  " + "-" * 52)
    baseline = None
    for name, arr, compressed in variants:
        t, size = time_variant(name, arr, compressed, workdir, args.repeat)
        if baseline is None:
            baseline = t
        print(f"  {name:<20}{t:>8.3f}s{human_bytes(size):>13}{baseline / t:>10.2f}x")
        if "f32" in name:
            del arr
            gc.collect()
    print(
        "\n  (speed-up is relative to the current on-disk format, "
        "compressed float64.)"
    )


if __name__ == "__main__":
    main()
