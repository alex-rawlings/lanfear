"""Null test of the boxlet resonance search on saved orbit results.

A box orbit is labelled a boxlet when its fundamental frequencies satisfy a
low-order commensurability ``|n . w| / max|w| < tol``. Some of those hits are
chance coincidences. Shuffling each axis's frequencies independently between
the box orbits keeps the frequency distributions but destroys every real
resonance, so the hit rate of the shuffled triples estimates the false-positive
rate. This prints, per resonance order and for the most common resonance
vectors, the hit fractions of the real and shuffled box frequencies at several
tolerances, and the implied purity ``1 - null / real``.

    python scripts/boxlet_null_test.py orbits.npz --tol 1e-2 2e-3 1e-3 5e-4
"""

import argparse
import collections

import numpy as np

import lanfear as lf
from lanfear.classify import (
    OrbitClass,
    OrbitShape,
    _find_resonances,
    _named_resonance_vectors,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file", type=str, help="OrbitResults .npz")
    ap.add_argument("--tol", type=float, nargs="+", default=[1e-2, 2e-3, 1e-3, 5e-4])
    ap.add_argument("--max-order", type=int, default=7)
    ap.add_argument("--shuffles", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    lf.set_verbosity("WARNING")

    res = lf.OrbitResults.load(args.file).drop_runaways()
    cl = res.classify(drop_runaways=False)
    regular_box = (
        res.ok & (cl.shape == OrbitShape.BOX) & (cl.labels != OrbitClass.IRREGULAR)
    )
    w = np.abs(np.asarray(res.fundamentals[regular_box], dtype=np.float64))
    rng = np.random.default_rng(args.seed)
    nulls = [
        np.stack([rng.permutation(w[:, a]) for a in range(3)], axis=1)
        for _ in range(args.shuffles)
    ]
    extra = _named_resonance_vectors(args.max_order)
    print(f"{len(w)} regular box orbits; {args.shuffles} independent shuffles")

    for tol in args.tol:
        real = collections.Counter(
            _find_resonances(w, args.max_order, tol, extra=extra)[1].tolist()
        )
        null = collections.Counter()
        for shuffled in nulls:
            null.update(
                _find_resonances(shuffled, args.max_order, tol, extra=extra)[1].tolist()
            )
        print(f"\ntol = {tol:g}: order  real   null   purity")
        for order in range(2, args.max_order + 1):
            r = real[order] / len(w)
            n = null[order] / (len(w) * args.shuffles)
            purity = 1 - n / r if r > 0 else float("nan")
            print(f"{'':12s}{order:5d}  {r:.3f}  {n:.3f}  {purity:6.2f}")

    tol = min(args.tol)
    vec_real = _find_resonances(w, args.max_order, tol, extra=extra)
    hits = collections.Counter(map(tuple, vec_real[0][vec_real[1] >= 2].tolist()))
    null_hits = collections.Counter()
    for shuffled in nulls:
        v, o, _, _ = _find_resonances(shuffled, args.max_order, tol, extra=extra)
        null_hits.update(map(tuple, v[o >= 2].tolist()))
    print(f"\nmost common resonance vectors at tol = {tol:g} (real / mean null):")
    for vector, count in hits.most_common(15):
        print(
            f"  {str(vector):>14s}  {count:6d} / {null_hits[vector] / args.shuffles:7.1f}"
        )


if __name__ == "__main__":
    main()
