"""Orbit classification from integrated + frequency-analysed orbits.

Following the standard picture (e.g. Carpintero & Aguilar 1998; Frigo et al.
2021), orbits are assigned to families from three cheap, already-reduced
per-orbit quantities:

* **Angular-momentum circulation.** A tube (loop) orbit circulates about one
  principal axis, so the corresponding component of angular momentum keeps its
  sign: ``circ_a = |<L_a>| / <|L_a|>`` is near 1. A box circulates about none
  (all ``circ_a`` small). The circulating axis names the tube: z -> short-axis
  tube, x -> long-axis tube, y -> (unstable) intermediate-axis tube.
* **Shape tensor.** Its smallest eigenvalue vanishes for a planar orbit in any
  orientation, identifying rosettes (2-D loops, typical of near-spherical
  potentials) and separating them from thick 3-D tubes.
* **Fundamental frequencies.** A low-order commensurability
  ``n_x w_x + n_y w_y + n_z w_z ~ 0`` among the axis fundamentals marks a
  resonant box (boxlet). Requires frequency data (:func:`analyse_family`).

The classification runs on the compact arrays in an :class:`OrbitResults`, so it
is trivially fast even for millions of orbits.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import gcd

import numpy as np


class OrbitClass(IntEnum):
    UNCLASSIFIED = 0
    BOX = 1
    BOXLET = 2  # resonant box (banana/fish/pretzel/...)
    SHORT_AXIS_TUBE = 3  # z-tube
    INNER_LONG_AXIS_TUBE = 4  # inner x-tube
    OUTER_LONG_AXIS_TUBE = 5  # outer x-tube
    INTERMEDIATE_AXIS_TUBE = 6  # y-tube (generally unstable)
    ROSETTE = 7  # planar loop


CLASS_NAMES = {c.value: c.name.lower() for c in OrbitClass}


@dataclass
class OrbitClassification:
    """Result of :func:`classify_orbits` (all arrays indexed like the orbits)."""

    labels: np.ndarray  # (N,) OrbitClass values
    circulation: np.ndarray  # (N,3) |<L_a>|/<|L_a|> per axis
    tube_axis: np.ndarray  # (N,) index of dominant circulation axis
    planarity: np.ndarray  # (N,) lambda_min/lambda_max of shape tensor
    resonance: np.ndarray  # (N,3) primitive resonance vector (0 if none)
    resonance_order: np.ndarray  # (N,) |n|_1 of the resonance (0 if none)

    @property
    def names(self) -> np.ndarray:
        """Per-orbit class name strings."""
        return np.array([CLASS_NAMES[int(v)] for v in self.labels])

    def counts(self) -> dict:
        """Mapping class name -> number of orbits."""
        vals, cnts = np.unique(self.labels, return_counts=True)
        return {CLASS_NAMES[int(v)]: int(c) for v, c in zip(vals, cnts)}

    def mask(self, cls: OrbitClass) -> np.ndarray:
        return self.labels == int(cls)


def _primitive_resonance_vectors(max_order: int):
    """Primitive integer triples n with |n|_1<=max_order, canonical sign, by order.

    The bound is on the L1 order |n_x|+|n_y|+|n_z| -- the physically meaningful
    resonance order -- not on the individual components, so only genuinely
    low-order commensurabilities (banana 2:-1:0, fish, pretzel, ...) qualify.
    """
    out = []
    rng = range(-max_order, max_order + 1)
    for nx in rng:
        for ny in rng:
            for nz in rng:
                order = abs(nx) + abs(ny) + abs(nz)
                if order == 0 or order > max_order:
                    continue
                if gcd(gcd(abs(nx), abs(ny)), abs(nz)) != 1:
                    continue
                first = nx if nx != 0 else (ny if ny != 0 else nz)
                if first < 0:  # canonical sign
                    continue
                out.append((order, (nx, ny, nz)))
    out.sort(key=lambda t: t[0])
    orders = np.array([o for o, _ in out], dtype=np.int64)
    vecs = np.array([v for _, v in out], dtype=np.float64)  # (M,3)
    return vecs, orders


def _find_resonances(w, max_order, tol, chunk=20000):
    """Lowest-order commensurability of each frequency triple w (N,3).

    Returns (vectors (N,3) int, orders (N,)); order 0 means none found.
    """
    vecs, orders = _primitive_resonance_vectors(max_order)
    N = len(w)
    res_vec = np.zeros((N, 3), dtype=np.int64)
    res_ord = np.zeros(N, dtype=np.int64)
    scale = np.maximum(np.max(np.abs(w), axis=1), 1e-30)  # (N,)
    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        # |n . w| / max|w| for every candidate; vecs are sorted by order, so the
        # first hit is the lowest-order commensurability.
        resid = np.abs(w[lo:hi] @ vecs.T)  # (n, M)
        norm = resid / scale[lo:hi, None]
        hit = norm < tol
        any_hit = hit.any(axis=1)
        first = np.argmax(hit, axis=1)  # first (lowest order)
        idx = np.where(any_hit)[0]
        res_vec[lo + idx] = vecs[first[idx]].astype(np.int64)
        res_ord[lo + idx] = orders[first[idx]]
    return res_vec, res_ord


def classify_orbits(
    results,
    circ_thresh: float = 0.7,
    freq_tol: float = 0.05,
    amp_frac: float = 0.05,
    planar_thresh: float = 0.02,
    resonance_max_order: int = 5,
    resonance_tol: float = 0.01,
    inner_outer_thresh: float = 0.2,
) -> OrbitClassification:
    """Classify the orbits in an :class:`~lanfear.OrbitResults`.

    Parameters
    ----------
    circ_thresh:
        ``circ_a`` above this counts as circulation about axis ``a`` (tube).
    freq_tol:
        a loop whose (amplitude-active) axis fundamentals are mutually equal to
        within this fraction is a rosette (1:1:1); a tube is 1:1:pi (two axes
        share a frequency, the circulation axis differs).
    amp_frac:
        an axis whose leading spectral amplitude exceeds this fraction of the
        largest is "active" (used to ignore silent axes in the 1:1:1 test).
    planar_thresh:
        without frequency data, a loop with shape-tensor
        ``lambda_min/lambda_max`` below this is taken to be a (planar) rosette.
    resonance_max_order, resonance_tol:
        search window and tolerance (``|n.w|/max|w|``) for the boxlet
        commensurability.
    inner_outer_thresh:
        long-axis tubes with a relative "hole" ``rho_x_min / rms_perp`` below
        this are labelled *inner*, else *outer* (a convex/non-convex proxy).
    """
    c = results.column
    status = c("status")
    N = len(status)

    Lm = np.stack([c("Lx_mean"), c("Ly_mean"), c("Lz_mean")], axis=1)
    La = np.stack([c("Lx_abs_mean"), c("Ly_abs_mean"), c("Lz_abs_mean")], axis=1)
    circ = np.abs(Lm) / np.maximum(La, 1e-30)  # (N,3)
    tube_axis = np.argmax(circ, axis=1)
    n_circ = np.sum(circ > circ_thresh, axis=1)

    # Shape tensor -> planarity (smallest / largest eigenvalue).
    S = np.zeros((N, 3, 3))
    S[:, 0, 0] = c("Sxx")
    S[:, 1, 1] = c("Syy")
    S[:, 2, 2] = c("Szz")
    S[:, 0, 1] = S[:, 1, 0] = c("Sxy")
    S[:, 0, 2] = S[:, 2, 0] = c("Sxz")
    S[:, 1, 2] = S[:, 2, 1] = c("Syz")
    evals = np.linalg.eigvalsh(S)  # ascending (N,3)
    planarity = evals[:, 0] / np.maximum(evals[:, 2], 1e-30)

    # Rosette test: are the active-axis fundamentals mutually 1:1:1?
    res_vec = np.zeros((N, 3), dtype=np.int64)
    res_ord = np.zeros(N, dtype=np.int64)
    if results.fundamentals is not None:
        w = np.abs(results.fundamentals)  # (N,3)
        amp = (
            results.lines[:, :, 0, 1] if results.lines is not None else np.ones_like(w)
        )
        active = amp > amp_frac * np.max(amp, axis=1, keepdims=True)
        n_active = np.sum(active, axis=1)
        w_hi = np.where(active, w, -np.inf).max(axis=1)
        w_lo = np.where(active, w, np.inf).min(axis=1)
        freq_111 = (n_active >= 2) & (
            w_hi <= (1.0 + freq_tol) * np.maximum(w_lo, 1e-30)
        )
        res_vec, res_ord = _find_resonances(w, resonance_max_order, resonance_tol)
    else:
        # No frequency data: fall back to the shape-tensor planarity.
        freq_111 = planarity < planar_thresh

    labels = np.full(N, OrbitClass.UNCLASSIFIED, dtype=np.int64)
    ok = status == 0
    is_loop = n_circ >= 1

    # Rosette: a loop whose axis fundamentals are all commensurate 1:1:1.
    labels[ok & is_loop & freq_111] = OrbitClass.ROSETTE

    # Tubes: a single circulation axis, not 1:1:1.
    tube = ok & is_loop & ~freq_111
    labels[tube & (tube_axis == 2)] = OrbitClass.SHORT_AXIS_TUBE
    labels[tube & (tube_axis == 1)] = OrbitClass.INTERMEDIATE_AXIS_TUBE
    # Long-axis (x) tubes split inner/outer by relative hole size.
    lat = tube & (tube_axis == 0)
    rms_perp = np.sqrt(np.maximum(c("Syy") + c("Szz"), 1e-30))
    hole = c("rho_x_min") / rms_perp
    labels[lat & (hole < inner_outer_thresh)] = OrbitClass.INNER_LONG_AXIS_TUBE
    labels[lat & (hole >= inner_outer_thresh)] = OrbitClass.OUTER_LONG_AXIS_TUBE

    # Boxes: no circulation. A low-order (>=2) resonance marks a boxlet.
    box = ok & ~is_loop
    labels[box] = OrbitClass.BOX
    labels[box & (res_ord >= 2)] = OrbitClass.BOXLET

    return OrbitClassification(
        labels=labels,
        circulation=circ,
        tube_axis=tube_axis,
        planarity=planarity,
        resonance=res_vec,
        resonance_order=res_ord,
    )
