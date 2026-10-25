"""Compare lanfear's orbit classification with the exact families of a perfect ellipsoid.

The perfect ellipsoid is integrable (its potential is of Staeckel form), so the
family of every orbit -- box, short-axis tube, inner or outer long-axis tube --
is known exactly. ``scripts/make_perfect_ellipsoid.py`` (run with AGAMA)
writes an equilibrium N-body model and the reference family of each particle,
optionally for a figure tumbling about its short axis (reference families then
from the body-frame circulation of long AGAMA integrations, plus Lyapunov
exponents and the inertial sense of short-axis tubes). This script runs
lanfear's normal pipeline on that snapshot (prepare with the model's known
pattern speed, SCF fit, integration, classification) for a random subsample of
particles and reports:

* how well the SCF fit reproduces the exact potential;
* the confusion matrix of lanfear's geometric shape (:attr:`OrbitClassification.shape`,
  i.e. before the irregular override, with boxlets counted as boxes) against
  the reference family, overall and in bins of the orbit-averaged radius;
* the confusion matrix restricted to orbits AGAMA finds regular (Lyapunov);
* the same with each rosette counted as the tube about its circulation axis
  (lanfear's rosettes are near-spherical loops, which have no Staeckel
  family of their own);
* lanfear's irregular/diffusing flags against AGAMA's chaotic orbits (in the
  static figure every orbit is regular), and the unresolved fraction;
* in a tumbling figure, lanfear's prograde/retrograde sense of the short-axis
  tubes against the sign of their net inertial winding about z, and the
  accuracy without the hole cut about the pattern axis (``hole_thresh=0``);
* a scan of ``circ_thresh`` (box/tube) and ``inner_outer_ratio`` (the
  frequency-ratio cut between inner and outer long-axis tubes) against the
  reference, i.e. the static calibration of milestone M10a. The inner/outer
  scan is scored on the orbits both lanfear and the reference call long-axis
  tubes, so that tube-axis errors do not mask it.

    python scripts/compare_perfect_ellipsoid.py --dir /orion/ptmp/arawling/orbits/agama

The integration is saved as ``lanfear_orbits_<n>.npz`` in ``--dir`` and reused
on later runs (``--reintegrate`` forces a new one). Under MPI (``srun``) the
orbits are spread over the ranks.
"""

import argparse
import json
import os

import numpy as np
from scipy.integrate import quad

import lanfear as lf
from lanfear.classify import OrbitShape

REFERENCE = ("box", "short_axis_tube", "inner_long_axis_tube", "outer_long_axis_tube")
# lanfear shape -> reference family name (shapes with no counterpart map to themselves).
SHAPE_TO_FAMILY = {
    int(OrbitShape.BOX): "box",
    int(OrbitShape.SHORT_AXIS_TUBE): "short_axis_tube",
    int(OrbitShape.INNER_LONG_AXIS_TUBE): "inner_long_axis_tube",
    int(OrbitShape.OUTER_LONG_AXIS_TUBE): "outer_long_axis_tube",
    int(OrbitShape.INTERMEDIATE_AXIS_TUBE): "intermediate_axis_tube",
    int(OrbitShape.ROSETTE): "rosette",
    int(OrbitShape.UNCLASSIFIED): "unclassified",
}
COLUMNS = (*REFERENCE, "intermediate_axis_tube", "rosette", "unclassified")


def exact_potential(points, mass, axes, G):
    """Exact perfect-ellipsoid potential (see make_perfect_ellipsoid.py).

    Parameters
    ----------
    points : numpy.ndarray
        (N, 3) positions.
    mass : float
        Total mass.
    axes : sequence of float
        (a, b, c).
    G : float
        Gravitational constant.

    Returns
    -------
    phi : numpy.ndarray
        (N,) potential.
    """
    a2 = np.asarray(axes, dtype=float) ** 2
    out = np.empty(len(points))
    for k, x in enumerate(points):
        x2 = np.asarray(x, dtype=float) ** 2

        def integrand(s):
            u = s / (1.0 - s)
            return (
                1.0
                / (np.sqrt(np.prod(a2 + u)) * (1.0 + np.sum(x2 / (a2 + u))))
                / (1.0 - s) ** 2
            )

        value, _ = quad(integrand, 0.0, 1.0, limit=200, epsabs=0, epsrel=1e-10)
        out[k] = -G * mass / np.pi * value
    return out


def confusion(reference, predicted):
    """Counts of (reference, predicted) family pairs.

    Parameters
    ----------
    reference, predicted : numpy.ndarray
        (N,) family names.

    Returns
    -------
    matrix : numpy.ndarray
        (len(REFERENCE), len(COLUMNS)) counts.
    """
    matrix = np.zeros((len(REFERENCE), len(COLUMNS)), dtype=np.int64)
    for i, ref in enumerate(REFERENCE):
        for j, col in enumerate(COLUMNS):
            matrix[i, j] = np.sum((reference == ref) & (predicted == col))
    return matrix


def print_confusion(matrix, title):
    """Print a confusion matrix with per-family recall and precision."""
    total = matrix.sum()
    print(f"\n{title} ({total} orbits; rows: reference, columns: lanfear)")
    header = "".join(f"{c[:12]:>13s}" for c in COLUMNS)
    print(f"{'':>22s}{header} {'recall':>8s}")
    for i, ref in enumerate(REFERENCE):
        row = "".join(f"{n:13d}" for n in matrix[i])
        recall = matrix[i, i] / max(matrix[i].sum(), 1)
        print(f"{ref:>22s}{row} {recall:8.3f}")
    precision = [
        matrix[j, j] / max(matrix[:, j].sum(), 1) for j in range(len(REFERENCE))
    ]
    print(f"{'precision':>22s}" + "".join(f"{p:13.3f}" for p in precision))
    print(f"accuracy {np.trace(matrix[:, : len(REFERENCE)]) / max(total, 1):.4f}")


def predicted_families(classification, fold_rosettes=False):
    """lanfear's shape, as reference family names.

    Parameters
    ----------
    classification : lanfear.OrbitClassification
        The classification.
    fold_rosettes : bool, optional
        Count each rosette as the tube about its circulation axis
        (:attr:`~lanfear.OrbitClassification.tube_axis`): a short-axis tube about
        z, an intermediate-axis tube about y, and about x an inner or outer
        long-axis tube by the classification's own frequency-ratio cut
        (:attr:`~lanfear.OrbitClassification.inner_outer_ratio`). A rosette is a
        loop whose frequencies are nearly equal, as in a nearly spherical
        potential; the Staeckel families have no counterpart for it.

    Returns
    -------
    families : numpy.ndarray
        (N,) family names.
    """
    # Wide enough for every name, so that folding never truncates one.
    width = max(len(name) for name in SHAPE_TO_FAMILY.values())
    families = np.array(
        [SHAPE_TO_FAMILY[int(s)] for s in classification.shape], dtype=f"<U{width}"
    )
    if fold_rosettes:
        rosette = classification.shape == OrbitShape.ROSETTE
        axis = classification.tube_axis
        w = np.abs(np.asarray(classification.fundamentals, dtype=float))
        with np.errstate(invalid="ignore", divide="ignore"):
            inner = (
                w[:, 0] / np.sqrt(w[:, 1] * w[:, 2]) < classification.inner_outer_ratio
            )
        families[rosette & (axis == 2)] = "short_axis_tube"
        families[rosette & (axis == 1)] = "intermediate_axis_tube"
        families[rosette & (axis == 0) & inner] = "inner_long_axis_tube"
        families[rosette & (axis == 0) & ~inner] = "outer_long_axis_tube"
    return families


def balanced_accuracy(reference, predicted, classes):
    """Mean recall over ``classes`` (orbits of other reference classes ignored)."""
    recalls = [
        np.mean(predicted[reference == c] == c)
        for c in classes
        if np.any(reference == c)
    ]
    return float(np.mean(recalls))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--dir", required=True, help="directory of make_perfect_ellipsoid.py"
    )
    ap.add_argument("--n-orbits", type=int, default=50_000)
    ap.add_argument("--n-max", type=int, default=24)
    ap.add_argument("--l-max", type=int, default=8)
    ap.add_argument("--periods", type=int, default=50)
    ap.add_argument("--samples", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reintegrate", action="store_true")
    args = ap.parse_args()
    lf.set_verbosity("INFO")

    try:
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.Get_rank()
    except Exception:
        rank = 0

    with open(os.path.join(args.dir, "perfect_ellipsoid_model.json")) as f:
        model = json.load(f)
    orbits_path = os.path.join(args.dir, f"lanfear_orbits_{args.n_orbits}.npz")

    results = None
    if os.path.exists(orbits_path) and not args.reintegrate:
        if rank == 0:
            results = lf.OrbitResults.load(orbits_path)
            print(f"loaded {orbits_path}")
    else:
        potential = subset = None
        if rank == 0:
            ps = lf.ParticleSystem.from_gadget_hdf5(
                os.path.join(args.dir, "perfect_ellipsoid_ics.hdf5")
            )
            ps.prepare(
                centre="shrinking_sphere",
                pattern_speed=model["pattern_speed"] or "none",
            )
            potential = lf.Potential.from_particles(
                ps, n_max=args.n_max, l_max=args.l_max
            )
            # SCF fit against the exact potential (in the prepared frame; the
            # exact potential is symmetric under axis-sign flips).
            rng = np.random.default_rng(args.seed)
            test = ps.pos[rng.choice(ps.n_particles, 300, replace=False)]
            exact = exact_potential(test, model["mass"], model["axes_kpc"], model["G"])
            scf = potential.potential(test) * potential.velocity_unit**2
            err = np.abs(scf / exact - 1.0)
            print(
                f"SCF (n_max={args.n_max}, l_max={args.l_max}) vs exact potential at "
                f"particle positions: median rel. error {np.median(err):.2e}, "
                f"90th pct {np.percentile(err, 90):.2e}, max {err.max():.2e}"
            )
            subset = ps.random_subset(
                args.n_orbits, rng=np.random.default_rng(args.seed)
            )
        results = lf.analyse_family(
            potential,
            subset,
            family="STAR",
            n_periods=args.periods,
            n_samples=args.samples,
            comm="auto",
        )
        if rank == 0:
            results.save(orbits_path)
    if rank != 0:
        return

    reference_file = np.load(os.path.join(args.dir, "perfect_ellipsoid_reference.npz"))
    row_of_id = {
        int(i): k for k, i in enumerate(reference_file["particle_ids"].tolist())
    }

    cl = results.classify()
    rows = np.array([row_of_id[int(i)] for i in cl.ids])
    reference = reference_file["particle_family"][rows]
    # AGAMA's normalised Lyapunov exponent: 0 without evidence of chaos.
    chaotic = reference_file["particle_lyapunov"][rows] > 0
    inertial_turns = reference_file["particle_inertial_turns_z"][rows]
    tumbling = bool(model["pattern_speed"])
    print(
        f"\npattern speed {model['pattern_speed']:.4g} km/s/kpc"
        + (
            f" (corotation at {model['corotation_radius_half_mass']:g} half-mass radii)"
            if tumbling
            else " (static)"
        )
        + f"; reference families by {model['reference_method']}; chaotic (AGAMA "
        f"Lyapunov) {chaotic.mean():.2%} of the classified orbits"
    )
    predicted = predicted_families(cl)
    known = np.isin(reference, REFERENCE)
    print(
        f"\n{len(cl.ids)} classified orbits; {int(np.sum(~known))} with an ambiguous "
        f"reference family left out"
    )
    print(
        f"inner/outer long-axis-tube cut: {cl.inner_outer_ratio:.3f} "
        f"({'adaptive' if cl.inner_outer_adaptive else 'fixed'})"
    )
    print_confusion(confusion(reference[known], predicted[known]), "All radii")
    print_confusion(
        confusion(reference[known & ~chaotic], predicted[known & ~chaotic]),
        "All radii, regular orbits (AGAMA Lyapunov)",
    )
    # Rosettes have no Staeckel counterpart: score them as the tube about their
    # circulation axis.
    folded = predicted_families(cl, fold_rosettes=True)
    rosette = cl.shape == OrbitShape.ROSETTE
    print(
        f"\nrosettes folded by tube axis: {int(rosette.sum())} "
        f"({', '.join(f'{n}: {int(np.sum(rosette & (folded == n)))}' for n in COLUMNS if np.any(rosette & (folded == n)))})"
    )
    print_confusion(
        confusion(reference[known], folded[known]), "All radii, rosettes folded"
    )
    print_confusion(
        confusion(reference[known & ~chaotic], folded[known & ~chaotic]),
        "All radii, regular orbits, rosettes folded",
    )

    radius = cl.radius_orbit_averaged / model["axes_kpc"][0]
    for lo, hi in ((0, 1), (1, 3), (3, 10), (10, np.inf)):
        sel = known & (radius >= lo) & (radius < hi)
        if sel.any():
            print_confusion(
                confusion(reference[sel], predicted[sel]), f"<r> in [{lo}, {hi}) a"
            )
            print(
                f"  with rosettes folded: accuracy "
                f"{np.mean(folded[sel] == reference[sel]):.4f}"
            )

    irregular = np.isin(cl.regularity_names, ["irregular", "diffusing"])
    print(
        f"\nflagged irregular: {irregular.mean():.3%} (spectral "
        f"{np.mean(cl.regularity_names == 'irregular'):.3%}, diffusing "
        f"{np.mean(cl.regularity_names == 'diffusing'):.3%}); unresolved "
        f"{cl.unresolved.mean():.3%}"
    )
    print(
        f"  of AGAMA's chaotic orbits ({int(chaotic.sum())}): {np.mean(irregular[chaotic]):.1%} "
        f"flagged (spectral {np.mean(cl.regularity_names[chaotic] == 'irregular'):.1%}); "
        f"of its regular ones: {np.mean(irregular[~chaotic]):.2%} flagged"
    )
    wrong = known & (predicted != reference)
    print(
        f"misclassified: {wrong.sum()} ({wrong.mean():.2%}); of these flagged irregular "
        f"{np.mean(irregular[wrong]):.1%}, unresolved {np.mean(cl.unresolved[wrong]):.1%} "
        f"(vs {np.mean(irregular[known & ~wrong]):.1%} / "
        f"{np.mean(cl.unresolved[known & ~wrong]):.1%} of the correct ones)"
    )

    if tumbling:
        # Sense of the short-axis tubes both sides agree on: the sign of the net
        # inertial winding about z (the pattern-speed axis, +z after prepare()).
        sat = (
            (reference == "short_axis_tube")
            & (predicted == "short_axis_tube")
            & ~chaotic
        )
        truth = np.where(inertial_turns[sat] > 0, "prograde", "retrograde")
        sense = cl.sense_names[sat]
        print(
            f"\nsense of {int(sat.sum())} regular short-axis tubes "
            f"(rows: reference, inertial winding; columns: lanfear):"
        )
        names = ("prograde", "retrograde", "locked", "librating", "none")
        print(f"{'':>12s}" + "".join(f"{n:>12s}" for n in names))
        for t in ("prograde", "retrograde"):
            print(
                f"{t:>12s}"
                + "".join(f"{np.sum((truth == t) & (sense == n)):12d}" for n in names)
            )
        agree = np.mean(
            sense[np.isin(sense, ("prograde", "retrograde"))]
            == truth[np.isin(sense, ("prograde", "retrograde"))]
        )
        print(f"  agreement where lanfear gives a sense: {agree:.4f}")
        no_hole = predicted_families(results.classify(hole_thresh=0.0))
        print(
            f"accuracy without the hole cut about the pattern axis: "
            f"{np.mean(no_hole[known] == reference[known]):.4f} (with: "
            f"{np.mean(predicted[known] == reference[known]):.4f}; "
            f"{int(np.sum(no_hole != predicted))} labels differ)"
        )

    # Threshold scans against the reference.
    print("\ncirc_thresh scan (balanced accuracy, box vs tube):")
    is_tube_ref = np.where(reference == "box", "box", "tube")
    for thresh in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        pred = predicted_families(results.classify(circ_thresh=thresh))
        is_tube = np.where(pred == "box", "box", "tube")
        score = balanced_accuracy(is_tube_ref[known], is_tube[known], ("box", "tube"))
        print(f"  circ_thresh {thresh:4.2f}: {score:.4f}")
    print("inner_outer_ratio scan (balanced accuracy, reference long-axis tubes):")
    lat = np.isin(
        reference, ("inner_long_axis_tube", "outer_long_axis_tube")
    ) & np.isin(
        cl.shape, [OrbitShape.INNER_LONG_AXIS_TUBE, OrbitShape.OUTER_LONG_AXIS_TUBE]
    )
    # The cut only acts on the frequency ratio, so scan it directly.
    w = np.abs(
        np.asarray(results.select(np.isin(results.ids, cl.ids)).fundamentals, float)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        lat_ratio = w[:, 0] / np.sqrt(w[:, 1] * w[:, 2])
    is_inner = reference == "inner_long_axis_tube"
    print(
        f"  {int(lat.sum())} orbits both call long-axis tubes ({int(np.sum(lat & is_inner))} "
        f"inner); |w_x|/sqrt(|w_y||w_z|) median inner "
        f"{np.nanmedian(lat_ratio[lat & is_inner]):.3f}, outer "
        f"{np.nanmedian(lat_ratio[lat & ~is_inner]):.3f}"
    )
    scores = []
    for ratio in np.round(np.arange(0.74, 0.961, 0.01), 2):
        pred_inner = lat_ratio < ratio
        score = 0.5 * (
            np.mean(pred_inner[lat & is_inner]) + np.mean(~pred_inner[lat & ~is_inner])
        )
        scores.append((score, ratio))
        print(f"  inner_outer_ratio {ratio:4.2f}: {score:.4f}")
    best, cut = max(scores)
    good = [r for sc, r in scores if sc >= best - 0.005]
    print(
        f"  best {best:.4f} at {cut:.2f}; within 0.005 of best for {min(good):.2f}-{max(good):.2f}"
    )


if __name__ == "__main__":
    main()
