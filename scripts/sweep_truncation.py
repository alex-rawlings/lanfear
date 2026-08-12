"""Sweep the SCF truncation orders (n_max, l_max) to pick good values.

Building the analytical potential needs a radial order ``n_max`` and an angular
order ``l_max``; too low under-resolves the field, too high wastes time and
amplifies shot noise. This script sweeps a grid of ``(n_max, l_max)``, building
and validating the potential against direct summation at each point, and
recommends the cheapest orders that reach a target accuracy -- feed the
recommendation to ``run_orbits_mpi.py``.

The grid is embarrassingly parallel, so each MPI rank handles a slice of the
``(n_max, l_max)`` combinations (round-robin) and the root gathers the errors.
The (prepared) particles are loaded once on the root and broadcast to every rank.

    # serial
    python scripts/sweep_truncation.py --file snap.hdf5

    # parallel: one rank per (n_max, l_max) combination is ideal
    srun --mpi=pmix -n 16 python scripts/sweep_truncation.py --file snap.hdf5

    # a quick demo on a synthetic triaxial system (no snapshot needed)
    python scripts/sweep_truncation.py --n 100000 --plot sweep.png

Set OMP_NUM_THREADS for per-rank threading (each build/validate uses OpenMP).
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lanfear as lf


def make_dummy_snapshot(path, n, a=3.0, m_total=1e10, flatten=(1.0, 0.85, 0.7), seed=5):
    """Write a mildly triaxial Hernquist snapshot (so l > 0 actually matters)."""
    import h5py

    rng = np.random.default_rng(seed)
    su = np.sqrt(rng.uniform(0, 1, n))
    r = a * su / (1 - su)
    mu = rng.uniform(-1, 1, n)
    az = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - mu**2)
    pos = np.stack([r * st * np.cos(az), r * st * np.sin(az), r * mu], axis=1)
    pos *= np.asarray(flatten)
    vel = rng.normal(0, 50, (n, 3))
    with h5py.File(path, "w") as f:
        f.create_group("Header").attrs["MassTable"] = np.zeros(6)
        g = f.create_group("PartType4")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=vel)
        g.create_dataset("Masses", data=np.full(n, m_total / n))
        g.create_dataset("ParticleIDs", data=np.arange(n, dtype=np.int64))


def subsample(particles, cap, seed=0):
    """Cap the number of field particles used for the sweep (BHs kept).

    Fewer particles make each build+validate much cheaper; the resulting error
    is a (slightly conservative) upper bound on the full-resolution error, so the
    recommended orders transfer to the full run.
    """
    field_idx = np.where(particles.species != "BH")[0]
    if cap <= 0 or field_idx.size <= cap:
        return particles
    bh_idx = np.where(particles.species == "BH")[0]
    rng = np.random.default_rng(seed)
    keep = rng.choice(field_idx, size=cap, replace=False)
    return particles.select(np.sort(np.concatenate([keep, bh_idx])))


def coeff_count(n_max, l_max):
    """Number of (n, l, m) coefficients (a proxy for build/eval cost)."""
    return (n_max + 1) * (l_max + 1) * (l_max + 2) // 2


def recommend(records, slack):
    """Recommend the cheapest orders that are near-converged.

    The recommendation is the cheapest ``(n_max, l_max)`` whose median error is
    within ``1 + slack`` of the *best* error measured on the grid -- i.e. the
    knee where extra orders stop helping. An absolute tolerance is deliberately
    not used to drive the choice: the median potential error is dominated by the
    monopole, so a loose tolerance would happily accept ``l_max = 0`` and throw
    away all the angular (flattening) structure that matters for orbits.

    Parameters
    ----------
    records : list of dict
        Each with keys ``n``, ``l``, ``median``.
    slack : float
        Fractional tolerance around the best grid error (e.g. 0.5 = within 50%).

    Returns
    -------
    rec : dict
        The recommended record.
    best_error : float
        The lowest median error found on the grid.
    """
    best_error = min(r["median"] for r in records)
    threshold = best_error * (1.0 + slack)
    candidates = [r for r in records if r["median"] <= threshold]
    rec = min(candidates, key=lambda r: (coeff_count(r["n"], r["l"]), r["median"]))
    return rec, best_error


def plot_sweep(grid, n_values, l_values, best, path):
    """Save a heatmap of the median error over the (n_max, l_max) grid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.1 * len(l_values) + 3, 0.9 * len(n_values) + 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        im = ax.imshow(np.log10(grid), origin="lower", aspect="auto", cmap="Reds")
    ax.set_xticks(range(len(l_values)))
    ax.set_xticklabels(l_values)
    ax.set_yticks(range(len(n_values)))
    ax.set_yticklabels(n_values)
    ax.set_xlabel(r"$l_{\max}$")
    ax.set_ylabel(r"$n_{\max}$")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\log_{10}$ median relative error")
    for i in range(len(n_values)):
        for j in range(len(l_values)):
            if np.isfinite(grid[i, j]):
                ax.text(
                    j,
                    i,
                    f"{grid[i, j]:.3%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=("w" if im.norm(np.log10(grid[i, j])) > 0.3 else "k"),
                )
    bi = int(np.where(n_values == best["n"])[0][0])
    bj = int(np.where(l_values == best["l"])[0][0])
    ax.scatter(
        [bj],
        [bi],
        marker="*",
        s=300,
        facecolors="#1187BD",
        edgecolors="k",
        linewidths=1.8,
        label="recommended",
        alpha=0.4,
    )
    ax.legend(loc="upper right")
    ax.set_title("SCF truncation sweep")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  saved heatmap to {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="MPI sweep over (n_max, l_max) to pick SCF truncation orders."
    )
    ap.add_argument("--file", type=str, help="Gadget snapshot (else a dummy is used)")
    ap.add_argument("--n", type=int, default=100_000, help="dummy particle count")
    ap.add_argument(
        "--n-max-values", type=int, default=[4, 8, 12, 16, 18, 20], nargs="*"
    )
    ap.add_argument("--l-max-values", type=int, default=[0, 2, 4, 6, 8, 10], nargs="*")
    ap.add_argument(
        "--subsample",
        type=int,
        default=200_000,
        help="cap field particles used for the sweep (0 = use all)",
    )
    ap.add_argument("--n-shells", type=int, default=16)
    ap.add_argument("--n-directions", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--slack",
        type=float,
        default=0.5,
        help="recommend the cheapest orders within (1+slack) of the best grid "
        "error (the convergence knee); default 0.5",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=0.01,
        help="absolute median-error target used only to annotate the "
        "recommendation (not to choose it)",
    )
    ap.add_argument("--plot", type=str, default=None, help="save a heatmap here")
    args = ap.parse_args()

    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
    except Exception:
        comm, rank, size = None, 0, 1

    n_values = np.sort(args.n_max_values)
    l_values = np.sort(args.l_max_values)
    combos = [(nv, lv) for nv in n_values for lv in l_values]

    # Root loads + prepares + subsamples once, then broadcasts to every rank.
    particles = None
    if rank == 0:
        if args.file is not None:
            path = args.file
        else:
            path = os.path.join(tempfile.mkdtemp(), "snap.hdf5")
            make_dummy_snapshot(path, args.n)
        particles = lf.ParticleSystem.from_gadget_hdf5(path)
        particles.prepare()
        particles = subsample(particles, args.subsample, seed=args.seed)
        print(
            f"[root] sweeping {len(combos)} (n_max, l_max) combos across "
            f"{size} rank(s); {particles.n_particles} particles "
            f"(subsample={args.subsample or 'all'})",
            flush=True,
        )
    if comm is not None:
        particles = comm.bcast(particles, root=0)

    # Each rank builds + validates its slice of the grid (no per-rank printing;
    # results are gathered and shown as a single aggregated table on the root).
    sweep_start = time.perf_counter()
    my_records = []
    for n_max, l_max in combos[rank::size]:
        t0 = time.perf_counter()
        pot = lf.Potential.from_particles(particles, n_max=n_max, l_max=l_max)
        vr = pot.validate(
            n_shells=args.n_shells, n_directions=args.n_directions, seed=args.seed
        )
        my_records.append(
            {
                "n": n_max,
                "l": l_max,
                "median": float(vr.median),
                "seconds": time.perf_counter() - t0,
            }
        )

    gathered = comm.gather(my_records, root=0) if comm is not None else [my_records]
    if rank != 0:
        return
    wall = time.perf_counter() - sweep_start
    records = [r for chunk in gathered for r in chunk]

    # Assemble the error grid (rows n_max, cols l_max) and print it.
    lookup = {(r["n"], r["l"]): r["median"] for r in records}
    grid = np.full((len(n_values), len(l_values)), np.nan)
    for i, nv in enumerate(n_values):
        for j, lv in enumerate(l_values):
            grid[i, j] = lookup.get((nv, lv), np.nan)

    print("\nmedian |Phi_scf - Phi_direct| / |Phi_direct|  (rows n_max, cols l_max)")
    header = "  n\\l " + "".join(f"{lv:>9d}" for lv in l_values)
    print(header)
    print("-" * len(header))
    for i, nv in enumerate(n_values):
        cells = "".join(f"{grid[i, j]:>9.3%}" for j in range(len(l_values)))
        print(f"{nv:>5d} {cells}")
    build_seconds = sum(r["seconds"] for r in records)
    print(
        f"\n{len(records)} combinations in {wall:.1f}s wall "
        f"({build_seconds:.1f}s total build+validate across {size} rank(s))"
    )

    best, best_error = recommend(records, args.slack)
    at_edge = best["n"] == n_values[-1] or best["l"] == l_values[-1]
    print()
    print(
        f"Recommended: n_max={best['n']} l_max={best['l']} "
        f"(median error {best['median']:.3%}, within {args.slack:.0%} of the best "
        f"grid error {best_error:.3%}; {coeff_count(best['n'], best['l'])} coefficients)"
    )
    if best["median"] <= args.tol:
        print(f"  meets the absolute target tol {args.tol:.1%}.")
    else:
        print(
            f"  NOTE: above the absolute target tol {args.tol:.1%}; the field may "
            f"need more particles or a disc basis if it is strongly flattened."
        )
    if at_edge:
        print(
            "  NOTE: the recommendation sits at the edge of the grid, so the error "
            "is still improving -- extend --n-max-values / --l-max-values to be sure."
        )
    print(f"  -> run_orbits_mpi.py --n-max {best['n']} --l-max {best['l']}")

    if args.plot is not None:
        plot_sweep(grid, n_values, l_values, best, args.plot)


if __name__ == "__main__":
    main()
