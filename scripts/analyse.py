"""Classify a saved orbit file and plot the family fractions.

    python scripts/analyse.py orbits.npz --figdir figures

If the results hold rotation-sense references (``run_orbits_mpi.py
--reference``), the co/counter/perpendicular/none fractions relative to each
are printed, by count and per family, and plotted against radius
(``rotation_sense_<name>.png``).
"""

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import lanfear as lf


def int_or_str(x):
    if x == "full":
        return x
    try:
        return int(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Expected an int or 'full', got {x}")


def report_rotation_senses(orbits, r_edges, figdir):
    """Print and plot the rotation senses relative to every reference.

    Parameters
    ----------
    orbits : lanfear.OrbitClassification
        Classified with ``references="all"`` (and possibly condensed; the
        per-family table uses its class labels).
    r_edges : numpy.ndarray
        Radial bin edges for the plot.
    figdir : str
        Figure directory.
    """
    for scheme in orbits.rotation_sense_schemes:
        name = scheme.split(":", 1)[1]
        counts = orbits.counts(scheme)
        total = sum(counts.values())
        summary = ", ".join(
            f"{label} {100 * n / total:.1f}%" for label, n in sorted(counts.items())
        )
        print(f"Rotation sense relative to '{name}': {summary}")
        for family, row in sorted(orbits.crosstab("class", scheme).items()):
            n = sum(row.values())
            parts = ", ".join(f"{k} {100 * v / n:.0f}%" for k, v in sorted(row.items()))
            print(f"    {family:<36s} {n:>8d}: {parts}")
        ax = orbits.plot_class_fractions(r_edges, scheme=scheme)
        ax.set_xscale("log")
        safe = re.sub(r"[^A-Za-z0-9_.+-]+", "_", name)
        ax.figure.savefig(os.path.join(figdir, f"rotation_sense_{safe}.png"), dpi=300)
        plt.close(ax.figure)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(type=str, help="Orbit file", dest="file")
    ap.add_argument("--figdir", type=str, help="figure directory", default="figures")
    ap.add_argument(
        "--r-max",
        type=float,
        default=20,
        help="plot particles out to this radius",
    )
    ap.add_argument(
        "--freq-map", action="store_true", help="plot frequency map", dest="freq_map"
    )
    ap.add_argument(
        "--loop-level", type=int_or_str, default="full", help="condense loop families"
    )
    ap.add_argument(
        "--box-level", type=int_or_str, default="full", help="condense box families"
    )
    ap.add_argument(
        "--pattern-level",
        type=int_or_str,
        default="full",
        help="condense pattern families",
    )
    args = ap.parse_args()
    lf.set_verbosity("INFO")
    os.makedirs(args.figdir, exist_ok=True)

    results = lf.OrbitResults.load(args.file)
    orbits = results.classify(references="all" if results.references else None)
    orbits = orbits.condense(
        loop_level=args.loop_level,
        box_level=args.box_level,
        pattern_level=args.pattern_level,
    )

    # Bin on the snapshot radius, in physical units (pass
    # quantity="radius_orbit_averaged" for the orbit-averaged radius).
    r_edges = np.concatenate(([0], np.geomspace(0.1, args.r_max, 11)))
    ax = orbits.plot_class_fractions(r_edges)
    ax.set_xscale("log")
    ax.figure.savefig(os.path.join(args.figdir, "class_fracs.png"), dpi=300)
    print("Done radial classification")

    ax = orbits.plot_class_histograms()
    ax.figure.savefig(os.path.join(args.figdir, "class_hists.png"), dpi=300)
    print("Done histogram")

    if orbits.rotation_sense_schemes:
        report_rotation_senses(orbits, r_edges, args.figdir)
        print("Done rotation senses")

    if args.freq_map:
        fig, ax = plt.subplots(1, 2, figsize=(6, 4))
        try:
            for axi in ax:
                orbits.plot_frequency_map(ax=axi)
            ax[0].set_xlim(0, 8)
            ax[0].set_ylim(0, 8)
            ax[0].get_legend().remove()
            fig.savefig(os.path.join(args.figdir, "freq_map.png"), dpi=300)
            print("Done frequency map")
        except ValueError as e:
            print(e)


if __name__ == "__main__":
    main()
