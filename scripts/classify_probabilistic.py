"""Label probabilities of saved orbits under their measurement errors.

Classifies perturbed copies of every orbit (see
``docs/probabilistic_classification.rst``), prints the expected family counts
with their measurement-error spread, and plots the distribution of the most
probable family's probability, the family probabilities of one orbit and the
mean family probabilities of the most common family.

    python scripts/classify_probabilistic.py orbits.npz --figdir figures
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import lanfear as lf


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(type=str, help="Orbit file", dest="file")
    ap.add_argument("--figdir", type=str, help="figure directory", default="figures")
    ap.add_argument(
        "--n-draws",
        type=int,
        default=32,
        dest="n_draws",
        help="number of perturbed copies of every orbit",
    )
    ap.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.8,
        dest="confidence_threshold",
        help="probability below which an orbit's family counts as ambiguous",
    )
    ap.add_argument(
        "--particle-id",
        type=int,
        default=None,
        dest="particle_id",
        help="particle to plot the family probabilities of (default: the most "
        "ambiguous one)",
    )
    args = ap.parse_args()
    lf.set_verbosity("INFO")
    os.makedirs(args.figdir, exist_ok=True)

    orbits = lf.OrbitResults.load(args.file)
    prob = orbits.classify_probabilistic(n_draws=args.n_draws)
    counts = prob.counts("family")
    sigma = prob.counts_uncertainty("family")
    print(f"Expected family counts of {len(prob.ids)} orbits (measurement error):")
    for name in sorted(counts, key=counts.get, reverse=True):
        print(f"  {name:>28s} {counts[name]:10.1f} +- {sigma[name]:.1f}")

    confident = prob.mask_confident(args.confidence_threshold, scheme="family")
    print(
        f"{int(confident.sum())} orbits keep their family with probability >= "
        f"{args.confidence_threshold}, {int((~confident).sum())} are ambiguous"
    )

    # How much of the population sits within measurement error of a threshold.
    p = prob.map_probability("family")
    fig, ax = plt.subplots(1, 2, sharex="all")
    fig.set_figwidth(fig.get_figwidth() * 2)
    ax[0].hist(p, bins=np.linspace(0, 1, prob.n_draws + 2) - 0.5 / prob.n_draws)
    ax[0].set_ylabel("number of orbits")
    ax[0].set_yscale("log")
    sorted_p = np.sort(p)
    ax[1].step(sorted_p, np.arange(1, len(p) + 1) / len(p), where="post")
    ax[1].set_ylabel("empirical CDF")
    ax[1].set_ylim(0, 1)
    for axi in ax:
        axi.axvline(args.confidence_threshold, color="k", ls="--", lw=1)
        axi.set_xlabel("probability of the most probable family")
    fig.savefig(os.path.join(args.figdir, "map_probability_hist.png"), dpi=300)
    print("Done probability histogram")

    particle_id = args.particle_id
    if particle_id is None:
        particle_id = int(prob.ids[np.argmin(p)])
    ax = prob.plot_probability_bar(id=particle_id, scheme="family")
    ax.figure.savefig(os.path.join(args.figdir, "particle_probabilities.png"), dpi=300)
    print(f"Done particle probabilities (id={particle_id})")

    top = max(counts, key=counts.get)
    ax = prob.plot_probability_bar(label=top, scheme="family")
    ax.figure.savefig(
        os.path.join(args.figdir, "family_mean_probabilities.png"), dpi=300
    )
    print(f"Done mean probabilities for family {top!r}")


if __name__ == "__main__":
    main()
