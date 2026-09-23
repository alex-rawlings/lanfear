import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import lanfear as lf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(type=str, help="Orbit file", dest="file")
    ap.add_argument("--figdir", type=str, help="figure directory", default="figures")
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="load a previously saved BayesianOrbitClassifier (.npz) instead of "
        "fitting one fresh",
    )
    ap.add_argument(
        "--save-model",
        type=str,
        default=None,
        dest="save_model",
        help="save the fitted (or reused) model to this path",
    )
    ap.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.8,
        dest="confidence_threshold",
        help="posterior probability below which an orbit counts as ambiguous",
    )
    ap.add_argument(
        "--particle-id",
        type=int,
        default=None,
        dest="particle_id",
        help="particle to plot the posterior bar chart for (default: the most "
        "ambiguous particle in the population)",
    )
    args = ap.parse_args()
    lf.set_verbosity("INFO")
    os.makedirs(args.figdir, exist_ok=True)

    orbits = lf.OrbitResults.load(args.file)

    # Fitting is self-supervised on classify_orbits()'s own labels (see
    # docs/probabilistic_classification.rst), so this works out of the box on
    # any OrbitResults with frequency data -- no separate training set needed.
    # A saved model can be reused across runs instead of refitting each time.
    model = lf.BayesianOrbitClassifier.load(args.model) if args.model else None
    prob = orbits.classify_probabilistic(model=model)
    print(f"Classified {len(prob.ids)} orbits: {prob.counts()}")

    if args.save_model:
        prob.model.save(args.save_model)
        print(f"Saved model to {args.save_model}")

    confident = prob.mask_confident(args.confidence_threshold)
    ambiguous = prob.mask_ambiguous(args.confidence_threshold)
    print(
        f"{int(confident.sum())} confidently classified "
        f"(posterior >= {args.confidence_threshold}), "
        f"{int(ambiguous.sum())} ambiguous, out of {len(prob.ids)}"
    )

    # Distribution of MAP posterior probability across the population -- how
    # much of it sits near a classification threshold vs. deep inside a class.
    valid = np.isfinite(prob.map_probability)
    fig, ax = plt.subplots(1, 2, sharex="all")
    fig.set_figwidth(fig.get_figwidth() * 2)

    ax[0].hist(prob.map_probability[valid], bins=30, range=(0, 1))
    ax[0].set_ylabel("number of orbits")

    # Empirical CDF: the exact step function through the sorted values
    # (F(p) = fraction of orbits with MAP probability <= p), not a binned
    # approximation.
    sorted_prob = np.sort(prob.map_probability[valid])
    ecdf = np.arange(1, len(sorted_prob) + 1) / len(sorted_prob)
    ax[1].step(sorted_prob, ecdf, where="post")
    ax[1].set_ylabel("empirical CDF")
    ax[1].set_ylim(0, 1)

    for axi in ax:
        axi.axvline(args.confidence_threshold, color="k", ls="--", lw=1)
        axi.set_xlabel("posterior probability of the MAP class")
    fig.savefig(os.path.join(args.figdir, "map_probability_hist.png"), dpi=300)
    print("Done confidence histogram")

    # Full posterior for a single particle -- by default the most ambiguous
    # one, to show a case where the extra information actually matters.
    particle_id = args.particle_id
    if particle_id is None:
        particle_id = int(prob.ids[valid][np.argmin(prob.map_probability[valid])])
    ax = prob.plot_probability_bar(id=particle_id)
    ax.figure.savefig(os.path.join(args.figdir, "particle_posterior.png"), dpi=300)
    print(f"Done particle posterior (id={particle_id})")

    # Mean posterior breakdown for the most populous class -- a soft view of
    # how "pure" that MAP-labelled population actually is.
    counts = prob.counts()
    counts.pop("unevaluable", None)
    if counts:
        top_class_name = max(counts, key=counts.get)
        top_class = next(k for k, v in prob.class_names.items() if v == top_class_name)
        ax = prob.plot_probability_bar(cls=top_class)
        ax.figure.savefig(
            os.path.join(args.figdir, "class_mean_posterior.png"), dpi=300
        )
        print(f"Done mean posterior for class {top_class_name!r}")


if __name__ == "__main__":
    main()
