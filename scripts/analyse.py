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
        "--r-max",
        type=float,
        default=20,
        help="plot particles out to this radius",
    )
    ap.add_argument(
        "--freq-map", action="store_true", help="plot frequency map", dest="freq_map"
    )
    args = ap.parse_args()
    lf.set_verbosity("INFO")
    os.makedirs(args.figdir, exist_ok=True)

    orbits = lf.OrbitResults.load(args.file).classify()

    # Bins are in physical units. By default plot_class_fractions bins on the
    # instantaneous snapshot radius; pass quantity="radius_orbit_averaged" to
    # bin on the orbit-averaged radius instead.
    r_edges = np.concatenate(([0], np.geomspace(0.1, args.r_max, 11)))
    ax = orbits.plot_class_fractions(r_edges)
    ax.set_xscale("log")
    ax.figure.savefig(os.path.join(args.figdir, "class_fracs.png"), dpi=300)
    print("Done radial classification")

    ax = orbits.plot_class_histograms()
    ax.figure.savefig(os.path.join(args.figdir, "class_hists.png"), dpi=300)
    print("Done histogram")

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
