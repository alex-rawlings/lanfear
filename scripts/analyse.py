import argparse
import os.path
import numpy as np
import lanfear as lf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(type=str, help="Orbit file", dest="file")
    ap.add_argument("--figdir", type=str, help="figure directory", default="./")
    ap.add_argument(
        "--freq-map", action="store_true", help="plot frequency map", dest="freq_map"
    )
    args = ap.parse_args()
    lf.set_verbosity("INFO")

    orbits = lf.OrbitResults.load(args.file).classify()

    # Bins are in physical units. By default plot_class_fractions bins on the
    # instantaneous snapshot radius; pass radius=orbits.radius_orbit_averaged to
    # bin on the orbit-averaged radius instead.
    ax = orbits.plot_class_fractions(np.geomspace(0.1, 20, 11))
    ax.set_xscale("log")
    ax.figure.savefig(os.path.join(args.figdir, "class_fracs.png"), dpi=300)
    print("Done radial classification")

    ax = orbits.plot_class_histograms()
    ax.figure.savefig(os.path.join(args.figdir, "class_hists.png"), dpi=300)
    print("Done histogram")

    if args.freq_map:
        try:
            ax = orbits.plot_frequency_map()
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.figure.savefig(os.path.join(args.figdir, "freq_map.png"), dpi=300)
            print("Done frequency map")
        except ValueError as e:
            print(e)


if __name__ == "__main__":
    main()
