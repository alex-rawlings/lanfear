import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lanfear as lf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(type=str, help="Orbit file", dest="file")
    args = ap.parse_args()
    lf.set_verbosity("INFO")

    orbits = lf.OrbitResults.load(args.file).classify()
    ax = orbits.plot_class_fractions(np.geomspace(0.1, 20, 11))
    ax.set_xscale("log")
    ax.figure.savefig("class_fracs.png", dpi=300)
    print("Done radial classification")

    ax = orbits.plot_class_histograms()
    ax.figure.savefig("class_hists.png", dpi=300)
    print("Done histogram")

    try:
        ax = orbits.plot_frequency_map()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.figure.savefig("freq_map.png", dpi=300)
        print("Done frequency map")
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()
