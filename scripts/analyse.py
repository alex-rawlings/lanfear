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
    ax = orbits.plot_class_fractions(np.geomspace(1e-2, 20, 11))
    ax.set_xscale("log")
    ax.figure.savefig("class_fracs.png", dpi=300)

    ax = orbits.plot_class_histograms()
    ax.figure.savefig("class_hists.png", dpi=300)


if __name__ == "__main__":
    main()
