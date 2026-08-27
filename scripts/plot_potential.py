import argparse
import os
import lanfear as lf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(type=str, help="Gadget snapshot file", dest="file")
    ap.add_argument("--n", type=int, default=20000, help="number of star particles")
    ap.add_argument("--n-max", type=int, default=10)
    ap.add_argument("--l-max", type=int, default=4)
    ap.add_argument("--centre", type=float, nargs="+", default=[0, 0, 0])
    ap.add_argument("--length", type=float, nargs="+", default=[10, 10])
    ap.add_argument("--plane", type=str, choices=["xy", "xz", "yz"], default="xy")
    ap.add_argument("--figdir", type=str, help="figure directory", default="figures")
    args = ap.parse_args()
    lf.set_verbosity("INFO")

    assert len(args.centre) == 3 and len(args.length) == 2

    particles = lf.ParticleSystem.from_gadget_hdf5(args.file)
    particles.prepare()
    print(f"BH positions:\n{particles.black_holes.pos}")
    potential = lf.Potential.from_particles(
        particles, n_max=args.n_max, l_max=args.l_max
    )
    axes = potential.plot_potential_plane(
        centre=args.centre, box_size=args.length, plane=args.plane
    )
    os.makedirs(args.figdir)
    axes[0].figure.savefig(os.path.join(args.figdir, "potential_slice.png"), dpi=300)

    POT_TOL = 0.01
    pot_result = potential.validate()
    assert pot_result.passed(
        POT_TOL
    ), f"median relerr {pot_result.median:.3%} exceeds tolerance {POT_TOL:.1%}"
    print(f"PASS: median agreement {pot_result.median:.3%} < {POT_TOL:.1%}")


if __name__ == "__main__":
    main()
