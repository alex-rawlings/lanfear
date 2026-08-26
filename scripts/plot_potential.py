import argparse
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
    axes[0].figure.savefig("potential_slice.png", dpi=300)


if __name__ == "__main__":
    main()
