import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lanfear as lf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(type=str, help="Gadget snapshot file", dest="file")
    ap.add_argument(type=int, help="particle ID", dest="id")
    ap.add_argument("--n-max", type=int, default=10)
    ap.add_argument("--l-max", type=int, default=4)
    args = ap.parse_args()

    # load particles and build potential
    # this is taken directly from 'run_orbits_mpi.py'
    particles = lf.ParticleSystem.from_gadget_hdf5(args.file)
    particles.prepare()
    potential = lf.Potential.from_particles(
        particles, n_max=args.n_max, l_max=args.l_max
    )

    # now calculate trajectory and plot
    traj = lf.ParticleTrajectory.from_particles(potential, particles, args.id)
    ax = traj.plot()
    ax[0].figure.savefig(f"trajectory_{args.id}.png", dpi=300)


if __name__ == "__main__":
    main()
