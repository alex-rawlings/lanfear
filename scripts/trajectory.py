import argparse
import os
import lanfear as lf

# Shared option parsing with the orbit driver, so a trajectory is rebuilt with
# the same centring and pattern speed as the run it is inspecting.
from run_orbits_mpi import pattern_speed_arg, resolve_centre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(type=str, help="Gadget snapshot file", dest="file")
    ap.add_argument(type=int, help="particle ID", dest="id")
    ap.add_argument("--n-max", type=int, default=18)
    ap.add_argument("--l-max", type=int, default=8)
    ap.add_argument("--figdir", type=str, help="figure directory", default="figures")
    ap.add_argument(
        "--centre",
        type=str,
        help="centre method; 'auto' (default) centres on the black hole(s) if "
        "the snapshot has any, otherwise with the shrinking sphere",
        default="auto",
        choices=["auto", "bh", "shrinking_sphere", "field", "STAR", "DM"],
    )
    ap.add_argument(
        "--pattern-speed",
        type=pattern_speed_arg,
        default="estimate",
        help="figure pattern speed: 'estimate' (default), 'none' (static), a "
        "number (about the aligned short axis) or an 'x,y,z' vector "
        "(velocity/length units). The trajectory is plotted in the frame "
        "co-rotating with the figure.",
    )
    args = ap.parse_args()

    # load particles and build potential
    # this is taken directly from 'run_orbits_mpi.py'
    particles = lf.ParticleSystem.from_gadget_hdf5(args.file)
    particles.prepare(
        centre=resolve_centre(particles, args.centre),
        pattern_speed=args.pattern_speed,
    )
    potential = lf.Potential.from_particles(
        particles, n_max=args.n_max, l_max=args.l_max
    )

    # now calculate trajectory and plot
    traj = lf.ParticleTrajectory.from_particles(potential, particles, args.id)
    ax = traj.plot()
    os.makedirs(args.figdir, exist_ok=True)
    ax[0].figure.savefig(
        os.path.join(args.figdir, f"trajectory_{args.id}.png"), dpi=300
    )


if __name__ == "__main__":
    main()
