"""Example / smoke test for MPI-parallel orbit integration.

Builds a synthetic Hernquist snapshot on the root rank, constructs the SCF
potential, and integrates + frequency-analyses all star orbits distributed
across MPI ranks. Prints a summary and a checksum; the checksum is independent of
the number of ranks, so running at several rank counts verifies the
decomposition is correct. The saved output carries the frequency data
(fundamentals + spectral lines), so a reloaded result supports the full
classification (rosette/boxlet/irregular) and the frequency map.

    # serial
    python scripts/run_orbits_mpi.py --n 20000

    # parallel (openmpi is loaded by load_py313)
    mpirun -n 4 python scripts/run_orbits_mpi.py --n 20000
    srun   -n 4 python scripts/run_orbits_mpi.py --n 20000

Set OMP_NUM_THREADS for per-rank threading (hybrid MPI+OpenMP).
"""

import argparse
import os
import tempfile
import time
import numpy as np
import lanfear as lf


def make_dummy_snapshot(path, n, a=3.0, m_total=1e10, seed=5):
    import h5py

    rng = np.random.default_rng(seed)
    su = np.sqrt(rng.uniform(0, 1, n))
    r = a * su / (1 - su)
    mu = rng.uniform(-1, 1, n)
    az = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - mu**2)
    pos = np.stack([r * st * np.cos(az), r * st * np.sin(az), r * mu], axis=1)
    Menc = m_total * (r / (r + a)) ** 2
    vc = np.sqrt(lf.Potential.DEFAULT_G * Menc / np.maximum(r, 1e-6))
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    vel = np.stack([-vc * np.sin(phi), vc * np.cos(phi), np.zeros(n)], axis=1)
    with h5py.File(path, "w") as f:
        f.create_group("Header").attrs["MassTable"] = np.zeros(6)
        g = f.create_group("PartType4")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=vel)
        g.create_dataset("Masses", data=np.full(n, m_total / n))
        g.create_dataset("ParticleIDs", data=np.arange(n, dtype=np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, help="Gadget snapshot file")
    ap.add_argument("--n", type=int, default=20000, help="number of star particles")
    ap.add_argument("--periods", type=int, default=50)
    ap.add_argument("--samples", type=int, default=2048)
    ap.add_argument("--n-lines", type=int, default=4, help="spectral lines per axis")
    ap.add_argument("--n-max", type=int, default=16)
    ap.add_argument("--l-max", type=int, default=4)
    ap.add_argument(
        "--r-max",
        type=float,
        default=None,
        help="only integrate particles within this radius (HO/physical units of "
        "the recentred system); the potential is still built from all particles",
    )
    ap.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="integrate a random subset of this many particles (default: all); "
        "the potential is still built from all particles",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for the --subsample draw (kept fixed so the checksum stays "
        "independent of the rank count)",
    )
    args = ap.parse_args()
    lf.set_verbosity("INFO")
    POT_TOL = 0.001  # 0.1% potential target agreement (median)

    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
    except Exception:
        rank, size = 0, 1

    potential = particles = to_integrate = None
    if rank == 0:
        lf.print_package_info()
        d = tempfile.mkdtemp()
        outfile = "lanfear_orbits/orbits.npz"
        if args.file is not None:
            path = args.file
            _dname, _ext = os.path.splitext(outfile)
            outfile = os.path.join(
                os.path.dirname(path),
                f"{_dname}_{os.path.basename(path).replace('.hdf5', _ext)}",
            )
        else:
            path = os.path.join(d, "snap.hdf5")
            make_dummy_snapshot(path, args.n)
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        particles = lf.ParticleSystem.from_gadget_hdf5(path)
        particles.prepare()
        potential = lf.Potential.from_particles(
            particles, n_max=args.n_max, l_max=args.l_max
        )
        pot_result = potential.validate()
        assert pot_result.passed(
            POT_TOL
        ), f"median relerr {pot_result.median:.3%} exceeds tolerance {POT_TOL:.1%}"
        print(f"\nPASS: median agreement {pot_result.median:.3%} < {POT_TOL:.1%}")
        # The potential is built from all particles above; optionally restrict
        # which orbits are integrated to those within --r-max (the potential is
        # unchanged). radius_mask composes with select like species_mask.
        to_integrate = particles
        if args.r_max is not None:
            to_integrate = particles.select(particles.radius_mask(args.r_max))
            print(
                f"[root] restricting integration to r < {args.r_max:g}: "
                f"{to_integrate.n_particles} of {particles.n_particles} particles",
                flush=True,
            )
        if args.subsample is not None:
            before = to_integrate.n_particles
            to_integrate = to_integrate.random_subset(
                args.subsample, rng=np.random.default_rng(args.seed)
            )
            print(
                f"[root] subsampling {to_integrate.n_particles} of {before} "
                f"particles for integration (seed={args.seed})",
                flush=True,
            )
        print(
            f"[root] {particles.n_particles} particles, scale_radius={particles.scale_radius:.3f}, "
            f"threads={os.environ.get('OMP_NUM_THREADS', '?')}",
            flush=True,
        )

    t0 = time.perf_counter()
    res = lf.analyse_family(
        potential,
        to_integrate,
        family="STAR",
        n_periods=args.periods,
        n_samples=args.samples,
        n_lines=args.n_lines,
        comm="auto",
    )
    dt = time.perf_counter() - t0

    if rank == 0:
        ok = res.ok
        # Checksum over the robust columns (exclude tiny-noise energy values) so
        # it is bit-reproducible across rank counts.
        checksum = float(np.sum(res.column("r_max")) + np.sum(res.column("period")))
        print(
            f"[{size} rank(s)] {len(res.ids)} orbits in {dt:.2f}s | "
            f"{np.mean(ok):.2%} ok | "
            f"median energy_drift={np.median(res.column('energy_drift')[ok]):.2e} | "
            f"checksum={checksum:.10e}",
            flush=True,
        )

        # save output
        res.save(outfile)


if __name__ == "__main__":
    main()
