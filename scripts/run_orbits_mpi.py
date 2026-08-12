"""Example / smoke test for MPI-parallel orbit integration.

Builds a synthetic Hernquist snapshot on the root rank, constructs the SCF
potential, and integrates all star orbits distributed across MPI ranks. Prints a
summary and a checksum; the checksum is independent of the number of ranks, so
running at several rank counts verifies the decomposition is correct.

    # serial
    python scripts/run_orbits_mpi.py --n 20000

    # parallel (openmpi is loaded by load_py313)
    mpirun -n 4 python scripts/run_orbits_mpi.py --n 20000
    srun   -n 4 python scripts/run_orbits_mpi.py --n 20000

Set OMP_NUM_THREADS for per-rank threading (hybrid MPI+OpenMP).
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    ap.add_argument("--periods", type=int, default=20)
    ap.add_argument("--samples", type=int, default=2048)
    ap.add_argument("--n-max", type=int, default=10)
    ap.add_argument("--l-max", type=int, default=4)
    args = ap.parse_args()
    lf.set_verbosity("INFO")

    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
    except Exception:
        rank, size = 0, 1

    potential = particles = None
    if rank == 0:
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
        potential.validate()
        print(
            f"[root] {particles.n_particles} particles, scale_radius={particles.scale_radius:.3f}, "
            f"threads={os.environ.get('OMP_NUM_THREADS', '?')}",
            flush=True,
        )

    t0 = time.perf_counter()
    res = lf.integrate_family(
        potential,
        particles,
        family="STAR",
        n_periods=args.periods,
        n_samples=args.samples,
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
