"""Orbit-integration tests (Milestone 2).

Covers the physics (energy conservation, closed/planar orbits, orbit-type
signatures) and the full ParticleSystem -> Potential -> integrate_family
pipeline. Runs serially; if launched under MPI (srun/mpirun -n P) it also
exercises the scatter/gather path and checks the result matches serial.

    python tests/test_orbits.py            # serial
    srun -n 4 python tests/test_orbits.py  # parallel
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lanfear as lf  # noqa: E402
from lanfear import _core  # noqa: E402


def build_hernquist_scf(n=200_000, n_max=12, l_max=4, seed=3):
    rng = np.random.default_rng(seed)
    su = np.sqrt(rng.uniform(0, 1, n))
    r = su / (1 - su)
    mu = rng.uniform(-1, 1, n)
    az = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - mu**2)
    pos = np.stack([r * st * np.cos(az), r * st * np.sin(az), r * mu], axis=1)
    mass = np.full(n, 1.0 / n)
    return _core.SCFPotential(n_max, l_max, pos, mass)


def test_physics():
    """Energy conservation and orbit-type signatures (serial, HO units).

    Geometric invariants (planarity, closed loops) hold exactly only in a
    spherically symmetric potential, so those are checked with an l_max=0
    monopole. Energy conservation is the integrator-correctness invariant and is
    checked in a realistic aspherical (l_max=4) field as well.
    """
    cols = _core.summary_columns()
    scf_sph = build_hernquist_scf(l_max=0)  # exactly spherical
    scf_real = build_hernquist_scf(l_max=4)  # aspherical (Poisson noise)

    def integ(scf, state, **kw):
        summ, _ = scf.integrate_orbit(np.asarray(state, float), **kw)
        return dict(zip(cols, summ))

    # Spherical potential: a planar circular orbit stays planar and circular,
    # with a fixed sign of Lz (a loop / tube orbit).
    r0 = 1.0
    vc = np.sqrt(-scf_sph.acceleration(r0, 0, 0)[0] * r0)
    d = integ(scf_sph, [r0, 0, 0, 0, vc, 0], n_periods=50, n_samples=8192)
    assert d["status"] == 0
    assert d["energy_drift"] < 1e-7, d["energy_drift"]
    assert d["z_abs_max"] < 1e-6, d["z_abs_max"]  # planar
    assert d["Lz_sign_changes"] == 0  # loop/tube
    assert (d["r_max"] - d["r_min"]) < 1e-2  # ~circular
    print(
        f"  spherical circular: edrift={d['energy_drift']:.1e} "
        f"dr={d['r_max']-d['r_min']:.1e} zmax={d['z_abs_max']:.1e}"
    )

    # Spherical potential: a radial (plunging) orbit passes through the centre,
    # so every angular-momentum component stays ~0.
    d = integ(scf_sph, [2.0, 0, 0, 0, 0, 0], n_periods=30, n_samples=8192)
    assert d["status"] == 0
    # Radial plunge repeatedly crosses r~0, the stiffest region, so its drift
    # tolerance is looser than a smooth loop orbit's.
    assert d["energy_drift"] < 1e-4, d["energy_drift"]
    assert d["r_min"] < 0.05, d["r_min"]
    assert max(d["Lx_abs_mean"], d["Ly_abs_mean"], d["Lz_abs_mean"]) < 1e-6
    print(
        f"  spherical radial  : edrift={d['energy_drift']:.1e} "
        f"r_min={d['r_min']:.2e}"
    )

    # Aspherical (realistic) potential: energy is still conserved to the
    # integrator tolerance even though the orbit is fully 3D.
    d = integ(scf_real, [1.5, 0.0, 0.3, 0.0, 0.6, 0.4], n_periods=50, n_samples=8192)
    assert d["status"] == 0
    assert d["energy_drift"] < 1e-6, d["energy_drift"]
    print(
        f"  aspherical 3D     : edrift={d['energy_drift']:.1e} "
        f"zmax={d['z_abs_max']:.2f}"
    )
    print("physics checks passed")


def make_snapshot(path, n=6000, a=3.0, m_total=1e10, seed=5):
    import h5py

    rng = np.random.default_rng(seed)
    su = np.sqrt(rng.uniform(0, 1, n))
    r = a * su / (1 - su)
    mu = rng.uniform(-1, 1, n)
    az = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - mu**2)
    pos = np.stack([r * st * np.cos(az), r * st * np.sin(az), r * mu], axis=1)
    # Circular-ish velocities so periods are well defined (km/s, G in gadget units).
    G = lf.Potential.DEFAULT_G
    Menc = m_total * (r / (r + a)) ** 2
    vc = np.sqrt(G * Menc / np.maximum(r, 1e-6))
    # tangential direction in the x-y plane
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    vel = np.zeros((n, 3))
    vel[:, 0] = -vc * np.sin(phi)
    vel[:, 1] = vc * np.cos(phi)
    mass = np.full(n, m_total / n)

    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        h.attrs["MassTable"] = np.zeros(6)
        g = f.create_group("PartType4")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=vel)
        g.create_dataset("Masses", data=mass)
        g.create_dataset("ParticleIDs", data=np.arange(n, dtype=np.int64))


def test_pipeline():
    """Full pipeline + MPI parity (if launched under MPI)."""
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
    except Exception:
        comm, rank, size = None, 0, 1

    potential = particles = None
    if rank == 0:
        d = tempfile.mkdtemp()
        path = os.path.join(d, "snap.hdf5")
        make_snapshot(path)
        particles = lf.ParticleSystem.from_gadget_hdf5(path)
        particles.prepare()
        potential = lf.Potential.from_particles(particles, n_max=10, l_max=2)

    res = lf.integrate_family(
        potential, particles, family="STAR", n_periods=15, n_samples=1024, comm="auto"
    )

    if rank == 0:
        frac_ok = np.mean(res.ok)
        med_drift = np.median(res.column("energy_drift")[res.ok])
        print(
            f"  [{size} rank(s)] integrated {len(res.ids)} orbits, "
            f"{frac_ok:.1%} ok, median energy_drift={med_drift:.2e}"
        )
        # Most orbits integrate cleanly; a minority of synthetic large-radius
        # particles fail the period estimate (status flag flags them). The real
        # correctness gate is energy conservation of the successful orbits.
        assert frac_ok > 0.8
        assert med_drift < 1e-4

        # Compare against an explicit serial run (comm=None) for parity.
        res_serial = lf.integrate_family(
            potential, particles, family="STAR", n_periods=15, n_samples=1024, comm=None
        )
        assert np.allclose(
            res.summary, res_serial.summary, atol=0, rtol=0
        ), "MPI result differs from serial"
        print("pipeline + MPI-parity checks passed")


if __name__ == "__main__":
    try:
        from mpi4py import MPI

        is_root = MPI.COMM_WORLD.Get_rank() == 0
    except Exception:
        is_root = True
    if is_root:
        print("== physics ==")
        test_physics()
        print("== pipeline ==")
    else:
        # non-root ranks skip the serial-only physics test
        pass
    test_pipeline()
    if is_root:
        print("\nALL ORBIT TESTS PASSED")
