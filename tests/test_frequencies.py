"""Frequency-extraction tests (Milestone 3).

Checks that NAFF recovers known orbital frequencies and that the full
analyse_family pipeline (with MPI parity, if launched under mpirun/srun)
produces the expected fundamental-frequency structure.

    python tests/test_frequencies.py            # serial
    mpirun -n 4 python tests/test_frequencies.py  # parallel
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lanfear as lf  # noqa: E402
from lanfear import _core  # noqa: E402


def build_hernquist_scf(n=200_000, n_max=12, l_max=0, seed=3):
    rng = np.random.default_rng(seed)
    su = np.sqrt(rng.uniform(0, 1, n))
    r = su / (1 - su)
    mu = rng.uniform(-1, 1, n)
    az = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - mu**2)
    pos = np.stack([r * st * np.cos(az), r * st * np.sin(az), r * mu], axis=1)
    return _core.SCFPotential(n_max, l_max, pos, np.full(n, 1.0 / n))


def test_naff_physics():
    """Fundamental frequencies of known orbits in a spherical potential."""
    scf = build_hernquist_scf(l_max=0)

    # Circular orbit: single frequency w0 on x and y, silent z.
    r0 = 1.0
    vc = np.sqrt(-scf.acceleration(r0, 0, 0)[0] * r0)
    summ, fund, lines, diff = scf.analyse_orbit(
        np.array([r0, 0, 0, 0, vc, 0]), n_periods=40, n_samples=8192, n_lines=4
    )
    d = dict(zip(_core.summary_columns(), summ))
    w0 = 2 * np.pi / d["period"]
    assert abs(abs(fund[0]) - w0) / w0 < 0.05
    assert abs(abs(fund[1]) - w0) / w0 < 0.05
    assert lines[2, 0, 1] < 1e-6  # z amplitude ~ 0
    # prograde loop: x and y fundamentals share a sign (same circulation sense)
    assert np.sign(fund[0]) == np.sign(fund[1])
    print(
        f"  circular: |w_x|={abs(fund[0]):.4f} |w_y|={abs(fund[1]):.4f} "
        f"w0={w0:.4f} (z silent)"
    )

    # Eccentric planar loop: radial and azimuthal frequencies differ, so x picks
    # up more than one strong line; still planar (z silent).
    summ, fund, lines, diff = scf.analyse_orbit(
        np.array([1.0, 0, 0, 0, 0.6 * vc, 0]), n_periods=40, n_samples=8192, n_lines=4
    )
    d = dict(zip(_core.summary_columns(), summ))
    assert d["status"] == 0
    assert lines[2, 0, 1] < 1e-6  # still planar
    # two distinct strong lines on x (radial vs azimuthal), well resolved
    ax_x = lines[0]
    strong = ax_x[ax_x[:, 1] > 0.05 * ax_x[0, 1]]
    assert len(strong) >= 2
    print(
        f"  eccentric: {len(strong)} strong x-lines, " f"leading |w|={abs(fund[0]):.4f}"
    )
    print("NAFF physics checks passed")


def test_body_period_lengthening():
    """Rotating figures integrate long enough to span ~n_periods body periods.

    A circular orbit at r = 1 of a Hernquist sphere has Omega_c = 0.5, so with
    a pattern speed of 0.4 its body-frame period 2 pi / |0.5 - 0.4| is 5 times
    its inertial one: the window is lengthened by the nearest power of two, 4,
    with the samples scaled alike, and the slow body-frame frequency 0.1 is
    then recovered. At 0.47 the ratio is ~17, beyond the cap of 8.
    """
    from lanfear.orbits import SUMMARY_COLUMNS

    scf = build_hernquist_scf(l_max=0)
    col = {name: SUMMARY_COLUMNS.index(name) for name in SUMMARY_COLUMNS}
    r0 = 1.0
    vc = np.sqrt(-scf.acceleration(r0, 0, 0)[0] * r0)
    state = [r0, 0.0, 0.0, 0.0, vc, 0.0]
    n_periods, n_samples = 20, 1024

    def run(omega, **kwargs):
        summ, traj = scf.integrate_orbit(
            state,
            n_periods=n_periods,
            n_samples=n_samples,
            return_trajectory=True,
            pattern_speed=(0.0, 0.0, omega),
            **kwargs,
        )
        period = summ[col["period"]]
        factor = summ[col["t_total"]] / (n_periods * period)
        return summ, len(traj), factor

    summ, rows, factor = run(0.0)
    assert summ[col["body_period"]] == summ[col["period"]]
    assert np.isclose(factor, 1.0) and rows == n_samples

    summ, rows, factor = run(0.4)
    period, body_period = summ[col["period"]], summ[col["body_period"]]
    omega_c = 2 * np.pi / period
    assert np.isclose(body_period, 2 * np.pi / abs(omega_c - 0.4))
    assert np.isclose(factor, 4.0) and rows == 4 * n_samples, rows
    assert summ[col["t_total"]] / body_period >= n_periods / np.sqrt(2)

    _, rows, factor = run(0.4, max_body_period_factor=1)  # disabled
    assert np.isclose(factor, 1.0) and rows == n_samples

    summ, _, factor = run(0.47)  # body/inertial ~ 17 > cap of 8
    assert np.isclose(factor, 8.0)
    assert summ[col["t_total"]] / summ[col["body_period"]] < n_periods / np.sqrt(2)

    # The slow body-frame frequency |Omega_c - Omega_p| is resolved.
    _, fund, _, _ = scf.analyse_orbit(
        state, n_periods=n_periods, n_samples=n_samples, pattern_speed=(0.0, 0.0, 0.4)
    )
    assert abs(abs(fund[0]) / abs(omega_c - 0.4) - 1) < 0.01, fund
    print(
        f"body-period lengthening OK: factor {factor:g}, body-frame frequency "
        f"{abs(fund[0]):.4f} (expected {abs(omega_c - 0.4):.4f})"
    )


def test_frequency_diffusion_regular():
    """Regular orbits in a spherical potential barely diffuse in frequency."""
    scf = build_hernquist_scf(l_max=0)
    vc = np.sqrt(-scf.acceleration(1.0, 0, 0)[0])
    for state in (
        [1.0, 0, 0, 0, vc, 0],
        [1.0, 0, 0, 0, 0.6 * vc, 0],
        [1.0, 0, 0.3, 0, 0.5 * vc, 0.2],
    ):
        _, _, _, diff = scf.analyse_orbit(
            np.array(state, float), n_periods=40, n_samples=8192, n_lines=4
        )
        assert diff.shape == (3,)
        assert np.all(diff < 1e-4), diff
    print("frequency diffusion (regular) OK")


def make_snapshot(path, n=6000, a=3.0, m_total=1e10, seed=5):
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


def test_pipeline():
    """analyse_family pipeline + MPI parity (if launched under MPI)."""
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
        make_snapshot(path, n=250)
        particles = lf.ParticleSystem.from_gadget_hdf5(path)
        particles.prepare(centre="shrinking_sphere")  # no BH particles
        potential = lf.Potential.from_particles(particles, n_max=10, l_max=2)

    res = lf.analyse_family(
        potential,
        particles,
        family="STAR",
        n_periods=20,
        n_samples=2048,
        n_lines=4,
        comm="auto",
    )

    if rank == 0:
        assert res.fundamentals is not None
        assert res.fundamentals.shape == (len(res.ids), 3)
        assert res.lines.shape == (len(res.ids), 3, 4, 2)
        ok = res.ok
        # For this near-spherical system each orbit is a planar loop, so its
        # three axis fundamentals should be similar: max/min ~ 1.
        tri = np.sort(np.abs(res.fundamentals[ok]), axis=1)
        ratio = tri[:, 2] / np.maximum(tri[:, 0], 1e-9)
        print(
            f"  [{size} rank(s)] {len(res.ids)} orbits, {np.mean(ok):.1%} ok, "
            f"median (max/min fundamental)={np.median(ratio):.3f}"
        )
        assert np.mean(ok) > 0.8
        assert np.median(ratio) < 1.6  # near-spherical -> near 1

        # MPI parity against an explicit serial run.
        res_s = lf.analyse_family(
            potential,
            particles,
            family="STAR",
            n_periods=20,
            n_samples=2048,
            n_lines=4,
            comm=None,
        )
        assert np.allclose(res.fundamentals, res_s.fundamentals, rtol=0, atol=0)
        assert np.allclose(res.lines, res_s.lines, rtol=0, atol=0)
        print("pipeline + MPI-parity checks passed")


if __name__ == "__main__":
    try:
        from mpi4py import MPI

        is_root = MPI.COMM_WORLD.Get_rank() == 0
    except Exception:
        is_root = True
    if is_root:
        print("== NAFF physics ==")
        test_naff_physics()
        test_body_period_lengthening()
        test_frequency_diffusion_regular()
        print("== pipeline ==")
    test_pipeline()
    if is_root:
        print("\nALL FREQUENCY TESTS PASSED")
