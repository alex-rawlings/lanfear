"""Figure-rotation (pattern-speed) tests.

Covers the single-snapshot pattern-speed estimator
(``ParticleSystem.estimate_pattern_speed``), the ``pattern_speed`` option of
``prepare()`` and its inheritance by the potentials, and orbit integration in a
rigidly rotating potential: orbits are integrated in the inertial frame and
recorded in the co-rotating frame, where the Jacobi integral is conserved.

    python tests/test_pattern_speed.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lanfear as lf  # noqa: E402
from lanfear import _core  # noqa: E402


def _hernquist_positions(n, rng):
    su = np.sqrt(rng.uniform(0, 1, n))
    r = su / (1 - su)
    mu = rng.uniform(-1, 1, n)
    az = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - mu**2)
    return np.stack([r * st * np.cos(az), r * st * np.sin(az), r * mu], axis=1)


def _tumbling_system(
    omega, squash=(1.0, 0.7, 0.5), n=60000, sigma=100.0, seed=0, scale=3.0
):
    """Squashed Hernquist figure whose density rotates rigidly at `omega`.

    Isotropic random velocities leave the density stationary on average, so
    adding the solid-body field ``omega x x`` makes the figure (not just the
    stars) rotate at exactly ``omega``.
    """
    rng = np.random.default_rng(seed)
    pos = _hernquist_positions(n, rng) * np.asarray(squash) * scale
    vel = rng.normal(0.0, sigma, (n, 3)) + np.cross(np.asarray(omega, float), pos)
    return lf.ParticleSystem(
        pos=pos,
        vel=vel,
        mass=np.full(n, 1e10 / n),
        ids=np.arange(n),
        species=np.full(n, "STAR"),
    )


def _rotation_matrix(axis, angle):
    """Rodrigues matrix for a rotation by `angle` about unit `axis`."""
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def test_estimator_recovers_rotation():
    """A rigidly tumbling triaxial figure yields its pattern speed."""
    for omega_true in ([0.0, 0.0, 40.0], [0.0, 0.0, -25.0], [15.0, 0.0, 30.0]):
        omega_true = np.asarray(omega_true)
        ps = _tumbling_system(omega_true)
        est = ps.estimate_pattern_speed()
        omega = est["pattern_speed"]
        # Compare each significant principal-frame component with its error.
        axes = est["principal_axes"]
        expected_p = axes.T @ omega_true
        err = np.abs(est["principal_pattern_speed"] - expected_p)
        assert np.all(err < 5 * est["uncertainty"]), (omega_true, est)
        assert np.linalg.norm(omega - omega_true) < 0.2 * np.linalg.norm(omega_true), (
            omega_true,
            omega,
        )
        print(
            f"  tumbling {omega_true}: estimate {np.round(omega, 2)} "
            f"+/- {np.round(est['uncertainty'], 2)}"
        )
    print("estimator recovery checks passed")


def test_estimator_static_figures():
    """Non-tumbling figures get a zero pattern speed."""
    cases = [
        ("triaxial, non-rotating", (1.0, 0.7, 0.5), 0.0),
        ("spherical + streaming", (1.0, 1.0, 1.0), 60.0),
        ("oblate + streaming", (1.0, 1.0, 0.5), 60.0),
    ]
    for seed, (name, squash, v_stream) in enumerate(cases):
        ps = _tumbling_system([0.0, 0.0, 0.0], squash=squash, seed=10 + seed)
        if v_stream:
            # Ordered streaming about z: the density of an axisymmetric figure
            # is unchanged by it, so it is not figure rotation.
            R = np.hypot(ps.pos[:, 0], ps.pos[:, 1]) + 1e-9
            ps.vel[:, 0] += -v_stream * ps.pos[:, 1] / R
            ps.vel[:, 1] += v_stream * ps.pos[:, 0] / R
        est = ps.estimate_pattern_speed()
        assert not np.any(est["pattern_speed"]), (name, est)
        print(f"  {name}: pattern speed {est['pattern_speed']} (static)")
    print("static-figure checks passed")


def test_prepare_options():
    """prepare() estimates by default; 'none', numbers and vectors also work."""
    omega_true = np.array([0.0, 0.0, 40.0])
    ps = _tumbling_system(omega_true, seed=1)
    ps.prepare(centre="shrinking_sphere")
    assert abs(abs(ps.pattern_speed[2]) - 40.0) < 8.0, ps.pattern_speed

    ps = _tumbling_system(omega_true, seed=1)
    ps.prepare(centre="shrinking_sphere", pattern_speed="none")
    assert not np.any(ps.pattern_speed)

    ps = _tumbling_system(omega_true, seed=1)
    ps.prepare(centre="shrinking_sphere", pattern_speed=12.5)
    np.testing.assert_array_equal(ps.pattern_speed, [0.0, 0.0, 12.5])

    ps = _tumbling_system(omega_true, seed=1)
    ps.prepare(centre="shrinking_sphere", pattern_speed=[1.0, 2.0, 3.0])
    np.testing.assert_array_equal(ps.pattern_speed, [1.0, 2.0, 3.0])

    for bad in ("sometimes", [1.0, 2.0], np.nan):
        try:
            _tumbling_system(omega_true, seed=1).prepare(
                centre="shrinking_sphere", pattern_speed=bad
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"pattern_speed={bad!r} should raise")

    # Subsets carry the pattern speed; align() rotates it with the system.
    sub = ps.select(ps.radius_mask(5.0))
    np.testing.assert_array_equal(sub.pattern_speed, ps.pattern_speed)
    rot = ps.align(bound_fraction=0.3)
    np.testing.assert_allclose(ps.pattern_speed, rot @ [1.0, 2.0, 3.0])
    print("prepare() pattern-speed option checks passed")


def test_potential_inherits():
    """from_particles copies the pattern speed; it can be reassigned."""
    ps = _tumbling_system([0.0, 0.0, 40.0], n=20000, seed=2)
    ps.prepare(centre="shrinking_sphere", pattern_speed=30.0)
    pot = lf.Potential.from_particles(ps, n_max=6, l_max=2)
    np.testing.assert_array_equal(pot.pattern_speed, [0.0, 0.0, 30.0])
    assert pot.rotating
    np.testing.assert_allclose(pot.pattern_speed_ho, pot.pattern_speed * pot.time_unit)
    pot.pattern_speed = "none"
    assert not pot.rotating and not np.any(pot.pattern_speed_ho)
    pot.pattern_speed = 5.0
    np.testing.assert_array_equal(pot.pattern_speed, [0.0, 0.0, 5.0])

    # Never prepared: a static figure.
    raw = _tumbling_system([0.0, 0.0, 40.0], n=20000, seed=2)
    raw.recentre(on="shrinking_sphere")
    assert not lf.Potential.from_particles(raw, n_max=6, l_max=2).rotating
    print("potential inheritance checks passed")


def _scf(squash=(1.0, 1.0, 1.0), l_max=0, n=100_000, seed=3):
    rng = np.random.default_rng(seed)
    pos = _hernquist_positions(n, rng) * np.asarray(squash)
    return _core.SCFPotential(10, l_max, pos, np.full(n, 1.0 / n))


def test_zero_pattern_speed_is_static():
    """An explicit zero pattern speed reproduces the static integration exactly."""
    scf = _scf(squash=(1.0, 0.8, 0.6), l_max=4)
    state = np.array([1.2, 0.1, 0.3, 0.05, 0.6, 0.2])
    s0, t0 = scf.integrate_orbit(state, 10, 1024, return_trajectory=True)
    s1, t1 = scf.integrate_orbit(
        state, 10, 1024, return_trajectory=True, pattern_speed=(0.0, 0.0, 0.0)
    )
    np.testing.assert_array_equal(s0, s1)
    np.testing.assert_array_equal(t0, t1)
    print("zero-pattern-speed identity check passed")


def test_rotating_spherical_equivalence():
    """Rotating a spherical potential leaves the inertial orbit unchanged.

    The co-rotating trajectory must then be the static (inertial) trajectory
    seen from the rotating frame: x_b = R(t)^T x, v_rot = R(t)^T v - Omega x x_b.
    This checks the frame transforms and their sign conventions.
    """
    scf = _scf(l_max=0)  # exactly spherical
    state = np.array([1.2, 0.0, 0.3, 0.1, 0.7, 0.2])
    omega = np.array([0.1, 0.2, 0.5])
    n_samples = 512
    s_static, x_static = scf.integrate_orbit(
        state, 5, n_samples, return_trajectory=True
    )
    s_rot, x_rot = scf.integrate_orbit(
        state, 5, n_samples, return_trajectory=True, pattern_speed=tuple(omega)
    )
    cols = list(_core.summary_columns())
    t_total = s_rot[cols.index("t_total")]
    times = np.linspace(0.0, t_total, n_samples)
    rate = np.linalg.norm(omega)
    axis = omega / rate
    expected = np.empty_like(x_static)
    for j, t in enumerate(times):
        r_t = _rotation_matrix(axis, rate * t)
        xb = r_t.T @ x_static[j, :3]
        expected[j, :3] = xb
        expected[j, 3:] = r_t.T @ x_static[j, 3:] - np.cross(omega, xb)
    np.testing.assert_allclose(x_rot, expected, atol=1e-6)

    # The (inertial) energy is conserved here too, so E_J = E - Omega . L with
    # L conserved: both columns agree with the static run's up to Omega . L.
    L0 = np.cross(state[:3], state[3:])
    e_static = s_static[cols.index("energy0")]
    np.testing.assert_allclose(s_rot[cols.index("energy0")], e_static - omega @ L0)
    print("rotating spherical-potential equivalence check passed")


def test_jacobi_conservation():
    """In a rotating triaxial potential E_J is conserved but E is not."""
    scf = _scf(squash=(1.0, 0.7, 0.5), l_max=4)
    state = np.array([1.0, 0.2, 0.2, 0.1, 0.5, 0.2])
    omega = np.array([0.0, 0.0, 0.3])
    summ, traj = scf.integrate_orbit(
        state, 20, 2048, return_trajectory=True, pattern_speed=tuple(omega)
    )
    cols = list(_core.summary_columns())
    assert summ[cols.index("status")] == 0
    assert summ[cols.index("energy_drift")] < 1e-6, summ[cols.index("energy_drift")]

    # The inertial energy is not conserved: the rotating figure does work.
    xb, v_rot = traj[:, :3], traj[:, 3:]
    v_in = v_rot + np.cross(omega, xb)
    energy = 0.5 * np.sum(v_in**2, axis=1) + scf.potential_batch(
        np.ascontiguousarray(xb)
    )
    assert np.ptp(energy) / abs(energy[0]) > 1e-3
    print(
        f"  E_J drift {summ[cols.index('energy_drift')]:.1e}, "
        f"E variation {np.ptp(energy) / abs(energy[0]):.1e}"
    )
    print("Jacobi-integral conservation check passed")


def test_pipeline_records_pattern_speed():
    """analyse_family integrates in the rotating potential and records it."""
    ps = _tumbling_system([0.0, 0.0, 40.0], n=20000, seed=4)
    ps.prepare(centre="shrinking_sphere")
    assert ps.pattern_speed is not None and np.any(ps.pattern_speed)
    pot = lf.Potential.from_particles(ps, n_max=6, l_max=2)
    sub = ps.random_subset(20, rng=np.random.default_rng(0))
    res = lf.analyse_family(
        pot, sub, family="STAR", n_periods=10, n_samples=512, comm=None
    )
    np.testing.assert_array_equal(res.pattern_speed, pot.pattern_speed)
    ok = res.ok
    assert ok.any()
    assert np.all(res.column("energy_drift")[ok] < 1e-4)

    with tempfile.TemporaryDirectory() as d:
        back = lf.OrbitResults.load(res.save(os.path.join(d, "orbits.npz")))
    np.testing.assert_array_equal(back.pattern_speed, res.pattern_speed)

    traj = lf.ParticleTrajectory.from_particles(pot, sub, int(sub.ids[0]), n_periods=5)
    np.testing.assert_array_equal(traj.pattern_speed, pot.pattern_speed)
    if traj.status == 0:
        drift = np.ptp(traj.energy) / abs(traj.energy[0])
        assert drift < 1e-5, drift
    print("pipeline pattern-speed checks passed")


if __name__ == "__main__":
    test_estimator_recovers_rotation()
    test_estimator_static_figures()
    test_prepare_options()
    test_potential_inherits()
    test_zero_pattern_speed_is_static()
    test_rotating_spherical_equivalence()
    test_jacobi_conservation()
    test_pipeline_records_pattern_speed()
    print("\nALL PATTERN-SPEED TESTS PASSED")
