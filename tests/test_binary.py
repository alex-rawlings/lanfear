"""SMBH-binary tests: BH kernel consistency, binary representation, and the
binary-interacting orbit flag.

    python tests/test_binary.py
"""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lanfear as lf  # noqa: E402
from lanfear import _core  # noqa: E402

G = lf.Potential.DEFAULT_G  # kpc, 1e10 Msun, km/s


def _hernquist_field(n, a=1.0, m_total=1.0, seed=11):
    """Isotropic Hernquist sphere (positions, velocities, masses)."""
    rng = np.random.default_rng(seed)
    su = np.sqrt(rng.uniform(0, 1, n))
    r = a * su / (1 - su)
    mu = rng.uniform(-1, 1, n)
    az = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - mu**2)
    pos = np.stack([r * st * np.cos(az), r * st * np.sin(az), r * mu], axis=1)
    # Random isotropic velocities at a fraction of the local circular speed, so
    # the population holds both loops and plunging (low-L) orbits.
    v_circ = np.sqrt(G * m_total * r / (r + a) ** 2)
    vdir = rng.normal(size=(n, 3))
    vdir /= np.linalg.norm(vdir, axis=1, keepdims=True)
    vel = vdir * (v_circ * rng.uniform(0.1, 0.9, n))[:, None]
    mass = np.full(n, m_total / n)
    return pos, vel, mass


def _system_with_binary(separation, rel_speed, n=20_000, m_bh=0.005, seed=11):
    """Hernquist field plus an equal-mass BH pair centred on the origin.

    The BHs sit at +-separation/2 along x, moving at +-rel_speed/2 along y (a
    circular relative orbit for rel_speed = sqrt(G M12 / separation)).
    """
    pos, vel, mass = _hernquist_field(n, seed=seed)
    bh_pos = np.array([[0.5 * separation, 0, 0], [-0.5 * separation, 0, 0]])
    bh_vel = np.array([[0, 0.5 * rel_speed, 0], [0, -0.5 * rel_speed, 0]])
    ps = lf.ParticleSystem(
        pos=np.vstack([pos, bh_pos]),
        vel=np.vstack([vel, bh_vel]),
        mass=np.concatenate([mass, [m_bh, m_bh]]),
        ids=np.arange(n + 2),
        species=np.array(["STAR"] * n + ["BH", "BH"]),
        centring="bh",
    )
    ps.estimate_scale_radius()
    return ps


def test_bh_potential_matches_core_kernel():
    """_bh_potential_ho uses the core's spline kernel, even inside the softening."""
    ps = _system_with_binary(separation=0.02, rel_speed=0.0, n=5000)
    # Large softening so the test points sit well inside the kernel.
    softening = 0.5
    with_bh = lf.Potential.from_particles(
        ps, n_max=4, l_max=2, bh_softening=softening, binary_treatment="separate"
    )
    field_only = lf.Potential.from_particles(
        ps.field, n_max=4, l_max=2, bh_softening=softening
    )
    rng = np.random.default_rng(0)
    pts_ho = rng.normal(scale=0.1, size=(200, 3))  # r << softening
    phi_field = with_bh._potential_ho(pts_ho) - with_bh._bh_potential_ho(pts_ho)
    assert np.allclose(phi_field, field_only._potential_ho(pts_ho), rtol=0, atol=1e-12)


def test_binary_properties_circular():
    """A circular equal-mass pair: a = separation, e ~ 0, correct influence radius."""
    d = 0.01
    ps = _system_with_binary(separation=d, rel_speed=np.sqrt(G * 0.01 / d))
    b = lf.binary_properties(ps)
    assert b is not None and b.bound and b.compact
    assert np.isclose(b.semimajor_axis, d, rtol=1e-10)
    assert b.eccentricity < 1e-10
    assert np.isclose(b.total_mass, 0.01)
    assert np.allclose(b.centre_of_mass, 0.0)
    # Hernquist enclosed mass r^2 / (1 + r)^2 = 2 * M12 -> r ~ 0.165 (a = 1).
    x = np.sqrt(0.02)
    assert np.isclose(b.influence_radius, x / (1 - x), rtol=0.1)
    # Not a binary unless there are exactly two BHs.
    assert lf.binary_properties(ps.field) is None


def test_binary_treatment():
    """auto combines a compact binary, keeps a wide pair, and the options force it."""
    d = 0.01
    compact = _system_with_binary(separation=d, rel_speed=np.sqrt(G * 0.01 / d))
    pot = lf.Potential.from_particles(compact, n_max=4, l_max=2)
    assert pot.n_black_holes == 1
    mass_ho, pos_ho, soft_ho = pot._bh_params[0]
    assert np.isclose(mass_ho * pot.field_mass, 0.01)
    assert np.allclose(pos_ho, 0.0)
    # The combined mass is softened on the binary's own scale, h = a.
    assert np.isclose(soft_ho * pot.scale_radius, d)
    # ... and the core uses it: inside h the BH term follows the spline kernel,
    # and outside it is exactly Keplerian.
    field_only = lf.Potential.from_particles(compact.field, n_max=4, l_max=2)
    pts_ho = np.array([[0.0, 0.0, 0.0], [0.3 * soft_ho, 0, 0], [3 * soft_ho, 0, 0]])
    phi_bh = pot._potential_ho(pts_ho) - field_only._potential_ho(pts_ho)
    assert np.allclose(phi_bh, pot._bh_potential_ho(pts_ho), rtol=1e-10)
    assert np.isclose(phi_bh[2], -mass_ho / (3 * soft_ho), rtol=1e-10)
    assert pot.binary is not None and pot.binary.compact

    forced = lf.Potential.from_particles(
        compact, n_max=4, l_max=2, binary_treatment="separate"
    )
    assert forced.n_black_holes == 2
    # Separate masses keep bh_softening (default 1e-3).
    assert all(np.isclose(p[2], 1e-3) for p in forced._bh_params)

    # Wide pair at rest: bound (radial) with a = separation / 2 = 0.5 > r_infl.
    wide = _system_with_binary(separation=1.0, rel_speed=0.0)
    pot = lf.Potential.from_particles(wide, n_max=4, l_max=2)
    assert pot.binary.bound and not pot.binary.compact
    assert np.isclose(pot.binary.semimajor_axis, 0.5)
    assert pot.n_black_holes == 2
    forced = lf.Potential.from_particles(
        wide, n_max=4, l_max=2, binary_treatment="point"
    )
    assert forced.n_black_holes == 1
    assert np.isclose(forced._bh_params[0][2] * forced.scale_radius, 0.5)

    # Unbound pair: never combined by auto.
    unbound = _system_with_binary(separation=0.01, rel_speed=1e4)
    pot = lf.Potential.from_particles(unbound, n_max=4, l_max=2)
    assert not pot.binary.bound and pot.n_black_holes == 2
    # Forced to combine, an unbound pair is softened on its separation.
    forced = lf.Potential.from_particles(
        unbound, n_max=4, l_max=2, binary_treatment="point"
    )
    assert np.isclose(forced._bh_params[0][2] * forced.scale_radius, 0.01)

    # The same logic runs for the multi-component builder.
    mc = lf.MultiComponentPotential.from_particles(
        compact, components={"STAR": lf.scf_component(n_max=4, l_max=2)}
    )
    assert mc.n_black_holes == 1 and mc.binary.compact

    with pytest.raises(ValueError, match="binary_treatment"):
        lf.Potential.from_particles(compact, n_max=4, l_max=2, binary_treatment="x")


def test_r_peri_resolves_plunge():
    """r_peri (min over integrator evaluations) is <= the sampled r_min."""
    rng = np.random.default_rng(3)
    n = 50_000
    su = np.sqrt(rng.uniform(0, 1, n))
    r = su / (1 - su)
    mu = rng.uniform(-1, 1, n)
    az = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - mu**2)
    pos = np.stack([r * st * np.cos(az), r * st * np.sin(az), r * mu], axis=1)
    scf = _core.SCFPotential(8, 0, pos, np.full(n, 1.0 / n))
    scf.add_black_hole(0.01, 0.0, 0.0, 0.0, 1e-4)
    cols = list(_core.summary_columns())
    # Nearly radial orbit, very coarsely sampled: the samples step over the
    # pericentre, the integrator's own evaluations do not.
    state = np.array([2.0, 0.0, 0.0, 0.0, 0.01, 0.0])
    summ, _ = scf.integrate_orbit(state, n_periods=10, n_samples=64)
    d = dict(zip(cols, summ))
    # Reference pericentre from very dense sampling.
    summ_fine, _ = scf.integrate_orbit(state, n_periods=10, n_samples=400_000)
    r_peri_true = dict(zip(cols, summ_fine))["r_min"]
    assert d["status"] == 0
    assert d["r_peri"] <= d["r_min"]
    assert d["r_min"] > 2 * r_peri_true  # coarse samples miss the pericentre
    assert abs(d["r_peri"] - r_peri_true) < 0.1 * r_peri_true, (
        d["r_peri"],
        r_peri_true,
    )


def test_flag_and_drop():
    """Binary-interacting orbits are flagged from r_peri and can be dropped."""
    d = 0.01
    ps = _system_with_binary(separation=d, rel_speed=np.sqrt(G * 0.01 / d), n=5000)
    pot = lf.Potential.from_particles(ps, n_max=6, l_max=2, bh_softening=1e-3)
    sub = ps.select(np.r_[np.arange(80), ps.n_particles - 2, ps.n_particles - 1])
    # Generous factor so a test-sized population has some flagged orbits.
    res = lf.analyse_family(
        pot,
        sub,
        family="STAR",
        n_periods=5,
        n_samples=512,
        comm=None,
        progress=False,
        binary_interaction_factor=20.0,
    )
    assert np.isclose(res.binary_semimajor_axis, d)
    flag = res.binary_interacting
    expected = res.column("r_peri") * res.length_unit < 20.0 * d
    assert np.array_equal(flag, expected)
    assert 0 < flag.sum() < len(flag), flag.sum()

    kept = res.drop_binary_interacting()
    assert len(kept.ids) == len(res.ids) - flag.sum()
    assert np.array_equal(kept.ids, res.ids[~flag])
    assert not kept.binary_interacting.any()
    assert len(res.classify(drop_binary_interacting=True).labels) == len(kept.ids)
    assert len(res.classify().labels) == len(res.ids)
    assert np.array_equal(res.to_dict()["binary_interacting"], flag)

    # Changing the factor after integration re-flags on access.
    res.binary_interaction_factor = 0.0
    assert not res.binary_interacting.any()
    res.binary_interaction_factor = 20.0

    with tempfile.TemporaryDirectory() as tmp:
        loaded = lf.OrbitResults.load(res.save(os.path.join(tmp, "orbits")))
    assert loaded.binary_semimajor_axis == res.binary_semimajor_axis
    assert loaded.binary_interaction_factor == 20.0
    assert np.array_equal(loaded.binary_interacting, flag)

    # Without a binary nothing is flagged, and dropping is a no-op.
    res.binary_semimajor_axis = None
    assert not res.binary_interacting.any()
    assert len(res.drop_binary_interacting().ids) == len(res.ids)


if __name__ == "__main__":
    test_bh_potential_matches_core_kernel()
    test_binary_properties_circular()
    test_binary_treatment()
    test_r_peri_resolves_plunge()
    test_flag_and_drop()
    print("binary checks passed")
