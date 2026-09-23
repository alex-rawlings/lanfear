"""Multi-component potential tests.

Covers MultiComponentPotential: the explicit-component-spec contract (no
default potential type per species), reduction to a single component's own
result when only one species is present, agreement with direct summation of a
two-species (disc + halo) system, per-component goodness-of-fit, pickling of
the composite C++ core, and interoperability with the orbit pipeline.

    python tests/test_multi_component.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lanfear as lf  # noqa: E402
from lanfear.orbits import OrbitResults  # noqa: E402


def exponential_disc_particles(n=20_000, Rd=3.0, z0=0.3, m_total=1e10, seed=7):
    """STAR-species exponential disc, matching tests/test_disc.py's fixture."""
    rng = np.random.default_rng(seed)
    R = rng.gamma(2.0, Rd, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    z = z0 * np.arctanh(rng.uniform(-1, 1, n))
    pos = np.stack([R * np.cos(phi), R * np.sin(phi), z], axis=1)
    return pos, np.full(n, m_total / n)


def hernquist_particles(n=100_000, a=20.0, m_total=1e11, seed=3):
    """DM-species Hernquist sphere, matching tests/test_pipeline.py's fixture."""
    rng = np.random.default_rng(seed)
    su = np.sqrt(rng.uniform(0, 1, n))
    r = a * su / (1.0 - su)
    mu = rng.uniform(-1, 1, n)
    az = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - mu**2)
    pos = np.stack([r * st * np.cos(az), r * st * np.sin(az), r * mu], axis=1)
    return pos, np.full(n, m_total / n)


def disc_plus_halo_system(seed=1):
    """A STAR exponential disc embedded in a DM Hernquist halo, one ParticleSystem."""
    star_pos, star_mass = exponential_disc_particles(seed=seed)
    dm_pos, dm_mass = hernquist_particles(seed=seed + 1)
    pos = np.concatenate([star_pos, dm_pos])
    mass = np.concatenate([star_mass, dm_mass])
    n = len(pos)
    ps = lf.ParticleSystem(
        pos=pos,
        vel=np.zeros((n, 3)),
        mass=mass,
        ids=np.arange(n),
        species=np.concatenate(
            [np.full(len(star_pos), "STAR"), np.full(len(dm_pos), "DM")]
        ),
    )
    ps.prepare(centre="shrinking_sphere")
    return ps


def test_missing_and_extra_component_raise():
    """Every present species needs an explicit spec; no stray specs either."""
    ps = disc_plus_halo_system()

    try:
        lf.MultiComponentPotential.from_particles(
            ps, components={"STAR": lf.disc_component(n_radial=6, n_vert=2)}
        )
        raise AssertionError("missing DM component spec should have raised")
    except ValueError as exc:
        assert "DM" in str(exc)

    try:
        lf.MultiComponentPotential.from_particles(
            ps,
            components={
                "STAR": lf.disc_component(n_radial=6, n_vert=2),
                "DM": lf.scf_component(n_max=6, l_max=4),
                "GAS": lf.scf_component(n_max=6, l_max=4),
            },
        )
        raise AssertionError("spec for absent species should have raised")
    except ValueError as exc:
        assert "GAS" in str(exc)

    try:
        lf.scf_component()
        raise AssertionError("scf_component without n_max/l_max should have raised")
    except TypeError:
        pass
    print("missing/extra component specs: OK")


def test_single_component_matches_standalone():
    """One-species composite must reduce exactly to that component alone.

    With a single species the composite's shared unit system collapses to the
    component's own (coord_scale = phi_weight = acc_weight = 1), so this is an
    exactness check on the C++ CompositePotential weighting, not just a
    goodness-of-fit tolerance.
    """
    star_pos, star_mass = exponential_disc_particles(n=5_000)
    n = len(star_pos)
    ps = lf.ParticleSystem(
        pos=star_pos,
        vel=np.zeros((n, 3)),
        mass=star_mass,
        ids=np.arange(n),
        species=np.full(n, "STAR"),
    )
    ps.prepare(centre="shrinking_sphere")

    disc = lf.DiscPotential.from_particles(ps, n_radial=6, n_vert=2)
    composite = lf.MultiComponentPotential.from_particles(
        ps, components={"STAR": lf.disc_component(n_radial=6, n_vert=2)}
    )

    pts = ps.pos[:200]
    assert np.allclose(composite.potential(pts), disc.potential(pts), rtol=1e-10)
    assert np.allclose(composite.acceleration(pts), disc.acceleration(pts), rtol=1e-10)
    assert abs(composite.scale_radius - disc.scale_radius) / disc.scale_radius < 1e-10
    print("single-component composite == standalone DiscPotential: OK")


def test_two_component_matches_direct_sum():
    """A disc+halo composite reproduces direct summation of every particle."""
    ps = disc_plus_halo_system()
    pot = lf.MultiComponentPotential.from_particles(
        ps,
        components={
            "STAR": lf.disc_component(n_radial=8, n_vert=3),
            "DM": lf.scf_component(n_max=10, l_max=6),
        },
    )
    result = pot.validate(n_points=2000)
    print(f"  composite vs direct sum: {result}")
    assert result.passed(tolerance=0.10)

    # Independent spot-check, built from scratch (not from pot's own cached
    # HO-unit arrays): brute-force PHYSICAL potential of every field particle
    # of BOTH species together, converted with pot's own velocity_unit, vs.
    # what pot.potential() (composite of the two independently-fit bases)
    # reports for the same physical points.
    rng = np.random.default_rng(0)
    field = ps.field
    probe = field.pos[rng.choice(field.n_particles, size=50, replace=False)]
    softening_phys = 0.02 * pot.scale_radius
    d = probe[:, None, :] - field.pos[None, :, :]
    r = np.sqrt(np.sum(d**2, axis=2) + softening_phys**2)
    direct_phys = -pot.G * np.sum(field.mass[None, :] / r, axis=1)
    direct_ho = direct_phys / pot.velocity_unit**2
    fit_ho = pot.potential(probe)
    rel = np.median(np.abs(fit_ho - direct_ho) / np.abs(direct_ho))
    print(f"  composite vs brute-force (independent check): median rel err={rel:.2%}")
    assert rel < 0.10


def test_validate_component():
    """validate_component() checks each species' own fit in isolation."""
    ps = disc_plus_halo_system()
    pot = lf.MultiComponentPotential.from_particles(
        ps,
        components={
            "STAR": lf.disc_component(n_radial=8, n_vert=3),
            "DM": lf.scf_component(n_max=10, l_max=6),
        },
    )
    star_result = pot.validate_component("STAR")
    dm_result = pot.validate_component("DM")
    print(f"  STAR component: {star_result}")
    print(f"  DM component: {dm_result}")
    assert star_result.passed(tolerance=0.10)
    assert dm_result.passed(tolerance=0.10)

    try:
        pot.validate_component("GAS")
        raise AssertionError("unknown component label should have raised")
    except KeyError:
        pass
    print("validate_component: OK")


def test_pickle_and_n_max_l_max():
    """Pickle round-trip of the composite core; n_max/l_max are None (not a bug)."""
    import pickle

    ps = disc_plus_halo_system()
    pot = lf.MultiComponentPotential.from_particles(
        ps,
        components={
            "STAR": lf.disc_component(n_radial=6, n_vert=2),
            "DM": lf.scf_component(n_max=6, l_max=4),
        },
    )
    assert pot.n_max is None and pot.l_max is None

    core2 = pickle.loads(pickle.dumps(pot.core))
    pts = ps.pos[:20] / pot.scale_radius
    assert np.allclose(
        pot.core.potential_batch(pts), core2.potential_batch(pts), atol=1e-12
    )
    assert np.allclose(
        pot.core.acceleration_batch(pts),
        core2.acceleration_batch(pts),
        atol=1e-12,
    )
    print("pickle round-trip + n_max/l_max: OK")


def test_orbit_pipeline():
    """analyse_family/OrbitResults accept a MultiComponentPotential unchanged."""
    ps = disc_plus_halo_system()
    pot = lf.MultiComponentPotential.from_particles(
        ps,
        components={
            "STAR": lf.disc_component(n_radial=6, n_vert=2),
            "DM": lf.scf_component(n_max=6, l_max=4),
        },
    )
    res = lf.analyse_family(
        pot,
        ps,
        family="STAR",
        n_periods=10,
        n_samples=512,
        comm=None,
        progress=False,
    )
    assert isinstance(res, OrbitResults)
    assert res.n_max is None and res.l_max is None
    assert np.mean(res.ok) > 0.8
    cl = res.classify()
    print(f"  orbit pipeline on MultiComponentPotential: families={cl.counts()}")
    print("orbit pipeline: OK")


if __name__ == "__main__":
    print("== missing/extra component specs ==")
    test_missing_and_extra_component_raise()
    print("== single component == standalone ==")
    test_single_component_matches_standalone()
    print("== two-component vs direct sum ==")
    test_two_component_matches_direct_sum()
    print("== validate_component ==")
    test_validate_component()
    print("== pickle + n_max/l_max ==")
    test_pickle_and_n_max_l_max()
    print("== orbit pipeline ==")
    test_orbit_pipeline()
    print("\nALL MULTI-COMPONENT TESTS PASSED")
