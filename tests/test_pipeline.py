"""End-to-end Milestone-1 test.

Generates a synthetic Gadget-4 HDF5 snapshot (a Hernquist sphere of stars plus
an off-centre black hole), then exercises the full python pipeline: load,
prepare, build the SCF potential, and validate against direct summation.

Run with::

    load_py313
    python tests/test_pipeline.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lanfear as lf  # noqa: E402


def make_hernquist_snapshot(path, n=200_000, a=3.0, m_total=1e10, seed=1):
    """Write a Hernquist sphere (PartType4) + one BH (PartType5) to HDF5.

    Units are arbitrary but internally consistent. The Hernquist scale radius
    is ``a`` and the total stellar mass is ``m_total``.
    """
    import h5py

    rng = np.random.default_rng(seed)
    su = np.sqrt(rng.uniform(0, 1, n))
    r = a * su / (1.0 - su)  # invert M(<r) for a Hernquist profile
    mu = rng.uniform(-1, 1, n)
    az = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - mu**2)
    pos = np.stack([r * st * np.cos(az), r * st * np.sin(az), r * mu], axis=1)
    vel = rng.normal(0, 50, (n, 3))  # velocities irrelevant to the potential
    mass = np.full(n, m_total / n)
    ids = np.arange(n, dtype=np.int64)

    # Displace the whole galaxy so recentring has something to do.
    pos += np.array([100.0, -50.0, 25.0])

    # One BH, deliberately off the stellar centre.
    bh_pos = np.array([[100.0 + 0.5 * a, -50.0, 25.0]])
    bh_mass = np.array([0.02 * m_total])

    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["NumPart_ThisFile"] = np.array([0, 0, 0, 0, n, 1])
        g4 = f.create_group("PartType4")
        g4.create_dataset("Coordinates", data=pos)
        g4.create_dataset("Velocities", data=vel)
        g4.create_dataset("Masses", data=mass)
        g4.create_dataset("ParticleIDs", data=ids)
        g5 = f.create_group("PartType5")
        g5.create_dataset("Coordinates", data=bh_pos)
        g5.create_dataset("Velocities", data=np.zeros((1, 3)))
        g5.create_dataset("Masses", data=bh_mass)
        g5.create_dataset("ParticleIDs", data=np.array([n], dtype=np.int64))
    return a, m_total


def test_radius_mask():
    """radius_mask composes with select and drives radius-limited integration."""
    rng = np.random.default_rng(4)
    n = 40_000
    a = 3.0
    su = np.sqrt(rng.uniform(0, 1, n))
    r = a * su / (1.0 - su)
    mu = rng.uniform(-1, 1, n)
    az = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - mu**2)
    pos = np.stack([r * st * np.cos(az), r * st * np.sin(az), r * mu], axis=1)
    menc = 1e10 * (r / (r + a)) ** 2
    vc = np.sqrt(lf.Potential.DEFAULT_G * menc / np.maximum(r, 1e-6))
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    vel = np.stack([-vc * np.sin(phi), vc * np.cos(phi), np.zeros(n)], axis=1)
    ps = lf.ParticleSystem(
        pos=pos,
        vel=vel,
        mass=np.full(n, 1e10 / n),
        ids=np.arange(n),
        species=np.full(n, "STAR"),
    )
    ps.prepare(check_figure_rotation=False)

    # Mask semantics: within r, shell, and composition with species_mask.
    rr = ps.radii()
    r_cut = float(np.median(rr))
    assert np.array_equal(ps.radius_mask(r_cut), rr < r_cut)
    assert np.array_equal(ps.radius_mask(r_cut, r_min=0.5), (rr >= 0.5) & (rr < r_cut))
    combined = ps.species_mask("STAR") & ps.radius_mask(r_cut)
    assert combined.sum() == int(np.sum(rr < r_cut))

    # The potential is built from ALL particles ...
    pot = lf.Potential.from_particles(ps, n_max=8, l_max=2)
    # ... but only the inner subset is integrated.
    inner = ps.select(ps.radius_mask(r_cut))
    assert inner.n_particles == int(np.sum(rr < r_cut))
    res = lf.analyse_family(
        pot, inner, family="STAR", n_periods=3, n_samples=512, progress=False
    )
    assert res is not None and len(res.ids) == inner.n_particles
    # The integrated IDs are exactly the inner particles (a strict subset).
    assert set(res.ids.tolist()) == set(inner.ids.tolist())
    assert len(res.ids) < ps.n_particles
    print(f"radius_mask OK: integrated {len(res.ids)} of {ps.n_particles} particles")


def test_random_subset():
    """random_subset draws without replacement and honours a passed rng."""
    n = 500
    ps = lf.ParticleSystem(
        pos=np.random.rand(n, 3),
        vel=np.random.rand(n, 3),
        mass=np.ones(n),
        ids=np.arange(n),
        species=np.full(n, "STAR"),
    )
    sub = ps.random_subset(100)
    assert isinstance(sub, lf.ParticleSystem)
    assert sub.n_particles == 100
    assert len(np.unique(sub.ids)) == 100  # no replacement
    assert set(sub.ids.tolist()) <= set(ps.ids.tolist())  # a genuine subset

    # A supplied rng makes the draw reproducible; None gives a fresh generator.
    a = ps.random_subset(50, rng=np.random.default_rng(7)).ids
    b = ps.random_subset(50, rng=np.random.default_rng(7)).ids
    assert np.array_equal(a, b)

    # Edge cases: over-sized -> full system, zero -> empty, negative -> error.
    assert ps.random_subset(9999).n_particles == n
    assert ps.random_subset(0).n_particles == 0
    try:
        ps.random_subset(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for negative n")
    print("random_subset OK")


def _make_system(flatten=(1.0, 1.0, 1.0), n=40_000, a=3.0, seed=6):
    """A Hernquist ParticleSystem, optionally squashed by ``flatten``."""
    rng = np.random.default_rng(seed)
    su = np.sqrt(rng.uniform(0, 1, n))
    r = a * su / (1.0 - su)
    mu = rng.uniform(-1, 1, n)
    az = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - mu**2)
    pos = np.stack([r * st * np.cos(az), r * st * np.sin(az), r * mu], axis=1)
    pos *= np.asarray(flatten)
    ps = lf.ParticleSystem(
        pos=pos,
        vel=np.zeros((n, 3)),
        mass=np.full(n, 1e10 / n),
        ids=np.arange(n),
        species=np.full(n, "STAR"),
    )
    ps.prepare(check_figure_rotation=False)
    return ps


def test_truncation_convergence():
    """The validation-error-vs-truncation sweep falls then plateaus."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.axes

    # A flattened system genuinely needs both radial and angular resolution.
    sweep = lf.Potential.truncation_convergence(
        _make_system(flatten=(1.0, 0.85, 0.7), n=20_000),
        n_max_values=[2, 6, 12],
        l_max_values=[0, 2, 4],
        n_shells=8,
        n_directions=24,
        seed=1,
    )
    assert isinstance(sweep, lf.TruncationSweep)
    assert sweep.median_error_vs_n.shape == (3,)
    assert sweep.median_error_vs_l.shape == (3,)
    # Higher order is no worse than the lowest (convergence, allowing a plateau).
    assert sweep.median_error_vs_n[-1] <= sweep.median_error_vs_n[0]
    assert sweep.median_error_vs_l[-1] <= sweep.median_error_vs_l[0]
    # A flattened system genuinely needs l > 0: l=0 is the worst angular point.
    assert sweep.median_error_vs_l[0] == sweep.median_error_vs_l.max()
    axes = sweep.plot()
    assert len(axes) == 2 and all(isinstance(a, matplotlib.axes.Axes) for a in axes)
    print(
        "truncation convergence OK: "
        f"err(n)={np.array2string(sweep.median_error_vs_n, precision=4)}, "
        f"err(l)={np.array2string(sweep.median_error_vs_l, precision=4)}"
    )


def test_shrinking_sphere():
    """Shrinking-sphere centring recovers an offset centre despite outliers."""
    rng = np.random.default_rng(0)
    n = 200_000
    su = np.sqrt(rng.uniform(0, 1, n))
    r = su / (1.0 - su)
    mu = rng.uniform(-1, 1, n)
    az = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - mu**2)
    pos = np.stack([r * st * np.cos(az), r * st * np.sin(az), r * mu], axis=1)
    true_c = np.array([12.0, -5.0, 3.0])
    true_v = np.array([80.0, 10.0, -40.0])
    pos += true_c
    vel = rng.normal(0, 50, (n, 3)) + true_v
    mass = np.full(n, 1.0 / n)

    # A distant 5% "stream" that badly biases the global centre of mass.
    m = n // 20
    opos = rng.normal(0, 3, (m, 3)) + np.array([400.0, 0.0, 0.0])
    ovel = rng.normal(0, 50, (m, 3)) + np.array([-300.0, 0.0, 0.0])
    all_pos = np.vstack([pos, opos])
    all_vel = np.vstack([vel, ovel])
    all_mass = np.concatenate([mass, np.full(m, 1.0 / n)])

    ps = lf.ParticleSystem(
        pos=all_pos.copy(),
        vel=all_vel.copy(),
        mass=all_mass.copy(),
        ids=np.arange(len(all_mass)),
        species=np.full(len(all_mass), "DM"),
    )

    # The naive COM is pulled far off by the stream; shrinking sphere is not.
    naive = np.average(all_pos, weights=all_mass, axis=0)
    assert np.linalg.norm(naive - true_c) > 15.0
    pos_c, vel_c = ps.shrinking_sphere_centre(use="all")
    assert np.allclose(pos_c, true_c, atol=0.1), pos_c
    assert np.allclose(vel_c, true_v, atol=3.0), vel_c

    # recentre() defaults to shrinking sphere and puts the blob at the origin.
    ps.recentre()
    assert np.allclose(np.median(ps.pos[:n], axis=0), 0.0, atol=0.05)
    assert np.allclose(np.average(ps.vel[:n], weights=mass, axis=0), 0.0, atol=3.0)
    print("shrinking-sphere centring OK")


def main():
    tol = 0.02  # 2% target agreement (median)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "snap.hdf5")
        a_true, m_true = make_hernquist_snapshot(path)

        ps = lf.ParticleSystem.from_gadget_hdf5(path)
        print(f"loaded {ps.n_particles} particles; " f"species: {set(ps.species)}")
        assert ps.n_particles == 200_001
        assert np.sum(ps.species == "BH") == 1

        ps.prepare()  # recentre on field COM, align, estimate scale radius
        print(f"scale radius estimate: {ps.scale_radius:.3f} (true a={a_true})")
        # Half-mass radius of Hernquist is (1+sqrt2) a, so estimate ~ a.
        assert abs(ps.scale_radius - a_true) / a_true < 0.1

        pot = lf.Potential.from_particles(ps, n_max=12, l_max=4)
        print(f"built SCF potential; n_black_holes={pot.n_black_holes}")
        assert pot.n_black_holes == 1

        result = pot.validate(n_shells=24, n_directions=64)
        print(f"validation vs direct summation: {result}")
        for rr, err in zip(result.radii, result.rel_error):
            print(f"    r_ho={rr:7.3f}  median relerr={err:.3%}")

        assert result.passed(
            tol
        ), f"median relerr {result.median:.3%} exceeds tolerance {tol:.1%}"
        print(f"\nPASS: median agreement {result.median:.3%} < {tol:.1%}")


if __name__ == "__main__":
    test_shrinking_sphere()
    test_radius_mask()
    test_random_subset()
    test_truncation_convergence()
    main()
