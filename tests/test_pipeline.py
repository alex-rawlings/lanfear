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
    n = 400  # only mask semantics and subset selection are checked
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
    ps.prepare(centre="shrinking_sphere", check_figure_rotation=False)

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
    # Only ID/count semantics are checked below (not orbit physics), so the
    # integration itself is kept as cheap as possible: 1 period, few samples.
    res = lf.analyse_family(
        pot, inner, family="STAR", n_periods=1, n_samples=64, progress=False
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


def test_select_slice():
    """select accepts a slice as well as boolean and index arrays."""
    rng = np.random.default_rng(3)
    n = 50
    ps = lf.ParticleSystem(
        pos=rng.normal(size=(n, 3)),
        vel=rng.normal(size=(n, 3)),
        mass=np.ones(n),
        ids=np.arange(n),
        species=np.full(n, "STAR"),
    )
    by_slice = ps.select(slice(5, 40, 3))
    by_index = ps.select(np.arange(5, 40, 3))
    assert np.array_equal(by_slice.ids, by_index.ids)
    assert np.array_equal(by_slice.pos, by_index.pos)
    assert np.array_equal(by_slice.species, by_index.species)
    print("select by slice OK")


def test_load_by_ids():
    """A chunked ID-selected load matches loading everything and selecting."""
    import h5py

    rng = np.random.default_rng(9)
    n_star, n_dm = 1000, 700
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "snap.hdf5")
        with h5py.File(path, "w") as f:
            f.create_group("Header").attrs["MassTable"] = np.array(
                [0.0, 2.5, 0.0, 0.0, 0.0, 0.0]
            )
            g4 = f.create_group("PartType4")
            g4.create_dataset("Coordinates", data=rng.normal(size=(n_star, 3)))
            g4.create_dataset("Velocities", data=rng.normal(size=(n_star, 3)))
            g4.create_dataset("Masses", data=rng.uniform(1, 2, n_star))
            # Shuffled uint32 IDs, as in many cosmological snapshots.
            g4.create_dataset(
                "ParticleIDs", data=rng.permutation(n_star).astype(np.uint32)
            )
            g1 = f.create_group("PartType1")  # masses from the MassTable
            g1.create_dataset("Coordinates", data=rng.normal(size=(n_dm, 3)))
            g1.create_dataset("Velocities", data=rng.normal(size=(n_dm, 3)))
            g1.create_dataset(
                "ParticleIDs", data=np.arange(n_star, n_star + n_dm, dtype=np.uint32)
            )

        full = lf.ParticleSystem.from_gadget_hdf5(path)
        wanted = rng.choice(n_star + n_dm, size=300, replace=False)
        # Duplicates, an ID beyond uint32 and an absent ID are ignored.
        request = np.r_[wanted, wanted[:10], 2**40, n_star + n_dm + 5]
        sub = lf.ParticleSystem.from_gadget_hdf5(path, ids=request, chunk_size=64)
        ref = full.select(np.isin(full.ids, wanted))

        assert sub.n_particles == 300
        assert np.array_equal(sub.ids, ref.ids)
        assert np.array_equal(sub.pos, ref.pos)
        assert np.array_equal(sub.vel, ref.vel)
        assert np.array_equal(sub.mass, ref.mass)
        assert np.array_equal(sub.species, ref.species)
        assert np.all(sub.mass[sub.species == "DM"] == 2.5)

        # Chunking without a selection reproduces the whole snapshot.
        chunked = lf.ParticleSystem.from_gadget_hdf5(path, chunk_size=64)
        assert np.array_equal(chunked.ids, full.ids)
        assert np.array_equal(chunked.pos, full.pos)

        for bad in ([], [10**6]):
            try:
                lf.ParticleSystem.from_gadget_hdf5(path, ids=bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"ids={bad} should raise ValueError")
    print("load by IDs OK")


def test_parallel_load():
    """Reading the blocks in worker processes gives exactly the serial result."""
    import h5py

    from lanfear.particle_system import _iter_gadget_hdf5

    rng = np.random.default_rng(17)
    n_star, n_dm = 900, 600
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "snap.hdf5")
        with h5py.File(path, "w") as f:
            hdr = f.create_group("Header")
            hdr.attrs["MassTable"] = np.array([0.0, 2.5, 0.0, 0.0, 0.0, 0.0])
            hdr.attrs["Time"] = 0.5
            par = f.create_group("Parameters")  # conversion done in the workers
            par.attrs["ComovingIntegrationOn"] = 1
            par.attrs["HubbleParam"] = 0.7
            g4 = f.create_group("PartType4")
            g4.create_dataset("Coordinates", data=rng.normal(size=(n_star, 3)))
            g4.create_dataset("Velocities", data=rng.normal(size=(n_star, 3)))
            g4.create_dataset("Masses", data=rng.uniform(1, 2, n_star))
            g4.create_dataset(
                "ParticleIDs", data=rng.permutation(n_star).astype(np.uint32)
            )
            g1 = f.create_group("PartType1")  # masses from the MassTable
            g1.create_dataset("Coordinates", data=rng.normal(size=(n_dm, 3)))
            g1.create_dataset("Velocities", data=rng.normal(size=(n_dm, 3)))
            g1.create_dataset(
                "ParticleIDs", data=np.arange(n_star, n_star + n_dm, dtype=np.uint32)
            )

        wanted = rng.choice(n_star + n_dm, size=200, replace=False)
        for ids in (wanted, None):
            # Many more blocks than workers, so the look-ahead window refills.
            serial = lf.ParticleSystem.from_gadget_hdf5(path, ids=ids, chunk_size=50)
            parallel = lf.ParticleSystem.from_gadget_hdf5(
                path, ids=ids, chunk_size=50, n_workers=3
            )
            for name in ("pos", "vel", "mass", "ids", "species"):
                assert np.array_equal(getattr(parallel, name), getattr(serial, name))

        # More workers than blocks is capped, not an error.
        few = lf.ParticleSystem.from_gadget_hdf5(path, n_workers=8)
        assert few.n_particles == n_star + n_dm

        # Abandoning the reader part-way shuts the pool down without hanging.
        reader = _iter_gadget_hdf5(path, chunk_size=20, n_workers=2)
        assert len(next(reader)[0]) == 20
        reader.close()

        # An error in a worker reaches the caller.
        with h5py.File(path, "a") as f:
            del f["PartType1/Velocities"]
        try:
            lf.ParticleSystem.from_gadget_hdf5(path, chunk_size=50, n_workers=2)
        except KeyError:
            pass
        else:
            raise AssertionError("a worker's KeyError should propagate")
    print("parallel load OK")


def test_periodic_unwrap():
    """A halo straddling the edges of a periodic box loads contiguous."""
    import h5py

    rng = np.random.default_rng(12)
    box, n = 100.0, 2000
    # A compact blob across the x = 0 and y = 0 faces, plus a few far-away
    # particles of the box that are not selected.
    truth = rng.normal(size=(n, 3)) + np.array([99.5, 0.5, 50.0])
    other = rng.uniform(30.0, 70.0, size=(50, 3))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "box.hdf5")
        with h5py.File(path, "w") as f:
            h = f.create_group("Header")
            h.attrs["MassTable"] = np.zeros(6)
            h.attrs["BoxSize"] = box
            f.create_group("Config").attrs["PERIODIC"] = 1
            g = f.create_group("PartType1")
            g.create_dataset("Coordinates", data=np.mod(np.r_[truth, other], box))
            g.create_dataset("Velocities", data=np.zeros((n + 50, 3)))
            g.create_dataset("Masses", data=np.ones(n + 50))
            g.create_dataset("ParticleIDs", data=np.arange(n + 50, dtype=np.uint64))

        halo = lf.ParticleSystem.from_gadget_hdf5(path, ids=np.arange(n))
        assert np.allclose(halo.box_size, box)
        assert np.allclose(halo.pos, truth)
        assert np.allclose(halo.select(slice(0, 5)).box_size, box)

        # periodic=False keeps the wrapped coordinates and no box size.
        raw = lf.ParticleSystem.from_gadget_hdf5(path, ids=np.arange(n), periodic=False)
        assert raw.box_size is None
        assert np.allclose(raw.pos, np.mod(truth, box))

        # Without the Config flag the box is only periodic when asked to be.
        with h5py.File(path, "a") as f:
            del f["Config"]
        assert lf.ParticleSystem.from_gadget_hdf5(path).box_size is None
        forced = lf.ParticleSystem.from_gadget_hdf5(
            path, ids=np.arange(n), periodic=True
        )
        assert np.allclose(forced.pos, truth)
    print("periodic unwrap OK")


def test_physical_units():
    """Cosmological snapshots load in physical units; others load as stored."""
    import h5py

    rng = np.random.default_rng(21)
    a, h, box, n = 0.5, 0.7, 100.0, 500
    pos = rng.normal(size=(n, 3)) + 50.0
    vel = rng.normal(size=(n, 3))
    mass = rng.uniform(1.0, 2.0, n)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cosmo.hdf5")
        with h5py.File(path, "w") as f:
            hdr = f.create_group("Header")
            hdr.attrs["MassTable"] = np.array([0.0, 3.0, 0.0, 0.0, 0.0, 0.0])
            hdr.attrs["BoxSize"] = box
            hdr.attrs["Time"] = a
            par = f.create_group("Parameters")
            par.attrs["ComovingIntegrationOn"] = 1
            par.attrs["HubbleParam"] = h
            f.create_group("Config").attrs["PERIODIC"] = 1
            g4 = f.create_group("PartType4")  # Gadget-4 scaling attributes
            for name, data, a_exp, h_exp in (
                ("Coordinates", pos, 1.0, -1.0),
                ("Velocities", vel, 0.5, 0.0),
                ("Masses", mass, 0.0, -1.0),
            ):
                g4.create_dataset(name, data=data)
                g4[name].attrs["a_scaling"] = a_exp
                g4[name].attrs["h_scaling"] = h_exp
            g4.create_dataset("ParticleIDs", data=np.arange(n))
            g1 = f.create_group("PartType1")  # no attributes: default exponents
            g1.create_dataset("Coordinates", data=pos)
            g1.create_dataset("Velocities", data=vel)
            g1.create_dataset("ParticleIDs", data=np.arange(n, 2 * n))

        ps = lf.ParticleSystem.from_gadget_hdf5(path)
        assert ps.scale_factor == a
        assert np.allclose(ps.box_size, box * a / h)
        for label in ("STAR", "DM"):
            sel = ps.species == label
            assert np.allclose(ps.pos[sel], pos * a / h)
            assert np.allclose(ps.vel[sel], vel * np.sqrt(a))
        assert np.allclose(ps.mass[ps.species == "STAR"], mass / h)
        assert np.allclose(ps.mass[ps.species == "DM"], 3.0 / h)

        # cosmological=False keeps the stored (comoving) values.
        raw = lf.ParticleSystem.from_gadget_hdf5(path, cosmological=False)
        assert raw.scale_factor is None
        assert np.allclose(raw.pos[raw.species == "STAR"], pos)
        assert np.allclose(raw.box_size, box)

        # An isolated run is already physical: Time is a time and HubbleParam
        # plays no part, so nothing is rescaled.
        with h5py.File(path, "a") as f:
            f["Parameters"].attrs["ComovingIntegrationOn"] = 0
            f["Header"].attrs["Time"] = 2.0
        iso = lf.ParticleSystem.from_gadget_hdf5(path)
        assert iso.scale_factor is None
        assert np.allclose(iso.pos[iso.species == "STAR"], pos)
        assert np.allclose(iso.mass[iso.species == "DM"], 3.0)

        # Forcing a cosmological reading needs Time and HubbleParam.
        with h5py.File(path, "a") as f:
            del f["Parameters"]
            del f["Header"].attrs["Time"]
        try:
            lf.ParticleSystem.from_gadget_hdf5(path, cosmological=True)
        except ValueError:
            pass
        else:
            raise AssertionError("missing Time should raise ValueError")
    print("physical units OK")


def _write_gadget3(path, pos, vel, mass, header):
    """Write a Gadget-3 style snapshot: Header only, no Parameters or Config."""
    import h5py

    with h5py.File(path, "w") as f:
        hdr = f.create_group("Header")
        hdr.attrs["MassTable"] = np.zeros(6)
        for key, value in header.items():
            hdr.attrs[key] = value
        g = f.create_group("PartType1")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=vel)
        g.create_dataset("Masses", data=mass)
        g.create_dataset("ParticleIDs", data=np.arange(len(mass), dtype=np.uint32))


def test_gadget3_header():
    """Gadget-3 snapshots: comoving runs and periodicity inferred from Header."""
    rng = np.random.default_rng(33)
    a, h, box, n = 0.5, 0.7, 100.0, 1000
    # A halo straddling the x = 0 face, in comoving coordinates.
    comoving = rng.normal(size=(n, 3)) + np.array([0.5, 50.0, 50.0])
    vel = rng.normal(size=(n, 3))
    mass = np.ones(n)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "g3_cosmo.hdf5")
        _write_gadget3(
            path,
            np.mod(comoving, box),
            vel,
            mass,
            {
                "Time": a,
                "Redshift": 1.0 / a - 1.0,
                "Omega0": 0.3,
                "HubbleParam": h,
                "BoxSize": box,
            },
        )
        ps = lf.ParticleSystem.from_gadget_hdf5(path)
        assert ps.scale_factor == a
        assert np.allclose(ps.box_size, box * a / h)
        assert np.allclose(ps.pos, comoving * a / h)
        assert np.allclose(ps.vel, vel * np.sqrt(a))
        assert np.allclose(ps.mass, mass / h)

        # An isolated Gadget-3 run: Time is a time, Redshift is 0, and the
        # leftover HubbleParam and BoxSize must not rescale or wrap anything.
        path = os.path.join(d, "g3_isolated.hdf5")
        pos = rng.normal(size=(n, 3)) * 30.0
        _write_gadget3(
            path,
            pos,
            vel,
            mass,
            {
                "Time": 2.0,
                "Redshift": 0.0,
                "Omega0": 0.0,
                "HubbleParam": 0.7,
                "BoxSize": 50.0,
            },
        )
        iso = lf.ParticleSystem.from_gadget_hdf5(path)
        assert iso.scale_factor is None and iso.box_size is None
        assert np.allclose(iso.pos, pos)
        assert np.allclose(iso.mass, mass)
    print("Gadget-3 header OK")


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
    ps.prepare(centre="shrinking_sphere", check_figure_rotation=False)
    return ps


def test_truncation_convergence():
    """The validation-error-vs-truncation sweep falls then plateaus."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.axes

    # A flattened system genuinely needs both radial and angular resolution.
    # Rebuilds the SCF expansion 6 times (3 n_max + 3 l_max points), so the
    # particle count and validate() sampling are kept as small as the
    # monotonic-convergence assertions below can tolerate.
    sweep = lf.Potential.truncation_convergence(
        _make_system(flatten=(1.0, 0.85, 0.7), n=5_000),
        n_max_values=[2, 6, 12],
        l_max_values=[0, 2, 4],
        n_shells=6,
        n_directions=16,
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

    # No BH particles, so the shrinking-sphere centre must be requested
    # explicitly; it puts the blob at the origin.
    ps.recentre(on="shrinking_sphere")
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
    test_select_slice()
    test_load_by_ids()
    test_parallel_load()
    test_periodic_unwrap()
    test_physical_units()
    test_gadget3_header()
    test_truncation_convergence()
    main()
