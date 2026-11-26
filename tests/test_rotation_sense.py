"""Rotation-sense references (SpinReference) and per-orbit rotation senses.

python tests/test_rotation_sense.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lanfear as lf  # noqa: E402
from lanfear import _core  # noqa: E402
from lanfear.rotation_sense import RotationSense  # noqa: E402


def _tilt(angle_deg):
    """Rotation by ``angle_deg`` about x."""
    a = np.radians(angle_deg)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(a), -np.sin(a)], [0.0, np.sin(a), np.cos(a)]]
    )


def _disc(n, rng, z0=0.3, normal_tilt=0.0, v_circ=200.0, dispersion=20.0):
    """Exponential disc rotating about its normal (tilted about x)."""
    R = rng.gamma(2.0, 3.0, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    z = z0 * np.arctanh(rng.uniform(-1, 1, n))
    pos = np.stack([R * np.cos(phi), R * np.sin(phi), z], axis=1)
    vel = np.stack([-np.sin(phi), np.cos(phi), np.zeros(n)], axis=1) * v_circ
    vel += rng.normal(size=(n, 3)) * dispersion
    rot = _tilt(normal_tilt)
    return pos @ rot.T, vel @ rot.T


def _sphere(n, rng, spin_axis=None, omega=5.0, dispersion=100.0, scale=20.0):
    """Uniform sphere of radius ``scale``, optionally in solid-body rotation."""
    r = scale * rng.uniform(0, 1, n) ** (1.0 / 3.0)
    u = rng.normal(size=(n, 3))
    pos = r[:, None] * u / np.linalg.norm(u, axis=1)[:, None]
    vel = rng.normal(size=(n, 3)) * dispersion
    if spin_axis is not None:
        vel += omega * np.cross(spin_axis, pos)
    return pos, vel


def _system(parts):
    """ParticleSystem from ``{label: (pos, vel)}``."""
    pos = np.concatenate([p for p, _ in parts.values()])
    vel = np.concatenate([v for _, v in parts.values()])
    species = np.concatenate(
        [np.full(len(p), label) for label, (p, _) in parts.items()]
    )
    n = len(pos)
    return lf.ParticleSystem(
        pos=pos, vel=vel, mass=np.full(n, 1.0 / n), ids=np.arange(n), species=species
    )


def _angle_deg(a, b):
    return np.degrees(np.arccos(np.clip(abs(float(a @ b)), -1.0, 1.0)))


DM_AXIS = _tilt(40.0) @ np.array([0.0, 0.0, 1.0])


def _misaligned_system(seed=4):
    rng = np.random.default_rng(seed)
    return _system(
        {
            "STAR": _disc(20_000, rng),
            "DM": _sphere(40_000, rng, spin_axis=DM_AXIS),
            "GAS": _sphere(5_000, rng),
        }
    )


def test_reference_axes_recovered():
    """Misaligned spins of two families are each recovered; noise is flagged."""
    ps = _misaligned_system()
    star = ps.reference_spin(family="STAR")
    dm = ps.reference_spin(family="DM")
    assert star.name == "STAR" and dm.name == "DM"
    assert star.significant and dm.significant
    assert _angle_deg(star.axis, [0.0, 0.0, 1.0]) < 1.0
    assert _angle_deg(dm.axis, DM_AXIS) < max(2.0, 3 * np.degrees(dm.angle_uncertainty))
    assert star.axis[2] > 0 and dm.axis @ DM_AXIS > 0  # signed, not just an axis
    assert star.frame is ps.frame and star.geometry is None
    gas = ps.reference_spin(family="GAS")
    assert not gas.significant
    field = ps.reference_spin()
    assert field.name == "field" and field.n_particles == ps.n_particles
    print("reference axes recovered")


def test_spherical_apertures():
    """Sphere and shell selections match radius_mask; names describe them."""
    ps = _misaligned_system()
    is_star = ps.species_mask("STAR")
    ref = ps.reference_spin(family="STAR", radius=5.0, geometry="spherical")
    assert ref.n_particles == int(np.sum(is_star & ps.radius_mask(5.0)))
    assert ref.name == "STAR_r<5" and ref.radius_outer == 5.0
    shell = ps.reference_spin(
        family=["STAR", "DM"], radius=(2.0, 8.0), geometry="spherical"
    )
    both = ps.species_mask("STAR", "DM")
    assert shell.n_particles == int(np.sum(both & ps.radius_mask(8.0, r_min=2.0)))
    assert shell.name == "STAR+DM_2<=r<8"
    smooth = ps.reference_spin(family="DM", radius=20.0, window="gaussian")
    assert smooth.n_particles == int(np.sum(ps.species_mask("DM")))
    assert _angle_deg(smooth.axis, DM_AXIS) < 3.0
    custom = ps.reference_spin(mask=ps.pos[:, 0] > 0, name="half")
    assert custom.name == "half"
    print("spherical apertures")


def test_cylindrical_aperture_tilted_disc():
    """The cylinder follows a tilted disc's own short axis."""
    rng = np.random.default_rng(9)
    ps = _system(
        {"STAR": _disc(20_000, rng, normal_tilt=30.0), "DM": _sphere(20_000, rng)}
    )
    normal = _tilt(30.0) @ np.array([0.0, 0.0, 1.0])
    ref = ps.reference_spin(family="STAR", radius=5.0, geometry="cylindrical")
    assert ref.geometry == "cylindrical"
    assert _angle_deg(ref.cylinder_axis, normal) < 1.0
    assert ref.cylinder_axis @ ref.net_angular_momentum > 0
    # Default z_max: 3 x median |z| about the disc's own axis.
    star = ps.species_mask("STAR")
    height = ps.pos[star] @ ref.cylinder_axis
    cyl_r = np.linalg.norm(ps.pos[star] - height[:, None] * ref.cylinder_axis, axis=1)
    expected = 3.0 * np.median(np.abs(height[cyl_r < 5.0]))
    assert np.isclose(ref.z_max, expected, rtol=1e-10)
    assert ref.n_particles == int(np.sum((cyl_r < 5.0) & (np.abs(height) < ref.z_max)))
    assert ref.name == f"STAR_R<5_|z|<{ref.z_max:.3g}"
    assert _angle_deg(ref.axis, normal) < 1.0
    fixed = ps.reference_spin(
        family="STAR", radius=5.0, geometry="cylindrical", z_max=1.0
    )
    assert fixed.z_max == 1.0
    sphere = ps.reference_spin(family="STAR", radius=5.0, geometry="spherical")
    assert sphere.n_particles != ref.n_particles
    print("cylindrical aperture follows a tilted disc")


def test_cylinder_separates_disc_from_spheroid():
    """A tilted disc inside a rotating stellar spheroid of the same family.

    With a disc-scale z_max the cylindrical reference follows the disc; a
    sphere of the same radius mixes in the spheroid's spin and points
    elsewhere. The default z_max (3 x median |z| of the selection) is inflated
    by the spheroid, so the cylinder then takes in part of it too.
    """
    rng = np.random.default_rng(13)
    disc = _disc(30_000, rng, normal_tilt=30.0)
    spheroid = _sphere(
        30_000,
        rng,
        spin_axis=np.array([0.0, 0.0, 1.0]),
        omega=40.0,
        dispersion=50.0,
        scale=5.0,
    )
    ps = _system(
        {
            "STAR": (
                np.concatenate([disc[0], spheroid[0]]),
                np.concatenate([disc[1], spheroid[1]]),
            )
        }
    )
    normal = _tilt(30.0) @ np.array([0.0, 0.0, 1.0])
    cylinder = ps.reference_spin(
        family="STAR", radius=5.0, geometry="cylindrical", z_max=0.5
    )
    sphere = ps.reference_spin(family="STAR", radius=5.0, geometry="spherical")
    assert _angle_deg(cylinder.cylinder_axis, normal) < 2.0
    assert _angle_deg(cylinder.axis, normal) < 5.0
    assert _angle_deg(sphere.axis, normal) > 5.0
    default = ps.reference_spin(family="STAR", radius=5.0, geometry="cylindrical")
    assert default.z_max > 2 * cylinder.z_max
    print("cylinder separates a tilted disc from the spheroid")


def test_auto_geometry():
    """'auto' follows the fitted component type and never guesses."""
    rng = np.random.default_rng(11)
    ps = _system({"STAR": _disc(4_000, rng), "DM": _sphere(12_000, rng)})
    ps.estimate_scale_radius()
    pot = lf.MultiComponentPotential.from_particles(
        ps,
        components={
            "STAR": lf.disc_component(n_radial=4, n_vert=2),
            "DM": lf.scf_component(n_max=4, l_max=2),
        },
    )
    assert (
        ps.reference_spin(family="STAR", radius=5.0, potential=pot).geometry
        == "cylindrical"
    )
    assert (
        ps.reference_spin(family="DM", radius=20.0, potential=pot).geometry
        == "spherical"
    )
    spec = lf.ReferenceSpec(family="STAR", radius=5.0)
    assert spec.resolve(ps, potential=pot).geometry == "cylindrical"
    for kwargs in (
        {"family": "STAR", "radius": 5.0},  # no potential
        {"family": ["STAR", "DM"], "radius": 5.0, "potential": pot},  # mixed
        {"family": "GAS", "radius": 5.0, "potential": pot},  # no such family
    ):
        try:
            ps.reference_spin(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"reference_spin({kwargs}) should have raised")
    # Without an aperture the geometry is not needed.
    assert ps.reference_spin(family="STAR").geometry is None
    for kwargs in (
        {"family": "NOPE"},
        {"family": "STAR", "radius": 1e-6, "geometry": "spherical"},
        {"mask": np.ones(ps.n_particles, bool)},  # mask without a name
        {"radius": (3.0, 2.0), "geometry": "spherical"},
        {"geometry": "conical"},
    ):
        try:
            ps.reference_spin(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"reference_spin({kwargs}) should have raised")
    print("auto geometry")


def _results_with_moments(means, seconds, status=None, frame=None):
    """OrbitResults whose only meaningful columns are the Lin_* moments."""
    n = len(means)
    cols = lf.SUMMARY_COLUMNS
    summary = np.zeros((n, len(cols)))
    if status is not None:
        summary[:, cols.index("status")] = status
    for k in range(n):
        for i, a in enumerate("xyz"):
            summary[k, cols.index(f"Lin_{a}_mean")] = means[k][i]
            for j, b in enumerate("xyz"):
                summary[k, cols.index(f"Lin_{''.join(sorted(a + b))}")] = seconds[k][
                    i, j
                ]
    return lf.OrbitResults(
        ids=np.arange(n),
        summary=summary,
        columns=cols,
        time_unit=1.0,
        n_periods=1,
        n_samples=1,
        fundamentals=np.zeros((n, 3)),
        lines=np.zeros((n, 3, 1, 2)),
        diffusion=np.zeros((n, 3)),
        planar_lines=np.zeros((n, 3, 1, 2)),
        length_unit=1.0,
        initial_radius=np.ones(n),
        frame=lf.Frame.identity() if frame is None else frame,
    )


def _z_reference(significant=True, frame=None):
    return lf.SpinReference(
        name="z",
        axis=np.array([0.0, 0.0, 1.0]),
        net_angular_momentum=np.array([0.0, 0.0, 1.0]),
        sigma=np.full(3, 0.01),
        angle_uncertainty=0.01,
        significant=significant,
        n_particles=100,
        frame=lf.Frame.identity() if frame is None else frame,
    )


def test_rotation_sense_labels():
    """Coherence, cosine and labels for hand-made moments."""
    tilted = np.array([0.7, 0.0, 0.7])
    polar = np.array([1.0, 0.0, 0.1])
    means = [
        np.array([0.0, 0.0, 1.0]),  # co
        np.array([0.0, 0.0, -1.0]),  # counter
        polar,  # perpendicular
        np.array([0.01, 0.0, 0.0]),  # box: no net circulation
        np.array([0.0, 0.0, 1.0]),  # failed orbit
        tilted,  # co at 45 degrees
    ]
    seconds = [np.outer(m, m) for m in means]
    seconds[3] = np.eye(3)
    status = np.array([0, 0, 0, 0, 2, 0])
    res = _results_with_moments(means, seconds, status)
    ref = res.add_reference(_z_reference())
    kappa = res.circulation_coherence()
    assert np.allclose(kappa[[0, 1, 2, 4, 5]], 1.0) and kappa[3] < 0.01
    cosine = res.reference_cosine("z")
    assert np.allclose(cosine[[0, 1]], [1.0, -1.0])
    assert np.isclose(cosine[5], np.sqrt(0.5))
    assert np.allclose(res.reference_sign_consistency(ref)[[0, 1, 2]], [1.0, -1.0, 1.0])
    S = RotationSense
    expected = [S.CO, S.COUNTER, S.PERPENDICULAR, S.NONE, S.NONE, S.CO]
    assert list(res.rotation_sense("z")) == expected
    two_way = res.rotation_sense("z", perpendicular=None)
    assert two_way[2] == S.CO
    assert res.rotation_sense("z", coherence_min=0.0)[3] == S.PERPENDICULAR
    res.add_reference(_z_reference(significant=False), overwrite=True)
    assert np.all(res.rotation_sense("z") == S.UNDEFINED)
    assert lf.ROTATION_SENSE_NAMES[int(S.COUNTER)] == "counter"
    print("rotation-sense labels")


def test_reference_frame_checks_and_storage():
    """Frames must match; references survive select() and save/load."""
    means = [np.array([0.0, 0.0, 1.0])] * 3
    res = _results_with_moments(means, [np.outer(m, m) for m in means])
    other = lf.Frame.identity().rotated(np.diag([1.0, -1.0, -1.0]))
    for call in (
        lambda: res.add_reference(_z_reference(frame=other)),
        lambda: res.rotation_sense(_z_reference(frame=other)),
    ):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("a reference in another frame should be rejected")
    res.add_reference(_z_reference())
    try:
        res.add_reference(_z_reference())
    except ValueError:
        pass
    else:
        raise AssertionError("a duplicate name should be rejected")
    try:
        res.rotation_sense("missing")
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown name should raise KeyError")
    unknown = _results_with_moments(means, [np.outer(m, m) for m in means])
    unknown.frame = None
    try:
        unknown.add_reference(_z_reference())
    except ValueError:
        pass
    else:
        raise AssertionError("results without a frame cannot take references")

    sub = res.select(np.array([True, False, True]))
    assert list(sub.references) == ["z"]
    sub.references.clear()
    assert "z" in res.references  # select() copies the dict

    ps = _misaligned_system()
    cyl = ps.reference_spin(family="STAR", radius=5.0, geometry="cylindrical")
    res.frame = ps.frame
    res.references.clear()
    res.add_reference(cyl)
    res.add_reference(ps.reference_spin(family="DM"))
    with tempfile.TemporaryDirectory() as d:
        loaded = lf.OrbitResults.load(res.save(os.path.join(d, "orbits")))
    assert list(loaded.references) == list(res.references)
    for name, ref in res.references.items():
        back = loaded.references[name]
        assert back.frame.matches(ref.frame, rtol=0, atol=0)
        for key in ("axis", "net_angular_momentum", "sigma"):
            assert np.array_equal(getattr(back, key), getattr(ref, key))
        for key in (
            "name",
            "family",
            "radius_inner",
            "radius_outer",
            "geometry",
            "z_max",
            "window",
            "significant",
            "n_particles",
            "significance",
            "angle_uncertainty",
        ):
            assert getattr(back, key) == getattr(ref, key), key
    assert np.array_equal(loaded.references[cyl.name].cylinder_axis, cyl.cylinder_axis)
    assert loaded.references["DM"].cylinder_axis is None
    print("reference frame checks and storage")


def test_rotation_sense_of_integrated_orbits():
    """In a spherical potential the labels follow the sign of L . z."""
    rng = np.random.default_rng(5)
    n = 40_000
    su = np.sqrt(rng.uniform(0, 1, n))
    r = su / (1 - su)
    u = rng.normal(size=(n, 3))
    pos = r[:, None] * u / np.linalg.norm(u, axis=1)[:, None]
    scf = _core.SCFPotential(8, 0, pos, np.full(n, 1.0 / n))
    states = np.zeros((12, 6))
    angles = np.radians(np.linspace(5, 175, 12))  # L tilt from +z
    states[:, 0] = 1.0
    states[:, 4] = 0.4 * np.cos(angles)
    states[:, 5] = 0.4 * np.sin(angles)  # L = x x v lies in the y-z plane
    summ, fund, lines, diff, planar = scf.analyse_batch(
        states, 10, 1024, 1e-10, 1e-9, 2
    )
    res = lf.OrbitResults(
        ids=np.arange(12),
        summary=summ,
        columns=lf.SUMMARY_COLUMNS,
        time_unit=1.0,
        n_periods=10,
        n_samples=1024,
        fundamentals=fund,
        lines=lines,
        diffusion=diff,
        planar_lines=planar,
        length_unit=1.0,
        initial_radius=np.ones(12),
        frame=lf.Frame.identity(),
    )
    res.add_reference(_z_reference())
    assert np.allclose(res.circulation_coherence(), 1.0, atol=1e-6)
    l0 = np.cross(states[:, :3], states[:, 3:])
    cos0 = l0[:, 2] / np.linalg.norm(l0, axis=1)
    assert np.allclose(res.reference_cosine("z"), cos0, atol=1e-6)
    S = RotationSense
    expected = np.where(
        np.abs(cos0) < 0.5, S.PERPENDICULAR, np.where(cos0 > 0, S.CO, S.COUNTER)
    )
    assert np.array_equal(res.rotation_sense("z"), expected)
    print("rotation sense of integrated orbits")


def test_reference_spec_parse():
    """The compact form round-trips into ReferenceSpec; bad forms raise."""
    parse = lf.ReferenceSpec.parse
    assert parse("field") == lf.ReferenceSpec()
    assert parse("DM") == lf.ReferenceSpec(family=("DM",))
    assert parse("STAR:5") == lf.ReferenceSpec(family=("STAR",), radius=5.0)
    assert parse("STAR+DM:2-8:spherical") == lf.ReferenceSpec(
        family=("STAR", "DM"), radius=(2.0, 8.0), geometry="spherical"
    )
    assert parse("STAR:5:cylindrical:z=1.0:name=disc") == lf.ReferenceSpec(
        family=("STAR",), radius=5.0, geometry="cylindrical", z_max=1.0, name="disc"
    )
    assert parse("DM:20:gaussian").window == "gaussian"
    assert parse("STAR:local:width=0.5") == lf.ReferenceSpec(
        family=("STAR",), local=True, width=0.5
    )
    for bad in ("", "STAR:five", "STAR:1-2-3", "STAR:5:5", "STAR:local:5", "STAR:z="):
        try:
            parse(bad)
        except ValueError:
            continue
        raise AssertionError(f"parse({bad!r}) should have raised")
    ps = _misaligned_system()
    ref = parse("STAR:5:spherical").resolve(ps)
    assert ref.name == "STAR_r<5"
    try:
        ps.reference_spin(family="STAR", radius=5.0, geometry="spherical", z_max=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("z_max with a spherical aperture should raise")
    print("reference spec parsing")


def _counter_core_system(seed=21, n=60_000, core=2.0):
    """A uniform sphere whose inner ``core`` counter-rotates about z."""
    rng = np.random.default_rng(seed)
    pos, vel = _sphere(
        n,
        rng,
        spin_axis=np.array([0.0, 0.0, 1.0]),
        omega=20.0,
        dispersion=20.0,
        scale=10.0,
    )
    vel[np.linalg.norm(pos, axis=1) < core] *= -1.0
    return _system({"STAR": (pos, vel)})


def test_profile_reference_follows_a_counter_rotating_core():
    """The local reference flips with the spin; the global one does not."""
    ps = _counter_core_system()
    default = ps.reference_spin_profile(family="STAR", geometry="spherical")
    assert default.name == "STAR_local" and default.geometry == "spherical"
    assert len(default.radii) == 12 and np.all(np.diff(default.radii) > 0)
    r = np.linalg.norm(ps.pos, axis=1)
    assert np.allclose(default.radii[[0, -1]], np.percentile(r, [10, 90]))
    # The default shells start outside this core, so cover it explicitly.
    profile = ps.reference_spin_profile(
        family="STAR", radii=np.geomspace(0.5, 8.0, 12), geometry="spherical"
    )
    axes, significant = profile.axes_at([0.8, 6.0])
    assert significant.all()
    assert axes[0, 2] < -0.95 and axes[1, 2] > 0.95
    explicit = lf.reference_spin_profile(
        ps, family="STAR", radii=[1.0, 5.0], geometry="spherical"
    )
    assert np.array_equal(explicit.radii, [1.0, 5.0])
    spec = lf.ReferenceSpec.parse("STAR:local:spherical")
    assert isinstance(spec.resolve(ps), lf.ProfileReference)
    for kwargs in (
        {"family": "STAR"},
        {"family": "STAR", "radii": [2.0, 1.0], "geometry": "spherical"},
        {"family": "STAR", "geometry": "spherical", "z_max": 1.0},
    ):
        try:
            ps.reference_spin_profile(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"reference_spin_profile({kwargs}) should have raised")
    print("profile reference follows a counter-rotating core")


def test_cylindrical_profile_of_a_tilted_disc():
    """A cylindrical profile follows a tilted disc's own axis."""
    rng = np.random.default_rng(23)
    ps = _system(
        {"STAR": _disc(30_000, rng, normal_tilt=30.0), "DM": _sphere(20_000, rng)}
    )
    normal = _tilt(30.0) @ np.array([0.0, 0.0, 1.0])
    profile = ps.reference_spin_profile(family="STAR", geometry="cylindrical")
    assert _angle_deg(profile.cylinder_axis, normal) < 1.0
    axes, significant = profile.axes_at(profile.radii)
    assert significant.all()
    assert max(_angle_deg(a, normal) for a in axes) < 3.0
    print("cylindrical profile of a tilted disc")


def _profile_z_flip():
    """A hand-made profile: spin -z at r = 1, +z at r = 10."""
    return lf.ProfileReference(
        name="flip",
        radii=np.array([1.0, 10.0]),
        net_angular_momentum=np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]),
        sigma=np.full((2, 3), 0.01),
        significant=np.array([True, True]),
        n_particles=100,
        frame=lf.Frame.identity(),
    )


def test_rotation_sense_against_a_profile():
    """Orbits are compared with the local direction at their r_mean."""
    lz = [-1.0, 1.0, -1.0, 1.0, 1.0]
    means = [np.array([0.0, 0.0, v]) for v in lz]
    res = _results_with_moments(means, [np.outer(m, m) for m in means])
    radii = [0.5, 12.0, 12.0, np.sqrt(10.0), 1.0]  # inside, beyond, flip, inner
    res.summary[:, lf.SUMMARY_COLUMNS.index("r_mean")] = radii
    res.add_reference(_profile_z_flip())
    S = RotationSense
    assert list(res.rotation_sense("flip")) == [
        S.CO,
        S.CO,
        S.COUNTER,
        S.UNDEFINED,
        S.COUNTER,
    ]
    assert np.allclose(res.reference_cosine("flip")[[0, 1, 2, 4]], [1, 1, -1, -1])
    with tempfile.TemporaryDirectory() as d:
        loaded = lf.OrbitResults.load(res.save(os.path.join(d, "orbits")))
    back = loaded.references["flip"]
    assert isinstance(back, lf.ProfileReference)
    for key in ("radii", "net_angular_momentum", "sigma", "significant"):
        assert np.array_equal(getattr(back, key), getattr(res.references["flip"], key))
    assert list(loaded.rotation_sense("flip")) == list(res.rotation_sense("flip"))
    print("rotation sense against a profile")


if __name__ == "__main__":
    test_reference_axes_recovered()
    test_spherical_apertures()
    test_cylindrical_aperture_tilted_disc()
    test_cylinder_separates_disc_from_spheroid()
    test_auto_geometry()
    test_rotation_sense_labels()
    test_reference_frame_checks_and_storage()
    test_rotation_sense_of_integrated_orbits()
    test_reference_spec_parse()
    test_profile_reference_follows_a_counter_rotating_core()
    test_cylindrical_profile_of_a_tilted_disc()
    test_rotation_sense_against_a_profile()
    print("\nALL ROTATION-SENSE TESTS PASSED")
