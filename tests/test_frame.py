"""Frame tracking: ParticleSystem.frame follows recentre/align/orient_axes.

python tests/test_frame.py
"""

import os
import sys
import tempfile
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lanfear as lf  # noqa: E402
from lanfear import _core  # noqa: E402


def _random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _input_system(seed=2, n=20_000):
    """A rotating triaxial blob, randomly rotated, offset and boosted."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n, 3)) * np.array([3.0, 2.0, 1.0])
    vel = rng.normal(size=(n, 3)) * 20.0
    vel += 40.0 * np.cross([0.0, 0.0, 1.0], pos) / np.linalg.norm(pos, axis=1)[:, None]
    rot = _random_rotation(rng)
    pos = pos @ rot.T + np.array([50.0, -20.0, 7.0])
    vel = vel @ rot.T + np.array([100.0, 30.0, -5.0])
    return lf.ParticleSystem(
        pos=pos,
        vel=vel,
        mass=np.full(n, 1.0 / n),
        ids=np.arange(n),
        species=np.full(n, "STAR"),
    )


def _prepared(seed=2):
    ps = _input_system(seed)
    pos_in, vel_in = ps.pos.copy(), ps.vel.copy()
    ps.prepare(centre="field", pattern_speed="none", check_figure_rotation=False)
    return ps, pos_in, vel_in


def test_frame_maps_input_to_analysis():
    """frame.to_frame reproduces pos/vel; from_frame inverts it."""
    ps0 = _input_system()
    assert ps0.frame.matches(lf.Frame.identity())
    ps, pos_in, vel_in = _prepared()
    rot = ps.frame.rotation
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(rot), 1.0)
    pos, vel = ps.frame.to_frame(pos_in, vel_in)
    assert np.allclose(pos, ps.pos, atol=1e-9)
    assert np.allclose(vel, ps.vel, atol=1e-9)
    back_pos, back_vel = ps.frame.from_frame(ps.pos, ps.vel)
    assert np.allclose(back_pos, pos_in, atol=1e-9)
    assert np.allclose(back_vel, vel_in, atol=1e-9)
    assert np.allclose(ps.frame.centre, [50.0, -20.0, 7.0], atol=0.2)
    print("frame maps input to analysis frame")


def test_frame_reproducible_and_carried():
    """prepare() twice gives matching frames; select() carries the frame."""
    a, _, _ = _prepared()
    b, _, _ = _prepared()
    assert a.frame.matches(b.frame)
    flipped = a.frame.rotated(np.diag([1.0, -1.0, -1.0]))
    assert not a.frame.matches(flipped)
    assert a.select(np.arange(10)).frame is a.frame
    assert a.field.frame is a.frame
    # A further recentre composes with the existing frame.
    ps_in = _input_system()
    a.recentre(on="shrinking_sphere")
    pos, vel = a.frame.to_frame(ps_in.pos, ps_in.vel)
    assert np.allclose(pos, a.pos, atol=1e-9)
    assert np.allclose(vel, a.vel, atol=1e-9)
    print("frame reproducible and carried by select()")


def test_frame_saved_with_results():
    """OrbitResults.save/load round-trips the frame."""
    ps, _, _ = _prepared()
    n = 3
    ncol = len(lf.SUMMARY_COLUMNS)
    res = lf.OrbitResults(
        ids=np.arange(n),
        summary=np.zeros((n, ncol)),
        columns=lf.SUMMARY_COLUMNS,
        time_unit=1.0,
        n_periods=1,
        n_samples=1,
        fundamentals=np.zeros((n, 3)),
        lines=np.zeros((n, 3, 1, 2)),
        diffusion=np.zeros((n, 3)),
        planar_lines=np.zeros((n, 3, 1, 2)),
        length_unit=1.0,
        initial_radius=np.ones(n),
        frame=ps.frame,
    )
    with tempfile.TemporaryDirectory() as d:
        loaded = lf.OrbitResults.load(res.save(os.path.join(d, "orbits")))
        assert loaded.frame is not None and loaded.frame.matches(
            ps.frame, rtol=0, atol=0
        )
        assert res.select(np.array([True, False, True])).frame is ps.frame
        res_none = lf.OrbitResults.load(
            replace(res, frame=None).save(os.path.join(d, "none"))
        )
        assert res_none.frame is None
    print("frame saved with results")


def test_frame_recorded_by_analyse_family():
    """analyse_family stores the particles' frame on the results."""
    ps, _, _ = _prepared()
    ps.estimate_scale_radius()
    pot = lf.Potential.from_particles(ps, n_max=4, l_max=2)
    sub = ps.select(np.arange(8))
    res = lf.analyse_family(
        pot, sub, family="STAR", n_periods=5, n_samples=256, comm=None, progress=False
    )
    assert res.frame is sub.frame
    print("frame recorded by analyse_family")


if __name__ == "__main__":
    assert _core is not None
    test_frame_maps_input_to_analysis()
    test_frame_reproducible_and_carried()
    test_frame_saved_with_results()
    test_frame_recorded_by_analyse_family()
    print("\nALL FRAME TESTS PASSED")
