"""Figure-rotation detection tests.

``ParticleSystem.detect_figure_rotation`` heuristically flags a non-axisymmetric
field with significant ordered rotation about its short axis (the regime in which
the static-potential classifier would mis-assign families). These tests check it
fires for a tumbling-like triaxial figure and stays quiet for spherical,
axisymmetric-disc, and non-rotating triaxial systems.

    python tests/test_figure_rotation.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lanfear as lf  # noqa: E402


def _build(n=60000, squash=(1, 1, 1), v_tan=0.0, sigma=50.0, seed=0):
    rng = np.random.default_rng(seed)
    su = np.sqrt(rng.uniform(0, 1, n))
    rad = su / (1 - su)
    mu = rng.uniform(-1, 1, n)
    az = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - mu**2)
    pos = np.stack([rad * st * np.cos(az), rad * st * np.sin(az), rad * mu], axis=1)
    pos *= np.asarray(squash) * 3.0
    vel = rng.normal(0, sigma, (n, 3))
    if v_tan:  # add net rotation about z (the short axis for these squashes)
        R = np.hypot(pos[:, 0], pos[:, 1]) + 1e-9
        vel[:, 0] += -v_tan * pos[:, 1] / R
        vel[:, 1] += v_tan * pos[:, 0] / R
    return lf.ParticleSystem(
        pos=pos,
        vel=vel,
        mass=np.full(n, 1e10 / n),
        ids=np.arange(n),
        species=np.full(n, "STAR"),
    )


def test_detection():
    cases = [
        ("spherical", (1, 1, 1), 0.0, False),
        ("axisymmetric disc + rotation", (1, 1, 0.2), 45.0, False),
        ("triaxial, non-rotating", (1, 0.7, 0.4), 0.0, False),
        ("triaxial + rotation", (1, 0.7, 0.4), 45.0, True),
        ("bar-like + rotation", (1, 0.85, 0.5), 40.0, True),
        ("nearly axisymmetric + rotation", (1, 0.97, 0.5), 45.0, False),
    ]
    for name, squash, v_tan, expect in cases:
        ps = _build(squash=squash, v_tan=v_tan, seed=abs(hash(name)) % 1000)
        d = ps.detect_figure_rotation()
        assert d["detected"] == expect, (
            f"{name}: detected={d['detected']} expected={expect} "
            f"(b/a={d['b_over_a']:.2f}, v_rot/sigma={d['rotation_measure']:.2f})"
        )
        # Sanity: a genuinely spherical system reports near-unity axis ratios.
        if name == "spherical":
            assert d["b_over_a"] > 0.9 and d["c_over_a"] > 0.9
        print(
            f"  [{'WARN' if d['detected'] else 'ok'}] {name}: "
            f"b/a={d['b_over_a']:.2f} c/a={d['c_over_a']:.2f} "
            f"v_rot/sigma={d['rotation_measure']:.2f}"
        )
    print("figure-rotation detection checks passed")


def test_prepare_runs_check():
    """prepare() invokes the check and the flag matches a direct call."""
    ps = _build(squash=(1, 0.7, 0.4), v_tan=45.0, seed=7)
    direct = ps.detect_figure_rotation()["detected"]
    ps.prepare()  # must not raise; runs the check internally
    # prepare() recentres/aligns but the detection outcome should be unchanged.
    assert ps.detect_figure_rotation()["detected"] == direct
    # And it can be disabled.
    _build(squash=(1, 0.7, 0.4), v_tan=45.0, seed=7).prepare(
        check_figure_rotation=False
    )
    print("prepare() figure-rotation wiring checks passed")


if __name__ == "__main__":
    print("== detection ==")
    test_detection()
    print("== prepare wiring ==")
    test_prepare_runs_check()
    print("\nALL FIGURE-ROTATION TESTS PASSED")
