"""Orbit-classification tests (Milestone 4).

Constructs orbits of known type (box, short-/long-axis tube, rosette) in
triaxial and near-spherical SCF potentials and checks the classifier assigns the
right family, plus a unit test of the resonance finder.

    python tests/test_classify.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lanfear import _core, OrbitClass  # noqa: E402
from lanfear.orbits import OrbitResults, SUMMARY_COLUMNS  # noqa: E402
from lanfear.classify import _find_resonances  # noqa: E402


def build_scf(flatten=(1.0, 1.0, 1.0), n_max=12, l_max=6, n=300_000, seed=3):
    rng = np.random.default_rng(seed)
    su = np.sqrt(rng.uniform(0, 1, n))
    r = su / (1 - su)
    mu = rng.uniform(-1, 1, n)
    az = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - mu**2)
    pos = np.stack([r * st * np.cos(az), r * st * np.sin(az), r * mu], axis=1)
    pos *= np.asarray(flatten)
    return _core.SCFPotential(n_max, l_max, pos, np.full(n, 1.0 / n))


def _classify_state(scf, state):
    summ, fund, lines = scf.analyse_batch(
        np.array([state], float), n_periods=40, n_samples=4096, n_lines=4
    )
    res = OrbitResults(
        ids=np.array([0]),
        summary=summ,
        columns=SUMMARY_COLUMNS,
        time_unit=1.0,
        n_periods=40,
        n_samples=4096,
        fundamentals=fund,
        lines=lines,
    )
    return res.classify()


def _vcirc(scf, x, y, z):
    a = np.array(scf.acceleration(x, y, z))
    rr = np.array([x, y, z])
    nr = np.linalg.norm(rr)
    return np.sqrt(max(-a @ rr / nr, 1e-6) * nr)


def test_resonance_finder():
    # banana 2:-1:0, a 1:1:1, and an incommensurate triple (no low-order res).
    w = np.array(
        [
            [0.40, 0.80, 0.137],  # 2 w_x - w_y = 0
            [0.30, 0.30, 0.30],  # 1:1:1
            [0.317, 0.482, 0.613],
        ]
    )  # incommensurate
    vec, order = _find_resonances(w, max_order=5, tol=0.01)
    assert tuple(vec[0]) == (2, -1, 0) and order[0] == 3
    assert order[1] == 2  # some 1:1 among equal freqs
    assert order[2] == 0  # no low-order resonance
    print("resonance finder OK")


def test_known_orbits():
    tri = build_scf(flatten=(1.0, 0.75, 0.5))  # x long, y intermediate, z short
    sph = build_scf(flatten=(1.0, 1.0, 1.0), l_max=4)

    cl = _classify_state(tri, [1.5, 1.0, 0.7, 0, 0, 0])  # released from rest
    assert cl.labels[0] == OrbitClass.BOX, cl.names[0]
    print(f"  box      -> {cl.names[0]}")

    cl = _classify_state(tri, [2.0, 0, 0.6, 0, 0.6, 0.15])  # circulate about z
    assert cl.labels[0] == OrbitClass.SHORT_AXIS_TUBE, cl.names[0]
    assert cl.circulation[0, 2] > 0.9
    print(f"  z-tube   -> {cl.names[0]}")

    v = _vcirc(tri, 0.3, 2.0, 0)
    cl = _classify_state(tri, [0.3, 2.0, 0, 0, 0, 0.9 * v])  # circulate about x
    assert cl.labels[0] in (
        OrbitClass.INNER_LONG_AXIS_TUBE,
        OrbitClass.OUTER_LONG_AXIS_TUBE,
    ), cl.names[0]
    assert cl.circulation[0, 0] > 0.9
    print(f"  x-tube   -> {cl.names[0]}")

    vc = np.sqrt(-np.array(sph.acceleration(2.0, 0, 0))[0] * 2.0)
    cl = _classify_state(sph, [2.0, 0, 0.3, 0, 0.9 * vc, 0.2 * vc])  # tilted loop
    assert cl.labels[0] == OrbitClass.ROSETTE, cl.names[0]
    print(f"  rosette  -> {cl.names[0]}")
    print("known-orbit classification OK")


def test_population():
    """A triaxial population classifies fully and yields a sensible family mix."""
    tri = build_scf(flatten=(1.0, 0.8, 0.6))
    rng = np.random.default_rng(11)
    n = 400
    # Random bound-ish initial conditions spanning box and tube regions.
    pos = rng.uniform(-3, 3, (n, 3))
    r = np.linalg.norm(pos, axis=1)
    keep = (r > 0.5) & (r < 5)
    pos = pos[keep]
    states = []
    for p in pos:
        vc = _vcirc(tri, *p)
        v = rng.uniform(-1, 1, 3)
        v *= rng.uniform(0, 1.0) * vc / (np.linalg.norm(v) + 1e-9)
        states.append([*p, *v])
    states = np.array(states)

    summ, fund, lines = tri.analyse_batch(
        states, n_periods=30, n_samples=2048, n_lines=4
    )
    res = OrbitResults(
        ids=np.arange(len(states)),
        summary=summ,
        columns=SUMMARY_COLUMNS,
        time_unit=1.0,
        n_periods=30,
        n_samples=2048,
        fundamentals=fund,
        lines=lines,
    )
    cl = res.classify()
    ok = res.ok
    # Every successfully integrated orbit gets a (non-UNCLASSIFIED) family.
    assert np.all(cl.labels[ok] != OrbitClass.UNCLASSIFIED)
    counts = cl.counts()
    print(f"  {int(ok.sum())} orbits classified; families: {counts}")
    # A triaxial potential should host both boxes and tubes.
    n_box = np.sum((cl.labels == OrbitClass.BOX) | (cl.labels == OrbitClass.BOXLET))
    n_tube = np.sum(
        (cl.labels >= OrbitClass.SHORT_AXIS_TUBE)
        & (cl.labels <= OrbitClass.INTERMEDIATE_AXIS_TUBE)
    )
    assert n_box > 0 and n_tube > 0
    print("population classification OK")


if __name__ == "__main__":
    print("== resonance finder ==")
    test_resonance_finder()
    print("== known orbits ==")
    test_known_orbits()
    print("== population ==")
    test_population()
    print("\nALL CLASSIFICATION TESTS PASSED")
