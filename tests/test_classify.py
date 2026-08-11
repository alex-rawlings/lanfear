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


def test_condense_families():
    """condense_families() folds subclasses into box / tube / unclassified."""
    from lanfear import OrbitClassification, OrbitFamily

    labels = np.array([int(c) for c in OrbitClass])  # one of every class, 0..7
    n = len(labels)
    zeros3 = np.zeros((n, 3))
    cl = OrbitClassification(
        labels=labels,
        circulation=zeros3,
        tube_axis=np.zeros(n, int),
        planarity=np.zeros(n),
        resonance=np.zeros((n, 3), int),
        resonance_order=np.zeros(n, int),
    )
    cond = cl.condense_families()
    assert isinstance(cond, OrbitClassification)
    assert set(cond.names) <= {"box", "tube", "unclassified"}

    expect = {
        "unclassified": "unclassified",
        "box": "box",
        "boxlet": "box",
        "short_axis_tube": "tube",
        "inner_long_axis_tube": "tube",
        "outer_long_axis_tube": "tube",
        "intermediate_axis_tube": "tube",
        "rosette": "tube",
    }
    for full_name, cond_name in zip(cl.names, cond.names):
        assert cond_name == expect[full_name], (full_name, cond_name)

    assert cond.counts() == {"unclassified": 1, "box": 2, "tube": 5}
    assert cond.mask(OrbitFamily.TUBE).sum() == 5
    assert cond.mask(OrbitFamily.BOX).sum() == 2
    # Diagnostic arrays are carried through, and re-condensing is a no-op.
    assert np.array_equal(cond.circulation, cl.circulation)
    assert cond.condense_families().counts() == cond.counts()
    print(f"condense_families OK: {cond.counts()}")


def test_plot_class_fractions():
    """plot_class_fractions bins by radius and returns a matplotlib axes."""
    import matplotlib

    matplotlib.use("Agg")  # headless
    from lanfear import OrbitClassification

    n = 6
    # radii 0.5..5.5; two classes (box, short-axis-tube) split across two bins.
    radius = np.array([0.5, 0.6, 0.7, 4.0, 4.5, 5.0])
    labels = np.array(
        [
            int(OrbitClass.BOX),
            int(OrbitClass.BOX),
            int(OrbitClass.SHORT_AXIS_TUBE),
            int(OrbitClass.BOX),
            int(OrbitClass.SHORT_AXIS_TUBE),
            int(OrbitClass.SHORT_AXIS_TUBE),
        ]
    )
    zeros3 = np.zeros((n, 3))
    cl = OrbitClassification(
        labels=labels,
        circulation=zeros3,
        tube_axis=np.zeros(n, int),
        planarity=np.zeros(n),
        resonance=np.zeros((n, 3), int),
        resonance_order=np.zeros(n, int),
        radius=radius,
    )
    edges = np.array([0.0, 1.0, 6.0])  # bin 0: 3 orbits, bin 1: 3 orbits

    ax = cl.plot_class_fractions(edges, per_bin=True)
    import matplotlib.axes

    assert isinstance(ax, matplotlib.axes.Axes)

    # Recover the plotted curves keyed by class name (the legend label).
    curves = {ln.get_label(): ln.get_ydata() for ln in ax.get_lines()}
    # Bin 0: 2 box / 1 tube of 3; bin 1: 1 box / 2 tube of 3.
    assert np.allclose(curves["box"], [2 / 3, 1 / 3])
    assert np.allclose(curves["short_axis_tube"], [1 / 3, 2 / 3])
    # Per-bin fractions across classes sum to 1 in each populated bin.
    assert np.allclose(sum(curves.values()), [1.0, 1.0])

    # Global normalisation: each count divided by the total (6).
    ax2 = cl.plot_class_fractions(edges, per_bin=False)
    curves2 = {ln.get_label(): ln.get_ydata() for ln in ax2.get_lines()}
    assert np.allclose(curves2["box"], [2 / 6, 1 / 6])
    assert np.allclose(curves2["short_axis_tube"], [1 / 6, 2 / 6])

    # Works on a condensed classification too (radius carried through).
    fam = cl.condense_families()
    ax3 = fam.plot_class_fractions(edges, per_bin=True)
    fam_curves = {ln.get_label(): ln.get_ydata() for ln in ax3.get_lines()}
    assert set(fam_curves) == {"box", "tube"}
    print("plot_class_fractions OK")


def test_plot_class_histograms():
    """plot_class_histograms bars the per-class counts with name x-labels."""
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.axes

    from lanfear import OrbitClassification

    labels = np.array(
        [int(OrbitClass.BOX)] * 3
        + [int(OrbitClass.SHORT_AXIS_TUBE)] * 2
        + [int(OrbitClass.ROSETTE)]
    )
    n = len(labels)
    cl = OrbitClassification(
        labels=labels,
        circulation=np.zeros((n, 3)),
        tube_axis=np.zeros(n, int),
        planarity=np.zeros(n),
        resonance=np.zeros((n, 3), int),
        resonance_order=np.zeros(n, int),
    )
    counts = cl.counts()
    ax = cl.plot_class_histograms()
    assert isinstance(ax, matplotlib.axes.Axes)

    # One bar per populated class, height == count, labelled by class name.
    bars = ax.patches
    assert len(bars) == len(counts)
    xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
    assert xtick_labels == list(counts.keys())
    heights = {name: bar.get_height() for name, bar in zip(xtick_labels, bars)}
    assert heights == {name: float(c) for name, c in counts.items()}
    print(f"plot_class_histograms OK: {counts}")


if __name__ == "__main__":
    print("== resonance finder ==")
    test_resonance_finder()
    print("== known orbits ==")
    test_known_orbits()
    print("== population ==")
    test_population()
    print("== condense families ==")
    test_condense_families()
    print("== plot class fractions ==")
    test_plot_class_fractions()
    print("== plot class histograms ==")
    test_plot_class_histograms()
    print("\nALL CLASSIFICATION TESTS PASSED")
