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
from lanfear.classify import _find_resonances, _latex_label  # noqa: E402

# Figures produced by the tests are written here (git-ignored, created on demand).
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def _save_figure(fig, name):
    """Save a test figure to tests/figures/<name>.png at 300 dpi.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    name : str
        Base filename (without extension).

    Returns
    -------
    path : str
        Full path the figure was written to.
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, name + ".png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return path


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
        length_unit=1.0,
        n_periods=40,
        n_samples=4096,
        initial_radius=np.array([np.linalg.norm(np.asarray(state)[:3])]),
        fundamentals=fund,
        lines=lines,
    )
    return res.classify()


def _vcirc(scf, x, y, z):
    a = np.array(scf.acceleration(x, y, z))
    rr = np.array([x, y, z])
    nr = np.linalg.norm(rr)
    return np.sqrt(max(-a @ rr / nr, 1e-6) * nr)


def test_irregular_detection():
    """_detect_irregular flags a 4th independent base frequency (Frigo 2021)."""
    from lanfear.classify import _detect_irregular, _lattice_vectors

    def make_lines(freqs_amps):
        """(3, nl) list of per-axis [(freq, amp), ...] -> (1,3,nl,2) array."""
        nl = max(len(a) for a in freqs_amps)
        out = np.zeros((1, 3, nl, 2))
        for a, rows in enumerate(freqs_amps):
            for k, (fr, am) in enumerate(rows):
                out[0, a, k] = (fr, am)
        return out

    # Three genuinely incommensurate fundamentals (independent over integers).
    fx, fy, fz = 0.2113, 0.3771, 0.5732
    fund = np.array([[fx, fy, fz]])

    # Regular orbit: every line is an exact integer combination of the three.
    regular = make_lines(
        [
            [(fx, 1.0), (2 * fx, 0.4), (fx + fy, 0.3), (fy - fx, 0.2)],
            [(fy, 1.0), (2 * fy, 0.4), (fx + fz, 0.3), (fz - fy, 0.2)],
            [(fz, 1.0), (2 * fz, 0.4), (fz - fx, 0.3), (fy + fz, 0.2)],
        ]
    )
    assert not _detect_irregular(fund, regular, 0.1, 0.02, 6)[0]

    # Place a strong 4th line in the largest gap of the 3-base combination
    # lattice, so it is provably not reducible to {fx, fy, fz}.
    combos = np.abs(_lattice_vectors(6) @ np.array([fx, fy, fz]))
    grid = np.unique(np.round(np.sort(combos), 6))
    grid = grid[(grid > 0.1) & (grid < 0.9)]
    k = int(np.argmax(np.diff(grid)))
    extra = 0.5 * (grid[k] + grid[k + 1])
    assert grid[k + 1] - grid[k] > 2 * 0.02 * fz  # comfortably outside tolerance

    irregular = make_lines(
        [
            [(fx, 1.0), (extra, 0.6), (2 * fx, 0.4), (fx + fy, 0.3)],
            [(fy, 1.0), (2 * fy, 0.4), (fx + fz, 0.3), (fz - fy, 0.2)],
            [(fz, 1.0), (2 * fz, 0.4), (fz - fx, 0.3), (fy + fz, 0.2)],
        ]
    )
    assert _detect_irregular(fund, irregular, 0.1, 0.02, 6)[0]

    # The same 4th line, but weak (below amp_frac), does not trigger irregular.
    weak = make_lines(
        [
            [(fx, 1.0), (extra, 0.03), (2 * fx, 0.4), (fx + fy, 0.3)],
            [(fy, 1.0), (2 * fy, 0.4), (fx + fz, 0.3), (fz - fy, 0.2)],
            [(fz, 1.0), (2 * fz, 0.4), (fz - fx, 0.3), (fy + fz, 0.2)],
        ]
    )
    assert not _detect_irregular(fund, weak, 0.1, 0.02, 6)[0]
    print("irregular detection OK")


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
    assert cl.labels[0] == OrbitClass.PIBOX, cl.names[0]
    print(f"  box      -> {cl.names[0]}")

    cl = _classify_state(tri, [2.0, 0, 0.6, 0, 0.6, 0.15])  # circulate about z
    assert cl.labels[0] == OrbitClass.SHORT_AXIS_TUBE, cl.names[0]
    assert cl.circulation[0, 2] > 0.9
    print(f"  z-tube   -> {cl.names[0]}")

    # Long-axis (x) tubes split inner/outer by morphology (Frigo et al. 2021,
    # after orbit-analysis): an inner tube is pinched at the waist (y-extent
    # peaks at the x-ends -> x_tube_ratio < 1), an outer tube is widest at the
    # centre (ratio >= 1). These two ICs are robust, reproducible examples.
    cl = _classify_state(tri, [-3.011, 0.766, 3.396, -0.192, -0.035, 0.203])
    assert cl.labels[0] == OrbitClass.INNER_LONG_AXIS_TUBE, cl.names[0]
    assert cl.circulation[0, 0] > 0.9
    print(f"  x-tube(in)  -> {cl.names[0]}")

    cl = _classify_state(tri, [1.727, -0.851, 3.367, -0.257, -0.015, -0.286])
    assert cl.labels[0] == OrbitClass.OUTER_LONG_AXIS_TUBE, cl.names[0]
    assert cl.circulation[0, 0] > 0.9
    print(f"  x-tube(out) -> {cl.names[0]}")

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
        length_unit=1.0,
        n_periods=30,
        n_samples=2048,
        initial_radius=np.linalg.norm(states[:, :3], axis=1),
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
    n_box = np.sum((cl.labels == OrbitClass.PIBOX) | (cl.labels == OrbitClass.BOXLET))
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
        "pibox": "box",
        "boxlet": "box",
        "short_axis_tube": "tube",
        "inner_long_axis_tube": "tube",
        "outer_long_axis_tube": "tube",
        "intermediate_axis_tube": "tube",
        "rosette": "tube",
        "irregular": "unclassified",  # neither box nor tube
    }
    for full_name, cond_name in zip(cl.names, cond.names):
        assert cond_name == expect[full_name], (full_name, cond_name)

    # unclassified (0) + irregular (8) both fold to the "unclassified" family.
    assert cond.counts() == {"unclassified": 2, "box": 2, "tube": 5}
    assert cond.mask(OrbitFamily.TUBE).sum() == 5
    assert cond.mask(OrbitFamily.BOX).sum() == 2
    # Diagnostic arrays are carried through, and re-condensing is a no-op.
    assert np.array_equal(cond.circulation, cl.circulation)
    assert cond.condense_families().counts() == cond.counts()


def test_get_class_ids():
    """get_class_ids() returns the particle IDs belonging to a family."""
    from lanfear import OrbitClassification, OrbitFamily

    labels = np.array(
        [
            int(OrbitClass.PIBOX),
            int(OrbitClass.SHORT_AXIS_TUBE),
            int(OrbitClass.PIBOX),
            int(OrbitClass.SHORT_AXIS_TUBE),
        ]
    )
    n = len(labels)
    zeros3 = np.zeros((n, 3))
    ids = np.array([10, 20, 30, 40])
    cl = OrbitClassification(
        labels=labels,
        circulation=zeros3,
        tube_axis=np.zeros(n, int),
        planarity=np.zeros(n),
        resonance=np.zeros((n, 3), int),
        resonance_order=np.zeros(n, int),
        ids=ids,
    )
    assert np.array_equal(cl.get_class_ids(OrbitClass.PIBOX), [10, 30])
    assert np.array_equal(cl.get_class_ids(OrbitClass.SHORT_AXIS_TUBE), [20, 40])
    assert cl.get_class_ids(OrbitClass.ROSETTE).size == 0

    # Carried through condense_families, and selects by condensed family too.
    cond = cl.condense_families()
    assert np.array_equal(cond.get_class_ids(OrbitFamily.BOX), [10, 30])
    assert np.array_equal(cond.get_class_ids(OrbitFamily.TUBE), [20, 40])

    # No IDs recorded -> a clear error rather than an AttributeError.
    cl_no_ids = OrbitClassification(
        labels=labels,
        circulation=zeros3,
        tube_axis=np.zeros(n, int),
        planarity=np.zeros(n),
        resonance=np.zeros((n, 3), int),
        resonance_order=np.zeros(n, int),
    )
    try:
        cl_no_ids.get_class_ids(OrbitClass.PIBOX)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when no IDs are recorded")
    print("get_class_ids OK")
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
            int(OrbitClass.PIBOX),
            int(OrbitClass.PIBOX),
            int(OrbitClass.SHORT_AXIS_TUBE),
            int(OrbitClass.PIBOX),
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

    # Recover the plotted curves keyed by legend label (the rendered LaTeX).
    curves = {ln.get_label(): ln.get_ydata() for ln in ax.get_lines()}
    # Bin 0: 2 pibox / 1 tube of 3; bin 1: 1 pibox / 2 tube of 3.
    assert np.allclose(curves[_latex_label("pibox")], [2 / 3, 1 / 3])
    assert np.allclose(curves[_latex_label("short_axis_tube")], [1 / 3, 2 / 3])
    # Per-bin fractions across classes sum to 1 in each populated bin.
    assert np.allclose(sum(curves.values()), [1.0, 1.0])

    # Global normalisation: each count divided by the total (6).
    ax2 = cl.plot_class_fractions(edges, per_bin=False)
    curves2 = {ln.get_label(): ln.get_ydata() for ln in ax2.get_lines()}
    assert np.allclose(curves2[_latex_label("pibox")], [2 / 6, 1 / 6])
    assert np.allclose(curves2[_latex_label("short_axis_tube")], [1 / 6, 2 / 6])

    # Works on a condensed classification too (radius carried through).
    fam = cl.condense_families()
    ax3 = fam.plot_class_fractions(edges, per_bin=True)
    fam_curves = {ln.get_label(): ln.get_ydata() for ln in ax3.get_lines()}
    assert set(fam_curves) == {_latex_label("box"), _latex_label("tube")}
    _save_figure(ax.figure, "class_fractions")
    print("plot_class_fractions OK")


def test_plot_class_histograms():
    """plot_class_histograms bars the per-class counts with name x-labels."""
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.axes

    from lanfear import OrbitClassification

    labels = np.array(
        [int(OrbitClass.PIBOX)] * 3
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

    # One bar per populated class, height == count, labelled by rendered LaTeX.
    bars = ax.patches
    assert len(bars) == len(counts)
    xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
    assert xtick_labels == [_latex_label(name) for name in counts]
    heights = {label: bar.get_height() for label, bar in zip(xtick_labels, bars)}
    assert heights == {_latex_label(name): float(c) for name, c in counts.items()}
    _save_figure(ax.figure, "class_histograms")
    print(f"plot_class_histograms OK: {counts}")


def test_plot_frequency_map():
    """plot_frequency_map scatters w-ratios, one collection per class, by class."""
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.axes

    from lanfear import OrbitClassification

    rng = np.random.default_rng(5)
    # Two classes with distinct frequency-ratio clouds.
    labels = np.array(
        [int(OrbitClass.PIBOX)] * 30 + [int(OrbitClass.SHORT_AXIS_TUBE)] * 20
    )
    n = len(labels)
    wz = np.ones(n)
    wx = np.concatenate([rng.normal(1.6, 0.05, 30), rng.normal(1.0, 0.05, 20)])
    wy = np.concatenate([rng.normal(1.3, 0.05, 30), rng.normal(1.0, 0.05, 20)])
    fundamentals = np.stack([wx, wy, wz], axis=1)
    cl = OrbitClassification(
        labels=labels,
        circulation=np.zeros((n, 3)),
        tube_axis=np.zeros(n, int),
        planarity=np.zeros(n),
        resonance=np.zeros((n, 3), int),
        resonance_order=np.zeros(n, int),
        fundamentals=fundamentals,
    )

    ax = cl.plot_frequency_map()
    assert isinstance(ax, matplotlib.axes.Axes)
    # One scatter collection per populated class, each carrying its class points.
    colls = ax.collections
    assert len(colls) == 2
    assert sum(c.get_offsets().shape[0] for c in colls) == n
    # Points sit at (wx/wz, wy/wz).
    offs = np.vstack([c.get_offsets() for c in colls])
    assert np.isclose(offs[:, 0].max(), wx.max(), atol=1e-6)
    _save_figure(ax.figure, "frequency_map")

    # Without frequency data it refuses rather than plotting nonsense.
    cl_nofreq = OrbitClassification(
        labels=labels,
        circulation=np.zeros((n, 3)),
        tube_axis=np.zeros(n, int),
        planarity=np.zeros(n),
        resonance=np.zeros((n, 3), int),
        resonance_order=np.zeros(n, int),
    )
    try:
        cl_nofreq.plot_frequency_map()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError without frequency data")
    print("plot_frequency_map OK")


def test_compare():
    """compare() matches particles by ID and reports family transitions."""
    from lanfear import OrbitClassification

    def make(ids, labels):
        ids = np.asarray(ids)
        labels = np.asarray(labels)
        n = len(ids)
        return OrbitClassification(
            labels=labels,
            circulation=np.zeros((n, 3)),
            tube_axis=np.zeros(n, int),
            planarity=np.zeros(n),
            resonance=np.zeros((n, 3), int),
            resonance_order=np.zeros(n, int),
            ids=ids,
        )

    # Particle 5 is only in "before", particle 6 only in "after" -> dropped.
    before = make(
        [1, 2, 3, 4, 5],
        [
            OrbitClass.PIBOX,
            OrbitClass.PIBOX,
            OrbitClass.SHORT_AXIS_TUBE,
            OrbitClass.ROSETTE,
            OrbitClass.PIBOX,
        ],
    )
    after = make(
        [4, 3, 2, 1, 6],  # deliberately out of order to exercise ID matching
        [
            OrbitClass.ROSETTE,  # id 4: rosette -> rosette (unchanged)
            OrbitClass.SHORT_AXIS_TUBE,  # id 3: unchanged
            OrbitClass.SHORT_AXIS_TUBE,  # id 2: box -> tube (changed)
            OrbitClass.PIBOX,  # id 1: unchanged
            OrbitClass.PIBOX,
        ],
    )

    cmp = before.compare(after)
    # Only ids {1,2,3,4} match, sorted.
    assert np.array_equal(cmp.ids, [1, 2, 3, 4])
    assert cmp.n_matched == 4
    # Only particle 2 changed family.
    assert np.array_equal(cmp.changed, [False, True, False, False])
    assert cmp.fraction_changed == 0.25

    # before/after labels are aligned to the matched, sorted IDs.
    assert np.array_equal(
        cmp.names_before, ["pibox", "pibox", "short_axis_tube", "rosette"]
    )
    assert np.array_equal(
        cmp.names_after, ["pibox", "short_axis_tube", "short_axis_tube", "rosette"]
    )

    rows, cols, matrix = cmp.transition_matrix()
    # Row "pibox" -> one stays pibox (id 1), one becomes short_axis_tube (id 2).
    box_row = matrix[rows.index("pibox")]
    assert box_row[cols.index("pibox")] == 1
    assert box_row[cols.index("short_axis_tube")] == 1
    assert matrix.sum() == cmp.n_matched

    # Works across the condense_families boundary (still ID-matched).
    fam_cmp = before.condense_families().compare(after.condense_families())
    assert fam_cmp.n_matched == 4
    assert set(fam_cmp.names_before) <= {"box", "tube", "unclassified"}

    # Missing IDs are an error.
    no_ids = make([1], [OrbitClass.PIBOX])
    no_ids.ids = None
    try:
        no_ids.compare(after)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when ids are missing")

    # Comparing a full classification against a condensed one is an error.
    try:
        before.compare(after.condense_families())
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched class schemes")

    # Sankey diagram: returns an Axes with one ribbon per non-zero transition.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.axes

    ax = cmp.plot_sankey()
    assert isinstance(ax, matplotlib.axes.Axes)
    _, _, matrix = cmp.transition_matrix()
    n_flows = int(np.count_nonzero(matrix))
    n_ribbons = sum(
        1 for p in ax.patches if isinstance(p, matplotlib.patches.PathPatch)
    )
    assert n_ribbons == n_flows
    _save_figure(ax.figure, "compare_sankey")
    print(f"compare OK: {cmp.n_matched} matched, {cmp.fraction_changed:.0%} changed")


if __name__ == "__main__":
    print("== irregular detection ==")
    test_irregular_detection()
    print("== resonance finder ==")
    test_resonance_finder()
    print("== known orbits ==")
    test_known_orbits()
    print("== population ==")
    test_population()
    print("== condense families ==")
    test_condense_families()
    print("== get class ids ==")
    test_get_class_ids()
    print("== plot class fractions ==")
    test_plot_class_fractions()
    print("== plot class histograms ==")
    test_plot_class_histograms()
    print("== plot frequency map ==")
    test_plot_frequency_map()
    print("== compare ==")
    test_compare()
    print("\nALL CLASSIFICATION TESTS PASSED")
