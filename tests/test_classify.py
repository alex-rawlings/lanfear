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
from lanfear.classify import (  # noqa: E402
    BoxletType,
    OrbitSense,
    OrbitShape,
    PatternResonance,
    Regularity,
    _BOX_CLASSES,
    _find_resonances,
    _named_resonance_vectors,
    _latex_label,
    _TUBE_CLASSES,
)

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
    summ, fund, lines, diff, planar = scf.analyse_batch(
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
        diffusion=diff,
        planar_lines=planar,
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
    vec, order, second, second_order = _find_resonances(w, max_order=5, tol=0.01)
    assert tuple(vec[0]) == (2, -1, 0) and order[0] == 3
    assert order[1] == 2  # some 1:1 among equal freqs
    assert second_order[1] == 2  # 1:1:1 has two independent 1:1 locks
    assert order[2] == 0 and second_order[2] == 0  # no low-order resonance

    # The order-7 pretzel (w_x:w_y = 3:4) is only found by the named search.
    pretzel = np.array([[0.75, 1.0, 0.613]])
    assert _find_resonances(pretzel, 5, 1e-3)[1][0] == 0
    vec, order, _, _ = _find_resonances(
        pretzel, 5, 1e-3, extra=_named_resonance_vectors(5)
    )
    assert tuple(vec[0]) == (4, -3, 0) and order[0] == 7
    named = _named_resonance_vectors(5)
    assert len(named) == 6 and (0, 3, -4) in named  # 3 planes x 2 orientations
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

    # Long-axis (x) tubes split inner/outer by where they are widest: an outer
    # tube at its waist (x_waist_ratio ~1), an inner one away from it. These
    # two ICs are robust, reproducible examples.
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


def test_known_boxlets():
    """Pinned boxlets (released from rest) in the (1, 0.75, 0.5) model.

    Found by searching 6000 rest-start box orbits; each sits on its resonance
    to far better than the 1e-3 tolerance, except the fish (4e-4).
    """
    tri = build_scf(flatten=(1.0, 0.75, 0.5))
    cases = [
        ("banana_xz", [-1.532, 0.066, 0.301], BoxletType.BANANA, 1, 0),
        ("fish_xy", [2.624, -1.257, -0.158], BoxletType.FISH, 2, 0),
        ("pretzel_xy", [-1.396, -2.44, -0.191], BoxletType.PRETZEL, 2, 0),
    ]
    for name, position, kind, plane, elongation in cases:
        cl = _classify_state(tri, [*position, 0.0, 0.0, 0.0])
        assert cl.labels[0] == OrbitClass.BOXLET, (name, cl.names[0])
        assert cl.boxlet_type[0] == kind and cl.boxlet_names[0] == name
        assert cl.boxlet_plane[0] == plane and cl.boxlet_elongation[0] == elongation
        assert not cl.boxlet_closed[0]
        print(f"  {name:10s} -> {cl.boxlet_names[0]} (resonance {cl.resonance[0]})")
    # Two independent resonances: a closed (periodic) boxlet.
    cl = _classify_state(tri, [-2.582, -0.041, -0.755, 0.0, 0.0, 0.0])
    assert cl.labels[0] == OrbitClass.BOXLET and cl.boxlet_closed[0]
    assert cl.boxlet_names[0] == "1:1_xy"
    assert tuple(cl.second_resonance[0]) == (0, 2, -1)
    print("known boxlets OK")


def test_resonance_false_positives():
    """Null test: chance commensurabilities of random frequency triples.

    Uniform random triples have no resonances, so every hit is a false
    positive. At the default tolerance (1e-3, orders <= 5 plus the named
    pretzel) about 3% are hit, against ~28% at the old 1e-2.
    """
    rng = np.random.default_rng(1)
    w = rng.uniform(0.2, 1.2, (20000, 3))
    extra = _named_resonance_vectors(5)
    rate = np.mean(_find_resonances(w, 5, 1e-3, extra=extra)[1] >= 2)
    loose = np.mean(_find_resonances(w, 5, 1e-2, extra=extra)[1] >= 2)
    assert rate < 0.05 and loose > 0.2, (rate, loose)
    print(f"resonance false positives: {rate:.3f} at 1e-3, {loose:.3f} at 1e-2")


def test_boxlet_description():
    """Boxlet types, names, planes, masks, counts and the frequency map."""
    n = 5
    w = np.array(
        [
            [0.5, 1.0, 0.7313],  # banana x-y (2, -1, 0), elongated along x
            [0.9, 0.5171, 0.45],  # banana x-z (1, 0, -2), elongated along z
            [0.4, 0.55, 0.7],  # spatial (1, -2, 1)
            [0.6, 0.6, 0.8813],  # 1:1 in x-y: planar, unnamed
            [0.2913, 0.4721, 0.6337],  # no resonance: pi-box
        ]
    )
    summary = np.zeros((n, len(SUMMARY_COLUMNS)))
    for name in ("Lx_abs_mean", "Ly_abs_mean", "Lz_abs_mean"):
        summary[:, SUMMARY_COLUMNS.index(name)] = 1.0
    summary[:, SUMMARY_COLUMNS.index("r_max")] = 3.0
    lines = np.zeros((n, 3, 4, 2))
    lines[:, :, 0, 0] = w
    lines[:, :, 0, 1] = 1.0
    res = OrbitResults(
        ids=np.arange(10, 10 + n),
        summary=summary,
        columns=SUMMARY_COLUMNS,
        time_unit=1.0,
        length_unit=1.0,
        n_periods=40,
        n_samples=4096,
        initial_radius=np.full(n, 2.0),
        fundamentals=w,
        lines=lines,
        diffusion=np.zeros((n, 3)),
        planar_lines=np.zeros((n, 3, 4, 2)),
    )
    cl = res.classify()
    expected = [
        BoxletType.BANANA,
        BoxletType.BANANA,
        BoxletType.SPATIAL,
        BoxletType.PLANAR,
        BoxletType.NONE,
    ]
    np.testing.assert_array_equal(cl.boxlet_type, expected)
    assert list(cl.boxlet_names) == [
        "banana_xy",
        "banana_xz",
        "resonant_1_-2_1",
        "1:1_xy",
        "",
    ]
    np.testing.assert_array_equal(cl.boxlet_plane, [2, 1, -1, 2, -1])
    np.testing.assert_array_equal(cl.boxlet_elongation, [0, 2, 0, 0, -1])
    assert cl.labels[4] == OrbitClass.PIBOX

    assert np.array_equal(cl.mask("banana"), cl.mask(BoxletType.BANANA))
    assert cl.mask("banana").sum() == 2 and cl.mask("banana_xz").sum() == 1
    assert cl.mask("boxlet").sum() == 4  # a family name still selects the family
    np.testing.assert_array_equal(cl.get_class_ids("spatial"), [12])
    assert cl.boxlet_counts() == {"banana": 2, "planar": 1, "spatial": 1}
    assert cl.boxlet_counts(detailed=True)["banana_xz"] == 1
    try:
        cl.mask("croissant")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown name should raise")
    ax = cl.plot_frequency_map(by="boxlet", resonance_lines=True)
    _save_figure(ax.figure, "frequency_map_boxlets")
    import matplotlib.pyplot as plt

    plt.close(ax.figure)
    print("boxlet description checks passed")


def test_y_tube_resonance_corroboration():
    """A candidate y-tube is only accepted if its x/z fundamentals actually
    lock 1:1 -- Carpintero & Aguilar's own frequency-domain definition of a
    y-tube -- and otherwise falls back to a boxlet or plain box. All three
    orbits here show the same spurious, finite-window circ_y (high L_y
    persistence, low L_x/L_z) that would previously have been labelled
    INTERMEDIATE_AXIS_TUBE regardless of the fundamentals:

    * orbit 0: w_x == w_z (locked) -> a genuine, corroborated y-tube.
    * orbit 1: w_x, w_z unlocked, but the triple sits exactly on the (2,-1,0)
      "banana" commensurability used in test_resonance_finder -- the
      signature of a resonant box orbit elongated along y (Merritt & Valluri
      1999; Poon & Merritt 2001) -- relabelled BOXLET.
    * orbit 2: w_x, w_z unlocked and the triple is genuinely incommensurate
      (no boxlet evidence either) -- neither confirmation holds, so it falls
      back to an unconfirmed PIBOX rather than a tube.
    """
    n = 3
    summary = np.zeros((n, len(SUMMARY_COLUMNS)))

    def set_col(name, values):
        summary[:, SUMMARY_COLUMNS.index(name)] = values

    set_col("status", [0, 0, 0])
    set_col("Lx_mean", [0.05, 0.05, 0.05])
    set_col("Ly_mean", [0.95, 0.95, 0.95])
    set_col("Lz_mean", [0.10, 0.10, 0.10])
    set_col("Lx_abs_mean", [1.0, 1.0, 1.0])
    set_col("Ly_abs_mean", [1.0, 1.0, 1.0])
    set_col("Lz_abs_mean", [1.0, 1.0, 1.0])
    set_col("r_mean", [2.0, 2.0, 2.0])

    fundamentals = np.array(
        [
            [0.500, 0.790, 0.500],  # w_x == w_z -> locked, genuine y-tube
            [0.40, 0.80, 0.137],  # 2 w_x - w_y = 0, unlocked -> resonant box
            [0.317, 0.482, 0.613],  # incommensurate, unlocked -> plain box
        ]
    )

    res = OrbitResults(
        ids=np.array([0, 1, 2]),
        summary=summary,
        columns=SUMMARY_COLUMNS,
        time_unit=1.0,
        length_unit=1.0,
        n_periods=40,
        n_samples=4096,
        initial_radius=np.array([2.0, 2.0, 2.0]),
        fundamentals=fundamentals,
        lines=np.ones((3, 3, 4, 2)),  # dummy: equal-amplitude, no irregularity
        diffusion=np.zeros((3, 3)),
        planar_lines=np.zeros((3, 3, 4, 2)),
    )
    cl = res.classify()
    assert cl.labels[0] == OrbitClass.INTERMEDIATE_AXIS_TUBE, cl.names[0]
    assert cl.circulation[0, 1] > 0.7
    assert cl.labels[1] == OrbitClass.BOXLET, cl.names[1]
    assert cl.labels[2] == OrbitClass.PIBOX, cl.names[2]
    print("y-tube resonance corroboration OK")


def test_population():
    """A triaxial population classifies fully and yields a sensible family mix."""
    tri = build_scf(flatten=(1.0, 0.8, 0.6), n=20_000)
    rng = np.random.default_rng(11)
    n = 120
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

    summ, fund, lines, diff, planar = tri.analyse_batch(
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
        diffusion=diff,
        planar_lines=planar,
    )
    cl = res.classify()
    ok = res.ok
    # Every successfully integrated orbit gets a (non-UNCLASSIFIED) family.
    assert np.all(cl.labels[ok] != OrbitClass.UNCLASSIFIED)
    counts = cl.counts()
    print(f"  {int(ok.sum())} orbits classified; families: {counts}")
    # A triaxial potential should host both boxes and tubes.
    n_box = np.sum(np.isin(cl.labels, _BOX_CLASSES))
    n_tube = np.sum(np.isin(cl.labels, _TUBE_CLASSES))
    assert n_box > 0 and n_tube > 0
    # The family label is IRREGULAR exactly when the regularity is not regular.
    np.testing.assert_array_equal(
        cl.labels[ok] == OrbitClass.IRREGULAR,
        np.isin(cl.regularity[ok], [Regularity.IRREGULAR, Regularity.DIFFUSING]),
    )
    assert np.all(cl.regularity[~ok] == Regularity.UNKNOWN)
    # The shape follows the regular family label.
    regular = ok & (cl.labels != OrbitClass.IRREGULAR)
    boxes = np.isin(cl.labels, _BOX_CLASSES)
    assert np.all(cl.shape[regular & boxes] == OrbitShape.BOX)
    assert np.all(cl.shape[regular & ~boxes] != OrbitShape.BOX)
    print("population classification OK")

    # classify_orbits records energy / angular momentum for fractions_by.
    assert np.allclose(
        cl.quantities["energy"], summ[:, SUMMARY_COLUMNS.index("energy0")]
    )
    assert np.all(cl.quantities["angular_momentum"][ok] >= 0)
    edges = np.quantile(cl.quantities["energy"][ok], np.linspace(0, 1, 4))
    edges[-1] += 1e-9
    assert cl.fractions_by(edges, quantity="energy").counts.sum() == ok.sum()

    # Frequency diffusion: orbits the spectral criterion calls irregular drift
    # far more than the rest, and the default cut flags high-diffusion orbits.
    rate = res.diffusion_rate()
    cl_off = res.classify(diffusion_threshold=None)
    irr = cl_off.labels == OrbitClass.IRREGULAR
    assert irr.any() and (~irr).any()
    assert np.nanmedian(rate[irr]) > 2 * np.nanmedian(rate[~irr])
    # The cut is on by default (0.1); None disables it.
    cut = 0.1
    flagged = ok & (rate > cut)
    assert flagged.any()
    assert np.all(cl.labels[flagged] == OrbitClass.IRREGULAR)
    assert np.array_equal(cl.labels[~flagged], cl_off.labels[~flagged])
    assert np.sum(cl.labels == OrbitClass.IRREGULAR) >= np.sum(
        cl_off.labels == OrbitClass.IRREGULAR
    )
    print("frequency diffusion classification OK")


def test_diffusion_drop_rule():
    """Extended orbits are chaotic only if their drift failed to fall."""
    n = 6
    summary = np.zeros((n, len(SUMMARY_COLUMNS)))
    col = {name: i for i, name in enumerate(SUMMARY_COLUMNS)}
    summary[:, col["Lz_abs_mean"]] = 1.0
    summary[:, col["Lz_mean"]] = 1.0  # z-circulating for all
    summary[:, col["Lx_abs_mean"]] = 1.0
    summary[:, col["Ly_abs_mean"]] = 1.0
    summary[:, col["x_waist_ratio"]] = 1.0
    freq = np.tile([[1.0, 1.0, 2.0]], (n, 1))  # 1:1:2, not 1:1:1
    lines = np.zeros((n, 3, 2, 2))
    lines[:, :, 0, 0] = freq
    lines[:, :, 0, 1] = 1.0
    # rate now:      [low, high, high, high, high, high]
    # rate before:   [nan, nan,  0.4,  0.9,  1.0,  nan]
    rate = np.array([1e-4, 0.5, 0.05, 0.6, 0.5, 0.05])
    prev = np.array([np.nan, np.nan, 0.4, 0.9, 1.0, 0.3])
    diffusion = np.tile(rate[:, None], (1, 3))
    res = OrbitResults(
        ids=np.arange(n),
        summary=summary,
        columns=SUMMARY_COLUMNS,
        time_unit=1.0,
        length_unit=1.0,
        n_periods=10,
        n_samples=100,
        initial_radius=np.ones(n),
        fundamentals=freq,
        lines=lines,
        diffusion=diffusion,
        planar_lines=np.zeros_like(lines),
    )
    irr = OrbitClass.IRREGULAR
    # No extension data: the plain threshold cut (0.1).
    assert list(res.classify().labels == irr) == [False, True, False, True, True, False]
    # With previous rates (NaN = never extended):
    #  0: low -> regular          1: never extended, high -> irregular (plain cut)
    #  2: 0.05 < thr -> regular   3: 0.6 vs 0.9*0.5=0.45 not falling -> irregular
    #  4: 0.5 vs 0.5*1.0 not fallen enough -> irregular
    #  5: 0.05 below the threshold -> regular
    res.diffusion_previous = prev
    got = res.classify().labels == irr
    assert list(got) == [False, True, False, True, True, False], got
    # A high orbit whose drift fell by more than the drop factor stays regular.
    res.diffusion_previous = np.array([np.nan, np.nan, 0.4, 5.0, 5.0, 0.3])
    got = res.classify().labels == irr
    assert list(got) == [False, True, False, False, False, False], got
    # Raising the drop factor demands a larger fall before calling it chaotic.
    res.diffusion_previous = np.array([np.nan, np.nan, 0.4, 1.0, 1.0, 0.3])
    assert (res.classify(diffusion_drop=0.5).labels == irr)[3]
    assert not (res.classify(diffusion_drop=0.9).labels == irr)[3]
    print("diffusion drop rule OK")


def _rotating_case():
    """Hand-built rotating-figure results exercising each frame-aware rule.

    The figure tumbles about z at Omega = 0.05 (HO units, window 2000):

    0. A z-loop (circ_z = 1) with a hole about z, winding at 0.3: a
       short-axis tube.
    1. A box precessing steadily about z: circ_z = 0.95, but it passes the
       axis (hole 0.01): a box (a short-axis tube without the hole cut, or
       when static).
    2. circ_z = 0.2 but winding steadily about z at 0.3: a box (circulation
       decides, not winding).
    3. A z-loop with body-frame x/y fundamentals 0.45 and z 0.5: in the
       inertial frame its in-plane frequency is 0.45 + 0.05 = 0.5, so 1:1:1, a
       rosette; static, it is a short-axis tube.
    4. Pattern-locked: x/y at |Omega|, z silent, winding backwards at
       -Omega in the body frame. Not a rosette (unless the lock check is off).
    """
    from dataclasses import replace

    omega, t_total = 0.05, 2000.0
    n = 5
    summary = np.zeros((n, len(SUMMARY_COLUMNS)))

    def set_col(name, values):
        summary[:, SUMMARY_COLUMNS.index(name)] = values

    set_col("period", [2 * np.pi] * n)
    set_col("t_total", [t_total] * n)
    set_col("r_mean", [2.0] * n)
    set_col("r_max", [3.0] * n)
    set_col("x_waist_ratio", [1.0] * n)
    set_col("Lx_abs_mean", [1.0] * n)
    set_col("Ly_abs_mean", [1.0] * n)
    set_col("Lz_abs_mean", [1.0] * n)
    set_col("Lz_mean", [1.0, 0.95, 0.2, 1.0, -1.0])
    set_col("rho_z_min", [0.6, 0.02, 0.6, 0.6, 0.6])  # holes 0.3, 0.01 r_mean
    locked = -omega * (1 + 1e-5)
    rates = np.array([0.3, 0.03, 0.3, 0.45, locked])
    set_col("phi_z_net", rates * t_total)
    set_col("phi_z_range", np.abs(rates) * t_total)
    set_col("phi_x_range", [1.0] * n)
    set_col("phi_y_range", [1.0] * n)
    set_col("radial_frequency", [0.5, 0.5, 0.005, 0.7, 0.7])

    incommensurate = [0.2913, 0.4721, 0.6337]
    fundamentals = np.array(
        [
            incommensurate,
            incommensurate,
            incommensurate,
            [0.45, 0.45, 0.5],
            [omega, omega, 0.37],
        ]
    )
    amplitudes = np.ones((n, 3))
    amplitudes[4, 2] = 0.01  # silent z
    lines = np.zeros((n, 3, 4, 2))
    lines[:, :, 0, 0] = fundamentals
    lines[:, :, 0, 1] = amplitudes
    planar = np.zeros((n, 3, 4, 2))
    planar[:, 2, 0, 0] = rates
    planar[:, 2, 0, 1] = 1.0
    res = OrbitResults(
        ids=np.arange(n),
        summary=summary,
        columns=SUMMARY_COLUMNS,
        time_unit=1.0,
        length_unit=1.0,
        n_periods=40,
        n_samples=4096,
        initial_radius=np.full(n, 2.0),
        fundamentals=fundamentals,
        lines=lines,
        diffusion=np.zeros((n, 3)),
        planar_lines=planar,
        pattern_speed=np.array([0.0, 0.0, omega]),
    )
    return res, replace(res, pattern_speed=np.zeros(3))


def test_rotating_hole_and_rosette():
    """Hole cut, frame-corrected rosette test and pattern-lock exclusion."""
    res, static = _rotating_case()
    cl = res.classify(diffusion_threshold=None)
    assert cl.labels[0] == OrbitClass.SHORT_AXIS_TUBE, cl.names
    assert cl.tube_axis[0] == 2
    assert cl.shape[1] == OrbitShape.BOX and cl.shape[2] == OrbitShape.BOX
    assert cl.labels[3] == OrbitClass.ROSETTE, cl.names
    assert cl.labels[4] == OrbitClass.SHORT_AXIS_TUBE, cl.names  # locked
    unlocked = res.classify(diffusion_threshold=None, pattern_lock_tol=None)
    assert unlocked.labels[4] == OrbitClass.ROSETTE

    no_hole = res.classify(diffusion_threshold=None, hole_thresh=0.0)
    assert no_hole.shape[1] == OrbitShape.SHORT_AXIS_TUBE

    # Static: no hole cut, no frame correction, no lock check.
    cs = static.classify(diffusion_threshold=None)
    assert cs.shape[1] == OrbitShape.SHORT_AXIS_TUBE and cs.shape[2] == OrbitShape.BOX
    assert cs.labels[3] == OrbitClass.SHORT_AXIS_TUBE  # 0.45 vs 0.5: not 1:1:1
    assert cs.labels[4] == OrbitClass.ROSETTE  # the 1:1 x/y lock alone
    print("rotating hole / rosette / pattern-lock checks passed")


def test_rotating_shared_line_not_boxlet():
    """A box whose in-plane fundamentals are one shared line is no 1:1 boxlet."""
    res, static = _rotating_case()
    for r in (res, static):
        r.fundamentals[2, :2] = 0.4  # x and y led by the same line
        r.lines[2, :2, 0, 0] = 0.4
    cs = static.classify(diffusion_threshold=None)
    assert cs.labels[2] == OrbitClass.BOXLET, cs.names  # static: a genuine 1:1
    cl = res.classify(diffusion_threshold=None)
    assert cl.labels[2] == OrbitClass.PIBOX, cl.names  # rotating: the precession
    assert cl.boxlet_type[2] == BoxletType.NONE
    # Apart by more than shared_line_tol resolutions: tested as a resonance.
    off = res.classify(diffusion_threshold=None, shared_line_tol=0.0)
    assert off.labels[2] == OrbitClass.BOXLET
    print("shared-line boxlet check passed")


def test_sense_labels():
    """Prograde/retrograde loops, and locked/librating orbits when rotating."""
    tri = build_scf(flatten=(1.0, 0.75, 0.5))
    for sign, expected in ((1.0, OrbitSense.PROGRADE), (-1.0, OrbitSense.RETROGRADE)):
        state = np.array([2.0, 0.0, 0.6, 0.0, 0.6, 0.15])
        state[3:] *= sign
        cl = _classify_state(tri, state)
        assert cl.labels[0] == OrbitClass.SHORT_AXIS_TUBE and cl.sense[0] == expected
    # A long-axis tube's sense is about x, with the sign of its L_x.
    state = [-3.011, 0.766, 3.396, -0.192, -0.035, 0.203]
    cl = _classify_state(tri, state)
    assert cl.tube_axis[0] == 0 and cl.sense[0] == OrbitSense.PROGRADE
    cl = _classify_state(tri, [*state[:3], *(-np.array(state[3:]))])
    assert cl.sense[0] == OrbitSense.RETROGRADE
    assert _classify_state(tri, [1.5, 1.0, 0.7, 0, 0, 0]).sense[0] == OrbitSense.NONE

    res, static = _rotating_case()
    col = SUMMARY_COLUMNS.index
    # Orbit 1 becomes an orbit librating around z: a hole, < 2 pi of azimuth.
    for r in (res, static):
        r.summary[1, col("Lz_mean")] = 0.2
        r.summary[1, col("phi_z_net")] = 1.0
        r.summary[1, col("phi_z_range")] = 3.0
        r.summary[1, col("rho_z_min")] = 0.5
    cl = res.classify(diffusion_threshold=None)
    assert list(cl.sense_names) == [
        "prograde",  # short-axis tube, winding forwards
        "librating",
        "none",  # box
        "prograde",  # rosette, inertial 0.5 > 0
        "locked",
    ], cl.sense_names
    cs = static.classify(diffusion_threshold=None)
    assert not np.isin(cs.sense, [OrbitSense.LOCKED, OrbitSense.LIBRATING]).any()
    assert cs.sense[4] == OrbitSense.RETROGRADE  # static: just a backwards loop
    print("sense label checks passed")


def _synthetic_loops(omega, specs, t_total=2000.0):
    """Hand-built results of orbits going round z in a figure tumbling at omega.

    ``specs`` holds one ``(omega_body, omega_r, sxx, syy, circulating)`` per
    orbit: the body-frame azimuthal frequency about z, the radial frequency,
    the in-plane second moments and whether it is a loop (``circ_z = 1``) or a
    librating non-loop (hole about z, azimuth spanning 3 rad).
    """
    n = len(specs)
    summary = np.zeros((n, len(SUMMARY_COLUMNS)))

    def set_col(name, values):
        summary[:, SUMMARY_COLUMNS.index(name)] = values

    omega_body = np.array([s[0] for s in specs])
    loop = np.array([s[4] for s in specs])
    set_col("t_total", [t_total] * n)
    set_col("period", [2 * np.pi] * n)
    set_col("r_mean", [2.0] * n)
    set_col("r_max", [3.0] * n)
    set_col("x_waist_ratio", [1.0] * n)
    set_col("Lx_abs_mean", [1.0] * n)
    set_col("Ly_abs_mean", [1.0] * n)
    set_col("Lz_abs_mean", [1.0] * n)
    set_col("Lz_mean", np.where(loop, np.sign(omega_body), 0.0))
    set_col("phi_z_net", np.where(loop, omega_body * t_total, 1.0))
    set_col("phi_z_range", np.where(loop, np.abs(omega_body) * t_total, 3.0))
    set_col("phi_x_range", [1.0] * n)
    set_col("phi_y_range", [1.0] * n)
    set_col("rho_z_min", np.where(loop, 0.6, 0.5))  # loops keep clear of z
    set_col("radial_frequency", [s[1] for s in specs])
    set_col("Sxx", [s[2] for s in specs])
    set_col("Syy", [s[3] for s in specs])
    fundamentals = np.tile([0.2913, 0.4721, 0.6337], (n, 1))  # no 1:1:1, no resonance
    lines = np.zeros((n, 3, 4, 2))
    lines[:, :, 0, 0] = fundamentals
    lines[:, :, 0, 1] = 1.0
    planar = np.zeros((n, 3, 4, 2))
    planar[:, 2, 0, 0] = np.where(loop, omega_body, 1.0 / t_total)
    planar[:, 2, 0, 1] = 1.0
    return OrbitResults(
        ids=np.arange(n),
        summary=summary,
        columns=SUMMARY_COLUMNS,
        time_unit=1.0,
        length_unit=1.0,
        n_periods=40,
        n_samples=4096,
        initial_radius=np.full(n, 2.0),
        fundamentals=fundamentals,
        lines=lines,
        diffusion=np.zeros((n, 3)),
        planar_lines=planar,
        pattern_speed=np.array([0.0, 0.0, omega]),
    )


def test_pattern_resonance_synthetic():
    """Each pattern-resonance tag from a hand-built orbit (Omega_p = 0.05)."""
    from dataclasses import replace

    omega = 0.05
    specs = [
        (0.3, 0.6, 2.0, 1.0, True),  # 2:1 prograde, along x: x1
        (0.3, 0.6, 1.0, 2.0, True),  # 2:1 prograde, across x: x2
        (-0.4, 0.8, 1.0, 1.0, True),  # 2:1 retrograde, inertial -0.35: x4
        (-0.03, 0.06, 1.0, 1.0, True),  # 2:1 retrograde, inertial +0.02: OLR
        (0.2, 0.8, 1.0, 1.0, True),  # 4:1
        (0.3, 0.5, 1.0, 1.0, True),  # no resonance
        (0.0005, 0.6, 1.0, 1.0, True),  # loop at corotation (< 1 resolution)
        (0.3, 0.005, 1.0, 1.0, True),  # radial frequency unresolved
        (0.0, 0.6, 1.0, 1.0, False),  # librating at corotation: trapped
    ]
    res = _synthetic_loops(omega, specs)
    cl = res.classify(diffusion_threshold=None)
    assert list(cl.pattern_resonance_names) == [
        "x1",
        "x2",
        "x4",
        "olr",
        "ultraharmonic",
        "none",
        "corotation",
        "unresolved",
        "corotation",
    ], cl.pattern_resonance_names
    np.testing.assert_array_equal(cl.corotation_trapped, [False] * 8 + [True])
    # A pattern too slow to complete 3 cycles in the window cannot define
    # corotation: those orbits are unresolved instead.
    slow = replace(res, pattern_speed=np.array([0.0, 0.0, 0.005]))
    slow_cl = slow.classify(diffusion_threshold=None)
    assert slow_cl.pattern_resonance[8] == PatternResonance.UNRESOLVED
    assert not slow_cl.corotation_trapped.any()
    assert not any(slow_cl.inconsistencies().values())
    assert np.isclose(cl.pattern_frequency_ratio[0], 2.0)
    assert np.isnan(cl.pattern_frequency_ratio[6])  # corotation: not tested

    static = replace(res, pattern_speed=np.zeros(3)).classify(diffusion_threshold=None)
    assert np.all(static.pattern_resonance == PatternResonance.NONE)
    assert not static.corotation_trapped.any()
    print("synthetic pattern-resonance checks passed")


def test_family_table_and_views():
    """Named families, scheme views, custom tables and consistency checks."""
    from dataclasses import replace
    from lanfear import FamilyRule, assign_families

    omega = 0.05
    specs = [
        (0.3, 0.6, 2.0, 1.0, True),  # x1
        (0.3, 0.6, 1.0, 2.0, True),  # x2
        (-0.4, 0.8, 1.0, 1.0, True),  # x4
        (0.3, 0.5, 1.0, 1.0, True),  # plain prograde z-tube
        (-0.3, 0.5, 1.0, 1.0, True),  # plain retrograde z-tube
        (0.0, 0.6, 1.0, 1.0, False),  # trapped at L4/L5
    ]
    res = _synthetic_loops(omega, specs)
    cl = res.classify(diffusion_threshold=None)
    assert list(cl.family_name_array) == [
        "x1",
        "x2",
        "x4",
        "prograde_short_axis_tube",
        "retrograde_short_axis_tube",
        "corotation_trapped",
    ], cl.family_name_array
    # The class label is the shape-level view of the same orbits.
    assert list(cl.names[:5]) == ["short_axis_tube"] * 5
    assert cl.counts("family")["x1"] == 1
    assert cl.view("family").counts() == cl.counts("family")
    np.testing.assert_array_equal(cl.mask("x4", scheme="family"), [0, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(cl.get_class_ids("x2", scheme="family"), [1])
    assert cl.counts("sense")["prograde"] == 3
    assert cl.view() is cl
    for bad in ("nonsense",):
        try:
            cl.view(bad)
        except ValueError:
            continue
        raise AssertionError("an unknown scheme should raise")
    try:
        cl.view("family").condense_families()
    except ValueError:
        pass
    else:
        raise AssertionError("condensing a view should raise")

    # Comparisons work on any scheme, but only between equally oriented results.
    comparison = cl.compare(cl, scheme="family")
    assert comparison.fraction_changed == 0.0
    try:
        cl.compare(replace(cl, orientation="spin"))
    except ValueError:
        pass
    else:
        raise AssertionError("comparing differently oriented results should raise")

    # A custom table: first match wins, unmatched orbits are "other".
    table = (
        FamilyRule(
            "bar_supporting", resonance=(PatternResonance.X1, PatternResonance.X2)
        ),
        FamilyRule("loop", shape=(OrbitShape.SHORT_AXIS_TUBE,)),
    )
    custom = res.classify(diffusion_threshold=None, family_table=table)
    assert list(custom.family_name_array) == ["bar_supporting"] * 2 + ["loop"] * 3 + [
        "other"
    ]
    for bad in ((FamilyRule("a"), FamilyRule("a")), (FamilyRule("other"),)):
        try:
            assign_families(
                {
                    k: np.zeros(1, int)
                    for k in ("shape", "boxlet", "sense", "resonance", "regularity")
                },
                bad,
            )
        except ValueError:
            continue
        raise AssertionError(f"{bad} should be rejected")

    # Consistency checks flag impossible combinations.
    assert not any(cl.inconsistencies().values())
    broken = replace(cl, sense=cl.sense.copy())
    broken.sense[0] = OrbitSense.RETROGRADE  # an x1 orbit cannot be retrograde
    assert (
        broken.inconsistencies()["x1/x2 (prograde in the body frame) but retrograde"]
        == 1
    )

    ax = cl.plot_frequency_map(by="family")
    ax2 = cl.plot_class_histograms(scheme="family")
    import matplotlib.pyplot as plt

    plt.close(ax.figure)
    plt.close(ax2.figure)
    print("family table / view checks passed")


def test_mass_weighting():
    """Aggregates weighted by particle mass, matched by ID."""
    import matplotlib.pyplot as plt
    from lanfear import ParticleSystem, particle_weights

    specs = [
        (0.3, 0.6, 2.0, 1.0, True),  # x1
        (0.3, 0.6, 1.0, 2.0, True),  # x2
        (0.3, 0.5, 1.0, 1.0, True),  # prograde tube
        (-0.3, 0.5, 1.0, 1.0, True),  # retrograde tube
        (0.0, 0.6, 1.0, 1.0, False),  # trapped
    ]
    res = _synthetic_loops(0.05, specs)
    res.ids = np.array([10, 11, 12, 13, 14])
    cl = res.classify(diffusion_threshold=None)
    # Particles in another order, with an extra one not classified.
    ids = np.array([14, 99, 12, 10, 13, 11])
    mass = np.array([5.0, 100.0, 3.0, 1.0, 4.0, 2.0])
    n = len(ids)
    ps = ParticleSystem(
        pos=np.zeros((n, 3)),
        vel=np.zeros((n, 3)),
        mass=mass,
        ids=ids,
        species=np.full(n, "STAR"),
    )
    weights = particle_weights(cl.ids, ps)
    np.testing.assert_array_equal(weights, [1.0, 2.0, 3.0, 4.0, 5.0])

    assert cl.counts() == {"short_axis_tube": 4, "pibox": 1}
    assert cl.counts(particles=ps) == {"pibox": 5.0, "short_axis_tube": 10.0}
    assert cl.counts("family", particles=ps)["retrograde_short_axis_tube"] == 4.0

    edges = [0.0, 5.0]  # one bin holding every orbit (radius 2)
    frac = cl.fractions_by(edges, particles=ps, scheme="family")
    by_name = dict(
        zip([frac.class_names[int(c)] for c in frac.labels], frac.fractions[:, 0])
    )
    assert np.isclose(by_name["x2"], 2.0 / 15.0) and frac.weighted
    unit = ParticleSystem(**{**ps.__dict__, "mass": np.ones(n)})
    np.testing.assert_allclose(
        cl.fractions_by(edges, particles=unit).fractions,
        cl.fractions_by(edges).fractions,
    )
    boot = cl.fractions_by(edges, particles=ps, n_bootstrap=200, seed=1)
    assert np.all(boot.lower <= boot.fractions + 1e-12)
    assert np.all(boot.fractions <= boot.upper + 1e-12)

    comparison = cl.compare(cl)
    _, _, matrix = comparison.transition_matrix(particles=ps)
    assert np.isclose(matrix.sum(), 15.0) and matrix.dtype == float
    ax = comparison.plot_sankey(particles=ps)
    plt.close(ax.figure)
    ax = cl.plot_class_histograms(particles=ps)
    assert ax.get_ylabel() == "mass"
    plt.close(ax.figure)
    ax = cl.plot_class_fractions(edges, particles=ps)
    plt.close(ax.figure)

    missing = ParticleSystem(
        **{**ps.__dict__, "ids": np.array([14, 99, 12, 10, 13, 77])}
    )
    duplicate = ParticleSystem(
        **{**ps.__dict__, "ids": np.array([14, 14, 12, 10, 13, 11])}
    )
    for bad in (missing, duplicate):
        try:
            cl.counts(particles=bad)
        except ValueError:
            continue
        raise AssertionError("unmatched or duplicate IDs should raise")
    print("mass-weighting checks passed")


def _x_tubes(ratios):
    """Hand-built static long-axis tubes with given |w_x| / sqrt(|w_y| |w_z|)."""
    n = len(ratios)
    summary = np.zeros((n, len(SUMMARY_COLUMNS)))
    for name, value in (
        ("Lx_mean", 1.0),
        ("Lx_abs_mean", 1.0),
        ("Ly_abs_mean", 1.0),
        ("Lz_abs_mean", 1.0),
        ("r_mean", 2.0),
        ("r_max", 3.0),
        ("t_total", 2000.0),
        ("phi_x_range", 1.0),
    ):
        summary[:, SUMMARY_COLUMNS.index(name)] = value
    fundamentals = np.stack([np.asarray(ratios), np.ones(n), np.ones(n)], axis=1)
    lines = np.zeros((n, 3, 4, 2))
    lines[:, :, 0, 0] = fundamentals
    lines[:, :, 0, 1] = 1.0
    return OrbitResults(
        ids=np.arange(n),
        summary=summary,
        columns=SUMMARY_COLUMNS,
        time_unit=1.0,
        length_unit=1.0,
        n_periods=40,
        n_samples=4096,
        initial_radius=np.full(n, 2.0),
        fundamentals=fundamentals,
        lines=lines,
        diffusion=np.zeros((n, 3)),
        planar_lines=np.zeros((n, 3, 4, 2)),
    )


def test_adaptive_inner_outer_cut():
    """The inner/outer cut is placed from the population, or falls back."""
    from lanfear import DEFAULT_INNER_OUTER_RATIO, adaptive_inner_outer_cut

    rng = np.random.default_rng(3)
    inner = rng.normal(0.72, 0.04, 400)
    outer = np.clip(rng.normal(0.945, 0.01, 600), None, 0.975)  # stay off 1:1:1
    cut, reason = adaptive_inner_outer_cut(np.concatenate([inner, outer]))
    assert cut is not None and 0.8 < cut < 0.9, (cut, reason)
    for label, values in (
        ("outer only", outer),
        ("inner only", inner),
        ("too few", np.concatenate([inner[:40], outer[:60]])),
    ):
        cut_single, reason = adaptive_inner_outer_cut(values)
        assert cut_single is None and reason, (label, cut_single)

    # In the classifier: adaptive by default, recorded, overridable.
    res = _x_tubes(np.concatenate([inner, outer]))
    cl = res.classify(diffusion_threshold=None)
    assert cl.inner_outer_adaptive and np.isclose(cl.inner_outer_ratio, cut)
    n_inner = int(np.sum(cl.labels == OrbitClass.INNER_LONG_AXIS_TUBE))
    assert abs(n_inner - 400) <= 5, n_inner
    fixed = res.classify(diffusion_threshold=None, inner_outer_ratio=0.7)
    assert not fixed.inner_outer_adaptive and fixed.inner_outer_ratio == 0.7
    assert np.sum(fixed.labels == OrbitClass.INNER_LONG_AXIS_TUBE) < n_inner
    # A single-family population falls back to the fixed default.
    single = _x_tubes(outer).classify(diffusion_threshold=None)
    assert not single.inner_outer_adaptive
    assert single.inner_outer_ratio == DEFAULT_INNER_OUTER_RATIO
    assert np.all(single.labels == OrbitClass.OUTER_LONG_AXIS_TUBE)
    print(f"adaptive inner/outer cut OK: {cut:.3f}")


def test_reliability_flags():
    """Regularity, unresolved frequencies and the near-corotation flag."""
    from dataclasses import replace

    omega = 0.05
    specs = [
        (0.3, 0.6, 2.0, 1.0, True),  # epsilon = 0.05 / 0.35
        (-0.03, 0.06, 1.0, 1.0, True),  # inertial 0.02: epsilon 2.5
        (0.0005, 0.6, 1.0, 1.0, True),  # at corotation: slow only in the body frame
        (0.3, 0.005, 1.0, 1.0, True),  # radial frequency unresolved
        (0.0, 0.6, 1.0, 1.0, False),  # librating: no epsilon
    ]
    res = _synthetic_loops(omega, specs)
    res.diffusion[0] = 1.0  # drifting frequencies
    cl = res.classify()
    assert list(cl.regularity_names) == ["diffusing"] + ["regular"] * 4
    assert cl.labels[0] == OrbitClass.IRREGULAR
    # The corotation loop's sense uses its inertial azimuthal frequency (~Omega_p,
    # well resolved), and its corotation tag needs no resolved body frequency.
    np.testing.assert_array_equal(cl.unresolved, [False, False, False, True, False])
    np.testing.assert_allclose(
        cl.pattern_speed_ratio[:4], [0.05 / 0.35, 2.5, 0.05 / 0.0505, 0.05 / 0.35]
    )
    assert np.isnan(cl.pattern_speed_ratio[4])
    # Default band (0.5, inf): at (0.99) and beyond (2.5) corotation.
    np.testing.assert_array_equal(cl.near_corotation, [False, True, True, False, False])
    narrow = res.classify(corotation_band=(0.5, 2.0))
    np.testing.assert_array_equal(
        narrow.near_corotation, [False, False, True, False, False]
    )
    wide = res.classify(corotation_band=(0.1, 3.0))
    np.testing.assert_array_equal(wide.near_corotation, [True, True, True, True, False])
    assert "pattern_speed_ratio" in cl.quantities

    static = replace(res, pattern_speed=np.zeros(3)).classify()
    assert not static.near_corotation.any() and not np.any(static.pattern_speed_ratio)
    print("reliability flag checks passed")


def test_known_pattern_orbits():
    """Pinned x1, x4 and L4-trapped orbits in tumbling test models.

    Found by searching random near-circular starts in the bar-like
    (1, 0.45, 0.35) model rotating at Omega_c(3) of the unsquashed sphere,
    and near-rest starts around the L4 point (0, 2.5, 0) of the milder
    (1, 0.8, 0.6) model rotating at the circular frequency there (strong
    bars destabilise L4/L5).
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_planar_spectra import _results

    bar = build_scf(flatten=(1.0, 0.45, 0.35))
    omega = np.sqrt((9 / 16) / 27)
    states = [
        [0.1249, -0.784, 0.0105, 0.7778, 0.119, -0.0005],  # x1
        [-0.0896, 0.4202, -0.0058, 0.4347, 0.0799, 0.005],  # x4
    ]
    cl = _results(bar, states, pattern_speed=(0.0, 0.0, omega)).classify()
    assert list(cl.pattern_resonance_names) == ["x1", "x4"], cl.pattern_resonance_names
    # The x1 orbit passes close to the centre (no hole about z): a box that
    # winds about the pattern axis, so it has a resonance but no loop sense.
    assert list(cl.shape_names) == ["box", "rosette"], cl.shape_names
    assert list(cl.sense_names) == ["none", "retrograde"]
    assert np.allclose(cl.pattern_frequency_ratio, 2.0, atol=1e-3)

    tri = build_scf(flatten=(1.0, 0.8, 0.6))
    omega = float(np.sqrt(-tri.acceleration(0.0, 2.5, 0.0)[1] / 2.5))
    state = [-0.4789, 2.3375, -0.0067, -0.4458, -0.0941, 0.0026]  # near L4
    cl = _results(tri, [state], pattern_speed=(0.0, 0.0, omega)).classify()
    assert cl.sense[0] == OrbitSense.LIBRATING and cl.corotation_trapped[0]
    assert cl.shape[0] == OrbitShape.BOX
    print("known pattern-resonance orbits OK")


def test_shape_labels():
    """Shapes follow the regular family, survive the irregular override."""
    res, _ = _rotating_case()
    res.diffusion[1] = 1.0  # make the box chaotic
    cl = res.classify()
    assert cl.labels[1] == OrbitClass.IRREGULAR and cl.shape[1] == OrbitShape.BOX
    assert list(cl.shape_names[[0, 3]]) == ["short_axis_tube", "rosette"]
    condensed = cl.condense_families()
    assert np.array_equal(condensed.shape, cl.shape)
    print("shape label checks passed")


def test_runaway():
    """Runaway orbits are flagged and left out of the classification by default.

    * orbit 1 runs away (``r_max`` far beyond the system radius);
    * orbits 0 and 2 are ordinary bound orbits.
    """
    import tempfile
    from dataclasses import replace

    n = 3
    summary = np.zeros((n, len(SUMMARY_COLUMNS)))

    def set_col(name, values):
        summary[:, SUMMARY_COLUMNS.index(name)] = values

    set_col("Lz_mean", [1.0] * n)
    set_col("Lz_abs_mean", [1.0] * n)
    set_col("Lx_abs_mean", [1.0] * n)
    set_col("Ly_abs_mean", [1.0] * n)
    set_col("x_waist_ratio", [1.0] * n)
    set_col("r_mean", [2.0] * n)
    set_col("r_max", [3.0, 1e4, 3.0])
    set_col("period", [2 * np.pi] * n)

    fundamentals = np.array([[0.31, 0.47, 0.61], [0.29, 0.43, 0.67], [0.3, 0.5, 0.7]])
    lines = np.zeros((n, 3, 4, 2))
    lines[:, :, 0, 0] = fundamentals
    lines[:, :, 0, 1] = 1.0

    res = OrbitResults(
        ids=np.arange(n),
        summary=summary,
        columns=SUMMARY_COLUMNS,
        time_unit=1.0,
        length_unit=1.0,
        n_periods=40,
        n_samples=4096,
        initial_radius=np.full(n, 2.0),
        fundamentals=fundamentals,
        lines=lines,
        diffusion=np.zeros((n, 3)),
        planar_lines=np.zeros_like(lines),
    )

    assert np.array_equal(res.runaway, [False, True, False])
    kept = res.drop_runaways()
    assert np.array_equal(kept.ids, [0, 2])
    assert not kept.runaway.any()
    assert np.array_equal(res.classify().ids, [0, 2])  # dropped by default
    assert len(res.classify(drop_runaways=False).labels) == n
    assert not replace(res, runaway_factor=1e4).runaway.any()
    assert np.array_equal(res.to_dict()["runaway"], res.runaway)

    # The runaway criterion survives a save/load round trip.
    res = replace(res, system_radius=2.0, runaway_factor=5.0)
    with tempfile.TemporaryDirectory() as tmp:
        loaded = OrbitResults.load(res.save(os.path.join(tmp, "orbits.npz")))
    assert loaded.system_radius == 2.0 and loaded.runaway_factor == 5.0
    assert np.array_equal(loaded.runaway, res.runaway)
    print("runaway flag OK")


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


def test_fractions_by():
    """fractions_by bins on any quantity and bootstraps sensible bounds."""
    from lanfear import OrbitClassification

    rng = np.random.default_rng(1)
    n = 4000
    energy = rng.uniform(0.0, 2.0, n)
    # Tube probability rises with energy, so the composition varies by bin.
    is_tube = rng.uniform(size=n) < energy / 2.0
    labels = np.where(is_tube, int(OrbitClass.SHORT_AXIS_TUBE), int(OrbitClass.PIBOX))
    zeros3 = np.zeros((n, 3))
    cl = OrbitClassification(
        labels=labels,
        circulation=zeros3,
        tube_axis=np.zeros(n, int),
        planarity=np.zeros(n),
        resonance=np.zeros((n, 3), int),
        resonance_order=np.zeros(n, int),
        quantities={"energy": energy},
    )
    edges = np.linspace(0.0, 2.0, 5)

    res = cl.fractions_by(edges, quantity="energy")
    assert res.lower is None
    assert np.allclose(res.fractions.sum(axis=0), 1.0)
    assert res.counts.sum() == n
    tube = res.fractions[list(res.labels).index(int(OrbitClass.SHORT_AXIS_TUBE))]
    assert np.all(np.diff(tube) > 0)  # composition follows the input trend

    # An explicit array behaves like the named quantity.
    assert np.allclose(cl.fractions_by(edges, quantity=energy).fractions, res.fractions)

    # Bootstrap: bounds bracket the estimate, are reproducible, and narrow ~1/sqrt(N).
    b1 = cl.fractions_by(edges, quantity="energy", n_bootstrap=300, seed=5)
    b2 = cl.fractions_by(edges, quantity="energy", n_bootstrap=300, seed=5)
    assert np.allclose(b1.lower, b2.lower)
    assert np.all(b1.lower <= b1.fractions + 1e-12)
    assert np.all(b1.upper >= b1.fractions - 1e-12)
    width = (b1.upper - b1.lower)[:, 1]
    expected = 2 * np.sqrt(tube[1] * (1 - tube[1]) / res.counts[:, 1].sum())
    assert np.all(width > 0.5 * expected) and np.all(width < 1.5 * expected)

    # per_bin=False normalises to the binned total.
    g = cl.fractions_by(edges, quantity="energy", per_bin=False)
    assert np.isclose(g.fractions.sum(), 1.0)

    # Errors: unknown / missing quantity, bad edges, bad confidence.
    for bad in (
        dict(edges=edges, quantity="nope"),
        dict(edges=edges, quantity="radius"),  # not recorded on this object
        dict(edges=[1.0], quantity="energy"),
        dict(edges=[0.0, 0.0, 1.0], quantity="energy"),
        dict(edges=edges, quantity="energy", confidence=1.5),
    ):
        try:
            cl.fractions_by(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")

    # Plot with the band; condensing keeps the extra quantities.
    import matplotlib

    matplotlib.use("Agg")
    ax = cl.condense_families().plot_class_fractions(
        edges, quantity="energy", n_bootstrap=100, seed=0
    )
    assert len(ax.collections) == 2  # one band per class
    _save_figure(ax.figure, "class_fractions_bootstrap")
    print("fractions_by OK")


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
    test_known_boxlets()
    test_resonance_false_positives()
    test_boxlet_description()
    print("== y-tube resonance corroboration ==")
    test_y_tube_resonance_corroboration()
    print("== population ==")
    test_population()
    test_diffusion_drop_rule()
    print("== runaway orbits ==")
    test_runaway()
    print("== rotating-figure rules ==")
    test_rotating_hole_and_rosette()
    test_rotating_shared_line_not_boxlet()
    test_sense_labels()
    test_pattern_resonance_synthetic()
    test_known_pattern_orbits()
    test_reliability_flags()
    test_adaptive_inner_outer_cut()
    test_mass_weighting()
    test_family_table_and_views()
    test_shape_labels()
    print("== condense families ==")
    test_condense_families()
    print("== get class ids ==")
    test_get_class_ids()
    print("== plot class fractions ==")
    test_plot_class_fractions()
    test_fractions_by()
    print("== plot class histograms ==")
    test_plot_class_histograms()
    print("== plot frequency map ==")
    test_plot_frequency_map()
    print("== compare ==")
    test_compare()
    print("\nALL CLASSIFICATION TESTS PASSED")
