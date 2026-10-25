"""Measurement-error probabilistic-classification tests.

Classifies perturbed copies of a triaxial population (reusing the
``build_scf``/``_vcirc`` helpers from ``test_classify.py``) and checks that the
probabilities are consistent with the deterministic labels, that orbits with
no measurement error are certain, that an orbit on a threshold is split, and
that the query API behaves.

    python tests/test_probabilistic_classify.py
"""

import functools
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lanfear import OrbitClass  # noqa: E402
from lanfear.classify import OrbitShape  # noqa: E402
from lanfear.orbits import OrbitResults, SUMMARY_COLUMNS  # noqa: E402
from lanfear.probabilistic_classify import (  # noqa: E402
    classify_orbits_probabilistic,
    measurement_uncertainties,
)

from test_classify import _vcirc, build_scf  # noqa: E402


@functools.lru_cache(maxsize=1)
def _shared_scf():
    """The one triaxial SCF potential every test in this file classifies orbits in."""
    return build_scf(flatten=(1.0, 0.8, 0.6), n=40_000, seed=7)


@functools.lru_cache(maxsize=4)
def _triaxial_population(n=800, seed=7, n_periods=8, n_samples=512, pattern_speed=None):
    """A triaxial population, integrated and frequency-analysed.

    With ``pattern_speed`` (HO units, a (3,) tuple) the figure tumbles at that
    rate.
    """
    omega = np.zeros(3) if pattern_speed is None else np.asarray(pattern_speed, float)
    tri = _shared_scf()
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-3, 3, (n, 3))
    r = np.linalg.norm(pos, axis=1)
    pos = pos[(r > 0.5) & (r < 5)]
    states = []
    for p in pos:
        vc = _vcirc(tri, *p)
        v = rng.uniform(-1, 1, 3)
        v *= rng.uniform(0, 1.0) * vc / (np.linalg.norm(v) + 1e-9)
        states.append([*p, *v])
    states = np.array(states)

    summ, fund, lines, diff, planar = tri.analyse_batch(
        states,
        n_periods=n_periods,
        n_samples=n_samples,
        n_lines=4,
        pattern_speed=tuple(omega),
    )
    return OrbitResults(
        ids=np.arange(len(states)),
        summary=summ,
        columns=SUMMARY_COLUMNS,
        time_unit=1.0,
        length_unit=1.0,
        n_periods=n_periods,
        n_samples=n_samples,
        initial_radius=np.linalg.norm(states[:, :3], axis=1),
        fundamentals=fund,
        lines=lines,
        diffusion=diff,
        planar_lines=planar,
        pattern_speed=omega,
    )


def _without_errors(results):
    """A copy whose half-window values equal the full-window ones (no error)."""
    summary = np.array(results.summary, copy=True)
    col = list(results.columns).index
    for a in "xyz":
        summary[:, col(f"L{a}_mean_h1")] = summary[:, col(f"L{a}_mean")]
        summary[:, col(f"L{a}_abs_mean_h1")] = summary[:, col(f"L{a}_abs_mean")]
        for h in (1, 2):
            summary[:, col(f"rho_{a}_min_h{h}")] = summary[:, col(f"rho_{a}_min")]
            summary[:, col(f"freq_{a}_h{h}")] = results.fundamentals[:, "xyz".index(a)]
    return replace(results, summary=summary)


def test_measurement_uncertainties():
    """Errors have the documented shapes, are non-negative, and need the half columns."""
    results = _triaxial_population()
    errors = measurement_uncertainties(results)
    n = len(results.ids)
    for name, shape in (
        ("circulation", (n, 3)),
        ("axis_distance", (n, 3)),
        ("frequency", (n, 3)),
        ("other_frequency", (n,)),
    ):
        assert errors[name].shape == shape, name
        assert np.all(np.isfinite(errors[name])), name
    assert np.all(errors["other_frequency"] >= 0)
    assert np.any(errors["circulation"] != 0) and np.any(errors["frequency"] != 0)

    keep = [c for c in results.columns if not c.endswith(("_h1", "_h2"))]
    old = replace(
        results,
        summary=results.summary[:, [list(results.columns).index(c) for c in keep]],
        columns=keep,
    )
    try:
        measurement_uncertainties(old)
    except ValueError:
        pass
    else:
        raise AssertionError("results without half-window columns should be rejected")
    print("measurement uncertainty checks passed")


def test_probabilities_consistent_with_deterministic():
    """Rows sum to 1; the most probable class is (almost always) the deterministic one."""
    results = _triaxial_population()
    prob = classify_orbits_probabilistic(results, n_draws=24)
    base = prob.base
    for scheme in prob.schemes:
        _, p = prob.probabilities(scheme)
        np.testing.assert_allclose(p.sum(axis=1), 1.0)
    agree = np.mean(prob.map_labels("class") == base.labels)
    assert agree >= 0.97, agree
    # A label every draw agrees on is the deterministic label.
    certain = prob.map_probability("family") == 1.0
    np.testing.assert_array_equal(
        prob.map_labels("family")[certain], base.family[certain]
    )
    assert certain.mean() > 0.5
    # Regularity is not perturbed, apart from the spectral test's tolerance,
    # which scales with the (perturbed) frequencies.
    assert np.mean(prob.map_probability("regularity") == 1.0) >= 0.99
    np.testing.assert_array_equal(prob.map_labels("regularity"), base.regularity)
    print(f"consistency checks passed: MAP class == deterministic for {agree:.1%}")


def test_no_errors_are_certain():
    """With zero measurement error every draw reproduces the deterministic labels."""
    results = _without_errors(_triaxial_population())
    prob = classify_orbits_probabilistic(results, n_draws=4)
    for scheme in prob.schemes:
        assert np.all(prob.map_probability(scheme) == 1.0), scheme
        assert np.all(prob.entropy(scheme) == 0.0), scheme
    np.testing.assert_array_equal(prob.map_labels("family"), prob.base.family)
    sigma = prob.counts_uncertainty("family")
    assert all(s == 0.0 for s in sigma.values()), sigma
    print("zero-error checks passed")


def test_threshold_orbit_is_split():
    """A short-axis tube moved onto the box/tube cut, with a large error, is split."""
    results = _without_errors(_triaxial_population())
    cl = results.classify()
    k = int(np.flatnonzero(cl.labels == OrbitClass.SHORT_AXIS_TUBE)[0])
    summary = np.array(results.summary, copy=True)
    col = list(results.columns).index
    abs_mean = summary[k, col("Lz_abs_mean")]
    sign = np.sign(summary[k, col("Lz_mean")])
    circ, spread = 0.9, 0.2  # the default circ_thresh; halves at 0.9 +- 0.2
    summary[k, col("Lz_mean")] = sign * circ * abs_mean
    summary[k, col("Lz_mean_h1")] = sign * (circ + spread) * abs_mean
    for a in "xy":  # no circulation about the other axes
        summary[k, col(f"L{a}_mean")] = 0.0
        summary[k, col(f"L{a}_mean_h1")] = 0.0
    results = replace(results, summary=summary)
    assert abs(measurement_uncertainties(results)["circulation"][k, 2]) > 0.15
    prob = classify_orbits_probabilistic(results, n_draws=64, seed=3)
    names, p = prob.probability_bar(index=k, scheme="shape")
    p_tube = p[names.index(OrbitShape.SHORT_AXIS_TUBE.name.lower())]
    p_box = p[names.index(OrbitShape.BOX.name.lower())]
    assert 0.25 < p_tube < 0.75 and 0.25 < p_box < 0.75, (p_tube, p_box)
    assert p_tube + p_box > 0.95
    print(f"threshold orbit split: P(tube) = {p_tube:.2f}, P(box) = {p_box:.2f}")


def test_query_api():
    """Counts, masks, bars, plots and to_hard behave."""
    results = _triaxial_population()
    prob = classify_orbits_probabilistic(results, n_draws=16)
    n = len(prob.ids)
    counts = prob.counts("family")
    assert abs(sum(counts.values()) - n) < 1e-6
    sigma = prob.counts_uncertainty("family")
    assert set(sigma) == set(counts) and all(s >= 0 for s in sigma.values())
    confident = prob.mask_confident(0.8)
    np.testing.assert_array_equal(confident, ~prob.mask_ambiguous(0.8))
    names, p = prob.probability_bar(id=int(prob.ids[0]))
    assert len(names) == len(p) and abs(p.sum() - 1) < 1e-9
    label = prob.map_names("class")[0]
    _, mean = prob.probabilities_for(label=label)
    assert abs(mean.sum() - 1) < 1e-9
    for bad in (dict(), dict(id=0, index=0)):
        try:
            prob.probability_bar(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"probability_bar({bad}) should raise")
    try:
        prob.probabilities("nonsense")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown scheme should raise")
    ax = prob.plot_probability_bar(index=0, scheme="family")
    assert ax.get_ylabel() == "probability"
    hard = prob.to_hard()
    np.testing.assert_array_equal(hard.labels, prob.map_labels("class"))
    np.testing.assert_array_equal(hard.family, prob.map_labels("family"))
    assert prob.label_counts["class"].dtype == np.uint8  # compact storage
    print("query API checks passed")


def test_rotating_figure():
    """A tumbling figure: pattern labels get probabilities too."""
    results = _triaxial_population(n=400, seed=11, pattern_speed=(0.0, 0.0, 0.05))
    prob = classify_orbits_probabilistic(results, n_draws=8)
    assert np.mean(prob.map_labels("class") == prob.base.labels) >= 0.95
    names, p = prob.probabilities("pattern_resonance")
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert "none" in names
    print("rotating-figure checks passed")


if __name__ == "__main__":
    test_measurement_uncertainties()
    test_probabilities_consistent_with_deterministic()
    test_no_errors_are_certain()
    test_threshold_orbit_is_split()
    test_query_api()
    test_rotating_figure()
    print("All probabilistic classification tests passed.")
