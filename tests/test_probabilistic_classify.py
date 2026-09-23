"""Bayesian probabilistic-classification tests.

Fits the self-supervised :class:`~lanfear.BayesianOrbitClassifier` on a
triaxial population (reusing the ``build_scf``/``_vcirc`` helpers from
``test_classify.py``) and checks it agrees with the deterministic classifier
away from its thresholds, that the per-particle posterior-query API behaves,
and that the fitted model round-trips through save/load.

    python tests/test_probabilistic_classify.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lanfear import OrbitClass  # noqa: E402
from lanfear.orbits import OrbitResults, SUMMARY_COLUMNS  # noqa: E402
from lanfear.probabilistic_classify import (  # noqa: E402
    FEATURE_NAMES,
    BayesianOrbitClassifier,
    classify_orbits_probabilistic,
    feature_matrix,
    fit_probabilistic_classifier,
)

from test_classify import _vcirc, build_scf  # noqa: E402


def _triaxial_population(n=3000, seed=7, n_periods=30, n_samples=2048):
    """A large-ish triaxial population, integrated and frequency-analysed."""
    tri = build_scf(flatten=(1.0, 0.8, 0.6), n=150_000, seed=seed)
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

    summ, fund, lines, diff = tri.analyse_batch(
        states, n_periods=n_periods, n_samples=n_samples, n_lines=4
    )
    results = OrbitResults(
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
    )
    return results


def test_feature_matrix_shape_and_alignment():
    """feature_matrix returns the documented shape and rejects mismatched inputs."""
    results = _triaxial_population(n=300, seed=1)
    cl = results.classify()
    features = feature_matrix(cl, results)
    assert features.shape == (len(results.ids), len(FEATURE_NAMES))

    other = results.classify()
    other.ids = other.ids[::-1].copy()  # break alignment
    try:
        feature_matrix(other, results)
        assert False, "expected a ValueError for mismatched ids"
    except ValueError:
        pass
    print("feature_matrix shape/alignment OK")


def test_fit_agrees_with_deterministic_classifier():
    """Confidently-away-from-threshold orbits get a MAP label matching classify_orbits."""
    results = _triaxial_population(n=4000, seed=11)
    cl = results.classify()
    model = fit_probabilistic_classifier(results, classification=cl)

    assert len(model.class_labels) >= 2
    assert np.all(model.n_training >= 5)
    print(
        f"  fitted classes: {model.class_names}, n_training={dict(zip(model.class_labels, model.n_training))}"
    )

    features = feature_matrix(cl, results)
    labels, map_probability, entropy = model.predict(features)

    valid = labels >= 0
    assert valid.any()
    # Posterior probabilities are well-formed.
    assert np.all(
        (map_probability[valid] >= 0) & (map_probability[valid] <= 1.0 + 1e-9)
    )
    assert np.all(entropy[valid] >= -1e-9)
    max_entropy = np.log2(len(model.class_labels))
    assert np.all(entropy[valid] <= max_entropy + 1e-6)

    # Orbits the deterministic classifier gave a real (trained) label to, and
    # which are confidently classified by the Bayesian model, should mostly
    # agree with it -- this is a self-supervised fit on that very label. The
    # one documented exception is IRREGULAR: an orbit can be flagged
    # irregular purely by the spectral (>3 base frequencies) test, which has
    # no continuous analogue in the reduced feature set, so a spectral-only
    # irregular orbit can look identical to a regular box/tube in every
    # feature the Bayesian model actually sees. Regular families are checked
    # strictly; IRREGULAR is only checked against chance.
    covered = np.isin(cl.labels, model.class_labels)
    confident = valid & covered & (map_probability >= 0.9)
    assert confident.sum() > 0

    regular = confident & (cl.labels != OrbitClass.IRREGULAR)
    assert regular.sum() > 0
    regular_agreement = (labels[regular] == cl.labels[regular]).mean()
    print(
        f"  {regular.sum()} confident regular-family orbits, {100*regular_agreement:.1f}% agree with classify_orbits"
    )
    assert regular_agreement > 0.9

    irregular = confident & (cl.labels == OrbitClass.IRREGULAR)
    if irregular.sum() > 0:
        irregular_agreement = (labels[irregular] == cl.labels[irregular]).mean()
        chance = 1.0 / len(model.class_labels)
        print(
            f"  {irregular.sum()} confident IRREGULAR orbits, {100*irregular_agreement:.1f}% agree (chance {100*chance:.1f}%)"
        )
        assert irregular_agreement > chance


def test_probability_bar_and_confidence_masks():
    """Per-particle posterior queries and confidence filtering behave as documented."""
    results = _triaxial_population(n=2500, seed=23)
    prob_cl = classify_orbits_probabilistic(results)

    assert prob_cl.labels.shape == (len(results.ids),)
    assert prob_cl.map_probability.shape == prob_cl.labels.shape
    assert prob_cl.entropy.shape == prob_cl.labels.shape

    valid = prob_cl.labels >= 0
    assert valid.any()
    some_id = int(prob_cl.ids[valid][0])
    names, proba = prob_cl.probability_bar(id=some_id)
    assert len(names) == len(proba) == len(prob_cl.model.class_labels)
    assert np.isclose(proba.sum(), 1.0)

    try:
        prob_cl.probability_bar(id=some_id, index=0)
        assert False, "expected ValueError when both id and index are given"
    except ValueError:
        pass
    try:
        prob_cl.probability_bar(id=-999999)
        assert False, "expected ValueError for an unknown id"
    except ValueError:
        pass

    confident = prob_cl.mask_confident(0.8)
    ambiguous = prob_cl.mask_ambiguous(0.8)
    # Confident and ambiguous partition the evaluable orbits, and are disjoint.
    assert not np.any(confident & ambiguous)
    assert np.array_equal(confident | ambiguous, np.isfinite(prob_cl.map_probability))

    if valid.any():
        cls = OrbitClass(int(prob_cl.labels[valid][0]))
        mask = prob_cl.mask(cls)
        if mask.any():
            names2, mean_proba = prob_cl.probabilities_for(cls=cls)
            assert np.isclose(mean_proba.sum(), 1.0)
    print("probability_bar / confidence-mask API OK")


def test_save_load_roundtrip(tmp_path=None):
    """A fitted model round-trips through save/load and predicts identically."""
    import tempfile

    results = _triaxial_population(n=1500, seed=31)
    model = fit_probabilistic_classifier(results)
    features = feature_matrix(results.classify(), results)
    proba_before = model.predict_prob(features)

    with tempfile.TemporaryDirectory() as d:
        path = model.save(os.path.join(d, "model"))
        reloaded = BayesianOrbitClassifier.load(path)

    assert np.array_equal(model.class_labels, reloaded.class_labels)
    proba_after = reloaded.predict_prob(features)
    finite = np.isfinite(proba_before)
    assert np.array_equal(finite, np.isfinite(proba_after))
    assert np.allclose(proba_before[finite], proba_after[finite])
    print("save/load round-trip OK")


def test_compact_storage_not_full_matrix():
    """The classification stores per-orbit summaries, not an (N, n_classes) matrix."""
    results = _triaxial_population(n=800, seed=41)
    prob_cl = classify_orbits_probabilistic(results)
    n = len(results.ids)
    for name in ("labels", "map_probability", "entropy"):
        arr = getattr(prob_cl, name)
        assert arr.shape == (n,), f"{name} should be shape (N,), got {arr.shape}"
    # The only (N, D) array kept is the reduced feature cache, D << n_classes typically not
    # the point -- the point is there is no (N, n_classes) array anywhere on the object.
    assert prob_cl.features.shape == (n, len(FEATURE_NAMES))
    print(
        "compact (label, probability, entropy) storage OK, no (N, n_classes) matrix cached"
    )


if __name__ == "__main__":
    test_feature_matrix_shape_and_alignment()
    test_fit_agrees_with_deterministic_classifier()
    test_probability_bar_and_confidence_masks()
    test_save_load_roundtrip()
    test_compact_storage_not_full_matrix()
    print("All probabilistic classification tests passed.")
