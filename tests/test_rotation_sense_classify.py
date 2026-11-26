"""Rotation senses in classify_orbits and the probabilistic classifier.

python tests/test_rotation_sense_classify.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lanfear as lf  # noqa: E402
from lanfear.probabilistic_classify import measurement_uncertainties  # noqa: E402

_CACHE = {}


def _population(n_orbits=160, seed=3):
    """A small rotating triaxial blob, prepared, fitted and integrated."""
    if "res" in _CACHE:
        return _CACHE["ps"], _CACHE["res"]
    rng = np.random.default_rng(seed)
    n = 30_000
    pos = rng.normal(size=(n, 3)) * np.array([3.0, 2.0, 1.5])
    r = np.linalg.norm(pos, axis=1)[:, None]
    vel = rng.normal(size=(n, 3)) * 40.0
    vel += 60.0 * np.cross([0.0, 0.0, 1.0], pos) / np.maximum(r, 0.5)
    ps = lf.ParticleSystem(
        pos=pos,
        vel=vel,
        mass=np.full(n, 1.0 / n),
        ids=np.arange(n),
        species=np.full(n, "STAR"),
    )
    ps.prepare(centre="field", pattern_speed="none", check_figure_rotation=False)
    pot = lf.Potential.from_particles(ps, n_max=6, l_max=4)
    sub = ps.random_subset(n_orbits, rng=np.random.default_rng(seed))
    res = lf.analyse_family(
        pot, sub, family="STAR", n_periods=20, n_samples=1024, comm=None, progress=False
    )
    res.add_reference(ps.reference_spin(family="STAR"))
    res.add_reference(lf.SpinReference.from_axis([1.0, 0.0, 0.0], ps.frame, "x"))
    _CACHE.update(ps=ps, res=res)
    return ps, res


_SCHEMES = (
    "labels",
    "family",
    "shape",
    "boxlet_type",
    "sense",
    "pattern_resonance",
    "regularity",
    "unresolved",
    "tube_axis",
)


def test_default_classification_unchanged():
    """Without references nothing is added and every label is the same."""
    _, res = _population()
    plain = res.classify()
    with_refs = res.classify(references="all")
    assert plain.rotation_senses is None and plain.rotation_references is None
    assert plain.rotation_sense_schemes == ()
    for attr in _SCHEMES:
        assert np.array_equal(getattr(plain, attr), getattr(with_refs, attr)), attr
    assert set(with_refs.quantities) - set(plain.quantities) == {
        "circulation_coherence",
        "reference_cosine:STAR",
        "reference_cosine:x",
    }
    print("default classification unchanged")


def test_rotation_sense_schemes():
    """The schemes match OrbitResults.rotation_sense and work everywhere."""
    ps, res = _population()
    cl = res.classify(references=["STAR", "x"], coherence_min=0.3, perpendicular=0.5)
    assert cl.rotation_sense_schemes == ("rotation_sense:STAR", "rotation_sense:x")
    assert cl.rotation_sense_thresholds == (0.3, 0.5)
    kept = res.drop_runaways()
    assert np.array_equal(cl.ids, kept.ids)
    for name in ("STAR", "x"):
        expected = kept.rotation_sense(name, coherence_min=0.3, perpendicular=0.5)
        assert np.array_equal(cl.rotation_senses[name], expected)
        view = cl.view(f"rotation_sense:{name}")
        assert np.array_equal(view.labels, expected)
        assert set(view.class_names.values()) == set(lf.ROTATION_SENSE_NAMES.values())
    names = cl.rotation_sense_names("STAR")
    assert set(names) <= set(lf.ROTATION_SENSE_NAMES.values())
    assert "co" in set(names)  # the blob rotates about the STAR spin

    counts = cl.counts("rotation_sense:STAR")
    assert sum(counts.values()) == len(cl.ids)
    mass = cl.counts("rotation_sense:STAR", particles=ps)
    assert np.isclose(sum(mass.values()), len(cl.ids) / ps.n_particles)
    co = cl.mask("co", scheme="rotation_sense:STAR")
    assert co.sum() == counts.get("co", 0)
    fractions = cl.fractions_by(
        [0.0, 2.0, 50.0], scheme="rotation_sense:STAR", particles=ps
    )
    assert fractions.weighted and np.allclose(np.nansum(fractions.fractions, axis=0), 1)
    by_cosine = cl.fractions_by([-1.0, 0.0, 1.0], quantity="reference_cosine:STAR")
    assert "cos" in by_cosine.quantity_label
    cl.fractions_by([0.0, 0.5, 1.0], quantity="circulation_coherence")

    table = cl.crosstab("family", "rotation_sense:STAR")
    assert sum(sum(row.values()) for row in table.values()) == len(cl.ids)
    family_counts = cl.counts("family")
    for family, row in table.items():
        assert sum(row.values()) == family_counts[family]
    masses = cl.crosstab("shape", "rotation_sense:x", particles=ps)
    assert all(isinstance(v, float) for row in masses.values() for v in row.values())
    print("rotation-sense schemes")


def test_rotation_sense_errors():
    """Unknown schemes and references raise."""
    _, res = _population()
    cl = res.classify(references="x")
    for scheme in ("rotation_sense:STAR", "rotation_sense:", "nonsense"):
        try:
            cl.view(scheme)
        except ValueError:
            continue
        raise AssertionError(f"view({scheme!r}) should have raised")
    empty = res.select(np.arange(len(res.ids)))
    empty.references.clear()
    try:
        empty.classify(references="all")
    except ValueError:
        pass
    else:
        raise AssertionError("references='all' with none stored should raise")
    try:
        res.classify(references="missing")
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown reference name should raise")
    print("rotation-sense errors")


def test_probabilistic_rotation_sense():
    """Each reference gets a probabilistic scheme; the default gets none."""
    _, res = _population()
    plain = lf.classify_orbits_probabilistic(res, n_draws=8, seed=1)
    assert not any(s.startswith("rotation_sense:") for s in plain.schemes)
    prob = lf.classify_orbits_probabilistic(res, n_draws=8, seed=1, references="all")
    # The same random draws: every other scheme's counts are unchanged.
    for scheme in plain.schemes:
        assert np.array_equal(prob.label_counts[scheme], plain.label_counts[scheme])
    for name in ("STAR", "x"):
        scheme = f"rotation_sense:{name}"
        assert scheme in prob.schemes
        _, p = prob.probabilities(scheme)
        assert np.allclose(p.sum(axis=1), 1.0)
        hard = prob.map_labels(scheme)
        assert hard.shape == (len(prob.ids),)
    hard = prob.to_hard()
    assert set(hard.rotation_senses) == {"STAR", "x"}
    errors = measurement_uncertainties(res.drop_runaways(), inertial=True)
    assert errors["inertial_angular_momentum"].shape == (len(prob.ids), 3)
    print("probabilistic rotation sense")


if __name__ == "__main__":
    test_default_classification_unchanged()
    test_rotation_sense_schemes()
    test_rotation_sense_errors()
    test_probabilistic_rotation_sense()
    print("\nALL ROTATION-SENSE CLASSIFICATION TESTS PASSED")
