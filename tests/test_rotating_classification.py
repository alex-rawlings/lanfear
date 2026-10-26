"""Regression tests: pinned orbits per family in a slowly tumbling triaxial figure.

The model is the triaxial SCF fit of :func:`test_classify.build_scf` with axis
ratios 1 : 0.79 : 0.5 (those of the perfect ellipsoids lanfear was calibrated
on), tumbling about its short axis z with corotation at 8 or 16 half-mass radii
of the spherical-equivalent Hernquist model -- the slow-tumbling range where the
classification was validated (see ``docs/tuning.rst``). The initial conditions
(HO units) were found by searching random starts and keeping, for each family,
an orbit far from every threshold whose label survives a 0.1% change of its
initial conditions.

    python tests/test_rotating_classification.py
"""

import functools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lanfear.classify import OrbitShape, Regularity  # noqa: E402
from test_classify import build_scf  # noqa: E402
from test_planar_spectra import _results  # noqa: E402

# Initial conditions and expected family at corotation radii 16 and 8. These
# orbits keep their family in the static figure too.
REGULAR_ORBITS = {
    "pibox": [-0.8449, 0.3377, -0.0591, -0.2085, 0.0708, 0.2114],
    "banana": [-2.797, 0.3953, 0.3242, 0.0265, 0.0325, -0.0188],
    "prograde_short_axis_tube": [-0.4485, -2.2026, -2.2314, 0.2484, 0.0672, -0.1743],
    "retrograde_short_axis_tube": [0.6521, -0.1246, 0.0956, 0.1457, -0.5321, 0.0316],
    "prograde_inner_long_axis_tube": [
        0.0941,
        -0.0554,
        0.2563,
        0.0604,
        -0.2371,
        -0.1161,
    ],
    "retrograde_inner_long_axis_tube": [
        -0.1592,
        0.0156,
        -0.5705,
        -0.0681,
        -0.1381,
        -0.1012,
    ],
    "prograde_outer_long_axis_tube": [
        0.8319,
        -0.2776,
        -0.2628,
        -0.166,
        -0.0044,
        -0.5094,
    ],
    "retrograde_outer_long_axis_tube": [
        0.7392,
        -0.4129,
        0.1583,
        -0.0472,
        0.1282,
        0.558,
    ],
}
# A strongly diffusing box: irregular static and at corotation radius 8.
IRREGULAR_ORBIT = [-0.3744, 0.4656, -1.8893, -0.0303, -0.0639, -0.0213]
# A short-axis tube in the static figure that, tumbling with corotation at 8
# half-mass radii, still winds steadily about z (circ_z ~1) but passes within
# 0.1% of its mean radius of the axis: a box by the hole cut about the pattern
# axis, a short-axis tube without it.
CENTROPHILIC_WINDING_ORBIT = [2.7859, 1.5476, -0.6433, 0.0649, -0.1524, 0.4105]

# A single orbit gives no population for the adaptive inner/outer cut.
INNER_OUTER_RATIO = 0.84


@functools.lru_cache(maxsize=None)
def _model():
    """The triaxial SCF potential (built once)."""
    return build_scf(flatten=(1.0, 0.79, 0.5))


def _pattern_speed(corotation_radius):
    """``Omega_c`` of the unit Hernquist sphere at that many half-mass radii."""
    r = corotation_radius * (1.0 + np.sqrt(2.0))
    return 1.0 / np.sqrt(r) / (r + 1.0)


def _classify(states, corotation_radius=np.inf, **kwargs):
    """Integrate and classify states, tumbling about z (static if inf)."""
    omega = 0.0 if np.isinf(corotation_radius) else _pattern_speed(corotation_radius)
    results = _results(_model(), states, pattern_speed=(0.0, 0.0, omega))
    return results, results.classify(
        inner_outer_ratio=INNER_OUTER_RATIO, drop_runaways=False, **kwargs
    )


def test_pinned_families_tumbling():
    """Every pinned orbit gets its family, unflagged, at both pattern speeds."""
    names = list(REGULAR_ORBITS)
    for corotation_radius in (16.0, 8.0):
        _, cl = _classify(list(REGULAR_ORBITS.values()), corotation_radius)
        assert list(cl.family_name_array) == names, (
            corotation_radius,
            cl.family_name_array,
        )
        assert not cl.unresolved.any() and not cl.near_corotation.any()
        assert np.all(cl.regularity == Regularity.REGULAR)
        assert cl.inconsistencies() == {k: 0 for k in cl.inconsistencies()}
    print("pinned families in the tumbling figure OK")


def test_pinned_families_static():
    """Slow tumbling keeps the static family of each pinned orbit."""
    _, cl = _classify(list(REGULAR_ORBITS.values()))
    assert list(cl.family_name_array) == list(REGULAR_ORBITS), cl.family_name_array
    print("pinned families in the static figure OK")


def test_irregular_orbit():
    """The diffusing box is irregular both static and tumbling."""
    for corotation_radius in (np.inf, 8.0):
        _, cl = _classify([IRREGULAR_ORBIT], corotation_radius)
        assert cl.family_name_array[0] == "irregular", corotation_radius
        assert cl.regularity[0] == Regularity.DIFFUSING
        assert cl.shape[0] == OrbitShape.BOX
    print("irregular orbit OK")


def test_hole_cut_about_pattern_axis():
    """A winding orbit through the centre is a box only by the hole cut."""
    results, cl = _classify([CENTROPHILIC_WINDING_ORBIT], 8.0)
    assert cl.circulation[0, 2] > 0.99
    assert results.winding_circulation()[0, 2] > 0.99
    assert results.hole_fraction()[0, 2] < 0.01
    assert cl.family_name_array[0] == "pibox", cl.family_name_array
    _, no_hole = _classify([CENTROPHILIC_WINDING_ORBIT], 8.0, hole_thresh=0.0)
    assert no_hole.shape[0] == OrbitShape.SHORT_AXIS_TUBE
    # Static, the same initial conditions give a short-axis tube, whatever the
    # hole cut (it acts only about the pattern-speed axis of a rotating figure).
    for hole_thresh in (0.03, 0.5):
        _, static = _classify([CENTROPHILIC_WINDING_ORBIT], hole_thresh=hole_thresh)
        assert static.family_name_array[0] == "retrograde_short_axis_tube"
    print("hole cut about the pattern axis OK")


if __name__ == "__main__":
    test_pinned_families_tumbling()
    test_pinned_families_static()
    test_irregular_orbit()
    test_hole_cut_about_pattern_axis()
